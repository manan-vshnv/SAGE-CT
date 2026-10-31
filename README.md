# SAGE-CT

Code for **SAGE-CT: A Lightweight Retrieval-Augmented Framework with Slice-Aware Latent Compression for Head Computed Tomography Report Generation**
(M. Vaishnav\*, C. S. Raghaw\*, P. K. Yalavarthy; \*equal contribution).

SAGE-CT generates the findings section of a brain CT report from 24 axial slices:

1. **Slice-aware CT encoding and scan–report alignment.** Stage-3 and stage-4 features of a frozen CQ500
   ResNet-101 (4×4 per slice, 384 tokens) are compressed into 64 latents by a Perceiver resampler with a
   slice–row–column rotary code on its keys. Stage 1 aligns the pooled latents with report embeddings (sigmoid
   contrastive loss).
2. **Case retrieval.** The five most similar training studies in the aligned space; their report clauses
   (frozen BERT embeddings) become knowledge tokens.
3. **Dual-context report generation.** A three-layer decoder attends to the latents and to the knowledge tokens
   through separate cross-attentions; a per-token gate decides how much retrieved knowledge enters each word.

About 28 M trainable parameters; trained with cross-entropy only; one configuration for both datasets.

## Layout

```
sagect/   model.py      encoder, alignment heads, decoder and knowledge gate
          train.py      Stage 1, retrieval index, Stage 2, greedy decoding, test
          data.py       splits, report text, annotation file
          metrics.py    BLEU-4, METEOR, ROUGE-L, CIDEr (PCRL's scripts) and finding F1
          findings_zh.py / findings_en.py   finding extraction for finding F1
          significance.py                   paired bootstrap test between two systems
prep/     cq500_slices.py, indinfm_slices.py, series.py   CT volumes -> 5 mm axial slices (brain window)
          encoder_cq500.py   ResNet-101 fine-tuned on CQ500 (PCRL / MEPNet recipe), then frozen
          features.py        frozen stage-3 / stage-4 features of the 24 slices
          annotations.py, clause_emb.py   reports, tokens and clause embeddings
```

## Setup

Python 3.11, one GPU, `pip install -r requirements.txt`, and a Java runtime for METEOR.

Two public repositories provide the scorer, the CTRG-Brain split and the entity list used by term F1:

```bash
git clone https://github.com/Chauncey-Jheng/PCRL-MRG third_party/PCRL-MRG   # tested at 016cf03
git clone https://github.com/YanzhaoShi/MEPNet third_party/MEPNet           # tested at 9370251
```

Locations are set through environment variables:

| Variable | Content |
|---|---|
| `SAGECT_WORK` | output folder for slices, features and runs (default `./work`) |
| `PCRL_DIR`, `MEPNET_DIR` | the two clones above (default `./third_party/...`) |
| `CTRG_DIR` | CTRG-Brain-263K abnormal studies (folder `异常`) |
| `CQ500_DIR` | CQ500 zip files and `reads.csv` |
| `INDINFM_DIR` | IndiNFM (private; access is restricted by the ethics approval) |

### Windows

The code runs natively on Windows (PowerShell) with a CUDA build of PyTorch from pytorch.org and a Java runtime
on `PATH`. Set the variables with `$env:`, and use `run.ps1` instead of `run.sh`:

```powershell
$env:SAGECT_WORK = "D:\sagect\work"
$env:CTRG_DIR    = "D:\data\CTRG-Brain-263K\异常"
python -m prep.annotations ctrg
.\run.ps1 ctrg
```

Alternatively, WSL2 (Ubuntu) runs the Linux commands below unchanged, with GPU support.

## Data

* **CTRG-Brain** (public): the 6001 abnormal studies, 7:1:2 split released with PCRL (4200 / 601 / 1200).
* **CQ500** (public): used only to fine-tune the slice encoder.
* **IndiNFM** (private, not distributed): official split (3610 / 378 / 996).

## Pipeline

Run from the repository root.

```bash
# slice encoder (once)
python -m prep.cq500_slices 0 1 && python -m prep.cq500_slices labels
python -m prep.encoder_cq500                       # -> work/encoder_cq500.pt

DS=ctrg                                            # or indinfm
python -m prep.indinfm_slices 0 1                  # IndiNFM only
python -m prep.annotations $DS                     # -> work/ann_$DS.json
python -m prep.features $DS                        # -> work/feat_s4_$DS, work/feat_s3_$DS
python -m prep.clause_emb $DS                      # -> work/clause_emb_$DS.pt
bash run.sh $DS                                    # five seeds -> work/runs/sagect_${DS}_s{0..4}
```

Each run folder holds `result.json` (test scores, selected epoch, parameter count), `pred_test.json`,
`retrieval.json` (the retrieved training studies of every study) and `model-best.pth`. The paper reports the mean
over seeds 0–4.

Significance against another system (e.g. a baseline's test predictions in the same `[{"id", "pred", "gt"}]` format):

```bash
python -m sagect.significance --ds $DS --a work/runs/sagect_${DS}_s*/pred_test.json --b other_pred_test.json
```

## Configuration and ablations

The defaults of `sagect.train` are the configuration of the paper: Stage 1 for 20 epochs (batch 64, learning
rate 3e-4), Stage 2 for 40 epochs (batch 16, learning rate 1e-4, one-cycle schedule), 64 latents, 5 retrieved
cases, image dropout 0.1, three decoder layers, label smoothing 0.1, validation every second epoch.

| Ablation | Flag |
|---|---|
| stage-4 features only | `--feat g4` |
| no slice–row–column rotary code | `--no_rope` |
| scalar gate / gate from the decoder state / without agreement term | `--gate scalar` / `state` / `nocos` |
| no retrieved knowledge | `--no_rag` |
| retrieval on raw CNN features | `--retr_space raw` |
| sensitivity | `--k`, `--n_lat`, `--p_drop`, `--layers` |

## Citation

```bibtex
@article{vaishnav2026sagect,
  title   = {{SAGE-CT}: A Lightweight Retrieval-Augmented Framework with Slice-Aware Latent Compression for Head Computed Tomography Report Generation},
  author  = {Vaishnav, Manan and Raghaw, Chandravardhan Singh and Yalavarthy, Phaneendra K.},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  note    = {under review},
  year    = {2026}
}
```
