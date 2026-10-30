#!/bin/bash
# Train and test SAGE-CT on one dataset, five seeds (paper configuration = the defaults of sagect.train).
#   bash run.sh ctrg        (or indinfm)
set -euo pipefail
DS=${1:?ctrg or indinfm}
for S in 0 1 2 3 4; do
  python -m sagect.train --ds "$DS" --seed "$S" --out "${SAGECT_WORK:-work}/runs/sagect_${DS}_s${S}"
done
