"""All file locations, set through environment variables (see README)."""
import os

WORK = os.environ.get("SAGECT_WORK", os.path.abspath("work"))  # slices, features, annotations, runs
PCRL_DIR = os.environ.get("PCRL_DIR", os.path.abspath("third_party/PCRL-MRG"))  # NLG scorer + CTRG-Brain split
MEPNET_DIR = os.environ.get("MEPNET_DIR", os.path.abspath("third_party/MEPNet"))  # CTRG-Brain medical-entity list
CTRG_DIR = os.environ.get("CTRG_DIR", "")  # CTRG-Brain-263K abnormal studies (folder 异常)
CQ500_DIR = os.environ.get("CQ500_DIR", "")  # CQ500 zip files and reads.csv
INDINFM_DIR = os.environ.get("INDINFM_DIR", "")  # private cohort: {train,test}.json and volumes


def work(*parts):
    return os.path.join(WORK, *parts)
