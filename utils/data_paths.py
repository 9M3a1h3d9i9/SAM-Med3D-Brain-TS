# ''' Example-1: manually list all dataset paths '''
# # img_datas = [
# # 'sam3d_train/medical_data_all/COVID_lesion/COVID1920_ct',
# # 'sam3d_train/medical_data_all/COVID_lesion/Chest_CT_Scans_with_COVID-19_ct',
# # 'sam3d_train/medical_data_all/adrenal/WORD_ct',
# # ]

# ''' Example-2: use glob to automatically list all dataset paths '''
# import os.path as osp
# from glob import glob

# PROJ_DIR = osp.dirname(osp.dirname(__file__))
# img_datas = glob(osp.join(PROJ_DIR, "data", "brain_pre_sam", "*", "*"))

''' Data paths for SAM-Med3D. Uses env var SAM_DATA_ROOT to switch splits. '''
import os
import os.path as osp
from glob import glob

PROJ_DIR = osp.dirname(osp.dirname(__file__))

# Default: use brain_pre_sam (all data). Override with env var:
#   SAM_DATA_ROOT=data/brain_train   (for training)
#   SAM_DATA_ROOT=data/brain_test    (for testing)
DATA_ROOT = os.environ.get("SAM_DATA_ROOT", "data/brain_pre_sam")

img_datas = glob(osp.join(PROJ_DIR, DATA_ROOT, "*", "*"))
print(f"[data_paths] Using DATA_ROOT={DATA_ROOT}, found {len(img_datas)} datasets")