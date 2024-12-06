from pathlib import Path

from clinical_ts.ecg_utils import prepare_data_acs
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *


target_fs=100
data_root=Path("./ecg_data/")
target_root=Path("./data")

data_folder_acs = data_root / "acs_database/"
target_folder_acs = target_root / ("acs_database" + str(target_fs))


df_acs, lbl_itos_acs,  mean_acs, std_acs = prepare_data_acs(data_folder_acs, target_fs=target_fs, channels=12, channel_stoi=channel_stoi_default, target_folder=target_folder_acs)


#reformat everything as memmap for efficiency
reformat_as_memmap(df_acs, target_folder_acs/("memmap.npy"),data_folder=target_folder_acs,delete_npys=True)

