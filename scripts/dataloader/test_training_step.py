import subprocess
import pkg_resources
import sys

required_packages = [
    "pandas",
    "numpy",
    "pytorch-lightning",
    "torch",
    "matplotlib",
    "Pillow",
    "seaborn",
    "PyYAML",
    "omegaconf",
    "torchvision"
]


#We first download the required packages
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])

import gdrive.GDriveHandler as GDH
import pandas as pd
import numpy as np
import pytorch_lightning.core as L
from torch.utils.data import DataLoader,Dataset
import GDriveDataloading as GDTL
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image,ImageShow
import numpy as np
import torch
import seaborn as sb
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
import torchvision.transforms as transforms

import main
import taming.modules.losses.vqperceptual
from taming.models.vqgan import LAPVQ

installed_packages = {pkg.key for pkg in pkg_resources.working_set}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])

root_dir = "/zhome/56/7/169073"

ID_DF_Path = rf"{root_dir}/taming-transformers/scripts/dataloader/gdrive/FFHQimages.csv"
config_path = fr"{root_dir}/taming-transformers/configs/gdrive_FFHQ.yaml"
scopes = ["https://www.googleapis.com/auth/drive.readonly"]
creds_path = fr"{root_dir}/taming-transformers/scripts/dataloader/gdrive/deep-learning-2023-405822-135193813109.json"
access_token_path = fr"{root_dir}/taming-transformers/scripts/dataloader/test.json"

GDDataloader = GDTL.ImagesDatamodule(ID_DF_Path,scopes,creds_path)
prop_dict = dict({"val":0.1,"test":0.1,"train":0.8})
GDDataloader.setup(prop_dict)

# Prepare CelebAHQ configurations
config_path = fr"{root_dir}/taming-transformers/configs/faceshq_vqgan.yaml"
celebAHQ_config = OmegaConf.load(config_path)
print(yaml.dump(OmegaConf.to_container(celebAHQ_config)))

# Init model with the chosen architecture and configurations
model = LAPVQ(**celebAHQ_config.model.params)


train_dataloader = iter(GDDataloader.train_dataloader())


for i in range(10):
    example_batch = next(train_dataloader).float()/255
    print("Loss:", model.training_step({"image": example_batch, "coord": np.zeros(32)}, 1,0))
    print("Loss:", model.training_step({"image": example_batch, "coord": np.zeros(32)}, 1,1))

