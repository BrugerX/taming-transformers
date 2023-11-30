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
from taming.models.cond_transformer import Net2NetTransformer

ID_DF_Path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\scripts\dataloader\gdrive\FFHQimages.csv"
config_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\configs\gdrive_FFHQ.yaml"
scopes = ["https://www.googleapis.com/auth/drive.readonly"]
creds_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\scripts\dataloader\gdrive\deep-learning-2023-405822-135193813109.json"
access_token_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\scripts\dataloader\test.json"

GDDataloader = GDTL.ImagesDatamodule(ID_DF_Path,scopes,creds_path)
prop_dict = dict({"val":0.1,"test":0.1,"train":0.8})
GDDataloader.setup(prop_dict)

train_dataloader = GDDataloader.train_dataloader()


example_batch = next(iter(train_dataloader))
example_batch = example_batch.float()/255

#Prepare CelebAHQ configurations
config_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\configs\faceshq_transformer.yaml"
celebAHQ_config = OmegaConf.load(config_path)
print(yaml.dump(OmegaConf.to_container(celebAHQ_config)))

#Init model with the chosen architecture and configurations
model = Net2NetTransformer(**celebAHQ_config.model.params)
print("Loss:",model.training_step({"image":example_batch,"coord":np.zeros(32)},1))
