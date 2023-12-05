import argparse
import pandas as pd
from PIL import Image
import torch
import yaml
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
import os
import taming.modules.losses.vqperceptual
import taming.models.vqgan
from taming.models.cond_transformer import Net2NetTransformer
from scipy.spatial.distance import euclidean

parser = argparse.ArgumentParser(
                    prog='Semantic-LR Datagathering',
                    description='Gathers data that we can use to look at the relationship between the semantic space and the latent space',
                    epilog='Text at the bottom of help')

parser.add_argument("-Nr",'--nr_worker',type=int)
parser.add_argument('-M', '--nr_paths',type=int)      # option that takes a value

args = parser.parse_args()
M = args.nr_paths
nr_worker = args.nr_worker

#We create a folder to host the experiment in:
xperiment_folder_dir = f"{nr_worker}SemanticsExperiment\\"

if not os.path.exists(xperiment_folder_dir):
    os.makedirs(xperiment_folder_dir)

# Prepare CelebAHQ configurations
config_path = fr"../configs/faceshq_transformer.yaml"
celebAHQ_config = OmegaConf.load(config_path)
print(yaml.dump(OmegaConf.to_container(celebAHQ_config)))

# Init model with the chosen architecture and configurations
model = Net2NetTransformer(**celebAHQ_config.model.params)

#Load checkpoints
ckpt_path = r"../configs/faceshq.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
model.load_state_dict(sd)
missing, unexpected = model.load_state_dict(sd, strict=False)

#Put model in evaluation mode
model.eval()
torch.set_grad_enabled(False)

#Name of DF with image paths and names
image_paths_DF_path = "image_paths_DF.csv"

#The directory to the images
try:
    image_paths_DF = pd.read_csv(f"{xperiment_folder_dir}{image_paths_DF_path}")
except:
    data_dir_path = r"../data/00000-20231203T065959Z-001/00000"

    data_directory = os.fsencode(data_dir_path)

    images = []

    for file in os.listdir(data_directory):
         filename = os.fsdecode(file)
         images += [{"filename":filename,"path":f"{data_dir_path}\\{filename}"}]

    image_paths_DF = pd.DataFrame.from_dict(images)
    image_paths_DF.to_csv(f"{xperiment_folder_dir}{image_paths_DF_path}")


# Takes a first stage model and returns a dataframe of distances between codes in the model's codebook
def codebook_distances(first_stage_model):
    results = []

    for codebook in first_stage_model.quantize.parameters():
        for idx,code in enumerate(codebook):
            for i in range(idx + 1, codebook.shape[0]):
                code_i = codebook[i]
                results.append({"idx_1":idx,"idx_2":i,"distance":euclidean(code,code_i)})

    return pd.DataFrame.from_dict(results)

codebook_edges_path = "codebook_edges.csv"

try:
    codebook_edges_DF = pd.read_csv(f"{xperiment_folder_dir}{codebook_edges_path}")
except:
    codebook_edges_DF = codebook_distances(model.first_stage_model)
    codebook_edges_DF.to_csv(f"{xperiment_folder_dir}{codebook_edges_path}")

def calculate_average_distance(row,df,source,target, weight = "distance"):
    return df[df[source] == row[target]][weight].mean()


def load_image_VQGAN(image_path):
     image_list = []

     for filename in glob.glob(image_path):
         im=Image.open(filename)

     transform = transforms.Compose([
     transforms.PILToTensor() ])

     #Our model takes floats in the range of [0,1]
     im = (transform(im).float()/255)

     return im.unsqueeze(0)

#index_popularity: A dict that records the count of each index
#Quantized_indices: A tensor/list of shape [[idx1,idx2,...,idx_n]]
def record_popularity(index_popularity,quantized_indices):

    for index in quantized_indices[0]:
        index = int(index)
        try:
            index_popularity[index] += 1
        except:
            index_popularity[index] = 1


def representation_worker(model,paths_list):
     image_array = []

     for image_path in paths_list:
          image_tensor = load_image_VQGAN(image_path)
          latent_representation = model.first_stage_model.encoder(image_tensor)
          quantized_representation,quantized_indices = model.encode_to_z(image_tensor)
          reconstruction = model.decode_to_img(quantized_indices,quantized_representation.shape)
          image_array += [{"image_path":image_path,"image_original":image_tensor,"latent_representation":latent_representation,"quantized_indices":quantized_indices,"image_reconstruction":reconstruction}]

     return image_array

import threading
import queue

def divide_paths_for_workers(num_workers, image_paths_list):
    # Calculate the number of paths per worker
    total_paths = len(image_paths_list)
    paths_per_worker = total_paths // num_workers
    remainder = total_paths % num_workers

    paths_per_thread = []
    start = 0

    # Distribute the paths among the workers
    for i in range(num_workers):
        # Add an extra path to some workers to distribute the remainder
        end = start + paths_per_worker + (1 if i < remainder else 0)
        paths_per_thread.append(image_paths_list[start:end])
        start = end

    return paths_per_thread


def parallel_representation_worker(model,paths_list, representation_queue):
    image_array = []

    for image_path in paths_list:
        image_tensor = load_image_VQGAN(image_path)
        latent_representation = model.first_stage_model.encoder(image_tensor)
        quantized_representation,quantized_indices = model.encode_to_z(image_tensor)
        reconstruction = model.decode_to_img(quantized_indices,quantized_representation.shape)
        image_array += [{"image_path":image_path,"image_original":image_tensor,"latent_representation":latent_representation,"quantized_indices":quantized_indices,"image_reconstruction":reconstruction}]


    representation_queue.put(image_array)

def worker(model,paths, result_queue):
    parallel_representation_worker(model,paths, result_queue)

#Name of DF with the different representations
representations_DF_path = "representation_DF.csv"
import time

start = time.time()
try:
    representations_DF = pd.read_csv(f"{xperiment_folder_dir}{representations_DF_path}")
except:

    image_arrays = []
    image_paths_list = image_paths_DF["path"].iloc[M*(nr_worker-1):M*nr_worker].tolist()


    representations = representation_worker(model,image_paths_list)
    representation_DF = pd.DataFrame.from_dict(representations)
    representation_DF.to_csv(f"{xperiment_folder_dir}{representations_DF_path}")


    representations_DF =  pd.DataFrame.from_dict(representations)
    representations_DF.to_csv(f"{xperiment_folder_dir}{representations_DF_path}")
    print(representations_DF)

empirical_codes_DF_path = "empirical_codes.csv"
try:
    empirical_codes_DF = pd.read_csv(f"{xperiment_folder_dir}{empirical_codes_DF_path}")
except:
    for row in representations_DF["quantized_indices"]:
        popularity_dict = dict()
        record_popularity(popularity_dict,row)

    empirical_codes_DF = pd.DataFrame.from_dict(popularity_dict.items()).rename(columns={0:"idx",1:"frequency"})
    empirical_codes_DF.to_csv(f"{xperiment_folder_dir}{empirical_codes_DF_path}")




