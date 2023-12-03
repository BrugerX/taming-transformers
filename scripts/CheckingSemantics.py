import random
import pandas as pd
from PIL import Image,ImageShow
import torch
import yaml
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import glob
import os
from taming.models.cond_transformer import Net2NetTransformer
from scipy.spatial.distance import euclidean

#Generate an ID to identify this experiment
random_generator = random.Random()
DF_id = random_generator.randint(0,10000)
print(f"EXPERIMENT ID: {DF_id}")

#We create a folder to host the experiment in:
xperiment_folder_dir = f"{DF_id}SemanticsExperiment\\"
if not os.path.exists(xperiment_folder_dir):
    os.makedirs(xperiment_folder_dir)

M = 20

# Prepare CelebAHQ configurations
config_path = fr"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\configs\faceshq_transformer.yaml"
celebAHQ_config = OmegaConf.load(config_path)
print(yaml.dump(OmegaConf.to_container(celebAHQ_config)))

# Init model with the chosen architecture and configurations
model = Net2NetTransformer(**celebAHQ_config.model.params)

#Load checkpoints
ckpt_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\configs\faceshq.ckpt"
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
    data_dir_path = r"C:\Users\DripTooHard\PycharmProjects\taming-transformers2\data\00000-20231203T065959Z-001\00000"

    data_directory = os.fsencode(data_dir_path)

    images = []

    for file in os.listdir(data_directory):
         filename = os.fsdecode(file)
         images += [{"filename":filename,"path":f"{data_dir_path}\\{filename}"}]

    image_paths_DF = pd.DataFrame.from_dict(images)
    image_paths_DF.to_csv(f"{xperiment_folder_dir}{image_paths_DF_path}")




def load_image_VQGAN(image_path):
     image_list = []
     for filename in glob.glob(image_path):
         im=Image.open(filename)

     transform = transforms.Compose([
     transforms.PILToTensor() ])

     #Our model takes floats in the range of [0,1]
     im = (transform(im).float()/255)

     return im.unsqueeze(0)


def representation_worker(paths_list):
    image_array = []

    for image_path in paths_list:
        image_tensor = load_image_VQGAN(image_path)
        latent_representation = model.first_stage_model.encoder(image_tensor)
        quantized_representation = model.first_stage_model.quant_conv(latent_representation)
        quantized_representation, _, _ = model.first_stage_model.quantize(quantized_representation)
        reconstruction = model.first_stage_model.decode(quantized_representation)
        image_array += [
            {"image_path": image_path, "image_original": image_tensor, "latent_representation": latent_representation,
             "quantized_representation": quantized_representation, "image_reconstruction": reconstruction}]

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
"""
This is the parallel version

def representation_worker(paths_list, result_queue):
    image_array = []

    for image_path in paths_list:
        image_tensor = load_image_VQGAN(image_path)
        latent_representation = model.first_stage_model.encoder(image_tensor)
        quantized_representation = model.first_stage_model.quant_conv(latent_representation)
        quantized_representation, _, _ = model.first_stage_model.quantize(quantized_representation)
        reconstruction = model.first_stage_model.decode(quantized_representation)
        image_array.append({"image_path": image_path, "image_original": image_tensor,
                            "latent_representation": latent_representation,
                            "quantized_representation": quantized_representation,
                            "image_reconstruction": reconstruction})

    result_queue.put(image_array)
"""
def worker(paths, result_queue):
    representation_worker(paths, result_queue)


#Name of DF with the different representations
representations_DF_path = "representation_DF.csv"
try:
    representations_DF = pd.read_csv(f"{xperiment_folder_dir}{representations_DF_path}")
except:

    num_workers = 1 #Number of cores used

    image_arrays = []
    image_paths_list = image_paths_DF["path"].iloc[0:M].tolist()

    """
    result_queue = queue.Queue()
    threads = []
    paths_per_thread = divide_paths_for_workers(num_workers, image_paths_list)


    # Create and start threads
    for paths in paths_per_thread:
        thread = threading.Thread(target=worker, args=(paths, result_queue))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Aggregate results
    all_results = []
    while not result_queue.empty():
        all_results.extend(result_queue.get())

    """

    representation_DF =  pd.DataFrame.from_dict(representation_worker(image_paths_list))
    representation_DF.to_csv(f"{xperiment_folder_dir}{representations_DF_path}")
    print("Done with latent representation DF")


def flatten_latent_representation(latent_rep):
    return latent_rep.reshape(-1, latent_rep.shape[1])

latent_representation_DF_path = "latent_representation_DF.csv"

try:
    latent_representations_DF = pd.read_csv(f"{xperiment_folder_dir}{latent_representation_DF_path}")
except:
    results = []
    # Iterate over each combination of rows
    for i in range(len(representation_DF)):
        for j in range(i+1, len(representation_DF)):  # Start from i+1 to avoid duplicate pairs and comparing with itself
            distance =  0
            row_i = representation_DF.iloc[i]
            row_j = representation_DF.iloc[j]

            # Flatten the latent representations
            flat_i = flatten_latent_representation(row_i['latent_representation'])
            flat_j = flatten_latent_representation(row_j['latent_representation'])

            # Ensure both have the same number of vectors
            if flat_i.shape[0] != flat_j.shape[0]:
                raise ValueError("Mismatch in the number of vectors in the latent representations")

            # Calculate distances between corresponding vectors
            for vec_index in range(flat_i.shape[0]):
                vec_i = flat_i[vec_index]
                vec_j = flat_j[vec_index]
                distance =+ euclidean(vec_i, vec_j)

            # Append the result as a dictionary
            results.append({
                "image_path_1": row_i['image_path'],
                "image_path_2": row_j['image_path'],
                "distance": distance
            })


    latent_representation_DF = pd.DataFrame.from_dict(results)
    latent_representation_DF.to_csv(f"{xperiment_folder_dir}{latent_representation_DF_path}")
    print("Done with latent RP DF")


quantized_representation_DF_path = "quantized_representation_DF.csv"

try:
    latent_representations_DF = pd.read_csv(f"{xperiment_folder_dir}{quantized_representation_DF_path}")
except:
    results = []
    # Iterate over each combination of rows
    for i in range(len(representation_DF)):
        for j in range(i+1, len(representation_DF)):  # Start from i+1 to avoid duplicate pairs and comparing with itself
            distance =  0
            row_i = representation_DF.iloc[i]
            row_j = representation_DF.iloc[j]

            # Flatten the latent representations
            flat_i = flatten_latent_representation(row_i['quantized_representation'])
            flat_j = flatten_latent_representation(row_j['quantized_representation'])

            # Ensure both have the same number of vectors
            if flat_i.shape[0] != flat_j.shape[0]:
                raise ValueError("Mismatch in the number of vectors in the latent representations")

            # Calculate distances between corresponding vectors
            for vec_index in range(flat_i.shape[0]):
                vec_i = flat_i[vec_index]
                vec_j = flat_j[vec_index]
                distance =+ euclidean(vec_i, vec_j)

            # Append the result as a dictionary
            results.append({
                "image_path_1": row_i['image_path'],
                "image_path_2": row_j['image_path'],
                "distance": distance
            })


    latent_representation_DF = pd.DataFrame.from_dict(results)
    latent_representation_DF.to_csv(f"{xperiment_folder_dir}{quantized_representation_DF_path}")
    print("Done with quantized rp DF")