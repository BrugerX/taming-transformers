import gdrive.GDriveHandler as GDH
import matplotlib.pyplot as plt
import pandas as pd


root_dir = "/zhome/56/7/169073"


Gdrive_api = GDH.GDrive_Handler(scopes,creds_path)
ID_DF = pd.read_csv(ID_DF_Path)

destination_dir = "/work3/s214590/FFHQ/"

for row in ID_DF:
    id = row["id"]
    file_name = row["name"]
    image = Gdrive_api.download_image_file(id)
    plt.imsave(f"{destination_dir}{file_name}")