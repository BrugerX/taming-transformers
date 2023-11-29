import pytorch_lightning.core as L
from torch.utils.data import DataLoader,Dataset
import yaml
import pandas as pd
import gdrive.GDriveHandler as GDH


class GDriveDataset(Dataset):
    def __init__(self,ID_DF,config_path, transform=None):
        """
        Initialize the dataset with Google Drive handler parameters.

        :param ID_DF A Pandas Dataframe in CSV format or a path to it, with a column "id" containing Google Drive IDs for each image
        :param scopes: Scopes for Google Drive API access.
        :param credentials_path: Path to the credentials file.
        :param token_path: Path to the token file.
        :param write_new_token: Whether to write a new token or use an existing one.
        :param transform: Optional transform to be applied on a sample.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        gdrive_config = config['gdrive']
        dataset_config = config['dataset']

        self.gdrive_handler = GDH.GDrive_Handler(gdrive_config['scopes'], gdrive_config['credentials_path'],
                                                 gdrive_config['write_new_token'],gdrive_config["token_path"])
        self.transform = transform

        if (type(ID_DF) == str):
            self.ID_DF = pd.read_csv(ID_DF)
        elif(type(ID_DF) == pd.DataFrame):
            self.ID_DF = ID_DF

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.ID_DF.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve the data item and its label at the specified index.

        :param idx: Index of the data item.
        """
        ith_row = self.ID_DF.iloc[idx]

        data = self.gdrive_handler.download_image_file(ith_row["id"])  # Method to load data from Google Drive
        if self.transform:
            data = self.transform(data)

        return data