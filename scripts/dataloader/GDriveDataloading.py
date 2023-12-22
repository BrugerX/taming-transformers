import pytorch_lightning.core as L
from torch.utils.data import DataLoader, Dataset
import yaml
import pandas as pd
from .gdrive.GDriveHandler import GDrive_Handler
import numpy as np
import torch
from torchvision import transforms

class FacesHQTransform:
    def __call__(self, np_img):
        """
        Convert a NumPy array image to a PyTorch tensor and normalize by dividing by 255.

        :param np_img: A NumPy array representing an image.
                       The array values are expected to be in the range [0, 255].
        :return: Normalized image as a torch.Tensor in the range [0, 1].
        """
        # Ensure the input is a NumPy array
        if not isinstance(np_img, np.ndarray):
            raise TypeError("Input is not a NumPy array")

        # Convert to PyTorch tensor and normalize
        tensor_img = torch.from_numpy(np_img).float() / 255.0

        # If the input array is a grayscale image (H, W),
        # you might want to add a channel dimension (H, W, C) where C=1
        if len(tensor_img.shape) == 2:
            tensor_img = tensor_img.unsqueeze(0)

        tensor_img = tensor_img.permute(2,0,1)

        return tensor_img





class DatasetPurposeMarker:
    def __int__(self):
        pass

    @staticmethod
    def mark_predefined_purpose_txt(ID_DF, purpose_txt_path, purpose):
        with open(purpose_txt_path) as f:
            names_list = f.read().split("\n")
            for name in names_list:
                ID_DF.loc[ID_DF["name"] == name, "purpose"] = purpose

    """

    @precondition: The proportions in the purpose_proportions sum to 1

    @args purpose_proportions: A dict in the format of {"purpose":%proportion in decimal for that purpose}

    """

    @staticmethod
    def mark_purpose(ID_DF, purpose_props: dict, overwrite_proportions=True):

        # First we make sure, that the proportions sum to 1/100%
        sum_proportions = 0
        for proportion in purpose_props.values():
            sum_proportions += proportion
        if (sum_proportions != 1):
            raise Exception(
                f"Sum of proportions of validation, test and training dataset doesn't sum to 100%.\n Instead it sums to: {sum_proportions}")

        if (overwrite_proportions):
            ID_DF["purpose"] = None

        size_of_DF = ID_DF.shape[0]

        for purpose, proportion in purpose_props.items():
            # TODO: We take % of the unmarked, but after the first iteration, this percentage will be lower than the intended percentage
            rows_to_mark = int(size_of_DF * proportion)
            unmarked_rows = ID_DF[ID_DF["purpose"].isna()]

            if (unmarked_rows.shape[0] < rows_to_mark):
                rows_to_mark = unmarked_rows.shape[0]

            indices_to_update = unmarked_rows.sample(rows_to_mark).index
            ID_DF.loc[indices_to_update, "purpose"] = purpose

        return ID_DF


class GDriveDataset(Dataset):
    def __init__(self, ID_DF, gdrive_handler, transform=None):
        """
        Initialize the dataset with Google Drive handler parameters.

        :param ID_DF A Pandas Dataframe in CSV format or a path to it, with a column "id" containing Google Drive IDs for each image
        :param transform: Optional transform to be applied on a sample.
        """

        self.gdrive_handler = gdrive_handler
        self.transform = transform

        if (type(ID_DF) == str):
            self.ID_DF = pd.read_csv(ID_DF)
        elif (type(ID_DF) == pd.DataFrame):
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


class ImagesDatamodule(L.LightningDataModule):
    def __init__(self, data_path, scopes, creds_path, num_workers=0, write_new_token=True,
                 gdrive_token_path="token.json", batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Google drive related
        self.scopes = scopes
        self.creds_path = creds_path
        self.write_new_token = write_new_token
        self.gdrive_token_path = gdrive_token_path

        # Instantiate our GDrive Handler
        self.GDriveHandler = GDrive_Handler(self.scopes, self.creds_path)

        # Get the dataset
        self.ID_DF = pd.read_csv(self.data_path)

    def prepare_data(self):
        # TODO: Add the thing, that uses the GDrive to get a dataset CSV
        pass

    """
    params: purpose_proportions: A dict containing the proportions of the validation, test and training dataset, in that order.
    It must contain every purpose, even if any of them are 0%.

    Example: {"val":0,"test":0.1,"train":0.9}

    """

    def setup(self, purpose_proportions, overwrite_purpose_markings=True):
        # Mark the dataframe
        self.ID_DF = DatasetPurposeMarker.mark_purpose(self.ID_DF, purpose_proportions, overwrite_purpose_markings)
        purpose_keys = list(purpose_proportions.keys())
        faceshq_transform = FacesHQTransform()

        # Get the different datasets
        self.valid_dataset = GDriveDataset(self.ID_DF[self.ID_DF["purpose"] == purpose_keys[0]], self.GDriveHandler,transform=faceshq_transform)
        self.test_dataset = GDriveDataset(self.ID_DF[self.ID_DF["purpose"] == purpose_keys[1]], self.GDriveHandler,transform=faceshq_transform)
        self.train_dataset = GDriveDataset(self.ID_DF[self.ID_DF["purpose"] == purpose_keys[2]], self.GDriveHandler,transform=faceshq_transform)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        self.GDriveHandler.shutdown()