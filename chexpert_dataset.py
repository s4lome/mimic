import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torch


class CheXpert_DataSet(Dataset):
    """
    A custom dataset class for the CheXpert dataset.

    Args:
        data_path (str): The path to the CheXpert dataset.
        pathology_dict (dict): A dictionary that maps integer keys (0 to 13) to
            disease names, as given by the MIMIC dataset. 
        transforms (callable, optional): A callable function/transform to apply to
            the images. Default is None.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Retrieves an image and its corresponding labels at the
            specified index.

    Attributes:
        data_path (str): The path to the CheXpert dataset.
        pathology_dict (dict): A dictionary containing disease names.
        annotations_file (pandas.DataFrame): The preprocessed annotations data.
        transforms (callable, optional): The transformation function for images.

    """

    def __init__(self, data_path, pathology_dict, transforms=None):
        """
        Initializes a CheXpert_DataSet instance.

        Args:
            data_path (str): The path to the CheXpert dataset.
            pathology_dict (dict): A dictionary that maps integer keys (0 to 13) to
                disease names, as given by the MIMIC dataset.
            transforms (callable, optional): A callable function/transform to apply to
                the images. Default is None.
        """

        self.data_path = data_path
        self.pathology_dict = pathology_dict
        self.annotations_file = preprocess_annotations(self.data_path, self.pathology_dict)
        self.transforms = transforms

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.annotations_file)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding labels at the specified index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            list: A list containing the image and its corresponding labels.
        """

        image_path = self.data_path + '/' + self.annotations_file['Path'][index]
        image = cv2.imread(image_path)
        label = self.annotations_file['Labels'][index]
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms
        if self.transforms:
            image = self.transforms(image=image)["image"]

        return [image], label


def preprocess_annotations(data_path, pathology_dict):
    """
    Preprocesses CheXpert dataset annotations.

    This function reads the CheXpert dataset annotations from a CSV file located
    at the specified `data_path`. It then reorders the columns in the DataFrame
    to match the order specified by the `pathology_dict` dictionary. The labels
    are concatenated into a 'Labels' column as a list. Finally, the function
    retains only the 'Path' and 'Labels' columns.

    Args:
        data_path (str): The path to the CheXpert dataset.
        pathology_dict (dict): A dictionary that maps integer keys (0 to 13)
            to disease names, as given by the MIMIC dataset.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame containing the 'Path' column
        with file paths and the 'Labels' column with a list of disease labels
        in the order specified by the `pathology_dict`.
    """
    # Read the CheXpert dataset annotations from the CSV file
    annotations_file = pd.read_csv(data_path + '/CheXpert-v1.0-small/' + 'valid.csv')
    
    # Filter for only Frontal
    #annotations_file = annotations_file[annotations_file['Frontal/Lateral']=='Frontal']
    
    annotations_file.reset_index(drop=True, inplace=True)
    # Create a list of the column names in the desired order
    reorder_mimic = [pathology_dict[i] for i in range(len(pathology_dict))]

    # Reorder the columns in the DataFrame
    reorder_mimic = annotations_file[reorder_mimic]

    # Replace the original columns with the reordered columns
    annotations_file = pd.concat((annotations_file.iloc[:, :5], reorder_mimic), axis=1)
    
    # Concatenate labels into a 'Labels' column as a list
    annotations_file['Labels'] = annotations_file.iloc[:, 5:].values.tolist()
    
    # Retain only the 'Path' and 'Labels' columns
    annotations_file = annotations_file[['Path', 'Labels']]
    
    return annotations_file