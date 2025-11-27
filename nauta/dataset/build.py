from torch.utils.data import DataLoader
from nauta.dataset.preprocessing import get_preprocessing_layer
from nauta.dataset.vtuad import VTUADFeature, VTUAD
import torch
import random
import numpy as np
import argparse
import yaml
from tqdm import tqdm

def create_data_loader(data, batch_size, shuffle=True):
    """Creates a pytorch dataloader from a Dataset.

    Args:
        data (Dataset): The desired dataset.
        batch_size (int): The size of the mini batch.
        shuffle (bool, optional): Indicates if the data needs to be shuffled. Defaults to True.

    Returns:
        DataLoader: The generated dataloader.
    """
    loader = DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=shuffle)

    return loader

def get_split_dataloader(config, split="test", shuffle=False):
    """Returns the desired dataloader for the selected split.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader : The desired dataloader object.
    """
    if config["dataset"]["type"] == "VTUADfeature":
        batch_size = config["dataset"]["batch_size"]
        dataset_path = config["dataset"][f"{split}_root_path"]
        preprocessings = config["dataset"]["preprocess"]
        num_of_classes = config["model"]["num_of_classes"]

        # Get the dataset and dataloader.
        dataset = VTUADFeature(
            dataset_path, num_of_classes=num_of_classes, preprocessing=preprocessings
        )
        dataloader = create_data_loader(dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        sample_rate = config["dataset"]["sample_rate"]
        number_of_samples = sample_rate * config["dataset"]["number_of_samples"]
        batch_size = config["dataset"]["batch_size"]

        metadata_path = config["dataset"][f"{split}_metadata"]

        pre_processing_type = config["dataset"]["preprocess"].lower()
        transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

        # Get the dataset and dataloader.
        dataset = VTUAD(
            metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        dataloader = create_data_loader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def get_dataset(config):
    """Returns the desired dataloaders for validation and train.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    if config["dataset"]["type"] == "VTUADfeature":
        batch_size = config["dataset"]["batch_size"]
        train_dataset_path = config["dataset"]["train_root_path"]
        validation_dataset_path = config["dataset"]["validation_root_path"]
        preprocessings = config["dataset"]["preprocess"]
        num_of_classes = config["model"]["num_of_classes"]

        # Get the training and validation.
        train_dataset = VTUADFeature(
            train_dataset_path, num_of_classes=num_of_classes, preprocessing=preprocessings
        )
        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

        validation_dataset = VTUADFeature(
            validation_dataset_path, num_of_classes=num_of_classes, preprocessing=preprocessings
        )
        validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, validation_dataloader
    else:
        sample_rate = config["dataset"]["sample_rate"]
        number_of_samples = sample_rate * config["dataset"]["number_of_samples"]
        batch_size = config["dataset"]["batch_size"]

        train_metadata_path = config["dataset"]["train_metadata"]
        validation_metadata_path = config["dataset"]["validation_metadata"]

        pre_processing_type = config["dataset"]["preprocess"].lower()
        transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

        # Get the training, validation and test dataloaders.
        train_dataset = VTUAD(
            train_metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)
        for input_data, target_data in tqdm(train_dataloader,
            desc=f"Train", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            ):

            print(input_data.shape)
            print(target_data.shape)
        validation_dataset = VTUAD(
            validation_metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, validation_dataloader

def create_parser():
    """Create the parser object.

    Returns:
        parser: The generated parser object with arguments
    """
    parser = argparse.ArgumentParser(description="Execute the training routine.")

    parser.add_argument(
        "--config_file",
        type=str,
        default="E:/Awork/target/underwater_snd/config_files/vtuad.yaml",
        help=""
    )

    return parser

def main():
    """The core of the training execution.
    Initializes all the variables and call the respective methods.
    """
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    print("Start training\n\n")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # Get the training, validation and test dataloaders.
    train_dataloader, validation_dataloader = get_dataset(args_list)

if __name__ == "__main__":
    main()
