from .mydataset import ImageFolder
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from prefetch_generator import BackgroundGenerator
import torchvision

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

number_work = 6

def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=16, return_id=False, balanced=False, val=False, val_data=None):

    source_folder = torchvision.datasets.ImageFolder(os.path.join(source_path), transform=transforms[source_path])

    target_folder_train = ImageFolder(os.path.join(target_path),
                                  transform=transforms[target_path],
                                  return_paths=False, return_id=return_id)
    if val:
        source_val_train = ImageFolder(val_data, transforms[source_path], return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms["eval"],
                                   return_paths=True)


    if balanced:
        freq = Counter(source_folder.targets)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.targets]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.targets))
        print("use balanced loader")
        source_loader = DataLoaderX(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=number_work)
    else:
        source_loader = DataLoaderX(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=number_work)

    target_loader = DataLoaderX(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=number_work)
    test_loader = DataLoaderX(
        eval_folder_test,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=number_work)
    if val:
        test_loader_source = DataLoaderX(
            source_val_test,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=number_work)
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader,target_loader, test_loader, target_folder_train


def get_loader_label(source_path, target_path, target_path_label, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):


    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_label = ImageFolder(os.path.join(target_path_label),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = DataLoaderX(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=number_work)
    else:
        source_loader = DataLoaderX(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=number_work)

    target_loader = DataLoaderX(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=number_work)
    target_loader_label = DataLoaderX(
        target_folder_label,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=number_work)
    test_loader = DataLoaderX(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_work)

    return source_loader, target_loader, target_loader_label, test_loader, target_folder_train



