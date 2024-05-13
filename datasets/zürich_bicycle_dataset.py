import os
import numpy as np
import random

import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(is_gray=True):
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.OpticalDistortion(p=0.3),
        A.ChannelShuffle(p=0.1),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=45, p=0.5),
        A.RandomScale(p=0.5)]
    if is_gray:
        transforms.append(A.ToGray(p=.5))
    return A.Compose(transforms)

class ZürichBicycleDataset(Dataset):

  def __init__(self, dir_container, is_gray=True, img_size=(224, 224), undersample_ratio=2.0, n_frames=5, is_one_class=True):
     self.dir_container = dir_container
     self.img_size = img_size
     self.n_frames = n_frames
     self.is_gray = is_gray
     self.one_class_problem = is_one_class
     self.transform = get_transforms(is_gray)
     self.undersample_ratio = undersample_ratio
     self.sequences, self.labels = self._prepare_dataset_data()
     self.sequences = list(np.concatenate(self.sequences))
     self.labels = list(np.concatenate(self.labels))
     self.indices = self._balance_classes()
  def _balance_classes(self):
    zero_indices = [i for i, label_seq in enumerate(self.labels) if np.any(label_seq == 0)]
    non_zero_indices = [i for i, label_seq in enumerate(self.labels) if np.any(label_seq != 0)]
    sampled_zero_indices = random.sample(zero_indices,
                min(int(len(non_zero_indices) * self.undersample_ratio), len(zero_indices)))
    return non_zero_indices + sampled_zero_indices
  def _prepare_dataset_data(self):
    sequences, labels = [], []
    for sequence_folder in sorted(os.listdir(self.dir_container)):
       sequence_path = os.path.join(self.dir_container, sequence_folder)
       images = sorted(
              [os.path.join(sequence_path, "images", img) for img in os.listdir(os.path.join(sequence_path,"images")) if img.endswith('.jpg')])
       label = self._read_labels(os.path.join(sequence_path, 'labels.txt'))
       if len(images) >= self.n_frames:
         sequences.append(images)
         labels.append(label)
    return sequences, labels
  def _read_labels(self, file_path):
      with open(file_path, 'r') as file:
          labels = np.array([int(line.strip()) for line in file.readlines()], dtype=np.float32)
      return labels

  def __getitem__(self, index):
    actual_idx = self.indices[index]

    frame_name = self.sequences[actual_idx]
    label = self.labels[actual_idx]

    image = cv2.imread(frame_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else np.zeros((*self.img_size, 3), dtype=np.uint8)
    image = self.transform(image=image)["image"]
    image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
    label =\
       torch.tensor(label, dtype=torch.float32) if self.one_class_problem else torch.tensor(label, dtype=torch.float32).long()
    image = torch.tensor(image, dtype=torch.float32)
    return image, label

  def __len__(self):
    return len(self.indices)


def zürich_collision_dataloaders(root_dir,
                                 batch_size=10,
                                 img_size=(224, 224),
                                 is_gray=True,
                                 undersample_scale=2.5,
                                 min_n_frames=5,
                                 is_multiclass_problem=True):

    train_dir = os.path.join(root_dir, 'training')
    test_dir = os.path.join(root_dir, 'testing')

    # Initialize datasets
    train_dataset = ZürichBicycleDataset(
        dir_container=train_dir,
        n_frames=min_n_frames,
        is_gray=is_gray,
        img_size=img_size,
        undersample_ratio=undersample_scale,
        is_one_class=not is_multiclass_problem
    )

    test_dataset = ZürichBicycleDataset(
        dir_container=test_dir,
        n_frames=min_n_frames,
        is_gray=is_gray,
        img_size=img_size,
        undersample_ratio=undersample_scale,
        is_one_class=not is_multiclass_problem
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True
    )

    return train_loader, test_loader
