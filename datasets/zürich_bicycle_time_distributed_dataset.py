import os
import cv2
import random

import kornia.augmentation
import numpy as np


import torch
import torch.utils.data as data
from torch.utils.data import Dataset

import kornia as K
from kornia.augmentation.container import AugmentationSequential

transform = AugmentationSequential(
      K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
      data_keys= ["input"],
      same_on_batch=True,
      random_apply=10,)


transform2 = AugmentationSequential(
      K.augmentation.RandomGrayscale(keepdim=False, same_on_batch=True, p=1.0),
      data_keys= ["input"],
      same_on_batch=True,
      random_apply=10,)


class ZürichTimeDistribBicycleDataset(Dataset):

    def __init__(self, dir_container, n_frames=5, is_gray=True, img_size=224, mean=None, std=None, sample_weight=1e-02):

       self.possibility, self.min_possibility = [], []
       self.sequences, self.awareness = [],[]
       self.n_frames = n_frames
       self.dir_container = dir_container
       self.full_len = 0
       self.is_gray = is_gray
       self.img_sz = img_size
       self.sample_weight = sample_weight
       self.INTER_MTH = cv2.INTER_AREA
       self.normalize = None if mean is None or std is None else K.augmentation.Normalize(mean, std)


       self.distortion_transforms = AugmentationSequential(
           AugmentationSequential(  #1
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomElasticTransform(alpha=(.3, .3), same_on_batch=True, p=1., padding_mode='border'),
               same_on_batch=True
           ),
           AugmentationSequential(  #2
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomThinPlateSpline(scale=.02, p=1., same_on_batch=True),
               same_on_batch=True
           ),
           AugmentationSequential(  #3
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomSharpness(sharpness=.8, p=1., same_on_batch=True)
           ),
           AugmentationSequential(  #4
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomGaussianBlur(kernel_size=5,
                                                 sigma=(8., 8.),
                                                 border_type=K.constants.BorderType.CIRCULAR.name,
                                                 p=.5, same_on_batch=True)
           ),
           AugmentationSequential(  #5
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomGaussianNoise(p=1., std=.1, same_on_batch=True)
           ),

           AugmentationSequential(  #6
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomSnow(p=1., same_on_batch=True)
           ),

           AugmentationSequential(  # 7
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomRain(p=1., keepdim=False, same_on_batch=True)
               #  number_of_drops=(50, 100), drop_height=(2, 5), drop_width=(-1, 1),
           ),

           AugmentationSequential(  # 8
               K.augmentation.RandomHorizontalFlip(p=.5, same_on_batch=True, ),
               K.augmentation.RandomRotation(degrees=5, p=1., same_on_batch=True)
           ),

           data_keys=['input'],
           random_apply=1,
           random_apply_weights=[0., 0., 0., 0., 1., 0., 0., 0.],
           same_on_batch=True
       )

       self.color_transforms = AugmentationSequential(
            AugmentationSequential(K.augmentation.RandomPlanckianJitter(same_on_batch=True, p=1.0),
                                   K.augmentation.RandomContrast(same_on_batch=True, p=.5)),
            AugmentationSequential(K.augmentation.RandomPlasmaContrast(same_on_batch=True, p=1.0),
                                   K.augmentation.RandomGamma(same_on_batch=True, p=.5)),
            data_keys=['input'],
            random_apply=1,
            random_apply_weights=[0.5, .5]
       )


       self.gray_transform = AugmentationSequential(
                          K.augmentation.RandomGrayscale(keepdim=False, same_on_batch=True, p=1.0),
                          data_keys= ["input"],
                          same_on_batch=True)

       self.resize = K.augmentation.Resize(size=self.img_sz, keepdim=False) #(self.img_sz, self.img_sz)

       self._prepare_dataset_data(dir_container)


       for idx in range(len(self.sequences)):
           self.possibility +=[np.random.rand(
               self.sequences[idx].shape[0]//self.n_frames -1)*self.sample_weight]
           self.min_possibility +=[float(np.min(self.possibility[-1]))]
           self.full_len +=self.sequences[idx].shape[0]


    def __getitem__(self, item):
        seq_ind = int(np.argmin(self.min_possibility))
        pick_ind = np.argmin(self.possibility[seq_ind])

        sequence, awareness = \
            self.get_sequence(seq_ind, pick_ind, item)

        sequence = self.prepare_frame_data(sequence)
        sequence = self.color_transforms(sequence)
        sequence = self.distortion_transforms(sequence)
        sequence = self.gray_transform(sequence) if self.is_gray else sequence
        sequence = sequence if self.normalize is None else self.normalize(sequence)
        sequence = self.resize(sequence)

        sequence = sequence[:,[0]] if self.is_gray else sequence
        delta = np.abs(1 - float(pick_ind) / float(self.possibility[seq_ind].shape[0]))
        self.possibility[seq_ind][pick_ind] += delta*self.sample_weight
        self.min_possibility[seq_ind] = np.min(self.possibility[seq_ind])
        return sequence, awareness

    def get_sequence(self, seq_id, pick_id, item):
        sequence = self.sequences[seq_id]
        awareness = self.awareness[seq_id]

        n_subseq = sequence.shape[0]//self.n_frames
        res_frames =\
            sequence.shape[0] - n_subseq*self.n_frames
        start = res_frames *(item % 2)
        end = sequence.shape[0]  - (res_frames if start == 0 else 0)
        sequence = sequence[start:end]

        frame_id = pick_id*self.n_frames + random.randint(0, self.n_frames-1)

        return sequence[frame_id:frame_id+self.n_frames],\
               awareness[frame_id+self.n_frames-1]

    def __len__(self):
        return self.full_len // self.n_frames

    def _prepare_dataset_data(self, dir_container, img_dir='images'):
        for sequence in os.listdir(dir_container):

            try:
                self.awareness.append(self._read_collisions(os.path.join(dir_container, sequence)))
            except Exception as e:
                print('error %s', str(e))
                continue
            frames = []
            for img_name in os.listdir(os.path.join(dir_container, sequence, img_dir)):
                image_name = os.path.join(dir_container, sequence, img_dir, img_name)
                frames.append(image_name)

            self.sequences.append(np.array(frames))

    def _read_collisions(self, dir_container, file_name='labels.txt'):
        with open(os.path.join(dir_container, file_name), 'r') as f:
          lines = f.readlines()
          collisions = [int(line) for line in lines]
          return collisions

    def prepare_frame_data(self, sequence):
        frame_seq =[]
        for img_name in sequence:
            if cv2.haveImageReader(img_name):
                frame = cv2.imread(img_name)
            else:
                frame = np.zeros((self.img_sz[0], self.img_sz[1], 3), np.uint8)

            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

            frame_seq.append(frame)


        out_tensor = torch.from_numpy(np.array(frame_seq)).float() / 255.0
        out_tensor = torch.permute(out_tensor, (0, 3, 1, 2))
        return out_tensor

def zürich_collision_dataloaders(root_dir, img_size=224, batch_sz=10, n_frames=10, is_gray=False):
    train_dataset = ZürichTimeDistribBicycleDataset(os.path.join(root_dir, 'training'),
                                                    img_size=img_size,
                                                    n_frames=n_frames,
                                                    is_gray=is_gray)

    test_dataset = ZürichTimeDistribBicycleDataset(os.path.join(root_dir, 'testing'),
                                                   img_size=img_size,
                                                   n_frames=n_frames,
                                                   is_gray=is_gray)

    train_loader = data.DataLoader(train_dataset, batch_sz, shuffle=True, num_workers=1,drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_sz, shuffle=True, num_workers=1, drop_last=True)
    return train_loader, test_loader








