"""
Dataset module for Handwritten Math Solutions.
"""

import os
from pathlib import Path
from typing import List, Optional, Set
import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TT


class MathDataset(Dataset):
    _all_cropped_files = None

    def __init__(self, path, phase, image_transform=None, target_transform=None):
        
        self.path = path
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.phase = phase

        csv = pd.read_csv(os.path.join(self.path, 'dataset/processed/merged_sorted.csv'))

        self.image_paths = []
        self.labels = []
        # Process the "cropped" folder
        if MathDataset._all_cropped_files is None:
            cropped_folder = os.path.join(self.path, 'dataset', 'raw', 'cropped')
            MathDataset._all_cropped_files = os.listdir(cropped_folder)
            random.shuffle(MathDataset._all_cropped_files)

        if phase == 'train':
            start, end = 0, 5600
        elif phase == 'valid':
            start, end = 5600, 6600
        else:  # test
            start, end = 6600, 7000

        selected_files = MathDataset._all_cropped_files[start:end]

        for i in selected_files:
            image_path = fr'.\cropped\{i}'
            try:
                label = csv[csv['Image Path'] == image_path]['LaTeX Label'].values[0]
                self.image_paths.append(image_path)
                self.labels.append(label)
            except:
                print(f"Image not found in CSV: {image_path}")

        # Process other folders
        for folder in ['Symbols', 'EnglishAlphabet', 'PersianNumbers']:
            folder_path = os.path.join(self.path, 'dataset', 'raw', folder)
            extra_files = []  # Clear list on each iteration

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                for fil in os.listdir(subfolder_path):
                    extra_files.append(fr'dataset\raw\{folder}\{subfolder}\{fil}')

            random.shuffle(extra_files)
            total_count = len(extra_files)
            train_count = int(0.8 * total_count)
            valid_count = int(0.2 * total_count)

            if phase == 'train':
                selected_files = extra_files[:train_count]
            elif phase == 'valid':
                selected_files = extra_files[train_count:train_count + valid_count]
            else:
                selected_files = extra_files[train_count + valid_count:]

            for image_path in selected_files:
                try:
                    label = csv[csv['Image Path'] == image_path]['LaTeX Label'].values[0]
                    self.image_paths.append(image_path)
                    self.labels.append(label)
                except:
                    pass

    def __getitem__(self, index):
        try:
          image_path = os.path.join(self.path, 'dataset/raw', self.image_paths[index].split('\\')[1], self.image_paths[index].split('\\')[2], self.image_paths[index].split('\\')[3])
        except:
          image_path = os.path.join(self.path, 'dataset/raw', self.image_paths[index].split('\\')[1], self.image_paths[index].split('\\')[2])

        # Apply target transformation if specified
        if self.target_transform:
            label = self.target_transform(self.labels[index])
        else:
            label = self.labels[index]

        # Open the image and apply transformations
        image = Image.open(image_path)
        if self.image_transform:
            image = self.image_transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
