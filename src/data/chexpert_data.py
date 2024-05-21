#https://github.com/Stomper10/CheXpert/blob/master/materials.py
"""Dataset class for CheXpert data.
Can also process other data, but CSV containing metadata has to have the same structure as
CheXpert CSV."""

import csv
from PIL import Image

import torch
from torch.utils.data import Dataset



class CheXpertDataSet(Dataset):
    def __init__(self, data_path, data_file, class_idx, policy, colour_input, transform = None):
        """
        data_path: path to where the data lives.
        data_file: path to the file containing image paths with corresponding labels.
        class_idx: Indices of findings/classes to be included.
        policy: name the policy with regard to the uncertainty labels.
        transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []

        with open(data_file, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header

            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                #keep only labels that should be included
                label = [label[i] for i in class_idx]

                for i in range(len(class_idx)):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeros":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0

                    if len(class_idx) == 1:
                        # just using binary classification, true label is "No Finding", swapping labels
                        label[i] = 1 - label[i]


                image_names.append(data_path + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        self.colour_input = colour_input
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and return the image and its labels'''

        image_name = self.image_names[index]
        image = Image.open(image_name).convert(self.colour_input) # RGB or L for greyscale
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
