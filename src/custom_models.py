from abc import ABCMeta, abstractmethod

import torchvision
import torch
from torch import nn


class CustomModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, version, out_size, input_layer_aggregation='sum', pre_trained=False):
        super(CustomModel, self).__init__()

        self._create_model(version, out_size, pre_trained)

        if input_layer_aggregation != 'repeat':
            self._rgb_to_grey_input(input_layer_aggregation)

    def forward(self, x):
        x = self.model(x)
        return x

    @abstractmethod
    def _create_model(self, version, out_size, pre_trained):
        """Set the model attribute according to the model type"""
        pass

    @abstractmethod
    def _rgb_to_grey_input(self, input_layer_aggregation):
        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding or averaging the weights of each existing dimension
        in order to retain pretrained parameters"""
        pass

    def get_n_params(self, trainable=True):
        """Return number of (trainable) parameters."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class DenseNet(CustomModel):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def _create_model(self, version, out_size, pre_trained):
        if version == 'DenseNet121':
            self.model = torchvision.models.densenet121(pretrained = pre_trained)
        else:
            raise NotImplementedError('Version of DenseNet not supported')
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def _rgb_to_grey_input(self, input_layer_aggregation):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding or averaging the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv0_weight = self.model.features.conv0.weight.clone()
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            if input_layer_aggregation == 'sum':
                self.model.features.conv0.weight = nn.Parameter(conv0_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights
            else:
                self.model.features.conv0.weight = nn.Parameter(conv0_weight.mean(dim=1,keepdim=True)) # way to keep pretrained weights


class ResNet(CustomModel):
    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """
    def _create_model(self, version, out_size, pre_trained):
        if version == 'ResNet50':
            self.model = torchvision.models.resnet50(pretrained=pre_trained)
        elif version == 'ResNet18':
            self.model = torchvision.models.resnet18(pretrained=pre_trained)
        else:
            raise NotImplementedError('Version of ResNet not supported')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def _rgb_to_grey_input(self, input_layer_aggregation):
        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv1_weight = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            if input_layer_aggregation == 'sum':
                self.model.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights
            else:
                self.model.conv1.weight = nn.Parameter(conv1_weight.mean(dim=1,keepdim=True)) # way to keep pretrained weights
