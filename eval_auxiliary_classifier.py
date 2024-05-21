import argparse
import torch
from trainer import Trainer, DenseNet121, ResNet50
from auxiliary_classifier import read_dataset
from torchvision import transforms
import os
from PIL import Image


class CheXpertDataset(torch.utils.data.Dataset):
    # CheXpert mean and std
    xray_mean = 0.5029
    xray_std = 0.2899

    def __init__(self, folder_dir, image_size, normalization):
        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels

        # Define list of image transformations
        image_transformation = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]

        if normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(self.xray_mean, self.xray_std))

        self.image_transformation = transforms.Compose(image_transformation)

        self.image_paths = [os.path.join(folder_dir, img_name) for img_name in os.listdir(folder_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert('L')

        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data)

        return image_data, os.path.split(image_path)[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test auxiliary classifier on images in folder')
    parser.add_argument('--model', default='DenseNet121', choices=['DenseNet121', 'ResNet50'], type=str,
                        help='Prediction model.')
    parser.add_argument('--label', default='Sex', choices=['Sex', 'Age'], type=str, help='Which label to predict.')
    parser.add_argument('--checkpoint', '-c', type=str, help='Trained model file.')
    parser.add_argument('--img_folder', '-f', type=str, required=False, help='Folder holding images')
    parser.add_argument('--run_all', action='store_true')
    args = parser.parse_args()

    num_classes = 1
    if args.run_all:
        for label in ['Sex', 'Age']:
            for model_type in ['DenseNet121', 'ResNet50']:
                if model_type == 'DenseNet121':
                    model = DenseNet121(num_classes, label=='Sex', colour_input='L', pre_trained=False).cuda()
                elif model_type == 'ResNet50':
                    model = ResNet50(num_classes, label=='Sex', colour_input='L', pre_trained=False).cuda()
                
                modelCheckpoint = torch.load(f'{args.checkpoint}/cpt_{label}_{model_type}.pth.tar')
                model.load_state_dict(modelCheckpoint['state_dict'])

                model.eval()
                
                for folder_name in os.listdir(args.img_folder):
                    print(f'===== Predicting {label} using {model_type} for images in {folder_name} =====')
                    
                    image_size = 224
                    dataset = CheXpertDataset(f'{args.img_folder}/{folder_name}', image_size, normalization=True)
                    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

                    with torch.no_grad():
                        for image, name in data_loader:
                            out = model(image.cuda())
                            print(f'{name}: {out.cpu().detach().numpy()}')
                    print('\n')
                    
    else:
        if args.model == 'DenseNet121':
            model = DenseNet121(num_classes, args.label=='Sex', colour_input='L', pre_trained=False).cuda()
        elif args.model == 'ResNet50':
            model = ResNet50(num_classes, args.label=='Sex', colour_input='L', pre_trained=False).cuda()

        modelCheckpoint = torch.load(args.checkpoint)
        model.load_state_dict(modelCheckpoint['state_dict'])

        model.eval()

        # read dataset
        image_size = 224
        if args.img_folder is not None:
            dataset = CheXpertDataset(args.img_folder, image_size, normalization=True)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

            with torch.no_grad():
                for image, name in data_loader:
                    out = model(image.cuda())
                    print(f'Predicting image {name}: {out}')
        else:
            _, dataset = read_dataset(args.label)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            Trainer.epochVal(model, data_loader, args.label, torch.nn.BCELoss() if args.label=="Sex" else torch.nn.MSELoss(), use_gpu=True)


