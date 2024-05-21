import os
#selected_gpus = [0] #configure this
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import torch
import time
import pandas as pd
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import inversefed
from trainer import Trainer, DenseNet121, ResNet50

DATA_BASE_PATH = '/dhc/dsets/ChestXrays/CheXpert/CheXpert-v1.0-large/CheXpert-v1.0'
image_size = 224
random_seed = 207


class CheXpertDataset(torch.utils.data.Dataset):
    # CheXpert mean and std
    xray_mean = 0.5029
    xray_std = 0.2899

    def __init__(self, folder_dir, dataframe, image_size, normalization):
        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
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

        # Get all image paths and image labels from dataframe
        for index, row in dataframe.iterrows():
            image_path = os.path.join(folder_dir, row['Path'])
            self.image_paths.append(image_path)
            self.image_labels.append(row['Label'])

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

        return image_data, torch.tensor(self.image_labels[index], dtype=torch.float)


def get_argparser():
    parser = argparse.ArgumentParser(description='Predict either sex or age from CXR')

    parser.add_argument('--model', default='DenseNet121', choices=['DenseNet121', 'ResNet50'], type=str,
                        help='Prediction model.')
    parser.add_argument('--trained_model', action='store_true', help='Use a trained model.')
    parser.add_argument('--max_epochs', default=1, type=int, help='How many epochs to train?')
    parser.add_argument('--batch_size', default=64, type=int, help='Which batch size to use?')
    parser.add_argument('--optim', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--betas', default=[0.9, 0.999], type=list, help='Adam Betas')
    parser.add_argument('--eps', default=1e-08, type=float, help='Adam Epsilon')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Adam Weight Decay')
    parser.add_argument('--label', default='Sex', choices=['Sex', 'Age'], type=str, help='Which label to predict.')
    parser.add_argument('--output_dir', default='logs', type=str, help='Where to print tensorboard log.')
    parser.add_argument('--deterministic', default=True)

    return parser


def read_dataset(label, plot_example_images=False):
    train_info = pd.read_csv(f'{DATA_BASE_PATH}/train.csv')
    valid_info = pd.read_csv(f'{DATA_BASE_PATH}/valid.csv')

    train_info = train_info[['Path', label]].rename(columns={label: "Label"})
    valid_info = valid_info[['Path', label]].rename(columns={label: "Label"})

    if label == "Sex":
        train_info.replace({'Female': 0, 'Male': 1}, inplace=True)
        valid_info.replace({'Female': 0, 'Male': 1}, inplace=True)

        train_info = train_info[~(train_info['Label'] == 'Unknown')]
        valid_info = valid_info[~(valid_info['Label'] == 'Unknown')]
    else:
        #max_age = 100
        #print(f'Maximum Age: {max_age}')
        #train_info.Label = train_info.Label.div(max_age)
        #valid_info.Label = valid_info.Label.div(max_age)
        print('Example labels:')
        print(train_info.Label.head(10))

    train_dataset = CheXpertDataset(folder_dir=os.path.split(DATA_BASE_PATH)[0], dataframe=train_info, image_size=image_size, normalization=True)
    valid_dataset = CheXpertDataset(folder_dir=os.path.split(DATA_BASE_PATH)[0], dataframe=valid_info, image_size=image_size, normalization=True)

    if plot_example_images:
        for i in range(2):
            plt.imshow(valid_dataset[i][0][0, :, :], cmap="Greys")
            plt.show()

    return train_dataset, valid_dataset


# --------------------
# Evaluation metrics
# --------------------

def compute_metrics(outputs, targets, losses):
    n_classes = outputs.shape[1]
    fpr, tpr, aucs, precision, recall = {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:,i], outputs[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:,i], outputs[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.mean(0).tolist()))}

    return metrics


# --------------------
# Train and evaluate
# --------------------
@torch.no_grad()
def evaluate(model, dataloader, loss_fn, setup):
    model.eval()

    targets, outputs, losses = [], [], []
    for x, target, idxs in dataloader:
        out = model(x.to(**setup))
        loss = loss_fn(out, target.to(**setup))

        outputs += [out.cpu()]
        targets += [target]
        losses += [loss.cpu()]

    return torch.cat(outputs), torch.cat(targets), torch.cat(losses)


def evaluate_single_model(model, dataloader, loss_fn, setup):
    outputs, targets, losses = evaluate(model, dataloader, loss_fn, setup)
    return compute_metrics(outputs, targets, losses)


def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, n_epochs, writer, setup):
    for epoch in range(n_epochs):
        # train
        model.train()
        for x, target in train_dataloader:
            out = model(x.to(**setup))
            loss = loss_fn(out, target.unsqueeze(1).to(**setup))

            predictions = out.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions.to(**setup) == target.to(**setup)).sum().item()
            accuracy = correct / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if scheduler and args.step >= args.lr_warmup_steps: scheduler.step()

            print(f'loss:{loss.item():.4f}, accuracy:{accuracy:.4f}')

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, setup)
        print(f'Evaluate metrics @ step {epoch}')
        print(f'AUC:{eval_metrics["aucs"]}\n')
        print('Loss:{eval_metrics["loss"]}\n')
        writer.add_scalar('eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
        for k, v in eval_metrics['aucs'].items():
            writer.add_scalar('eval_auc_class_{}'.format(k), v, args.step)

        # save eval metrics
        #save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    setup = inversefed.utils.system_startup(args)

    # not entirely reproducible... only on GPU
    if args.deterministic:
        inversefed.utils.set_deterministic()
        inversefed.utils.set_random_seed(random_seed)

    num_classes = 1
    train_dataset, valid_dataset = read_dataset(args.label)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model == 'DenseNet121':
        model = DenseNet121(num_classes, args.label=='Sex', colour_input='L', pre_trained=True).cuda()
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes, args.label=='Sex', colour_input='L', pre_trained=True).cuda()
    else:
        model = None
        exit('Model not supported')

    Trainer.train(model, train_dataloader, valid_dataloader, vars(args), args.output_dir, use_gpu=True)
