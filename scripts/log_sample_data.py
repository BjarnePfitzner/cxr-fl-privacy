import wandb
from omegaconf import OmegaConf
import torchvision.transforms as transforms

from train_FL import init_clients


wandb.init(project='CXR_Reconstruction',
           entity="bjarnepfitzner",
           name="Sample Data",
           settings=wandb.Settings(start_method="fork"))

chexpert_client_ids = [0]
mendeley_client_ids = [0]

data_cfg = OmegaConf.create({
    "resize": 224,
    "augment": False,
    "front_lat": "frontal",
    "class_idx": [0],
    "policy": "zeros",
    "colour_input": "L",
})
clients = init_clients(data_cfg, 1, 'sum', False, True,
                       chexpert_client_ids, mendeley_client_ids)

# show images for
for client in clients:
    observed_labels = []
    for single_img in client.test_data:
        if single_img[1] not in observed_labels:
            observed_labels.append(single_img[1])
            #wandb.log({f"{client.name}/{single_img[1]}": wandb.Image(transforms.ToPILImage()(single_img[0]))})
            wandb.log({f"{client.name}/{single_img[1]}": wandb.Image(single_img[0])})
        if len(observed_labels) == 2:
            break
# for batch in clients[0].train_loader:
#     .show()
#     logging.debug(batch[1][0])
#
#  for batch in clients[1].val_loader:
#     transforms.ToPILImage()(batch[0][0]).show()
#     logging.debug(batch[1][0])

# get labels and indices of data from dataloaders
# modify chexpert_data dataloader to also return indices for this
# for i in [12,13,14,15,16,17,18,19]:
#     logging.debug("Client ", i)
#     for batch in clients[i].train_loader:
#         logging.debug("Labels: ", batch[1])
#         logging.debug("Inidces: ", batch[2])
