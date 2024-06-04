from torch import optim


class Client:

    """Class for instantiating a single client.
    Mainly for tracking relevant objects during the course of training for each client.
    Init args:
        name (str): Name of client."""

    def __init__(self, name):

        """Placeholders for attributes."""

        self.name = name

        # datasets and loaders
        # dataloaders track changes in associated datasets
        # so we need to uniquely associate constant datasets with the client
        self.train_data = None
        self.train_loader = None
        self.val_data = None
        self.val_loader = None
        self.test_data = None
        self.test_loader = None

        self.n_data = None # size of training dataset
        self.output_path = None # name of output path for storing results
        self.selected_rounds = 0 # counter for rounds where client was selected

        # local model objects
        self.model_params = None # state dict of model
        self.model = None
        self.optimizer = None

        # individual privacy objects/parameters
        self.privacy_engine = None
        self.delta = None
        self.grad_norm = []

    def init_optimizer(self, train_cfg):

        """Initialize client optimizer and set it as client attribute.
        Args:
            cfg (DictConfig): Training config dictionary with optimizer information.
        Returns nothing."""

        if self.model is not None:
            if train_cfg.optimizer == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=train_cfg.lr, # setting optimizer & scheduler
                                            weight_decay=train_cfg.weight_decay)
            if train_cfg.optimizer.startswith("SGD"):
                self.optimizer = optim.SGD(self.model.parameters(), lr=train_cfg.lr,
                                           momentum=(0.9 if train_cfg.optimizer == "SGDM" else 0.0))
        else:
            print("self.model is currently None. Optimizer cannot be initialized.")

    def get_data_len(self):

        """Return number of data points (int) currently held by client, all splits taken together."""

        n_data = 0
        for data in [self.train_data, self.val_data, self.test_data]:
            if data is not None:
                n_data += len(data)

        return n_data
