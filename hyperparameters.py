import torch


class Hyperparameters:
    def __init__(self):
        self.batch_size = 64
        self.block_size = 256
        self.max_iter = 2000
        self.eval_every = 400
        self.learning_rate = 3e-4
        self.eval_iter = 200
        self.d_model = 384
        self.n_head = 6
        self.n_layers = 6
        self.dropout_p = 0.2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_hyperparameters(self):
        return {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "max_iter": self.max_iter,
            "eval_every": self.eval_every,
            "learning_rate": self.learning_rate,
            "eval_iter": self.eval_iter,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_layers": self.n_layers,
            "dropout_p": self.dropout_p,
        }
