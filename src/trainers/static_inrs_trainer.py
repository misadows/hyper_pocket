import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid, save_image
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from firelab.base_trainer import BaseTrainer
from firelab.config import Config
from tqdm import tqdm
from torchvision.utils import save_image

from src.models.inrs import StaticINRs
from src.utils.training_utils import construct_optimizer, construct_scheduler
from src.dataloaders.load_data import load_data


class StaticINRsTrainer(BaseTrainer):
    """
    This trainer trains a series of INR modules in a batch
    These INRs can have some shared parameters.
    We need it to debug an INR architecture/init
    """
    def __init__(self, config: Config):
        super(StaticINRsTrainer, self).__init__(config)

        print(self.config)

    def init_models(self):
        self.model = StaticINRs(self.config).to(self.device_name)

    def init_dataloaders(self):
        self.random = np.random.RandomState(self.config.random_seed)
        dataset = load_data(self.config.data)
        self.images = [dataset[i][0] for i in self.random.randint(0, len(dataset), self.config.hp.num_inrs)]
        self.images = torch.stack(self.images).to(self.device_name)

        self.save_images(self.images, 'GT')

    def init_optimizers(self):
        self.optim = construct_optimizer(self.model, self.config.hp.optim)
        self.scheduler = construct_scheduler(self.optim, self.config.hp.scheduler)

    def after_init_hook(self):
        self.input_coords = self.model.inrs.generate_input_coords(self.config.hp.num_inrs, self.config.data.target_img_size)
        self.input_coords = self.input_coords.to(self.device_name)

        self.writer.add_text('config', self.config.to_markdown(), self.num_iters_done)

    def _run_training(self):
        iterator = tqdm(range(self.config.hp.max_num_epochs))

        for epoch in iterator:
            self.train_on_batch(self.input_coords)
            if self.config.logging.log_activations_freq > 0 and self.num_iters_done % self.config.logging.log_activations_freq == 0:
                self.log_activations()

            self.num_iters_done += 1
            self.num_epochs_done += 1
            self.on_epoch_done()

        self.model.eval()
        print(f'Final MSE loss: {F.l1_loss(self.compute_preds(self.input_coords), self.images):.03f}')

    def train_on_batch(self, input_coords: Tensor):
        preds = self.compute_preds(input_coords)
        loss = F.l1_loss(preds, self.images)

        self.optim.zero_grad()
        loss.backward()
        if self.config.hp.has('optim.grad_clip_val'):
            grad_norm = clip_grad_norm_(self.model.parameters(), self.config.hp.optim.grad_clip_val)
            self.writer.add_scalar('grad_norm', grad_norm, self.num_iters_done)
        self.optim.step()

        self.writer.add_scalar('loss/train', loss.item(), self.num_iters_done)

    def on_epoch_done(self):
        self._try_to_validate()
        self.scheduler.step()

    @torch.no_grad()
    def log_activations(self):
        _, activations = self.model(self.input_coords, return_activations=True)

        for act_log_name, activations_values in activations.items():
            self.writer.add_histogram(f'activations/{act_log_name}', activations_values, self.num_iters_done)

    def compute_preds(self, input_coords: Tensor):
        if self.config.hp.inr.type == 'hier_fourier_inr':
            preds = self.model.inrs.generate_image(self.model.external_params, self.config.data.target_img_size)
        else:
            preds = self.model(input_coords)
        preds = preds.view(preds.shape[0], preds.shape[1], self.config.data.target_img_size, self.config.data.target_img_size)

        return preds

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        preds = self.compute_preds(self.input_coords)
        loss = F.l1_loss(preds, self.images)

        self.writer.add_scalar('loss/val', loss.item(), self.num_epochs_done)
        self.save_images(preds, 'preds')

    def save_images(self, images: Tensor, tag: str):
        """
        Images: [batch_size, n_channels, img_size, img_size]
        """
        images = images.cpu() / 2 + 0.5 # [-1, 1] => [0, 1]
        samples_grid = make_grid(images, nrow=8)
        self.writer.add_image(f'images/{tag}', samples_grid, self.num_iters_done)
        # save_image(samples_grid, os.path.join(self.paths.custom_data_path, f'samples_iter_{self.num_iters_done}.png'))
