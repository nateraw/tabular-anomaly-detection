import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Callback


class LatentSpaceVisualizationCallback(Callback):

    def __init__(self, latent_sample_limit: int = 50000):
        self.latent_sample_limit = latent_sample_limit

    def on_train_epoch_start(self, trainer, pl_module):
        self.latent_samples = None
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        if self.latent_samples is not None and self.latent_samples.shape[0] >= self.latent_sample_limit:
            return

        latent = outputs['latent'].detach().cpu()

        if self.latent_samples == None:
            self.latent_samples = latent
        elif self.latent_samples.shape[0] >= self.latent_sample_limit:
            pass
        else:
            self.latent_samples = torch.cat((self.latent_samples, latent))
    
    def on_train_epoch_end(self, trainer, pl_module):
        samples = pl_module.prior.sample([1000])
        fig = plt.figure()
        plt.scatter(self.latent_samples[:, 0], self.latent_samples[:, 1], c='C0', marker="o", edgecolors='w', linewidth=0.5)
        plt.scatter(samples[:, 0], samples[:, 1], c='C1', marker="o", edgecolors='w', linewidth=0.5)
        pl_module.logger.experiment.add_figure('latent_viz', fig, pl_module.global_step)
        plt.close()
