import pytorch_lightning as pl
import torch
import numpy as np
from torch import optim
from torch import nn
from torch.distributions import Independent, Normal, MixtureSameFamily, Categorical

from .modeling import Encoder, Decoder


class AdversarialAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        autoencoder_dims=(256, 64, 16, 4, 2),
        discriminator_dims=(2, 256, 16, 4, 1),
        learning_rate_enc: float = 1e-3,
        learning_rate_dec: float = 1e-3,
        learning_rate_dis: float = 1e-5,
        tau: int = 5,
        radius: float = 0.8,
        sigma: float = 0.01,
        predictions_file: str = 'predictions.pt',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(self.hparams.input_dim, *self.hparams.autoencoder_dims)
        self.decoder = Decoder(self.hparams.input_dim, *reversed(self.hparams.autoencoder_dims))
        self.discriminator = Decoder(self.hparams.discriminator_dims[0], *self.hparams.discriminator_dims[:-1])

        self.reconstruction_criterion_categorical = nn.BCEWithLogitsLoss()
        self.reconstruction_criterion_numeric = nn.MSELoss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()

        self.automatic_optimization=False
        self._init_prior()

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        encoder_optimizer, decoder_optimizer, discriminator_optimizer = self.optimizers()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()


        batch_cat, batch_num = batch
        batch_features = torch.cat((batch_cat, batch_num), 1)

        # 1. Train Autoencoder - How well can it reconstruct inputs?

        latent = self.encoder(batch_features)
        recon = self.decoder(latent)
        recon_loss_cat = self.reconstruction_criterion_categorical(input=recon[:, :batch_cat.shape[-1]], target=batch_cat)
        recon_loss_num = self.reconstruction_criterion_numeric(input=recon[:, batch_cat.shape[-1]:], target=batch_num)
        recon_loss = recon_loss_cat + recon_loss_num

        self.manual_backward(recon_loss)
        self.log('r_loss', recon_loss, prog_bar=True)
        decoder_optimizer.step()
        encoder_optimizer.step()

        # 2. Train Discriminator - how well can it tell a sampled prior from the generated latent space?
        # Sampled prior is real sample (labels = 1), Latent space is fake sample (labels = 0)

        self.discriminator.eval()

        sampled_prior = self.prior.sample([batch_features.shape[0]]).to(self.device)
        pred_prior = self.discriminator(sampled_prior)
        pred_prior_target = torch.ones(pred_prior.shape).to(self.device)
        discriminator_loss_real = self.discriminator_criterion(target=pred_prior_target, input=pred_prior)

        latent = self.encoder(batch_features)
        pred_latent = self.discriminator(latent)
        pred_latent_target = torch.zeros(pred_latent.shape).to(self.device)
        discriminator_loss_fake = self.discriminator_criterion(target=pred_latent_target, input=pred_latent) 

        discriminator_loss = discriminator_loss_fake + discriminator_loss_real

        self.manual_backward(discriminator_loss)
        self.log('d_loss', discriminator_loss, prog_bar=True)
        discriminator_optimizer.step()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        # 3. Train Generator - How well can it generate samples that fool the discriminator?

        latent = self.encoder(batch_features)
        pred_latent = self.discriminator(latent)
        pred_latent_target = torch.FloatTensor(torch.ones(pred_latent.shape)).to(self.device) # fake -> 1
        generator_loss = self.discriminator_criterion(target=pred_latent_target, input=pred_latent)

        self.manual_backward(generator_loss)
        self.log('g_loss', generator_loss, prog_bar=True)
        encoder_optimizer.step()

        return {'latent': latent}

    def training_step_end(self, outputs):
        return outputs

    def test_step(self, batch, batch_idx):
        batch_features = torch.cat(batch, 1)
        latent = self.encoder(batch_features)
        recon = self.decoder(latent)
        self.write_prediction('latent', latent, self.hparams.predictions_file)
        self.write_prediction('recon', recon, self.hparams.predictions_file)

    def configure_optimizers(self):
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.hparams.learning_rate_enc)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.hparams.learning_rate_dec)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate_dis)
        return [encoder_optimizer, decoder_optimizer, discriminator_optimizer]

    def _init_prior(self):
        x_centroid = (self.hparams.radius * np.sin(np.linspace(0, 2 * np.pi, self.hparams.tau, endpoint=False)) + 1) / 2
        y_centroid = (self.hparams.radius * np.cos(np.linspace(0, 2 * np.pi, self.hparams.tau, endpoint=False)) + 1) / 2
        mu_gauss = np.vstack([x_centroid, y_centroid]).T
        mix = Categorical(torch.ones(self.hparams.tau,))
        comp = Independent(Normal(torch.tensor(mu_gauss, dtype=torch.float32), torch.tensor(self.hparams.sigma)), 1)
        self.prior = MixtureSameFamily(mix, comp)
