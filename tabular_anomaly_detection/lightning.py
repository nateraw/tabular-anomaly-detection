import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from huggingface_hub import ModelHubMixin
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import cached_download, hf_hub_url
from torch import nn, optim
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal

from .modeling import Decoder, Encoder

logger = logging.getLogger(__name__)


class AdversarialAutoencoderLightning(pl.LightningModule, ModelHubMixin):
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
        predictions_file: str = "predictions.pt",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(self.hparams.input_dim, *self.hparams.autoencoder_dims)
        self.decoder = Decoder(self.hparams.input_dim, *reversed(self.hparams.autoencoder_dims))
        self.discriminator = Decoder(self.hparams.discriminator_dims[0], *self.hparams.discriminator_dims[:-1])

        self.reconstruction_criterion_categorical = nn.BCEWithLogitsLoss()
        self.reconstruction_criterion_numeric = nn.MSELoss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()

        self.automatic_optimization = False
        self._init_prior()

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        (
            encoder_optimizer,
            decoder_optimizer,
            discriminator_optimizer,
        ) = self.optimizers()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        batch_cat, batch_num = batch
        batch_features = torch.cat((batch_cat, batch_num), 1)

        # 1. Train Autoencoder - How well can it reconstruct inputs?

        latent = self.encoder(batch_features)
        recon = self.decoder(latent)
        recon_loss_cat = self.reconstruction_criterion_categorical(
            input=recon[:, : batch_cat.shape[-1]], target=batch_cat
        )
        recon_loss_num = self.reconstruction_criterion_numeric(input=recon[:, batch_cat.shape[-1] :], target=batch_num)
        recon_loss = recon_loss_cat + recon_loss_num

        self.manual_backward(recon_loss)
        self.log("r_loss", recon_loss, prog_bar=True)
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
        self.log("d_loss", discriminator_loss, prog_bar=True)
        discriminator_optimizer.step()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

        # 3. Train Generator - How well can it generate samples that fool the discriminator?

        latent = self.encoder(batch_features)
        pred_latent = self.discriminator(latent)
        pred_latent_target = torch.FloatTensor(torch.ones(pred_latent.shape)).to(self.device)  # fake -> 1
        generator_loss = self.discriminator_criterion(target=pred_latent_target, input=pred_latent)

        self.manual_backward(generator_loss)
        self.log("g_loss", generator_loss, prog_bar=True)
        encoder_optimizer.step()

        return {"latent": latent}

    def training_step_end(self, outputs):
        return outputs

    def test_step(self, batch, batch_idx):
        batch_features = torch.cat(batch, 1)
        latent = self.encoder(batch_features)
        recon = self.decoder(latent)
        self.write_prediction("latent", latent, self.hparams.predictions_file)
        self.write_prediction("recon", recon, self.hparams.predictions_file)

    def configure_optimizers(self):
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.hparams.learning_rate_enc)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.hparams.learning_rate_dec)
        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate_dis)
        return [encoder_optimizer, decoder_optimizer, discriminator_optimizer]

    def _init_prior(self):
        x_centroid = (self.hparams.radius * np.sin(np.linspace(0, 2 * np.pi, self.hparams.tau, endpoint=False)) + 1) / 2
        y_centroid = (self.hparams.radius * np.cos(np.linspace(0, 2 * np.pi, self.hparams.tau, endpoint=False)) + 1) / 2
        mu_gauss = np.vstack([x_centroid, y_centroid]).T
        mix = Categorical(
            torch.ones(
                self.hparams.tau,
            )
        )
        comp = Independent(
            Normal(
                torch.tensor(mu_gauss, dtype=torch.float32),
                torch.tensor(self.hparams.sigma),
            ),
            1,
        )
        self.prior = MixtureSameFamily(mix, comp)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, config=self.hparams, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str],
        strict: bool = True,
        map_location: Optional[str] = "cpu",
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration from huggingface-hub.
        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
        train the model, you should first set it back in training mode with ``model.train()``.
        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Can be either:
                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - You can add `revision` by appending `@` at the end of model_id simply like this: ``dbmdz/bert-base-german-cased@main``
                      Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id,
                      since we use a git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            model_kwargs (:obj:`Dict`, `optional`)::
                model_kwargs will be passed to the model during initialization
        .. note::
            Passing :obj:`use_auth_token=True` is required when you want to use a private model.
        """

        model_id = pretrained_model_name_or_path
        map_location = torch.device(map_location)

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

        if os.path.isdir(model_id) and CONFIG_NAME in os.listdir(model_id):
            config_file = os.path.join(model_id, CONFIG_NAME)
        else:
            try:
                config_url = hf_hub_url(model_id, filename=CONFIG_NAME, revision=revision)
                config_file = cached_download(
                    config_url,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
            except requests.exceptions.RequestException:
                logger.warning("config.json NOT FOUND in HuggingFace Hub")
                config_file = None

        if os.path.isdir(model_id):
            print("LOADING weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_url = hf_hub_url(model_id, filename=PYTORCH_WEIGHTS_NAME, revision=revision)
            model_file = cached_download(
                model_url,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_kwargs.update(config)

        model = cls(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model
