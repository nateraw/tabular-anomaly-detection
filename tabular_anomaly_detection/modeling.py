from argparse import Namespace
from typing import List, Tuple, Union
import requests

import torch
import torch.nn.functional as F
from huggingface_hub import ModelHubMixin
from torch import nn
import numpy as np
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from typing import Dict, Optional
from huggingface_hub.constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import cached_download, hf_hub_url
import json
import logging
import os

logger = logging.getLogger(__name__)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, activation=nn.LeakyReLU, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        self.activation = activation(**kwargs) if activation is not None else None

    def forward(self, x):
        if self.activation is None:
            return self.fc(x)
        return self.activation(self.fc(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, *dims):
        super().__init__()
        dims = (input_dim,) + dims
        self.layers = nn.Sequential(
            *[Dense(dims[i], dims[i + 1], negative_slope=0.4, inplace=True) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, output_dim, *dims):
        super().__init__()
        self.layers = nn.Sequential(
            *[Dense(dims[i], dims[i + 1], negative_slope=0.4, inplace=True) for i in range(len(dims) - 1)]
            + [Dense(dims[-1], output_dim, activation=None)]
        )

    def forward(self, x):
        return self.layers(x)


class PriorMixture:
    def __init__(self, tau: int = 5, radius: float = 0.8, sigma: float = 0.01):
        x_centroid = (radius * np.sin(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
        y_centroid = (radius * np.cos(np.linspace(0, 2 * np.pi, tau, endpoint=False)) + 1) / 2
        mu_gauss = np.vstack([x_centroid, y_centroid]).T
        mix = Categorical(torch.ones(tau))
        comp = Independent(Normal(torch.tensor(mu_gauss, dtype=torch.float32), torch.tensor(sigma)), 1)
        self.prior = MixtureSameFamily(mix, comp)

    def __call__(self, num_samples):
        return self.prior.sample([num_samples])

    def sample(self, *args, **kwargs):
        return self.prior.sample(*args, **kwargs)

class AdversarialAutoencoder(nn.Module, ModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        autoencoder_dims=(256, 64, 16, 4, 2),
        discriminator_dims=(2, 256, 16, 4, 1),
        tau: int = 5,
        radius: float = 0.8,
        sigma: float = 0.01
    ):
        super().__init__()
        self.config = {
            'input_dim': input_dim,
            'autoencoder_dims': autoencoder_dims,
            'discriminator_dims': discriminator_dims,
            'tau': tau,
            'radius': radius,
            'sigma': sigma,
        }
        self.encoder = Encoder(input_dim, *autoencoder_dims)
        self.decoder = Decoder(input_dim, *reversed(autoencoder_dims))
        self.discriminator = Decoder(discriminator_dims[0], *discriminator_dims[:-1])
        self.prior = PriorMixture(tau, radius, sigma)
        self.reconstruction_criterion_categorical = nn.BCEWithLogitsLoss()
        self.reconstruction_criterion_numeric = nn.MSELoss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()

    def forward(self, cat_batch, num_batch):
        latent = self.encoder(torch.cat((cat_batch, num_batch), dim=1))
        recon = self.decoder(latent)
        recon_loss_cat = self.reconstruction_criterion_categorical(
            input=recon[:, :cat_batch.shape[-1]], target=cat_batch
        )
        recon_loss_num = self.reconstruction_criterion_numeric(input=recon[:, cat_batch.shape[-1] :], target=num_batch)
        recon_loss = recon_loss_cat + recon_loss_num
        return latent, recon, recon_loss

    def discriminate(self, x):
        return self.discriminator(x)

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, config=self.config, **kwargs)

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
