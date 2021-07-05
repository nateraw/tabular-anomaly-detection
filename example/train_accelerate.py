from tabular_anomaly_detection.modeling import AdversarialAutoencoder, PriorMixture
from tabular_anomaly_detection import TabularCollator, TabularFeatureExtractor
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from accelerate import Accelerator
import torch
import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


# Model Args
model_name_or_path: str = None
autoencoder_dims = (256, 64, 16, 4, 2)
discriminator_dims = (2, 256, 16, 4, 1)
learning_rate_enc: float = 1e-3
learning_rate_dec: float = 1e-3
learning_rate_dis: float = 1e-5
tau: int = 7
radius: float = 0.8
sigma: float = 0.01

# Data Args
data_path = 'fraud_dataset_v2.csv'
categorical_cols = ["KTOSL", "PRCTR", "BSCHL", "HKONT", "BUKRS", "WAERS"]
numeric_cols = ["DMBTR", "WRBTR"]
label_col = None
batch_size = 256

# Training Args
num_epochs = 100
seed = 42
use_fp16 = True
use_cpu = False
update_progress_bar_every = 20
latent_sample_limit = 50000
save_every = 5
save_dir = 'outputs/'
Path(save_dir).mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def training_function():

    # Initialize Acclerator
    accelerator = Accelerator(fp16=use_fp16, cpu=use_cpu)

    seed_everything(seed)

    # Init Dataset
    ds = load_dataset("csv", data_files=data_path)["train"]
    feature_extractor = TabularFeatureExtractor(
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols,
        label_column=label_col,
    ).fit(ds)
    ds = ds.map(feature_extractor, batched=True, remove_columns=ds.column_names)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TabularCollator(feature_extractor),
    )

    # Init Models
    model = AdversarialAutoencoder(
        feature_extractor.feature_dim,
        autoencoder_dims,
        discriminator_dims,
        tau,
        radius,
        sigma
    ) if model_name_or_path is None else AdversarialAutoencoder.from_pretrained(model_name_or_path)

    encoder_optimizer = Adam(model.encoder.parameters(), lr=learning_rate_enc)
    decoder_optimizer = Adam(model.decoder.parameters(), lr=learning_rate_dec)
    discriminator_optimizer = Adam(model.discriminator.parameters(), lr=learning_rate_dis)

    # Prepare everything
    model, encoder_optimizer, decoder_optimizer, discriminator_optimizer, loader = accelerator.prepare(
        model,
        encoder_optimizer,
        decoder_optimizer,
        discriminator_optimizer,
        loader
    )

    progress_bar = tqdm(range(num_epochs * len(loader)), disable=not accelerator.is_main_process)
    summary_writer = SummaryWriter(f"{save_dir}/runs")

    for epoch in range(num_epochs):
        progress_bar.set_description_str("Epoch: %s" % epoch)

        model.train()

        latent_samples = None

        for step, batch in enumerate(loader):
            global_step = epoch * len(loader) + step

            model.encoder.zero_grad()
            model.decoder.zero_grad()
            model.discriminator.zero_grad()

            cat_batch, num_batch = batch
            batch_features = torch.cat((cat_batch, num_batch), 1)

            # 1. Train Autoencoder - How well can it reconstruct inputs?
            latent, recon, recon_loss = model(cat_batch, num_batch)

            accelerator.backward(recon_loss)
            decoder_optimizer.step()
            encoder_optimizer.step()

            # 2. Train Discriminator - how well can it tell a sampled prior from the generated latent space?
            # Sampled prior is real sample (labels = 1), Latent space is fake sample (labels = 0)

            model.discriminator.eval()

            sampled_prior = model.prior.sample([batch_features.shape[0]]).to(accelerator.device)
            pred_prior = model.discriminator(sampled_prior)
            pred_prior_target = torch.ones(pred_prior.shape).to(accelerator.device)
            discriminator_loss_real = model.discriminator_criterion(target=pred_prior_target, input=pred_prior)

            latent = model.encoder(batch_features)
            pred_latent = model.discriminator(latent)
            pred_latent_target = torch.zeros(pred_latent.shape).to(accelerator.device)
            discriminator_loss_fake = model.discriminator_criterion(target=pred_latent_target, input=pred_latent)

            discriminator_loss = discriminator_loss_fake + discriminator_loss_real

            accelerator.backward(discriminator_loss)
            discriminator_optimizer.step()
            model.encoder.zero_grad()
            model.decoder.zero_grad()
            model.discriminator.zero_grad()

            # 3. Train Generator - How well can it generate samples that fool the discriminator?

            latent = model.encoder(batch_features)
            pred_latent = model.discriminator(latent)
            pred_latent_target = torch.FloatTensor(torch.ones(pred_latent.shape)).to(accelerator.device)
            generator_loss = model.discriminator_criterion(target=pred_latent_target, input=pred_latent)

            accelerator.backward(generator_loss)
            encoder_optimizer.step()

            metrics = {'r_loss': recon_loss.item(), 'd_loss': discriminator_loss.item(), 'g_loss': generator_loss.item()}

            if step % update_progress_bar_every == 0:
                progress_bar.set_postfix(ordered_dict=None, refresh=True, **metrics)
                progress_bar.update(update_progress_bar_every)

            for k, v in metrics.items():
                summary_writer.add_scalar(f"train/{k}", v, global_step)

            if latent_samples is None:
                latent_samples = latent.detach().cpu()
            elif latent_samples.shape[0] > latent_sample_limit:
                pass
            else:
                latent_samples = torch.cat((latent_samples, latent.detach().cpu()))

        # Add remaining steps that weren't updated to pbar
        progress_bar.update(step % update_progress_bar_every)

        # Save a figure to tensorboard logs
        prior_samples = model.prior.sample([1000])
        fig = plt.figure()
        plt.scatter(
            latent_samples[:, 0],
            latent_samples[:, 1],
            c="C0",
            marker="o",
            edgecolors="w",
            linewidth=0.5,
        )
        plt.scatter(
            prior_samples[:, 0],
            prior_samples[:, 1],
            c="C1",
            marker="o",
            edgecolors="w",
            linewidth=0.5,
        )
        summary_writer.add_figure("latent_viz", fig, global_step)
        plt.close()

        # Save model
        if (epoch + 1) % save_every == 0:
            model.save_pretrained(f"{save_dir}/epoch_{epoch}/")

    model.save_pretrained(f"{save_dir}/final/")
    progress_bar.close()
    summary_writer.close()


if __name__ == '__main__':
    training_function()
