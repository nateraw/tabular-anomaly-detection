from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import requests
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from tabular_anomaly_detection import (
    AdversarialAutoencoderLightning,
    LatentSpaceVisualizationCallback,
    TabularCollator,
    TabularFeatureExtractor,
)


def get_example_dataset():
    url = "https://raw.githubusercontent.com/GitiHubi/deepAI/master/data/fraud_dataset_v2.csv"
    data_file = Path("fraud_dataset_v2.csv")
    if not data_file.exists():
        r = requests.get(url)
        data_file.write_text(r.text)
    return load_dataset("csv", data_files="fraud_dataset_v2.csv")["train"]


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--categorical_cols",
        type=str,
        default=",".join(["KTOSL", "PRCTR", "BSCHL", "HKONT", "BUKRS", "WAERS"]),
    )
    parser.add_argument(
        "--numeric_cols", type=str, default=",".join(["DMBTR", "WRBTR"])
    )
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed)

    if args.data_path is not None:
        ds = load_dataset("csv", data_files=args.data_path)["train"]
    else:
        ds = get_example_dataset()

    feature_extractor = TabularFeatureExtractor(
        categorical_columns=args.categorical_cols.split(","),
        numeric_columns=args.numeric_cols.split(","),
        label_column=args.label_col,
    ).fit(ds)

    ds = ds.map(feature_extractor, batched=True, remove_columns=ds.column_names)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TabularCollator(feature_extractor),
    )

    model = AdversarialAutoencoderLightning(feature_extractor.feature_dim)

    if args.use_wandb:
        args.logger = WandbLogger(name=f"{datetime.now()}", project="fin-anomaly")

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[LatentSpaceVisualizationCallback()]
    )
    trainer.fit(model, loader)
