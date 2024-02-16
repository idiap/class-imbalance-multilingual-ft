# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
This script is used to fine-tune a HuggingFace model with PyTorch Lightning.
It uses a dataset from the HuggingFace datasets library.
It supports training with class weight for each language, training the language classifier,
and entropy maximization of the language classification.
"""

import argparse
import json
import os
from datetime import datetime
import numpy as np
import random
import pytorch_lightning as pl
import torch
from functools import partial
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import datasets

from hfpl import HFModelForPl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        nargs="+",
        help="""
            Path to the configuration file(s). 
            Arguments in the configuration file(s) will be overwritten by command line arguments
            """,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Name of the HuggingFace model",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to load from",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset from the HuggingFace datasets library",
    )
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset on disk")
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the label column in the dataset",
    )
    parser.add_argument(
        "--lang_label_col",
        type=str,
        default="lang_label",
        help="Name of the language label column in the dataset",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        nargs="+",
        help="Name(s) of the training split(s)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="val",
        nargs="+",
        help="Name(s) of the validation split(s)",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        nargs="+",
        help="Name(s) of the test split(s)",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of labels in the classification task",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--lm_lr", type=float, default=5e-5, help="Learning rate for the language model"
    )
    parser.add_argument(
        "--cls_lr", type=float, default=5e-4, help="Learning rate for the classifier"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Whether to use per-language class weights for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--tokenize_pairs", action="store_true", help="Whether to tokenize input pairs"
    )
    parser.add_argument(
        "--freeze_lm",
        action="store_true",
        default=False,
        help="Whether to freeze the language model during training",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to use early stopping during training",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="val_loss",
        help="Metric to monitor for saving the best model",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--no_train", action="store_true", help="Whether to skip the training phase"
    )
    parser.add_argument(
        "--mask_entropy_max_coef",
        type=float,
        default=0.0,
        help="Coefficient for mask entropy maximization loss",
    )
    parser.add_argument(
        "--mask_entropy_max",
        type=bool,
        default=False,
        help="Whether to apply mask entropy maximization",
    )
    parser.add_argument(
        "--entropy_max_coef",
        type=float,
        default=0.0,
        help="Coefficient for entropy maximization loss of the language classifier",
    )
    parser.add_argument(
        "--gradient_reversal_coef",
        type=float,
        default=0.0,
        help="Coefficient for gradient reversal of the language classifier",
    )
    parser.add_argument(
        "--language_cls",
        action="store_true",
        default=False,
        help="Whether to enable language classification",
    )
    parser.add_argument(
        "--pretrain_lid_head",
        type=int,
        default=0,
        help="Number of pretraining steps for the language identification head",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lang_entropy",
        help="Name of the Weights & Biases project",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the experiment, also used in Weights & Biases",
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="ID of the Weights & Biases run to resume"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)"
    )

    args = parser.parse_args()

    if args.config is not None:
        if isinstance(args.config, str):
            args.config = [args.config]
        for conf_name in args.config:
            with open(conf_name, "r") as f:
                parser.set_defaults(**json.load(f))
        args = parser.parse_args()

    return args


def main():
    # Args parsing and dataset loading

    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # pl.seed_everything(args.seed)

    if args.dataset_name is not None:
        dataset = datasets.load_dataset(args.dataset_name)
    elif args.dataset_path is not None:
        dataset = datasets.load_from_disk(args.dataset_path)
    else:
        raise ValueError("You must specify either a dataset name or a dataset path")
    if args.entropy_max_coef > 0 and not args.language_cls:
        raise ValueError(
            "Entropy loss is only available when language compensation is enabled"
        )
    if args.gradient_reversal_coef > 0 and not args.language_cls:
        raise ValueError(
            "Gradient reversal is only available when language compensation is enabled"
        )

    if isinstance(args.train_split, str):
        args.train_split = [args.train_split]

    if isinstance(args.val_split, str):
        args.val_split = [args.val_split]

    # Dataset preprocessing

    splits_to_keep = []
    splits_to_keep.extend(args.train_split)
    splits_to_keep.extend(args.val_split)

    if isinstance(args.test_split, str):
        splits_to_keep.append(args.test_split)
    else:
        splits_to_keep.extend(args.test_split)

    dataset_only_splits_to_keep = datasets.DatasetDict(
        {split: dataset[split] for split in splits_to_keep}
    )
    dataset = dataset_only_splits_to_keep

    label_encoder = LabelEncoder()
    label_encoder.fit(dataset[args.train_split[0]][args.label_col])

    def encode_with_label_encoder(x):
        return {"label": label_encoder.transform([x[args.label_col]])}

    # print("encode_with_label_encoder hash ", Hasher.hash(encode_with_label_encoder))
    dataset = dataset.map(encode_with_label_encoder)

    label_encoder_lang = LabelEncoder()
    if (args.dataset_path is not None) and ("speech_dataset" in args.dataset_path):
        label_encoder_lang.fit(["DE", "FR", "IT"])
    else:
        all_labels_all_splits = []
        for split in dataset:
            all_labels_all_splits.extend(dataset[split][args.lang_label_col])
        label_encoder_lang.fit(all_labels_all_splits)

    def encode_with_label_encoder_lang(x):
        return {"lang_label": label_encoder_lang.transform([x[args.lang_label_col]])}

    dataset = dataset.map(
        encode_with_label_encoder_lang,
    )

    def select_lang(x, lang, lang_label_col):
        return x[lang_label_col] == lang

    if args.use_class_weights:
        class_weight = {}
        for lang in label_encoder_lang.classes_:
            # print(
            #     "partial function hash ",
            #     Hasher.hash(
            #         partial(select_lang, lang=lang, lang_label_col=args.lang_label_col)
            #     ),
            # )
            class_weight[label_encoder_lang.transform([lang])[0]] = (
                compute_class_weight(
                    "balanced",
                    classes=label_encoder.transform(label_encoder.classes_),
                    y=np.array(
                        dataset[args.train_split[0]].filter(
                            partial(
                                select_lang,
                                lang=lang,
                                lang_label_col=args.lang_label_col,
                            )
                        )["label"]
                    ).squeeze(),
                )
            )
        print("Class weight", class_weight)
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = args.num_labels
    config.num_lang_labels = args.num_lang_labels = len(label_encoder_lang.classes_)
    config.problem_type = None
    config.id2label = {i: int(label) for i, label in enumerate(label_encoder.classes_)}
    config.label2id = {int(label): i for i, label in enumerate(label_encoder.classes_)}

    wandb_name = args.name
    if args.use_class_weights:
        wandb_name += "_cw"
    if args.freeze_lm:
        wandb_name += "_lid"
    if args.entropy_max_coef > 0:
        wandb_name += f"_ent{args.entropy_max_coef}"
    if args.gradient_reversal_coef > 0:
        wandb_name += f"_gr{args.gradient_reversal_coef}"
    if args.mask_entropy_max:
        wandb_name += f"_mask_ent{args.mask_entropy_max_coef}"

    wandb_name += f"_{args.model_name.split('/')[-1]}"

    wandb_logger = WandbLogger(
        save_dir="./wandb",
        project=args.wandb_project,
        name=wandb_name,
        id=args.wandb_id,
        resume="allow",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.tokenize_pairs:

        def tokenize_pairs(x):
            return tokenizer(
                x["premise"],
                x["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        # print("tokenize_pairs hash ", Hasher.hash(tokenize_pairs))
        dataset = dataset.map(
            tokenize_pairs,
            batched=True,
        )
    else:

        def tokenize(x):
            return tokenizer(x[args.text_col], truncation=True)

        # print("tokenize hash ", Hasher.hash(tokenize))
        dataset = dataset.map(
            tokenize,
            batched=True,
        )

    columns_to_change_to_torch = set(
        dataset[args.train_split[0]].column_names
    ).intersection(
        ["input_ids", "attention_mask", "token_type_ids", "label", "lang_label"]
    )
    dataset.set_format(
        type="torch",
        columns=list(columns_to_change_to_torch),
    )

    columns_to_remove = set(dataset[args.train_split[0]].column_names).difference(
        ["input_ids", "attention_mask", "token_type_ids", "label", "lang_label"]
    )
    dataset = dataset.remove_columns(list(columns_to_remove))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_datasets = datasets.concatenate_datasets(
        [dataset[split] for split in args.train_split]
    )

    print(train_datasets)
    print("One datapoint", train_datasets[0])

    val_datasets = datasets.concatenate_datasets(
        [dataset[split] for split in args.val_split]
    )

    total_steps = (
        args.epochs
        * len(train_datasets)
        // (args.batch_size * args.gradient_accumulation_steps)
    )

    pl_model_kwargs = dict(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        pretrain_lid_head=args.pretrain_lid_head,
        train_lm=not args.freeze_lm,
        language_cls=args.language_cls,
        entropy_max_coef=args.entropy_max_coef,
        gradient_reversal_coef=args.gradient_reversal_coef,
        lm_lr=args.lm_lr,
        lid_lr=args.cls_lr,
        cls_lr=args.cls_lr,
        total_steps=total_steps,
        test_dataset_names=args.test_split,
        class_weight=class_weight if args.use_class_weights else None,
        mask_entropy_max=args.mask_entropy_max,
        mask_entropy_max_coef=args.mask_entropy_max_coef,
    )

    if args.load_from_checkpoint is not None:
        if not torch.cuda.is_available():
            model = HFModelForPl.load_from_checkpoint(
                args.load_from_checkpoint,
                map_location=torch.device("cpu"),
                strict=False,
                **pl_model_kwargs,
            )
        else:
            model = HFModelForPl.load_from_checkpoint(
                args.load_from_checkpoint, strict=False, **pl_model_kwargs
            )
    else:
        model = HFModelForPl(
            args.model_name,
            model_kwargs={
                "config": config,
            },
            tokenizer=tokenizer,
            **pl_model_kwargs,
        )

    checkpoint_folder = (
        (
            args.dataset_path.split("/")[-1]
            if args.dataset_path is not None
            else args.dataset_name
        )
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    os.makedirs(f"./models/{wandb_name}/{checkpoint_folder}")
    args_dict = vars(args)
    args_dict["label_encoder"] = label_encoder.classes_.tolist()
    args_dict["label_encoder_lang"] = label_encoder_lang.classes_.tolist()
    args_dict["wandb_id"] = wandb_logger.experiment.id

    with open(
        f"./models/{wandb_name}/{checkpoint_folder}/args.json",
        "w",
    ) as f:
        json.dump(args_dict, f)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor=args.metric_for_best_model,
        dirpath=f"./models/{wandb_name}/{checkpoint_folder}",
        filename="model-{epoch:02d}-{" + args.metric_for_best_model + ":.2f}",
        save_top_k=1,
        mode="min",
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(TQDMProgressBar(refresh_rate=50))

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=0.1,
        callbacks=callbacks,
        precision=16,
        logger=wandb_logger,
        deterministic=True,
    )

    if not args.no_train:
        trainer.fit(
            model=model,
            train_dataloaders=DataLoader(
                train_datasets,
                batch_size=args.batch_size,
                collate_fn=data_collator,
                shuffle=True,
                # pin_memory=True,
            ),
            val_dataloaders=DataLoader(
                val_datasets,
                batch_size=args.batch_size,
                collate_fn=data_collator,
                num_workers=0,
                # pin_memory=True,
            ),
        )

    if isinstance(args.test_split, str):
        args.test_split = [args.test_split]

    if args.test_split:
        trainer.test(
            # model=model if args.no_train else checkpoint_callback.best_model_path,
            ckpt_path="best",
            dataloaders=[
                DataLoader(
                    dataset[split],
                    batch_size=args.batch_size,
                    collate_fn=data_collator,
                )
                for split in args.test_split
            ],
        )


if __name__ == "__main__":
    main()
