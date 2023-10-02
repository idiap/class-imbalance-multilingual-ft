# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    PreTrainedTokenizer,
)
import os
from hfpl import HFModelForPl
import datasets
import shap
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    model = parser.add_argument_group("model")
    model.add_argument("--model_path", type=str, default=None)
    model.add_argument("--model_name", type=str, default=None)
    model.add_argument("--pl_model_path", type=str, default=None)

    data = parser.add_argument_group("data")
    data.add_argument("--dataset_path", type=str, default=None)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--text_pair", type=str, default=None)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--subsample", type=int, default=None)
    return parser.parse_args()


def find_best_model(path_to_model):
    # find the best model
    best_model = None
    best_score = -1
    for model in os.listdir(path_to_model):
        if "ckpt" in model:
            score = float(model.split("=")[-1].split(".ckpt")[0])
            if score > best_score:
                best_score = score
                best_model = model
    return os.path.join(path_to_model, best_model)


def main():
    args = parse_args()

    # load the emotion dataset
    dataset = datasets.load_from_disk(args.dataset_path)
    dataset = dataset[args.split]
    data = pd.DataFrame(
        {"text": dataset[args.text_col], "label": dataset[args.label_col]}
    )

    if args.subsample is not None:
        data = data.sample(args.subsample, random_state=0)

    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    # load the model and tokenizer

    if os.path.isdir(args.pl_model_path):
        path_to_model = find_best_model(args.pl_model_path)
        output_path = os.path.join(args.pl_model_path, "shap_values.pkl")
    else:
        path_to_model = args.pl_model_path
        output_path = os.path.join(
            os.path.dirname(args.pl_model_path), "shap_values.pkl"
        )

    model = HFModelForPl.load_from_checkpoint(path_to_model, map_location=device).lm
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    # build a pipeline object to do predictions
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    explainer = shap.Explainer(pred)
    shap_values = explainer(data["text"])

    if args.output_path is not None:
        output_path = args.output_path
    pickle.dump(shap_values, open(output_path, "wb"))
    print("SHAP values saved to", output_path)

if __name__ == "__main__":
    main()
