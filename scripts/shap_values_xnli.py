# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os
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

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_col", type=str, default="premise")
    parser.add_argument("--text_pair", type=str, default="hypothesis")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--subsample", type=int, default=None)

    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--job_array_idx", type=int, default=None)
    parser.add_argument("--job_array_total", type=int, default=None)
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
    print("Using model ", path_to_model)
    model = HFModelForPl.load_from_checkpoint(path_to_model, map_location=device).lm
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    print("Model loaded")
    combined_sent = dataset.map(
        lambda x: {
            "combined_sent": x[args.text_col]
            + " "
            + tokenizer.sep_token
            + x[args.text_pair]
        },
        keep_in_memory=True,
    )["combined_sent"]
    data = pd.DataFrame(
        {
            "text": combined_sent,
            "label": dataset[args.label_col],
        }
    )

    print(data["text"][0])
    if args.subsample is not None:
        data = data.sample(args.subsample, random_state=args.seed)

    if args.job_array_total is not None and args.job_array_idx is not None:
        # split the data into chunks
        data = np.array_split(data, args.job_array_total)[args.job_array_idx - 1]
    else:
        raise ValueError("Must specify both job array total and idx")

    # build a pipeline object to do predictions
    def f(x):
        outputs = []
        # print("hi", x)
        for _x in x:
            encoding = torch.tensor([tokenizer.encode(_x)]).to(device)
            # print(encoding.tolist())
            token_type_ids = torch.tensor([[0] * len(encoding[0])]).to(device)
            # set token_type_ids to 1 for the second sentence for sequence pairs
            if tokenizer.sep_token_id in encoding:
                token_type_ids[
                    0, encoding.tolist()[0].index(tokenizer.sep_token_id) + 1 :
                ] = 1
            # print(list(zip(encoding[0], token_type_ids[0])))

            output = (
                model(input_ids=encoding, token_type_ids=token_type_ids)[0]
                .detach()
                .softmax(-1)
                .to("cpu")
                .numpy()
            )
            outputs.append(output[0])
        outputs = np.array(outputs)
        return outputs

    explainer = shap.Explainer(
        f, tokenizer, output_names=["contradiction", "entailment", "neutral"]
    )

    shap_values = explainer(data["text"])
    if args.output_path is not None:
        output_path = args.output_path
        if args.job_array_total is not None and args.job_array_idx is not None:
            if "pck" in output_path:
                output_path = output_path.replace(
                    ".pck", f"_{args.job_array_idx}_{args.job_array_total}.pck"
                )
            else:
                output_path = (
                    output_path + f"_{args.job_array_idx}_{args.job_array_total}.pkl"
                )
    pickle.dump(shap_values, open(output_path, "wb"))
    print("SHAP values saved to", output_path)


if __name__ == "__main__":
    main()
