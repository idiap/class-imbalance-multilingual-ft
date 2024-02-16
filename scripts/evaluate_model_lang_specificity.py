# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import seaborn as sns
import datasets
import pickle
import os
import json

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import wandb

import hfpl
from hfpl import HFModelForPl
import argparse
import torch
from torch import inference_mode
from transformers.pipelines import FeatureExtractionPipeline


class PoolerFeatureExtractionPipeline(FeatureExtractionPipeline):
    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs[1]
        if self.framework == "pt":
            if "pooler_output" in model_outputs:
                return model_outputs.pooler_output.tolist()
            else:
                return model_outputs.last_hidden_state[:, 0].tolist()
        elif self.framework == "tf":
            return model_outputs[1].numpy().tolist()


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Load the dataset
    dataset = datasets.load_from_disk(
        args.dataset_path,
    )
    if args.dataset_split is not None:
        dataset = dataset[args.dataset_split]
    dataset_name = args.dataset_path.split("/")[-1] + (
        "_" + args.dataset_split if args.dataset_split is not None else ""
    )
    dataset = dataset.shuffle(seed=0)
    if args.subsample is not None:
        dataset = dataset.select(range(args.subsample))

    if args.languages is not None:
        print(f"Languages present in the dataset: {np.unique(dataset[args.label_col])}")
        print(f"Languages to keep: {args.languages}")
        dataset = dataset.filter(lambda x: x[args.label_col] in args.languages)

    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    # Load the language identification pipeline
    if args.pl_model_path is not None:
        model_name = "/".join(args.pl_model_path.split("/")[-3:])
        model_pl = HFModelForPl.load_from_checkpoint(
            args.pl_model_path, map_location=device
        )
        model = model_pl.lm.base_model
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        lid = PoolerFeatureExtractionPipeline(
            model=model,
            device=device,
            tokenizer=tokenizer,
            tokenize_kwargs={"truncation": True},
        )
    elif args.model_path is not None:
        model_name = args.model_path.split("/")[-1]
        model = AutoModel.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        lid = PoolerFeatureExtractionPipeline(
            model=model,
            device=device,
            tokenizer=tokenizer,
            tokenize_kwargs={"truncation": True},
        )
    elif args.model_name is not None:
        model_name = args.model_name
        lid = PoolerFeatureExtractionPipeline(
            model=model_name,
            device=device,
            tokenize_kwargs={"truncation": True},
        )
    elif args.model_folder is not None:
        files_in_folder = os.listdir(args.model_folder)
        # find checkpoint, raise error if there are multiple
        checkpoints = [f for f in files_in_folder if f.endswith(".ckpt")]
        if len(checkpoints) > 1:
            raise ValueError("Multiple checkpoints found in folder")
        elif len(checkpoints) == 0:
            raise ValueError("No checkpoint found in folder")
        else:
            model_pl = HFModelForPl.load_from_checkpoint(
                os.path.join(args.model_folder, checkpoints[0]), map_location=device
            )
            model = model_pl.lm.base_model
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            lid = PoolerFeatureExtractionPipeline(
                model=model,
                device=device,
                tokenizer=tokenizer,
                tokenize_kwargs={"truncation": True},
            )
            model_name = model_pl.lm.config._name_or_path

            if "args.json" not in files_in_folder:
                raise ValueError("No args.json file found in folder")
            else:
                args_training = json.load(
                    open(os.path.join(args.model_folder, "args.json"))
                )
                print("Using wandb run name and id from training")

    elif args.load_embeddings is None:
        raise ValueError(
            "Either model_path, model_name or load_embeddings must be provided"
        )

    if args.load_embeddings is None:
        embeddings = []
        if args.tokenize_pairs is not None:
            with inference_mode():
                for sample in tqdm(dataset):
                    embeddings.append(
                        lid(
                            sample[args.tokenize_pairs[0]],
                            text_pair=sample[args.tokenize_pairs[1]],
                        )[0]
                    )
        else:
            with inference_mode():
                for sample in tqdm(dataset):
                    embeddings.append(lid(sample[args.text_col])[0])

    else:
        model_name = args.load_embeddings.split("/")[-1]
        embeddings = pickle.load(open(args.load_embeddings, "rb"))

    # Save the embeddings
    if args.save_embeddings:
        if args.load_embeddings is None:
            if not os.path.exists(os.path.join(args.output_path, "embeddings")):
                os.makedirs(os.path.join(args.output_path, "embeddings"))
            pickle.dump(
                embeddings,
                open(
                    os.path.join(
                        args.output_path,
                        "embeddings",
                        args.name + f"{model_name}_embeddings.pkl",
                    ),
                    "wb",
                ),
            )
        else:
            print(
                "Cannot save embeddings if load_embeddings is provided (embeddings already saved)"
            )

    if args.wandb_project is not None:
        using_wandb = True
        wandb.init(
            project=(
                args_training["wandb_project"]
                if args.model_folder is not None
                else args.wandb_project
            ),
            name=args.name if args.name is not None else model_name,
            config={
                "model_name": model_name,
            },
            id=(
                args_training["wandb_id"]
                if args.model_folder is not None
                else args.wandb_id if args.wandb_id is not None else None
            ),
        )
    else:
        # If wandb_project is None then we are not using wandb
        using_wandb = False
        # Instead writing the results to a file in the folder where the model is saved
        if args.model_folder is not None:
            out_path_res = os.path.join(
                args.model_folder, f"results_{dataset_name}.txt"
            )
        elif args.pl_model_path is not None:
            out_path_res = os.path.join(
                os.path.dirname(args.pl_model_path), f"results_{dataset_name}.txt"
            )
        elif args.model_path is not None:
            out_path_res = os.path.join(
                os.path.dirname(args.model_path), f"results_{dataset_name}.txt"
            )
        else:
            raise ValueError("model_folder or model_path must be provided")
        file_res = open(out_path_res, "w")

    model.eval()

    # Trying to predict the language from the embeddings using a logistic regression and CV
    # X = np.array([e[0][0] for e in embeddings])
    X = np.array(embeddings)
    y = dataset[args.label_col]
    clf = LogisticRegressionCV(
        random_state=args.seed,
        max_iter=1000,
        Cs=1,
        n_jobs=2,
        cv=5,
        multi_class="ovr" if args.ovr else "multinomial",
    ).fit(X, y)

    # If args.ovr is not true then the score is the same for all languages
    # Thus, there is no need to log the score for each language
    if not args.ovr:
        scores = clf.scores_
        mean_across_l = np.array(list(scores.values())).mean(axis=1)
        if using_wandb:
            wandb.log({f"{dataset_name}/logreg_cv_score": mean_across_l.mean()})
            wandb.log({f"{dataset_name}/logreg_cv_score_std": mean_across_l.std()})
        else:
            file_res.write(f"Logreg CV score: {mean_across_l.mean()}\n")
            file_res.write(f"Logreg CV score std: {mean_across_l.std()}\n")
    else:
        dict_mean = {k: np.mean(score) for k, score in clf.scores_.items()}
        if using_wandb:
            wandb.log(
                wandb.Table(
                    columns=list(dict_mean.keys()), data=list(dict_mean.values())
                )
            )
        else:
            file_res.write(f"Logreg CV score: {dict_mean}\n")

    if args.upload_embeddings:
        # Create a dataframe with the embeddings and the corresponding language

        df_ebd = pd.Series(embeddings).rename("embeddings").to_frame()
        df_ebd["language"] = dataset[args.label_col]
        if args.pl_model_path is not None:
            preds = (
                model_pl.lm.classifier(torch.tensor(X).to(device).float())
                .cpu()
                .detach()
                .softmax(-1)
                .numpy()
            )
            df_ebd["preds"] = preds.argmax(axis=1)
            if "id2label" in model_pl.lm.config.__dict__:
                for k, v in model_pl.lm.config.id2label.items():
                    df_ebd[f"preds_{v}"] = preds[:, k]
            else:
                for c in range(preds.shape[1]):
                    df_ebd[f"preds_{c}"] = preds[:, c]
            print(df_ebd.head(10))
        # Save the embeddings
        if using_wandb:
            wandb.log({"lang_id/embeddings": wandb.Table(dataframe=df_ebd)})

    # Reduce the dimensionality of the embeddings
    if args.dim_red == "TSNE":
        dim_red = TSNE(n_components=2, perplexity=100)
    elif args.dim_red == "PCA":
        dim_red = PCA(n_components=2)
    else:
        raise ValueError("dim_red must be either TSNE or PCA")

    # Plot the embeddings using t-SNE
    if args.produce_plot:
        if args.subsample_for_plot is not None:
            df_ebd = df_ebd.sample(args.subsample_for_plot, random_state=args.seed)
        plt.figure(figsize=(10, 10))
        if (
            (args.draw_decision_boundary)
            and (args.load_embeddings is None)
            and (args.pl_model_path is not None)
        ):
            print("Plotting decision boundary")
            hfpl.plot_decision_boundary(
                infer_func=lambda x: model_pl.lm.classifier(
                    torch.tensor(x).to(device).float()
                )
                .cpu()
                .detach()
                .numpy(),
                pca=dim_red,
                x_min=df_ebd[0].min() - 0.1 * (df_ebd[0].max() - df_ebd[0].min()),
                x_max=df_ebd[0].max() + 0.1 * (df_ebd[0].max() - df_ebd[0].min()),
                y_min=df_ebd[1].min() - 0.1 * (df_ebd[1].max() - df_ebd[1].min()),
                y_max=df_ebd[1].max() + 0.1 * (df_ebd[1].max() - df_ebd[1].min()),
                n_points=1000,
            )

        sns.scatterplot(data=df_ebd, x=0, y=1, hue="language", alpha=0.5)
        name = args.name if args.name is not None else model_name
        out_path_plot = os.path.join(
            args.output_path, "plots", f"{name}_{args.dim_red}.png"
        )
        plt.savefig(out_path_plot)

    if using_wandb:
        wandb.finish()
    else:
        file_res.close()


def parse_args():
    parser = argparse.ArgumentParser()
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--pl_model_path", type=str, default=None)
    model.add_argument("--model_path", type=str, default=None)
    model.add_argument("--model_name", type=str, default=None)
    model.add_argument(
        "--model_folder",
        type=str,
        default=None,
        help="""Path to folder containing the model 
                and the args.json file as produced by fine_tune_w_pl.py""",
    )
    parser.add_argument("--load_embeddings", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/wikipedia",
    )
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="language")
    parser.add_argument("--dim_red", type=str, default="PCA")
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--produce_plot", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--subsample_for_plot", type=int, default=None)
    parser.add_argument("--draw_decision_boundary", action="store_true")
    parser.add_argument("--ovr", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--languages", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--upload_embeddings", action="store_true")
    parser.add_argument("--tokenize_pairs", nargs=2, default=None)
    parser.add_argument("--save_embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
