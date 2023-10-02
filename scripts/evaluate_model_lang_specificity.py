# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import v_measure_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns
import datasets
import pickle
import os

# from optimum.pipelines import pipeline
from transformers import (
    AutoModel,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import wandb
import sys

import hfpl
from hfpl import HFModelForPl
import argparse
import torch
from torch import inference_mode
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import FeatureExtractionPipeline


class PoolerFeatureExtractionPipeline(FeatureExtractionPipeline):
    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs[1]
        if self.framework == "pt":
            return model_outputs[1].tolist()
        elif self.framework == "tf":
            return model_outputs[1].numpy().tolist()

def main():
    args = parse_args()

    # Load the dataset
    dataset = datasets.load_from_disk(
        args.dataset_path,
    )
    if args.dataset_split is not None:
        dataset = dataset[args.dataset_split]
    dataset = dataset.shuffle(seed=0)
    if args.subsample is not None:
        dataset = dataset.select(range(args.subsample))

    if args.languages is not None:
        dataset = dataset.filter(lambda x: x[args.label_col] in args.languages)

    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    # Load the language identification pipeline
    if args.pl_model_path is not None:
        model_name = args.pl_model_path.split("/")[-1]
        model_pl = HFModelForPl.load_from_checkpoint(
            args.pl_model_path, map_location=device
        )
        model = model_pl.lm.bert
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
            device=device,
            tokenize_kwargs={"truncation": True},
        )
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
        name_save_embeddings = (
            os.path.join(
                args.output_path,
                "embeddings",
                args.wandb_run_name + f"{model_name}_embeddings.pkl",
            ),
        )
        # assert args.load_embeddings == name_save_embeddings, "The pickle path must be the same as the one used to save the embeddings"
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
                        args.wandb_run_name + f"{model_name}_embeddings.pkl",
                    ),
                    "wb",
                ),
            )
        else:
            print(
                "Cannot save embeddings if load_embeddings is provided (embeddings are already saved)"
            )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": model_name,
        },
    )

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
    # Save average of the scores across folds
    # wandb.log({"logreg_cv_score": clf.scores_})

    # If args.ovr is not true then the score is the same for all languages and there is no need to log the score for each language
    if not args.ovr:
        scores = clf.scores_
        mean_across_l = np.array(list(scores.values())).mean(axis=1)
        wandb.log({"logreg_cv_score": mean_across_l.mean()})
        wandb.log({"logreg_cv_score_std": mean_across_l.std()})
    else:
        dict_mean = {k: np.mean(score) for k, score in clf.scores_.items()}
        wandb.log(
            wandb.Table(columns=list(dict_mean.keys()), data=list(dict_mean.values()))
        )
        # for l, score in clf.scores_.items():
        #     wandb.log({f"logreg_cv_score_{l}": np.mean(score)})
        #     wandb.log({f"logreg_cv_score_std_{l}": np.std(score)})

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
        wandb.log({"embeddings": wandb.Table(dataframe=df_ebd)})

    # Reduce the dimensionality of the embeddings
    if args.dim_red == "TSNE":
        dim_red = TSNE(n_components=2, perplexity=100)
    elif args.dim_red == "PCA":
        dim_red = PCA(n_components=2)
    else:
        raise ValueError("dim_red must be either TSNE or PCA")
    embeddings_dimred = dim_red.fit_transform(X)

    # Clustering
    df = pd.DataFrame(embeddings_dimred, columns=[0, 1])
    df["language"] = dataset[args.label_col]
    v_measures = []
    for rand_state in range(10):
        kmeans = KMeans(
            n_clusters=df["language"].nunique(), random_state=rand_state, n_init="auto"
        ).fit(df[[0, 1]])
        df["cluster"] = kmeans.labels_
        v_measures.append(v_measure_score(df["language"], df["cluster"]))

    # Save the clustering scores
    wandb.log({"v_measure": np.mean(v_measures)})
    wandb.log({"v_measure_std": np.std(v_measures)})

    # Plot the embeddings using t-SNE
    if args.produce_plot:
        if args.subsample_for_plot is not None:
            df = df.sample(args.subsample_for_plot, random_state=args.seed)
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
                x_min=df[0].min() - 0.1 * (df[0].max() - df[0].min()),
                x_max=df[0].max() + 0.1 * (df[0].max() - df[0].min()),
                y_min=df[1].min() - 0.1 * (df[1].max() - df[1].min()),
                y_max=df[1].max() + 0.1 * (df[1].max() - df[1].min()),
                n_points=1000,
            )

        sns.scatterplot(data=df, x=0, y=1, hue="language", alpha=0.5)
        name = args.wandb_run_name if args.wandb_run_name is not None else model_name
        out_path = os.path.join(args.output_path, "plots", f"{name}_{args.dim_red}.png")
        plt.savefig(out_path)
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--pl_model_path", type=str, default=None)
    model.add_argument("--model_path", type=str, default=None)
    model.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
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
    parser.add_argument("--wandb_project", type=str, default="posterior_lang_analysis")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--upload_embeddings", action="store_true")
    parser.add_argument("--tokenize_pairs", nargs=2, default=None)
    parser.add_argument("--save_embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
