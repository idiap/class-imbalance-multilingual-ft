# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datasets
import argparse

LANGUAGE_GROUP = {"de": 0, "en": 0, "zh": 0, "fr": 1, "es": 1, "ja": 1}
LANGUAGE_COLOR = {
    "en": "#00bbff",
    "de": "#0800ff",
    "zh": "#0bd915",
    "fr": "#ffe600",
    "es": "#ff8800",
    "ja": "#ff2f00",
}

sns.set(font_scale=1.6, style="whitegrid")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shap_values_bal",
        type=str,
        required=True,
        help="Path to shap values in pickle format",
    )
    parser.add_argument(
        "--shap_values_imbal",
        type=str,
        required=True,
        help="Path to shap values in pickle format",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the folder to save plots"
    )
    parser.add_argument(
        "--filename", type=str, required=True, help="Filename for the plot"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--dataset_split", type=str, help="Dataset split")
    parser.add_argument("--subsample", type=int, help="Subsample size")
    parser.add_argument(
        "--threshold", type=float, help="Threshold for cumulative sum", default=0.01
    )
    parser.add_argument("--axlim", type=float, help="Limit for x-axis", default=0.2)
    parser.add_argument("--remove_legend", action="store_true", help="Remove legend")
    which_dataset = parser.add_mutually_exclusive_group(required=True)
    which_dataset.add_argument(
        "--amazon",
        action="store_true",
        help="Whether we are using the Amazon dataset (default)",
    )
    which_dataset.add_argument(
        "--xnli", action="store_true", help="Whether we are using the XNLI dataset"
    )
    return parser.parse_args()


def shap_to_pandas(shap_values, data=None):
    df_shap = pd.DataFrame(
        {
            "values": shap_values.values,
            "base_values": list(shap_values.base_values),
            "feature_names": shap_values.feature_names,
        }
    )

    df_shap["overall_shap_values"] = df_shap.apply(
        lambda x: x["values"].sum(axis=0) + x["base_values"], axis=1
    )
    df_shap["predictions"] = df_shap["overall_shap_values"].apply(
        lambda x: np.argmax(x)
    )

    if data is not None:
        df_shap["language"] = data.reset_index()["language"]
        df_shap["language_group"] = df_shap["language"].map(LANGUAGE_GROUP)

    return df_shap


def print_prediction_distribution(df):
    df_copy = df.copy()
    df_copy["predictions"] = df_copy["overall_shap_values"].apply(
        lambda x: np.argmax(x)
    )
    df_copy["language_group"] = df_copy["language"].map(LANGUAGE_GROUP)
    # round to 1 decimal and multiply by 100
    # then format by assing percentage sign and & in between to be latex reads
    print(
        df_copy.groupby("language_group")["predictions"]
        .value_counts(normalize=True)
        .apply(lambda x: round(x * 100, 1))
        .apply(lambda x: str(x) + "\%")
        .unstack()
        .apply(lambda x: " & ".join(x), axis=1)
    )


def shap_bal_and_imbal_to_pandas(shap_values_bal, shap_values_imbal, data=None):
    df_shap_bal = shap_to_pandas(shap_values_bal, data=data)
    df_shap_imbal = shap_to_pandas(shap_values_imbal, data=data)

    print("Balanced prediction distribution")
    print_prediction_distribution(df_shap_bal)
    print("Imbalanced prediction distribution")
    print_prediction_distribution(df_shap_imbal)

    df_shap_all = pd.concat(
        [
            df_shap_bal.add_suffix("_bal"),
            df_shap_imbal.add_suffix("_imbal"),
        ],
        axis=1,
    )
    if (
        "language_bal" in df_shap_all.columns
        and "language_imbal" in df_shap_all.columns
    ):
        # Sanity check: language should be the same
        assert df_shap_all["language_bal"].equals(df_shap_all["language_imbal"])
        assert df_shap_all["language_group_bal"].equals(
            df_shap_all["language_group_imbal"]
        )
        df_shap_all.rename(
            columns={
                "language_bal": "language",
                "language_group_bal": "language_group",
            },
            inplace=True,
        )
        df_shap_all.drop(
            columns=["language_imbal", "language_group_imbal"], inplace=True
        )

    df_shap_all["diff_shap_values"] = df_shap_all.apply(
        lambda x: x["values_imbal"] - x["values_bal"], axis=1
    )
    df_shap_all["diff_base_values"] = df_shap_all.apply(
        lambda x: x["base_values_imbal"] - x["base_values_bal"], axis=1
    )
    return df_shap_all


def sum_shap_values_with_threshold_mask(values, mask, threshold=0.01):
    return {
        "positive": np.ma.masked_array(values, mask=~(mask > threshold)).sum(axis=0),
        "neutral": np.ma.masked_array(values, mask=~(np.abs(mask) <= threshold)).sum(
            axis=0
        ),
        "negative": np.ma.masked_array(values, mask=~(mask < -threshold)).sum(axis=0),
    }


def main():
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset_path)
    if args.dataset_split is not None:
        dataset = dataset[args.dataset_split]

    if args.amazon:
        data = pd.DataFrame(
            {
                "text": dataset["review_body"],
                "label": dataset["stars"],
                "language": dataset["language"],
            }
        )
    elif args.xnli:
        data = pd.DataFrame(
            {
                "text": [
                    dataset["premise"][i] + dataset["hypothesis"][i]
                    for i in range(len(dataset["premise"]))
                ],
                "label": dataset["label"],
                "language": dataset["language"],
            }
        )
    if args.subsample is not None:
        data = data.sample(args.subsample, random_state=0)

    shap_values_bal = pickle.load(open(args.shap_values_bal, "rb"))
    shap_values_imbal = pickle.load(open(args.shap_values_imbal, "rb"))

    df_shap_all = shap_bal_and_imbal_to_pandas(
        shap_values_bal, shap_values_imbal, data=data
    )

    print(df_shap_all.columns)
    # Compute cumulative sum of shap values
    df_cumsum_diff_shap = df_shap_all.apply(
        lambda x: pd.Series(
            sum_shap_values_with_threshold_mask(
                x["diff_shap_values"], x["values_bal"], args.threshold
            )
        ),
        axis=1,
    )

    df_shap_all = pd.concat([df_shap_all, df_cumsum_diff_shap], axis=1)
    print(df_shap_all.columns)

    df_shap_contrib_per_token_type_melted = (
        df_shap_all[["positive", "neutral", "negative", "language", "diff_base_values"]]
        .rename(columns={"diff_base_values": "base_values"})
        .reset_index(level=0, inplace=False)
        .melt(
            id_vars=["language"],
            value_vars=["positive", "neutral", "negative", "base_values"],
            var_name="token_type",
            value_name="shap_values",
        )
    )

    if args.amazon:
        df_shap_contrib_per_token_type_melted["label"] = pd.Series(
            [[1, 2, 3, 4, 5] for _ in range(len(df_shap_contrib_per_token_type_melted))]
        )
    elif args.xnli:
        df_shap_contrib_per_token_type_melted["label"] = pd.Series(
            [
                ["entailment", "neutral", "contradiction"]
                for _ in range(len(df_shap_contrib_per_token_type_melted))
            ]
        )
    df_shap_contrib_per_token_type_melted = (
        df_shap_contrib_per_token_type_melted.explode(["shap_values", "label"])
    )
    df_shap_contrib_per_token_type_melted["shap_values"] = (
        df_shap_contrib_per_token_type_melted["shap_values"].apply(
            lambda x: None if isinstance(x, np.ma.core.MaskedConstant) else x
        )
    )
    df_shap_contrib_per_token_type_melted["language_color"] = (
        df_shap_contrib_per_token_type_melted["language"].apply(
            lambda x: LANGUAGE_COLOR[x]
        )
    )

    dict_rename_token_type = {
        "base_values": "base\nvalue",
        "positive": "pos.",
        "neutral": "neut.",
        "negative": "neg.",
    }

    df_shap_contrib_per_token_type_melted["token_type"] = (
        df_shap_contrib_per_token_type_melted["token_type"].apply(
            lambda x: dict_rename_token_type[x]
        )
    )

    if args.amazon:
        fig = sns.catplot(
            df_shap_contrib_per_token_type_melted.query(
                "label == 1 | label == 5"
            ).rename(columns={"shap_values": "average contribution"}),
            y="token_type",
            x="average contribution",
            kind="bar",
            col="label",
            hue="language",
            hue_order=["fr", "es", "ja", "de", "en", "zh"],
            palette=LANGUAGE_COLOR,
            estimator=np.nanmean,
            dodge=True,
        )

    elif args.xnli:
        fig = sns.catplot(
            df_shap_contrib_per_token_type_melted.query(
                "label == 'entailment' | label == 'contradiction'"
            ).rename(columns={"shap_values": "average contribution"}),
            y="token_type",
            x="average contribution",
            kind="bar",
            col="label",
            hue="language",
            hue_order=["fr", "en"],
            palette=LANGUAGE_COLOR,
            estimator=np.nanmean,
            dodge=True,
        )

    fig.set(ylabel=None).set(xlabel="Avg. cumulative difference")
    if args.remove_legend:
        fig._legend.remove()
    plt.xlim((-args.axlim, args.axlim))

    if args.filename is not None:
        plt.savefig(f"{args.output_dir}/{args.filename}.png")
    else:
        plt.savefig(
            f"{args.output_dir}/cumulative_diff_shap_{args.dataset_split}_subsample_{args.subsample}.png"
        )


if __name__ == "__main__":
    main()
