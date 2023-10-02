# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
This script creates an imbalanced version of the XNLI dataset. The languages are balanced, the overall labels are balanced,
but the labels within each language are imbalanced. The imbalance is created by subsampling. The number of examples
in each language is specified by the `total_size` argument. The seed is specified by the `seed`.
We use a config file to specify the imbalance. The config file is a json file with the following format:
{
    "en": "1:2:3",
    "fr": "3:1:2",
    "es": "2:3:1",
    ...
}
Where the numbers specify the ratio of each label. The labels are "contradiction", "entailment", and "neutral".
"""
import datasets
import pandas as pd
import json


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="data/xnli")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="The output directory"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config_xnli_imbal.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = json.load(open(args.config_file))
    output_dict = {}
    for lang in config.keys():
        output_dict[lang] = {}
        output_dict[lang]["bal"] = {"train": [], "validation": [], "test": []}
        output_dict[lang]["imbal"] = {"train": [], "validation": [], "test": []}
        output_dict[lang]["switched"] = {"train": [], "validation": [], "test": []}

        xnli_dataset_onelang = datasets.load_dataset("xnli", language=lang)
        for split in ["train", "validation", "test"]:
            xnli_dataset_onelang[split] = xnli_dataset_onelang[split].add_column(
                "language", [lang] * len(xnli_dataset_onelang[split])
            )

        size_dict = {}
        size_dict["train"] = len(xnli_dataset_onelang["train"])
        size_dict["validation"] = len(xnli_dataset_onelang["validation"])
        size_dict["test"] = len(xnli_dataset_onelang["test"])

        class_dist = {}

        class_dist["imbal"] = [int(x) for x in config[lang]["dist"].split(":")]
        class_dist["imbal"] = [
            x / max(class_dist["imbal"]) for x in class_dist["imbal"]
        ]

        class_dist["bal"] = [sum(class_dist["imbal"]) / len(class_dist["imbal"])] * len(
            class_dist["imbal"]
        )

        class_dist["switched"] = [
            int(x) for x in config[config[lang]["switch_with"]]["dist"].split(":")
        ]
        class_dist["switched"] = [
            x / max(class_dist["switched"]) for x in class_dist["switched"]
        ]

        for i, label in enumerate([0, 1, 2]):
            xnli_dataset_onelang_label = xnli_dataset_onelang.filter(
                lambda example: example["label"] == label
            )
            for k in ["bal", "imbal", "switched"]:
                for split in ["train", "validation", "test"]:
                    size = int((size_dict[split] * class_dist[k][i]) / 3) - 1
                    xnli_dataset_onelang_label_split = xnli_dataset_onelang_label[
                        split
                    ].select(range(size))
                    output_dict[lang][k][split].append(xnli_dataset_onelang_label_split)

        for k in ["bal", "imbal", "switched"]:
            for split in ["train", "validation", "test"]:
                output_dict[lang][k][split] = datasets.concatenate_datasets(
                    output_dict[lang][k][split]
                )
            
    rolled_output_dict = {}
    concatenated_datasets_by_language = {}
    for k in ["bal", "imbal", "switched"]:
        for split in ["train", "validation", "test"]:
            if f"{split}_{k}" not in concatenated_datasets_by_language.keys():
                concatenated_datasets_by_language[f"{split}_{k}"] = []
            for lang in output_dict.keys():
                rolled_output_dict[f"{split}_{k}_{lang}"] = output_dict[lang][k][split]
                concatenated_datasets_by_language[f"{split}_{k}"].append(
                    rolled_output_dict[f"{split}_{k}_{lang}"]
                )
            concatenated_datasets_by_language[
                f"{split}_{k}"
            ] = datasets.concatenate_datasets(
                concatenated_datasets_by_language[f"{split}_{k}"]
            )

    datasets.DatasetDict(
        **concatenated_datasets_by_language, **rolled_output_dict
    ).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
