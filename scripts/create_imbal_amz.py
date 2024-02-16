# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""
This script creates an imbalanced version of the Amazon reviews dataset.
The languages are balanced, the overall labels are balanced, but the labels within each language
are imbalanced. The imbalance is created by subsampling. The number of examples in each language 
is specified by the `total_size` argument. The seed is specified by the `seed`. We use a config 
file to specify the imbalance. The config file is a json file with the following format:
{
    "en": "1:2:3",
    "fr": "3:1:2",
    "es": "2:3:1",
    ...
}
Where the numbers specify the ratio of each label.
The labels are "contradiction", "entailment", and "neutral".
"""
import json

import datasets


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="amazon_reviews_multi")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/amz_imbal_v2",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config_amz_imbal.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--lang_col", type=str, default="language")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = json.load(open(args.config_file))
    print(config)
    ds = datasets.load_dataset(args.dataset_name)
    splits = list(ds.keys())
    new_ds = {f"{split}_imbal": [] for split in splits}
    for split in splits:
        new_ds[f"{split}_bal"] = []
    for lang in config.keys():
        ds_monolingual = ds.filter(lambda x: x[args.lang_col] == lang)
        labels = set(ds_monolingual[splits[0]][args.label_col])
        labels = sorted(labels)
        number_of_labels = len(labels)
        dict_sizes = {
            split: len(ds_monolingual[split]) for split in ds_monolingual.keys()
        }
        class_dist = [int(x) for x in config[lang]["dist"].split(":")]
        class_dist = [x / max(class_dist) for x in class_dist]

        class_dist_bal = [sum(class_dist) / len(class_dist)] * len(class_dist)

        class_split_sizes = {
            split: [
                int(x * dict_sizes[split] / number_of_labels) - 1 for x in class_dist
            ]
            for split in ds_monolingual.keys()
        }
        class_split_sizes_bal = {
            split: [
                int(x * dict_sizes[split] / number_of_labels) - 1
                for x in class_dist_bal
            ]
            for split in ds_monolingual.keys()
        }

        if "switch_with" in config[lang]:
            class_dist_switched = [
                int(x) for x in config[config[lang]["switch_with"]]["dist"].split(":")
            ]
            class_dist_switched = [
                x / max(class_dist_switched) for x in class_dist_switched
            ]

        for split in ds_monolingual.keys():
            for i, label in enumerate(labels):
                ds_monolingual_label = ds_monolingual.filter(
                    lambda example: example[args.label_col] == label
                )
                new_ds[f"{split}_imbal"].append(
                    ds_monolingual_label[split].select(
                        range(class_split_sizes[split][i])
                    )
                )
                new_ds[f"{split}_bal"].append(
                    ds_monolingual_label[split].select(
                        range(class_split_sizes_bal[split][i])
                    )
                )

    print(new_ds)
    new_ds = {
        split: datasets.concatenate_datasets(new_ds[split]) for split in new_ds.keys()
    }

    datasets.DatasetDict(
        **new_ds, **{f"{split}_orig": ds[split] for split in ds.keys()}
    ).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
