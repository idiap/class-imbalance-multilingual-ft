# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import datasets
import argparse
import os

languages = [
    "af",
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "kk",
    "ko",
    "mr",
    "nl",
    "pt",
    "ru",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "ur",
    "vi",
    "yo",
    "zh",
]

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, default="./datasets")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num_samples", type=int, default=1000)
    args = argparser.parse_args()
    return args

def main():

    args = parse_args()
    monolingual_datasets = {}
    for language in languages:
        monolingual_datasets[language] = datasets.load_dataset(
            "graelo/wikipedia", "20230601." + language, split="train"
        )

    # Select random samples from each dataset and concatenate them, adding a language label

    output_datasets = []

    for language, dataset in monolingual_datasets.items():
        dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(args.num_samples))
        dataset = dataset.add_column("language", [language] * len(dataset))
        output_datasets.append(dataset)

    wikipedia = datasets.interleave_datasets(
        output_datasets,
    )

    # save the dataset in huggingface format
    wikipedia.save_to_disk(args.output_dir)

if __name__ == "__main__":
    main()
