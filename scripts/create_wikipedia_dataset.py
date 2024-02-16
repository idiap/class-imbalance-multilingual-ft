# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
"""
This script creates a dataset from the Wikipedia dataset for each language specified in the languages list.
We select a fixed number of samples for each language and concatenate them into a single dataset. We also add a language label to each sample.
The resulting dataset is saved in the output_dir in huggingface format.
By default, the script saves 1000 samples for each language. We use this dataset to measure language identification in the latent space of our models.
"""
import datasets
import argparse

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
