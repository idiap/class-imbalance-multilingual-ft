<!--
SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>

SPDX-License-Identifier: GPL-3.0-only
-->

# Language-specific class imbalance

This repository contains the code for our paper *"Understanding the effects of language-specific class imbalance in multilingual fine-tuning"* which was accepted to EACL Findings 2024. You can find the paper on [publidiap](https://publidiap.idiap.ch/attachments/papers/2024/Jung_EACLFINDINGS2024_2024.pdf).

## Requirements

Please install the conda environment and add the custom module using the following commands:

```
conda env create -f environment.yml
conda activate pl-hf
conda develop hfpl
```

The HFPL module contains a class called HFModelForPl, which is meant to facilitate the use of huggingface models with Pytorch Lightning and to add the language identifier head and calculate the various losses we experiment with in the paper, along with handling the per-language class weighing. 

## Data

The datasets are created from publicly available datasets on the huggingface hub. There is no need to download the datasets manually.

To create the XNLI dataset, please run the following command:

```
python scripts/create_imbal_xnli.py --output_dir ./datasets/xnli_fr_en --config_file scripts/config_xnli_imbal.json
```

To create the amazon reviews dataset, please run the following command:

```
python scripts/create_imbal_amz.py --output_dir ./datasets/amz_reviews --config_file scripts/config_amz_imbal.json
```

To create the wikipedia dataset, please run the following command:

```
python scripts/create_wikipedia_dataset.py --output_dir ./datasets/wikipedia
```
(This dataset is just a subsample of the "graelo/wikipedia" dataset on the huggingface hub.)

## Training

We provide one training script and multiple config files for the different experiments in the paper. Tracking is done with Weights and Biases. The "config" argument takes in multiple names of configs found in the configs folder so that you can choose which model to train, which dataset to train it on, and whether to train it with out modified class weighing approach and/or the mask entropy maximization loss introduced in Annex A.3 of our paper. You can use the following template:

```
python scripts/fine_tune_w_pl.py \
    --config <mbert|xlmr> \
    <xnli_bal|xnli_imbal|amz_reviews_bal|amz_reviews_imbal> \
    <deterministic> \
    [class_w] \
    [mask_entropy_max]
```

The script will save the best model according to validation loss. 


## Evaluating language specificity of latent space

Once a model has been trained, you can evaluate the language specificity of the latent space using the following command:

For the amazon reviews dataset:
```
python scripts/evaluate_model_lang_specificity.py \
    --pl_model_path <PATH_TO_MODEL> \
    --dataset_path ./datasets/amz_reviews \
    --dataset_split "test_orig" \
    --text_col "review_body"
```

For the XNLI dataset:
```
python scripts/evaluate_model_lang_specificity.py \
    --pl_model_path <PATH_TO_MODEL> \
    --dataset_path "./datasets/xnli_fr_en" \
    --dataset_split "test_bal" \
    --tokenize_pairs premise hypothesis
```


For the wikipedia dataset:
```
python scripts/evaluate_model_lang_specificity.py \
    --pl_model_path <PATH_TO_MODEL> \
    --dataset_path "./datasets/wikipedia_lid_huge" \
    --languages fr en de es ja zh
```

The results will be saved in the same directory as the model checkpoint in a file called results_{dataset_name}.txt.

## Shap values

Computing the shap values on the test sets of the datasets is unfortunately very inefficient. We provide the code to compute the shap values on the test sets of the datasets, but we do not recommend running it, or at least not on the full test sets.

You can compute the SHAP values on XNLI using the following command:

```
python3 scripts/shap_values.py \
    --pl_model_path <PATH_TO_MODEL> \
    --dataset_path "./datasets/xnli_fr_en" \
    --label_col "label" \
    --text_col "premise" \
    --text_pair "hypothesis" \
    --split "test_bal"
```

And on the Amazon Reviews dataset using the following command:

```
python3 scripts/shap_values.py \
    --pl_model_path <PATH_TO_MODEL>\
    --dataset_path ./datasets/amz_reviews \
    --label_col "stars" \
    --text_col "review_body" \
    --subsample 4000 \
    --split "test_orig" 
```

Once the shap values have been calculated, they will be saved in the same directory as the model checkpoint.

## Cumulative difference in SHAP values

One script takes in the SHAP values from the models trained on the balanced and imbalanced datasets and computes the cumulative difference in SHAP values, and plots the average over the test sets, by language. You can use the following command:

For the Amazon Reviews dataset:
```
python .scripts/plot_cumulative_diff_shap.py \
    --shap_values_bal <PATH_TO_SHAP_BAL> \
    --shap_values_imbal <PATH_TO_SHAP_IMBAL> \
    --dataset_path "./datasets/amz_reviews" \
    --dataset_split "test_orig" \
    --subsample 4000 \
    --threshold 0.01 \
    --filename <FILENAME> \
    --amazon \
    --axlim 0.199 \
    --remove_legend \
    --output_dir "./plots"
```

For the XNLI dataset:
```
python scripts/plot_cumulative_diff_shap.py \
    --shap_values_bal 
    --shap_values_imbal 
    --dataset_path "./datasets/xnli_fr_en" \
    --dataset_split "test_bal" \
    --threshold 0.01 \
    --filename <FILENAME> \
    --xnli \
    --axlim 0.199 \
    --remove_legend \
    --output_dir "./plots"
```

## Citation

Please use the following BibTeX entry to cite our work:
```
@inproceedings{
anonymous2024understanding,
    title={Understanding the effects of language-specific class-imbalance in multilingual fine-tuning},
    author={Anonymous},
    booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
    year={2024},
    url={https://openreview.net/forum?id=FiUMOgLplM}
}
```