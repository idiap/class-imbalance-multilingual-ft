<!--
SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>

SPDX-License-Identifier: GPL-3.0-only
-->

# language-label-bias

This repository contains the code for our paper "Understanding the effects of language-specific class-imbalance in multilingual fine-tuning".

## Requirements

Please install the conda environment and add the custom module using the following commands:

```
conda env create -f environment.yml
conda activate lang_imbal
conda develop hfpl
```

## Data

The datasets are created from publicly available datasets on the huggingface hub. There is no need to download the datasets manually.

To create the XNLI dataset, please run the following command:

```
conda activate lang_imbal
python scripts/create_imbal_xnli.py --output_dir ./datasets/xnli --config_file scripts/config_xnli_imbal.json
```

To create the amazon reviews dataset, please run the following command:

```
conda activate lang_imbal
python scripts/imbalance_dataset.py --dataset_name amazon_reviews_multi --output_dir ./datasets/amz_reviews --config_file scripts/config_amz_imbal.json
```

To create the wikipedia dataset, please run the following command:

```
conda activate lang_imbal
python scripts/create_wikipedia_dataset.py --output_dir ./datasets/wikipedia
```
(This dataset is just a subsample of the "graelo/wikipedia" dataset on the huggingface hub.)

## Training

We provide one training script and multiple config files for the different experiments in the paper. For now, logging is done on Weights and Biases. You can use the following command to train a model:

```
conda activate lang_imbal
python scripts/fine_tune_w_pl.py --config configs/xnli_imbal.json
```

You can replace the config file with any of the following : amz_reviews_bal.json, amz_reviews_imbal.json, amz_reviews_imbal_cw.json, xnli_bal.json, xnli_imbal.json, xnli_imbal_cw.json.

## Evaluating language specificity of latent space

Once a model has been trained, you can evaluate the language specificity of the latent space using the following command:

```
conda activate lang_imbal
python scripts/evaluate_model_lang_specificity.py \
    --pl_model_path PATH_TO_MODEL \
    --wandb_run_name NAME_OF_WANDB_RUN 
```

## Shap values

Computing the shap values on the test sets of the datasets is unfortunately very inefficient. We provide the code to compute the shap values on the test sets of the datasets, but we do not recommend running it, or at least not on the full test sets. We will provide the shap values for the test sets of the datasets in the near future.

You can compute the SHAP values on XNLI using the following command:

```
conda activate lang_imbal
python3 scripts/shap_values_xnli.py \
    --pl_model_path PATH_TO_MODEL \
    --dataset_path "./datasets/xnli" \
    --label_col "label" \
    --text_col "premise" \
    --text_pair "hypothesis" \
    --split "test_bal"
```

And on the amazon reviews dataset using the following command:

```
conda activate lang_imbal
python3 scripts/shap_values.py \
    --pl_model_path PATH_TO_MODEL\
    --dataset_path ./datasets/amz_reviews \
    --label_col "stars" \
    --text_col "review_body" \
    --subsample 4000 \
    --split "test" 
```

Once the shap values have been calculated, they will be saved in the same directory as the model checkpoint.
In the notebooks folder, there are two notebooks used to generate the plots in the paper. The first one is used to generate the plots for the XNLI dataset, and the second one is used to generate the plots for the amazon reviews dataset. You can add the paths to the pickled shap values in the notebook and run it to generate the plots.