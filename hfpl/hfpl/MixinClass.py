# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import pytorch_lightning as pl
import torch
import warnings
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import balanced_accuracy_score



class HFModelForPl(pl.LightningModule):

    """
    General case class for huggingface models with pytorch lightning for sequence classification
    There are two classifiers on top of the language model : one out of the box from huggingface which is used to make predictions, and one which is used to language identification.
    The gradients of the language identifier head do not propagate to the language model.
    If specified however, the language model can be updated by adding an entropy loss to the total loss such that the language model becomes more language agnostic.
    """

    def __init__(
        self,
        model_name,
        model_kwargs={},
        tokenizer=None,
        tokenizer_kwargs={},
        pretrain_lid_head=0,
        train_lm=True,
        lm_lr=5e-5,
        lid_lr=1e-4,
        cls_lr=1e-4,
        total_steps=None,
        language_cls=False,
        class_weight=None,
        entropy_max_coef=0.0,
        gradient_reversal_coef=0.0,
        gradient_accumulation_steps=1,
        test_dataset_names=None,
        mask_entropy_max=False,
        mask_entropy_max_coef = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        if test_dataset_names is not None:
            assert isinstance(test_dataset_names, list)
            assert len(test_dataset_names) == len(set(test_dataset_names))

        if gradient_reversal_coef > 0:
            if not language_cls:
                warnings.warn(
                    "Gradient reversal is enabled but language classification is disabled, so there is no forward pass through the LID head. This is probably not what you want."
                )
        if entropy_max_coef > 0:
            if not language_cls:
                warnings.warn(
                    "Entropy maximization is enabled but language classification is disabled, so there is no forward pass through the LID head. This is probably not what you want."
                )

        if class_weight is not None:
            # class_weight should be a dict with keys being the language ids and values being a list of weights for each class
            assert isinstance(class_weight, dict)
            assert all(isinstance(k, (int, np.int_)) for k in class_weight.keys()) , f"Keys of class_weight should be integers, they are {type(list(class_weight.keys())[0])}"
            assert all(isinstance(v, (list, np.ndarray)) for v in class_weight.values()) , f"Values of class_weight should be lists, they are {type(list(class_weight.values())[0])}"
            self.class_weight = class_weight
        else:
            self.class_weight = None
        self.model = model_name
        self.tokenizer = tokenizer
        self.automatic_optimization = False
        self.lm = AutoModelForSequenceClassification.from_pretrained(
            model_name, **model_kwargs
        )

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )

        if "num_lang_labels" in self.lm.config.to_dict():
            self.lid_head = torch.nn.Linear(
                self.lm.config.hidden_size, self.lm.config.num_lang_labels
            )
        else:
            # This is just a placeholder
            self.lid_head = torch.nn.Linear(
                self.lm.config.hidden_size, 1
            )

    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True
        labels = kwargs.pop("labels", None)
        lang_label = kwargs.pop("lang_label", None)

        self.unfreeze_lm()
        self.unfreeze_cls()
        self.unfreeze_lid()

        if self.hparams.mask_entropy_max:
            # We add a row to the input with only mask tokens

            kwargs["input_ids"] = torch.cat(
                [kwargs["input_ids"], (kwargs["input_ids"][-1] != self.tokenizer.pad_token_id) * self.tokenizer.mask_token_id], dim=0
            )
            kwargs["attention_mask"] = torch.cat(
                [kwargs["attention_mask"], kwargs["attention_mask"][-1]], dim=0
            )

            if "token_type_ids" in kwargs:
                kwargs["token_type_ids"] = torch.cat(
                    [kwargs["token_type_ids"], kwargs["token_type_ids"][-1]], dim=0
                )

        output = self.lm(**kwargs)

        if self.hparams.mask_entropy_max:
            # We pop the last row of the output
            output_mask = output["logits"][-1, :]
            output["logits"] = output["logits"][:-1, :]
            output["hidden_states"] = [hs[:-1, :, :] for hs in output["hidden_states"]]

            distribution = output_mask.softmax(dim=-1)
            output["mask_entropy_loss"] = (distribution * distribution.log()).mean().mul(self.hparams.mask_entropy_max_coef)

        if labels is not None:
            if self.hparams.class_weight is not None:
                assert lang_label is not None
                losses = []
                for lang_id in lang_label.unique():
                    idx_lang = lang_label.squeeze() == lang_id
                    if not idx_lang.any():
                        continue
                    losses.append(
                        torch.nn.functional.cross_entropy(
                            output.logits[idx_lang, :],
                            labels[idx_lang].squeeze(-1),
                            weight=torch.tensor(self.class_weight[lang_id.item()], device=self.device, requires_grad=False).float(),
                            reduction="none",
                        )
                    )
                loss = torch.cat(losses).mean()
            else:
                loss = torch.nn.functional.cross_entropy(
                    output.logits, labels.squeeze(), weight=None, reduction="mean"
                )
            output["loss"] = loss
        else:
            output["loss"] = torch.tensor(float("nan"))
        lang_pred = self.lid_head(output.hidden_states[-1][:, 0, :])

        if (lang_label is not None) and (not any(lang_label >= self.lm.config.num_lang_labels)):
            # This basically covers the case when we have a batch with an unknown language
            # We still predict the language but we don't compute the loss
            # Otherwise it results in a cuda-device side assertion error
            output["lang_loss"] = torch.nn.functional.cross_entropy(
                lang_pred, lang_label.squeeze(), reduction="mean"
            )
        else:
            output["lang_loss"] = torch.tensor(float("nan"))
        output["lang_label"] = lang_label
        output["lang_logits"] = lang_pred
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        dict_log = {
            "train_loss": loss,
            "train_acc": (outputs.logits.argmax(dim=-1) == batch["labels"].squeeze())
            .float()
            .mean(),
            "train_bal_acc": balanced_accuracy_score(
                batch["labels"].squeeze().cpu().numpy(),
                outputs.logits.argmax(dim=-1).cpu().numpy(),
            ),
        }
        if self.hparams.mask_entropy_max:
            dict_log["train_mask_entropy_loss"] = outputs.mask_entropy_loss
            loss += outputs.mask_entropy_loss

        optim_lm, optim_lid = self.optimizers()
        scheduler_lm, scheduler_lid = self.lr_schedulers()

        if self.hparams.language_cls:
            if self.hparams.entropy_max_coef > 0:
                entropy_loss = (
                    outputs.lang_logits.softmax(dim=-1)
                    .log()
                    .mean()
                    .mul(self.hparams.entropy_max_coef)
                )
                dict_log["train_entropy_loss"] = entropy_loss
                loss += entropy_loss
            lang_loss = torch.nn.functional.cross_entropy(
                outputs.lang_logits, batch["lang_label"].squeeze(), reduction="mean"
            )
            if self.hparams.gradient_reversal_coef > 0:
                loss -= self.hparams.gradient_reversal_coef * lang_loss
            self.manual_backward(
                lang_loss / self.hparams.gradient_accumulation_steps, retain_graph=True
            )
            if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
                self.freeze_lm()
                self.freeze_cls()
                self.unfreeze_lid()
                optim_lid.step()
                optim_lid.zero_grad()
                scheduler_lid.step()
                self.unfreeze()

            dict_log["train_lang_loss"] = lang_loss
            dict_log["train_lang_acc"] = (
                (outputs.lang_logits.argmax(dim=-1) == batch["lang_label"].squeeze())
                .float()
                .mean()
            )

        if self.hparams.train_lm:
            self.unfreeze_lm()
            self.unfreeze_cls()
            self.freeze_lid()
            self.manual_backward(loss / self.hparams.gradient_accumulation_steps)
            self.unfreeze_lid()

            if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0 and not (
                batch_idx < self.hparams.pretrain_lid_head
            ):
                self.freeze_lid()
                optim_lm.step()
                optim_lm.zero_grad()
                scheduler_lm.step()
                self.unfreeze_lid()

        self.log_dict(dict_log, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        dict_log = {
            "val_loss": loss,
            "val_acc": (outputs.logits.argmax(dim=-1) == batch["labels"].squeeze())
            .float()
            .mean(),
            "val_bal_acc": balanced_accuracy_score(
                batch["labels"].squeeze().cpu().numpy(),
                outputs.logits.argmax(dim=-1).cpu().numpy(),
            ),
        }
        if self.hparams.language_cls:
            dict_log["val_lang_acc"] = (
                outputs.lang_logits.argmax(dim=-1) == batch["lang_label"].squeeze()
            ).float().mean()
            dict_log["val_lang_loss"] = torch.nn.functional.cross_entropy(
                outputs.lang_logits, batch["lang_label"].squeeze(), reduction="mean"
            )
            dict_log["val_lang_bal_acc"] = balanced_accuracy_score(
                batch["lang_label"].squeeze().cpu().numpy(),
                outputs.lang_logits.argmax(dim=-1).cpu().numpy(),
            ),
        self.log_dict(dict_log, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self(**batch)
        dict_log = {
            "test_loss": outputs.loss,
            "test_acc": (outputs.logits.argmax(dim=-1) == batch["labels"].squeeze())
            .float()
            .mean(),
            "test_bal_acc": balanced_accuracy_score(
                batch["labels"].squeeze().cpu().numpy(),
                outputs.logits.argmax(dim=-1).cpu().numpy(),
            ),
        }
        if self.hparams.language_cls:
            dict_log["test_lang_acc"] = (
                outputs.lang_logits.argmax(dim=-1) == batch["lang_label"].squeeze()
            ).float().mean(),
        add_dataloader_idx = True
        if self.hparams.test_dataset_names is not None and dataloader_idx is not None:
            add_dataloader_idx = False
            dict_new_name = {}
            for k, v in dict_log.items():
                dict_new_name[
                    f"{k}/{self.hparams.test_dataset_names[dataloader_idx]}"
                ] = v
            dict_log = dict_new_name

        self.log_dict(dict_log, on_epoch=True, add_dataloader_idx=add_dataloader_idx)

    def freeze_lm(self):
        for params in self.lm.base_model.parameters():
            params.requires_grad = False

    def unfreeze_lm(self):
        for params in self.lm.base_model.parameters():
            params.requires_grad = True

    def freeze_lid(self):
        for params in self.lid_head.parameters():
            params.requires_grad = False

    def unfreeze_lid(self):
        for params in self.lid_head.parameters():
            params.requires_grad = True

    def freeze_cls(self):
        for params in self.lm.classifier.parameters():
            params.requires_grad = False

    def unfreeze_cls(self):
        for params in self.lm.classifier.parameters():
            params.requires_grad = True

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            torch.optim.RAdam(
                [
                    {"params": self.lm.base_model.parameters(), "lr": self.hparams.lm_lr},
                    {
                        "params": self.lm.classifier.parameters(),
                        "lr": self.hparams.cls_lr,
                    },
                ],
                eps=1e-8,
            )
        )
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizers[0],
                    start_factor=1,
                    total_iters=self.hparams.total_steps,
                    end_factor=0.01,
                ),
                "interval": "step",
            }
        ]

        optimizers.append(
            torch.optim.RAdam(
                self.lid_head.parameters(), lr=self.hparams.cls_lr, eps=1e-8
            )
        )
        schedulers.append(
            {
                "scheduler": torch.optim.lr_scheduler.LinearLR(
                    optimizers[1],
                    start_factor=1,
                    total_iters=self.hparams.total_steps,
                    end_factor=0.01,
                ),
                "interval": "step",
            }
        )
        return optimizers, schedulers
