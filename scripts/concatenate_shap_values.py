# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import os
import shap
import pickle
import re
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--folder", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix
    list_of_files = os.listdir(args.folder)
    print(list_of_files)
    list_of_files = [
        file for file in list_of_files if re.match(prefix + "[0-9_]*.pkl", file)
    ]
    # list_of_files = [file for file in list_of_files if file.startswith(prefix)]
    # list_of_files = [file for file in list_of_files if file.endswith(".pck")]

    indices = [file.split("_") for file in list_of_files]
    # print(indices)
    assert all([not re.match("[0-9]", idx[0]) for idx in indices])
    assert all([not re.match("[0-9]", idx[1]) for idx in indices])
    assert all([re.match("[0-9]", idx[2]) for idx in indices])
    total_index = indices[0][3]
    assert all([idx[3] == total_index for idx in indices])

    list_of_files = [os.path.join(args.folder, file) for file in list_of_files]

    print(len(list_of_files))
    assert len(list_of_files) == int(total_index.split(".")[0])
    dict_of_files = {
        int(indices[i][2]): list_of_files[i] for i in range(len(list_of_files))
    }

    # print(f"Concatenating {len(list_of_files)} files:", "\n", "\n".join(list_of_files))
    shap_values = []
    for i in range(1, len(list_of_files) + 1):
        file = dict_of_files[i]
        print(f"Loading file {file}")
        shap_values.append(pickle.load(open(file, "rb")))
    pickle.dump(
        shap.Explanation(
            values=np.concatenate(
                [shap_values[i].values for i in range(len(shap_values))], axis=0
            ),
            base_values=np.concatenate(
                [shap_values[i].base_values for i in range(len(shap_values))], axis=0
            ),
            feature_names=[
                shap_values[i].feature_names[j]
                for i in range(len(shap_values))
                for j in range(len(shap_values[i].feature_names))
            ],
            data=tuple(
                [
                    shap_values[i].data[j]
                    for i in range(len(shap_values))
                    for j in range(len(shap_values[i].data))
                ]
            ),
        ),
        open(os.path.join(args.folder, prefix + "_cat.pck"), "wb"),
    )


if __name__ == "__main__":
    main()
