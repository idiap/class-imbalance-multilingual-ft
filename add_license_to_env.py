# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Vincent Jung <vincent.jung@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse

import requests
from importlib import metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="environment.yml",
        help="Path to environment.yml file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    package_license_map = {}
    for name, name_bis in metadata.packages_distributions().items():
        for subname in name_bis:
            package_license_map[subname] = "No license found, " + str(
                metadata.metadata(subname)["Home-page"]
            )
            if metadata.metadata(subname)["License"] is not None:
                package_license_map[subname] = (
                    str(metadata.metadata(subname)["License"])
                    + ", "
                    + str(metadata.metadata(subname)["Home-page"])
                )
            elif (metadata.metadata(subname)["Home-page"] is not None) and (
                "github.com" in metadata.metadata(subname)["Home-page"]
            ):
                api_url = (
                    "https://api.github.com/repos/"
                    + metadata.metadata(subname)["Home-page"].split("github.com/")[1]
                )
                if api_url.endswith("/"):
                    api_url = api_url[:-1]
                api_return = requests.get(
                    api_url,
                    headers={
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                        "Authorization": "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"
                    },
                )
                print(api_url)
                print("Status code: " + str(api_return.status_code))
                if api_return.status_code == 200:
                    print(api_return.json()["license"])
                    if "license" in api_return.json():
                        package_license_map[subname] = (
                            api_return.json()["license"]["spdx_id"]
                            + ", "
                            + str(metadata.metadata(subname)["Home-page"])
                        )
            else:
                pass

    with open(args.file, "r") as f:
        lines = f.readlines()

    started_listing_dependencies = False
    with open(
        args.file[: args.file.rindex(".")]
        + "_wlicense"
        + args.file[args.file.rindex(".") :],
        "w",
    ) as f:
        for i, line in enumerate(lines):
            if line.startswith("dependencies:"):
                started_listing_dependencies = True
            if started_listing_dependencies:
                if line.strip().startswith("- "):
                    package_name = line[4 : min(line.find("="), len(line) - 1)]
                    if package_name in package_license_map:
                        if not ("\n" in str(package_license_map[package_name])):
                            line = (
                                line[:-1]
                                + " #"
                                + str(package_license_map[package_name])
                                + "\n"
                            )
                        else:
                            line = (
                                line
                                + "\n".join(
                                    [
                                        "# " + i
                                        for i in package_license_map[
                                            package_name
                                        ].split("\n")
                                    ]
                                )
                                + "\n"
                            )

            f.write(line)


if __name__ == "__main__":
    main()
