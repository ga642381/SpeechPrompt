import argparse
import os
import re
import subprocess
from collections import Counter
from itertools import groupby
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm

from fluent_commands.dataset import FluentCommandsDataset, get_dataloader


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.taskrc = self.config["taskrc"]
        self.datarc = self.config["datarc"]
        self.fairseqrc = self.config["fairseqrc"]
        self.get_dataset()

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        for split in ["train", "valid", "test"]:
            dataset = FluentCommandsDataset(self.df[split], self.datarc["root_path"], self.Sy_intent)
            dataloader = get_dataloader(
                dataset=dataset,
                batch_size=self.datarc["batch_size"],
                num_workers=self.datarc["num_workers"],
            )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def get_dataset(self):
        self.df = {}
        train_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "train_data.csv"))
        valid_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "valid_data.csv"))
        test_df = pd.read_csv(os.path.join(self.datarc["root_path"], "data", "test_data.csv"))

        Sy_intent = {"action": {}, "object": {}, "location": {}}

        values_per_slot = []
        count = 0
        for slot in ["action", "object", "location"]:
            slot_values = Counter(train_df[slot])
            for index, value in enumerate(slot_values):
                if self.taskrc["no_overlap"]:
                    index = index + count
                Sy_intent[slot][value] = index
                Sy_intent[slot][index] = value
            count += len(slot_values)
            values_per_slot.append(len(slot_values))
        self.values_per_slot = values_per_slot
        self.Sy_intent = Sy_intent
        self.df["train"] = train_df
        self.df["valid"] = valid_df
        self.df["test"] = test_df

    def class2index(self, file_name):
        for key, df in self.df.items():
            if file_name in list(df["path"].values):
                act_idx = self.Sy_intent["action"][df[df["path"] == file_name]["action"].values[0]]
                obj_idx = self.Sy_intent["object"][df[df["path"] == file_name]["object"].values[0]]
                loc_idx = self.Sy_intent["location"][df[df["path"] == file_name]["location"].values[0]]

                return f"{act_idx} {obj_idx} {loc_idx}"

    def postprocess(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "preprocessed"),
            exist_ok=True,
        )

        for split in ["train", "valid", "test"]:
            quantized_file_path = Path(self.datarc["output_path"], "quantized", f"{split}")
            output_path = Path(self.datarc["output_path"], "preprocessed", f"{split}")

            with open(quantized_file_path, "r") as f:
                with open(output_path, "w") as f_output:
                    for line in tqdm(f.readlines(), desc=split):
                        file_name, tokens = line.rstrip("\n").split("|")

                        if self.config["merge"]:
                            token_list = tokens.split()
                            merged_tokens_list = [x[0] for x in groupby(token_list)]
                            tokens = " ".join(merged_tokens_list)

                        preprocessed_line = f"{file_name}|{tokens}|{self.class2index(file_name)}\n"

                        f_output.write(preprocessed_line)

    def quantized(self):
        os.makedirs(
            os.path.join(self.datarc["output_path"], "quantized"),
            exist_ok=True,
        )

        python_file = Path("../speech2unit/clustering/quantize_with_kmeans.py")

        for split in ["train", "valid", "test"]:
            manifest_path = Path(self.datarc["output_path"], "manifest", f"{split}.manifest")
            output_path = Path(self.datarc["output_path"], "quantized", f"{split}")

            subprocess.call(
                [
                    "python",
                    python_file,
                    "--feature_type",
                    self.fairseqrc["feature_type"],
                    "--kmeans_model_path",
                    self.fairseqrc["km_model_path"],
                    "--acoustic_model_path",
                    self.fairseqrc["ssl_model_path"],
                    "--layer",
                    str(self.fairseqrc["layer"]),
                    "--manifest_path",
                    manifest_path,
                    "--out_quantized_file_path",
                    output_path,
                    "--extension",
                    ".wav",
                    "--full_file_name",
                ]
            )
