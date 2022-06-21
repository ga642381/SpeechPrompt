import argparse
import os
import re
import subprocess
from itertools import groupby
from pathlib import Path

import torchaudio
from tqdm import tqdm

from speech_commands.dataset import (SpeechCommandsDataset,
                                     SpeechCommandsTestingDataset,
                                     get_dataloader, split_dataset)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.taskrc = self.config["taskrc"]
        self.datarc = self.config["datarc"]
        self.fairseqrc = self.config["fairseqrc"]

    def generate_manifest(self):
        os.makedirs(
            os.path.join(self.datarc["root_path"], "temp", "_background_noise_"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.datarc["output_path"], "manifest"),
            exist_ok=True,
        )

        split_data_list = {}
        split_data_list["train"], split_data_list["valid"] = split_dataset(
            Path(self.datarc["root_path"], "speech_commands_v0.01")
        )

        for split in ["train", "valid", "test"]:
            if split == "test":
                dataset = SpeechCommandsTestingDataset(root_path=self.datarc["root_path"])
                dataloader = get_dataloader(
                    dataset=dataset,
                    batch_size=self.datarc["batch_size"],
                    num_workers=self.datarc["num_workers"],
                    balanced=False,
                )

            else:
                dataset = SpeechCommandsDataset(data_list=split_data_list[split], root_path=self.datarc["root_path"])
                dataloader = get_dataloader(
                    dataset=dataset,
                    batch_size=self.datarc["batch_size"],
                    num_workers=self.datarc["num_workers"],
                    balanced=True,
                )

            with open(Path(self.datarc["output_path"], "manifest", f"{split}.manifest"), "w") as f:
                root_path = self.datarc["root_path"]
                f.write(f"{root_path}\n")
                for wavs, labels, audio_pathes in tqdm(dataloader, desc=split):
                    for wav, label, audio_path in zip(wavs, labels, audio_pathes):
                        if not audio_path.exists():
                            torchaudio.save(audio_path, wav.unsqueeze(0), 16000)

                        relative_path = audio_path.relative_to(self.datarc["root_path"])
                        f.write(f"{relative_path}\t{str(len(wav))}\n")

    def class2index(self, file_name):
        CLASSES = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "_unknown_",
            "_silence_",
        ]

        mapping = {CLASSES[i]: i for i in range(len(CLASSES))}

        class_name = file_name.split("/")[1]

        if class_name == "_background_noise_":
            class_name = "_silence_"
        elif class_name not in CLASSES:
            class_name = "_unknown_"

        return mapping[class_name]

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

                        if self.taskrc["merge"]:
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
