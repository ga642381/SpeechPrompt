"""
from s3prl toolkit
"""

import re
import os
from random import randint
from pathlib import Path
import hashlib
from typing import List, Tuple, Union

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio.sox_effects import apply_effects_file

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

EFFECTS = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]


class SpeechCommandsBaseDataset(Dataset):
    """12-class Speech Commands base dataset."""

    def __init__(self):
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}
        self.class_num = 12
        self.data = []

    def __getitem__(self, idx):
        class_name, audio_path = self.data[idx]
        wav, _ = apply_effects_file(str(audio_path), EFFECTS)
        wav = wav.squeeze(0)

        return wav, self.class2index[class_name], audio_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        """Collate a mini-batch of data."""
        return zip(*samples)


class SpeechCommandsDataset(SpeechCommandsBaseDataset):
    """Training and validation dataset."""

    def __init__(self, data_list, root_path, **kwargs):
        super().__init__()

        data = [
            (class_name, audio_path)
            if class_name in self.class2index.keys()
            else ("_unknown_", audio_path)
            for class_name, audio_path in data_list
        ]
        data += [
            ("_silence_", audio_path)
            for audio_path in Path(
                root_path, "speech_commands_v0.01", "_background_noise_"
            ).glob("*.wav")
        ]

        class_counts = {class_name: 0 for class_name in CLASSES}
        for class_name, _ in data:
            class_counts[class_name] += 1

        sample_weights = [
            len(data) / class_counts[class_name] for class_name, _ in data
        ]

        self.data = data
        self.sample_weights = sample_weights
        self.root_path = root_path

    def __getitem__(self, idx):
        wav, label, audio_path = super().__getitem__(idx)

        # _silence_ audios are longer than 1 sec.
        if label == self.class2index["_silence_"]:
            random_offset = randint(0, len(wav) - 16000)
            wav = wav[random_offset : random_offset + 16000]
            audio_path = Path(
                self.root_path,
                "temp",
                "_background_noise_",
                f"{audio_path.stem}-{random_offset}{audio_path.suffix}",
            )

        return wav, label, audio_path


class SpeechCommandsTestingDataset(SpeechCommandsBaseDataset):
    """Testing dataset."""

    def __init__(self, root_path, **kwargs):
        super().__init__()

        self.data = [
            (class_dir.name, audio_path)
            for class_dir in Path(root_path, "speech_commands_test_set_v0.01").iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]


def get_dataloader(dataset, batch_size, num_workers, drop_last=False, balanced=False):
    sampler = (
        WeightedRandomSampler(dataset.sample_weights, len(dataset.sample_weights))
        if balanced
        else None
    )

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )


def split_dataset(
    root_dir: Union[str, Path], max_uttr_per_class=2 ** 27 - 1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split Speech Commands into 3 set.
    
    Args:
        root_dir: speech commands dataset root dir
        max_uttr_per_class: predefined value in the original paper
    
    Return:
        train_list: [(class_name, audio_path), ...]
        valid_list: as above
    """
    train_list, valid_list = [], []

    for entry in Path(root_dir).iterdir():
        if not entry.is_dir() or entry.name == "_background_noise_":
            continue

        for audio_path in entry.glob("*.wav"):
            speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
            hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
            percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                100.0 / max_uttr_per_class
            )

            if percentage_hash < 10:
                valid_list.append((entry.name, audio_path))
            elif percentage_hash < 20:
                pass  # testing set is discarded
            else:
                train_list.append((entry.name, audio_path))

    return train_list, valid_list
