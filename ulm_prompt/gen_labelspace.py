import itertools
import json
from pathlib import Path

import numpy as np

from downstream_tasks import splits, task_dataset, tasks, unit_models

for unit_model in unit_models:
    result = {}
    for task in tasks:
        S = set()
        for split in splits:
            # === path === #
            data_dir = Path(f"./datasets/{unit_model}/{task_dataset[task]}/data")
            in_path = data_dir / split
            if not in_path.is_file():
                print(f"[INFO] Data not found in {data_dir}")
                break
            # === read === #
            with open(in_path, "r") as f:
                data = f.readlines()

            data = [d.split("|") for d in data]
            for d in data:
                labels = d[2].split()
                labels = [int(l) for l in labels]
                for l in labels:
                    S.add(l)
        labels = [str(l) for l in list(S)]
        if len(labels) != 0:
            result[task] = labels

    # === write === #
    labelspace_path = Path(f"./datasets/{unit_model}/labelspace.json")
    with open(labelspace_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] saved file to {labelspace_path}")
