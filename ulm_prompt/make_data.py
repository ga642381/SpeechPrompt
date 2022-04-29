import itertools
from pathlib import Path

from downstream_tasks import splits, task_dataset, tasks, unit_models

for split, task, unit_model in itertools.product(splits, tasks, unit_models):
    dataset = task_dataset[task]
    # == in == #
    data_dir = Path(f"./datasets/{unit_model}/{dataset}/data")
    in_path = data_dir / split
    if not in_path.is_file():
        print(f"[INFO] File not found: {in_path}")
        continue

    # == out == #
    dest_dir = Path(f"./datasets/{unit_model}/{dataset}/data_prompt")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / (split + ".txt")

    with open(in_path, "r") as f:
        data = f.readlines()
    data = [d.split("|") for d in data]
    out_data = [d[1] + " " + "<s>" + " " + d[2] for d in data]

    with open(out_path, "w") as f:
        f.writelines(out_data)
    print(f"[INFO] saved file to {out_path}")
