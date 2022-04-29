import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from downstream_tasks import splits, task_dataset, tasks, unit_models

# ===== CONFIG ===== #
random.seed(999)
method = "freq"  # freq / kmeans (kmeans doesn't work well in the preliminary study)
# ===== CONFIG ===== #

# (1) Kmeans method
def gen_kmeans_metadata(labelspaces, unit_embeds, file_path):
    def find_nearest(cs, values):
        result = []
        for c in cs:
            candidates = sorted(range(len(values)), key=lambda i: norm(values[i] - c))
            for i in candidates:
                if i not in result:
                    result.append(i)
                    break
        return result

    # === main === #
    result = {}
    for task in labelspaces.keys():
        labelspace = labelspaces[task]
        # == Kmeans == #
        unit_kmeans = KMeans(n_clusters=len(labelspace), random_state=0).fit(unit_embeds)
        centroids = unit_kmeans.cluster_centers_
        c_nearest = find_nearest(centroids, unit_embeds)
        random.shuffle(c_nearest)

        # == mapping == #
        mapping = {}
        for i in range(len(labelspace)):
            mapping[str(labelspace[i])] = str(c_nearest[i])
        result[task] = mapping
    # === write file === #
    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] saved kmeans centroids metadata: {file_path}")


# (2) freq. top-k method
def gen_freq_metadata(labelspaces, unit_model, file_path):
    result = {}
    for task in labelspaces.keys():
        labelspace = labelspaces[task]
        if len(labelspace) == 0:
            print(f"[INFO] labelspace of task: {task} is None")
            break
        data_dir = Path(f"./datasets/{unit_model}/{task_dataset[task]}/data_prompt")
        # === freq_stats === #
        src_count_dict = {}
        tgt_count_dict = {}
        with open(data_dir / "train.txt") as f:
            data = f.readlines()
        srcs = [d.split("<s>")[0].strip() for d in data]
        tgts = [d.split("<s>")[1].strip() for d in data]
        for src in srcs:
            units = src.split()
            for u in units:
                if u not in src_count_dict.keys():
                    src_count_dict[u] = 0
                else:
                    src_count_dict[u] += 1

        for tgt in tgts:
            units = tgt.split()
            for u in units:
                if u not in tgt_count_dict.keys():
                    tgt_count_dict[u] = 0
                else:
                    tgt_count_dict[u] += 1

        src_sorted = sorted(src_count_dict, key=src_count_dict.get, reverse=True)
        tgt_sorted = sorted(tgt_count_dict, key=tgt_count_dict.get, reverse=True)
        # === mapping === #
        mapping = {}
        for l in labelspace:
            i = tgt_sorted.index(l)
            mapping[str(l)] = str(src_sorted[i])
        result[task] = mapping

    # === write file === #
    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[INFO] saved freq. stats unit metadata: {file_path}")


def verbalize(labels: str, v_table: dict):
    labels = labels.split()
    v_labels = [v_table[l] for l in labels]
    return " ".join(v_labels)


def main(unit_model):
    # === stage 1: get label space === #
    labelspace_path = Path(f"./datasets/{unit_model}/labelspace.json")
    with open(labelspace_path, "r") as f:
        task_labelspace = json.load(f)

    # === stage 2: generate metadata === #
    verbal_path = Path(f"./datasets/{unit_model}/verbal.json")
    if not verbal_path.is_file():
        if method == "kmeans":
            ckpt = torch.load("./pretrained_models/{unit_model}_lm/checkpoint_best.pt")
            unit_embeds = ckpt["model"]["decoder.embed_tokens.weight"][4:].numpy()
            gen_kmeans_metadata(task_labelspace, unit_embeds, verbal_path)
        elif method == "freq":
            gen_freq_metadata(task_labelspace, unit_model, verbal_path)

    # === stage 3: verbalize === #
    with open(verbal_path, "r") as f:
        v_table = json.load(f)

    for task, split in itertools.product(tasks, splits):
        # === dir === #
        data_dir = Path(f"./datasets/{unit_model}/{task_dataset[task]}/data_prompt")
        dest_dir = Path(f"./datasets/{unit_model}/{task_dataset[task]}/data_{method}")
        dest_dir.mkdir(exist_ok=True)

        # === file === #
        in_path = data_dir / f"{split}.txt"
        out_path = dest_dir / f"{split}.txt"
        with open(in_path, "r") as f:
            data = f.readlines()

        # === process === #
        sep = "<s>"
        data = [d.strip().split(sep) for d in data]
        with open(out_path, "w") as f:
            for d in data:
                src = d[0].strip()
                tgt = d[1].strip()
                tgt = verbalize(tgt, v_table[task])
                line = src + " " + sep + " " + tgt + "\n"
                f.write(line)


if __name__ == "__main__":
    for unit_model in unit_models:
        main(unit_model)

# === 2D TSNE === #
# tsne = TSNE(n_components=2, learning_rate="auto", init="random")
# labels = unit_kmeans.labels_
# XY = tsne.fit_transform(unit_embeds)
# u_labels = np.unique(labels)
# for i in u_labels:
#     plt.scatter(XY[labels == i, 0], XY[labels == i, 1])
# plt.legend()
# plt.show()
