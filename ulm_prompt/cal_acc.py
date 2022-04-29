import json
from random import sample

from downstream_tasks import task_dataset

prompt_task = "IC"
model_date = "20220328"  # This means the number(code) of the model
sample_date = "20220328"  # This means the day you sampled
unit_model = "hubert100"

n_correct = 0
with open(f"./samples/samples_{sample_date}/samples_{prompt_task}_{unit_model}_{model_date}.json") as f:
    # === read === #
    data = json.load(f)
    ids = data.keys()
    # === cal === #
    for id in ids:
        label = data[id]["label"]
        predict = data[id]["predict"]
        if label == predict:
            n_correct += 1
print(f"correct prediction number: {n_correct}")
print(f"total data number:{len(data)}")
print(f"acc: {n_correct / len(data)}")
