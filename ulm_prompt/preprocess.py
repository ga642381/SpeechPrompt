import json
import os

from downstream_tasks import task_dataset, tasks, unit_models

if __name__ == "__main__":
    for unit_model in unit_models:
        for task in tasks:
            unit_transcribed_dir = f"./datasets/{unit_model}/{task_dataset[task]}/data_freq"
            # 1
            dict_path = f"./pretrained_models/{unit_model}_lm/dict.txt"
            # 2
            destdir = os.path.join(f"./data-bins/{unit_model}/{task}-data-bin")
            # 3
            train_data = os.path.join(unit_transcribed_dir, "train.txt")
            valid_data = os.path.join(unit_transcribed_dir, "valid.txt")
            test_data = os.path.join(unit_transcribed_dir, "test.txt")
            if os.path.exists(destdir):
                print(f"[INFO] {destdir} exists! SKIP!")
                continue
            cmd = f"fairseq-preprocess --only-source --srcdict {dict_path}\
            --trainpref {train_data} --validpref {valid_data} --testpref {test_data}\
            --destdir {destdir} --workers 16"
            os.system(cmd)
