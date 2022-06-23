#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Sample from a trained LM; hacked fairseq-interactive
"""
import ast
import json
import os
import random
from collections import namedtuple
from email.policy import default
from pathlib import Path
from unittest import result

import numpy as np
import torch
import tqdm
from fairseq import checkpoint_utils, options, tasks, utils

Batch = namedtuple("Batch", "ids src_tokens src_lengths")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def slice_by_length(data, max_len=400):
    def slice(src, src_len, max_len=400):
        # === get n_splice === #
        n_slice = 1
        l = src_len
        while l > max_len:
            n_slice *= 2
            l /= 2
        l = int(l)
        # === slice === #
        result = []
        s = src.split()
        for i in range(n_slice):
            s_tokens = s[i * l : (i + 1) * l]
            if s_tokens[-1] != "<s>":
                s_tokens.append("<s>")
            s_str = " ".join(s_tokens)
            result.append(s_str)
        return result

    for i, d in enumerate(data):
        if d["src_len"] > max_len:
            src = d["src"]
            srcs = slice(src, d["src_len"], max_len)  # function
            for i, s in enumerate(srcs):
                a = {
                    "id": f'{d["id"]}_{i}',
                    "file_name": d["file_name"],
                    "src": s,
                    "label": d["label"],
                    "src_len": len(s[i].split()),
                    "label_len": d["label_len"],
                }
                data.append(a)


def make_batches(lines, args, task, max_positions):
    tokens = [task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long() for src_str in lines]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.dataset.max_tokens,
        max_sentences=args.dataset.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.dataset.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch["id"],
            src_tokens=batch["net_input"]["src_tokens"],
            src_lengths=batch["net_input"]["src_lengths"],
        )


def main(args):
    arg_input = args.input_src
    arg_slicing = args.slicing
    arg_output = args.output
    arg_raw_file = args.raw_file
    arg_debug = args.debug
    arg_sample_size = args.samples_per_prompt
    arg_task_prompt = args.task_prompt

    try:
        from fairseq.dataclass.utils import convert_namespace_to_omegaconf

        args = convert_namespace_to_omegaconf(args)
    except:
        pass

    # if args.max_tokens is None and args.max_sentences is None:
    if args.common.seed is not None:
        np.random.seed(args.common.seed)
        utils.set_torch_seed(args.common.seed)

    if args.generation.sampling:
        args.generation.nbest = args.generation.beam = arg_sample_size

    task = tasks.setup_task(args.task)

    overrides = ast.literal_eval(args.common_eval.model_overrides)

    # load base prompt model
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.common_eval.path.split(os.pathsep),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # load prompt and sep token
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    model_dict = models[0].state_dict()
    for name, param in model_dict.items():
        if "prompt" in name or "sep" in name:
            model_dict[name] = torch.load(arg_task_prompt)["model"][name]
        else:
            continue
    models[0].load_state_dict(model_dict)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.generation.replace_unk)

    max_positions = utils.resolve_max_positions(task.max_positions(), *[model.max_positions() for model in models])

    # ========== #
    output_path = Path(arg_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {}
    output_file = output_path

    data = []
    with open(arg_input, "r") as f:
        lines = f.readlines()

    with open(arg_raw_file, "r") as f:
        raws = f.readlines()
        file_names = [r.split("|")[0] for r in raws]

    # random.seed(0)
    # lines = random.sample(lines, 64)
    # lines = sorted(lines, key=lambda l: len(l.split()), reverse=True)
    data = []
    split = [x.split("<s>") for x in lines]
    srcs = [x[0] + "<s>" for x in split]
    labels = [x[1].strip() for x in split]
    for i in range(len(file_names)):
        data.append(
            {
                "id": i,
                "file_name": file_names[i],
                "src": srcs[i],
                "label": labels[i],
                "src_len": len(srcs[i].split()),
                "label_len": len(labels[i].split()),
            }
        )
    if arg_slicing:
        slice_by_length(data)
    # data = sorted(data, key=lambda d: d["src_len"], reverse=True)[100:200]
    prompts = [d["src"] for d in data]
    # ========== #

    # if args.generation.prefix_size >= 0:
    #     prompts = [" ".join(l.split()[: args.generation.prefix_size]) for l in prompts]

    # if arg_debug:
    #     prompts = prompts[:10]

    # ===== generate sequence ===== #
    generator = task.build_generator(models, args.generation)
    start_id = 0
    pbar = tqdm.tqdm(total=len(data))
    for batch in make_batches(prompts, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        src_tokens = src_tokens.cuda()
        src_lengths = src_lengths.cuda()

        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
        }

        results = []
        translations = task.inference_step(generator, models, sample)
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((i + start_id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.common_eval.post_process)

            # Process top predictions
            for hypo_id, hypo in enumerate(hypos):
                _hypo_tokens, hypo_str, _alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.common_eval.post_process,
                )

                detok_hypo_str = hypo_str
                utterance = detok_hypo_str
                prediction = utterance.removeprefix(src_str).strip()
                assert utterance != prediction  # important when the generated sequence (max-len-b) is shorter than src

                data_point = {
                    "file_name": data[id]["file_name"],
                    "src": data[id]["src"],
                    "label": data[id]["label"],
                    "predict": prediction,
                }

                out_data[data[id]["id"]] = data_point
            pbar.update(1)
        start_id += len(results)

    # === write result === #
    with open(output_file, "w") as f:
        json.dump(out_data, f, indent=4)


def cli_main():
    # === arguments === #
    #  hacked the fairseq cmd
    #  can be better...
    parser = options.get_interactive_generation_parser()
    parser.add_argument("--prompt_task", type=str, default="IC")
    parser.add_argument("--unit_model", type=str, default="hubert100")
    parser.add_argument("--model_date", type=str, default="00000000")
    parser.add_argument("--sample_date", type=str, default="00000000")
    args = parser.parse_args()

    parser.add_argument("--output", type=str, default=None, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--input_src", type=str, default=None, required=True)
    parser.add_argument("--slicing", action="store_true")
    parser.add_argument("--task_prompt", type=str, default=None)
    parser.add_argument("--raw_file", type=str, default=None)

    prompt_task = args.prompt_task
    model_date = args.model_date  # This means the number of the model
    sample_date = args.sample_date  # This means the day you sample, and this means today!
    unit_model = args.unit_model
    verbal = "freq"  # prompt (naive) or freq
    from downstream_tasks import task_dataset

    input_args = [
        f"data-bins/{unit_model}/{prompt_task}-data-bin/",
        f"--path=./checkpoints/{prompt_task}_{unit_model}_checkpoints_{model_date}/base_prompt_model.pt",
        f"--task_prompt=./checkpoints/{prompt_task}_{unit_model}_checkpoints_{model_date}/checkpoint_best.pt",
        "--user-dir=./prompt_lm_module",
        "--task=language_modeling",
        "--sampling",
        "--sampling-topk=1",
        "--seed=1",
        f"--input_src=datasets/{unit_model}/{task_dataset[prompt_task]}/data_{verbal}/test.txt",
        f"--raw_file=datasets/{unit_model}/{task_dataset[prompt_task]}/data/test",
        f"--output=samples/samples_{sample_date}/samples_{prompt_task}_{unit_model}_{model_date}.json",
        "--max-len-a=0",
        "--max-len-b=1500",
        "--num-workers=12",
        "--prefix-size=-1",
        "--batch-size=32",
        "--fp16",
        "--samples-per-prompt=1",
    ]
    args = options.parse_args_and_arch(parser, input_args)

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    main(args)


if __name__ == "__main__":
    cli_main()
