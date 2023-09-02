from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

from os import listdir
import os
import json
import math
import pandas as pd

# env_setting
cur_dir = os.path.dirname(os.path.abspath(__file__))
with open(cur_dir + "/../data/env_config.json") as json_file:
    env_config = json.load(json_file)

prj_path = env_config["prj_path"]
data_dir = prj_path + "data/"
model_path_format = env_config["model_load_dir"] + "{model_id}"
cache_dir = env_config["cache_dir"]
output_file_dir = prj_path + "data/output_logs/"

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)

def load_model(model_id, device_id="cpu"):
    if model_id == "gpt":
        return GPT2LMHeadModel.from_pretrained(model_id, cache_dir=cache_dir).to(device_id)
    else:
        return GPT2LMHeadModel.from_pretrained(model_path_format.format(model_id=model_id)).to(device_id)


def tokenize_function(examples):
    output = gpt2_tokenizer(examples['text'], max_length=1024, truncation=True)
    return output


def get_model_config(model_id):
    model_config = {}
    model_config["model_path"] = model_path_format.format(model_id=model_id)
    json_path = model_config["model_path"] + "/trainer_state.json"
    with open(json_path, "r") as json_file:
        trainer_state = json.load(json_file)
        model_config["val_loss"] = trainer_state["best_metric"]
        model_config["val_ppl"] = math.exp(trainer_state["best_metric"])
        model_config["train_steps"] = int(
            trainer_state["best_model_checkpoint"].split("/")[-1].split("-")[-1])
        model_config["train_eps"] = float(trainer_state["epoch"])
    return model_config


def get_model_path(model_id):
    model_path_format = "/rds/general/user/jj1122/home/projects/m2d2/dataset/{model_id}/models"
    ckpt_path_format = "/checkpoint-{ckpt}"

    if model_id == "gpt2":
        model_path = "gpt2"
#         ckpt = "zs"
    else:
        model_path = model_path_format.format(model_id=model_id)
        l_dir = listdir(model_path)

        if all([len(x.split(".")) == 1 for x in l_dir]):
            ckpt = max([int(x.split("-")[1]) for x in l_dir])
            model_path += ckpt_path_format.format(ckpt=ckpt)
#         else:
#             ckpt = "final"
    return model_path

def _parse_code_submodule(row):
    list_code = row.code.split(".")
    row["trace_id"] = ".".join(row["code"].split(".")[:-1])
    #     row["component_id"] = ".".join(row["code"].split(".")[1:-1])
    is_in_layer = row["code"].startswith("transformer.h")

    if is_in_layer:
        row["layer"] = int(list_code[2])
        row["module"] = list_code[3]
    else:
        row["module"] = list_code[1]
        if row["module"] == "ln_f":
            row["layer"] = int(99)
        elif row["module"] in ["wte", "wpe"]:
            row['layer'] = int(-1)

    if row["module"] in ["attn", "mlp"]:
        row["submodule"] = list_code[-2]
    else:
        row["submodule"] = row["module"]
    is_investigated = not (row["submodule"].startswith("w") or row["submodule"].startswith("ln"))

    row["w_or_b"] = list_code[-1]
    row["is_in_layer"] = is_in_layer
    row["is_investigated"] = is_investigated

    return row


def get_module_info():
    # df_submodule
    dict_n_parmas = {
        tup[0]: tup[1].numel()
        for tup in GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir) \
            .to("cpu").named_parameters()
    }
    df_submodule = pd.DataFrame.from_dict(dict_n_parmas, orient="index").reset_index()
    df_submodule.columns = ["code", "params"]
    df_submodule = df_submodule.apply(lambda row: _parse_code_submodule(row), axis=1)
    df_params = df_submodule.groupby("trace_id")["params"].sum().to_frame().reset_index()
    df_params.columns = ["trace_id", "total"]

    df_submodule["display_id"] = df_submodule["layer"].astype(str) + "." + df_submodule["module"]

    df_submodule.loc[df_submodule.module != df_submodule.submodule, "display_id"] \
        = df_submodule.loc[df_submodule.module != df_submodule.submodule, "display_id"] \
            + "." + df_submodule.loc[df_submodule.module != df_submodule.submodule, "submodule"]

    df_submodule = df_submodule.merge(df_params, on=["trace_id"])

    # df_module
    df_module = df_submodule[:]
    df_module["trace_id"] = df_module["trace_id"].apply(
        lambda sub_trace_id: ".".join(sub_trace_id.split(".")[:-1])
        if sub_trace_id.split(".")[-2] in ["attn", "mlp"] else sub_trace_id
    )
    df_module = df_module \
        .groupby(["trace_id", "is_in_layer", "is_investigated", "layer", "module"], sort=False, as_index=False) \
        [["params"]].sum()
    df_module["display_id"] = df_module["layer"].astype(str) + "." + df_module["module"]
    return df_module, df_submodule


def get_dataset_info():
    df_dataset = pd.read_csv(data_dir + "dataset_info.tsv", sep="\t")
    # df_dataset = df_dataset[["dataset_id", "latex_id", "dataset", "source", "category"]]
    df_dataset = df_dataset.reset_index(names=["idx"])
    df_dataset_ratio = pd.read_csv(data_dir + "dataset_ratio.tsv", sep="\t")
    return df_dataset, df_dataset_ratio


