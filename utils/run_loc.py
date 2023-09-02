import pickle

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import torch
import argparse
import datasets
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
from os import listdir
import json
import math
import sys
import os
from util_helper import get_model_config, get_module_info, get_dataset_info, tokenize_function, load_model
import pickle as pkl

def main():
    # job_cd
    job_cd = sys.argv[2]
    # list_job_cd = job_cd.split("_")
    job_gubun = job_cd
    # prn_prop = list_job_cd[1]
    # ds_nm = "_".join(list_job_cd[2:])
    assert job_gubun in ["locw", "loca"]
    # assert prn_prop in ["top", "bot"]

    # env_setting
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(cur_dir + "/../data/env_config.json") as json_file:
        env_config = json.load(json_file)

    prj_path = env_config["prj_path"]
    data_dir = prj_path + "data/"
    cache_dir = env_config["cache_dir"]
    device_id = "cpu" if job_gubun == "locw" else "gpu"
    # model_path_format = env_config["model_load_dir"] + "{model_id}"
    output_file_dir = prj_path + "data/output_logs/"

    # base_conf_path = data_dir + "clm_base_config.json"
    # job_conf_path = output_file_dir + f"{job_cd}.json"

    # if job_gubun in ["prnw", "prnbase"]:
    #     df_local_result_path = data_dir + "localisation_weight.tsv"
    # elif job_gubun == "prna":
    #     df_local_result_path = data_dir + "localisation_activation.tsv"

    ############################## DATA #####################################
    # dataset_info
    df_dataset, df_dataset_ratio = get_dataset_info()
    list_local_models = df_dataset.loc[
        (df_dataset.valid)
        & (df_dataset.localisation),
        "dataset"
    ].tolist()

    # model_info
    config_models = {model_id: {} for model_id in list_local_models}
    for model_id in list_local_models:
        config_models[model_id] = get_model_config(model_id)

    # module_info
    df_module, df_submodule = get_module_info()

    base_model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir).to(device_id)
    n_total_params = sum(p.numel() for p in base_model.parameters())

    if job_gubun == "locw":
        # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir).to(device_id)
        def get_diff_from_tensors_sub(a, b):
            return torch.abs(a - b).mean().item()

        # get diff_weights
        dict_diff_weights = {}
        for k, v_dict in config_models.items():
            print(f"Get difference in weights of gpt2 and {k}")
            tuned_model = GPT2LMHeadModel.from_pretrained(v_dict["model_path"]).to(device_id)
            for tuned_tup, base_tup in zip(list(tuned_model.named_parameters()), list(base_model.named_parameters())):
                assert tuned_tup[0] == base_tup[0]
                if tuned_tup[0].endswith("weight"):
                    dict_diff_weights[k + "___" + tuned_tup[0]] = get_diff_from_tensors_sub(tuned_tup[1], base_tup[1])
            del tuned_model

        # dict_diff_weights to DataFrame
        df_diff_weights = pd.DataFrame.from_dict(dict_diff_weights, orient='index').reset_index()
        df_diff_weights.columns = ["code", "diff"]
        df_diff_weights["dataset"] = df_diff_weights["code"].str.split("___").apply(lambda l: l[0])
        df_diff_weights["code"] = df_diff_weights["code"].str.split("___").apply(lambda l: l[1])
        df_diff_weights = df_diff_weights.merge(df_submodule, on=["code"])
        df_diff_weights = df_diff_weights.merge(df_dataset, on=["dataset"])

        for dataset, d in config_models.items():
            for k, v in d.items():
                if k != "model_path":
                    df_diff_weights.loc[df_diff_weights.dataset == dataset, k] = v
        df_diff_weights["log_train_steps"] = np.log(df_diff_weights['train_steps'])
        df_diff_weights['scaled_diff'] = df_diff_weights['diff'] / df_diff_weights["train_steps"] * 10 ** 4

        df_diff_weights.loc[df_diff_weights.is_investigated, "in_rank"] \
            = df_diff_weights.loc[df_diff_weights.is_investigated].groupby(["dataset"])["diff"].rank(ascending=False)
        df_diff_weights["in_rank"] = df_diff_weights["in_rank"].fillna(-1).astype(int)
        df_diff_weights.to_pickle(data_dir + "df_diff_weights.pkl")

        # get all_rank
        df_all_ranking = df_diff_weights \
            .groupby(["is_investigated", "layer", "display_id", "trace_id", "module", "submodule", "total"]) \
            [["diff", "scaled_diff"]].mean() \
            .reset_index()

        df_all_ranking.loc[df_all_ranking.is_investigated, "all_rank"] \
            = df_all_ranking.loc[df_all_ranking.is_investigated]["scaled_diff"].rank(ascending=False)
        df_all_ranking["all_rank"] = df_all_ranking["all_rank"].fillna(-1).astype(int)

        df_all_ranking.loc[df_all_ranking.is_investigated, "accum_params"] \
            = df_all_ranking.loc[df_all_ranking.is_investigated].sort_values("all_rank")["total"].cumsum()
        df_all_ranking["accum_params"] = df_all_ranking["accum_params"].fillna(-1).astype(int)

        df_all_ranking.loc[df_all_ranking.is_investigated, "threshold"] \
            = pd.cut(df_all_ranking.loc[df_all_ranking.is_investigated, "accum_params"], 4, labels=([25, 50, 75, 100]))
        df_all_ranking["threshold"] = df_all_ranking["threshold"].astype(float).fillna(-1).astype(int)

        df_local_weights = df_all_ranking.sort_values(["all_rank", "layer"])
        df_local_weights["display_id"] = df_local_weights["display_id"].apply(
            lambda col: str(int(col.split(".")[0]) + 1) + "." + ".".join(col.split(".")[1:])
        )

        df_local_weights = df_local_weights \
            .groupby(["threshold"], as_index=False, sort=False) \
            .agg({"display_id": [list], "total": ["count", "sum"]})

        df_local_weights.columns = ["threshold", "modules", "mod_cnt", "params"]
        df_local_weights["prop"] = df_local_weights["params"] / n_total_params
        df_local_weights["accum_params"] = df_local_weights["params"].cumsum()
        df_local_weights["accum_prop_params"] = df_local_weights["accum_params"] / n_total_params
        df_local_weights["modules"] = df_local_weights["modules"].apply(lambda l: ", ".join(l))
        df_local_weights.to_csv(data_dir + "localisation_weight.tsv", sep="\t", index=False)
    if job_gubun == "loca":
        # list_trace_ids
        list_trace_module_ids = df_module.loc[df_module.is_investigated].trace_id.unique()

        # DATA
        def save_clean_activation(m_id):
            def save_clean_activation_hook(module, _input, _output):
                if m_id.endswith('attn'):
                    clean_activations[m_id] = _output[0].detach()
                elif m_id.endswith('mlp'):
                    clean_activations[m_id] = _output.detach()
            return save_clean_activation_hook

        def add_noise_to_input(module, _input, _output):
            std = torch.std(_output)
            return _output + (std * 1.5) * torch.randn(_output[0].shape).to(device_id)

        def restore_activation(m_id):
            def restore_activation_hook(module, _input, _output):
                clean_activation = clean_activations[m_id]
                if m_id.endswith('attn'):
                    return tuple([clean_activation, tuple([_output[1][0], _output[1][1]])])
                elif m_id.endswith('mlp'):
                    return clean_activation
            return restore_activation_hook


        for model_id in list_local_models:
            output_file = output_file_dir + f"{job_gubun}_{model_id}.json"

            dataset = load_dataset("machelreid/m2d2", model_id, cache_dir=cache_dir)
            ds_test = dataset["test"].filter(lambda x: x['text'] != '')
            tokenized_datasets = ds_test.map(
                tokenize_function,
                batched=True,
                num_proc=8,
                remove_columns='text',  # TODO
                load_from_cache_file=True,
            )
            len_sentences = len(tokenized_datasets)

            # MODEL & HOOK
            clean_model = load_model(model_id, device_id)
            clean_model.eval()
            for m_id in list_trace_module_ids:
                clean_model.get_submodule(m_id).register_forward_hook(save_clean_activation(m_id))

                # Second run: corrupted run def
            corrupted_model = load_model(model_id, device_id)
            corrupted_model.eval()
            corrupted_model.get_submodule("transformer.wte").register_forward_hook(add_noise_to_input)

            # Third run: restored run def
            restored_model = load_model(model_id, device_id)
            restored_model.get_submodule("transformer.wte").register_forward_hook(add_noise_to_input)

            # RUN
            try:
                with open(output_file, 'r') as json_file:
                    list_results = json.load(json_file)
                for i, d in enumerate(list_results):
                    if len(d) == 0:
                        done_idx = i - 1
                        break
            except:
                list_results = [{} for x in range(len_sentences)]
                done_idx = 0

            for sentence_idx, data in enumerate(tokenized_datasets):
                if done_idx > sentence_idx: continue
                if sentence_idx % 1000 == 0:
                    print(f"sentence_idx: {sentence_idx}")

                inputs = torch.tensor(data['input_ids']).to(device_id)

                # First run: clean run
                clean_activations = {}
                with torch.no_grad():
                    clean_outputs = clean_model(inputs, labels=inputs.clone())
                    clean_loss = np.exp(clean_outputs.loss.item())

                # Second run: corrupted run
                with torch.no_grad():
                    corrupted_outputs = corrupted_model(inputs, labels=inputs.clone())
                    corrupted_loss = np.exp(corrupted_outputs.loss.item())

                # Third run: corrupted-with-restoration run
                restored_loss = {}
                with torch.no_grad():
                    for m_id in list_trace_module_ids:
                        hook = restored_model.get_submodule(m_id).register_forward_hook(restore_activation(m_id))
                        restored_outputs = restored_model(inputs, labels=inputs.clone())
                        restored_loss[m_id] = np.exp(restored_outputs.loss.item())
                        hook.remove()

                list_results[sentence_idx]['clean_loss'] = clean_loss
                list_results[sentence_idx]['corrupted_loss'] = corrupted_loss
                list_results[sentence_idx]['restored_loss'] = restored_loss

                if sentence_idx % 1000 == 0:
                    with open(output_file, 'w') as json_file:
                        json.dump(list_results, json_file)

            with open(output_file, 'w') as json_file:
                json.dump(list_results, json_file)

        dict_causal_effects = {k: {} for k in list_local_models}
        for ds_nm in list_local_models:
            print(ds_nm)
            dict_causal_effects[ds_nm]= {}
            file_path = output_file_dir + f"{job_gubun}_{ds_nm}.json"
            with open(file_path, "r") as json_file:
                dict_causal_effects[ds_nm]["differences"] = json.load(json_file)
                dict_causal_effects[ds_nm]["path"] = file_path
                dict_causal_effects[ds_nm]["total"] = len(dict_causal_effects[ds_nm]["differences"])
                dict_causal_effects[ds_nm]["sents"] = len(
                    [x for x in dict_causal_effects[ds_nm]["differences"] if len(x) > 0])
                print("{sents} / {total} / {is_same} / {file_path}".format(
                    sents=dict_causal_effects[ds_nm]["sents"],
                    total=dict_causal_effects[ds_nm]["total"],
                    is_same=dict_causal_effects[ds_nm]["total"] == dict_causal_effects[ds_nm][
                        "sents"],
                    file_path=file_path
                ))

        for ds_nm, d_value in dict_causal_effects.items():
            print(ds_nm)

            list_differences = d_value["differences"]
            list_df_temp = list(range(len(list_differences)))

            for i, d in enumerate(list_differences):
                TE = -(d['clean_loss'] - d["corrupted_loss"])
                IE = {}

                for m_id in list_trace_module_ids:
                    IE[m_id] = -(d["restored_loss"][m_id] - d["corrupted_loss"])

                list_df_temp[i] = pd.DataFrame.from_dict(IE, orient="index").reset_index()
                list_df_temp[i].columns = ["trace_id", "IE"]
                list_df_temp[i]["TE"] = TE
                list_df_temp[i]["clean_loss"] = d["clean_loss"]
                list_df_temp[i]["corrupted_loss"] = d["corrupted_loss"]
                list_df_temp[i]["restored_loss"] = [d["restored_loss"][m_id] for m_id in list_trace_module_ids]
                list_df_temp[i]["sent_id"] = i
                list_df_temp[i]["dataset"] = ds_nm

            df = pd.concat(list_df_temp, axis=0, ignore_index=True).reset_index(drop=True)
            df["list_code"] = df["trace_id"].apply(lambda code: code.split("."))
            df["layer"] = df["list_code"].apply(lambda l: int(l[2]))
            df["module"] = df["list_code"].apply(lambda l: l[3])
            del list_df_temp, df["list_code"]

            d_value["df"] = df
            d_value["df_agg"] = df.groupby(["dataset", "trace_id", "layer", "module"])\
                .agg({f"IE": ['mean', 'std']})

        with open("raw_causal_effects.pkl", "wb") as pkl_file:
            pickle.dump(dict_causal_effects, pkl_file)

        list_df_temp = []
        for ds_nm in dict_causal_effects.keys():
            df_temp = dict_causal_effects[ds_nm]["omcd"]["df"]
            df_temp["ranking"] = df_temp.groupby("sent_id")["omcd_IE_original"].rank(ascending=False)
            df_temp = df_temp.groupby(["trace_id"], as_index=False, sort=False)["ranking"].agg(["mean", "std"])
            df_temp["ranking_v2"] = df_temp["mean"].rank()
            df_temp["ds_nm"] = ds_nm
            list_df_temp.append(df_temp)

        df_ranking = pd.concat(list_df_temp, axis=0).reset_index(drop=True)
        del list_df_temp
        df_ranking = df_ranking.merge(
            df_module[["trace_id", "layer", "module", "display_id"]], on="trace_id", how='left')
        df_ranking = df_ranking.merge(
            df_dataset[["dataset", "source", "category"]].rename({"dataset": "ds_nm"}, axis=1),
            on="ds_nm", how='left')

        df_ranking_result = df_ranking.groupby(["trace_id"])[["ranking_v2"]].mean().rank()
        df_ranking_result.columns = ["ranking"]
        df_ranking_result = df_ranking_result.reset_index()
        df_ranking_result = df_module.merge(df_ranking_result.sort_values("ranking"), on="trace_id", how='left')
        df_ranking_result["ranking"] = df_ranking_result["ranking"].fillna(-1)

        df_ranking_result.loc[df_ranking_result.is_investigated, "accum_params"] \
            = df_ranking_result.loc[df_ranking_result.is_investigated].sort_values("ranking")["params"].cumsum()
        df_ranking_result["accum_params"] = df_ranking_result["accum_params"].fillna(-1).astype(int)

        df_ranking_result.loc[df_ranking_result.is_investigated, "threshold"] \
            = pd.cut(
                df_ranking_result.loc[df_ranking_result.is_investigated, "accum_params"],
                bins=4, labels=(["25", "50", "75", "100"]))
        df_ranking_result["threshold"] = df_ranking_result["threshold"].astype(float).fillna(-1).astype(int)

        df_local_activations = df_ranking_result.sort_values(["threshold", "ranking"])
        df_local_activations["display_id"] = df_local_activations["display_id"].apply(
            lambda col: str(int(col.split(".")[0]) + 1) + "." + ".".join(col.split(".")[1:])
        )
        df_local_activations = df_local_activations \
            .groupby(["threshold"], as_index=False, sort=False) \
            .agg({
            "display_id": [list],
            "params": ["count", "sum"],
        })

        df_local_activations.columns = ["threshold", "modules", "mod_cnt", "params"]
        df_local_activations["prop"] = df_local_activations["params"] / n_total_params
        df_local_activations["accum_params"] = df_local_activations["params"].cumsum()
        df_local_activations["accum_prop_params"] = df_local_activations["accum_params"] / n_total_params
        df_local_activations["modules"] = df_local_activations["modules"].apply(lambda l: ", ".join(l))

        df_local_activations.to_csv("localisation_activation_.tsv", sep="\t", index=False)