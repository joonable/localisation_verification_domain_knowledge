# Localisation and Verification of Where Domain Knowledge is Stored in Auto-Regressive Language Models

## Abstract
The beginning of large pre-trained language models (LMs), such as BERT and GPT families, based on the Transformer architecture, has transformed the paradigm of natural language processing (NLP). However, understanding their internal mechanisms becomes crucial for effective utilisation. Recent research has dissected pre-trained LMs to uncover their underlying behaviours, leading to methods for localising and analysing stored information. Despite these advances, previous studies focused primarily on factual knowledge and relied on structured templates, limiting real-world applications.

To address this limitation, we propose novel localisation methods that extend from factual to domain knowledge, accommodating unstructured text. We employ both causal mediation analysis using activations and weight-based perspectives, yielding two distinct localisation results for specific (sub)modules. This approach offers a comprehensive understanding of where domain knowledge is stored. We empirically demonstrate that these approaches outperform previous methods, validating neural components' influence on domain information. Additionally, we verify the findings through model-editing and compression experiments. Through an extensive evaluation involving over a hundred GPT-2 models, we establish the efficacy of our approaches, enabling their universal applicability. Our work contributes to advancing the understanding of pre-trained LMs and their practical utilisation in various domains.

## Jobs
![experiment_overview.png](imgs%2Fexperiment_overview_git.jpg)

### 1) clm job
- A clm job is to fine-tune the smallest gpt-2 with a specific domain dataset in the M2D2 dataset, employing causal language modelling (clm).
- clm job_cd: ``clm_{dataset_nm}``

### 2) loc job
- a loc job is to localise where domain knowledge is stored in the model.
- To set specific models utilised in localisation, you need to fill ``TRUE`` of ``FALSE`` in ``localisation`` column in ``./data/dataset_info.tsv``.
- This job requires the fine-tuned models for localisation because the aggregated result from the models is employed during the process.
- loc job_cd: 
  - ``locw``: localisation with wights
  - ``locq``: localisation with activations
  
### 3) pft job
- a pft job is to edit specific components in pre-trained gpt-2, using partial fine-tuning. 
- This job requires ``localisation_weight.tsv`` and ``localisation_activation.tsv`` in the ``data`` directory, which are two sets of rankings from localisation with weights and activations, respectively.
- One of three different rankings {loc_rank} decides the neural components to be **edited** depending on the {edt_prop}. 
- pft job_cd: ``pft{loc_rank}_{edt_prop}_{dataset_nm}``
  - ``{loc_rank}``: w(weight), a(activation), and base(baselines)
  - ``{edt_prop}``: 25, 50, and 75
  - e.g. ``pftw_50_nlin_l1``: partially fine-tuning the **top 50%** of (sub)modules of the result from the **localisation with weights**, using **nlin_l1** dataset.  

### 4) prn job
- a prn job is to compress specific components in pre-trained gpt-2, using pruning. 
- This job requires ``localisation_weight.tsv`` and ``localisation_activation.tsv`` in the ``data`` directory, which are two sets of rankings from localisation with weights and activations, respectively.
- One of three different rankings {loc_rank} decides the neural components to be **compressed** depending on the {comp_grp}. 
- pft job_cd: ``pft{loc_rank}_{edt_prop}_{dataset_nm}``
  - ``{loc_rank}``: w(weight), a(activation), and base(baselines)
  - ``{edt_prop}``: top(top 50%) and bot(bottom 50%)
  - e.g. ``prna_bot_nlin_l1``: partially fine-tuning the **bottom 50%** of (sub)modules of the result from the **localisation with activations**, using **nlin_l1** dataset.

### 5) eval job
- a eval job is to evaluate the models with the test dataset in in-domain or out-domain datasets.
- eval job_cd: ``eval{in_or_out}_{model_category}_{model_id}``
  -  ``{in_or_out}``: in(in-domain dataset) and out(out-domain dataset)
  - ``{model_category}``: ft(fine-tuned model), gpt2(pre-trained gpt-2), and , ver(the edited/compressed models for verification)
  - ``{model_id}``: this id is equivalent to the name of datasets for ft and gpt2 or the their job code for ver.
  - e.g. ``evalin_ver_prna_bot_nlin_l1``, ``evalin_ft_nlin_l1``, ``evalout_gpt2_nlin_l1``

## How to run the code?
You can specify the job_codes in the ``list_job_cds`` in ``run_job.sh``. Depending on the job_cd, the corresponding python file is executed.

```shell
#!/bin/bash
list_job_cds=("clm_nlin_l1" "prna_bot_nlin_l1" "evalin_ver_prna_bot_nlin_l1")

for _job_cd in "${list_job_cds[@]}"; do
  if [[ "$_job_cd" == "clm"* ]]; then
    python ./utils/run_clm.py --job_cd _job_cd
  elif [[ "$_job_cd" == "loc"* ]]; then
    python ./utils/run_loc.py --job_cd _job_cd
  elif [[ "$_job_cd" == "pft"* ]]; then
    python ./utils/run_pft.py --job_cd _job_cd
  elif [[ "$_job_cd" == "prn"* ]]; then
    python ./utils/run_prn.py --job_cd _job_cd
  elif [[ "$_job_cd" == "eval"* ]]; then
    python ./utils/run_eval.py --job_cd _job_cd
  else
    echo "WRONG job_cd"
  fi
  echo "$_job_cd is executed"
  sleep 5
done
```