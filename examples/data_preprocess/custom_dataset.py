# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import pandas as pd
import datasets
import random
import json
from datasets import Dataset, load_dataset, concatenate_datasets, Features, Value
from collections import defaultdict
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def balance_sample(dataset, balance_sample_key, target_total, each_sample_key_num=None):
    # 按 data_source 分组
    grouped = defaultdict(list)
    for row in dataset:
        grouped[row[balance_sample_key]].append(row)

    balance_sample_keys = list(grouped.keys())
    num_balance_sample_keys = len(balance_sample_keys)

    if target_total == 0 and each_sample_key_num is None:
        #没有指定目标数量，按照最小类别数据量进行均衡采样
        base_per_balance_sample_key =  min(dataset.filter(lambda x: x[balance_sample_key] == key).num_rows for key in balance_sample_keys)
    elif each_sample_key_num is not None:
        # 每个类别的样本数
        base_per_balance_sample_key = each_sample_key_num
    else:
        # 每个语言的目标样本数（初步分配）
        base_per_balance_sample_key = target_total // num_balance_sample_keys

    # 存放采样结果
    sampled_data = []
    shortage = 0  # 记录不足的数量
    balance_sample_num = defaultdict(int)
    # 第一次采样：尽量取 base_per_data_source
    for key in balance_sample_keys:
        rows = grouped[key]
        if len(rows) >= base_per_balance_sample_key:
            sampled = random.sample(rows, base_per_balance_sample_key)
        else:
            sampled = rows[:]  # 全部取
            shortage += base_per_balance_sample_key - len(rows)
        sampled_data.extend(sampled)

    # 第二次补齐：从样本多的data_source中补足 shortage
    if each_sample_key_num is None and shortage > 0:
        print("第二次补齐：从样本多的data_source中补足!!!!!!!!!!!!")
        # 找出还有剩余的data_source
        surplus_pool = []
        for key in balance_sample_keys:
            rows = grouped[key]
            already_taken = sum(1 for r in sampled_data if r[balance_sample_key] == key)
            remaining = len(rows) - already_taken
            if remaining > 0:
                surplus_pool.extend([r for r in rows if r not in sampled_data])

        if len(surplus_pool) >= shortage:
            sampled_data.extend(random.sample(surplus_pool, shortage))
            
        else:
            # 如果剩余也不够，就全取
            sampled_data.extend(surplus_pool)
            

    for data in sampled_data:
        balance_sample_num[data[balance_sample_key]] += 1

    # 转回 Dataset
    balanced_dataset = Dataset.from_list(sampled_data)

    print("balance_sample_num: ", balance_sample_num)
    return balanced_dataset

def sft_data_balance_sample():
    #按照语言、为空和非空的均衡采样 
    # data_path = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_language_balabce_sample_10000.jsonl"
    data_path = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balabce_sample_10000.jsonl"
    none_data_path = "/data_large/liangxiaoyun/data/online_complete_data/250530_250630_completion_empty/output_001_eos_filtered_thres_0.14_sample_2000.jsonl"
    save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balance_sample_10000_none_1000.parquet"
    dataset = datasets.load_dataset(
            "json",
            data_files=data_path,
            split="train"
        )
    none_dataset = datasets.load_dataset(
            "json",
            data_files=none_data_path,
            split="train"
        )

    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")
            answer_raw = example.pop("completion", "")
            question = question_raw
            solution = answer_raw
            data = {
                "data_source": "custom_data",
                "prompt": question,
                "ability": "code_completion",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("train", "250530_250630_completion_v1"), with_indices=True)
    none_dataset = none_dataset.map(function=make_map_fn("train", "250530_250630_completion_v1_none"), with_indices=True)
     
    # 只保留指定列
    target_columns = ["data_source",
                "prompt",
                "ability",
                "reward_model",
                "extra_info"]  # 你想保留的列名
    ds1 = dataset.select_columns(target_columns)
    ds2 = none_dataset.select_columns(target_columns)

    # 合并
    ds_merged = concatenate_datasets([ds1, ds2])
    ds_merged.to_parquet(save_parquet_file)  # 保存成一个 parquet 文件
    print(f"ds1: {len(ds1)}; ds2: {len(ds2)}; ds_merged: {len(ds_merged)}")
    print("save_path: ", save_parquet_file)
    

def to_parquet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data.jsonl")
    parser.add_argument("--data_source", default="custom_data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--fim_format", default="v1")
    parser.add_argument("--balance_sample_key", default=None)
    parser.add_argument("--sample_num", default=0)
    parser.add_argument("--each_sample_key_num", default=None)

    args = parser.parse_args()
    # data_path = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000.jsonl"#"openai/gsm8k"
    try:
        dataset = datasets.load_dataset(
            "json",
            data_files=args.data_path,
            split="train"
        )
    except:
        df = pd.read_json(args.data_path, lines=True, dtype=str)  # 全部读成字符串
        dataset = Dataset.from_pandas(df)

    sample_num = int(args.sample_num)
     # 按data_source均衡采样
    if args.balance_sample_key:
        # 获取每种data_source的样本数
        sample_dataset = balance_sample(dataset, args.balance_sample_key, sample_num, int(args.each_sample_key_num))
        print("ori_data_length: ", len(dataset), "sampled_data_length: ", len(sample_dataset))
        save_path = args.data_path.replace(".jsonl", f"_balance_{args.balance_sample_key}_sample-{len(sample_dataset)}.parquet")
    elif sample_num != 0 and sample_num < len(dataset):
        print("in sample! ori_data_num: ", len(dataset))
        sample_dataset = dataset.train_test_split(test_size=sample_num, seed=42)["test"]
        print("in sample! after_sample_data_num: ", len(dataset))
        save_path = args.data_path.replace(".jsonl", f"_sample-{sample_num}.parquet")
    else:
        sample_dataset = dataset
        save_path = args.data_path.replace(".jsonl", ".parquet")

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")
            answer_raw = example.pop("completion", "")

            
            # atomic_information = example.pop("atomic_information")
            # related_content = atomic_information["related_content"]
            # suffix = atomic_information["suffix_content"]
            # prefix = atomic_information["prefix_content"]
            # if args.fim_format=="v1":
            #     ds_prompt = f"{related_content}<｜fim▁end｜>{suffix}<｜fim▁begin｜>{prefix}"
            # if args.fim_format=="v2":
            #     ds_prompt = f"{related_content}<｜fim▁end｜>{suffix}<｜fim▁begin｜>{prefix}<｜fim▁hole｜>"
            # elif args.fim_format=="v3":
            #     ds_prompt = f"{related_content}<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>"
            # else:#默认 v1
            #     ds_prompt = f"{related_content}<｜fim▁end｜>{suffix}<｜fim▁begin｜>{prefix}"
            # question_raw = ds_prompt
            # answer_raw = example.pop("backend_infer_result1")
            
            # import pdb;pdb.set_trace()

            question = question_raw
            solution = answer_raw
            data = {
                "data_source": args.data_source,
                "prompt": question,
                "ability": "code_completion",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    dataset = sample_dataset.map(function=make_map_fn(args.split), with_indices=True)
    

        
    dataset.to_parquet(save_path)
    print("save_path: ", save_path)

#python3 custom_dataset.py --data_source online_accept_test_data --data_path /data_train/liangxiaoyun/datas/online_complete_data/compl-0719-v1.1/0724_online_2_20250717-20250719_full.jsonl --split test
#python3 custom_dataset.py --data_source 250530_250630_deduplicated_001_less_8000 --data_path /data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000.jsonl --split train

def parquet_merge():
    # 读取两个 parquet 文件
    # parquet_file_1 = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_language_balabce_sample_10000_none_1000.parquet"
    parquet_file_1 = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balance_sample_10000_none_1000.parquet"
    parquet_file_2 = "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet"
    save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balabce_sample_10000_none_1000_serious_badcase_2434.parquet"
    
    ds1 = load_dataset("parquet", data_files=parquet_file_1, split="train")
    ds2 = load_dataset("parquet", data_files=parquet_file_2, split="train")

    # 只保留指定列
    target_columns = ["data_source",
                "prompt",
                "ability",
                "reward_model",
                "extra_info"]  # 你想保留的列名
    ds1 = ds1.select_columns(target_columns)
    ds2 = ds2.select_columns(target_columns)

    #ds2 更改数据类型
    # 用 map 给字典加 key
    def add_key(example, data_source):
        example["extra_info"]["data_source"] = data_source
        example["data_source"] = data_source
        return example

    
    ds1 = ds1.map(lambda x: add_key(x, "250530_250630_completion_v1"))
    ds2 = ds2.map(lambda x: add_key(x, "250701-250730_serious_badcase"))
    # import pdb;pdb.set_trace()

    # 合并
    ds_merged = concatenate_datasets([ds1, ds2])
    ds_merged.to_parquet(save_parquet_file)  # 保存成一个 parquet 文件
    print(f"ds1: {len(ds1)}; ds2: {len(ds2)}; ds_merged: {len(ds_merged)}")
    print("save_path: ", save_parquet_file)

def jsonl_parquet_merge_to_parquet():
    # data_list = {
    #     "250530_250630_completion_v1": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balabce_sample_10000.jsonl",
    #     "250530_250630_completion_v1_output_empty": "/data_large/liangxiaoyun/data/online_complete_data/250530_250630_completion_empty/output_001_eos_filtered_thres_0.14_sample_2000.jsonl",
    #     "250701-250730_serious_badcase": "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet"
    # }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_output_length_language_balance_sample_10000_none_2000.parquet"
    
    # data_list = {
    #     "250530_250630_completion_v1": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balanced_sample_20000.jsonl",
    #     "250530_250630_completion_v1_output_empty": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001_eos_filtered_thres_0.14_cursor_type_output_length_language_balanced_sample_2000.jsonl",
    #     "250701-250730_serious_badcase": "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet"
    # }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_20000_none_2000_serious_badcase_2434.parquet"
    # gt_key = "completion"

    # data_list = {
    #     "250530_250630_completion_v1": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balanced_sample_10000.jsonl",
    #     "250530_250630_completion_v1_output_empty": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001_eos_filtered_thres_0.14_cursor_type_output_length_language_balanced_sample_2000.jsonl",
    #     "250701-250730_serious_badcase": "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet"
    # }
    # data_list_repeat_num = {
    #     # "250701-250730_serious_badcase": 1
    # }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_10000_none_2000_serious_badcase_2434.parquet"
    # gt_key = "completion"

    # data_list = {
    #     "250530_250630_completion_v1": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balanced_sample_10000.jsonl",
    #     "250530_250630_completion_v1_output_empty": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001_eos_filtered_thres_0.14_cursor_type_output_length_language_balanced_sample_10000.jsonl",
    #     "250701-250730_serious_badcase": "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet"
    # }
    # data_list_repeat_num = {
    #     "250701-250730_serious_badcase": 1
    # }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_10000_none_10000_serious_badcase_2434-dul4.parquet"
    # gt_key = "completion"

    # data_list = {
    #     "test_python_output_empty_data": "/data_fast/jiaruiyu/workstation/user_data_analysis/Completion_User_Seq_Miner/data/test_dataset/2508_05_07_python_testset/deduplicated/random_selected/labelled/output/empty.jsonl",
    #     "test_python_output_not_empty_data": "/data_fast/jiaruiyu/workstation/user_data_analysis/Completion_User_Seq_Miner/data/test_dataset/2508_05_07_python_testset/deduplicated/random_selected/labelled/output/not_empty.jsonl"
    # }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/test_python_output.parquet"
    # gt_key = "reference_code"

    data_list = {
        "250530_250630_completion_v1": [
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_python_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_go_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_java_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_cpp_c_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_javascript_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_typescript_cursor_type_output_length_balanced_sample_10000.jsonl",
            "/data_train/liangxiaoyun/datas/online_complete_data/compl-20250820-20250906/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001-004_less_8000_filter0819_others_cursor_type_output_length_balanced_sample_10000.jsonl"
            ], 
        # "250530_250630_completion_v1_output_empty": "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001_eos_filtered_thres_0.14_cursor_type_output_length_language_balanced_sample_10000.jsonl",
        "250530_250630_completion_v1_output_empty": [
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_go_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_python_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_java_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_javascript_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_typescript_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_cpp_c_balanced_sample_5000.jsonl",
            "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/output_001-006_eos_filtered_thres_0.14_cursor_type_output_length_others_balanced_sample_5000.jsonl",
        ],
        "250701-250730_serious_badcase": "/data_large/liangxiaoyun/data/online_complete_data/250701-250730_badcase/250701-250730_badcase_serious_train_data_balance_data_source_sample-2434.parquet",
        "250701_250831_ast_error": "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_1-4_inference_over_1time_sft_sp.jsonl",
    }
    data_list_repeat_num = {
        "250701-250730_serious_badcase": 6,
        "250701_250831_ast_error": 2
    }
    # save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_60000_none_10000_serious_badcase_2434-dul4.parquet"
    save_parquet_file = "/data_train/liangxiaoyun/datas/completion_sft_datas/train_data_merge_user_distillation/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_deduplicated_001_less_8000_filter0819_cursor_type_output_length_language_balance_sample_60000_none_30000_serious_badcase_2434-dul6_ast_error_7988-dul2.parquet"
    gt_key = "completion"

    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question_raw = example.pop("prompt")
            answer_raw = example.pop(gt_key, "") if data_source != "test_python_output_empty_data" else ""
            question = question_raw
            solution = answer_raw
            data = {
                "data_source": data_source,
                "prompt": question,
                "ability": "code_completion",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "data_source": data_source,
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn
        
    def add_key(example, data_source):
        example["extra_info"]["data_source"] = data_source
        example["data_source"] = data_source
        return example

    dataset_list = []
    ori_dataset_length = defaultdict(int)
    dataset_length = defaultdict(int)
    for data_source, data_paths in data_list.items():
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        for data_path in data_paths:
            print("data_path: ", data_path)
            if data_path.endswith(".jsonl"):
                dataset = datasets.load_dataset(
                        "json",
                        data_files=data_path,
                        split="train"
                    )
                
                dataset = dataset.map(function=make_map_fn("train", data_source), with_indices=True)
        
            else: #parquet
                dataset = load_dataset("parquet", data_files=data_path, split="train")
                dataset = dataset.map(lambda x: add_key(x, data_source))

            # 只保留指定列
            target_columns = ["data_source",
                        "prompt",
                        "ability",
                        "reward_model",
                        "extra_info"]  # 你想保留的列名
            ds = dataset.select_columns(target_columns)
            
            if data_source in data_list_repeat_num.keys():
                repeat_num = data_list_repeat_num[data_source]
            else:
                repeat_num = 1
            for r in range(repeat_num):
                dataset_list.append(ds)    
                dataset_length[data_source] += len(ds)
            ori_dataset_length[data_source] += len(ds)

    # 合并
    ds_merged = concatenate_datasets(dataset_list)
    ds_merged.to_parquet(save_parquet_file)  # 保存成一个 parquet 文件
    print("dataset_length: ", dataset_length, "ori_dataset_length: ", ori_dataset_length, "final_length: ", len(ds_merged))
    print("save_path: ", save_parquet_file)

if __name__ == "__main__":
    # to_parquet()
    # parquet_merge()
    # sft_data_balance_sample()
    jsonl_parquet_merge_to_parquet()
    # language_num = defaultdict(int)
    # with open("/data_fast/jiaruiyu/workstation/user_data_analysis/Completion_User_Seq_Miner/data/250530_250630_completion_v1.1_sp_training_0805_empty_fim_deduplication_0.75_5/eos_filtered_thres_0.14/output_002.jsonl", "r") as f:
    #     for line in f.readlines():
    #         data =json.loads(line)
    #         language_num[data["language"]] += 1
    # sorted_language_num = dict(sorted(language_num.items(), key=lambda x: x[1], reverse=True))
    # print(sorted_language_num)