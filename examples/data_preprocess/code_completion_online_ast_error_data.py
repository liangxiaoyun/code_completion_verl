import os
import json
import re
from collections import defaultdict
from glob import glob
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("/data_train/liangxiaoyun/projects/safety")
sys.path.append("/data_train/liangxiaoyun/projects/verl/verl/utils/reward_score/code_completion_graders")
from ast_graders import ASTErrorGrader
from dataset import JsonlDataset, batch_process_key, my_collate_fn
from badcase_detect import Evaluation, completion_filter, prompt_filter
from combine_sft_train_data import count_tokens, md5_of_file, badcase_detection, extract_content
TEST_BADCASE_LIST = ["repeat", "dirty", "redundant_space", "comment_code"]
BAD_CASE_EVAL = Evaluation(test_badcase_list=TEST_BADCASE_LIST)

class OnlineData():
    def __init__(self, data_path):
        self.data_path = data_path
        self.filter_num = defaultdict(int)
        self.model_path = "/data_fast/jiaruiyu/workstation/user_data_analysis/LLM_post_training/output/250530_250630_completion_v1_sp_training_0731_deduplication_0.75_5_50k_user_50k_github_from_v4.0"
        self.samplingparams = SamplingParams(temperature=0.5, max_tokens=215, n=5, stop=["<|EOS|>", "<|EOT|>"])
        self.vllm_model = LLM(model=self.model_path, enforce_eager=True, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.ast_judge = ASTErrorGrader()

    def data_filter(self, data):
        #badcase检测
        if not re.search("empty", data["data_source"]):
            badcase_type = badcase_detection(data, detect_key="completion", need_completion_filter=True, train_type="sft", special_token="v1")
        else:
            badcase_type = badcase_detection(data, detect_key="completion", need_completion_filter=False, train_type="sft", special_token="v1")
        if badcase_type is not None:
            self.filter_num[badcase_type] += 1
            return True
        return False

    def is_ast_error(self, completion, prompt):
        extra_info = {"question": prompt}
        reward = self.ast_judge.compute_reward(completion, "", extra_info)
        if reward:
            return False
        return True
    
    def main(self):
        # data["data_source"] = "250701_250831_ast_error"
        save_path = self.data_path.replace(".jsonl", "_inference.jsonl")
        print("data_path: {}; model_path: {}; save_path: {}".format(self.data_path, self.model_path, save_path))
        start_idx, end_idx = 0, -1
        if os.path.exists(save_path):
            output_data = open(save_path, 'a')
            with open(save_path, 'r') as f:
                line_count = sum(1 for _ in f)
                print(f"Output file exists. Appending to {save_path}. Line count: {line_count}")
            if line_count >= 1:
                meta_data_flag = True
            # Update start index
            start_idx = max(0, line_count - 1)
        else:
            output_data = open(save_path, 'w')
        # Load data
        if end_idx == -1:
            end_idx = None
        print(f"start_idx: {start_idx}; end_idx: {end_idx}")
        dataset = JsonlDataset(self.data_path)
        data_keys = dataset.data_keys
        dataset = dataset[start_idx:end_idx]
        data_num = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=my_collate_fn)
        all_ast_error_num = 0
        for batch in tqdm(dataloader, desc="Inference"):
            # import pdb;pdb.set_trace()
            batch_prompts = batch_process_key(batch, prompt_key="infer_prompt")
            #推理
            batch_output = self.vllm_model.generate(batch_prompts, self.samplingparams)
            all_ast_error = True
            for idx, output in enumerate(batch_output):
                save_case = {"completion_result": [], "data_source": "250701_250831_ast_error"}
                for k in data_keys:
                    save_case[k] = batch[k][idx].item() if isinstance(batch[k][idx], torch.Tensor) else batch[k][idx]
                for out in output.outputs:
                    completion = out.text 
                    completion_length = len(out.token_ids)
                    is_ast_error = self.is_ast_error(completion, batch_prompts[idx])
                    print(f"!!!!!!!!!!!completion-{completion_length}-{is_ast_error}!!!!!!!!!!!!")
                    print(completion)
                    if not is_ast_error:
                        all_ast_error = False
                    save_case["completion_result"].append({
                        "completion": completion,
                        "completion_length": completion_length,
                        "is_ast_error": is_ast_error
                    })
                output_data.write(json.dumps(save_case, ensure_ascii=False) + "\n")
            if all_ast_error:
                all_ast_error_num += 1 
        print(f"data_num: {data_num}, all_ast_error_num: {all_ast_error_num}, all_ast_error_rate: {all_ast_error_num/data_num}")
        print("save_path: ", save_path)

def ast_error_data_to_train():
    data_path_list = ["/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_0001_inference.jsonl",
        "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_0002_inference.jsonl",
        "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_0003_inference.jsonl",
        "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_0004_inference.jsonl"]
    save_data_path = "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_1-4_inference_over_1time_sft_sp.jsonl"

    sf = open(save_data_path, "w")
    ori_data_num, keep_data_num = 0, 0
    for data_path in data_path_list:
        ast_error_num = defaultdict(int)
        empty_ast_error_num = defaultdict(int)
        data_num = 0
        with open(data_path, "r") as f:
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                if "completion_result" not in data.keys(): continue
                error_num, empty_error_num = 0, 0
                ori_data_num += 1
                for completion in data["completion_result"]:
                    if completion["is_ast_error"]:
                        # import pdb;pdb.set_trace()
                        error_num += 1
                        if completion["completion"] == "":
                            empty_error_num += 1
                ast_error_num[error_num] += 1
                empty_ast_error_num[empty_error_num] += 1
                data_num += 1
                #save
                if error_num > 1: #6次里面有3次及以上是ast_error
                    save_case ={
                        "data_source": data["data_source"],
                        "prompt": data["infer_prompt"],
                        "completion": data["backend_infer_result1"],
                        "language": data["language"]}
                    sf.write(json.dumps(save_case, ensure_ascii=False) + "\n")
                    keep_data_num += 1
        print(f"data_path: {data_path}, data_num: {data_num}, ast_error_num: {ast_error_num}, empty_ast_error_num: {empty_ast_error_num}")
    # sf.close()
    print(f"ori_data_num: {ori_data_num}; keep_data_num: {keep_data_num}")
    print(f"save_data_path: {save_data_path}")

if __name__ == "__main__":
    data_path = "/data_large_v2/liangxiaoyun/datas/online_datas/coml-07-08-ast-error-filter-dedup-final/coml-07-08-ast-error-data-dedup_0004.jsonl"
    # online = OnlineData(data_path)
    # online.main()
    ast_error_data_to_train()
