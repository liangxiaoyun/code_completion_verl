from typing import Dict, Any, List
from verl.utils.reward_score.code_completion_graders.compute_score import ScoreComputer, DataSourceConfig, GraderConfig

reward_config = {
    # 通用评分配置
    "default": DataSourceConfig(
        name="default",
        graders=[
            GraderConfig("levenshtein", 'levenshtein_similarity', 0.2),
            GraderConfig("jaro_winkler", 'jaro_winkler_similarity', 0.2),
            GraderConfig("length_similarity", 'generate_length', 0.2),
            GraderConfig("combined_badcase", "badcase_detection", 0.4),
        ],
        description="通用评分配置"
    ),
}

def compute_score(data_source: str, solution_str: str, ground_truth: str,
                 extra_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    向后兼容的接口函数
    
    Args:
        data_source: 数据源
        solution_str: 模型输出
        ground_truth: 真实答案
        extra_info: 额外信息
        
    Returns:
        评分结果
    """
    reward_computer = ScoreComputer(config=reward_config)
    data_source = extra_info.get("data_source", "default")
    return reward_computer.compute_score(solution_str, ground_truth, extra_info, data_source)


# # Copyright 2020-2025 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import re
# import sys
# sys.path.append('/data_train/liangxiaoyun/projects/safety')
# from badcase_detect import Evaluation

# from difflib import SequenceMatcher
# import Levenshtein
# import jellyfish
# import math
# BadcaseDetect = Evaluation(test_badcase_list=["repeat", "dirty", "redundant_space", "comment_code"])

# def _is_code_similar(text1, text2, similarity_threshold=0.98):
#     """判断两个文本是否相似"""
#     matcher = SequenceMatcher(None, text1, text2)
#     similarity = matcher.ratio()
#     # if similarity >= similarity_threshold:
#     #     return True
#     # return False
#     return similarity

# def first_line_similar_as_gt_reward(completions, ground_truth, **kwargs):
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise
#     # return [1.0 if _is_code_similar(c.strip().split("\n")[0], gt.strip().split("\n")[0]) else 0.0 for c, gt in zip(completions, ground_truth)]
#         return [_is_code_similar(c.strip().split("\n")[0], gt.strip().split("\n")[0]) for c, gt in zip(completions, ground_truth)]

# def _lcp_space_optimized(completion, reference):
#     completion = re.sub(r'\s+', ' ', completion)
#     reference = re.sub(r'\s+', ' ', reference)
#     min_len = min(len(completion), len(reference))
#     for i in range(min_len):
#         if completion[i] != reference[i]:
#             return i
#     similar = min_len / len(reference)
#     return similar

# def levenshtein_similarity_reward(completion, ground_truth):
#     return Levenshtein.ratio(completion, ground_truth)  # 返回相似度 [0,1]

# def jaro_winkler_similarity_reward(completion, ground_truth):
#     if hasattr(jellyfish, "jaro_winkler_similarity"):
#         reward = jellyfish.jaro_winkler_similarity(completion, ground_truth) # 返回相似度 [0,1]
#     elif hasattr(jellyfish, "jaro_winkler"):
#         reward = jellyfish.jaro_winkler(completion, ground_truth) # 返回相似度 [0,1]
#     return reward

# def generate_length_reward(completion, ground_truth):
#     diff = abs(len(completion) - len(ground_truth))
#     reward = math.exp(-diff)
#     return reward

# def badcase_reward(completion, ground_truth, extra_info):
#     reward = 1.0
#     flag = False
#     prompt = extra_info["question"]
#     case = extract_content(prompt, special_token="v1")
#     for badcase_type in ["repeat", "dirty", "redundant_space", "comment_code"]:
#         try:
#             flag, _ = BadcaseDetect.test_badcase_detection[badcase_type](case, completion)
#         except Exception as e:
#             print(e)
#         if flag:
#             reward = 0.0
#             break
#     return reward

# #####################badcase reward########################
# def extract_content(prompt, special_token="v1"):
#     #special_token v1: <｜fim▁end｜>  <｜fim▁begin｜>
#     #special_token v2: <fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>
#     language, current_filename = "", ""
#     case = {}
#     #get language
#     language_start = prompt.find("language:")
#     if language_start != -1:
#         if special_token=="v1":
#             language = prompt[language_start:].split("\n")[0].split("<｜fim▁end｜>")[0].replace("language:", "")
#         else:
#             language = prompt[language_start:].split("\n")[0].split("<fim_suffix>")[0].replace("language:", "")
#     case["language"] = language.strip()

#     #get filename
#     filename_start = [m.start() for m in re.finditer(r'<filename>', prompt)]
#     if len(filename_start) == 0:
#         filename_start = [m.start() for m in re.finditer(r'<｜filename｜>', prompt)]
#     if len(filename_start):
#         if prompt[max(filename_start)-len("<neighbor>"):max(filename_start)] == "<neighbor>" or prompt[max(filename_start)-len("<neighbor>"):max(filename_start)] == "<｜neighbor｜>" :
#             current_filename = ""
#         else:
#             if special_token=="v1":
#                 current_filename = prompt[max(filename_start):].split("\n")[0].split("<｜fim▁end｜>")[0].replace("<filename>", "").replace("<｜filename｜>", "")
#             else:
#                 current_filename = prompt[max(filename_start):].split("\n")[0].split("<fim_suffix>")[0].replace("<filename>", "").replace("<｜filename｜>", "")

#     case["filename"] = current_filename

#     #get suffix, prefix
#     if special_token=="v1":
#         if "<｜fim▁end｜>" not in prompt:
#             suffix = ""
#             related_content = ""
#             if "<filename>" in prompt:
#                 prefix = "\n".join(prompt.split("<filename>")[-1].split("\n")[1:])
#             else:
#                 prefix = prompt
#         else:
#             related_content = prompt.split("<｜fim▁end｜>")[0]
#             suffix = prompt.split("<｜fim▁end｜>")[-1].split("<｜fim▁begin｜>")[0]    
#             prefix = prompt.split("<｜fim▁begin｜>")[-1]
#     else:
#         if "<fim_suffix>" not in prompt:
#             suffix = ""
#             related_content = ""
#             if "<filename>" in prompt:
#                 prefix = "\n".join(prompt.split("<filename>")[-1].split("\n")[1:])
#                 related_content = prompt.split(prefix)[0]
#             else:
#                 prefix = prompt
#         else:
#             related_content = prompt.split("<fim_suffix>")[0]
#             suffix = prompt.split("<fim_suffix>")[-1].split("<fim_prefix>")[0]    
#             prefix = prompt.split("<fim_prefix>")[-1].replace("<fim_middle>", "")


#     case["related_content"] = related_content
#     case["suffix"] = suffix
#     case["prefix"] = prefix
#     return case

# def badcase_repeat_reward(prompt_infos, completions, **kwargs):
#     rewards = []
#     for prompt_info, completion in zip(prompt_infos, completions):
#         flag = False
#         try:
#             flag, _ = BadcaseDetect.test_badcase_detection["repeat"](prompt_info, completion)
#         except Exception as e:
#             print(e)
#         reward = 0.0 if flag else 1.0
#         rewards.append(reward)
#     return rewards

# def badcase_dirty_reward(prompt_infos, completions, **kwargs):
#     rewards = []
#     for prompt_info, completion in zip(prompt_infos, completions):
#         flag = False
#         try:
#             flag, _ = BadcaseDetect.test_badcase_detection["dirty"](prompt_info, completion)
#         except Exception as e:
#             print(e)
#         reward = 0.0 if flag else 1.0
#         rewards.append(reward)
#     return rewards

# def badcase_redundant_space_reward(prompt_infos, completions, **kwargs):
#     rewards = []
#     for prompt_info, completion in zip(prompt_infos, completions):
#         flag = False
#         try:
#             flag, _ = BadcaseDetect.test_badcase_detection["redundant_space"](prompt_info, completion)
#         except Exception as e:
#             print(e)
#         reward = 0.0 if flag else 1.0
#         rewards.append(reward)
#     return rewards

# def badcase_comment_code_reward(prompt_infos, completions, **kwargs):
#     rewards = []
#     for prompt_info, completion in zip(prompt_infos, completions):
#         flag = False
#         try:
#             flag, _ = BadcaseDetect.test_badcase_detection["comment_code"](prompt_info, completion)
#         except Exception as e:
#             print(e)
#         reward = 0.0 if flag else 1.0
#         rewards.append(reward)
#     return rewards

# #####################badcase reward########################



# def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
#     r"""
#     Reward function that checks if the reasoning process is enclosed within `"<think>"` and `"</think>"` tags. The
#     function returns a reward of 1.0 if the format is correct, otherwise 0.0.

#     Args:
#         completions (`list[list[dict[str, str]]]`):
#             List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
#             containing the key `"content"` with the value being the text of the completion.
#         **kwargs:
#             Additional keyword arguments. This function does not use them, but they are required in the function
#             signature to ensure compatibility with trainers like [`GRPOTrainer`].

#     Returns:
#         `list[float]`:
#             A list of rewards, where each reward is 1.0 if the completion matches the expected format, otherwise 0.0.

#     Example:
#     ```python
#     >>> from trl.rewards import think_format_reward
#     >>> completions = [
#     ...     [{"content": "<think>\nThis is my reasoning.\n</think>\nThis is my answer."}],
#     ...     [{"content": "<think>\nThis is my reasoning.\nThis is my answer."}],
#     ... ]
#     >>> think_format_reward(completions)
#     [1.0, 0.0]
#     ```
#     """
#     pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]


# def compute_score_with_badcase(data_source, solution_str, ground_truth, extra_info=None):
#     lev_weight = 0.2
#     jaro_weight = 0.2
#     length_weight = 0.2
#     badcase_weight = 0.4
#     try:
#         score_lev = levenshtein_similarity_reward(solution_str, ground_truth)
#         score_jaro =  jaro_winkler_similarity_reward(solution_str, ground_truth)
#         score_length = generate_length_reward(solution_str, ground_truth)
#         score_badcase = badcase_reward(solution_str, ground_truth, extra_info)
#     except Exception as e:
#         print(f"get_format_score_and_hunks failed\nerror:\n{e}\nmodel output:\n{model_output}\n=============\n")
#         return 0.0

#     return score_lev * lev_weight + score_jaro * jaro_weight + score_length * length_weight + score_badcase * badcase_weight

# def compute_score(data_source, solution_str, ground_truth, extra_info=None):
#     try:
#         score_badcase = badcase_reward(solution_str, ground_truth, extra_info)
#     except Exception as e:
#         print(f"get_format_score_and_hunks failed\nerror:\n{e}\nmodel output:\n{model_output}\n=============\n")
#         return 0.0

#     return score_badcase

# if __name__ == "__main__":
#     print(_lcp_space_optimized("abcde\nfg\ntl", "abcdde\nfg\ntl"))