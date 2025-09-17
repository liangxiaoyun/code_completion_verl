# badcase_graders.py
import sys
import re
from typing import Dict, Any
from verl.utils.reward_score.code_completion_graders.utils import extract_content
from verl.utils.reward_score.code_completion_graders.base_grader import BaseGrader
sys.path.append('/data_train/liangxiaoyun/projects/safety')
from badcase_detect import Evaluation

# 初始化BadcaseDetect
BadcaseDetect = Evaluation(test_badcase_list=["repeat", "dirty", "redundant_space", "comment_code"])

class BadcaseBaseGrader(BaseGrader):
    """Badcase检测的基类"""
    
    def __init__(self, badcase_type: str):
        super().__init__()
        self.badcase_type = badcase_type
        
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not extra_info or "question" not in extra_info:
            return 1.0
            
        prompt = extra_info["question"]
        prompt_info = extract_content(prompt, special_token="v1")
        
        try:
            flag, _ = BadcaseDetect.test_badcase_detection[self.badcase_type](prompt_info, completion)
            return 0.0 if flag else 1.0
        except Exception as e:
            print(f"Error in {self.badcase_type} detection: {e}")
            return 1.0


class RepeatBadcaseGrader(BadcaseBaseGrader):
    """重复检测评分器"""
    def __init__(self):
        super().__init__("repeat")


class DirtyBadcaseGrader(BadcaseBaseGrader):
    """脏数据检测评分器"""
    def __init__(self):
        super().__init__("dirty")


class RedundantSpaceBadcaseGrader(BadcaseBaseGrader):
    """冗余空格检测评分器"""
    def __init__(self):
        super().__init__("redundant_space")


class CommentCodeBadcaseGrader(BadcaseBaseGrader):
    """注释代码检测评分器"""
    def __init__(self):
        super().__init__("comment_code")


class CombinedBadcaseGrader(BaseGrader):
    """组合所有badcase检测的评分器"""
    
    def __init__(self):
        super().__init__()
        self.graders = [
            RepeatBadcaseGrader(),
            DirtyBadcaseGrader(),
            RedundantSpaceBadcaseGrader(),
            CommentCodeBadcaseGrader()
        ]
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        """检测所有badcase类型，任何一个检测到就返回0"""
        for grader in self.graders:
            reward = grader.compute_reward(completion, ground_truth, extra_info, **kwargs)
            if reward == 0.0:  # 检测到badcase
                return 0.0
        return 1.0