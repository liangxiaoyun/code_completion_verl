# format_graders.py
import re
from verl.utils.reward_score.code_completion_graders.base_grader import BaseGrader
from typing import Dict, Any, List

class ThinkFormatGrader(BaseGrader):
    """思考格式评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        # 处理特殊的消息格式
        if isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict) and "content" in completion[0]:
                completion = completion[0]["content"]
        
        pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
        match = re.match(pattern, completion, re.DOTALL | re.MULTILINE)
        return 1.0 if match else 0.0


class EmptyOutputGrader(BaseGrader):
    """空输出评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        return 1.0 if completion == "" else 0.0


class MinLengthGrader(BaseGrader):
    """简单长度评分器，检查输出是否大于指定长度"""
    
    def __init__(self, min_length: int = 1):
        super().__init__()
        self.min_length = min_length
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        return float(len(completion) > self.min_length)