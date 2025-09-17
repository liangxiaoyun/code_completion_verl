# base_grader.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Union

class BaseGrader(ABC):
    """抽象基类，所有Grader都需要继承这个类"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute_reward(self, completion: str, ground_truth: str = None, 
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        """
        计算奖励分数
        
        Args:
            completion: 模型生成的完成内容
            ground_truth: 真实答案（可选）
            extra_info: 额外信息字典
            **kwargs: 其他参数
            
        Returns:
            奖励分数 (float)
        """
        pass