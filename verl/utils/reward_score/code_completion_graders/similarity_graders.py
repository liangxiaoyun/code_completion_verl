# similarity_graders.py
import re
import math
import Levenshtein
import jellyfish
from difflib import SequenceMatcher
from verl.utils.reward_score.code_completion_graders.base_grader import BaseGrader
from typing import Dict, Any

class FirstLineSimilarityGrader(BaseGrader):
    """首行相似度评分器"""
    
    def __init__(self, similarity_threshold: float = 0.98):
        super().__init__()
        self.similarity_threshold = similarity_threshold
    
    def _is_code_similar(self, text1: str, text2: str) -> float:
        """判断两个文本的相似度"""
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
    
    def compute_reward(self, completion: str, ground_truth: str = None, 
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        return self._is_code_similar(
            completion.strip().split("\n")[0], 
            ground_truth.strip().split("\n")[0]
        )


class LevenshteinSimilarityGrader(BaseGrader):
    """Levenshtein相似度评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        return Levenshtein.ratio(completion, ground_truth)


class JaroWinklerSimilarityGrader(BaseGrader):
    """Jaro-Winkler相似度评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
            
        if hasattr(jellyfish, "jaro_winkler_similarity"):
            return jellyfish.jaro_winkler_similarity(completion, ground_truth)
        elif hasattr(jellyfish, "jaro_winkler"):
            return jellyfish.jaro_winkler(completion, ground_truth)
        else:
            return 0.0


class LengthRewardGrader(BaseGrader):
    """长度奖励评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
        
        if completion == "":
            return 0.0
        
        diff = abs(len(completion) - len(ground_truth))
        return math.exp(-diff)


class LCPSpaceOptimizedGrader(BaseGrader):
    """空间优化的最长公共前缀评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        if not ground_truth:
            return 0.0
            
        completion = re.sub(r'\s+', ' ', completion)
        reference = re.sub(r'\s+', ' ', ground_truth)
        min_len = min(len(completion), len(reference))
        
        for i in range(min_len):
            if completion[i] != reference[i]:
                return i / len(reference)
        
        return min_len / len(reference)