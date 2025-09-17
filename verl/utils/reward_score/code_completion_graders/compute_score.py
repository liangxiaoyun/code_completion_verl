# compute_score.py
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field
from verl.utils.reward_score.code_completion_graders.base_grader import BaseGrader
from verl.utils.reward_score.code_completion_graders.similarity_graders import (
    FirstLineSimilarityGrader,
    LevenshteinSimilarityGrader,
    JaroWinklerSimilarityGrader,
    LCPSpaceOptimizedGrader,
    LengthRewardGrader
)
from verl.utils.reward_score.code_completion_graders.badcase_graders import (
    CombinedBadcaseGrader,
    RepeatBadcaseGrader,
    DirtyBadcaseGrader,
    RedundantSpaceBadcaseGrader,
    CommentCodeBadcaseGrader
)
from verl.utils.reward_score.code_completion_graders.format_graders import (
    ThinkFormatGrader,
    EmptyOutputGrader, 
    MinLengthGrader
)
from verl.utils.reward_score.code_completion_graders.ast_graders import (
    ASTStopGrader,
    ASTErrorGrader
)

@dataclass
class GraderConfig:
    """单个Grader的配置"""
    name: str
    show_name: str
    weight: float = 1.0
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """数据源的配置"""
    name: str
    graders: List[GraderConfig]
    description: str = ""


class ScoreComputer:
    """
    可配置的评分计算器
    
    Example:
        # 方式1: 使用默认配置
        computer = ScoreComputer()
        
        # 方式2: 使用配置字典
        config = {
            "default": {
                "graders": [
                    {"name": "levenshtein", "weight": 0.4},
                    {"name": "jaro_winkler", "weight": 0.3},
                    {"name": "length", "weight": 0.3}
                ]
            }
        }
        computer = ScoreComputer(config=config)
        
    """
    
    # 所有可用的Grader类
    AVAILABLE_GRADERS = {
        'levenshtein': LevenshteinSimilarityGrader,
        'jaro_winkler': JaroWinklerSimilarityGrader,
        'length_similarity': LengthRewardGrader,
        'first_line_similarity': FirstLineSimilarityGrader,
        'lcp_space_optimized': LCPSpaceOptimizedGrader,
        'combined_badcase': CombinedBadcaseGrader,
        'repeat': RepeatBadcaseGrader,
        'dirty': DirtyBadcaseGrader,
        'redundant_space': RedundantSpaceBadcaseGrader,
        'comment_code': CommentCodeBadcaseGrader,
        'empty': EmptyOutputGrader,
        'min_length': MinLengthGrader,
        'think_format': ThinkFormatGrader,
        'ast_stop': ASTStopGrader,
        'ast_error': ASTErrorGrader,
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化ScoreComputer
        
        Args:
            config: 配置字典
            config_path: 配置文件路径（支持json/yaml）
        """
        self.grader_instances = {}  # 缓存的grader实例
        self.data_source_configs = {}  # 数据源配置
        
        # 加载配置
        if config:
            self._load_config_from_dict(config)
        else:
            self._load_default_config()

    
    def _load_config_from_dict(self, config: Dict[str, Any]):
        """从字典加载配置"""
        for data_source, source_config in config.items():
            graders = source_config.graders
            self.data_source_configs[data_source] = DataSourceConfig(
                name=data_source,
                graders=graders,
                description=source_config.description
            )
    
    def _load_default_config(self):
        """加载默认配置"""
        default_config = {
            # 通用评分配置
            "default": DataSourceConfig(
                name="default",
                graders=[
                    GraderConfig("levenshtein", 'levenshtein_similarity', 0.4),
                    GraderConfig("jaro_winkler", 'jaro_winkler_similarity', 0.3),
                    GraderConfig("length_similarity", 'generate_length', 0.3)
                ],
                description="通用评分配置"
            ),
            
            # Badcase检测配置
            "250701-250730_serious_badcase": DataSourceConfig(
                name="250701-250730_serious_badcase",
                graders=[
                    GraderConfig("combined_badcase", "badcase_detection", 0.5),
                    GraderConfig("min_length", "badcase_generate_length", 0.5, params={"min_length": 1})
                ],
                description="Badcase检测数据"
            ),
            
            # 空输出检测配置
            "250530_250630_completion_v1_output_empty": DataSourceConfig(
                name="250530_250630_completion_v1_output_empty",
                graders=[
                    GraderConfig("empty", "empty", 1.0)
                ],
                description="空输出检测"
            ),
            
            "test_python_output_empty_data": DataSourceConfig(
                name="test_python_output_empty_data",
                graders=[
                    GraderConfig("empty", "empty", 1.0)
                ],
                description="Python空输出检测"
            ),
            
            # AST停止检测配置
            "open_source_ast_stop": DataSourceConfig(
                name="open_source_ast_stop",
                graders=[
                    GraderConfig("ast_stop", "ast_stop", 0.5),
                    GraderConfig("length_similarity", "ast_stop_generate_length", 0.5)
                ],
                description="AST停止检测"
            ),
            
            # AST错误检测配置
            "250701_250831_ast_error": DataSourceConfig(
                name="250701_250831_ast_error",
                graders=[
                    GraderConfig("ast_error", "ast_error", 0.5),
                    GraderConfig("min_length", "ast_error_generate_length", 0.5, params={"min_length": 1})
                ],
                description="AST错误检测"
            )
        }
        
        self.data_source_configs = default_config
    
    def get_grader_instance(self, grader_name: str, params: Dict[str, Any] = None) -> BaseGrader:
        """
        获取或创建Grader实例
        
        Args:
            grader_name: Grader名称
            params: 初始化参数
            
        Returns:
            Grader实例
        """
        # 生成缓存key
        cache_key = f"{grader_name}_{json.dumps(params or {}, sort_keys=True)}"
        
        # 检查缓存
        if cache_key in self.grader_instances:
            return self.grader_instances[cache_key]
        
        # 创建新实例
        if grader_name not in self.AVAILABLE_GRADERS:
            raise ValueError(f"Unknown grader: {grader_name}")
        
        grader_class = self.AVAILABLE_GRADERS[grader_name]
        if params:
            grader = grader_class(**params)
        else:
            grader = grader_class()
        
        # 缓存实例
        self.grader_instances[cache_key] = grader
        return grader
        
    def compute_score(self, solution_str: str, ground_truth: str = None,
                     extra_info: Dict[str, Any] = None,
                     data_source: str = None) -> Dict[str, Any]:
        """
        计算评分
        
        Args:
            solution_str: 模型输出
            ground_truth: 真实答案
            extra_info: 额外信息
            data_source: 数据源（可以从extra_info中获取）
            
        Returns:
            包含总分和各项分数的字典
        """
        try:
            # 确定数据源
            if data_source is None and extra_info:
                data_source = extra_info.get("data_source", "default")
            if data_source is None:
                data_source = "default"
            
            # 获取配置
            if data_source not in self.data_source_configs:
                print(f"Unknown data source: {data_source}, using default config")
                data_source = "default"
            
            config = self.data_source_configs[data_source]
            
            # 计算各项分数
            scores = {}
            weighted_score = 0
            total_weight = 0
            
            for grader_config in config.graders:
                if not grader_config.enabled:
                    continue
                
                # 获取grader实例
                grader = self.get_grader_instance(
                    grader_config.name,
                    grader_config.params
                )
                
                # 计算分数
                score = grader.compute_reward(
                    solution_str,
                    ground_truth,
                    extra_info
                )
                
                # 记录分数
                scores[f"{grader_config.show_name}_reward"] = score
                weighted_score += score * grader_config.weight
            
            # 返回结果
            result = {"score": weighted_score}
            result.update(scores)
            
            return result
            
        except Exception as e:
            print(f"compute_score failed\nerror:\n{e}\nmodel output:\n{solution_str}\n")
            return {"score": 0.0}