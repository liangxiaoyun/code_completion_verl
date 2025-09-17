import re
from verl.utils.reward_score.code_completion_graders.base_grader import BaseGrader
from typing import Dict, Any, List, Tuple, Optional
from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser
from verl.utils.reward_score.code_completion_graders.utils import extract_content

class ASTErrorGrader(BaseGrader):
    """AST错误检测评分器"""
    
    def __init__(self):
        super().__init__()
        # 缓存parser以提高性能
        self.parser_cache = {}
        self.supported_languages = {
            "python", "go", "cpp", "c", "javascript", "typescript", "java", "scala", "kotlin"
        }
    def _is_position_in_node(self, node: Node, position: Tuple[int, int]):
        if position[0] < node.start_point[0] or position[0] > node.end_point[0]:
            return False
        if position[0] == node.start_point[0] and position[1] < node.start_point[1]:
            return False
        if position[0] == node.end_point[0] and position[1] > node.end_point[1]:
            return False
        return True

    def _get_cursor_types(self, node: Node, position: Tuple[int, int], prev_types: List[str]) -> List[List[str]]:
        in_node = self._is_position_in_node(node, position)
        if not in_node:
            return []
        outputs = []
        if node.children: # 如果有子节点，则对子节点递归获取子类型

            children_outputs = []
            for child in node.children:
                children_outputs.extend(self._get_cursor_types(child, position, prev_types + [node.type]))

            if children_outputs:
                outputs.extend(children_outputs)
            else: # 如果子节点没有输出，说明光标在当前节点内，但不在子节点内（比如空行、空格等）
                outputs.append(prev_types + [node.type])
        else:
            outputs.append(prev_types + [node.type])

        return outputs

    
    def _has_ast_error(self, prefix: str, suffix: str, completion: str, parser: Parser) -> bool:
        """检查代码补全是否引入了AST错误"""
        if not prefix or not suffix:
            return False
        # 获取补全前后代码
        prev_code = prefix + suffix # 补全前的代码
        next_code = prefix + completion + suffix # 补全后的代码
        mock_next_code = prefix + completion + "a" + suffix # mock补全后的代码，添加一个a避免多数补全不完整的情况

        # 获取光标位置
        prev_prefix_lines = prefix.split("\n")
        prev_position = (len(prev_prefix_lines) - 1, len(prev_prefix_lines[-1]))
        next_prefix_lines = (prefix + completion).split("\n")
        next_position = (len(next_prefix_lines) - 1, len(next_prefix_lines[-1]))
        mock_next_position = next_position

        # 获取光标所处节点类型
        prev_root = parser.parse(prev_code.encode("utf-8")).root_node
        prev_cursor_types = self._get_cursor_types(prev_root, prev_position, [])
        next_root = parser.parse(next_code.encode("utf-8")).root_node
        next_cursor_types = self._get_cursor_types(next_root, prev_position, []) + self._get_cursor_types(next_root, next_position, [])
        mock_next_root = parser.parse(mock_next_code.encode("utf-8")).root_node
        mock_next_cursor_types = self._get_cursor_types(mock_next_root, prev_position, []) + self._get_cursor_types(mock_next_root, mock_next_position, [])

        error_in_prev = False
        for node_types in prev_cursor_types:
            if "ERROR" in node_types:
                error_in_prev = True
        error_in_next = False
        for node_types in next_cursor_types:
            if "ERROR" in node_types:
                error_in_next = True
        error_in_mock_next = False
        for node_types in mock_next_cursor_types:
            if "ERROR" in node_types:
                error_in_mock_next = True

        if (not error_in_prev) and error_in_next and error_in_mock_next:
            return True

        return False
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        """检测语法结构正确性问题：存在问题返回0，否则返回1"""
        reward = 1.0
        
        if not extra_info or "question" not in extra_info:
            return reward
        
        prompt = extra_info["question"]
        prompt_info = extract_content(prompt, special_token="v1")
        
        language = prompt_info["language"].lower()
        prefix = prompt_info["prefix"]
        suffix = prompt_info["suffix"]
        
        # 检查语言是否支持
        if language not in self.supported_languages:
            return reward
        
        try:
            # 获取或创建parser（使用缓存）
            if language not in self.parser_cache:
                self.parser_cache[language] = get_parser(language)
            parser = self.parser_cache[language]
            
            # 检查是否有AST错误
            ast_error = self._has_ast_error(prefix, suffix, completion, parser)
            if ast_error:
                return 0.0
        except Exception as e:
            print(f"Error in AST error detection: {e}")
        
        return reward

class ASTStopGrader(BaseGrader):
    """AST停止判断评分器"""
    
    def compute_reward(self, completion: str, ground_truth: str = None,
                      extra_info: Dict[str, Any] = None, **kwargs) -> float:
        # TODO: 实现语法结构自动停止正确性检测
        # 停止正确，reward为1，否则为0
        return 1.0  # 占位实现