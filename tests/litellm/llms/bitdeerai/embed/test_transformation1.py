import os
import sys
from typing import Dict,List,Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bitdeerai.embed.transformation import BitdeerAIEmbeddingConfig

import pytest

class TestBitdeerAIEmbeddingConfig:
    def test_get_supported_openai_params(self):
        """验证 get_supported_openai_params 返回空列表"""
        supported_params = BitdeerAIEmbeddingConfig.get_supported_openai_params()
        assert isinstance(supported_params, list)
        assert len(supported_params) == 0, "Expected empty list"

    @pytest.mark.parametrize(
        "non_default_params, optional_params, model, drop_params, expected",
        [
            # 基础测试：直接返回 optional_params
            (
                {"temperature": 0.5},
                {"input": "test text", "model": "bitdeerai/test"},
                "model",
                {"drop": "this"},
                {"input": "test text", "model": "bitdeerai/test"},
            ),
            # 测试无 optional_params（空字典）
            (
                {},
                {},
                "model",
                None,
                {},
            ),
            # 测试非默认参数与 model 的存在不影响结果
            (
                {"user": "test_user"},
                {"input": "another text"},
                "another_model",
                {"unknown_param": "value"},
                {"input": "another text"},
            ),
        ],
    )
    def test_map_openai_params(
        self,
        non_default_params,
        optional_params,
        model,
        drop_params,
        expected,
    ):
        """验证 map_openai_params 正确返回 optional_params（直通）"""
        result = BitdeerAIEmbeddingConfig.map_openai_params(
            non_default_params,
            optional_params,
            model,
            drop_params=drop_params,
        )
        assert result == expected, f"Expected {expected}, got {result}"