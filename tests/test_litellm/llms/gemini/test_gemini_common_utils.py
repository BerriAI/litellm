import pytest

from litellm.llms.gemini.common_utils import GeminiModelInfo


class TestGeminiModelInfo:
    """Test suite for GeminiModelInfo class"""

    def test_process_model_name_normal_cases(self):
        """Test process_model_name with normal model names"""
        gemini_model_info = GeminiModelInfo()

        # Test with normal model names
        models = [
            {"name": "models/gemini-1.5-flash"},
            {"name": "models/gemini-1.5-pro"},
            {"name": "models/gemini-2.0-flash-exp"},
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-flash",
            "gemini/gemini-1.5-pro",
            "gemini/gemini-2.0-flash-exp",
        ]

        assert result == expected

    def test_process_model_name_edge_cases(self):
        """Test process_model_name with edge cases that could be affected by strip() vs replace()"""
        gemini_model_info = GeminiModelInfo()

        # Test edge cases where model names end with characters from "models/"
        # These would be incorrectly processed if using strip("models/") instead of replace("models/", "")
        models = [
            {
                "name": "models/gemini-1.5-pro"
            },  # ends with 'o' - would become "gemini-1.5-pr" with strip()
            {
                "name": "models/test-model"
            },  # ends with 'l' - would become "gemini/test-mode" with strip()
            {
                "name": "models/custom-models"
            },  # ends with 's' - would become "gemini/custom-model" with strip()
            {
                "name": "models/demo"
            },  # ends with 'o' - would become "gemini/dem" with strip()
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-pro",  # 'o' should be preserved
            "gemini/test-model",  # 'l' should be preserved
            "gemini/custom-models",  # 's' should be preserved
            "gemini/demo",  # 'o' should be preserved
        ]

        assert result == expected

    def test_process_model_name_empty_list(self):
        """Test process_model_name with empty list"""
        gemini_model_info = GeminiModelInfo()

        result = gemini_model_info.process_model_name([])

        assert result == []

    def test_process_model_name_no_models_prefix(self):
        """Test process_model_name with model names that don't have 'models/' prefix"""
        gemini_model_info = GeminiModelInfo()

        models = [
            {"name": "gemini-1.5-flash"},  # No "models/" prefix
            {"name": "custom-model"},
        ]

        result = gemini_model_info.process_model_name(models)

        expected = [
            "gemini/gemini-1.5-flash",
            "gemini/custom-model",
        ]

        assert result == expected
