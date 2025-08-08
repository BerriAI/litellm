"""
Test file for Vertex AI Online Prediction
"""

import unittest

from .transformation import VertexAIOnlinePredictionTransformation
from .types import EndpointConfig


class TestVertexAIOnlinePredictionTransformation(unittest.TestCase):
    """Test cases for VertexAIOnlinePredictionTransformation"""

    def test_parse_endpoint_from_model_simple(self):
        """Test parsing endpoint from simple model format"""
        model = "vertex_ai/endpoints/1234567890123456789"
        result = VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
        
        self.assertEqual(result.endpoint_id, "1234567890123456789")
        self.assertEqual(result.project_id, "")
        self.assertEqual(result.location, "")

    def test_parse_endpoint_from_model_full(self):
        """Test parsing endpoint from full model format"""
        model = "vertex_ai/my-project/us-central1/endpoints/1234567890123456789"
        result = VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
        
        self.assertEqual(result.endpoint_id, "1234567890123456789")
        self.assertEqual(result.project_id, "my-project")
        self.assertEqual(result.location, "us-central1")

    def test_parse_endpoint_from_model_invalid(self):
        """Test parsing invalid model format"""
        model = "invalid/model/format"
        
        with self.assertRaises(ValueError):
            VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)

    def test_validate_endpoint_config(self):
        """Test endpoint configuration validation"""
        endpoint_config = EndpointConfig(
            project_id="",
            location="",
            endpoint_id="1234567890123456789"
        )
        
        result = VertexAIOnlinePredictionTransformation.validate_endpoint_config(
            endpoint_config, "my-project", "us-central1"
        )
        
        self.assertEqual(result.project_id, "my-project")
        self.assertEqual(result.location, "us-central1")
        self.assertEqual(result.endpoint_id, "1234567890123456789")

    def test_validate_endpoint_config_missing_project(self):
        """Test validation with missing project ID"""
        endpoint_config = EndpointConfig(
            project_id="",
            location="us-central1",
            endpoint_id="1234567890123456789"
        )
        
        with self.assertRaises(ValueError):
            VertexAIOnlinePredictionTransformation.validate_endpoint_config(
                endpoint_config, None, "us-central1"
            )

    def test_create_prediction_url(self):
        """Test prediction URL creation"""
        url = VertexAIOnlinePredictionTransformation.create_prediction_url(
            project_id="my-project",
            location="us-central1",
            endpoint_id="1234567890123456789"
        )
        
        expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:predict"
        self.assertEqual(url, expected)

    def test_create_prediction_url_raw(self):
        """Test raw prediction URL creation"""
        url = VertexAIOnlinePredictionTransformation.create_prediction_url(
            project_id="my-project",
            location="us-central1",
            endpoint_id="1234567890123456789",
            use_raw_predict=True
        )
        
        expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:rawPredict"
        self.assertEqual(url, expected)

    def test_transform_optional_params_to_prediction_params(self):
        """Test transformation of optional parameters"""
        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stop": ["END", "STOP"]
        }
        
        result = VertexAIOnlinePredictionTransformation.transform_optional_params_to_prediction_params(
            optional_params
        )
        
        self.assertEqual(result.temperature, 0.7)
        self.assertEqual(result.max_tokens, 100)
        self.assertEqual(result.top_p, 0.9)
        self.assertEqual(result.stop_sequences, ["END", "STOP"])

    def test_transform_optional_params_to_prediction_params_stop_string(self):
        """Test transformation with string stop parameter"""
        optional_params = {
            "stop": "END"
        }
        
        result = VertexAIOnlinePredictionTransformation.transform_optional_params_to_prediction_params(
            optional_params
        )
        
        self.assertEqual(result.stop_sequences, ["END"])


if __name__ == "__main__":
    unittest.main() 