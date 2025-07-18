"""
Simple standalone test for Vertex AI Online Prediction transformation logic
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules directly
import types
import transformation


def test_parse_endpoint_from_model_simple():
    """Test parsing endpoint from simple model format"""
    model = "vertex_ai/endpoints/1234567890123456789"
    result = transformation.VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
    
    assert result.endpoint_id == "1234567890123456789"
    assert result.project_id == ""
    assert result.location == ""
    print("✓ test_parse_endpoint_from_model_simple passed")


def test_parse_endpoint_from_model_full():
    """Test parsing endpoint from full model format"""
    model = "vertex_ai/my-project/us-central1/endpoints/1234567890123456789"
    result = transformation.VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
    
    assert result.endpoint_id == "1234567890123456789"
    assert result.project_id == "my-project"
    assert result.location == "us-central1"
    print("✓ test_parse_endpoint_from_model_full passed")


def test_parse_endpoint_from_model_invalid():
    """Test parsing invalid model format"""
    model = "invalid/model/format"
    
    try:
        transformation.VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ test_parse_endpoint_from_model_invalid passed")


def test_validate_endpoint_config():
    """Test endpoint configuration validation"""
    endpoint_config = types.EndpointConfig(
        project_id="",
        location="",
        endpoint_id="1234567890123456789"
    )
    
    result = transformation.VertexAIOnlinePredictionTransformation.validate_endpoint_config(
        endpoint_config, "my-project", "us-central1"
    )
    
    assert result.project_id == "my-project"
    assert result.location == "us-central1"
    assert result.endpoint_id == "1234567890123456789"
    print("✓ test_validate_endpoint_config passed")


def test_create_prediction_url():
    """Test prediction URL creation"""
    url = transformation.VertexAIOnlinePredictionTransformation.create_prediction_url(
        project_id="my-project",
        location="us-central1",
        endpoint_id="1234567890123456789"
    )
    
    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:predict"
    assert url == expected
    print("✓ test_create_prediction_url passed")


def test_create_prediction_url_raw():
    """Test raw prediction URL creation"""
    url = transformation.VertexAIOnlinePredictionTransformation.create_prediction_url(
        project_id="my-project",
        location="us-central1",
        endpoint_id="1234567890123456789",
        use_raw_predict=True
    )
    
    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:rawPredict"
    assert url == expected
    print("✓ test_create_prediction_url_raw passed")


def run_all_tests():
    """Run all tests"""
    print("Running Vertex AI Online Prediction tests...")
    print("=" * 50)
    
    test_parse_endpoint_from_model_simple()
    test_parse_endpoint_from_model_full()
    test_parse_endpoint_from_model_invalid()
    test_validate_endpoint_config()
    test_create_prediction_url()
    test_create_prediction_url_raw()
    
    print("=" * 50)
    print("All tests passed! 🎉")


if __name__ == "__main__":
    run_all_tests() 