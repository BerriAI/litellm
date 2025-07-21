"""
Minimal test for Vertex AI Online Prediction core logic
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointConfig:
    """Configuration for Vertex AI endpoint"""
    
    project_id: str
    location: str
    endpoint_id: str
    model_name: Optional[str] = None


class VertexAIOnlinePredictionTransformation:
    """Transformation logic for Vertex AI online prediction"""

    @staticmethod
    def parse_endpoint_from_model(model: str) -> EndpointConfig:
        """
        Parse endpoint configuration from model string
        
        Expected format: vertex_ai/endpoints/{endpoint_id}
        or: vertex_ai/{project_id}/{location}/endpoints/{endpoint_id}
        """
        if not model.startswith("vertex_ai/"):
            raise ValueError(f"Invalid model format: {model}")
        
        parts = model.split("/")
        
        if len(parts) == 3 and parts[1] == "endpoints":
            # Format: vertex_ai/endpoints/{endpoint_id}
            return EndpointConfig(
                project_id="",  # Will be set from environment or params
                location="",    # Will be set from environment or params
                endpoint_id=parts[2]
            )
        elif len(parts) == 5 and parts[3] == "endpoints":
            # Format: vertex_ai/{project_id}/{location}/endpoints/{endpoint_id}
            return EndpointConfig(
                project_id=parts[1],
                location=parts[2],
                endpoint_id=parts[4]
            )
        else:
            raise ValueError(f"Invalid model format: {model}")

    @staticmethod
    def validate_endpoint_config(
        endpoint_config: EndpointConfig,
        vertex_project: Optional[str],
        vertex_location: Optional[str]
    ) -> EndpointConfig:
        """
        Validate and fill in missing endpoint configuration
        """
        if not endpoint_config.project_id:
            if not vertex_project:
                raise ValueError("Project ID is required for Vertex AI online prediction")
            endpoint_config.project_id = vertex_project
        
        if not endpoint_config.location:
            if not vertex_location:
                endpoint_config.location = "us-central1"  # Default location
            else:
                endpoint_config.location = vertex_location
        
        return endpoint_config

    @staticmethod
    def create_prediction_url(
        project_id: str,
        location: str,
        endpoint_id: str,
        use_raw_predict: bool = False
    ) -> str:
        """
        Create the prediction URL for Vertex AI endpoint
        """
        base_url = f"https://{location}-aiplatform.googleapis.com"
        endpoint_path = f"v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        
        if use_raw_predict:
            return f"{base_url}/{endpoint_path}:rawPredict"
        else:
            return f"{base_url}/{endpoint_path}:predict"


def test_parse_endpoint_from_model_simple():
    """Test parsing endpoint from simple model format"""
    model = "vertex_ai/endpoints/1234567890123456789"
    result = VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
    
    assert result.endpoint_id == "1234567890123456789"
    assert result.project_id == ""
    assert result.location == ""


def test_parse_endpoint_from_model_full():
    """Test parsing endpoint from full model format"""
    model = "vertex_ai/my-project/us-central1/endpoints/1234567890123456789"
    result = VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
    
    assert result.endpoint_id == "1234567890123456789"
    assert result.project_id == "my-project"
    assert result.location == "us-central1"


def test_parse_endpoint_from_model_invalid():
    """Test parsing invalid model format"""
    model = "invalid/model/format"
    
    try:
        VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_validate_endpoint_config():
    """Test endpoint configuration validation"""
    endpoint_config = EndpointConfig(
        project_id="",
        location="",
        endpoint_id="1234567890123456789"
    )
    
    result = VertexAIOnlinePredictionTransformation.validate_endpoint_config(
        endpoint_config, "my-project", "us-central1"
    )
    
    assert result.project_id == "my-project"
    assert result.location == "us-central1"
    assert result.endpoint_id == "1234567890123456789"


def test_create_prediction_url():
    """Test prediction URL creation"""
    url = VertexAIOnlinePredictionTransformation.create_prediction_url(
        project_id="my-project",
        location="us-central1",
        endpoint_id="1234567890123456789"
    )
    
    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:predict"
    assert url == expected


def test_create_prediction_url_raw():
    """Test raw prediction URL creation"""
    url = VertexAIOnlinePredictionTransformation.create_prediction_url(
        project_id="my-project",
        location="us-central1",
        endpoint_id="1234567890123456789",
        use_raw_predict=True
    )
    
    expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/1234567890123456789:rawPredict"
    assert url == expected


def run_all_tests():
    """Run all tests"""
    test_parse_endpoint_from_model_simple()
    test_parse_endpoint_from_model_full()
    test_parse_endpoint_from_model_invalid()
    test_validate_endpoint_config()
    test_create_prediction_url()
    test_create_prediction_url_raw()


if __name__ == "__main__":
    run_all_tests() 