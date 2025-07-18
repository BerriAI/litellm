"""
Transformation logic for Vertex AI Supervised Fine-Tuning
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from .types import (
    FineTuningHyperparameters,
    FineTuningJobCreate,
    FineTuningJobStatus,
    DatasetValidationResult,
    FineTuningCostEstimate,
)


class VertexAIFineTuningTransformation:
    """Transformation logic for Vertex AI supervised fine-tuning"""

    @staticmethod
    def validate_hyperparameters(hyperparameters: Dict[str, Any]) -> FineTuningHyperparameters:
        """
        Validate and transform hyperparameters
        """
        # Set defaults for missing parameters
        defaults = {
            "epoch_count": 3,
            "learning_rate_multiplier": 1.0,
            "adapter_size": "medium",
            "warmup_steps": 0,
            "weight_decay": 0.01
        }
        
        # Merge with provided hyperparameters
        merged_params = {**defaults, **hyperparameters}
        
        # Validate ranges
        if not (1 <= merged_params["epoch_count"] <= 10):
            raise ValueError("epoch_count must be between 1 and 10")
        
        if not (0.1 <= merged_params["learning_rate_multiplier"] <= 10.0):
            raise ValueError("learning_rate_multiplier must be between 0.1 and 10.0")
        
        if merged_params["adapter_size"] not in ["small", "medium", "large"]:
            raise ValueError("adapter_size must be 'small', 'medium', or 'large'")
        
        if not (0 <= merged_params["warmup_steps"] <= 1000):
            raise ValueError("warmup_steps must be between 0 and 1000")
        
        if not (0.0 <= merged_params["weight_decay"] <= 0.1):
            raise ValueError("weight_decay must be between 0.0 and 0.1")
        
        return FineTuningHyperparameters(**merged_params)

    @staticmethod
    def validate_dataset_format(file_uri: str) -> DatasetValidationResult:
        """
        Validate dataset format and structure
        """
        errors = []
        warnings = []
        
        # Check file extension
        if not file_uri.endswith(('.jsonl', '.json', '.csv')):
            errors.append("Dataset file must have .jsonl, .json, or .csv extension")
        
        # Check if it's a GCS URI
        if not file_uri.startswith('gs://'):
            warnings.append("File URI should be a Google Cloud Storage URI (gs://)")
        
        # Additional validation would be done by reading the file
        # This is a basic validation - in practice, you'd want to read and validate the content
        
        return DatasetValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            format=file_uri.split('.')[-1] if '.' in file_uri else None
        )

    @staticmethod
    def create_fine_tuning_request(
        job_create: FineTuningJobCreate
    ) -> Dict[str, Any]:
        """
        Create Vertex AI fine-tuning request
        """
        # Validate hyperparameters
        hyperparameters = None
        if job_create.hyperparameters:
            hyperparameters = job_create.hyperparameters.dict(exclude_none=True)
        
        # Create the request
        request = {
            "baseModel": job_create.model,
            "supervisedTuningSpec": {
                "trainingDatasetUri": job_create.training_file
            }
        }
        
        # Add validation dataset if provided
        if job_create.validation_file:
            request["supervisedTuningSpec"]["validationDataset"] = job_create.validation_file
        
        # Add hyperparameters if provided
        if hyperparameters:
            request["supervisedTuningSpec"]["hyperParameters"] = hyperparameters
        
        # Add display name if suffix provided
        if job_create.suffix:
            request["tunedModelDisplayName"] = job_create.suffix
        
        return request

    @staticmethod
    def transform_vertex_response_to_job_status(
        response: Dict[str, Any],
        job_id: str
    ) -> FineTuningJobStatus:
        """
        Transform Vertex AI response to job status
        """
        # Extract supervised tuning spec
        supervised_tuning_spec = response.get("supervisedTuningSpec", {})
        
        # Extract hyperparameters
        hyperparameters = None
        if "hyperParameters" in supervised_tuning_spec:
            hyperparameters = FineTuningHyperparameters(**supervised_tuning_spec["hyperParameters"])
        
        # Convert timestamps
        created_at = None
        if "createTime" in response:
            created_at = VertexAIFineTuningTransformation._parse_timestamp(response["createTime"])
        
        finished_at = None
        if "endTime" in response:
            finished_at = VertexAIFineTuningTransformation._parse_timestamp(response["endTime"])
        
        # Extract error if any
        error = None
        if response.get("state") == "JOB_STATE_FAILED":
            error = response.get("error", {}).get("message", "Unknown error")
        
        return FineTuningJobStatus(
            id=job_id,
            status=response.get("state", "JOB_STATE_UNKNOWN"),
            created_at=created_at,
            finished_at=finished_at,
            fine_tuned_model=response.get("tunedModelDisplayName"),
            training_file=supervised_tuning_spec.get("trainingDatasetUri", ""),
            validation_file=supervised_tuning_spec.get("validationDataset"),
            hyperparameters=hyperparameters,
            error=error
        )

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> Optional[int]:
        """
        Parse ISO timestamp to Unix timestamp
        """
        try:
            from datetime import datetime
            # Remove 'Z' and parse
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return int(dt.timestamp())
        except Exception:
            return None

    @staticmethod
    def create_fine_tuning_url(
        project_id: str,
        location: str
    ) -> str:
        """
        Create the fine-tuning URL
        """
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/tuningJobs"

    @staticmethod
    def create_job_status_url(
        project_id: str,
        location: str,
        job_id: str
    ) -> str:
        """
        Create the job status URL
        """
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/tuningJobs/{job_id}"

    @staticmethod
    def estimate_fine_tuning_cost(
        model: str,
        training_file_size_mb: float,
        hyperparameters: FineTuningHyperparameters
    ) -> FineTuningCostEstimate:
        """
        Estimate the cost of fine-tuning
        """
        # This is a simplified cost estimation
        # In practice, you'd want to use more sophisticated cost models
        
        # Base costs per hour (approximate)
        base_costs = {
            "gemini-1.0-pro": 2.0,
            "gemini-2.0-flash": 1.5,
            "gemini-2.5-pro": 3.0,
            "claude-3-opus": 4.0,
            "claude-3-sonnet": 2.5,
            "claude-3-haiku": 1.0,
        }
        
        # Get base cost for model
        base_cost_per_hour = base_costs.get(model, 2.0)
        
        # Estimate duration based on dataset size and hyperparameters
        estimated_hours = max(1, training_file_size_mb / 100) * hyperparameters.epoch_count
        
        # Calculate total cost
        total_cost = base_cost_per_hour * estimated_hours
        
        # Cost breakdown
        cost_breakdown = {
            "compute": total_cost * 0.8,
            "storage": total_cost * 0.1,
            "network": total_cost * 0.1
        }
        
        return FineTuningCostEstimate(
            estimated_cost_usd=total_cost,
            estimated_duration_hours=estimated_hours,
            cost_breakdown=cost_breakdown,
            factors={
                "model": model,
                "dataset_size_mb": training_file_size_mb,
                "epochs": hyperparameters.epoch_count,
                "adapter_size": hyperparameters.adapter_size
            }
        )

    @staticmethod
    def validate_model_supports_fine_tuning(model: str) -> bool:
        """
        Check if a model supports fine-tuning
        """
        # Models that support fine-tuning
        supported_models = [
            "gemini-1.0-pro",
            "gemini-2.0-flash", 
            "gemini-2.5-pro",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "meta-llama/Llama-2-7b-chat",
            "meta-llama/Llama-2-13b-chat",
            "mistral-7b-instruct",
            "mistral-large"
        ]
        
        # Check exact match
        if model in supported_models:
            return True
        
        # Check pattern matches (for custom models)
        patterns = [
            r"^gemini-\d+\.\d+.*$",
            r"^claude-\d+.*$",
            r"^meta-llama/.*$",
            r"^mistral-.*$"
        ]
        
        for pattern in patterns:
            if re.match(pattern, model):
                return True
        
        return False

    @staticmethod
    def extract_job_id_from_response(response: Dict[str, Any]) -> str:
        """
        Extract job ID from Vertex AI response
        """
        # The job ID is typically in the 'name' field
        name = response.get("name", "")
        
        # Extract the job ID from the name
        # Format: projects/{project}/locations/{location}/tuningJobs/{job_id}
        parts = name.split("/")
        if len(parts) >= 6:
            return parts[-1]
        
        # Fallback to using the name as the ID
        return name

    @staticmethod
    def transform_job_list_response(
        response: Dict[str, Any]
    ) -> List[FineTuningJobStatus]:
        """
        Transform list response to job status objects
        """
        jobs = []
        
        for job_data in response.get("tuningJobs", []):
            job_id = VertexAIFineTuningTransformation.extract_job_id_from_response(job_data)
            job_status = VertexAIFineTuningTransformation.transform_vertex_response_to_job_status(
                job_data, job_id
            )
            jobs.append(job_status)
        
        return jobs 