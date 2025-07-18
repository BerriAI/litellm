"""
Type definitions for Vertex AI Supervised Fine-Tuning
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class FineTuningHyperparameters(BaseModel):
    """Hyperparameters for supervised fine-tuning"""
    
    epoch_count: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of training epochs"
    )
    learning_rate_multiplier: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Learning rate multiplier"
    )
    adapter_size: Optional[Literal["small", "medium", "large"]] = Field(
        default="medium",
        description="Size of the adapter"
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=64,
        description="Training batch size"
    )
    warmup_steps: Optional[int] = Field(
        default=0,
        ge=0,
        le=1000,
        description="Number of warmup steps"
    )
    weight_decay: Optional[float] = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Weight decay coefficient"
    )


class FineTuningJobCreate(BaseModel):
    """Request to create a fine-tuning job"""
    
    model: str = Field(description="Base model to fine-tune")
    training_file: str = Field(description="URI of training data file")
    validation_file: Optional[str] = Field(
        default=None,
        description="URI of validation data file"
    )
    hyperparameters: Optional[FineTuningHyperparameters] = Field(
        default=None,
        description="Fine-tuning hyperparameters"
    )
    suffix: Optional[str] = Field(
        default=None,
        description="Suffix for the fine-tuned model name"
    )
    vertex_project: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID"
    )
    vertex_location: Optional[str] = Field(
        default="us-central1",
        description="Vertex AI location"
    )


class FineTuningJobStatus(BaseModel):
    """Status of a fine-tuning job"""
    
    id: str = Field(description="Fine-tuning job ID")
    status: Literal[
        "JOB_STATE_QUEUED",
        "JOB_STATE_PENDING", 
        "JOB_STATE_RUNNING",
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLING",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_ERROR"
    ] = Field(description="Current job status")
    created_at: Optional[int] = Field(
        default=None,
        description="Job creation timestamp"
    )
    finished_at: Optional[int] = Field(
        default=None,
        description="Job completion timestamp"
    )
    fine_tuned_model: Optional[str] = Field(
        default=None,
        description="Name of the fine-tuned model"
    )
    training_file: str = Field(description="Training file used")
    validation_file: Optional[str] = Field(
        default=None,
        description="Validation file used"
    )
    hyperparameters: Optional[FineTuningHyperparameters] = Field(
        default=None,
        description="Hyperparameters used"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Training progress (0.0 to 1.0)"
    )
    trained_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens trained"
    )
    estimated_finish: Optional[int] = Field(
        default=None,
        description="Estimated completion timestamp"
    )


class DatasetValidationResult(BaseModel):
    """Result of dataset validation"""
    
    is_valid: bool = Field(description="Whether the dataset is valid")
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )
    row_count: Optional[int] = Field(
        default=None,
        description="Number of rows in the dataset"
    )
    format: Optional[str] = Field(
        default=None,
        description="Detected dataset format"
    )


class FineTuningCostEstimate(BaseModel):
    """Cost estimate for fine-tuning"""
    
    estimated_cost_usd: float = Field(description="Estimated cost in USD")
    estimated_duration_hours: float = Field(description="Estimated duration in hours")
    cost_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of costs by component"
    )
    factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Factors affecting the cost estimate"
    )


class FineTuningJobMetrics(BaseModel):
    """Training metrics for a fine-tuning job"""
    
    job_id: str = Field(description="Fine-tuning job ID")
    training_loss: Optional[float] = Field(
        default=None,
        description="Training loss"
    )
    validation_loss: Optional[float] = Field(
        default=None,
        description="Validation loss"
    )
    learning_rate: Optional[float] = Field(
        default=None,
        description="Current learning rate"
    )
    epoch: Optional[int] = Field(
        default=None,
        description="Current epoch"
    )
    step: Optional[int] = Field(
        default=None,
        description="Current training step"
    )
    timestamp: Optional[int] = Field(
        default=None,
        description="Timestamp of the metrics"
    )


class FineTuningJobList(BaseModel):
    """List of fine-tuning jobs"""
    
    jobs: List[FineTuningJobStatus] = Field(
        default_factory=list,
        description="List of fine-tuning jobs"
    )
    next_page_token: Optional[str] = Field(
        default=None,
        description="Token for pagination"
    )


# Type aliases for convenience
Hyperparameters = FineTuningHyperparameters
JobCreate = FineTuningJobCreate
JobStatus = FineTuningJobStatus
ValidationResult = DatasetValidationResult
CostEstimate = FineTuningCostEstimate
JobMetrics = FineTuningJobMetrics
JobList = FineTuningJobList 