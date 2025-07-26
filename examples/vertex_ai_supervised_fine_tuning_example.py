#!/usr/bin/env python3
"""
Vertex AI Supervised Fine-Tuning Example

This example demonstrates how to use LiteLLM's Vertex AI supervised fine-tuning functionality.

Prerequisites:
1. Google Cloud project with Vertex AI enabled
2. Service account with appropriate permissions
3. Training data uploaded to Google Cloud Storage
4. Environment variables set:
   - GOOGLE_APPLICATION_CREDENTIALS
   - VERTEX_AI_PROJECT
   - VERTEX_AI_LOCATION

Usage:
    python vertex_ai_supervised_fine_tuning_example.py
"""

import os
import time
from typing import Dict, Any

import litellm


def create_fine_tuning_job_example():
    """Example of creating a supervised fine-tuning job"""
    
    print("=== Creating Vertex AI Supervised Fine-Tuning Job ===")
    
    # Example 1: Basic fine-tuning with Gemini
    try:
        job = litellm.create_fine_tuning_job(
            model="gemini-1.0-pro",  # Base model to fine-tune
            training_file="gs://my-bucket/training-data.jsonl",
            validation_file="gs://my-bucket/validation-data.jsonl",
            hyperparameters={
                "epoch_count": 3,
                "learning_rate_multiplier": 1.0,
                "adapter_size": "medium"
            },
            suffix="my-custom-model",  # Suffix for the fine-tuned model
            custom_llm_provider="vertex_ai",
            vertex_project="my-project",
            vertex_location="us-central1"
        )
        
        print(f"✅ Fine-tuning job created successfully!")
        print(f"Job ID: {job.id}")
        print(f"Status: {job.status}")
        print(f"Model: {job.model}")
        print(f"Created at: {job.created_at}")
        
        return job.id
        
    except Exception as e:
        print(f"❌ Error creating fine-tuning job: {e}")
        return None


def monitor_fine_tuning_job(job_id: str):
    """Example of monitoring a fine-tuning job"""
    
    print(f"\n=== Monitoring Fine-Tuning Job {job_id} ===")
    
    try:
        # Get job status
        job = litellm.retrieve_fine_tuning_job(
            fine_tuning_job_id=job_id,
            custom_llm_provider="vertex_ai",
            vertex_project="my-project",
            vertex_location="us-central1"
        )
        
        print(f"Job Status: {job.status}")
        print(f"Progress: {getattr(job, 'progress', 'N/A')}")
        print(f"Fine-tuned model: {getattr(job, 'fine_tuned_model', 'N/A')}")
        print(f"Training file: {getattr(job, 'training_file', 'N/A')}")
        print(f"Validation file: {getattr(job, 'validation_file', 'N/A')}")
        
        # Check if job is complete
        if job.status in ["succeeded", "failed", "cancelled"]:
            print(f"Job completed with status: {job.status}")
            if job.status == "succeeded":
                print(f"Fine-tuned model available: {getattr(job, 'fine_tuned_model', 'N/A')}")
        else:
            print("Job is still running...")
            
    except Exception as e:
        print(f"❌ Error retrieving job status: {e}")


def list_fine_tuning_jobs_example():
    """Example of listing fine-tuning jobs"""
    
    print("\n=== Listing Fine-Tuning Jobs ===")
    
    try:
        jobs = litellm.list_fine_tuning_jobs(
            limit=10,
            custom_llm_provider="vertex_ai",
            vertex_project="my-project",
            vertex_location="us-central1"
        )
        
        print(f"Found {len(jobs.data)} fine-tuning jobs:")
        for job in jobs.data:
            print(f"  - ID: {job.id}")
            print(f"    Status: {job.status}")
            print(f"    Model: {job.model}")
            print(f"    Created: {job.created_at}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing jobs: {e}")


def cancel_fine_tuning_job_example(job_id: str):
    """Example of cancelling a fine-tuning job"""
    
    print(f"\n=== Cancelling Fine-Tuning Job {job_id} ===")
    
    try:
        job = litellm.cancel_fine_tuning_job(
            fine_tuning_job_id=job_id,
            custom_llm_provider="vertex_ai",
            vertex_project="my-project",
            vertex_location="us-central1"
        )
        
        print(f"✅ Job cancelled successfully!")
        print(f"Status: {job.status}")
        
    except Exception as e:
        print(f"❌ Error cancelling job: {e}")


def use_fine_tuned_model_example(fine_tuned_model: str):
    """Example of using a fine-tuned model for predictions"""
    
    print(f"\n=== Using Fine-Tuned Model {fine_tuned_model} ===")
    
    try:
        # Use the fine-tuned model for completion
        response = litellm.completion(
            model=fine_tuned_model,
            messages=[
                {"role": "user", "content": "Hello, how can you help me today?"}
            ],
            custom_llm_provider="vertex_ai",
            vertex_project="my-project",
            vertex_location="us-central1"
        )
        
        print(f"✅ Response from fine-tuned model:")
        print(f"Content: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error using fine-tuned model: {e}")


def create_training_data_example():
    """Example of creating training data in the correct format"""
    
    print("\n=== Training Data Format Example ===")
    
    # JSONL format (recommended)
    jsonl_example = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I make coffee?"},
                {"role": "assistant", "content": "To make coffee, you need coffee grounds, hot water, and a brewing method. Here's a simple process: 1. Boil water, 2. Add coffee grounds to a filter, 3. Pour hot water over the grounds, 4. Let it brew for 3-4 minutes."}
            ]
        }
    ]
    
    print("JSONL Format Example:")
    for item in jsonl_example:
        print(f"  {item}")
    
    # CSV format
    csv_example = [
        {"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."},
        {"prompt": "How do I make coffee?", "completion": "To make coffee, you need coffee grounds, hot water, and a brewing method..."}
    ]
    
    print("\nCSV Format Example:")
    for item in csv_example:
        print(f"  {item}")


def hyperparameter_tuning_example():
    """Example of different hyperparameter configurations"""
    
    print("\n=== Hyperparameter Tuning Examples ===")
    
    # Conservative settings for small datasets
    conservative_params = {
        "epoch_count": 2,
        "learning_rate_multiplier": 0.5,
        "adapter_size": "small",
        "warmup_steps": 100,
        "weight_decay": 0.01
    }
    
    # Aggressive settings for large datasets
    aggressive_params = {
        "epoch_count": 5,
        "learning_rate_multiplier": 2.0,
        "adapter_size": "large",
        "warmup_steps": 500,
        "weight_decay": 0.05
    }
    
    # Balanced settings (default)
    balanced_params = {
        "epoch_count": 3,
        "learning_rate_multiplier": 1.0,
        "adapter_size": "medium",
        "warmup_steps": 0,
        "weight_decay": 0.01
    }
    
    print("Conservative Settings (small datasets):")
    print(f"  {conservative_params}")
    
    print("\nAggressive Settings (large datasets):")
    print(f"  {aggressive_params}")
    
    print("\nBalanced Settings (default):")
    print(f"  {balanced_params}")


def main():
    """Main function demonstrating the complete workflow"""
    
    print("🚀 Vertex AI Supervised Fine-Tuning with LiteLLM")
    print("=" * 50)
    
    # Check environment variables
    required_env_vars = [
        "GOOGLE_APPLICATION_CREDENTIALS",
        "VERTEX_AI_PROJECT", 
        "VERTEX_AI_LOCATION"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these environment variables before running the example.")
        return
    
    # Show training data format
    create_training_data_example()
    
    # Show hyperparameter examples
    hyperparameter_tuning_example()
    
    # List existing jobs
    list_fine_tuning_jobs_example()
    
    # Create a new fine-tuning job
    job_id = create_fine_tuning_job_example()
    
    if job_id:
        # Monitor the job
        monitor_fine_tuning_job(job_id)
        
        # Example: Cancel job (uncomment to test)
        # cancel_fine_tuning_job_example(job_id)
        
        # Example: Use fine-tuned model (uncomment when job completes)
        # use_fine_tuned_model_example("projects/my-project/locations/us-central1/models/fine-tuned-model")
    
    print("\n✅ Example completed!")


if __name__ == "__main__":
    main() 