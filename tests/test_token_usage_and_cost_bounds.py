import pytest
from typing import Dict, Any

def test_token_usage_aggregation():
    responses = [
        {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
        {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
        {"prompt_tokens": 200, "completion_tokens": 90, "total_tokens": 290}
    ]
    
    total_prompt = sum(r["prompt_tokens"] for r in responses)
    total_completion = sum(r["completion_tokens"] for r in responses)
    total_combined = sum(r["total_tokens"] for r in responses)
    
    assert total_prompt == 400
    assert total_completion == 165
    assert total_combined == 565
    assert total_prompt + total_completion == total_combined

def test_cost_calculation_boundary_zero_tokens():
    input_cost_per_token = 0.0000015
    output_cost_per_token = 0.0000020
    
    prompt_tokens = 0
    completion_tokens = 0
    
    cost = (prompt_tokens * input_cost_per_token) + (completion_tokens * output_cost_per_token)
    assert cost == 0.0

def test_cost_calculation_precision():
    input_cost_per_million = 2.50
    output_cost_per_million = 10.00
    
    prompt_tokens = 100_000
    completion_tokens = 50_000
    
    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = round(input_cost + output_cost, 4)
    
    assert input_cost == 0.25
    assert output_cost == 0.50
    assert total_cost == 0.75