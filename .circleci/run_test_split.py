#!/usr/bin/env python3
"""
Script to run pytest based on test split configurations defined in test_splits.json.
Usage: python .circleci/run_test_split.py <job_name> <part_number> [additional pytest args...]
Example: python .circleci/run_test_split.py logging_testing 1 --junitxml=test-results/junit-part1.xml
"""

import json
import sys
import subprocess
from pathlib import Path


def load_test_splits():
    """Load test split configuration from JSON file."""
    config_path = Path(__file__).parent / "test_splits.json"
    with open(config_path, "r") as f:
        return json.load(f)


def build_pytest_command(job_config, extra_args):
    """Build pytest command from job configuration."""
    cmd = ["python", "-m", "pytest"]
    
    # Add test files/directories
    cmd.extend(job_config["test_files"])
    
    # Add standard pytest options
    cmd.extend([
        "--cov=litellm",
        "--cov-report=xml",
        "-s",
        "-v",
        f"--durations=10",
        f"-n={job_config['parallel_workers']}",
        f"--timeout={job_config['timeout']}",
        "-vv",
        "--log-cli-level=INFO",
        "--tb=short"
    ])
    
    # Add excluded files if specified
    if "exclude_files" in job_config:
        for exclude in job_config["exclude_files"]:
            cmd.extend(["--ignore", exclude])
    
    # Add any extra arguments passed from command line
    if extra_args:
        cmd.extend(extra_args)
    
    return cmd


def main():
    if len(sys.argv) < 3:
        print("Usage: python .circleci/run_test_split.py <job_name> <part_number> [additional pytest args...]")
        print("Example: python .circleci/run_test_split.py logging_testing 1 --junitxml=test-results/junit-part1.xml")
        sys.exit(1)
    
    job_name = sys.argv[1]
    part_number = sys.argv[2]
    extra_args = sys.argv[3:] if len(sys.argv) > 3 else []
    part_key = f"part{part_number}"
    
    # Load configuration
    test_splits = load_test_splits()
    
    if job_name not in test_splits:
        print(f"Error: Job '{job_name}' not found in test_splits.json")
        print(f"Available jobs: {', '.join(test_splits.keys())}")
        sys.exit(1)
    
    if part_key not in test_splits[job_name]:
        print(f"Error: Part '{part_key}' not found for job '{job_name}'")
        print(f"Available parts: {', '.join(test_splits[job_name].keys())}")
        sys.exit(1)
    
    job_config = test_splits[job_name][part_key]
    
    # Print job info
    print(f"Running {job_name} - {part_key}")
    print(f"Description: {job_config['description']}")
    print(f"Resource class: {job_config['resource_class']}")
    print(f"Parallel workers: {job_config['parallel_workers']}")
    print(f"Timeout: {job_config['timeout']}s")
    print()
    
    # Build and run pytest command
    cmd = build_pytest_command(job_config, extra_args)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run pytest
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

