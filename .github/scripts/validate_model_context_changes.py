#!/usr/bin/env python3
"""
Pre-commit hook to validate that changes to model context JSON files include cited sources.
"""

import json
import sys
import os
import re
from typing import List, Dict, Any

def check_for_citations(commit_message: str, pr_description: str = "") -> bool:
    """
    Check if the commit message or PR description contains source citations.
    
    Args:
        commit_message: The git commit message
        pr_description: Optional PR description (if available)
    
    Returns:
        bool: True if citations are found, False otherwise
    """
    text_to_check = f"{commit_message} {pr_description}".lower()
    
    # Common citation patterns
    citation_patterns = [
        r'https?://[^\s]+',  # URLs
        r'source[s]?:',      # "Source:" or "Sources:"
        r'reference[s]?:',   # "Reference:" or "References:"
        r'citation[s]?:',    # "Citation:" or "Citations:"
        r'api\s+documentation',
        r'endpoint[s]?:',
        r'documentation:',
        r'from\s+https?://',
        r'based\s+on\s+https?://',
        r'according\s+to\s+https?://',
        r'per\s+https?://',
        r'via\s+https?://',
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, text_to_check):
            return True
    
    return False

def validate_json_structure(file_path: str) -> bool:
    """
    Validate that the JSON file has proper structure.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        bool: True if JSON is valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ JSON validation failed for {file_path}: {e}")
        return False

def check_model_context_files(staged_files: List[str]) -> List[str]:
    """
    Check if any staged files are model context files.
    
    Args:
        staged_files: List of staged file paths
    
    Returns:
        List of model context files that are staged
    """
    model_context_patterns = [
        'model_prices_and_context_window.json',
        '**/model_context*.json',
        '**/context_window*.json'
    ]
    
    model_context_files = []
    for file_path in staged_files:
        if any(pattern.replace('**/', '').replace('*', '') in file_path for pattern in model_context_patterns):
            model_context_files.append(file_path)
    
    return model_context_files

def main():
    """Main function to validate model context changes."""
    # Get staged files
    try:
        import subprocess
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                              capture_output=True, text=True, check=True)
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        print("❌ Failed to get staged files")
        sys.exit(1)
    
    # Check for model context files
    model_context_files = check_model_context_files(staged_files)
    
    if not model_context_files:
        print("✅ No model context files modified")
        sys.exit(0)
    
    print(f"🔍 Found {len(model_context_files)} model context file(s) in staging area:")
    for file in model_context_files:
        print(f"  - {file}")
    
    # Validate JSON structure
    json_valid = True
    for file_path in model_context_files:
        if not validate_json_structure(file_path):
            json_valid = False
    
    if not json_valid:
        print("❌ JSON validation failed")
        sys.exit(1)
    
    # Check for citations in commit message
    try:
        commit_message = os.environ.get('COMMIT_MSG', '')
        if not commit_message:
            # Try to get the commit message from git
            result = subprocess.run(['git', 'log', '-1', '--pretty=%B'], 
                                  capture_output=True, text=True, check=True)
            commit_message = result.stdout.strip()
    except subprocess.CalledProcessError:
        commit_message = ''
    
    # Check for citations
    if check_for_citations(commit_message):
        print("✅ Source citations found in commit message")
        sys.exit(0)
    else:
        print("❌ No source citations found in commit message")
        print("\n📝 Please include source citations in your commit message or PR description.")
        print("Examples:")
        print("  - Source: https://openrouter.ai/api/v1/models")
        print("  - Reference: API documentation at https://example.com/docs")
        print("  - Based on: https://provider.com/pricing")
        print("\nThis helps maintain data integrity and traceability.")
        sys.exit(1)

if __name__ == "__main__":
    main()