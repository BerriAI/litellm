"""
SpanAttributes Value Usage Checker

This script ensures that all SpanAttributes enum references in the OpenTelemetry integration
are properly accessed with the .value property. This is important because:

1. Without .value, the enum object itself is used instead of its string value
2. This can cause type errors or unexpected behavior in OpenTelemetry exporters
3. It's a consistent pattern that should be followed for all enum usage

Example of correct usage:
    span.set_attribute(key=SpanAttributes.LLM_USER.value, value="user123")

Example of incorrect usage:
    span.set_attribute(key=SpanAttributes.LLM_USER, value="user123")

The script checks both through AST parsing (for accurate code analysis) and regex
(for backup coverage) to find any violations.

Usage:
    python tests/code_coverage_tests/check_spanattributes_value_usage.py
    python tests/code_coverage_tests/check_spanattributes_value_usage.py --debug
"""

import argparse
import ast
import os
import re
from typing import List, Tuple
import sys

# Add parent directory to path so we can import litellm
sys.path.insert(0, os.path.abspath("../.."))
import litellm


class SpanAttributesUsageChecker(ast.NodeVisitor):
    """
    Checks if SpanAttributes is used without .value when setting attributes in safe_set_attribute calls
    and other attribute setting methods in opentelemetry.py.
    
    This is important to ensure consistent enum value access and prevent type errors
    when sending data to OpenTelemetry exporters.
    """
    def __init__(self, debug=False):
        self.violations = []
        self.debug = debug
        
    def visit_Call(self, node):
        # Check if this is a call to safe_set_attribute or set_attribute
        if isinstance(node.func, ast.Attribute) and node.func.attr in ['safe_set_attribute', 'set_attribute']:
            # Look for the 'key' parameter
            for keyword in node.keywords:
                if keyword.arg == 'key':
                    # Check if the value is a SpanAttributes member without .value
                    if isinstance(keyword.value, ast.Attribute) and \
                       isinstance(keyword.value.value, ast.Name) and \
                       keyword.value.value.id == 'SpanAttributes':
                        
                        # Get the source code for this attribute
                        try:
                            attr_source = ast.unparse(keyword.value)
                            if not attr_source.endswith('.value'):
                                if self.debug:
                                    print(f"AST found violation: {node.lineno}: {attr_source}")
                                self.violations.append((node.lineno, f"{attr_source} used without .value"))
                        except AttributeError:
                            # For Python < 3.9, ast.unparse doesn't exist
                            # Fallback to our best guess
                            if keyword.value.attr != 'value' and not hasattr(keyword.value, 'value'):
                                violation_msg = f"SpanAttributes.{keyword.value.attr} used without .value"
                                if self.debug:
                                    print(f"AST found violation: {node.lineno}: {violation_msg}")
                                self.violations.append((node.lineno, violation_msg))
        # Continue the visit
        self.generic_visit(node)

def check_file(file_path: str, debug: bool = False) -> List[Tuple[int, str]]:
    """
    Analyze a Python file to check for SpanAttributes usage without .value
    
    Args:
        file_path: Path to the Python file to check
        debug: Whether to print debug information
        
    Returns:
        List of (line_number, message) tuples identifying violations
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # First try AST parsing for accurate code structure analysis
    try:
        tree = ast.parse(content)
        checker = SpanAttributesUsageChecker(debug=debug)
        checker.visit(tree)
        violations = checker.violations
        
        # Also do a regex check for backup/extra coverage
        # This catches cases that might be missed by AST parsing
        
        # Split content into lines for more precise analysis
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Skip lines that contain ".value" after "SpanAttributes."
            # This prevents false positives for correct usage
            if re.search(r"SpanAttributes\.[A-Z_][A-Z0-9_]*\.value", line):
                if debug:
                    print(f"Line {i} skipped - contains .value: {line.strip()}")
                continue
            
            # Pattern: Looking for "key=SpanAttributes.ENUM_NAME" without .value at the end
            pattern = r"key\s*=\s*SpanAttributes\.[A-Z_][A-Z0-9_]*(?!\.value)"
            match = re.search(pattern, line)
            
            if match:
                # Check if this violation was already found by AST
                if not any(i == line_num for line_num, _ in violations):
                    if debug:
                        print(f"Regex found violation: {i}: {match.group(0)}")
                    violations.append((i, f"SpanAttributes used without .value: {match.group(0)}"))
        
        return violations
    
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return []

def main():
    """
    Main function to run the SpanAttributes usage check on the OpenTelemetry integration file.
    
    Exits with code 1 if violations are found, 0 otherwise.
    """
    parser = argparse.ArgumentParser(description='Check for SpanAttributes used without .value')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Path to the OpenTelemetry integration file
    target_file = os.path.join("litellm", "integrations", "opentelemetry.py")
    
    if not os.path.exists(target_file):
        # Try alternate path for local development
        target_file = os.path.join("..", "..", "litellm", "integrations", "opentelemetry.py")
    
    if not os.path.exists(target_file):
        print(f"Error: Could not find file at {target_file}")
        exit(1)
    
    violations = check_file(target_file, debug=args.debug)
    
    if violations:
        print(f"Found {len(violations)} SpanAttributes without .value in {target_file}:")
        
        # Sort violations by line number for better readability
        violations.sort(key=lambda x: x[0])
        
        for line, message in violations:
            print(f"  Line {line}: {message}")
        print("\nDirect enum reference can cause errors. Always use .value with SpanAttributes enums.")
        exit(1)
    else:
        print(f"All SpanAttributes are used correctly with .value in {target_file}")
        exit(0)

if __name__ == "__main__":
    main()