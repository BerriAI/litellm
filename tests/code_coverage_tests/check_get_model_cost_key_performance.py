"""
Code quality check: Ensure _get_model_cost_key only uses O(1) operations.

Simple pattern-based check for O(n) operations in _get_model_cost_key.
"""

import re
import os


def _function_has_on_operations(all_lines, func_name, visited=None):
    """
    Check if a function contains O(n) operations by searching for it in the file.
    Recursively checks called functions as well.
    """
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if func_name in visited:
        return False
    visited.add(func_name)
    
    func_start = None
    func_end = None
    
    for i, line in enumerate(all_lines):
        if func_start is None and f'def {func_name}(' in line:
            func_start = i
        elif func_start is not None:
            # Function ends when we hit next def at module level
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.startswith('def '):
                func_end = i
                break
    
    if func_start is None or func_end is None:
        return False
    
    # Check function body for O(n) patterns
    func_lines = all_lines[func_start:func_end]
    
    for line in func_lines:
        # Skip comments and docstrings
        line_stripped = line.strip()
        if line_stripped.startswith('#') or line_stripped.startswith('"""') or line_stripped.startswith("'''"):
            continue
        
        # Check for for loops
        if re.search(r'\bfor\s+\w+\s+in\s+', line):
            return True
        # Check for while loops
        if re.search(r'\bwhile\s+', line):
            return True
        # Check for comprehensions
        if re.search(r'\[.*\s+for\s+.*\s+in\s+', line) or re.search(r'\{.*\s+for\s+.*\s+in\s+', line):
            return True
        
        # Recursively check called functions (check all, don't skip any in recursive checks)
        func_call_match = re.search(r'\b([a-z_][a-z0-9_]*)\s*\(', line)
        if func_call_match:
            called_func = func_call_match.group(1)
            if called_func.startswith('_'):
                if _function_has_on_operations(all_lines, called_func, visited):
                    return True
    
    return False


def check_get_model_cost_key_performance():
    """
    Check that _get_model_cost_key doesn't contain O(n) operations.
    """
    utils_file = "./litellm/utils.py"
    
    if not os.path.exists(utils_file):
        print(f"Warning: File {utils_file} does not exist.")
        return []
    
    with open(utils_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find the _get_model_cost_key function
    func_start = None
    func_end = None
    
    for i, line in enumerate(lines):
        if func_start is None and 'def _get_model_cost_key(' in line:
            func_start = i
        elif func_start is not None:
            # Function ends when we hit next def at module level (no indentation)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and line.startswith('def '):
                func_end = i
                break
    
    if func_start is None:
        print("Warning: Could not find _get_model_cost_key function")
        return []
    
    if func_end is None:
        func_end = len(lines)
    
    # Extract function body
    func_lines = lines[func_start:func_end]
    problematic_lines = []
    
    # Track if we're inside a docstring
    in_docstring = False
    docstring_quote = None
    
    # Check for O(n) patterns
    for i, line in enumerate(func_lines, start=func_start + 1):
        line_stripped = line.strip()
        
        # Track docstring state (handle both single-line and multi-line docstrings)
        if not in_docstring:
            if line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                docstring_quote = '"""' if line_stripped.startswith('"""') else "'''"
                # Check if it's a single-line docstring
                if line_stripped.count(docstring_quote) >= 2:
                    in_docstring = False  # Single-line, skip this line
                    continue
                else:
                    in_docstring = True
                    continue
        else:
            # Inside multi-line docstring, check for closing quote
            if docstring_quote is not None and docstring_quote in line:
                in_docstring = False
                docstring_quote = None
            continue  # Skip all lines inside docstring
        
        # Skip comments
        if line_stripped.startswith('#'):
            continue
        
        # Check for for loops
        if re.search(r'\bfor\s+\w+\s+in\s+', line):
            # Allow helper function calls (they're conditional)
            if not re.search(r'(_rebuild_model_cost_lowercase_map|_handle_stale_map_entry_rebuild|_handle_new_key_with_scan)', line):
                problematic_lines.append((i, "for loop", line_stripped))
        
        # Check for while loops
        if re.search(r'\bwhile\s+', line):
            problematic_lines.append((i, "while loop", line_stripped))
        
        # Check for comprehensions
        if re.search(r'\[.*\s+for\s+.*\s+in\s+', line) or re.search(r'\{.*\s+for\s+.*\s+in\s+', line):
            problematic_lines.append((i, "comprehension", line_stripped))
        
        # Check for problematic function calls
        problematic_funcs = ['enumerate', 'zip', 'map', 'filter', 'sorted', 'any', 'all', 'sum', 'max', 'min']
        for func in problematic_funcs:
            if re.search(rf'\b{func}\s*\(', line):
                problematic_lines.append((i, f"call to {func}()", line_stripped))
        
        # Check for calls to functions that might have O(n) operations
        # Allow known helper functions that are conditional
        allowed_helpers = [
            '_rebuild_model_cost_lowercase_map',
            '_handle_stale_map_entry_rebuild',
            '_handle_new_key_with_scan',
        ]
        
        # Check for function calls (pattern: function_name(...), but not function definitions)
        # Skip function definitions (def function_name(...))
        if not re.search(r'\bdef\s+', line):
            func_call_match = re.search(r'\b([a-z_][a-z0-9_]*)\s*\(', line)
            if func_call_match:
                func_name = func_call_match.group(1)
                # If it's a call to a function that might have O(n) operations, check it
                if func_name not in allowed_helpers and func_name.startswith('_'):
                    # Check if this function has O(n) operations
                    if _function_has_on_operations(lines, func_name):
                        problematic_lines.append((i, f"call to {func_name}() which contains O(n) operations", line_stripped))
    
    return problematic_lines


def main():
    """Main function to check _get_model_cost_key performance requirements."""
    problematic_lines = check_get_model_cost_key_performance()
    
    if problematic_lines:
        print("\nERROR: Found O(n) operations in _get_model_cost_key:")
        for line_num, operation, context in problematic_lines:
            print(f"  Line {line_num}: {operation} - {context}")
        
        print("\nWARNING: Only O(1) lookup operations are acceptable in _get_model_cost_key.")
        print("Any O(n) operations will cause severe CPU overhead.")
        
        raise Exception(
            f"Found {len(problematic_lines)} O(n) operation(s) in _get_model_cost_key. "
            f"This violates the performance requirement."
        )
    else:
        print("OK: No O(n) operations found in _get_model_cost_key. Performance requirement satisfied.")


if __name__ == "__main__":
    main()
