import ast
import os
import re
from typing import List, Dict, Any


class SensitiveLogDetector(ast.NodeVisitor):
    """
    Detects logger.info() statements that might log sensitive request/response data.
    """
    
    def __init__(self):
        self.violations = []
        self.current_file = None
    
    def set_file(self, file_path: str):
        """Set the current file being analyzed"""
        self.current_file = file_path
    
    def visit_Call(self, node):
        """Visit function calls to detect logger.info() with sensitive data"""
        if self._is_logger_info_call(node):
            # Check all arguments to the logger.info() call
            for arg in node.args:
                if self._contains_sensitive_data(arg):
                    violation = {
                        "file": self.current_file,
                        "line": node.lineno,
                        "call": self._get_call_string(node),
                        "reason": self._get_violation_reason(arg),
                        "arg": self._get_arg_string(arg)
                    }
                    self.violations.append(violation)
        
        self.generic_visit(node)
    
    def _is_logger_info_call(self, node) -> bool:
        """Check if this is a logger.info() call"""
        if not isinstance(node.func, ast.Attribute):
            return False
        
        # Check for various logger patterns:
        # logger.info(), verbose_logger.info(), verbose_proxy_logger.info(), etc.
        if node.func.attr == "info":
            if isinstance(node.func.value, ast.Name):
                logger_name = node.func.value.id
                return any(pattern in logger_name.lower() for pattern in ["logger", "log"])
        
        return False
    
    def _contains_sensitive_data(self, arg) -> bool:
        """Check if the argument might contain sensitive data"""
        # Convert argument to string for analysis
        arg_str = self._get_arg_string(arg).lower()
        
        # Skip obvious non-sensitive patterns
        non_sensitive_patterns = [
            r'^["\'][\w\s\-_:.,!?]*["\']$',  # Simple static strings
            r'^["\'][^{%]*["\']$',  # Strings without format placeholders
        ]
        
        # Skip common safe phrases that contain sensitive keywords
        safe_phrases = [
            r'request\s+(completed|finished|started|processing)',
            r'response\s+(sent|received|processed)',
            r'data\s+(inserted|updated|deleted|saved)\s+into',
            r'(successfully|failed)\s+(request|response)',
            r'(starting|ending|completed)\s+(request|response)',
            r'no\s+(usage\s+)?data\s+found',
            r'found\s+\d+.*records',
            r'exported\s+\d+.*records',
        ]
        
        for pattern in non_sensitive_patterns:
            if re.search(pattern, arg_str):
                # Check if it's a safe phrase first
                for safe_pattern in safe_phrases:
                    if re.search(safe_pattern, arg_str, re.IGNORECASE):
                        return False
                
                # Then check if the static string mentions sensitive keywords
                if not any(keyword in arg_str for keyword in 
                          ['request', 'response', 'data', 'body', 'payload', 'token', 'auth', 'credential']):
                    return False
        
        # Direct variable/attribute patterns that are likely sensitive
        sensitive_patterns = [
            r'\brequest\b(?!\s*(id|status|method))',  # request but not request_id, request_status, request_method
            r'\bresponse\b(?!\s*(status|code|time))',  # response but not response_status, response_code
            r'\bdata\b(?=[\.\[\s]|$)',  # data followed by . [ space or end
            r'\bbody\b(?=[\.\[\s]|$)',
            r'\bpayload\b(?=[\.\[\s]|$)',
            r'\bmessages?\b(?=[\.\[\s]|$)',
            r'\bcontent\b(?=[\.\[\s]|$)',
            r'\binput\b(?=[\.\[\s]|$)',
            r'\boutput\b(?=[\.\[\s]|$)',
            r'\bargs\b(?=[\.\[\s]|$)',
            r'\bkwargs\b(?=[\.\[\s]|$)',
            r'\bparams\b(?=[\.\[\s]|$)',
            r'\bheaders\b(?=[\.\[\s]|$)',
            r'\bapi_key\b',
            r'\btoken\b(?!\s*(name|id))',  # token but not token_name, token_id
            r'\bauth\b(?=[\.\[\s]|$)',
            r'\bcredentials?\b'
        ]
        
        # Check for direct variable references with context
        for pattern in sensitive_patterns:
            if re.search(pattern, arg_str):
                return True
        
        # Check for format strings that might interpolate sensitive data
        if self._is_format_string_with_sensitive_data(arg):
            return True
        
        # Check for JSON dumps or string formatting of objects
        if self._is_object_serialization(arg):
            return True
            
        return False
    
    def _is_format_string_with_sensitive_data(self, arg) -> bool:
        """Check if this is a format string that might contain sensitive data"""
        # Check for f-strings
        if isinstance(arg, ast.JoinedStr):
            for value in arg.values:
                if isinstance(value, ast.FormattedValue):
                    value_str = self._get_arg_string(value.value).lower()
                    if any(pattern in value_str for pattern in 
                          ['request', 'response', 'data', 'body', 'content', 'messages']):
                        return True
        
        # Check for .format() calls
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
            if arg.func.attr == "format":
                # Check the base string for suspicious patterns
                base_str = self._get_arg_string(arg.func.value).lower()
                if "{}" in base_str or "{" in base_str:
                    # Check format arguments for sensitive data
                    for format_arg in arg.args:
                        format_str = self._get_arg_string(format_arg).lower()
                        if any(pattern in format_str for pattern in 
                              ['request', 'response', 'data', 'body', 'content']):
                            return True
        
        return False
    
    def _is_object_serialization(self, arg) -> bool:
        """Check if this is serializing an object that might contain sensitive data"""
        arg_str = self._get_arg_string(arg)
        
        # Check for json.dumps() calls
        if isinstance(arg, ast.Call):
            if (isinstance(arg.func, ast.Attribute) and 
                arg.func.attr == "dumps" and
                isinstance(arg.func.value, ast.Name) and
                arg.func.value.id == "json"):
                return True
            
            # Check for str() calls on potentially sensitive objects
            if (isinstance(arg.func, ast.Name) and arg.func.id == "str" and
                len(arg.args) > 0):
                obj_str = self._get_arg_string(arg.args[0]).lower()
                if any(pattern in obj_str for pattern in 
                      ['request', 'response', 'data', 'body']):
                    return True
        
        return False
    
    def _get_violation_reason(self, arg) -> str:
        """Get a human-readable reason for the violation"""
        arg_str = self._get_arg_string(arg).lower()
        
        if 'request' in arg_str:
            return "Potentially logging request data"
        elif 'response' in arg_str:
            return "Potentially logging response data"
        elif any(pattern in arg_str for pattern in ['data', 'body', 'payload', 'content']):
            return "Potentially logging sensitive data/body/content"
        elif any(pattern in arg_str for pattern in ['messages', 'input', 'output']):
            return "Potentially logging message/input/output data"
        elif any(pattern in arg_str for pattern in ['api_key', 'token', 'auth', 'credentials']):
            return "Potentially logging authentication data"
        else:
            return "Potentially logging sensitive data"
    
    def _get_call_string(self, node) -> str:
        """Get string representation of the function call"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                return f"{self._get_arg_string(node.func)}(...)"
        except:
            return "logger.info(...)"
    
    def _get_arg_string(self, arg) -> str:
        """Get string representation of an argument"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(arg)
            else:
                # Fallback for older Python versions
                if isinstance(arg, ast.Name):
                    return arg.id
                elif isinstance(arg, ast.Attribute):
                    return f"{self._get_arg_string(arg.value)}.{arg.attr}"
                elif isinstance(arg, ast.Str):
                    return repr(arg.s)
                elif isinstance(arg, ast.Constant):
                    return repr(arg.value)
                else:
                    return str(type(arg).__name__)
        except:
            return "unknown"


def check_sensitive_logging(base_dir: str) -> List[Dict[str, Any]]:
    """
    Check for logger.info() statements that might log sensitive data.
    
    Args:
        base_dir: Base directory to scan (typically the litellm root)
        
    Returns:
        List of violations found
    """
    detector = SensitiveLogDetector()
    all_violations = []
    
    # Directories to scan - only main litellm codebase
    scan_dirs = [
        "litellm",
        "enterprise"  # Include enterprise directory if it exists
    ]
    
    # Directories to exclude (third-party code, venvs, etc.)
    exclude_dirs = {
        "venv", "venv313", ".venv", "env", ".env",
        "node_modules", "__pycache__", ".git",
        "build", "dist", ".tox", "clean_env",
        "litellm_env", "myenv", "py313_env",
        "venv_sip_bypass", "mypyc_env"
    }
    
    for scan_dir in scan_dirs:
        dir_path = os.path.join(base_dir, scan_dir)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist, skipping.")
            continue
        
        print(f"Scanning directory: {dir_path}")
        
        for root, dirs, files in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Skip if we're in a virtual environment or third-party directory
            relative_root = os.path.relpath(root, base_dir)
            if any(excluded in relative_root.split(os.sep) for excluded in exclude_dirs):
                continue
            
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, base_dir)
                    
                    # Skip files that are clearly third-party or generated
                    if any(excluded in relative_path for excluded in exclude_dirs):
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            tree = ast.parse(content)
                        
                        detector.set_file(relative_path)
                        detector.visit(tree)
                        
                    except SyntaxError as e:
                        print(f"Warning: Syntax error in file {relative_path}: {e}")
                        continue
                    except UnicodeDecodeError as e:
                        print(f"Warning: Unicode decode error in file {relative_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing file {relative_path}: {e}")
                        continue
    
    return detector.violations


def main():
    """Main function to run the sensitive logging check"""
    # Get the base directory (assume we're running from tests/code_coverage_tests/)
    ###################
    # Running locally
    ###################
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = os.path.join(current_dir, "..", "..")
    # base_dir = os.path.abspath(base_dir)

    ###################
    # Running in CI/CD
    ###################
    base_dir = "./litellm"  # Adjust this path as needed
    
    print(f"Checking for sensitive logging in: {base_dir}")
    
    violations = check_sensitive_logging(base_dir)
    
    if violations:
        print(f"\n❌ Found {len(violations)} potential violations:")
        print("=" * 80)
        
        for i, violation in enumerate(violations, 1):
            print(f"\n{i}. {violation['file']}:{violation['line']}")
            print(f"   Reason: {violation['reason']}")
            print(f"   Call: {violation['call']}")
            print(f"   Argument: {violation['arg']}")
        
        print("\n" + "=" * 80)
        print("⚠️  SECURITY WARNING:")
        print("These logger.info() statements may log sensitive request/response data.")
        print("Consider changing them to logger.debug() or removing sensitive data.")
        print("This is critical for PII compliance and security.")
        print("Please contact @ishaan-jaff for more details about this check. DO NOT VIOLATE THIS CHECK.")
        
        return 1  # Exit with error code
    else:
        print("\n✅ No sensitive logging violations found!")
        return 0


if __name__ == "__main__":
    exit(main())
