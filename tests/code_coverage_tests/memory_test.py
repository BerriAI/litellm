"""
Memory Violation Detection Test

Detects bad memory patterns in the LiteLLM codebase that can lead to memory leaks or OOMs.

The detector uses a modular pattern-based system. To add detection for new memory patterns:

1. Create a Pattern subclass implementing get_pattern_name(), visit_assign(), and check_cleanup()
   - You can extend the Pattern class with additional methods as needed for your detection logic
2. Add the pattern to MemoryViolationDetector.DEFAULT_PATTERNS

Currently detects: queue.get() / queue.get_nowait() operations where variables aren't set to None.
"""

import ast
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence


class Pattern(ABC):
    """Base class for memory violation detection patterns"""
    
    @abstractmethod
    def get_pattern_name(self) -> str:
        """Return unique identifier for this violation type"""
        pass
    
    @abstractmethod
    def visit_assign(self, node: ast.Assign, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect memory-sensitive operations in assignment. Returns list of {line, var_name, call} dicts."""
        pass
    
    @abstractmethod
    def check_cleanup(self, operations: List[Dict[str, Any]], function_body: List[ast.stmt], 
                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify variables are set to None. Returns list of violation dicts."""
        pass


class QueueGetPattern(Pattern):
    """Detects queue.get()/get_nowait() operations that aren't cleared"""
    
    def get_pattern_name(self) -> str:
        return "queue_reference_not_cleared"
    
    def visit_assign(self, node: ast.Assign, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect queue.get() or queue.get_nowait() calls where object name contains 'queue'"""
        operations = []
        
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr in ("get", "get_nowait"):
                obj_name = context["get_attr_string"](func.value)
                if "queue" in obj_name.lower() and node.targets and isinstance(node.targets[0], ast.Name):
                    operations.append({
                        "line": node.lineno,
                        "var_name": node.targets[0].id,
                        "call": context["get_call_string"](node.value),
                    })
        
        return operations
    
    def check_cleanup(self, operations: List[Dict[str, Any]], function_body: List[ast.stmt],
                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flag queue variables that aren't set to None"""
        violations = []
        is_var_set_to_none = context["is_var_set_to_none"]
        current_function = context["current_function"]
        file_path = context["file_path"]
        
        queue_vars = {op["var_name"]: op["line"] for op in operations}
        
        for var_name, line_num in queue_vars.items():
            if not is_var_set_to_none(var_name, function_body):
                violations.append({
                    "line": line_num,
                    "type": self.get_pattern_name(),
                    "var_name": var_name,
                    "function": current_function,
                    "file_path": file_path,
                    "message": (
                        f"Queue variable '{var_name}' in function "
                        f"'{current_function}' is not set to None after use. "
                        f"If the runtime is overwhelmed, this can cause OOM (Out of Memory) errors."
                    ),
                })
        
        return violations


class MemoryViolationDetector(ast.NodeVisitor):
    """AST visitor that detects memory violations using registered patterns"""
    
    DEFAULT_PATTERNS: List[Pattern] = [QueueGetPattern()]

    def __init__(self, file_path: str, patterns: Optional[Sequence[Pattern]] = None):
        self.file_path = file_path
        self.violations: List[Dict[str, Any]] = []
        self.current_function: Optional[str] = None
        self.patterns = self.DEFAULT_PATTERNS if patterns is None else patterns
        
        self.pattern_operations: Dict[str, List[Dict[str, Any]]] = {
            pattern.get_pattern_name(): [] for pattern in self.patterns
        }
        
        self._context = {
            "get_call_string": self._get_call_string,
            "get_attr_string": self._get_attr_string,
            "is_var_set_to_none": self._is_var_set_to_none,
            "current_function": None,
            "file_path": file_path,
        }

    def visit_FunctionDef(self, node):
        """Track function scope and check cleanup after visiting"""
        old_function = self.current_function
        self.current_function = node.name
        self._context["current_function"] = node.name
        
        for pattern_name in self.pattern_operations:
            self.pattern_operations[pattern_name] = []
        
        self.generic_visit(node)
        self._check_function_cleanup(node)
        
        self.current_function = old_function
        self._context["current_function"] = old_function

    def visit_AsyncFunctionDef(self, node):
        """Track async function scope and check cleanup after visiting"""
        old_function = self.current_function
        self.current_function = node.name
        self._context["current_function"] = node.name
        
        for pattern_name in self.pattern_operations:
            self.pattern_operations[pattern_name] = []
        
        self.generic_visit(node)
        self._check_function_cleanup(node)
        
        self.current_function = old_function
        self._context["current_function"] = old_function

    def visit_Assign(self, node):
        """Detect memory-sensitive operations in assignments"""
        for pattern in self.patterns:
            operations = pattern.visit_assign(node, self._context)
            self.pattern_operations[pattern.get_pattern_name()].extend(operations)
        
        self.generic_visit(node)

    def _check_function_cleanup(self, node):
        """Check cleanup for all detected operations"""
        for pattern in self.patterns:
            operations = self.pattern_operations[pattern.get_pattern_name()]
            if operations:
                violations = pattern.check_cleanup(operations, node.body, self._context)
                self.violations.extend(violations)

    def _is_var_set_to_none(self, var_name: str, body: List[ast.stmt]) -> bool:
        """Check if variable is set to None after its initial assignment"""
        assignment_line = None
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            assignment_line = node.lineno
                            break
                    if assignment_line:
                        break
            if assignment_line:
                break
        
        if not assignment_line:
            return False
        
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name and node.lineno > assignment_line:
                            if isinstance(node.value, ast.Constant) and node.value.value is None:
                                return True
                            try:
                                NameConstant = getattr(ast, "NameConstant", None)
                                if NameConstant and isinstance(node.value, NameConstant):
                                    if getattr(node.value, "value", None) is None:
                                        return True
                            except (AttributeError, TypeError):
                                pass
        return False

    def _get_call_string(self, node: ast.Call) -> str:
        """Get string representation of function call"""
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(node)
            elif isinstance(node.func, ast.Attribute):
                return f"{self._get_attr_string(node.func.value)}.{node.func.attr}()"
            return str(node)
        except Exception:
            return str(node)

    def _get_attr_string(self, node: ast.AST) -> str:
        """Get string representation of attribute access"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attr_string(node.value)}.{node.attr}"
        return str(node)


def check_file_for_memory_violations(file_path: str, patterns: Optional[Sequence[Pattern]] = None) -> List[Dict[str, Any]]:
    """Check a single file for memory violations"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if "test" in file_path.lower() or "__pycache__" in file_path:
            return []
        
        tree = ast.parse(content, filename=file_path)
        detector = MemoryViolationDetector(file_path, patterns)
        detector.visit(tree)
        return detector.violations
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def check_directory_for_memory_violations(directory_path: str, ignore_patterns: Optional[List[str]] = None,
                                          patterns: Optional[Sequence[Pattern]] = None) -> List[Dict[str, Any]]:
    """Recursively scan directory for memory violations"""
    if ignore_patterns is None:
        ignore_patterns = ["__pycache__", ".pyc", "site-packages", "venv", ".venv", "env", ".env", "node_modules", "tests"]
    
    all_violations = []
    for root, _dirs, files in os.walk(directory_path):
        if any(pattern in root for pattern in ignore_patterns):
            continue
        for file in files:
            if file.endswith(".py"):
                violations = check_file_for_memory_violations(os.path.join(root, file), patterns)
                all_violations.extend(violations)
    return all_violations


def main():
    """Run memory violation detection on codebase"""
    codebase_path = "./litellm"
    
    print("=" * 80)
    print("MEMORY VIOLATION DETECTION TEST")
    print("=" * 80)
    print(f"Scanning: {codebase_path}")
    print(f"Active patterns: {', '.join(p.get_pattern_name() for p in MemoryViolationDetector.DEFAULT_PATTERNS)}")
    print()
    
    violations = check_directory_for_memory_violations(codebase_path)
    
    if violations:
        by_type = {}
        for v in violations:
            vtype = v["type"]
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(v)
        
        print("🚨 MEMORY VIOLATIONS FOUND:")
        print("=" * 80)
        
        total = len(violations)
        for vtype, vlist in by_type.items():
            print(f"\n📋 {vtype.upper().replace('_', ' ')}: {len(vlist)} violation(s)")
            print("-" * 80)
            for v in vlist[:10]:
                print(f"  ❌ {v['file_path'] if 'file_path' in v else 'unknown'}:{v['line']}")
                print(f"     Function: {v['function']}")
                print(f"     Variable: {v['var_name']}")
                print(f"     {v['message']}")
                print()
            if len(vlist) > 10:
                print(f"     ... and {len(vlist) - 10} more violations of this type")
        
        print("=" * 80)
        print(f"🚨 TOTAL VIOLATIONS: {total}")
        print()
        print("💡 RECOMMENDATIONS:")
        print("   1. Set queue variables to None after use: obj = queue.get(); ...; obj = None")
        print("   2. Use bounded queues to prevent unbounded accumulation")
        print("   3. Process items faster than they're added, or drain queues periodically")
        print("=" * 80)
        
        first_v = violations[0]
        raise Exception(
            f"🚨 Found {total} memory violations! "
            f"First violation: {first_v.get('file_path', 'unknown')}:{first_v['line']} - "
            f"{first_v['message']}"
        )
    else:
        print("✅ No memory violations found!")


if __name__ == "__main__":
    main()
