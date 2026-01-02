"""
Memory Violation Detection Test

Detects bad memory patterns in the LiteLLM codebase that can lead to memory leaks or OOMs.

The detector uses a modular pattern-based system. To add detection for new memory patterns:

1. Create a Pattern subclass implementing get_pattern_name(), visit_assign(), and check_cleanup()
   - You can extend the Pattern class with additional methods as needed for your detection logic
2. Add the pattern to MemoryViolationDetector.DEFAULT_PATTERNS

Currently detects:
- queue.get() / queue.get_nowait() operations where variables aren't set to None
- Class-level data structures that have add operations during runtime without size limits:
  * Built-in: list, dict, set
  * Collections: deque, defaultdict, Counter, OrderedDict, ChainMap
  * Queues: queue.Queue, asyncio.Queue (if unbounded, i.e., no maxsize parameter)
  * Heap operations: heapq.heappush(), heapq.heapreplace(), heapq.heappushpop() on class-level lists
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


class UnboundedDataStructurePattern(Pattern):
    """Detects class-level data structures (lists, dicts, sets) that can grow unbounded"""
    
    def get_pattern_name(self) -> str:
        return "unbounded_data_structure"
    
    def visit_assign(self, node: ast.Assign, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect list/dict/set creations that are at class level"""
        operations = []
        
        # Check if this is a data structure creation
        is_data_structure = False
        structure_type = None
        
        if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
            is_data_structure = True
            if isinstance(node.value, ast.List):
                structure_type = "list"
            elif isinstance(node.value, ast.Dict):
                structure_type = "dict"
            elif isinstance(node.value, ast.Set):
                structure_type = "set"
        elif isinstance(node.value, ast.Call):
            # Check for list(), dict(), set() calls
            func = node.value.func
            if isinstance(func, ast.Name):
                if func.id in ("list", "dict", "set"):
                    is_data_structure = True
                    structure_type = func.id
            elif isinstance(func, ast.Attribute):
                # Handle cases like collections.defaultdict(list), collections.deque(), etc.
                obj_name = context["get_attr_string"](func.value)
                attr_name = func.attr
                
                # Check for collections module data structures
                if "collections" in obj_name.lower() or "collections" in str(func.value):
                    if attr_name in ("deque", "defaultdict", "Counter", "OrderedDict", "ChainMap"):
                        # For deque, we track it and let size checks determine if it's bounded
                        # (deque with maxlen parameter is bounded, but we detect that via size checks)
                        is_data_structure = True
                        structure_type = attr_name
                    elif attr_name in ("list", "dict", "set"):
                        # collections.defaultdict(list) pattern
                        is_data_structure = True
                        structure_type = "defaultdict" if "defaultdict" in obj_name.lower() else attr_name
                # Check for queue.Queue, asyncio.Queue (if unbounded)
                elif "queue" in obj_name.lower() or "asyncio" in obj_name.lower():
                    if attr_name == "Queue":
                        # Check if maxsize is set (bounded queue)
                        has_maxsize = False
                        for keyword in node.value.keywords:
                            if keyword.arg == "maxsize":
                                has_maxsize = True
                                break
                        if not has_maxsize:
                            is_data_structure = True
                            structure_type = "queue"
                # Direct attribute access like deque(), Counter(), etc.
                elif attr_name in ("deque", "defaultdict", "Counter", "OrderedDict", "ChainMap"):
                    is_data_structure = True
                    structure_type = attr_name
        
        if is_data_structure and node.targets and isinstance(node.targets[0], ast.Name):
            scope = context.get("current_scope", "function")
            # Only track if it's at class level (not module level)
            if scope == "class":
                operations.append({
                    "line": node.lineno,
                    "var_name": node.targets[0].id,
                    "structure_type": structure_type,
                    "scope": scope,
                    "call": context["get_call_string"](node.value) if isinstance(node.value, ast.Call) else f"{structure_type}()",
                })
        
        return operations
    
    def check_cleanup(self, operations: List[Dict[str, Any]], function_body: List[ast.stmt],
                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flag persistent data structures that have add operations without size limits"""
        violations = []
        current_function = context["current_function"]
        current_scope = context.get("current_scope", "function")
        file_path = context["file_path"]
        get_attr_string = context["get_attr_string"]
        
        # Skip if this is initialization code (module-level, class-level, or __init__ methods)
        # Only flag operations in regular methods/functions that can be called during runtime
        is_initialization = (
            current_scope in ("module", "class") or
            current_function in ("__init__", "__new__", "__class_init__") or
            current_function is None  # Module-level code
        )
        
        if is_initialization:
            return violations  # Don't flag initialization code
        
        # Track which variables have add operations and size checks
        var_add_operations = {}  # var_name -> list of lines with add operations
        var_size_checks = {}  # var_name -> has size limit check
        
        # Build a set of variable names to check
        tracked_vars = {op["var_name"]: op for op in operations}
        
        # Scan body for operations on these variables
        for stmt in function_body:
            for node in ast.walk(stmt):
                # Check for method calls that add items
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    obj_name = get_attr_string(node.func.value)
                    
                    # Check if this is an add operation on one of our tracked variables
                    for var_name, op in tracked_vars.items():
                        structure_type = op["structure_type"]
                        
                        # Match variable name (exact or as attribute)
                        if obj_name == var_name or obj_name.endswith(f".{var_name}") or obj_name.endswith(f"['{var_name}']"):
                            # Check for add operations
                            add_ops = {
                                "list": ["append", "extend", "insert"],
                                "dict": ["update", "setdefault"],
                                "set": ["add", "update"],
                                "deque": ["append", "appendleft", "extend", "extendleft", "insert"],
                                "defaultdict": ["update", "setdefault"],
                                "Counter": ["update"],
                                "OrderedDict": ["update", "setdefault"],
                                "ChainMap": ["new_child"],
                                "queue": ["put", "put_nowait"],
                            }
                            
                            if attr_name in add_ops.get(structure_type, []):
                                if var_name not in var_add_operations:
                                    var_add_operations[var_name] = []
                                var_add_operations[var_name].append(node.lineno)
                            
                            # Check for size limit checks (len() calls, maxsize/maxlen attributes)
                            if (attr_name in ("__len__",) or 
                                "maxsize" in attr_name.lower() or 
                                "max_size" in attr_name.lower() or
                                attr_name == "maxlen"):  # For deque
                                var_size_checks[var_name] = True
                
                # Check for heapq operations on tracked lists (heapq.heappush, heapq.heappop)
                if isinstance(node, ast.Call):
                    func = node.func
                    # Check for heapq.heappush(list_var, item) or heapq.heappop(list_var)
                    if isinstance(func, ast.Attribute):
                        func_obj = get_attr_string(func.value)
                        func_name = func.attr
                        # Check if it's a heapq operation
                        if func_obj == "heapq" and func_name in ("heappush", "heapreplace", "heappushpop"):
                            # First argument should be our tracked variable
                            if len(node.args) > 0:
                                arg_name = get_attr_string(node.args[0])
                                for var_name, op in tracked_vars.items():
                                    if op["structure_type"] == "list" and (
                                        arg_name == var_name or arg_name.endswith(f".{var_name}")
                                    ):
                                        if var_name not in var_add_operations:
                                            var_add_operations[var_name] = []
                                        var_add_operations[var_name].append(node.lineno)
                
                # Check for dict item assignment: dict[key] = value
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript):
                            target_name = get_attr_string(target.value)
                            for var_name in tracked_vars:
                                if target_name == var_name or target_name.endswith(f".{var_name}"):
                                    if var_name not in var_add_operations:
                                        var_add_operations[var_name] = []
                                    var_add_operations[var_name].append(node.lineno)
                
                # Check for augmented assignment: list += [...]
                if isinstance(node, ast.AugAssign):
                    target_name = get_attr_string(node.target)
                    for var_name in tracked_vars:
                        if target_name == var_name or target_name.endswith(f".{var_name}"):
                            if var_name not in var_add_operations:
                                var_add_operations[var_name] = []
                            var_add_operations[var_name].append(node.lineno)
                
                # Check for size comparisons in conditionals
                if isinstance(node, (ast.If, ast.While, ast.Assert)):
                    test = getattr(node, "test", None)
                    if test:
                        for comp_node in ast.walk(test):
                            if isinstance(comp_node, ast.Compare):
                                left_str = get_attr_string(comp_node.left) if hasattr(comp_node, "left") else ""
                                # Check for len() calls
                                if isinstance(comp_node.left, ast.Call):
                                    call_func = comp_node.left.func
                                    if isinstance(call_func, ast.Name) and call_func.id == "len":
                                        if len(comp_node.left.args) > 0:
                                            arg_name = get_attr_string(comp_node.left.args[0])
                                            for var_name in tracked_vars:
                                                if arg_name == var_name or arg_name.endswith(f".{var_name}"):
                                                    # Check if comparing to a limit
                                                    for comparator in comp_node.comparators:
                                                        if isinstance(comparator, ast.Constant):
                                                            var_size_checks[var_name] = True
                                                        elif isinstance(comparator, ast.Name):
                                                            # Could be a constant like MAX_SIZE
                                                            if "max" in comparator.id.lower() or "limit" in comparator.id.lower():
                                                                var_size_checks[var_name] = True
                                                        # Handle deprecated ast.Num for Python < 3.8
                                                        try:
                                                            Num = getattr(ast, "Num", None)
                                                            if Num and isinstance(comparator, Num):
                                                                var_size_checks[var_name] = True
                                                        except (AttributeError, TypeError):
                                                            pass
                                # Check for direct variable comparisons
                                for var_name in tracked_vars:
                                    if var_name in left_str:
                                        for comparator in comp_node.comparators:
                                            if isinstance(comparator, ast.Constant):
                                                var_size_checks[var_name] = True
                                            # Handle deprecated ast.Num for Python < 3.8
                                            try:
                                                Num = getattr(ast, "Num", None)
                                                if Num and isinstance(comparator, Num):
                                                    var_size_checks[var_name] = True
                                            except (AttributeError, TypeError):
                                                pass
        
        # Flag violations: persistent structures with add operations but no size checks
        for op in operations:
            var_name = op["var_name"]
            structure_type = op["structure_type"]
            
            if var_name in var_add_operations and var_name not in var_size_checks:
                violations.append({
                    "line": op["line"],
                    "type": self.get_pattern_name(),
                    "var_name": var_name,
                    "function": current_function or "class-level",
                    "file_path": file_path,
                    "message": (
                        f"Class-level {structure_type} '{var_name}' "
                        f"has add operations (lines {var_add_operations[var_name]}) but no size limit checks. "
                        f"This can lead to unbounded memory growth and OOM errors during runtime."
                    ),
                })
        
        return violations


class MemoryViolationDetector(ast.NodeVisitor):
    """AST visitor that detects memory violations using registered patterns"""
    
    DEFAULT_PATTERNS: List[Pattern] = [QueueGetPattern(), UnboundedDataStructurePattern()]

    def __init__(self, file_path: str, patterns: Optional[Sequence[Pattern]] = None):
        self.file_path = file_path
        self.violations: List[Dict[str, Any]] = []
        self.current_function: Optional[str] = None
        self.current_scope: str = "module"  # Track current scope: module, class, function
        self.patterns = self.DEFAULT_PATTERNS if patterns is None else patterns
        self.ast_tree: Optional[ast.Module] = None  # Store full AST for module-level checks
        
        self.pattern_operations: Dict[str, List[Dict[str, Any]]] = {
            pattern.get_pattern_name(): [] for pattern in self.patterns
        }
        
        # Track class-level operations separately (for checking in functions)
        self.class_level_operations: Dict[str, List[Dict[str, Any]]] = {
            pattern.get_pattern_name(): [] for pattern in self.patterns
        }
        
        self._context = {
            "get_call_string": self._get_call_string,
            "get_attr_string": self._get_attr_string,
            "is_var_set_to_none": self._is_var_set_to_none,
            "current_function": None,
            "current_scope": "module",
            "file_path": file_path,
        }

    def visit_ClassDef(self, node):
        """Track class scope"""
        old_scope = self.current_scope
        self.current_scope = "class"
        self._context["current_scope"] = "class"
        
        self.generic_visit(node)
        
        self.current_scope = old_scope
        self._context["current_scope"] = old_scope

    def visit_FunctionDef(self, node):
        """Track function scope and check cleanup after visiting"""
        old_function = self.current_function
        old_scope = self.current_scope
        self.current_function = node.name
        self.current_scope = "function"
        self._context["current_function"] = node.name
        self._context["current_scope"] = "function"
        
        for pattern_name in self.pattern_operations:
            self.pattern_operations[pattern_name] = []
        
        self.generic_visit(node)
        self._check_function_cleanup(node)
        
        self.current_function = old_function
        self.current_scope = old_scope
        self._context["current_function"] = old_function
        self._context["current_scope"] = old_scope

    def visit_AsyncFunctionDef(self, node):
        """Track async function scope and check cleanup after visiting"""
        old_function = self.current_function
        old_scope = self.current_scope
        self.current_function = node.name
        self.current_scope = "function"
        self._context["current_function"] = node.name
        self._context["current_scope"] = "function"
        
        for pattern_name in self.pattern_operations:
            self.pattern_operations[pattern_name] = []
        
        self.generic_visit(node)
        self._check_function_cleanup(node)
        
        self.current_function = old_function
        self.current_scope = old_scope
        self._context["current_function"] = old_function
        self._context["current_scope"] = old_scope

    def visit_Assign(self, node):
        """Detect memory-sensitive operations in assignments"""
        for pattern in self.patterns:
            operations = pattern.visit_assign(node, self._context)
            # Track function-level operations
            self.pattern_operations[pattern.get_pattern_name()].extend(operations)
            # Track class-level operations separately (for checking in functions)
            for op in operations:
                if op.get("scope") == "class":
                    self.class_level_operations[pattern.get_pattern_name()].append(op)
        
        self.generic_visit(node)

    def _check_function_cleanup(self, node):
        """Check cleanup for all detected operations"""
        for pattern in self.patterns:
            operations = self.pattern_operations[pattern.get_pattern_name()]
            if operations:
                violations = pattern.check_cleanup(operations, node.body, self._context)
                self.violations.extend(violations)
            
            # For UnboundedDataStructurePattern, also check if this function modifies class-level structures
            if isinstance(pattern, UnboundedDataStructurePattern):
                class_ops = self.class_level_operations[pattern.get_pattern_name()]
                if class_ops and self.current_function not in ("__init__", "__new__", "__class_init__", None):
                    # Check if this regular function modifies class-level structures
                    violations = pattern.check_cleanup(class_ops, node.body, self._context)
                    self.violations.extend(violations)
    
    def _check_module_level_cleanup(self):
        """Check cleanup for module/class level operations"""
        # Module-level operations are now checked when visiting functions
        # This method is kept for potential future use but doesn't need to do anything
        # since we only want to flag runtime modifications in functions, not initialization code
        pass

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
        detector.ast_tree = tree  # Store AST for potential future use
        detector.visit(tree)
        # Class-level operations are checked when visiting functions
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
        
        print("MEMORY VIOLATIONS FOUND:")
        print("=" * 80)
        
        total = len(violations)
        for vtype, vlist in by_type.items():
            print(f"\n{vtype.upper().replace('_', ' ')}: {len(vlist)} violation(s)")
            print("-" * 80)
            for v in vlist[:10]:
                print(f"  [VIOLATION] {v['file_path'] if 'file_path' in v else 'unknown'}:{v['line']}")
                print(f"     Function: {v['function']}")
                print(f"     Variable: {v['var_name']}")
                print(f"     {v['message']}")
                print()
            if len(vlist) > 10:
                print(f"     ... and {len(vlist) - 10} more violations of this type")
        
        print("=" * 80)
        print(f"TOTAL VIOLATIONS: {total}")
        print()
        print("RECOMMENDATIONS:")
        print("   1. Set queue variables to None after use: obj = queue.get(); ...; obj = None")
        print("   2. Use bounded queues to prevent unbounded accumulation")
        print("   3. Process items faster than they're added, or drain queues periodically")
        print("   4. For class-level data structures (lists, dicts, sets) that are modified at runtime:")
        print("      - Add size limit checks: if len(data) >= MAX_SIZE: ...")
        print("      - Implement periodic cleanup or use bounded collections")
        print("      - Consider using collections.deque with maxlen for lists")
        print("=" * 80)
        
        first_v = violations[0]
        raise Exception(
            f"Found {total} memory violations! "
            f"First violation: {first_v.get('file_path', 'unknown')}:{first_v['line']} - "
            f"{first_v['message']}"
        )
    else:
        print("OK No memory violations found!")


if __name__ == "__main__":
    main()
