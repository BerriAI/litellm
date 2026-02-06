import ast
import os


def check_for_litellm_module_deletion(base_dir):
    """
    Checks for code patterns that delete litellm modules from sys.modules
    in the test_litellm directory.
    
    Specifically looks for patterns like:
    for module in list(sys.modules.keys()):
        if module.startswith("litellm"):
            del sys.modules[module]
    """
    problematic_files = []
    test_dir = os.path.join(base_dir, "test_litellm")
    
    if not os.path.exists(test_dir):
        print(f"Warning: Directory {test_dir} does not exist.")
        return []

    print(f"Checking directory: {test_dir}")
    
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        tree = ast.parse(f.read())
                except SyntaxError:
                    print(f"Warning: Syntax error in file {file_path}")
                    continue
                
                # Check for litellm module deletion patterns
                if has_litellm_module_deletion(tree):
                    relative_path = os.path.relpath(file_path, base_dir)
                    problematic_files.append(relative_path)
                    print(f"Found litellm module deletion in: {relative_path}")
    
    return problematic_files


def has_litellm_module_deletion(tree):
    """
    Checks if the AST contains patterns that delete litellm modules from sys.modules.
    
    Looks for:
    1. Loops over sys.modules.keys()
    2. Conditions checking if module startswith "litellm"
    3. del sys.modules[module] statements
    """
    class LiteLLMDeletionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.has_sys_modules_loop = False
            self.has_litellm_check = False
            self.has_del_sys_modules = False
            self.current_for_target = None
        
        def visit_For(self, node):
            # Check if we're looping over sys.modules.keys()
            if (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Attribute) and
                isinstance(node.iter.func.value, ast.Attribute) and
                isinstance(node.iter.func.value.value, ast.Name) and
                node.iter.func.value.value.id == "sys" and
                node.iter.func.value.attr == "modules" and
                node.iter.func.attr == "keys"):
                
                self.has_sys_modules_loop = True
                if isinstance(node.target, ast.Name):
                    self.current_for_target = node.target.id
                
                # Check the body of the for loop
                for stmt in node.body:
                    self.visit(stmt)
            
            # Also check for list(sys.modules.keys()) pattern
            elif (isinstance(node.iter, ast.Call) and
                  isinstance(node.iter.func, ast.Name) and
                  node.iter.func.id == "list" and
                  len(node.iter.args) == 1 and
                  isinstance(node.iter.args[0], ast.Call) and
                  isinstance(node.iter.args[0].func, ast.Attribute) and
                  isinstance(node.iter.args[0].func.value, ast.Attribute) and
                  isinstance(node.iter.args[0].func.value.value, ast.Name) and
                  node.iter.args[0].func.value.value.id == "sys" and
                  node.iter.args[0].func.value.attr == "modules" and
                  node.iter.args[0].func.attr == "keys"):
                
                self.has_sys_modules_loop = True
                if isinstance(node.target, ast.Name):
                    self.current_for_target = node.target.id
                
                # Check the body of the for loop
                for stmt in node.body:
                    self.visit(stmt)
            
            self.generic_visit(node)
        
        def visit_If(self, node):
            # Check for conditions like module.startswith("litellm")
            if (isinstance(node.test, ast.Call) and
                isinstance(node.test.func, ast.Attribute) and
                isinstance(node.test.func.value, ast.Name) and
                node.test.func.value.id == self.current_for_target and
                node.test.func.attr == "startswith" and
                len(node.test.args) == 1 and
                isinstance(node.test.args[0], ast.Constant) and
                node.test.args[0].value == "litellm"):
                
                self.has_litellm_check = True
                
                # Check the body of the if statement
                for stmt in node.body:
                    self.visit(stmt)
            
            self.generic_visit(node)
        
        def visit_Delete(self, node):
            # Check for del sys.modules[module]
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and
                    isinstance(target.value, ast.Attribute) and
                    isinstance(target.value.value, ast.Name) and
                    target.value.value.id == "sys" and
                    target.value.attr == "modules" and
                    isinstance(target.slice, ast.Name) and
                    target.slice.id == self.current_for_target):
                    
                    self.has_del_sys_modules = True
            
            self.generic_visit(node)
    
    visitor = LiteLLMDeletionVisitor()
    visitor.visit(tree)
    
    return (visitor.has_sys_modules_loop and 
            visitor.has_litellm_check and 
            visitor.has_del_sys_modules)


def main():
    """
    Main function to check for litellm module deletion patterns in test files.
    """
    # local dir 
    #tests_dir = "../../tests/"
    
    # ci/cd dir
    tests_dir = "./tests/"
    
    problematic_files = check_for_litellm_module_deletion(tests_dir)
    
    if problematic_files:
        print("\nERROR: Found files that delete litellm modules from sys.modules:")
        for file_path in problematic_files:
            print(f"  - {file_path}")
        
        raise Exception(
            f"Found {len(problematic_files)} file(s) that delete litellm modules from sys.modules. "
            f"This can cause import issues and test failures. Files: {problematic_files}"
        )
    else:
        print("✓ No litellm module deletion patterns found in test_litellm directory.")


if __name__ == "__main__":
    main()
