import ast
import os

class EnterpriseImportFinder(ast.NodeVisitor):
    def __init__(self):
        self.unsafe_imports = []
        self.current_file = None
        self.in_try_block = False
        self.try_blocks = []

    def visit_Try(self, node):
        # Track that we're entering a try block
        self.in_try_block = True
        self.try_blocks.append(node)
        # Visit all nodes in the try block
        for item in node.body:
            self.visit(item)
        # Visit except blocks
        for handler in node.handlers:
            for item in handler.body:
                self.visit(item)
        # Visit else block if it exists
        for item in node.orelse:
            self.visit(item)
        # Visit finally block if it exists
        for item in node.finalbody:
            self.visit(item)
        # We're leaving the try block
        self.try_blocks.pop()
        self.in_try_block = len(self.try_blocks) > 0

    def visit_Import(self, node):
        # Check for direct imports of litellm_enterprise
        for name in node.names:
            if "litellm_enterprise" in name.name or "enterprise" in name.name:
                if not self.in_try_block:
                    self.unsafe_imports.append({
                        "file": self.current_file,
                        "line": node.lineno,
                        "import": name.name,
                        "context": "direct import"
                    })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Check for from litellm_enterprise imports
        if node.module and ("litellm_enterprise" in node.module or "enterprise" in node.module):
            if not self.in_try_block:
                self.unsafe_imports.append({
                    "file": self.current_file,
                    "line": node.lineno,
                    "import": f"from {node.module}",
                    "context": "from import"
                })
        self.generic_visit(node)

def find_unsafe_enterprise_imports_in_file(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    finder = EnterpriseImportFinder()
    finder.current_file = file_path
    finder.visit(tree)
    return finder.unsafe_imports

def find_unsafe_enterprise_imports_in_directory(directory):
    unsafe_imports = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imports = find_unsafe_enterprise_imports_in_file(file_path)
                if imports:
                    unsafe_imports.extend(imports)
    return unsafe_imports

if __name__ == "__main__":
    # Check for unsafe enterprise imports in the litellm directory
    directory_path = "./litellm"
    unsafe_imports = find_unsafe_enterprise_imports_in_directory(directory_path)
    
    if unsafe_imports:
        print("🚨 UNSAFE ENTERPRISE IMPORTS FOUND (not in try-except blocks):")
        for imp in unsafe_imports:
            print(f"File: {imp['file']}")
            print(f"Line: {imp['line']}")
            print(f"Import: {imp['import']}")
            print(f"Context: {imp['context']}")
            print("---")
        
        # Raise exception to fail CI/CD
        raise Exception(
            "🚨 Unsafe enterprise imports found. All enterprise imports must be wrapped in try-except blocks."
        )
    else:
        print("✅ No unsafe enterprise imports found.")
