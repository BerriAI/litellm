import unittest
import ast
from typing import List


def add_parent_links(tree: ast.AST) -> None:
    """Add parent links to all nodes in the AST"""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


class LabelsCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.labels_calls: List[tuple[int, str]] = []  # (line_number, code)
        self.in_prometheus_logger = False

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name == "PrometheusLogger":
            self.in_prometheus_logger = True
            self.generic_visit(node)
            self.in_prometheus_logger = False
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if (
            self.in_prometheus_logger
            and isinstance(node, ast.Attribute)
            and node.attr == "labels"
        ):
            # Find the full statement by walking up to parent nodes
            stmt = node
            while hasattr(stmt, "parent") and not isinstance(stmt, ast.stmt):
                stmt = stmt.parent
            if hasattr(stmt, "parent"):
                self.labels_calls.append((node.lineno, ast.unparse(stmt)))
        self.generic_visit(node)


class TestPrometheusLogger(unittest.TestCase):
    def test_no_direct_labels_calls(self):
        """Test that PrometheusLogger never calls .labels directly"""

        # Read the prometheus.py file
        with open("../../litellm/integrations/prometheus.py", "r") as f:
            content = f.read()

        # Parse the file and add parent links
        tree = ast.parse(content)
        add_parent_links(tree)

        # Visit the AST
        visitor = LabelsCallVisitor()
        visitor.visit(tree)

        # Check for violations
        if visitor.labels_calls:
            violation_messages = []
            for line_num, code in visitor.labels_calls:
                violation_messages.append(f"Line {line_num}: {code}")

            self.fail(
                "Found direct .labels calls in PrometheusLogger:\n"
                + "\n".join(violation_messages)
                + "\nPlease use PrometheusMetricManager instead."
            )


if __name__ == "__main__":
    unittest.main()
