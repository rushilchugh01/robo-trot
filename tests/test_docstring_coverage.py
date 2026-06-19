import ast
from pathlib import Path


def test_robo_trot_classes_and_functions_have_docstrings():
    missing = []
    for path in sorted(Path("robo_trot").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{path}:{node.lineno} {type(node).__name__} {node.name}")

    assert missing == []
