import ast
from pathlib import Path


PRODUCTION_ROOTS = (Path("robo_trot"), Path("data"), Path("scripts"))
MIN_DOCSTRING_LINES = 2
MIN_MATH_HEAVY_DOCSTRING_LINES = 4
MATH_HEAVY_NAMES = {
    "acos",
    "arccos",
    "arctan2",
    "cos",
    "cross",
    "dot",
    "hypot",
    "inv",
    "norm",
    "sin",
    "sqrt",
    "tan",
}
MATH_HEAVY_NAME_PARTS = ("ik", "quat", "rotate", "yaw")
MATH_DOC_MARKERS = ("=", "formula", "equation", "radian", "rad", "frame")


def production_python_files() -> list[Path]:
    """Return authored production Python files covered by docstring lint.

    Local review helpers under scripts/debug are intentionally ignored.
    """
    paths: list[Path] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "debug" in path.parts or path.name.endswith("_debug.py"):
                continue
            paths.append(path)
    return paths


def docstring_lines(node: ast.AST) -> list[str]:
    """Return non-empty docstring lines for a class or function node.

    Whitespace-only lines do not count toward the minimum.
    """
    docstring = ast.get_docstring(node) or ""
    return [line.strip() for line in docstring.splitlines() if line.strip()]


def is_math_heavy(node: ast.AST) -> bool:
    """Return whether a function should have a math-oriented docstring.

    The heuristic catches trig, quaternion/rotation, phase, and IK helpers.
    """
    lowered_name = getattr(node, "name", "").lower()
    if any(part in lowered_name for part in MATH_HEAVY_NAME_PARTS):
        return True
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr.lower() in MATH_HEAVY_NAMES:
            return True
        if isinstance(child, ast.Name) and child.id.lower() in MATH_HEAVY_NAMES:
            return True
    return False


def has_math_explanation(lines: list[str]) -> bool:
    """Return whether docstring lines contain explicit math/frame detail.

    Math-heavy methods should state equations, units, or coordinate frames.
    """
    lowered = "\n".join(lines).lower()
    return any(marker in lowered for marker in MATH_DOC_MARKERS)


def test_production_classes_and_functions_have_substantive_docstrings():
    """Lint production classes and functions for docstring coverage.

    Every class/function/method needs at least two non-empty docstring lines.
    """
    failures = []
    for path in production_python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                lines = docstring_lines(node)
                label = f"{path}:{node.lineno} {type(node).__name__} {node.name}"
                if len(lines) < MIN_DOCSTRING_LINES:
                    failures.append(f"{label} docstring has {len(lines)} non-empty line(s)")
                    continue
                if is_math_heavy(node) and len(lines) < MIN_MATH_HEAVY_DOCSTRING_LINES:
                    failures.append(f"{label} math-heavy docstring has {len(lines)} non-empty line(s)")
                    continue
                if is_math_heavy(node) and not has_math_explanation(lines):
                    failures.append(f"{label} math-heavy docstring lacks equation/unit/frame detail")

    assert failures == []
