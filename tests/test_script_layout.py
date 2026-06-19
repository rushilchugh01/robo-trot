from pathlib import Path


def test_scripts_root_contains_only_package_marker():
    """Keep runnable scripts grouped under purpose-specific subfolders."""
    root_python_files = sorted(path.name for path in Path("scripts").glob("*.py"))

    assert root_python_files == ["__init__.py"]


def test_core_script_subfolders_exist():
    """Require the script groups used by the project documentation."""
    expected = {"assets", "data", "policy", "robot", "teacher"}
    actual = {path.name for path in Path("scripts").iterdir() if path.is_dir() and not path.name.startswith("__")}

    assert expected.issubset(actual)
