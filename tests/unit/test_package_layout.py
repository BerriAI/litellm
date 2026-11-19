import os

TESTS_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_ci_timing_probe_unit_package_has_initializer():
    assert os.path.isfile(os.path.join(TESTS_UNIT_DIR, "__init__.py"))


def test_every_directory_under_tests_unit_is_a_package():
    missing = []
    for root, dirs, _files in os.walk(TESTS_UNIT_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if not os.path.isfile(os.path.join(root, "__init__.py")):
            missing.append(os.path.relpath(root, TESTS_UNIT_DIR))
    assert missing == []
