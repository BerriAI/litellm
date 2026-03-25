import importlib.util
import os


def _load_module():
    script_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "ci_cd",
            "check_committed_build_artifacts.py",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "check_committed_build_artifacts", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_build_artifact_paths_detects_generated_outputs():
    module = _load_module()

    paths = [
        "litellm-proxy-extras/dist/package.whl",
        "build/output.txt",
        "src/archive.tar.gz",
        "docs/readme.md",
    ]

    assert module.find_build_artifact_paths(paths) == [
        "litellm-proxy-extras/dist/package.whl",
        "build/output.txt",
        "src/archive.tar.gz",
    ]


def test_find_build_artifact_paths_ignores_source_files():
    module = _load_module()

    paths = [
        ".github/workflows/test-release-hygiene.yml",
        "litellm/proxy/proxy_server.py",
        "tests/test_litellm/test_check_committed_build_artifacts.py",
    ]

    assert module.find_build_artifact_paths(paths) == []
