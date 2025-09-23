import os
import socket
import httpx

TIMEOUT = int(os.getenv("NDSMOKE_TIMEOUT", "240"))
import pytest


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _api():
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    return host, port


def _skip_if_unavailable(tool: str):
    host, port = _api()
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping live docker ndsmoke")
    if not _can_connect(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")
    # quick which via auto-run python
    payload = {
        "messages": [
            {"role": "system", "content": "Reply with only a python code block that prints the path to a tool."},
            {"role": "user", "content": f"Use Python to check if '{tool}' is on PATH: print(__import__('shutil').which('{tool}'))."},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 1,
        "enable_repair": False,
    }
    url = f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Treat missing tool as skip; final answer contains path or 'None'
    messages = data.get("messages", [])
    joined = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    if "None" in joined or joined.strip() == "":
        pytest.skip(f"tool '{tool}' not available in container")


@pytest.mark.ndsmoke
def test_lang_python_live_optional():
    _skip_if_unavailable("python3")
    host, port = _api()
    prompt = ("""Implement a Python function compress_runs(s: str) -> str that compresses runs of repeated characters like aaabbc -> a3b2c1. Return only Python code in a code block and in __main__ print two tests.""")
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python using tools. If tests fail, fix and try again."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True
    assert isinstance((data.get("final_answer") or "").strip(), str)


@pytest.mark.ndsmoke
def test_lang_javascript_live_optional():
    _skip_if_unavailable("node")
    host, port = _api()
    prompt = (
        """Implement a Python function `parse_csv(line: str) -> list[str]` that parses a single CSV line per RFC 4180: commas separate fields; quotes can wrap fields; double quote inside quoted field is escaped by doubling. Handle empty fields and trailing commas. Return only python code in a code block, and in `__main__` print: parse_csv("a,b,c"); parse_csv(""hello, world"",x,"y""); parse_csv(",,");""",
        " For example 'aaabbc' -> 'a3b2c1'. Handle empty strings and Unicode. Return only python code, then run tests"
        " that print compress_runs('xxxyzzzz') and compress_runs('') to stdout."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to run Node."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_c_live_optional():
    _skip_if_unavailable("gcc")
    host, port = _api()
    prompt = (
        "Write a Python script that writes a C program rotate.c that rotates an N x N matrix 90 degrees clockwise in-place. The program should read N on the first line, then N lines of N integers, rotate, and print the rotated matrix. Treat warnings as errors."
        " Compile with `gcc -O2 -Wall -Werror rotate.c -o rotate` and run it. Print outputs. Return only python code."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to compile and run C."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_cpp_live_optional():
    _skip_if_unavailable("g++")
    host, port = _api()
    prompt = (
        "Write a Python script that writes a C++20 program lru.cpp that implements an LRU cache class template LRU<K,V> with get/put and capacity. In main(), exercise the cache and print hits/misses."
        " Compile with `g++ -O2 -std=c++20 -Wall -Werror lru.cpp -o lru` and run it. Return only python code."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to compile and run C++."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_go_live_optional():
    _skip_if_unavailable("go")
    host, port = _api()
    prompt = (
        "Write a Python script that writes a Go program main.go that merges K sorted integer slices concurrently using goroutines and channels, and prints the merged result. Then run `go run main.go`."
        " Return only python code, and surface stdout/stderr."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to run Go."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_java_live_optional():
    _skip_if_unavailable("javac")
    host, port = _api()
    prompt = (
        "Write a Python script that writes a Java program Main.java implementing a Trie with insert/search/startsWith, then prints a few queries. Compile with javac and run with `java Main`."
        " Return only python code."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to compile and run Java."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_rust_live_optional():
    _skip_if_unavailable("rustc")
    host, port = _api()
    prompt = (
        "Write a Python script that writes a Rust program main.rs that reads multiple lines from stdin and prints a histogram of word frequencies (case-insensitive, stripping punctuation). Use a HashMap; compile with rustc and run it."
        " Return only python code."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to compile and run Rust."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True


@pytest.mark.ndsmoke
def test_lang_asm_live_optional():
    _skip_if_unavailable("nasm")
    host, port = _api()
    prompt = (
        "Write a Python script that writes an x86_64 Linux assembly program sum.asm (NASM) that reads newline-separated integers from stdin using syscalls, computes the sum, and prints it. Assemble with `nasm -felf64 sum.asm -o sum.o`, link with `ld -o sum sum.o`, and run it."
        " Return only python code."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python; inside Python use subprocess to assemble and run."},
            {"role": "user", "content": prompt},
        ],
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/granite3.3:8b"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url = f"http://{host}:{port}/agent/run"
    data = httpx.post(url, json=payload, timeout=TIMEOUT).json()
    assert data.get("ok") is True