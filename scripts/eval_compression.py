"""
Prompt Compression Evaluation Harness
======================================
Compare model performance on coding tasks with and without prompt compression.

Usage:
    python scripts/eval_compression.py --model gpt-4o --problems 5
    python scripts/eval_compression.py --model claude-sonnet-4-20250514 --problems 12 --runs 3
    python scripts/eval_compression.py --model gpt-4o-mini --padding-factor 50

The harness runs each problem in two modes:
  1. **baseline** — raw prompt sent directly to the model.
  2. **compressed** — prompt is padded with distractor context, then
     ``litellm.compress()`` removes the noise before sending.

This measures whether compression preserves the signal the model needs
to solve the task while reducing token usage.

Set --padding-factor to control how much distractor context is injected
(higher = more tokens to compress away).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import litellm
from litellm.types.utils import CallTypes

# ---------------------------------------------------------------------------
# Problem definitions (HumanEval-style)
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "id": "has_close_elements",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def has_close_elements(numbers: List[float], threshold: float) -> bool:
                \"\"\"Check if in given list of numbers, are any two numbers closer to each other than
                given threshold.
                >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
                False
                >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
                True
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
            assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
            assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
            assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
            assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True
            assert has_close_elements([], 0.5) == False
            print("PASSED")
        """
        ),
    },
    {
        "id": "separate_paren_groups",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def separate_paren_groups(paren_string: str) -> List[str]:
                \"\"\"Input to this function is a string containing multiple groups of nested parentheses.
                Your goal is to separate those groups into separate strings and return the list of those.
                Separate groups are balanced (each open brace is properly closed) and not nested within each other.
                Ignore any spaces in the input string.
                >>> separate_paren_groups('( ) (( )) (( )( ))')
                ['()', '(())', '(()())']
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
            assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
            assert separate_paren_groups('(()(()))') == ['(()(()))']
            assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
            print("PASSED")
        """
        ),
    },
    {
        "id": "truncate_number",
        "prompt": textwrap.dedent(
            """\
            def truncate_number(number: float) -> float:
                \"\"\"Given a positive floating point number, it can be decomposed into
                an integer part (largest integer smaller than given number) and decimals
                (leftover part always smaller than 1).
                Return the decimal part of the number.
                >>> truncate_number(3.5)
                0.5
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert truncate_number(3.5) == 0.5
            assert abs(truncate_number(1.33) - 0.33) < 1e-6
            assert abs(truncate_number(123.456) - 0.456) < 1e-6
            print("PASSED")
        """
        ),
    },
    {
        "id": "below_zero",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def below_zero(operations: List[int]) -> bool:
                \"\"\"You're given a list of deposit and withdrawal operations on a bank account that starts with
                zero balance. Your task is to detect if at any point the balance of account falls below zero, and
                at that point function should return True. Otherwise it should return False.
                >>> below_zero([1, 2, 3])
                False
                >>> below_zero([1, 2, -4, 5])
                True
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert below_zero([]) == False
            assert below_zero([1, 2, -3, 1, 2, -3]) == False
            assert below_zero([1, 2, -4, 5, 6]) == True
            assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False
            assert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) == True
            assert below_zero([1, -2]) == True
            print("PASSED")
        """
        ),
    },
    {
        "id": "mean_absolute_deviation",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def mean_absolute_deviation(numbers: List[float]) -> float:
                \"\"\"For a given list of input numbers, calculate Mean Absolute Deviation
                around the mean of this dataset.
                Mean Absolute Deviation is the average absolute difference between each
                element and a centerpoint (mean in this case):
                MAD = average | x - x_mean |
                >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
                1.0
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
            assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6
            assert abs(mean_absolute_deviation([1.0, 1.0, 1.0, 1.0]) - 0.0) < 1e-6
            print("PASSED")
        """
        ),
    },
    {
        "id": "intersperse",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def intersperse(numbers: List[int], delimiter: int) -> List[int]:
                \"\"\"Insert a number 'delimiter' between every two consecutive elements of input list `numbers`.
                >>> intersperse([], 4)
                []
                >>> intersperse([1, 2, 3], 4)
                [1, 4, 2, 4, 3]
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert intersperse([], 7) == []
            assert intersperse([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]
            assert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]
            print("PASSED")
        """
        ),
    },
    {
        "id": "parse_nested_parens",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def parse_nested_parens(paren_string: str) -> List[int]:
                \"\"\"Input to this function is a string represented multiple groups of nested parentheses separated by spaces.
                For each of the groups, output the deepest level of nesting of parentheses.
                E.g. (()()) has maximum two levels of nesting while ((())) has three.
                >>> parse_nested_parens('(()()) ((())) () ((())())')
                [2, 3, 1, 3]
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert parse_nested_parens('(()()) ((())) () ((())())') == [2, 3, 1, 3]
            assert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]
            assert parse_nested_parens('(()(())((())))') == [4]
            print("PASSED")
        """
        ),
    },
    {
        "id": "filter_by_substring",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def filter_by_substring(strings: List[str], substring: str) -> List[str]:
                \"\"\"Filter an input list of strings only for ones that contain given substring.
                >>> filter_by_substring([], 'a')
                []
                >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
                ['abc', 'bacd', 'array']
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert filter_by_substring([], 'john') == []
            assert filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxuj', 'xxx'], 'xxx') == ['xxx', 'xxxuj', 'xxx']
            assert filter_by_substring(['xxx', 'asd', 'aaber', 'john doe', 'xxxuj', 'xxx'], 'xx') == ['xxx', 'xxxuj', 'xxx']
            assert filter_by_substring(['grunt', 'hierarchial', 'abc', 'hierarchial'], 'hi') == ['hierarchial', 'hierarchial']
            print("PASSED")
        """
        ),
    },
    {
        "id": "sum_product",
        "prompt": textwrap.dedent(
            """\
            from typing import List, Tuple

            def sum_product(numbers: List[int]) -> Tuple[int, int]:
                \"\"\"For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
                Empty sum should be equal to 0 and empty product should be equal to 1.
                >>> sum_product([])
                (0, 1)
                >>> sum_product([1, 2, 3, 4])
                (10, 24)
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert sum_product([]) == (0, 1)
            assert sum_product([1, 1, 1]) == (3, 1)
            assert sum_product([100, 0]) == (100, 0)
            assert sum_product([3, 5, 7]) == (15, 105)
            assert sum_product([10]) == (10, 10)
            print("PASSED")
        """
        ),
    },
    {
        "id": "max_element",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def max_element(l: List[int]) -> int:
                \"\"\"Return maximum element in the list.
                >>> max_element([1, 2, 3])
                3
                >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
                123
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert max_element([1, 2, 3]) == 3
            assert max_element([5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10]) == 124
            assert max_element([-1, -2, -3]) == -1
            print("PASSED")
        """
        ),
    },
    {
        "id": "fizz_buzz",
        "prompt": textwrap.dedent(
            """\
            def fizz_buzz(n: int) -> int:
                \"\"\"Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
                >>> fizz_buzz(50)
                0
                >>> fizz_buzz(78)
                2
                >>> fizz_buzz(79)
                3
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert fizz_buzz(50) == 0
            assert fizz_buzz(78) == 2
            assert fizz_buzz(79) == 3
            assert fizz_buzz(100) == 3
            assert fizz_buzz(200) == 6
            assert fizz_buzz(4000) == 192
            print("PASSED")
        """
        ),
    },
    {
        "id": "sort_by_binary_len",
        "prompt": textwrap.dedent(
            """\
            from typing import List

            def sort_array(arr: List[int]) -> List[int]:
                \"\"\"Sort an array of non-negative integers according to number of ones in their binary
                representation in ascending order. For equal number of ones, sort based on decimal value.
                >>> sort_array([1, 5, 2, 3, 4])
                [1, 2, 4, 3, 5]
                >>> sort_array([-2, -3, -4, -5, -6])
                [-6, -5, -4, -3, -2]
                >>> sort_array([1, 0, 2, 3, 4])
                [0, 1, 2, 4, 3]
                \"\"\"
        """
        ),
        "tests": textwrap.dedent(
            """\
            assert sort_array([1, 5, 2, 3, 4]) == [1, 2, 4, 3, 5]
            assert sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
            assert sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 4, 3]
            assert sort_array([]) == []
            assert sort_array([2, 5, 77, 4, 5, 3, 5, 7, 2, 3, 4]) == [2, 2, 4, 4, 3, 3, 5, 5, 5, 7, 77]
            assert sort_array([3, 6, 44, 12, 32, 5]) == [32, 3, 5, 6, 12, 44]
            print("PASSED")
        """
        ),
    },
]

# Distractor code snippets injected as prior conversation context.
# These are plausible but irrelevant to the actual task, forcing the
# compressor to identify and drop them.
DISTRACTOR_SNIPPETS = [
    # distractor 0 — database connection pool
    textwrap.dedent(
        """\
        # db_pool.py
        import threading
        from contextlib import contextmanager

        class ConnectionPool:
            def __init__(self, dsn, min_size=2, max_size=10):
                self._dsn = dsn
                self._min_size = min_size
                self._max_size = max_size
                self._pool = []
                self._lock = threading.Lock()
                self._initialize()

            def _initialize(self):
                for _ in range(self._min_size):
                    self._pool.append(self._create_connection())

            def _create_connection(self):
                import psycopg2
                return psycopg2.connect(self._dsn)

            @contextmanager
            def acquire(self):
                conn = self._checkout()
                try:
                    yield conn
                finally:
                    self._checkin(conn)

            def _checkout(self):
                with self._lock:
                    if self._pool:
                        return self._pool.pop()
                    if len(self._pool) < self._max_size:
                        return self._create_connection()
                raise RuntimeError("Pool exhausted")

            def _checkin(self, conn):
                with self._lock:
                    self._pool.append(conn)

            def close_all(self):
                with self._lock:
                    for conn in self._pool:
                        conn.close()
                    self._pool.clear()
    """
    ),
    # distractor 1 — HTTP retry logic
    textwrap.dedent(
        """\
        # http_retry.py
        import time
        import random
        import requests
        from functools import wraps

        class RetryConfig:
            def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
                self.max_retries = max_retries
                self.base_delay = base_delay
                self.max_delay = max_delay
                self.backoff_factor = backoff_factor

        def retry_with_backoff(config=None):
            if config is None:
                config = RetryConfig()

            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    last_exception = None
                    for attempt in range(config.max_retries + 1):
                        try:
                            return func(*args, **kwargs)
                        except (requests.ConnectionError, requests.Timeout) as e:
                            last_exception = e
                            if attempt == config.max_retries:
                                break
                            delay = min(
                                config.base_delay * (config.backoff_factor ** attempt),
                                config.max_delay
                            )
                            jitter = random.uniform(0, delay * 0.1)
                            time.sleep(delay + jitter)
                    raise last_exception
                return wrapper
            return decorator

        @retry_with_backoff(RetryConfig(max_retries=5))
        def fetch_data(url, params=None):
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
    """
    ),
    # distractor 2 — LRU cache implementation
    textwrap.dedent(
        """\
        # lru_cache.py
        from collections import OrderedDict
        from threading import RLock

        class LRUCache:
            def __init__(self, capacity=128):
                self._capacity = capacity
                self._cache = OrderedDict()
                self._lock = RLock()
                self._hits = 0
                self._misses = 0

            def get(self, key, default=None):
                with self._lock:
                    if key in self._cache:
                        self._cache.move_to_end(key)
                        self._hits += 1
                        return self._cache[key]
                    self._misses += 1
                    return default

            def put(self, key, value):
                with self._lock:
                    if key in self._cache:
                        self._cache.move_to_end(key)
                    self._cache[key] = value
                    if len(self._cache) > self._capacity:
                        self._cache.popitem(last=False)

            def delete(self, key):
                with self._lock:
                    self._cache.pop(key, None)

            def clear(self):
                with self._lock:
                    self._cache.clear()

            @property
            def stats(self):
                total = self._hits + self._misses
                hit_rate = self._hits / total if total else 0.0
                return {"hits": self._hits, "misses": self._misses, "hit_rate": hit_rate}

            def __len__(self):
                return len(self._cache)

            def __contains__(self, key):
                return key in self._cache
    """
    ),
    # distractor 3 — CSV report generator
    textwrap.dedent(
        """\
        # report_gen.py
        import csv
        import io
        from datetime import datetime, timedelta

        class ReportGenerator:
            def __init__(self, title, columns):
                self.title = title
                self.columns = columns
                self.rows = []

            def add_row(self, **kwargs):
                row = {col: kwargs.get(col, "") for col in self.columns}
                self.rows.append(row)

            def to_csv(self):
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(self.rows)
                return output.getvalue()

            def summary(self):
                numeric_cols = []
                for col in self.columns:
                    try:
                        vals = [float(r[col]) for r in self.rows if r[col] != ""]
                        if vals:
                            numeric_cols.append({
                                "column": col,
                                "min": min(vals),
                                "max": max(vals),
                                "mean": sum(vals) / len(vals),
                                "count": len(vals),
                            })
                    except (ValueError, TypeError):
                        continue
                return numeric_cols

            def filter_rows(self, predicate):
                gen = ReportGenerator(self.title, self.columns)
                gen.rows = [r for r in self.rows if predicate(r)]
                return gen

            def date_range_report(self, date_col, start, end):
                def in_range(row):
                    try:
                        d = datetime.fromisoformat(row[date_col])
                        return start <= d <= end
                    except (ValueError, KeyError):
                        return False
                return self.filter_rows(in_range)
    """
    ),
    # distractor 4 — async task queue
    textwrap.dedent(
        """\
        # task_queue.py
        import asyncio
        import logging
        from dataclasses import dataclass, field
        from enum import Enum
        from typing import Any, Callable, Coroutine

        logger = logging.getLogger(__name__)

        class TaskStatus(Enum):
            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        @dataclass
        class Task:
            id: str
            func: Callable[..., Coroutine]
            args: tuple = ()
            kwargs: dict = field(default_factory=dict)
            status: TaskStatus = TaskStatus.PENDING
            result: Any = None
            error: str = ""
            retries: int = 0
            max_retries: int = 3

        class AsyncTaskQueue:
            def __init__(self, concurrency=5):
                self._queue = asyncio.Queue()
                self._concurrency = concurrency
                self._tasks = {}
                self._workers = []

            async def submit(self, task: Task):
                self._tasks[task.id] = task
                await self._queue.put(task)

            async def _worker(self):
                while True:
                    task = await self._queue.get()
                    task.status = TaskStatus.RUNNING
                    try:
                        task.result = await task.func(*task.args, **task.kwargs)
                        task.status = TaskStatus.COMPLETED
                    except Exception as e:
                        task.retries += 1
                        if task.retries <= task.max_retries:
                            task.status = TaskStatus.PENDING
                            await self._queue.put(task)
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            logger.error(f"Task {task.id} failed: {e}")
                    finally:
                        self._queue.task_done()

            async def start(self):
                self._workers = [
                    asyncio.create_task(self._worker())
                    for _ in range(self._concurrency)
                ]

            async def wait(self):
                await self._queue.join()

            async def shutdown(self):
                for w in self._workers:
                    w.cancel()
    """
    ),
    # distractor 5 — config parser with env var interpolation
    textwrap.dedent(
        """\
        # config_parser.py
        import os
        import re
        import json
        from pathlib import Path

        _ENV_PATTERN = re.compile(r'\\$\\{([A-Z_][A-Z0-9_]*)(?::-(.*?))?\\}')

        class ConfigError(Exception):
            pass

        class Config:
            def __init__(self, data=None):
                self._data = data or {}

            @classmethod
            def from_file(cls, path):
                p = Path(path)
                if not p.exists():
                    raise ConfigError(f"Config file not found: {path}")
                with open(p) as f:
                    raw = json.load(f)
                return cls(cls._interpolate(raw))

            @classmethod
            def _interpolate(cls, obj):
                if isinstance(obj, str):
                    return cls._interpolate_string(obj)
                if isinstance(obj, dict):
                    return {k: cls._interpolate(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [cls._interpolate(item) for item in obj]
                return obj

            @classmethod
            def _interpolate_string(cls, s):
                def replacer(match):
                    var_name = match.group(1)
                    default = match.group(2)
                    value = os.environ.get(var_name)
                    if value is None:
                        if default is not None:
                            return default
                        raise ConfigError(f"Required env var {var_name} is not set")
                    return value
                return _ENV_PATTERN.sub(replacer, s)

            def get(self, key, default=None):
                keys = key.split(".")
                obj = self._data
                for k in keys:
                    if isinstance(obj, dict) and k in obj:
                        obj = obj[k]
                    else:
                        return default
                return obj

            def require(self, key):
                val = self.get(key)
                if val is None:
                    raise ConfigError(f"Required config key missing: {key}")
                return val
    """
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    problem_id: str
    mode: str  # "baseline" or "compressed"
    passed: bool
    generated_code: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    compression_ratio: float = 0.0
    error: str = ""


@dataclass
class BenchmarkReport:
    model: str
    timestamp: str
    num_problems: int
    num_runs: int
    padding_factor: int
    baseline: dict = field(default_factory=dict)
    compressed: dict = field(default_factory=dict)
    per_problem: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM caller (uses litellm)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a Python coding assistant. Complete the function below. "
    "Return ONLY the Python code (the complete function), no explanation, "
    "no markdown fences."
)


def call_llm(model: str, messages: list[dict]) -> dict:
    """Call model via litellm. Returns dict with response text and usage."""
    t0 = time.time()
    resp = litellm.completion(
        model=model, messages=messages, temperature=0.0, max_tokens=2048
    )
    latency_ms = (time.time() - t0) * 1000

    text = resp.choices[0].message.content or ""
    usage = resp.usage

    return {
        "text": text,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# Code extraction & execution
# ---------------------------------------------------------------------------


def extract_code(raw: str) -> str:
    """Pull code out of the LLM response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def run_tests(code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute generated code + tests in a subprocess. Returns (passed, error_msg)."""
    full = code + "\n\n" + tests
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0 and "PASSED" in result.stdout:
                return True, ""
            err = result.stderr.strip() or result.stdout.strip()
            return False, err[:500]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# Context building — pad the prompt with distractors
# ---------------------------------------------------------------------------


def build_messages(
    problem: dict,
    padding_factor: int = 0,
) -> list[dict]:
    """
    Build a message list for a problem.

    When ``padding_factor`` > 0, distractor code snippets are injected as
    prior user messages (simulating a long coding session) so there is
    enough context for compression to act on.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_MSG}]

    if padding_factor > 0:
        for i in range(padding_factor):
            snippet = DISTRACTOR_SNIPPETS[i % len(DISTRACTOR_SNIPPETS)]
            messages.append(
                {
                    "role": "user",
                    "content": f"Here is some code from our codebase:\n\n{snippet}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Got it, I've reviewed that code. What would you like me to help with?",
                }
            )

    messages.append(
        {
            "role": "user",
            "content": (
                "Complete the following Python function. Return ONLY the code.\n\n"
                + problem["prompt"]
            ),
        }
    )
    return messages


# ---------------------------------------------------------------------------
# Single problem evaluation
# ---------------------------------------------------------------------------


def eval_problem(
    problem: dict,
    model: str,
    padding_factor: int,
    use_compression: bool,
    compression_trigger: int,
    embedding_model: Optional[str],
) -> RunResult:
    """Evaluate a single problem in either baseline or compressed mode."""
    mode = "compressed" if use_compression else "baseline"
    messages = build_messages(problem, padding_factor=padding_factor)

    compression_ratio = 0.0

    if use_compression:
        result = litellm.compress(
            messages=messages,
            model=model,
            call_type=CallTypes.completion,
            compression_trigger=compression_trigger,
            embedding_model=embedding_model,
        )
        messages = result["messages"]
        compression_ratio = result["compression_ratio"]

    try:
        resp = call_llm(model, messages)
        code = extract_code(resp["text"])
        passed, error = run_tests(code, problem["tests"])

        return RunResult(
            problem_id=problem["id"],
            mode=mode,
            passed=passed,
            generated_code=code,
            prompt_tokens=resp["prompt_tokens"],
            completion_tokens=resp["completion_tokens"],
            total_tokens=resp["total_tokens"],
            latency_ms=resp["latency_ms"],
            compression_ratio=compression_ratio,
            error=error,
        )
    except Exception as e:
        return RunResult(
            problem_id=problem["id"],
            mode=mode,
            passed=False,
            generated_code="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0,
            compression_ratio=compression_ratio,
            error=str(e)[:500],
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[RunResult]) -> dict:
    """Compute aggregate stats from a list of RunResults."""
    if not results:
        return {}
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    return {
        "pass_rate": round(passed / total * 100, 1),
        "passed": passed,
        "total": total,
        "avg_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in results)),
        "avg_completion_tokens": round(
            statistics.mean(r.completion_tokens for r in results)
        ),
        "avg_total_tokens": round(statistics.mean(r.total_tokens for r in results)),
        "avg_latency_ms": round(statistics.mean(r.latency_ms for r in results), 1),
        "median_latency_ms": round(statistics.median(r.latency_ms for r in results), 1),
        "avg_compression_ratio": round(
            statistics.mean(r.compression_ratio for r in results), 4
        ),
    }


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


def run_benchmark(
    model: str,
    num_problems: int = 0,
    num_runs: int = 1,
    padding_factor: int = 20,
    compression_trigger: int = 2000,
    embedding_model: Optional[str] = None,
) -> dict:
    """
    Run the full benchmark.

    Parameters:
        model: LLM model name (litellm format).
        num_problems: How many problems to run (0 = all).
        num_runs: Number of runs per mode.
        padding_factor: How many distractor snippets to inject. Each snippet
            adds ~400-600 tokens. 20 snippets ≈ 10k tokens of noise.
        compression_trigger: Token count above which compression activates.
        embedding_model: Optional embedding model for semantic scoring.
    """
    problems = PROBLEMS[:num_problems] if num_problems > 0 else PROBLEMS

    print(f"\n{'=' * 60}")
    print("Prompt Compression Eval Harness")
    print(f"{'=' * 60}")
    print(f"Model:              {model}")
    print(f"Problems:           {len(problems)}")
    print(f"Runs per mode:      {num_runs}")
    print(f"Padding factor:     {padding_factor}")
    print(f"Compression trigger:{compression_trigger} tokens")
    print(f"Embedding model:    {embedding_model or 'None (BM25 only)'}")
    print(f"{'=' * 60}\n")

    baseline_results: list[RunResult] = []
    compressed_results: list[RunResult] = []

    for run_i in range(num_runs):
        if num_runs > 1:
            print(f"--- Run {run_i + 1}/{num_runs} ---")

        for p in problems:
            # Baseline (with padding, but no compression)
            print(f"  [{p['id']}] baseline ... ", end="", flush=True)
            r = eval_problem(
                p,
                model,
                padding_factor=padding_factor,
                use_compression=False,
                compression_trigger=compression_trigger,
                embedding_model=embedding_model,
            )
            baseline_results.append(r)
            print("PASS" if r.passed else f"FAIL ({r.error[:60]})")

            # Compressed
            print(f"  [{p['id']}] compressed ... ", end="", flush=True)
            r = eval_problem(
                p,
                model,
                padding_factor=padding_factor,
                use_compression=True,
                compression_trigger=compression_trigger,
                embedding_model=embedding_model,
            )
            compressed_results.append(r)
            status = "PASS" if r.passed else f"FAIL ({r.error[:60]})"
            print(f"{status}  (ratio: {r.compression_ratio:.2%})")

    # Aggregate
    base_agg = aggregate(baseline_results)
    comp_agg = aggregate(compressed_results)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Baseline (with {padding_factor} distractor snippets, no compression):")
    print(
        f"    Pass rate:         {base_agg['pass_rate']}% ({base_agg['passed']}/{base_agg['total']})"
    )
    print(f"    Avg prompt tokens: {base_agg['avg_prompt_tokens']}")
    print(f"    Avg total tokens:  {base_agg['avg_total_tokens']}")
    print(f"    Avg latency:       {base_agg['avg_latency_ms']}ms")

    print(f"\n  Compressed (litellm.compress → then call model):")
    print(
        f"    Pass rate:         {comp_agg['pass_rate']}% ({comp_agg['passed']}/{comp_agg['total']})"
    )
    print(f"    Avg prompt tokens: {comp_agg['avg_prompt_tokens']}")
    print(f"    Avg total tokens:  {comp_agg['avg_total_tokens']}")
    print(f"    Avg latency:       {comp_agg['avg_latency_ms']}ms")
    print(f"    Avg compression:   {comp_agg['avg_compression_ratio']:.2%}")

    token_savings = base_agg["avg_prompt_tokens"] - comp_agg["avg_prompt_tokens"]
    token_pct = (
        round(token_savings / base_agg["avg_prompt_tokens"] * 100, 1)
        if base_agg["avg_prompt_tokens"]
        else 0
    )
    latency_diff = base_agg["avg_latency_ms"] - comp_agg["avg_latency_ms"]
    pass_diff = comp_agg["pass_rate"] - base_agg["pass_rate"]

    print(f"\n  Delta (compressed vs baseline):")
    print(f"    Token savings:     {token_savings} tokens ({token_pct}%)")
    print(f"    Latency delta:     {latency_diff:+.1f}ms")
    print(f"    Pass rate delta:   {pass_diff:+.1f}%")

    # Save JSON report
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"eval_report_{ts}.json"
    report = {
        "model": model,
        "timestamp": ts,
        "num_problems": len(problems),
        "num_runs": num_runs,
        "padding_factor": padding_factor,
        "compression_trigger": compression_trigger,
        "embedding_model": embedding_model,
        "baseline": base_agg,
        "compressed": comp_agg,
        "baseline_results": [asdict(r) for r in baseline_results],
        "compressed_results": [asdict(r) for r in compressed_results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt Compression Evaluation Harness"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="Model name (litellm format)"
    )
    parser.add_argument(
        "--problems", type=int, default=0, help="Number of problems (0 = all)"
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per mode")
    parser.add_argument(
        "--padding-factor",
        type=int,
        default=20,
        help="Number of distractor snippets to inject (default: 20, ~10k tokens)",
    )
    parser.add_argument(
        "--compression-trigger",
        type=int,
        default=2000,
        help="Token count threshold to trigger compression (default: 2000)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model for semantic scoring (e.g. text-embedding-3-small)",
    )
    args = parser.parse_args()

    run_benchmark(
        model=args.model,
        num_problems=args.problems,
        num_runs=args.runs,
        padding_factor=args.padding_factor,
        compression_trigger=args.compression_trigger,
        embedding_model=args.embedding_model,
    )
