"""
Experimental provider stub: codex-cli-agent (a.k.a. codex-agent)

Phase 0/1: This registers a CustomLLM that intentionally raises a
501 Not Implemented until the actual runner/streaming is added.

Registration is gated by env var LITELLM_ENABLE_CODEX_AGENT=1 and performed
from litellm.__init__ when the flag is set.

Provider names registered:
- "codex-agent"
- "codex_cli_agent" (alias)
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterator,
    Optional,
    Union,
    cast,
)

import httpx

import litellm
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import EmbeddingResponse, ImageResponse, ModelResponse


PROVIDER_NAME_PRIMARY = "codex-agent"
PROVIDER_NAME_ALIAS = "codex_cli_agent"


class CodexCLIProvider(CustomLLM):
    """Minimal stub provider for Phase 0/1.

    All methods raise 501 Not Implemented to signal the feature gate until
    the spawn/stream logic is implemented in a later phase.
    """

    # ------------ config helpers ------------
    # ------------ concurrency & inflight (class-wide) ------------
    _lock = threading.Lock()
    _current_concurrency: int = 0
    _default_max_concurrency: int = int(
        os.getenv("LITELLM_CODEX_MAX_CONCURRENCY", "0") or 0
    )
    _inflight: dict[str, float] = {}
    _inflight_ttl: float = float(
        os.getenv("LITELLM_CODEX_INFLIGHT_TTL_SECONDS", "30") or 30
    )
    _redis_client = None  # lazy init if REDIS_URL provided and redis installed
    _cache_lock = threading.Lock()
    _cache: dict[str, tuple[str, float]] = {}
    _runs_lock = threading.Lock()
    _runs: dict[str, subprocess.Popen] = {}
    _per_key_counts: dict[str, int] = {}

    @dataclass
    class RunConfig:
        binary_path: str
        args: list[str] = field(default_factory=list)
        env: dict[str, str] = field(default_factory=dict)
        artifacts_root: Path = Path("local/artifacts")
        run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
        first_byte_seconds: float = 10.0
        idle_seconds: float = 60.0
        max_seconds: float = 600.0
        graceful_seconds: float = 3.0
        work_dir: Path = Path("local/artifacts")
        per_line_cap: int = 8192
        transcript_cap: int = 10 * 1024 * 1024
        sanitize_data_urls: bool = True
        # limits / dedupe
        max_concurrency: int = 0  # 0 = unlimited (falls back to class default)
        inflight_ttl: float = 30.0
        cache_ttl: float = 0.0  # 0 = disabled
        use_pty: bool = False

    def _truthy(self, v: Optional[str]) -> bool:
        return str(v).strip() in {"1", "true", "True", "yes", "on"}

    def _build_run_config(  # noqa: PLR0915
        self, optional_params: dict, litellm_params: dict
    ) -> "CodexCLIProvider.RunConfig":
        extra_body = optional_params.get("extra_body", {}) or {}
        # Env overrides allowlist: comma-separated names in env; if absent, allow nothing by default.
        allowlist_env = {
            k.strip()
            for k in os.getenv(
                "LITELLM_CODEX_ENV_ALLOWLIST", "OPENAI_API_KEY,GOOGLE_API_KEY"
            ).split(",")
            if k.strip()
        }
        env_overrides: dict[str, str] = {}
        if isinstance(extra_body.get("env_overrides"), dict):
            env_overrides = {
                k: str(v)
                for k, v in extra_body["env_overrides"].items()
                if k in allowlist_env
            }

        # Base env
        run_env = os.environ.copy()
        run_env.update(env_overrides)

        # Binary resolution (per-request only if allowed) → env → CODEX_HOME → PATH → fallback
        smoke_mode = self._truthy(os.getenv("LITELLM_SMOKE_MODE"))
        allow_req_bin = self._truthy(os.getenv("LITELLM_CODEX_ALLOW_REQUEST_BINARY"))

        def _is_exec(p: Optional[str]) -> bool:
            return bool(p and os.path.isfile(p) and os.access(p, os.X_OK))

        # 1) Per-request override when allowed (smoke or explicit allow)
        binary_path: Optional[str] = None
        candidate_val = extra_body.get("codex_binary_path")
        if (smoke_mode or allow_req_bin) and isinstance(candidate_val, str):
            candidate = candidate_val.strip()
            candidate_abs = os.path.abspath(candidate)
            if _is_exec(candidate_abs):
                binary_path = candidate_abs

        # 2) Server-level absolute binary
        if not binary_path:
            srv_bin = os.getenv("LITELLM_CODEX_BINARY_PATH")
            if _is_exec(srv_bin):
                binary_path = srv_bin  # type: ignore

        # 3) CODEX_HOME: handle both file path or directory containing bin/codex
        if not binary_path:
            codex_home = os.getenv("CODEX_HOME")
            if codex_home:
                codex_home = codex_home.strip()
                # If points directly to an executable
                if _is_exec(codex_home):
                    binary_path = codex_home
                else:
                    # Assume it's a directory; look for <CODEX_HOME>/bin/codex
                    potential = os.path.join(codex_home, "bin", "codex")
                    if _is_exec(potential):
                        binary_path = potential

        # 4) PATH resolution then fallback
        if not binary_path:
            binary_path = which("codex") or "/usr/local/bin/codex"

        if not _is_exec(binary_path):
            raise CustomLLMError(
                status_code=501,
                message=(
                    "Codex CLI binary not found or not executable. "
                    "Set LITELLM_CODEX_BINARY_PATH or CODEX_HOME in your environment, "
                    "or (if allowed) pass extra_body.codex_binary_path; ensure PATH includes 'codex'."
                ),
            )

        # Args (defaults + per-request + mapped CLI flags)
        args: list[str] = []
        default_args_env = os.getenv("LITELLM_CODEX_DEFAULT_ARGS")
        if default_args_env:
            try:
                args.extend(shlex.split(default_args_env, posix=(os.name != "nt")))
            except Exception:
                pass
        if isinstance(extra_body.get("codex_args"), list):
            args.extend([str(x) for x in extra_body["codex_args"]])

        # Map convenience fields to Codex CLI flags (headless)
        # --model
        cli_model = extra_body.get("codex_cli_model")
        if isinstance(cli_model, str) and cli_model.strip():
            args.extend(["--model", cli_model.strip()])
        # --approval-mode
        approval = extra_body.get("codex_approval_mode")
        if isinstance(approval, str) and approval.strip():
            args.extend(["--approval-mode", approval.strip()])
        # --sandbox <level>
        sandbox = extra_body.get("codex_sandbox")
        if isinstance(sandbox, str) and sandbox.strip():
            args.extend(["--sandbox", sandbox.strip()])
        # -i / --image (multiple)
        images = extra_body.get("codex_images")
        if isinstance(images, (list, tuple)):
            for p in images:
                if isinstance(p, str) and p.strip():
                    args.extend(["-i", p.strip()])
        # --config
        cfg_path = extra_body.get("codex_config_path")
        if isinstance(cfg_path, str) and cfg_path.strip():
            args.extend(["--config", cfg_path.strip()])
        # --cd / -C DIR (only if caller wants to override; default working dir is cfg.work_dir)
        cd_dir = extra_body.get("codex_cd")
        if isinstance(cd_dir, str) and cd_dir.strip():
            args.extend(["-C", cd_dir.strip()])
        # --profile
        profile = extra_body.get("codex_profile")
        if isinstance(profile, str) and profile.strip():
            args.extend(["--profile", profile.strip()])
        # --color
        color = extra_body.get("codex_color")
        if isinstance(color, str) and color.strip():
            args.extend(["--color", color.strip()])
        # --json
        if bool(extra_body.get("codex_json")):
            args.append("--json")
        # --output-last-message FILE
        olm = extra_body.get("codex_output_last_message")
        if isinstance(olm, str) and olm.strip():
            args.extend(["--output-last-message", olm.strip()])
        # --skip-git-repo-check
        if bool(extra_body.get("codex_skip_git_repo_check")):
            args.append("--skip-git-repo-check")
        # --oss
        if bool(extra_body.get("codex_oss")):
            args.append("--oss")
        # --full-auto
        if bool(extra_body.get("codex_full_auto")):
            args.append("--full-auto")
        # --no-localhost
        if bool(extra_body.get("codex_no_localhost")):
            args.append("--no-localhost")
        # --verbose
        if bool(extra_body.get("codex_verbose")):
            args.append("--verbose")
        # --yolo (dangerous)
        if bool(extra_body.get("codex_yolo")):
            args.append("--yolo")

        # Apply default guardrails:
        # - Enforce sandbox default to read-only if not specified
        # - Block --yolo unless LITELLM_CODEX_ALLOW_YOLO=1
        # - Downgrade --sandbox danger-full-access unless LITELLM_CODEX_ALLOW_DANGER_FULL_ACCESS=1
        # - Default approval-mode to 'never' if not provided
        allow_yolo = self._truthy(os.getenv("LITELLM_CODEX_ALLOW_YOLO"))
        allow_dfa = self._truthy(os.getenv("LITELLM_CODEX_ALLOW_DANGER_FULL_ACCESS"))

        # Remove yolo flags if not allowed
        if not allow_yolo:
            args = [
                a
                for a in args
                if a not in ("--yolo", "--dangerously-bypass-approvals-and-sandbox")
            ]

        # Ensure sandbox present; track index for value replacement
        sandbox_idx = None
        for i, a in enumerate(args):
            if a in ("--sandbox", "-s") and i + 1 < len(args):
                sandbox_idx = i + 1
                break
        if sandbox_idx is None:
            args.extend(["--sandbox", "read-only"])
            sandbox_idx = len(args) - 1
        else:
            # Downgrade DFA if not allowed
            if not allow_dfa and args[sandbox_idx].strip().lower() in (
                "danger",
                "danger-full-access",
                "danger_full_access",
                "full-access",
                "dangerously-bypass",
            ):
                args[sandbox_idx] = "read-only"

        # Ensure approval-mode present (default: never)
        if "--approval-mode" not in args:
            args.extend(["--approval-mode", "never"])

        # Timeouts
        def _get_float(key: str, env_key: str, default: float) -> float:
            v = extra_body.get(key)
            if isinstance(v, (int, float)):
                return float(v)
            ev = os.getenv(env_key)
            try:
                return float(ev) if ev is not None else default
            except Exception:
                return default

        first_byte = _get_float(
            "codex_first_byte_seconds", "LITELLM_CODEX_FIRST_BYTE_SEC", 10.0
        )
        idle = _get_float("codex_idle_timeout_seconds", "LITELLM_CODEX_IDLE_SEC", 60.0)
        max_run = _get_float("codex_max_run_seconds", "LITELLM_CODEX_MAX_SEC", 600.0)
        graceful = _get_float(
            "codex_graceful_shutdown_seconds", "LITELLM_CODEX_GRACEFUL_SEC", 3.0
        )

        # Artifacts
        artifacts_root = Path(
            extra_body.get("artifacts_root")
            or os.getenv("LITELLM_CODEX_ARTIFACTS_ROOT")
            or "local/artifacts"
        )
        artifacts_root.mkdir(parents=True, exist_ok=True)
        # Workspace selection: persistent by key or ephemeral
        persistence_key = extra_body.get("working_dir_persistence_key")
        if isinstance(persistence_key, str) and persistence_key.strip():
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", persistence_key.strip())[:64]
            work_dir = artifacts_root / "persistent" / safe
        else:
            run_id = f"run_{uuid.uuid4().hex[:8]}"
            work_dir = artifacts_root / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        sanitize = extra_body.get("sanitize_data_urls")
        if sanitize is None:
            sanitize = self._truthy(os.getenv("LITELLM_CODEX_SANITIZE", "1"))

        # Concurrency & inflight
        max_cc = 0
        if isinstance(extra_body.get("codex_max_concurrency"), int):
            max_cc = int(extra_body["codex_max_concurrency"])  # request override
        inflight_ttl = _get_float(
            "inflight_ttl_seconds", "LITELLM_CODEX_INFLIGHT_TTL_SECONDS", 30.0
        )
        cache_ttl = _get_float(
            "cache_ttl_seconds", "LITELLM_CODEX_CACHE_TTL_SECONDS", 0.0
        )
        use_pty = bool(
            extra_body.get("use_pty", False)
            or self._truthy(os.getenv("LITELLM_CODEX_USE_PTY", "0"))
        )

        return CodexCLIProvider.RunConfig(
            binary_path=binary_path,
            args=args,
            env=run_env,
            artifacts_root=artifacts_root,
            work_dir=work_dir,
            first_byte_seconds=first_byte,
            idle_seconds=idle,
            max_seconds=max_run,
            graceful_seconds=graceful,
            sanitize_data_urls=bool(sanitize),
            max_concurrency=max_cc,
            inflight_ttl=inflight_ttl,
            cache_ttl=cache_ttl,
            use_pty=use_pty,
        )

    # ------------ process helpers ------------
    _DATA_URL_RE = re.compile(r"data:[^,]*;base64,([A-Za-z0-9+/=]+)")

    def _sanitize_text(self, s: str, cap: int) -> str:
        if not s:
            return s

        def _repl(m: re.Match[str]) -> str:
            payload = m.group(1)
            return f"data:[sanitized {len(payload)} bytes]"

        s2 = self._DATA_URL_RE.sub(_repl, s)
        if len(s2) > cap:
            return s2[: cap - 12] + " [truncated]"
        return s2

    def _build_stdin_from_messages(self, messages: list) -> Optional[str]:
        try:
            parts: list[str] = []
            for m in messages or []:
                role = (
                    m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                )
                content = (
                    m.get("content")
                    if isinstance(m, dict)
                    else getattr(m, "content", None)
                )
                if role == "user" and content:
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        # content parts may be dicts with type/text
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                parts.append(str(c["text"]))
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
        return None

    def _run_subprocess_stream(  # noqa: PLR0915
        self,
        cfg: "CodexCLIProvider.RunConfig",
        stdin_data: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Spawn subprocess synchronously and yield sanitized stdout lines.

        Applies first-byte, idle, and max-runtime timeouts. Kills process on timeout.
        """
        # Build command: prefer headless 'exec' subcommand and read prompt from stdin ('-')
        # If a user explicitly passed a subcommand via codex_args, we do not filter it out; otherwise add 'exec'.
        cmd = [cfg.binary_path]
        if not any(a in ("exec", "debug", "tui") for a in cfg.args):
            cmd.append("exec")
        cmd += cfg.args
        # Ensure stdin mode by default for prompt input
        if "-" not in cmd:
            cmd.append("-")
        if extra_args:
            cmd.extend(extra_args)
        creationflags = 0
        popen_kwargs: dict = {}
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            popen_kwargs["creationflags"] = creationflags
            popen_kwargs["start_new_session"] = False
        else:
            popen_kwargs["start_new_session"] = True

        use_pty = bool(cfg.use_pty and os.name != "nt")
        # Common queue for both branches; define once to satisfy type checker
        q: queue.Queue[tuple[str, float]]
        # Derive a stable run_id name from work_dir
        work_dir_name: str = cfg.work_dir.name if isinstance(cfg.work_dir, Path) else ""
        if use_pty:
            import pty

            master_fd, slave_fd = pty.openpty()
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                text=True,
                bufsize=1,
                cwd=str(cfg.work_dir),
                env=cfg.env,
                **popen_kwargs,
            )
            # register run
            self._register_run(work_dir_name, proc)

            q = queue.Queue()

            def _pty_reader(fd, qref):
                import os as _os

                try:
                    while True:
                        try:
                            data = _os.read(fd, 1024)
                        except OSError:
                            break
                        if not data:
                            break
                        for line in data.decode(errors="replace").splitlines():
                            qref.put((line, time.time()))
                finally:
                    try:
                        _os.close(fd)
                    except Exception:
                        pass

            t_out = threading.Thread(
                target=_pty_reader, args=(master_fd, q), daemon=True
            )
            t_out.start()
        else:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(cfg.work_dir),
                env=cfg.env,
                **popen_kwargs,
            )
            # register run
            self._register_run(work_dir_name, proc)

            q = queue.Queue()

            def _reader(pipe, qref):
                try:
                    for line in iter(pipe.readline, ""):
                        qref.put((line, time.time()))
                finally:
                    try:
                        pipe.close()
                    except Exception:
                        pass

            t_out = threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True)  # type: ignore[arg-type]
            t_out.start()
        # If we have stdin payload, send it now then close stdin
        if not use_pty and stdin_data is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_data)
                if not stdin_data.endswith("\n"):
                    proc.stdin.write("\n")
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass

        start = time.time()
        last = start
        saw_first = False
        try:
            while True:
                try:
                    line, ts = q.get(timeout=0.1)
                    if not saw_first:
                        saw_first = True
                    last = ts
                    yield self._sanitize_text(line.rstrip("\n"), cfg.per_line_cap)
                except queue.Empty:
                    # Check timeouts
                    now = time.time()
                    if (now - start) > cfg.max_seconds:
                        # Graceful termination by OS
                        if os.name != "nt":
                            try:
                                os.killpg(proc.pid, signal.SIGTERM)
                            except Exception:
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                        else:
                            # Windows: try CTRL_BREAK_EVENT, else terminate
                            try:
                                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                            except Exception:
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                        try:
                            proc.wait(timeout=cfg.graceful_seconds)
                        except Exception:
                            if os.name != "nt":
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            else:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                        break
                    if saw_first:
                        if (now - last) > cfg.idle_seconds:
                            if os.name != "nt":
                                try:
                                    os.killpg(proc.pid, signal.SIGTERM)
                                except Exception:
                                    try:
                                        proc.terminate()
                                    except Exception:
                                        pass
                            else:
                                try:
                                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                                except Exception:
                                    try:
                                        proc.terminate()
                                    except Exception:
                                        pass
                            try:
                                proc.wait(timeout=cfg.graceful_seconds)
                            except Exception:
                                if os.name != "nt":
                                    try:
                                        os.killpg(proc.pid, signal.SIGKILL)
                                    except Exception:
                                        try:
                                            proc.kill()
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            break
                    # If first byte hasn't arrived, keep waiting until max_seconds
                # Exit when process ended, reader finished, and queue drained
                if proc.poll() is not None and q.empty():
                    # if reader thread still alive, give it a brief moment to flush
                    if t_out.is_alive():
                        time.sleep(0.05)
                        continue
                    break
        finally:
            try:
                proc.stdout and proc.stdout.close()
                proc.stderr and proc.stderr.close()
            except Exception:
                pass
            self._deregister_run(work_dir_name)

    # ------------ concurrency / inflight helpers ------------
    def _get_max_concurrency(self, cfg: "CodexCLIProvider.RunConfig") -> int:
        # request override if > 0; else class default; else unlimited (0)
        return (
            cfg.max_concurrency
            if cfg.max_concurrency > 0
            else self._default_max_concurrency
        )

    def _make_dedupe_key(
        self, stdin_payload: Optional[str], cfg: "CodexCLIProvider.RunConfig"
    ) -> str:
        data = {
            "bin": cfg.binary_path,
            "args": cfg.args,
            "input": (stdin_payload or "")[:2048],
        }
        s = json.dumps(data, sort_keys=True)
        import hashlib

        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _cleanup_inflight(self) -> None:
        now = time.time()
        to_del = [
            k for k, ts in self._inflight.items() if now - ts > self._inflight_ttl
        ]
        for k in to_del:
            self._inflight.pop(k, None)

    def _try_redis_acquire(self, key: str, ttl: float) -> Optional[bool]:
        url = os.getenv("LITELLM_CODEX_REDIS_URL")
        if not url:
            return None
        try:
            if self._redis_client is None:
                import redis  # type: ignore

                self._redis_client = redis.Redis.from_url(url)
            # SET key NX EX ttl
            ok = self._redis_client.set(
                f"codex_inflight:{key}", b"1", ex=int(ttl), nx=True
            )
            return bool(ok)
        except Exception:
            return None

    def _get_redis(self):
        url = os.getenv("LITELLM_CODEX_REDIS_URL")
        if not url:
            return None
        try:
            if self._redis_client is None:
                import redis  # type: ignore

                self._redis_client = redis.Redis.from_url(url)
            return self._redis_client
        except Exception:
            return None

    def _cache_get(self, key: str) -> Optional[str]:
        # Try Redis first
        r = self._get_redis()
        if r is not None:
            try:
                val = r.get(f"codex_cache:{key}")
                if val is None:
                    return None
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="replace")
                return str(val)
            except Exception:
                pass
        # Fallback to in-memory
        now = time.time()
        with self._cache_lock:
            tup = self._cache.get(key)
            if not tup:
                return None
            val, exp = tup
            if now > exp:
                self._cache.pop(key, None)
                return None
            return val

    def _cache_set(self, key: str, value: str, ttl: float) -> None:
        if ttl <= 0:
            return
        # Redis first
        r = self._get_redis()
        if r is not None:
            try:
                r.set(f"codex_cache:{key}", value.encode("utf-8"), ex=int(ttl))
                return
            except Exception:
                pass
        # In-memory fallback
        with self._cache_lock:
            self._cache[key] = (value, time.time() + ttl)

    # Per-key concurrency helpers
    def _get_per_key_id(self, optional_params: dict) -> str:
        meta = (
            optional_params.get("metadata", {})
            if isinstance(optional_params, dict)
            else {}
        )
        key = (
            meta.get("team")
            or meta.get("user")
            or optional_params.get("user")
            or "anonymous"
        )
        return str(key)

    def _get_per_key_limit(self, cfg: "CodexCLIProvider.RunConfig") -> int:
        return (
            int(cfg.per_key_max_concurrency)
            if hasattr(cfg, "per_key_max_concurrency")
            else 0
        )

    def _inc_per_key(self, key_id: str, limit: int) -> bool:
        if limit <= 0:
            return True
        with self._lock:
            current = self._per_key_counts.get(key_id, 0)
            if current >= limit:
                return False
            self._per_key_counts[key_id] = current + 1
            return True

    def _dec_per_key(self, key_id: str) -> None:
        with self._lock:
            if key_id in self._per_key_counts and self._per_key_counts[key_id] > 0:
                self._per_key_counts[key_id] -= 1

    def _acquire(self, cfg: "CodexCLIProvider.RunConfig", key: Optional[str]) -> None:
        max_cc = self._get_max_concurrency(cfg)
        with self._lock:
            # Inflight dedupe (optional)
            if key is not None:
                self._cleanup_inflight()
                redis_ok = self._try_redis_acquire(key, cfg.inflight_ttl)
                if redis_ok is False:
                    raise CustomLLMError(
                        status_code=429, message="duplicate run in-flight"
                    )
                if redis_ok is None:
                    # fallback to in-memory
                    if key in self._inflight:
                        raise CustomLLMError(
                            status_code=429, message="duplicate run in-flight"
                        )
                    self._inflight[key] = time.time()

            if max_cc and self._current_concurrency >= max_cc:
                # Saturated
                # Clean up inflight key if we set one in this call
                if key is not None:
                    self._inflight.pop(key, None)
                raise CustomLLMError(
                    status_code=429, message="concurrency limit reached"
                )
            self._current_concurrency += 1

    def _release(self, key: Optional[str]) -> None:
        with self._lock:
            if self._current_concurrency > 0:
                self._current_concurrency -= 1
            if key is not None:
                self._inflight.pop(key, None)

    # ------------ run registry / cancellation ------------
    def _register_run(self, run_id: str, proc: subprocess.Popen) -> None:
        with self._runs_lock:
            self._runs[run_id] = proc

    def _deregister_run(self, run_id: str) -> None:
        with self._runs_lock:
            self._runs.pop(run_id, None)

    def list_runs(self) -> list[str]:
        with self._runs_lock:
            return list(self._runs.keys())

    def cancel_run(self, run_id: str) -> bool:
        with self._runs_lock:
            proc = self._runs.get(run_id)
        if proc is None:
            return False
        try:
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                except Exception:
                    proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass
            return True
        finally:
            self._deregister_run(run_id)

    # ------------ provider interface ------------

    def completion(  # noqa: PLR0915
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> Union[ModelResponse, "litellm.CustomStreamWrapper"]:
        optional_params = optional_params or {}
        cfg = self._build_run_config(
            optional_params=optional_params, litellm_params=litellm_params or {}
        )
        stdin_payload = self._build_stdin_from_messages(messages)
        # Optionally fetch images referenced by URL and attach via -i
        extra_attach_args: list[str] = []
        try:
            eb = optional_params.get("extra_body", {}) or {}
            allow_net = self._truthy(os.getenv("LITELLM_CODEX_ALLOW_NET", "0"))
            fetch_images = bool(eb.get("codex_fetch_images", False)) and allow_net
            if fetch_images and isinstance(messages, list):
                url_re = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
                text = "\n".join(
                    [
                        (m.get("content") if isinstance(m, dict) else "") or ""
                        for m in messages
                    ]
                )
                urls = url_re.findall(text)
                img_urls = [
                    u
                    for u in urls
                    if re.search(r"\.(png|jpe?g|gif|webp)(\?.*)?$", u, re.IGNORECASE)
                ]
                if img_urls:
                    import httpx

                    img_dir = cfg.work_dir / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    timeout_val = float(eb.get("codex_fetch_timeout_seconds", 10.0))
                    with httpx.Client(
                        timeout=timeout_val, follow_redirects=True
                    ) as client:
                        for u in img_urls[:4]:
                            try:
                                r = client.get(u)
                                if r.status_code == 200 and r.content:
                                    basename = re.sub(
                                        r"[^A-Za-z0-9_.-]",
                                        "_",
                                        u.split("/")[-1].split("?")[0],
                                    )
                                    if not re.search(
                                        r"\.(png|jpe?g|gif|webp)$",
                                        basename,
                                        re.IGNORECASE,
                                    ):
                                        ext = "." + (u.split(".")[-1].split("?")[0])
                                        basename = f"img_{uuid.uuid4().hex[:8]}{ext}"
                                    fpath = img_dir / basename
                                    fpath.write_bytes(r.content)
                                    extra_attach_args.extend(["-i", str(fpath)])
                            except Exception:
                                continue
        except Exception:
            pass
        skip_guard = bool(optional_params.get("codex_skip_guard"))
        key = self._make_dedupe_key(stdin_payload, cfg)
        # Cache short-circuit (non-stream)
        if cfg.cache_ttl > 0:
            cached = self._cache_get(key)
            if cached is not None:
                from litellm.types.utils import Message, Choices

                msg = Message(role="assistant", content=cached)
                return ModelResponse(
                    choices=[Choices(message=msg, finish_reason="stop", index=0)]
                )
        # Per-key concurrency check
        per_key_id = self._get_per_key_id(optional_params)
        per_key_limit = self._get_per_key_limit(cfg)
        if per_key_limit and not self._inc_per_key(per_key_id, per_key_limit):
            raise CustomLLMError(
                status_code=429, message="per-key concurrency limit reached"
            )
        if not skip_guard:
            self._acquire(cfg, key)
        try:
            transcript_parts: list[str] = []
            total = 0
            for line in self._run_subprocess_stream(
                cfg, stdin_data=stdin_payload, extra_args=extra_attach_args
            ):
                if not line:
                    continue
                transcript_parts.append(line)
                total += len(line) + 1
                if total >= cfg.transcript_cap:
                    transcript_parts.append("[transcript truncated]")
                    break
            text_out = "\n".join(transcript_parts)
            # Populate basic message content
            from litellm.types.utils import Message, Choices

            msg = Message(role="assistant", content=text_out)
            resp = ModelResponse(
                choices=[Choices(message=msg, finish_reason="stop", index=0)]
            )
            if cfg.cache_ttl > 0 and text_out:
                self._cache_set(key, text_out, cfg.cache_ttl)
            return resp
        finally:
            if not skip_guard:
                self._release(key)
            if per_key_limit:
                self._dec_per_key(per_key_id)

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        optional_params = optional_params or {}
        cfg = self._build_run_config(
            optional_params=optional_params, litellm_params=litellm_params or {}
        )
        stdin_payload = self._build_stdin_from_messages(messages)
        key = self._make_dedupe_key(stdin_payload, cfg)
        self._acquire(cfg, key)
        try:
            for line in self._run_subprocess_stream(cfg, stdin_data=stdin_payload):
                if line is None:
                    continue
                yield GenericStreamingChunk(
                    text=line,
                    tool_use=None,
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                )
        finally:
            self._release(key)

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> Union[
        Coroutine[Any, Any, Union[ModelResponse, "litellm.CustomStreamWrapper"]],
        Union[ModelResponse, "litellm.CustomStreamWrapper"],
    ]:
        # Delegate to sync completion in a thread for Phase 1; avoid double-guard
        def _run_sync(skip_guard: bool) -> ModelResponse:
            op = dict(optional_params or {})
            if skip_guard:
                op["codex_skip_guard"] = True
            return cast(
                ModelResponse,
                self.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=op,
                    acompletion=acompletion,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                ),
            )

        import asyncio

        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        # Apply guard at async wrapper level
        cfg = self._build_run_config(
            optional_params=optional_params or {}, litellm_params=litellm_params or {}
        )
        stdin_payload = self._build_stdin_from_messages(messages)
        key = self._make_dedupe_key(stdin_payload, cfg)
        # Per-key concurrency check
        per_key_id = self._get_per_key_id(optional_params or {})
        per_key_limit = self._get_per_key_limit(cfg)
        if per_key_limit and not self._inc_per_key(per_key_id, per_key_limit):
            raise CustomLLMError(
                status_code=429, message="per-key concurrency limit reached"
            )
        self._acquire(cfg, key)
        try:
            if loop and loop.is_running():
                import functools

                return await loop.run_in_executor(
                    None, functools.partial(_run_sync, True)
                )
            else:
                return _run_sync(True)
        finally:
            self._release(key)
            if per_key_limit:
                self._dec_per_key(per_key_id)

    async def astreaming(  # type: ignore[misc, override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict = {},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        # Delegate to sync streaming wrapped into async generator
        async def _gen() -> AsyncIterator[GenericStreamingChunk]:
            cfg = self._build_run_config(
                optional_params=optional_params or {},
                litellm_params=litellm_params or {},
            )
            stdin_payload = self._build_stdin_from_messages(messages)
            # Use a thread to iterate sync generator and push into asyncio queue
            q: "asyncio.Queue[Optional[GenericStreamingChunk]]"  # type: ignore
            import asyncio

            loop = asyncio.get_running_loop()
            q = asyncio.Queue()

            key = self._make_dedupe_key(stdin_payload, cfg)
            # acquire before starting worker
            self._acquire(cfg, key)

            def _worker():
                try:
                    for line in self._run_subprocess_stream(
                        cfg, stdin_data=stdin_payload
                    ):
                        chunk = GenericStreamingChunk(
                            text=line,
                            tool_use=None,
                            is_finished=False,
                            finish_reason="",
                            usage=None,
                        )
                        asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop)
                    self._release(key)

            th = threading.Thread(target=_worker, daemon=True)
            th.start()
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item

        async for c in _gen():
            yield c

    def image_generation(
        self,
        model: str,
        prompt: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model_response: ImageResponse,
        optional_params: dict,
        logging_obj: Any,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ImageResponse:
        raise CustomLLMError(
            status_code=501, message="Image generation not supported by codex-agent"
        )

    async def aimage_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        optional_params: dict = {},
        logging_obj: Any = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ImageResponse:
        raise CustomLLMError(
            status_code=501, message="Image generation not supported by codex-agent"
        )

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        raise CustomLLMError(
            status_code=501, message="Embedding not supported by codex-agent"
        )

    async def aembedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        raise CustomLLMError(
            status_code=501, message="Embedding not supported by codex-agent"
        )


def register() -> None:
    """Register the provider into litellm.custom_provider_map (Phase 0/1).

    Idempotent: calling multiple times won’t duplicate entries.
    """
    names = {PROVIDER_NAME_PRIMARY, PROVIDER_NAME_ALIAS}
    # Avoid duplicate registrations
    existing = {item["provider"] for item in litellm.custom_provider_map}
    to_add = [name for name in names if name not in existing]
    if not to_add:
        return
    handler = CodexCLIProvider()
    for name in to_add:
        litellm.custom_provider_map.append(
            {
                "provider": name,
                "custom_handler": handler,
            }
        )
