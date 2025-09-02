import asyncio
import json
import os
import time
import uuid
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, UserAPIKeyAuth


# Paths and default config
ARCAGI_ROOT = os.environ.get("ARCAGI_DATA_ROOT", "/data/arcagi")
ARCAGI_CONFIG_PATH = os.path.join(ARCAGI_ROOT, "config.json")
ARCAGI_DATASETS_DIR = os.path.join(ARCAGI_ROOT, "datasets")
ARCAGI_REPORTS_DIR = os.path.join(ARCAGI_ROOT, "reports")

router = APIRouter()
logger = logging.getLogger("litellm.proxy.arcagi")


# In-memory job manager for progress + events
@dataclass
class JobState:
    job_id: str
    dataset_id: str
    model_id: str
    total: int
    completed: int = 0
    status: str = "running"  # running|done|error
    error: Optional[str] = None
    report_id: Optional[str] = None
    queue: asyncio.Queue = None


JOBS: Dict[str, JobState] = {}


def _ensure_dirs() -> None:
    global ARCAGI_ROOT, ARCAGI_CONFIG_PATH, ARCAGI_DATASETS_DIR, ARCAGI_REPORTS_DIR
    try:
        os.makedirs(ARCAGI_ROOT, exist_ok=True)
        os.makedirs(ARCAGI_DATASETS_DIR, exist_ok=True)
        os.makedirs(ARCAGI_REPORTS_DIR, exist_ok=True)
    except PermissionError:
        # Fallback to a local path under the current working directory
        fallback_root = os.path.join(os.getcwd(), "data", "arcagi")
        ARCAGI_ROOT = fallback_root
        ARCAGI_CONFIG_PATH = os.path.join(ARCAGI_ROOT, "config.json")
        ARCAGI_DATASETS_DIR = os.path.join(ARCAGI_ROOT, "datasets")
        ARCAGI_REPORTS_DIR = os.path.join(ARCAGI_ROOT, "reports")
        logger.warning(
            "ARC-AGI: Permission denied for %s; falling back to %s",
            "/data/arcagi",
            fallback_root,
        )
        os.makedirs(ARCAGI_ROOT, exist_ok=True)
        os.makedirs(ARCAGI_DATASETS_DIR, exist_ok=True)
        os.makedirs(ARCAGI_REPORTS_DIR, exist_ok=True)


def _ensure_config() -> dict:
    _ensure_dirs()
    if not os.path.exists(ARCAGI_CONFIG_PATH):
        # Create a minimal placeholder config with no datasets
        with open(ARCAGI_CONFIG_PATH, "w") as f:
            json.dump({"datasets": []}, f, indent=2)
    with open(ARCAGI_CONFIG_PATH, "r") as f:
        return json.load(f)


async def arcagi_bootstrap() -> None:
    """Ensure config and directories exist. Trigger dataset download if missing."""
    cfg = _ensure_config()
    datasets_cfg = cfg.get("datasets", [])
    missing: List[str] = []
    for d in datasets_cfg:
        did = d.get("id")
        if not did:
            continue
        path = os.path.join(ARCAGI_DATASETS_DIR, f"{did}.jsonl")
        if not os.path.exists(path):
            missing.append(did)
    if missing:
        logger.info("ARC-AGI bootstrap: missing datasets detected: %s", ", ".join(missing))
        try:
            res = await _download_datasets(restrict_ids=missing, overwrite=False)
            logger.info("ARC-AGI bootstrap: dataset download result: %s", json.dumps(res))
        except Exception as e:
            logger.warning("ARC-AGI bootstrap: failed to download datasets: %s", str(e))
    else:
        logger.info("ARC-AGI bootstrap: all datasets present; no download needed")


@router.get("/arcagi/config", dependencies=[Depends(user_api_key_auth)])
async def get_config() -> dict:
    cfg = _ensure_config()
    return cfg


def _hf_dataset_specs(dataset_id: str) -> Tuple[str, Optional[str]]:
    """Map our IDs to Hugging Face dataset names/configs.
    Returns (name, config) where config may be None.
    """
    if dataset_id == "hellaswag":
        return ("hellaswag", None)
    if dataset_id == "arc_easy":
        return ("ai2_arc", "ARC-Easy")
    if dataset_id == "arc_challenge":
        return ("ai2_arc", "ARC-Challenge")
    return (dataset_id, None)


async def _download_datasets(restrict_ids: Optional[List[str]] = None, overwrite: bool = True) -> dict:
    """Internal helper to download/export datasets to JSONL.
    - restrict_ids: limit to these dataset ids; otherwise use all from config
    - overwrite: if False, skip files that already exist (startup bootstrapping)
    """
    cfg = _ensure_config()
    if restrict_ids is not None:
        target = [d for d in cfg.get("datasets", []) if d.get("id") in restrict_ids]
    else:
        target = cfg.get("datasets", [])

    downloaded: List[str] = []
    skipped: List[str] = []
    errors: Dict[str, str] = {}

    # Lazy import to avoid hard dependency if library isn't installed
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        logger.warning("ARC-AGI datasets library not installed; cannot download: %s", str(e))
        for d in target:
            dataset_id = d.get("id")
            if not dataset_id:
                continue
            if not overwrite:
                # on bootstrap, just log skip
                skipped.append(dataset_id)
            else:
                marker_path = os.path.join(ARCAGI_DATASETS_DIR, f"{dataset_id}.marker")
                with open(marker_path, "w") as f:
                    f.write("datasets lib not installed; marker created")
                skipped.append(dataset_id)
            errors[dataset_id] = "datasets library not installed"
        return {"status": "partial", "downloaded": downloaded, "skipped": skipped, "errors": errors}

    for d in target:
        dataset_id = d.get("id")
        if not dataset_id:
            continue
        out_path = os.path.join(ARCAGI_DATASETS_DIR, f"{dataset_id}.jsonl")
        if os.path.exists(out_path) and not overwrite:
            skipped.append(dataset_id)
            continue

        name, config = _hf_dataset_specs(dataset_id)
        try:
            if config:
                ds = load_dataset(name, config, split="validation")
            else:
                ds = load_dataset(name, split="validation")
        except Exception:
            try:
                if config:
                    ds = load_dataset(name, config, split="train")
                else:
                    ds = load_dataset(name, split="train")
            except Exception as e:
                skipped.append(dataset_id)
                errors[dataset_id] = f"failed to load dataset: {e}"
                continue

        with open(out_path, "w", encoding="utf-8") as f:
            count = 0
            for item in ds:
                norm = _normalize_item(dataset_id, item)
                if norm is None:
                    continue
                f.write(json.dumps(norm) + "\n")
                count += 1
                if count >= 200:
                    break
        downloaded.append(dataset_id)

    status_summary = "ok" if not errors else ("partial" if downloaded else "error")
    return {"status": status_summary, "downloaded": downloaded, "skipped": skipped, "errors": errors}


@router.post("/arcagi/datasets/download", dependencies=[Depends(user_api_key_auth)])
async def download_datasets(request: Request) -> dict:
    """Download/refresh datasets defined in config. Overwrites existing files if present.
    If `datasets` list provided in JSON body, restrict to those; otherwise use config.json.
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    restrict_ids: Optional[List[str]] = body.get("datasets") if isinstance(body, dict) else None
    res = await _download_datasets(restrict_ids=restrict_ids, overwrite=True)
    logger.info("ARC-AGI manual dataset download: %s", json.dumps(res))
    return res


def _normalize_item(dataset_id: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize dataset items to {question, choices, answer}.
    Returns None if cannot normalize.
    """
    try:
        if dataset_id == "hellaswag":
            # HellaSwag fields: 'ctx', 'endings', 'label'
            ctx = item.get("ctx") or item.get("context")
            endings = item.get("endings")
            label = item.get("label")
            if ctx and isinstance(endings, list) and label is not None:
                return {
                    "question": f"{ctx}\n\nWhich ending is most plausible?",
                    "choices": endings,
                    "answer": int(label),
                }
        else:
            # ARC: item has 'question', 'choices':{'text':[], 'label':[]}, 'answerKey'
            q = item.get("question")
            choices = item.get("choices") or {}
            texts = choices.get("text")
            labels = choices.get("label")
            ans_key = item.get("answerKey")
            if q and isinstance(texts, list) and isinstance(labels, list) and ans_key:
                # answer index
                idx = labels.index(ans_key) if ans_key in labels else None
                if idx is not None and idx < len(texts):
                    return {
                        "question": q,
                        "choices": texts,
                        "answer": idx,
                    }
    except Exception:
        return None
    return None


def _sample_questions(dataset_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Read normalized JSONL if present; otherwise fallback to a small built-in set."""
    path = os.path.join(ARCAGI_DATASETS_DIR, f"{dataset_id}.jsonl")
    out: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
                    if len(out) >= limit:
                        break
        except Exception:
            out = []
    if out:
        return out

    # Fallback questions (dataset-agnostic)
    fallback = [
        {
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": 2,
        },
        {
            "question": "2 + 2 = ?",
            "choices": ["3", "4", "5", "22"],
            "answer": 1,
        },
        {
            "question": "Which is the largest ocean?",
            "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
            "answer": 3,
        },
        {
            "question": "H2O is also known as?",
            "choices": ["Oxygen", "Hydrogen", "Water", "Helium"],
            "answer": 2,
        },
        {
            "question": "The chemical symbol for Gold is?",
            "choices": ["Ag", "Au", "Gd", "Go"],
            "answer": 1,
        },
    ]
    return fallback[:limit]


async def _call_model(model_id: str, prompt: str) -> Tuple[bool, str, float]:
    """Call the configured model via internal router. Returns (ok, content, latency)."""
    start = time.time()
    try:
        # Resolve llm_router lazily to avoid circular imports
        from litellm.proxy.proxy_server import llm_router
        import litellm

        messages = [{"role": "user", "content": prompt}]

        if llm_router is not None and model_id:
            resp = await llm_router.acompletion(
                model=model_id,
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )
        else:
            # fallback to using litellm directly
            resp = await litellm.acompletion(
                model=model_id,
                messages=messages,
                max_tokens=256,
                temperature=0.2,
            )

        # Extract content
        content = None
        try:
            content = resp.choices[0].message["content"]  # type: ignore[index]
        except Exception:
            content = str(resp)
        latency = time.time() - start
        return True, content or "", latency
    except Exception as e:
        latency = time.time() - start
        return False, f"ERROR: {e}", latency


def _check_correct(answer_idx: int, model_text: str, choices: List[str]) -> bool:
    try:
        # Simple heuristics: look for choice text or a letter/index
        correct_text = choices[answer_idx].strip().lower()
        t = (model_text or "").strip().lower()
        if correct_text and correct_text in t:
            return True
        # Also check index mentions (e.g., "option 3")
        if str(answer_idx) in t:
            return True
        # Check letter labels A-D
        label = chr(ord('A') + answer_idx)
        if label.lower() in t or ("option " + label.lower()) in t:
            return True
    except Exception:
        return False
    return False


def _build_markdown_report(model_id: str, dataset_id: str, results: List[Dict[str, Any]], total_time: float) -> str:
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = (correct / len(results)) * 100 if results else 0.0
    avg_latency = (total_time / len(results)) if results else 0.0
    lines: List[str] = []
    lines.append("# ARC-AGI Benchmark Report")
    lines.append("")
    lines.append(f"- Model: `{model_id}`")
    lines.append(f"- Dataset: `{dataset_id}`")
    lines.append(f"- Questions: {len(results)}")
    lines.append(f"- Correct: {correct}")
    lines.append(f"- Accuracy: {accuracy:.1f}%")
    lines.append(f"- Avg latency: {avg_latency:.2f}s")
    lines.append(f"- Total time: {total_time:.2f}s")
    lines.append("")
    lines.append("## Details")
    for i, r in enumerate(results, 1):
        is_correct: bool = bool(r.get("correct"))
        status = "✅ Correct" if is_correct else "❌ Incorrect"
        latency = r.get("latency", 0)
        lines.append(f"### Q{i} — {status} · {latency:.2f}s")
        lines.append("")
        q = str(r.get("question", "")).strip()
        lines.append("> " + q.replace("\n", " "))
        lines.append("")
        choices = r.get("choices") or []
        ans = int(r.get("answer", 0)) if r.get("answer") is not None else -1
        for idx, ch in enumerate(choices):
            label = chr(ord('A') + idx)
            text = str(ch).strip()
            suffix = " ⟵" if idx == ans else ""
            lines.append(f"- ({label}) {text}{suffix}")
        lines.append("")
        chosen = (r.get("model_text") or "").strip()
        if chosen:
            lines.append("#### Model output")
            lines.append("\n" + "```text\n" + chosen + "\n```" + "\n")
        # Add horizontal rule between questions, except after the last one
        if i < len(results):
            lines.append("\n---\n")
    return "\n".join(lines)


@router.post("/arcagi/benchmark/run", dependencies=[Depends(user_api_key_auth)])
async def run_benchmark(request: Request) -> dict:
    body = await request.json()
    model_id = body.get("model_id")
    dataset_id = body.get("dataset_id")
    if not isinstance(model_id, str) or not isinstance(dataset_id, str):
        raise HTTPException(status_code=400, detail={"error": "model_id and dataset_id required"})

    cfg = _ensure_config()
    valid_ids = {d.get("id") for d in cfg.get("datasets", []) if d.get("id")}
    if dataset_id not in valid_ids:
        raise HTTPException(status_code=400, detail={"error": "invalid dataset_id"})

    questions = _sample_questions(dataset_id, limit=10)
    job_id = str(uuid.uuid4())
    job = JobState(job_id=job_id, dataset_id=dataset_id, model_id=model_id, total=len(questions))
    job.queue = asyncio.Queue()
    JOBS[job_id] = job

    async def worker():
        start_total = time.time()
        results: List[Dict[str, Any]] = []
        try:
            for idx, q in enumerate(questions):
                prompt = q["question"]
                choices = q.get("choices", [])
                answer = int(q.get("answer", 0))
                ok, text, latency = await _call_model(model_id, _build_prompt(prompt, choices))
                correct = False
                if ok:
                    correct = _check_correct(answer, text, choices)
                results.append(
                    {
                        "question": prompt,
                        "choices": choices,
                        "answer": answer,
                        "model_text": text,
                        "ok": ok,
                        "latency": latency,
                        "correct": correct,
                    }
                )
                job.completed = idx + 1
                percent = int(job.completed * 100 / max(job.total, 1))
                await job.queue.put(json.dumps({
                    "type": "progress",
                    "completed": job.completed,
                    "total": job.total,
                    "percent": percent,
                }))

            total_time = time.time() - start_total
            md = _build_markdown_report(model_id, dataset_id, results, total_time)
            report_id = str(uuid.uuid4())
            report_path = os.path.join(ARCAGI_REPORTS_DIR, f"{report_id}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)
            job.status = "done"
            job.report_id = report_id
            await job.queue.put(json.dumps({"type": "done", "report_id": report_id}))
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            await job.queue.put(json.dumps({"type": "error", "message": str(e)}))
        finally:
            # Slight delay to allow SSE consumer to read final message
            await asyncio.sleep(0.25)
            # Mark queue completion
            await job.queue.put("__CLOSE__")

    asyncio.create_task(worker())
    return {"job_id": job_id}


def _build_prompt(question: str, choices: List[str]) -> str:
    if not choices:
        return question
    letters = [chr(ord('A') + i) for i in range(len(choices))]
    opts = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(len(choices))])
    return f"{question}\n\nOptions:\n{opts}\n\nAnswer with the best option letter only."


@router.get("/arcagi/benchmark/stream/{job_id}", dependencies=[Depends(user_api_key_auth)])
async def stream_benchmark(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={"error": "job not found"})

    async def event_gen() -> AsyncGenerator[bytes, None]:
        while True:
            msg = await job.queue.get()
            if msg == "__CLOSE__":
                break
            yield f"data: {msg}\n\n".encode("utf-8")

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/arcagi/report/{report_id}", dependencies=[Depends(user_api_key_auth)])
async def get_report(report_id: str) -> Response:
    """Return report markdown and delete file immediately after read (as per requirement)."""
    if not report_id or "/" in report_id or ".." in report_id:
        raise HTTPException(status_code=400, detail={"error": "invalid report_id"})
    path = os.path.join(ARCAGI_REPORTS_DIR, f"{report_id}.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail={"error": "report not found"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Delete after reading, before responding
        try:
            os.remove(path)
        except Exception:
            pass
        return PlainTextResponse(content, media_type="text/markdown")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
