#!/usr/bin/env python3
"""
Generate an LTX video through LiteLLM and save the resulting MP4 locally.

Examples:

Text-to-video:
    poetry run python cookbook/ltx_video_generation.py \
      --prompt "A slow cinematic drone shot over snowy mountains at sunrise"

Image-to-video with a local image:
    poetry run python cookbook/ltx_video_generation.py \
      --prompt "The camera slowly pushes in while clouds drift overhead" \
      --input-reference ./assets/reference.jpg \
      --model ltx/ltx-2-3-pro

Notes:
- This script uses LiteLLM's LTX integration directly.
- On this branch, LTX video bytes are stored locally and retrieved in-process,
  so generation and download must happen in the same script run.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import litellm


DEFAULT_MODEL = "ltx/ltx-2-3-fast"
DEFAULT_SECONDS = "5"
DEFAULT_SIZE = "1920x1080"


def _serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)
    if hasattr(response, "dict"):
        return response.dict(exclude_none=True)
    if isinstance(response, dict):
        return response
    return {"response": str(response)}


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "ltx_video"


def _default_output_path(model: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_suffix = _sanitize_filename_part(model.split("/", 1)[-1])
    return Path.cwd() / f"ltx_{model_suffix}_{timestamp}.mp4"


def _looks_like_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:", "ltx://"))


def _file_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_input_reference(input_reference: Optional[str]) -> Optional[str]:
    if input_reference is None:
        return None

    if _looks_like_uri(input_reference):
        return input_reference

    candidate = Path(input_reference).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Input reference was not found: {candidate}. "
            "Pass an HTTPS URL, data URI, ltx:// upload URI, or a local file path."
        )

    return _file_to_data_uri(candidate)


def _load_extra_body(
    extra_body: Optional[str], extra_body_file: Optional[str]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    if extra_body_file:
        file_payload = json.loads(Path(extra_body_file).read_text())
        if not isinstance(file_payload, dict):
            raise ValueError("--extra-body-file must contain a JSON object")
        payload.update(file_payload)

    if extra_body:
        inline_payload = json.loads(extra_body)
        if not isinstance(inline_payload, dict):
            raise ValueError("--extra-body must be a JSON object")
        payload.update(inline_payload)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and download an LTX video using LiteLLM."
    )
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for video generation."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LTX model to call through LiteLLM. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--seconds",
        default=DEFAULT_SECONDS,
        help=f"Clip duration in seconds. Default: {DEFAULT_SECONDS}",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help=f"Output resolution. Default: {DEFAULT_SIZE}",
    )
    parser.add_argument(
        "--input-reference",
        help=(
            "Optional image reference for image-to-video. Supports HTTPS URLs, "
            "data URIs, ltx:// upload URIs, or a local file path."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Optional LTX-specific FPS override.",
    )
    parser.add_argument(
        "--camera-motion",
        help="Optional LTX-specific camera motion value.",
    )
    parser.add_argument(
        "--generate-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to ask LTX to generate audio.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LTX_API_KEY"),
        help="LTX API key. Defaults to the LTX_API_KEY environment variable.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Optional custom LTX API base URL.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the generated MP4. Defaults to a timestamped file in the current directory.",
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help="Extra JSON body fields to pass through to LTX, as a JSON object string.",
    )
    parser.add_argument(
        "--extra-body-file",
        default=None,
        help="Path to a JSON file containing extra body fields to pass through to LTX.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print(
            "Missing LTX API key. Set LTX_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    try:
        input_reference = _normalize_input_reference(args.input_reference)
        extra_body = _load_extra_body(args.extra_body, args.extra_body_file)
    except Exception as exc:
        print(f"Failed to prepare request inputs: {exc}", file=sys.stderr)
        return 1

    if args.fps is not None:
        extra_body["fps"] = args.fps
    if args.camera_motion:
        extra_body["camera_motion"] = args.camera_motion
    if args.generate_audio is not None:
        extra_body["generate_audio"] = args.generate_audio

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(args.model)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating with model: {args.model}")
    print(f"Prompt: {args.prompt}")
    if input_reference:
        input_source = "local file converted to data URI"
        if args.input_reference and _looks_like_uri(args.input_reference):
            input_source = args.input_reference
        print(f"Input reference: {input_source}")
    print(f"Output path: {output_path}")

    try:
        response = litellm.video_generation(
            model=args.model,
            prompt=args.prompt,
            input_reference=input_reference,
            seconds=args.seconds,
            size=args.size,
            api_key=args.api_key,
            api_base=args.api_base,
            extra_body=extra_body or None,
        )
    except Exception as exc:
        print(f"Video generation failed: {exc}", file=sys.stderr)
        return 1

    response_payload = _serialize_response(response)
    print("\nGeneration response:")
    print(json.dumps(response_payload, indent=2, sort_keys=True))

    video_id = getattr(response, "id", None) or response_payload.get("id")
    if not video_id:
        print("No video_id was returned by LiteLLM.", file=sys.stderr)
        return 1

    try:
        video_bytes = litellm.video_content(
            video_id=video_id,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    except Exception as exc:
        print(f"Video download failed: {exc}", file=sys.stderr)
        return 1

    output_path.write_bytes(video_bytes)
    print(f"\nSaved {len(video_bytes)} bytes to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
