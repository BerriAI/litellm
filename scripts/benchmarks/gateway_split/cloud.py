import argparse
import hashlib
import itertools
import json
import logging
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from analyze_bifrost import Record, records, summarize
from bifrost_bench import FrozenModel
from matrix import TOPOLOGIES, Job, jobs, shard
from pydantic import TypeAdapter
from run import ROOT, Options, run, stop

LOGGER: Final = logging.getLogger(__name__)


class CLI(FrozenModel):
    command: Literal["plan", "run", "merge"]
    manifest: Path
    output: Path
    inputs: Path
    bifrost: Path
    suite_root: Path
    job: str | None
    shard_index: int
    shard_count: int
    machine_class: str | None


class Manifest(FrozenModel):
    schema_version: Literal[1] = 1
    jobs: tuple[Job, ...]


class Metadata(FrozenModel):
    platform: str
    cpu: str
    logical_cpus: int
    memory_bytes: int
    machine_class: str
    python: str
    rust: str
    git_sha: str
    versions: Mapping[str, str]
    harness_sha256: Mapping[str, str]
    bifrost_binary_sha256: str
    bifrost_patch_sha256: str
    resolved_http_parser: str
    dependency_lock_sha256: str
    parameters: Options


def check_job(job: Job, path: Path) -> tuple[Record, ...]:
    rows: Final = tuple(records((path,)))
    expected: Final = frozenset(itertools.product(TOPOLOGIES, range(job.repetitions)))
    actual: Final = frozenset((row.variant, row.repetition) for row in rows)
    if len(rows) != job.expected_rows or actual != expected:
        raise ValueError(f"Incomplete or duplicate trials for {job.id}")
    if not all(
        row.profile == job.profile
        and row.prompt_bytes == job.prompt_bytes
        and row.stream == (job.response == "sse")
        and row.concurrency == job.concurrency
        for row in rows
    ):
        raise ValueError(f"Workload differs from manifest for {job.id}")
    metadata: Final = Metadata.model_validate_json(path.with_suffix(".metadata.json").read_bytes())
    options: Final = metadata.parameters
    if (options.profile, options.sizes, options.responses, options.concurrencies) != (
        job.profile,
        (job.prompt_bytes,),
        (job.response == "sse",),
        (job.concurrency,),
    ):
        raise ValueError(f"Parameters differ from manifest for {job.id}")
    if (options.seconds, options.repetitions, options.warmup, options.seed) != (
        job.seconds,
        job.repetitions,
        job.warmup_seconds,
        job.seed,
    ):
        raise ValueError(f"Timing differs from manifest for {job.id}")
    probes: Final = TypeAdapter(tuple[dict[str, object], ...]).validate_json(
        path.with_suffix(".probes.json").read_bytes()
    )
    if {probe.get("variant") for probe in probes} != set(TOPOLOGIES):
        raise ValueError(f"Missing topology probes for {job.id}")
    for probe in probes:
        if not all(
            probe.get(key) is True
            for key in (
                "body_digest",
                "anthropic_sdk_messages",
                "slow_upload_verified",
                "headers",
                "status_passthrough",
                "cancelled_upstream",
            )
        ):
            raise ValueError(f"Failed correctness probe for {job.id}")
    return rows


def compatibility(metadata: Metadata) -> str:
    return json.dumps(metadata.model_dump(exclude={"parameters"}), sort_keys=True)


def merge(manifest: Manifest, source: Path, output: Path) -> None:
    paths: Final = tuple(source / job.id / "trials.jsonl" for job in manifest.jobs)
    rows: Final = tuple(itertools.chain.from_iterable(check_job(job, path) for job, path in zip(manifest.jobs, paths)))
    metadata: Final = tuple(
        Metadata.model_validate_json(path.with_suffix(".metadata.json").read_bytes()) for path in paths
    )
    if len(frozenset(compatibility(item) for item in metadata)) != 1:
        raise ValueError("Cannot combine different hardware classes, revisions, dependencies or benchmark binaries")
    summaries: Final = tuple(
        summarize(
            tuple(
                row
                for row in rows
                if row.variant == variant
                and row.profile == job.profile
                and row.prompt_bytes == job.prompt_bytes
                and row.stream == (job.response == "sse")
                and row.concurrency == job.concurrency
            ),
            job.repetitions,
        )
        for job, variant in itertools.product(manifest.jobs, TOPOLOGIES)
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(tuple(item.model_dump() for item in summaries), indent=2) + "\n")
    (output / "trials.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows))
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    LOGGER.info("Validated %s jobs, %s trials, four topologies per workload", len(manifest.jobs), len(rows))


def execute(job: Job, options: Options) -> None:
    if options.output.exists():
        check_job(job, options.output)
        metadata: Final = Metadata.model_validate_json(options.output.with_suffix(".metadata.json").read_bytes())
        current: Final = {ROOT / name: digest for name, digest in metadata.harness_sha256.items()}
        if not all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in current.items()):
            raise ValueError("Existing output belongs to different benchmark sources")
        if metadata.bifrost_binary_sha256 != hashlib.sha256(options.bifrost.read_bytes()).hexdigest():
            raise ValueError("Existing output belongs to a different Bifrost binary")
        if metadata.parameters != options:
            raise ValueError("Existing output belongs to different run parameters")
        LOGGER.info("Already complete %s", job.id)
        return
    run(options)
    check_job(job, options.output)


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run", "merge"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "matrix.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/cloud")
    parser.add_argument("--inputs", type=Path, default=ROOT / "results/cloud")
    parser.add_argument("--bifrost", type=Path, default=ROOT / "runtime/tools/suite")
    parser.add_argument("--suite-root", type=Path, default=ROOT / "runtime/tools/bifrost-benchmarking")
    parser.add_argument("--job")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--machine-class")
    raw: Final = CLI.model_validate(vars(parser.parse_args()))
    command: Final = TypeAdapter(str).validate_python(raw.command)
    manifest_path: Final = TypeAdapter(Path).validate_python(raw.manifest)
    output: Final = TypeAdapter(Path).validate_python(raw.output).resolve()
    if command == "plan":
        manifest_path.write_text(Manifest(jobs=jobs()).model_dump_json(indent=2) + "\n")
        LOGGER.info("Wrote 18 jobs, 216 measured trials to %s", manifest_path)
        return
    manifest: Final = Manifest.model_validate_json(manifest_path.read_bytes())
    if len(frozenset(job.id for job in manifest.jobs)) != len(manifest.jobs):
        raise ValueError("Manifest contains duplicate job IDs")
    if command == "merge":
        merge(manifest, TypeAdapter(Path).validate_python(raw.inputs), output)
        return
    machine_class: Final = TypeAdapter(str).validate_python(raw.machine_class)
    job_id: Final = raw.job
    selected: Final = (
        tuple(job for job in manifest.jobs if job.id == job_id)
        if job_id
        else shard(
            manifest.jobs,
            TypeAdapter(int).validate_python(raw.shard_index),
            TypeAdapter(int).validate_python(raw.shard_count),
        )
    )
    if not selected:
        raise ValueError("No jobs selected")
    for job in selected:
        LOGGER.info("Running %s", job.id)
        execute(
            job,
            Options(
                bifrost=TypeAdapter(Path).validate_python(raw.bifrost).resolve(),
                suite_root=TypeAdapter(Path).validate_python(raw.suite_root).resolve(),
                output=output / job.id / "trials.jsonl",
                profile=job.profile,
                sizes=(job.prompt_bytes,),
                concurrencies=(job.concurrency,),
                responses=(job.response == "sse",),
                seconds=job.seconds,
                repetitions=job.repetitions,
                warmup=job.warmup_seconds,
                seed=job.seed,
                machine_class=machine_class,
            ),
        )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
