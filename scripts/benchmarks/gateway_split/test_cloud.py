import itertools
import json
import tempfile
import unittest
from pathlib import Path
from typing import Final

from analyze_bifrost import Record, Summary
from bifrost_bench import Stats
from cloud import Manifest, Metadata, check_job, merge
from matrix import TOPOLOGIES, Job, jobs, shard
from pydantic import TypeAdapter
from run import Options


def trial(job: Job, variant: str, repetition: int) -> Record:
    return Record(
        generator="bifrost",
        variant=variant,
        prompt_bytes=job.prompt_bytes,
        body_bytes=job.prompt_bytes + 100,
        stream=job.response == "sse",
        concurrency=job.concurrency,
        repetition=repetition,
        python_body="stream",
        python_client="aiohttp",
        rust_client="hyper-stream",
        http_parser="httptools",
        event_loop="uvloop",
        profile=job.profile,
        gateway_cpu_s=0.2,
        gateway_cpu_cores=0.1,
        gateway_peak_rss_bytes=1024,
        loadgen_cpu_cores=0.1,
        backend_completed=10,
        backend_requests=10,
        backend_cancelled=0,
        stats=Stats(
            requests=10,
            rate=5,
            success_rate=100,
            mean_latency_ms=1,
            p50_latency_ms=1,
            p95_latency_ms=2,
            p99_latency_ms=3,
            max_latency_ms=4,
            elapsed_seconds=2,
            throughput_rps=5 + repetition,
            first_event_p50_ms=0.5,
            first_event_p95_ms=1,
            first_event_p99_ms=2,
            status_code_counts={"200": 10},
            drop_reasons={},
        ),
    )


def fixture(root: Path, job: Job, machine_class: str = "test-machine") -> Path:
    path: Final = root / job.id / "trials.jsonl"
    path.parent.mkdir(parents=True)
    rows: Final = tuple(
        trial(job, variant, repetition) for variant, repetition in itertools.product(TOPOLOGIES, range(job.repetitions))
    )
    path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
    metadata: Final = Metadata(
        platform="test-platform",
        cpu="test-cpu",
        logical_cpus=4,
        memory_bytes=1024,
        machine_class=machine_class,
        python="test-python",
        rust="test-rust",
        git_sha="test-revision",
        versions={},
        harness_sha256={},
        bifrost_binary_sha256="test-binary",
        bifrost_patch_sha256="test-patch",
        resolved_http_parser="h11",
        dependency_lock_sha256="test-lock",
        parameters=Options(
            output=path,
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
    path.with_suffix(".metadata.json").write_text(metadata.model_dump_json())
    probes: Final = tuple(
        {
            "variant": variant,
            "body_digest": True,
            "anthropic_sdk_messages": True,
            "slow_upload_verified": True,
            "headers": True,
            "status_passthrough": True,
            "cancelled_upstream": True,
        }
        for variant in TOPOLOGIES
    )
    path.with_suffix(".probes.json").write_text(json.dumps(probes))
    return path


class CloudContractTests(unittest.TestCase):
    def test_shards_cover_each_workload_exactly_once(self) -> None:
        planned: Final = jobs()
        for count in (1, 5, 113):
            with self.subTest(count=count):
                self.assertCountEqual(
                    tuple(
                        job.id
                        for job in itertools.chain.from_iterable(shard(planned, index, count) for index in range(count))
                    ),
                    tuple(job.id for job in planned),
                )

    def test_invalid_shard_assignment_is_rejected(self) -> None:
        for index, count in ((0, 0), (-1, 2), (2, 2)):
            with self.subTest(index=index, count=count), self.assertRaises(ValueError):
                shard(jobs(), index, count)

    def test_merger_includes_every_topology_and_repetition(self) -> None:
        selected: Final = jobs(repetitions=3)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            for job in selected:
                fixture(root, job)
            merge(Manifest(jobs=selected), root, root / "merged")
            combined: Final = (root / "merged/trials.jsonl").read_text().splitlines()
            self.assertEqual(len(combined), sum(job.expected_rows for job in selected))
            summaries: Final = TypeAdapter(tuple[Summary, ...]).validate_json(
                (root / "merged/summary.json").read_bytes()
            )
            self.assertEqual(len(summaries), len(selected) * len(TOPOLOGIES))
            self.assertTrue(all(row.rps.median == 6 for row in summaries))

    def test_duplicate_cannot_replace_a_missing_topology(self) -> None:
        job: Final = jobs(repetitions=1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path: Final = fixture(Path(temporary), job)
            rows: Final = path.read_text().splitlines()
            path.write_text("\n".join((rows[0], rows[0], *rows[2:])) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
                check_job(job, path)

    def test_missing_trial_is_rejected(self) -> None:
        job: Final = jobs(repetitions=1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path: Final = fixture(Path(temporary), job)
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
                check_job(job, path)

    def test_different_cloud_hardware_cannot_be_merged(self) -> None:
        selected: Final = jobs(repetitions=1)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            root: Final = Path(temporary)
            fixture(root, selected[0], "machine-one")
            fixture(root, selected[1], "machine-two")
            with self.assertRaisesRegex(ValueError, "different hardware"):
                merge(Manifest(jobs=selected), root, root / "merged")


if __name__ == "__main__":
    unittest.main()
