import argparse
import hashlib
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

from cases import extension_path
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Measurement(FrozenModel):
    median_ns: PositiveFloat


class CodSpeedStats(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    median_ns: PositiveFloat


class CodSpeedBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)
    uri: str = Field(min_length=1)
    stats: CodSpeedStats


class CodSpeedInstrument(BaseModel):
    type: Literal["walltime"]


class CodSpeedResults(BaseModel):
    model_config = ConfigDict(frozen=True)
    instrument: CodSpeedInstrument
    benchmarks: tuple[CodSpeedBenchmark, ...] = Field(min_length=1)


class Report(FrozenModel):
    environment: dict[str, str]
    revision: str
    extension_sha256: str
    measurements: dict[str, Measurement] = Field(min_length=1)


class Ceiling(FrozenModel):
    baseline_ns: PositiveFloat
    ceiling_ns: PositiveFloat
    unit: str = "ns/call"


class Budgets(FrozenModel):
    schema_version: int = 1
    baseline_revision: str = ""
    baseline_extension_sha256: str = ""
    environment: dict[str, str]
    cases: dict[str, Ceiling]


class Arguments(FrozenModel):
    mode: Literal["check", "baseline"]
    paths: tuple[Path, ...]
    output: Path
    budgets: Path


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(root: Path) -> dict[str, str]:
    return {
        "runner": os.environ.get("BOUNDARY_RUNNER", "local"),
        "architecture": platform.machine(),
        "os": platform.system(),
        "python": platform.python_version(),
        "rust": command("rustc", "--version"),
        "cargo_lock": digest(root / "litellm-rust/Cargo.lock"),
        "uv_lock": digest(root / "uv.lock"),
        "benchmark_dependencies": digest(Path(__file__).with_name("requirements.txt")),
        "rust_toolchain": digest(root / "rust-toolchain.toml"),
        "profile": "release",
        "features": "abi3,bench,extension-module",
    }


def measurements_from_codspeed(results: CodSpeedResults) -> dict[str, Measurement]:
    uris: Final = tuple(benchmark.uri for benchmark in results.benchmarks)
    if len(set(uris)) != len(uris):
        raise ValueError("Duplicate CodSpeed benchmark IDs")
    return {benchmark.uri: Measurement(median_ns=benchmark.stats.median_ns) for benchmark in results.benchmarks}


def from_codspeed(results: CodSpeedResults, root: Path) -> Report:
    return Report(
        environment=environment(root),
        revision=command("git", "rev-parse", "HEAD"),
        extension_sha256=digest(extension_path()),
        measurements=measurements_from_codspeed(results),
    )


def violations(report: Report, budgets: Budgets) -> tuple[str, ...]:
    return (
        *(
            ("No reviewed budgets; establish ceilings from five stable CodSpeed walltime reports",)
            if not budgets.cases
            else ()
        ),
        *(("Unsupported budget schema",) if budgets.schema_version != 1 else ()),
        *(("Environment differs from the calibrated baseline",) if report.environment != budgets.environment else ()),
        *(f"Missing measurement: {name}" for name in sorted(budgets.cases.keys() - report.measurements.keys())),
        *(f"Unbudgeted measurement: {name}" for name in sorted(report.measurements.keys() - budgets.cases.keys())),
        *(
            f"Invalid budget: {name}"
            for name, ceiling in budgets.cases.items()
            if ceiling.unit != "ns/call"
            or ceiling.ceiling_ns < ceiling.baseline_ns
            or ceiling.ceiling_ns > ceiling.baseline_ns * 1.20 + 0.001
        ),
        *(
            f"Over budget: {name}: {value.median_ns:.1f} ns > {budgets.cases[name].ceiling_ns:.1f} ns"
            for name, value in report.measurements.items()
            if name in budgets.cases and value.median_ns > budgets.cases[name].ceiling_ns
        ),
    )


def calibrate(reports: tuple[Report, ...]) -> Budgets:
    if len(reports) != 5:
        raise ValueError("Calibration requires exactly five independent reports")
    first: Final = reports[0]
    if first.environment.get("runner") != "codspeed-macro":
        raise ValueError("Budgets must be calibrated on codspeed-macro")
    if any(
        report.environment != first.environment
        or report.revision != first.revision
        or report.extension_sha256 != first.extension_sha256
        or report.measurements.keys() != first.measurements.keys()
        for report in reports
    ):
        raise ValueError("Calibration reports must have identical environments, builds, and cases")
    medians: Final = {
        name: tuple(report.measurements[name].median_ns for report in reports) for name in first.measurements
    }
    unstable: Final = tuple(
        name
        for name, values in medians.items()
        if any(abs(value / statistics.median(values) - 1) > 0.10 for value in values)
    )
    if unstable:
        raise ValueError(f"Unstable calibration cases: {', '.join(unstable)}")
    return Budgets(
        environment=first.environment,
        baseline_revision=first.revision,
        baseline_extension_sha256=first.extension_sha256,
        cases={
            name: Ceiling(baseline_ns=statistics.median(values), ceiling_ns=1.20 * statistics.median(values))
            for name, values in medians.items()
        },
    )


def summary(report: Report, budgets: Budgets) -> str:
    return "\n".join(
        (
            "| Case | Measured ns/call | Ceiling ns/call |",
            "|---|---:|---:|",
            *(
                f"| {name} | {value.median_ns:.1f} | {budgets.cases[name].ceiling_ns if name in budgets.cases else 'uncalibrated'} |"
                for name, value in report.measurements.items()
            ),
            "",
            *violations(report, budgets),
        )
    )


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "baseline"))
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=Path, default=Path(__file__).with_name("budgets.json"))
    args: Final = Arguments.model_validate(vars(parser.parse_args()))
    reports: Final = tuple(Report.model_validate_json(path.read_text()) for path in args.paths)
    if args.mode == "baseline":
        calibrated: Final = calibrate(reports)
        args.output.write_text(calibrated.model_dump_json(indent=2) + "\n")
        return 0
    if len(reports) != 1:
        parser.error("check requires exactly one report")
    budgets: Final = Budgets.model_validate_json(args.budgets.read_text())
    text: Final = summary(reports[0], budgets)
    args.output.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return int(bool(violations(reports[0], budgets)))


if __name__ == "__main__":
    sys.exit(main())
