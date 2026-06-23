#!/usr/bin/env python3
import configparser
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Dict, List, Optional, Set, Tuple

from packaging.requirements import Requirement
import requests

DEFAULT_TRANSITIVE_PIN_PACKAGES = (
    "aiofiles",
    "anyio",
    "async-generator",
    "azure-keyvault",
    "colorlog",
    "filelock",
    "grpc-google-iam-v1",
    "h11",
    "hf-xet",
    "jaraco-context",
    "redis",
    "requests-toolbelt",
    "starlette",
    "tornado",
    "tzdata",
    "urllib3",
    "wheel",
)

# SPDX license expressions (PEP 639 "License-Expression") join identifiers with
# the uppercase operators OR / AND / WITH. The split is case-sensitive: the
# lowercase "-or-later" inside an identifier such as "GPL-2.0-or-later" is part
# of the identifier, not an operator.
_SPDX_OPERATOR_SPLIT = re.compile(r"\s+(?:OR|AND)\s+")
_SPDX_WITH_SUFFIX = re.compile(r"\s+WITH\s+.*", re.DOTALL)


@dataclass
class PackageLicense:
    name: str
    version: Optional[str]
    license_type: Optional[str]
    is_authorized: bool
    reason: str


class LicenseChecker:
    def __init__(
        self, config_file: Path = Path("./tests/code_coverage_tests/liccheck.ini")
    ):
        if not config_file.exists():
            print(f"Error: Config file {config_file} not found")
            sys.exit(1)

        self.config = configparser.ConfigParser(allow_no_value=True)
        self.config.read(config_file)

        # Initialize license sets
        self.authorized_licenses = self._parse_license_list(
            "Licenses", "authorized_licenses"
        )
        self.unauthorized_licenses = self._parse_license_list(
            "Licenses", "unauthorized_licenses"
        )

        # Parse authorized packages
        self.authorized_packages = self._parse_authorized_packages()

        # Initialize cache
        self.cache_file = Path("license_cache.json")
        self.license_cache: Dict[str, str] = {}
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.license_cache = json.load(f)

        # Track package results
        self.package_results: List[PackageLicense] = []

    @staticmethod
    def _normalize_package_name(package_name: str) -> str:
        """Canonicalize package names so '-', '_' and '.' compare equivalently."""
        return re.sub(r"[-_.]+", "-", package_name).lower()

    def _parse_license_list(self, section: str, option: str) -> Set[str]:
        """Parse license list from config, handling comments and whitespace."""
        if not self.config.has_option(section, option):
            return set()

        licenses = set()
        for line in self.config.get(section, option).split("\n"):
            line = line.strip().lower()
            if line and not line.startswith("#"):
                licenses.add(line)
        return licenses

    def _parse_authorized_packages(self) -> Dict[str, Dict[str, str]]:
        """Parse authorized packages with their version specs and comments."""
        authorized = {}
        if self.config.has_section("Authorized Packages"):
            for package, spec in self.config.items("Authorized Packages"):
                if not package.startswith("#"):
                    package = self._normalize_package_name(package.strip())
                    parts = spec.split("#", 1)
                    version_spec = parts[0].strip()
                    comment = parts[1].strip() if len(parts) > 1 else ""
                    authorized[package] = {
                        "version_spec": version_spec,
                        "comment": comment,
                    }
        return authorized

    def get_package_license_from_pypi(
        self, package_name: str, version: str
    ) -> Optional[str]:
        """Fetch license information for a package from PyPI.

        Prefers the PEP 639 SPDX expression (``info.license_expression``),
        falls back to the legacy free-text ``info.license`` field, and as a
        last resort derives the license from the ``License :: OSI Approved ::
        ...`` trove classifiers.
        """
        try:
            url = f"https://pypi.org/pypi/{package_name}/{version}/json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            info = response.json().get("info", {}) or {}
            return (
                info.get("license_expression")
                or info.get("license")
                or self._license_from_classifiers(info.get("classifiers") or [])
            )
        except Exception as e:
            print(
                f"Warning: Failed to fetch license for {package_name} {version}: {str(e)}"
            )
            return None

    @staticmethod
    def _license_from_classifiers(classifiers: List[str]) -> Optional[str]:
        """Derive a license name from the ``License :: OSI Approved :: ...`` trove classifiers."""
        prefix = "License :: OSI Approved :: "
        for classifier in classifiers:
            if classifier.startswith(prefix):
                license_name = classifier[len(prefix) :].strip()
                if license_name:
                    return license_name
        return None

    @staticmethod
    def _split_spdx_expression(license_str: str) -> Optional[List[str]]:
        """Split an SPDX license expression into its component identifiers.

        Returns ``None`` when the string is not a recognizable SPDX expression
        (for example a free-text license blob), so callers fall back to
        whole-string matching.
        """
        if "OR" not in license_str and "AND" not in license_str:
            return None

        components: List[str] = []
        normalized = license_str.replace("(", " ").replace(")", " ")
        for part in _SPDX_OPERATOR_SPLIT.split(normalized):
            # Drop any "WITH <exception>" suffix: the exception qualifies the
            # preceding license, it is not itself a license to authorize.
            identifier = _SPDX_WITH_SUFFIX.sub("", part).strip()
            if not identifier:
                continue
            # SPDX short-form identifiers are single whitespace-free tokens; a
            # component with internal whitespace means this is free text.
            if any(char.isspace() for char in identifier):
                return None
            components.append(identifier)

        return components if len(components) > 1 else None

    def is_license_acceptable(self, license_str: Optional[str]) -> Tuple[bool, str]:
        """Check if a license (or compound SPDX expression) is acceptable."""
        if not license_str:
            return False, "Unknown license"

        components = self._split_spdx_expression(license_str)
        if components is None:
            return self._is_single_license_acceptable(license_str)

        # Compound SPDX expression: conservatively require every component to
        # be acceptable on its own (the safe direction for a CI gate).
        for component in components:
            is_acceptable, reason = self._is_single_license_acceptable(component)
            if not is_acceptable:
                return False, f"{reason} (in SPDX expression '{license_str}')"
        return True, f"All SPDX components authorized: {', '.join(components)}"

    def _is_single_license_acceptable(self, license_str: str) -> Tuple[bool, str]:
        """Check if a single license identifier is acceptable based on configured lists."""
        if not license_str:
            return False, "Unknown license"

        # Normalize license string to handle common variations
        normalized_license = license_str.lower()
        normalized_license = normalized_license.replace("-", " ").replace("_", " ")

        # Special case for BSD licenses
        if "bsd" in normalized_license:
            if any(
                variation in normalized_license
                for variation in ["3 clause", "3-clause", "new", "simplified"]
            ):
                return True, "Matches authorized license: BSD 3-Clause"

        # Check unauthorized licenses first
        for unauth in self.unauthorized_licenses:
            if unauth in normalized_license:
                return False, f"Matches unauthorized license: {unauth}"

        # Then check authorized licenses
        for auth in self.authorized_licenses:
            if auth in normalized_license:
                return True, f"Matches authorized license: {auth}"

        return False, "License not in authorized list"

    def check_package(self, package_name: str, version: Optional[str] = None) -> bool:
        """Check if a specific package version is compliant."""
        package_lower = self._normalize_package_name(package_name)

        # Check if package is in authorized packages list
        if package_lower in self.authorized_packages:
            pkg_info = self.authorized_packages[package_lower]

            # If there's a comment, consider it manually verified
            if pkg_info.get("comment"):
                result = PackageLicense(
                    name=package_name,
                    version=version,
                    license_type=pkg_info["comment"],
                    is_authorized=True,
                    reason="Manually verified in config",
                )
                self.package_results.append(result)
                print(f"✅ {package_name}: Manually verified - {pkg_info['comment']}")
                return True

            # If no comment, proceed with license check but package is considered authorized
            license_type = self.get_package_license_from_pypi(
                package_name, version or ""
            )
            if license_type:
                is_acceptable, reason = self.is_license_acceptable(license_type)
                result = PackageLicense(
                    name=package_name,
                    version=version,
                    license_type=license_type,
                    is_authorized=True,  # Package is authorized even if license check fails
                    reason=f"Listed in authorized packages - {license_type}",
                )
                self.package_results.append(result)
                print(
                    f"✅ {package_name}: {license_type} (Listed in authorized packages)"
                )
                return True

        # If package is not authorized or authorization check failed, proceed with normal license check
        cache_key = f"{package_name}:{version}" if version else package_name

        if cache_key in self.license_cache:
            license_type = self.license_cache[cache_key]
        else:
            license_type = self.get_package_license_from_pypi(
                package_name, version or ""
            )
            if license_type:
                self.license_cache[cache_key] = license_type

        if not license_type:
            result = PackageLicense(
                name=package_name,
                version=version,
                license_type=None,
                is_authorized=False,
                reason="Could not determine license",
            )
            self.package_results.append(result)
            print(f"⚠️  Warning: Could not determine license for {package_name}")
            return False

        is_acceptable, reason = self.is_license_acceptable(license_type)

        result = PackageLicense(
            name=package_name,
            version=version,
            license_type=license_type,
            is_authorized=is_acceptable,
            reason=reason,
        )
        self.package_results.append(result)

        if is_acceptable:
            print(f"✅ {package_name}: {license_type}")
        else:
            print(f"❌ {package_name}: {license_type} - {reason}")

        return is_acceptable

    def _load_requirements(
        self, requirements_file: Optional[Path] = None
    ) -> List[Requirement]:
        """Load pinned requirements from a file or from the repo defaults."""
        try:
            if requirements_file is not None:
                with open(requirements_file) as f:
                    requirement_lines = f.readlines()
            else:
                with open("pyproject.toml", "rb") as f:
                    pyproject = tomllib.load(f)
                with open("uv.lock", "rb") as f:
                    lock_data = tomllib.load(f)

                requirement_lines = list(pyproject["project"].get("dependencies", []))
                for extra_reqs in (
                    pyproject["project"].get("optional-dependencies", {}).values()
                ):
                    requirement_lines.extend(extra_reqs)
                for group_reqs in pyproject.get("dependency-groups", {}).values():
                    requirement_lines.extend(group_reqs)

                lock_versions: Dict[str, List[str]] = {}
                for package in lock_data.get("package", []):
                    source = package.get("source", {})
                    if "registry" not in source:
                        continue

                    normalized_name = self._normalize_package_name(package["name"])
                    version = package.get("version")
                    if not version:
                        continue
                    versions = lock_versions.setdefault(normalized_name, [])
                    if version not in versions:
                        versions.append(version)

                # Preserve the coverage that used to come from requirements.txt for
                # explicitly pinned transitives/security fixes without broadening the
                # default check to every package variant in the lockfile.
                for package_name in DEFAULT_TRANSITIVE_PIN_PACKAGES:
                    for version in lock_versions.get(package_name, []):
                        requirement_lines.append(f"{package_name}=={version}")

                # Preserve declaration order while removing duplicates.
                requirement_lines = list(dict.fromkeys(requirement_lines))

            return [
                Requirement(line.split("#")[0].strip())
                for line in requirement_lines
                if line.split("#")[0].strip() and not line.startswith("#")
            ]
        except Exception as e:
            source = requirements_file or "pyproject.toml + uv.lock"
            raise RuntimeError(
                f"Error parsing requirements from {source}: {str(e)}"
            ) from e

    def check_requirements(self, requirements_file: Optional[Path] = None) -> bool:
        """Check all packages from a requirements file or the default repo deps."""
        source = requirements_file or "pyproject.toml + uv.lock"
        print(f"\nChecking licenses for packages in {source}...")

        try:
            requirements = self._load_requirements(requirements_file)
        except RuntimeError as e:
            print(str(e))
            return False

        all_compliant = True

        for req in requirements:
            # Prefer a lower-bound/exact version (a real released version) for the
            # PyPI license lookup. ``next(iter(req.specifier))`` returns an
            # arbitrary clause; for a range like ``>=1.0,<2.0`` that can be the
            # upper bound (``2.0``) — a version that may not exist on PyPI and
            # would 404 to an "unknown" license.
            try:
                floor_versions = [
                    spec.version
                    for spec in req.specifier
                    if spec.operator in (">=", "==", "===", "~=", ">")
                ]
                if floor_versions:
                    version = floor_versions[0]
                else:
                    version = (
                        next(iter(req.specifier)).version if req.specifier else None
                    )
            except StopIteration:
                version = None

            if not self.check_package(req.name, version):
                all_compliant = False

        # Save updated cache
        with open(self.cache_file, "w") as f:
            json.dump(self.license_cache, f, indent=2)

        return all_compliant


def main():
    req_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    checker = LicenseChecker()

    # Check requirements
    if not checker.check_requirements(req_file):
        # Get lists of problematic packages
        unverified = [p for p in checker.package_results if not p.license_type]
        invalid = [
            p for p in checker.package_results if p.license_type and not p.is_authorized
        ]

        # Print detailed information about problematic packages
        if unverified:
            print("\n❌ Packages with unknown licenses:")
            for pkg in unverified:
                version_str = f" ({pkg.version})" if pkg.version else ""
                print(f"- {pkg.name}{version_str}")

        if invalid:
            print("\n❌ Packages with unauthorized licenses:")
            for pkg in invalid:
                version_str = f" ({pkg.version})" if pkg.version else ""
                print(f"- {pkg.name}{version_str}: {pkg.license_type}")

        # Only error if there are packages that aren't explicitly authorized
        unhandled_packages = [
            p
            for p in (unverified + invalid)
            if checker._normalize_package_name(p.name)
            not in checker.authorized_packages
        ]

        if unhandled_packages:
            print("\n❌ Error: Found packages that need verification:")
            for pkg in unhandled_packages:
                version_str = f" ({pkg.version})" if pkg.version else ""
                license_str = (
                    f" - {pkg.license_type}"
                    if pkg.license_type
                    else " - Unknown license"
                )
                print(f"- {pkg.name}{version_str}{license_str}")
            print(
                "\nAdd these packages to the [Authorized Packages] section in liccheck.ini with a comment about their license verification."
            )
            print("Example:")
            print("package-name: >=1.0.0  # MIT license manually verified")
            sys.exit(1)
    else:
        print("\n✅ All dependencies have acceptable licenses.")

    sys.exit(0)


if __name__ == "__main__":
    main()
