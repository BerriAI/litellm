#!/usr/bin/env python3
import sys

import requests
from packaging.requirements import Requirement
from pathlib import Path
import json
from typing import Dict, List, Optional, Set, Tuple
import configparser
import re
from dataclasses import dataclass


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
                    package = package.strip().lower()
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
        """Fetch license information for a package from PyPI."""
        try:
            url = f"https://pypi.org/pypi/{package_name}/{version}/json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("info", {}).get("license")
        except Exception as e:
            print(
                f"Warning: Failed to fetch license for {package_name} {version}: {str(e)}"
            )
            return None

    def is_license_acceptable(self, license_str: str) -> Tuple[bool, str]:
        """Check if a license is acceptable based on configured lists."""
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
        package_lower = package_name.lower()

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

    def check_requirements(self, requirements_file: Path) -> bool:
        """Check all packages in a requirements file."""
        print(f"\nChecking licenses for packages in {requirements_file}...")

        try:
            with open(requirements_file) as f:
                requirements = [
                    Requirement(line.strip())
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except Exception as e:
            print(f"Error parsing {requirements_file}: {str(e)}")
            return False

        all_compliant = True

        for req in requirements:
            try:
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
    # req_file = "../../requirements.txt" ## LOCAL TESTING
    req_file = "./requirements.txt"
    checker = LicenseChecker()

    # Check requirements
    if not checker.check_requirements(Path(req_file)):
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
            if p.name.lower() not in checker.authorized_packages
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
