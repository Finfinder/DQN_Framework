import argparse
import re
import sys
from pathlib import Path

_VERSION_RE = re.compile(r'__version__\s*=\s*["\'](?P<version>\d+\.\d+\.\d+)["\']')
_EXPECTED_VERSION_RE = re.compile(r"^v?(?P<version>\d+\.\d+\.\d+)$")
_README_BADGE_RE = re.compile(
    r"https://img\.shields\.io/badge/version-(?P<version>\d+\.\d+\.\d+)-"
)


class VersionConsistencyError(RuntimeError):
    pass


def _read_text(path):
    if not path.exists():
        raise VersionConsistencyError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def _extract_version(path):
    match = _VERSION_RE.search(_read_text(path))
    if not match:
        raise VersionConsistencyError(
            f"Could not read version from {path.as_posix()}"
        )
    return match.group("version")


def _extract_readme_badge_version(path):
    match = _README_BADGE_RE.search(_read_text(path))
    if not match:
        raise VersionConsistencyError(
            f"Could not read version badge from {path.as_posix()}"
        )
    return match.group("version")


def _normalize_expected_version(expected_version):
    if expected_version is None:
        return None

    match = _EXPECTED_VERSION_RE.fullmatch(expected_version.strip())
    if not match:
        raise VersionConsistencyError(
            "Invalid expected version: use X.Y.Z or vX.Y.Z"
        )
    return match.group("version")


def _validate_expected_version(actual, expected, source_label):
    if actual != expected:
        raise VersionConsistencyError(
            f"Inconsistent version in {source_label}: expected {expected}, found {actual}"
        )


def _extract_changelog_section(path, version):
    lines = _read_text(path).splitlines()
    header_prefix = f"## [{version}]"
    start_index = None
    end_index = len(lines)

    for index, line in enumerate(lines):
        if line.startswith(header_prefix):
            start_index = index
            break

    if start_index is None:
        raise VersionConsistencyError(
            f"CHANGELOG.md does not contain a section for version {version}"
        )

    for index in range(start_index + 1, len(lines)):
        if lines[index].startswith("## ["):
            end_index = index
            break

    section_lines = lines[start_index:end_index]
    body_lines = section_lines[1:]
    if not any(line.strip() for line in body_lines):
        raise VersionConsistencyError(
            f"CHANGELOG.md section for version {version} is empty"
        )

    return "\n".join(section_lines).strip() + "\n"


def validate_version_consistency(repo_root, expected_version=None):
    version_path = repo_root / "version.py"
    readme_path = repo_root / "README.md"
    changelog_path = repo_root / "CHANGELOG.md"

    version = _extract_version(version_path)
    normalized_expected = _normalize_expected_version(expected_version)

    if normalized_expected is not None:
        _validate_expected_version(version, normalized_expected, "version.py")

    readme_version = _extract_readme_badge_version(readme_path)
    _validate_expected_version(readme_version, version, "README.md")
    _extract_changelog_section(changelog_path, version)
    return version


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Validates version.py, README badge and CHANGELOG consistency."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing version.py, README.md and CHANGELOG.md",
    )
    parser.add_argument(
        "--expected-version",
        help="Optional expected version in X.Y.Z or vX.Y.Z format",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        validate_version_consistency(
            Path(args.repo_root),
            expected_version=args.expected_version,
        )
    except VersionConsistencyError as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
