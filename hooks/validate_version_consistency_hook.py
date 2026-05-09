from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def resolve_shell() -> tuple[str, list[str]]:
    pwsh_path = shutil.which("pwsh")
    if pwsh_path:
        return pwsh_path, ["-NoProfile"]

    powershell_path = shutil.which("powershell")
    if powershell_path:
        return powershell_path, ["-NoProfile", "-ExecutionPolicy", "Bypass"]

    raise FileNotFoundError("PowerShell executable was not found. Install pwsh or ensure powershell is available.")


def main() -> int:
    repository_root = Path.cwd()
    automation_root = Path(__file__).resolve().parents[1]
    validator_path = automation_root / "scripts" / "validate-version-consistency.ps1"

    try:
        shell_path, shell_args = resolve_shell()
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    command = [
        shell_path,
        *shell_args,
        "-File",
        str(validator_path),
        "-RepositoryRoot",
        str(repository_root),
        "-VersionTargetsPath",
        str(repository_root / ".github" / "versioning" / "version-targets.json"),
        "-ReadmeTargetsPath",
        str(repository_root / ".github" / "versioning" / "readme-targets.json"),
    ]

    completed = subprocess.run(command, check=False, stdout=subprocess.DEVNULL)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())