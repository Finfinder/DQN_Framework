from pathlib import Path

import pytest
from scripts.validate_version_consistency import (
    VersionConsistencyError,
    main,
    validate_version_consistency,
)


def _write_fixture_repo(
    tmp_path,
    version="1.2.3",
    readme_version=None,
    changelog_version=None,
    changelog_body=None,
):
    (tmp_path / "version.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )

    badge_version = readme_version or version
    (tmp_path / "README.md").write_text(
        "# DQN Framework\n\n"
        f"[![Version](https://img.shields.io/badge/version-{badge_version}-green)]()\n",
        encoding="utf-8",
    )

    section_version = changelog_version or version
    section_body = changelog_body if changelog_body is not None else "- Added tests\n"
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        f"## [{section_version}] - 2026-04-26\n\n"
        f"{section_body}\n"
        "## [1.2.2] - 2026-04-20\n\n"
        "- Previous release\n",
        encoding="utf-8",
    )
    return tmp_path


class TestValidateVersionConsistency:

    def test_happy_path(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path)

        assert validate_version_consistency(repo_root) == "1.2.3"

    def test_accepts_expected_version_with_leading_v(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path)

        assert validate_version_consistency(repo_root, expected_version="v1.2.3") == "1.2.3"

    def test_raises_when_version_py_mismatch(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path, version="1.2.2")

        with pytest.raises(VersionConsistencyError, match="version.py"):
            validate_version_consistency(repo_root, expected_version="1.2.3")

    def test_raises_when_readme_badge_mismatch(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path, readme_version="1.2.2")

        with pytest.raises(VersionConsistencyError, match="README.md"):
            validate_version_consistency(repo_root)

    def test_raises_when_changelog_section_missing(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path, changelog_version="1.2.2")

        with pytest.raises(VersionConsistencyError, match="CHANGELOG.md"):
            validate_version_consistency(repo_root)

    def test_raises_when_readme_badge_is_invalid(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path)
        (repo_root / "README.md").write_text("# DQN Framework\n", encoding="utf-8")

        with pytest.raises(VersionConsistencyError, match="badge"):
            validate_version_consistency(repo_root)

    def test_raises_when_changelog_section_is_empty(self, tmp_path):
        repo_root = _write_fixture_repo(tmp_path, changelog_body="")

        with pytest.raises(VersionConsistencyError, match="empty"):
            validate_version_consistency(repo_root)

    @pytest.mark.parametrize("expected_version", ["main", "1.2", "release-1.2.3"])
    def test_raises_when_expected_version_format_is_invalid(self, tmp_path, expected_version):
        repo_root = _write_fixture_repo(tmp_path)

        with pytest.raises(VersionConsistencyError, match="Invalid expected version"):
            validate_version_consistency(repo_root, expected_version=expected_version)

    def test_main_returns_clean_error_without_traceback(self, tmp_path, capsys):
        repo_root = _write_fixture_repo(tmp_path, readme_version="1.2.2")

        exit_code = main([
            "--repo-root",
            str(repo_root),
        ])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "README.md" in captured.err
        assert "Traceback" not in captured.err


class TestWorkflowContracts:

    def test_ci_workflow_runs_version_validator(self):
        workflow_text = (
            Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")

        assert "uses: ./.github/workflows/reusable-version-consistency.yml" in workflow_text
        assert "repository-ref: ${{ github.sha }}" in workflow_text
        assert "Finfinder/AI_Instruction" not in workflow_text
        assert "scripts/validate_version_consistency.py" not in workflow_text

    def test_release_workflow_runs_version_validator(self):
        workflow_text = (
            Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")

        assert "uses: ./.github/workflows/reusable-version-consistency.yml" in workflow_text
        assert "uses: ./.github/workflows/reusable-next-version-request.yml" in workflow_text
        assert (
            "uses: softprops/action-gh-release@153bb8e04406b158c6c84fc1615b65b24149a1fe"
            in workflow_text
        )
        assert "source-repository: ${{ github.repository }}" in workflow_text
        assert "repository-ref: ${{ github.ref }}" in workflow_text
        assert "expected-version: ${{ github.ref_name }}" in workflow_text
        assert "expected-release-version: ${{ github.ref_name }}" in workflow_text
        assert "Finfinder/AI_Instruction" not in workflow_text
        assert "softprops/action-gh-release@v2" not in workflow_text
        assert "scripts/validate_version_consistency.py" not in workflow_text
        assert "Validate next version request" not in workflow_text

    def test_open_next_version_branch_uses_local_reusable_workflow(self):
        workflow_text = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "open-next-version-branch.yml"
        ).read_text(encoding="utf-8")
        reusable_workflow_text = (
            Path(__file__).resolve().parent.parent
            / ".github"
            / "workflows"
            / "reusable-open-next-version-branch.yml"
        ).read_text(encoding="utf-8")

        assert "uses: ./.github/workflows/reusable-open-next-version-branch.yml" in workflow_text
        assert "artifact-name: next-version-request" in workflow_text
        assert "base-branch: main" in workflow_text
        assert "commit_created" in reusable_workflow_text
        assert "branch_name" in reusable_workflow_text
        assert "Finfinder/AI_Instruction" not in workflow_text

    def test_pre_commit_hook_runs_version_validator(self):
        hook_text = (
            Path(__file__).resolve().parent.parent / ".pre-commit-config.yaml"
        ).read_text(encoding="utf-8")

        assert "validate-version-consistency" in hook_text
        assert "hooks/validate_version_consistency_hook.py" in hook_text
        assert "language: python" in hook_text
        assert "pass_filenames: false" in hook_text
