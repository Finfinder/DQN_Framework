---
# Synchronizacja metadanych GitHub

Pliki `.github/gh-sync.json` oraz `.github/issue-seed.json` (jeśli występują) są źródłem prawdy dla metadanych repozytorium i dla seedowania issue. Sekwencja robocza: `sync` potem `seed`.

Przykładowe komendy PowerShell (dry-run):

```powershell
.\scripts\sync-github-meta.ps1 -DryRun > sync-dry-run.log
.\scripts\seed-github-issues.ps1 -DryRun >> sync-dry-run.log
```

Przykładowe komendy PowerShell (Apply):

```powershell
.\scripts\sync-github-meta.ps1 -Apply
.\scripts\seed-github-issues.ps1 -Apply
```

Backup przed Apply:
- Eksportuj REST repo (np. `gh api repos/:owner/:repo > backup.json`).
- Sprawdź kodowanie (UTF-8) i unikalność identyfikatorów.

Checklist przed Apply:
- Backup REST
- Sprawdzenie UTF-8
- Unikatowe `sourceId`
- Sprawdź prefix tytułów issue (np. `DQN-\d+`)

Jeśli nie ma lokalnych skryptów `scripts/sync-github-meta.ps1` / `scripts/seed-github-issues.ps1` — dostosuj je lub zignoruj workflow. Workflow zaprojektowany tak, żeby nie kończyć się błędem w ich braku.

Dostosuj prefix issue (np. TB-, IA-, AR-, SEQ-, DQN-) jeśli używasz lokalnego seed.

Króciutka checklista walidacji:
- UTF-8
- `sourceId` unikatowe
- Tytuły issue zgodne z konwencją

---
