#!/usr/bin/env python3
"""Sync local matt-debutts directory with sk hosts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    candidates = [start] + list(start.parents)
    # First pass: look for the sync_with_sk package (dir with __init__.py).
    # Checked before .git so that a .git repo nested inside the parent
    # (e.g. matt-debutts/.git) doesn't short-circuit before we reach
    # the directory that actually contains the package.
    for candidate in candidates:
        pkg = candidate / "sync_with_sk"
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            return candidate
    # Fallback: nearest .git root.
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("Unable to locate repository root (missing .git directory).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync local matt-debutts tree to sk hosts via ssh/rsync/scp."
    )
    parser.add_argument("--upload", action="store_true", help="Sync local files to remote hosts.")
    parser.add_argument("--download", action="store_true", help="Sync remote files to local.")
    parser.add_argument(
        "--pull-remote-only",
        action="store_true",
        help="Deprecated alias for --download (now includes changed files).",
    )
    parser.add_argument(
        "--hosts",
        nargs="*",
        default=None,
        help="Hosts to sync. If omitted (or passed with no values), sync all default hosts.",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Local matt-debutts root. Defaults to directory containing this script.",
    )
    parser.add_argument(
        "--remote-root",
        default=None,
        help="Optional remote root override (applies to all hosts).",
    )
    parser.add_argument("--code-only", action="store_true", help="Sync code-only files.")
    parser.add_argument("--data-only", action="store_true", help="Sync data-only files.")
    parser.add_argument("--csv-only", action="store_true", help="Sync CSV-only files.")
    parser.add_argument(
        "--all-text-only",
        action="store_true",
        help="Only consider files ending with all_text.csv (data-only filter).",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Case-insensitive substring filter on relative paths. Repeatable.",
    )
    parser.add_argument(
        "--speed-download",
        action="store_true",
        help="Fast data mode: tar-stream grouped folders when enabled in config.",
    )
    parser.add_argument(
        "--delete-remote-only",
        action="store_true",
        help="Delete remote-only files after confirmation.",
    )
    parser.add_argument("--yes", action="store_true", help="Auto-confirm destructive actions.")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without executing.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full SSH scan of remote (slow). Default: use state file as remote proxy.",
    )
    parser.add_argument(
        "--max-hash-bytes",
        type=int,
        default=None,
        help="Hash files <= this size unless treated as data files.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to state cache JSON file. Defaults to <local-root>/.sync_with_sk_state.json",
    )
    parser.add_argument(
        "--pull-dbs",
        action="store_true",
        help="Download all out/*/analysis.db files from a remote host via rsync.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="With --pull-dbs: auto-merge downloaded DBs into local copies.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries for transfer operations."
    )
    return parser.parse_args()


HOST_ROOTS = {
    "sk1": "/lfs/skampere1/0/alexspan/matt-debutts",
    "sk2": "/lfs/skampere2/0/alexspan/matt-debutts",
    "sk3": "/lfs/skampere3/0/alexspan/matt-debutts",
}


def pull_dbs(host: str, local_root: Path, dry_run: bool = False,
             merge: bool = False) -> int:
    """Download all out/*/analysis.db files from *host* into local out/.

    Uses a single tar-over-SSH stream to grab only the *.db files, avoiding
    the slow ``find`` scan of 100k+ article directories.

    When *merge* is True, remote DBs land as analysis.remote.db and are
    merged into local analysis.db (local wins on conflicts).
    """
    import shutil

    remote_root = HOST_ROOTS.get(host)
    if remote_root is None:
        print(f"Unknown host {host!r}. Known hosts: {', '.join(HOST_ROOTS)}")
        return 1

    remote_out = f"{remote_root}/out"
    local_out = local_root / "out"
    # When merging, download into a temp dir so local DBs stay intact.
    if merge:
        import tempfile
        staging_dir = Path(tempfile.mkdtemp(prefix="pull_dbs_"))
    else:
        staging_dir = local_out

    staging_dir.mkdir(parents=True, exist_ok=True)

    # Single tar stream: on the remote, glob out/*/analysis.db and stream
    # them as tar. This skips the expensive recursive find entirely.
    # The glob out/*/analysis.db only matches depth-1 subdirs.
    tar_remote = (
        f"cd {remote_root} && "
        f"tar cf - out/*/analysis.db out/*/._analysis.db 2>/dev/null"
    )
    print(f"[pull-dbs] Streaming *.db files from {host}:{remote_out} …")
    if dry_run:
        print("  (dry-run, skipping transfer)")
        return 0

    # Back up existing local DBs before anything changes.
    if local_out.exists():
        for db in sorted(local_out.glob("*/analysis.db")):
            bak = db.with_name("analysis.db.local-bak")
            print(f"  backing up {db.relative_to(local_root)} → {bak.name}")
            shutil.copy2(str(db), str(bak))

    # Stream tar over SSH → extract locally.
    extract_dir = staging_dir if merge else local_root
    cmd = f"ssh {host} '{tar_remote}' | tar xf - -C '{extract_dir}'"
    print(f"  running: {cmd}")
    cp = subprocess.run(cmd, shell=True)
    if cp.returncode != 0:
        print(f"  tar stream failed (exit {cp.returncode})")
        return 1

    # List what we got.
    downloaded = sorted(Path(extract_dir).glob("out/*/analysis.db"))
    for db in downloaded:
        size_mb = db.stat().st_size / (1024 * 1024)
        print(f"  downloaded: {db.relative_to(extract_dir)} ({size_mb:.0f}M)")

    if not downloaded:
        print("  No analysis.db files received.")
        return 0

    # Merge phase.
    if merge:
        scripts_dir = local_root / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from merge_analysis_dbs import merge as merge_fn  # noqa: WPS433

        local_out.mkdir(exist_ok=True)
        for remote_db in downloaded:
            # e.g. staging/out/ap/analysis.db
            source_name = remote_db.parent.name  # "ap"
            local_dir = local_out / source_name
            local_dir.mkdir(exist_ok=True)
            local_db = local_dir / "analysis.db"
            remote_copy = local_dir / "analysis.remote.db"
            shutil.copy2(str(remote_db), str(remote_copy))

            if local_db.exists():
                # Merge smaller (local) into bigger (remote), then swap in.
                print(f"  merging local ({local_db.stat().st_size // (1024*1024)}M) "
                      f"into remote ({remote_copy.stat().st_size // (1024*1024)}M) "
                      f"for {source_name}")
                merge_fn(str(remote_copy), str(local_db))
                # remote_copy now has the merged result — make it the new DB.
                shutil.move(str(remote_copy), str(local_db))
            else:
                print(f"  no local DB for {source_name}, using remote as-is")
                shutil.move(str(remote_copy), str(local_db))

        # Clean up staging dir.
        shutil.rmtree(str(staging_dir), ignore_errors=True)

    print("[pull-dbs] Done.")
    return 0


def main() -> int:
    args = parse_args()
    if args.pull_remote_only:
        args.download = True

    local_root = Path(args.local_root).resolve() if args.local_root else Path(__file__).resolve().parent

    # ---------- pull-dbs shortcut (bypasses full sync machinery) ----------
    if args.pull_dbs:
        hosts = args.hosts if args.hosts else ["sk3"]
        for host in hosts:
            rc = pull_dbs(host, local_root, dry_run=args.dry_run, merge=args.merge)
            if rc != 0:
                return rc
        return 0

    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(repo_root))

    from sync_with_sk.core import ProjectConfig, run_sync  # noqa: WPS433

    config = ProjectConfig(
        name="matt-debutts",
        local_root=local_root,
        scope_root=local_root,
        default_hosts=["sk1", "sk2", "sk3"],
        host_roots=HOST_ROOTS,
        fallback_remote_root="/lfs/skampere3/0/alexspan/matt-debutts",
        data_dirs=set(),
        data_suffixes=set(),
        data_allowlist=set(),
        code_suffixes=set(),
        include_unknown=True,
        ignore_paths=[local_root / ".sync-ignore"],
        max_hash_bytes=5 * 1024 * 1024,
        max_download_bytes=None,
        state_file_path=local_root / ".sync_with_sk_state.json",
        speed_grouping="data-dir",
    )

    return run_sync(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
