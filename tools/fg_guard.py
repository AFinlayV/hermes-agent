"""Foreground command guard: per-command memory cap + runtime limit + clear kill line.

Card t_fe2b14bc (2026-11-16): a worker's foreground terminal command (e.g. a test
that allocates 1.9 GB) could freeze the whole VPS — the background spawn path gets a
``systemd-run --user --scope`` MemoryMax (#70716) but the foreground path ran bare,
and nothing bounded runtime inside the worker, so an OOM killed the *worker* itself
(the card re-ran the same test, twice) and swap filled until the box thrashed.

Mechanism, not instructions in a prompt (card requirement):
  * When the spawning process runs in a ``hermes-worker-*.scope`` cgroup (kanban
    background executors, cron) and ``systemd-run --user`` is usable, foreground
    commands are wrapped in a fresh transient scope with ``MemoryMax``
    (``TERMINAL_FG_MEMORY_MAX_MB``, floor 1200M, cap 2 GiB; safety bound = the
    spawning cgroup's ``memory.max`` / half of physical RAM) and
    ``MemorySwapMax=0`` (the November freeze: the escape hatch was swap, not RAM)
    plus ``RuntimeMaxSec`` (``TERMINAL_FG_RUNTIME_SECONDS``, 0 disables; default 1800)
    so the scope outlives the command's own timeout backstop.
  * When a command is killed by that scope, bash sees exit 137/143/124 and the
    terminal result gains a line the card can read and act on:
    ``[killed: memory cap 1200M (systemd scope MemorySwapMax=0; SIGKILL)]`` —
    with which runtime limit fired when recognizable. The card reports blocked
    instead of retrying.
  * Everything is probed and cached; any failure falls back to today's argv
    unchanged (never a new way for a terminal command to break).
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional

_IS_LINUX = (os.name == "posix" and shutil.which("systemd-run") is not None
             and Path("/run/systemd/system").exists())

_MIN_FG_MEMORY_MAX_BYTES = 1200 * 1024 * 1024
_DEFAULT_FG_MEMORY_MAX_BYTES = 1200 * 1024 * 1024
_FG_MEMORY_MAX_CAP_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_FG_RUNTIME_SECONDS = 1800
_MIN_FG_RUNTIME_SECONDS = 60

_SCOPE_PROBE_LOCK = threading.Lock()
_SCOPE_PROBABLE: Optional[bool] = None
_SCOPE_PROBED_AT = 0.0
_SCOPE_PROBE_TTL_SECONDS = 60.0


def _fg_memory_max_bytes() -> int:
    """MemoryMax for the foreground guard scope: ``TERMINAL_FG_MEMORY_MAX_MB`` when
    it tightens the safe bound (the spawning cgroup's ``memory.max``, else half of
    physical RAM, capped at 2 GiB), floor 1200 MiB — the card's ceiling."""
    override_bytes: Optional[int] = None
    override = os.getenv("TERMINAL_FG_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
            # A tiny override would silently re-create the unbounded path; floor it.
            override_bytes = max(parsed, _MIN_FG_MEMORY_MAX_BYTES)
        except ValueError:
            pass
    candidates: List[int] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(
                    encoding="utf-8-sig").strip()
                if raw.isdigit() and int(raw) >= _MIN_FG_MEMORY_MAX_BYTES:
                    candidates.append(int(raw))
                break
    except (OSError, ValueError):
        pass
    try:
        physical = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(min(_FG_MEMORY_MAX_CAP_BYTES, max(_MIN_FG_MEMORY_MAX_BYTES, physical // 2)))
    except (OSError, ValueError, TypeError):
        pass
    safe = min(candidates) if candidates else _DEFAULT_FG_MEMORY_MAX_BYTES
    if override_bytes:
        return min(override_bytes, safe)
    return min(_DEFAULT_FG_MEMORY_MAX_BYTES, safe)


def _fg_runtime_seconds() -> int:
    """RuntimeMaxSec for the guard scope. 0 disables the runtime ceiling."""
    raw = os.getenv("TERMINAL_FG_RUNTIME_SECONDS", "").strip()
    try:
        seconds = int(raw) if raw else _DEFAULT_FG_RUNTIME_SECONDS
    except ValueError:
        return _DEFAULT_FG_RUNTIME_SECONDS
    if seconds == 0:
        return 0
    return max(_MIN_FG_RUNTIME_SECONDS, seconds)


def _in_worker_scope_cgroup() -> bool:
    """True when this process already lives in a ``hermes-worker-*.scope`` cgroup
    (kanban background executors, dispatch scopes) — the population an unbounded
    command can freeze and the one whose scope OOM kills our own worker."""
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::") and "hermes-worker-" in line and line.rstrip().endswith(".scope"):
                return True
    except OSError:
        pass
    return False


def systemd_run_user_scope_usable() -> bool:
    """Can a fresh ``systemd-run --user --scope`` actually start here? The binary
    can exist while the user D-Bus is absent (system services, some containers);
    probe a real /bin/true scope, cache briefly."""
    global _SCOPE_PROBABLE, _SCOPE_PROBED_AT
    if not _IS_LINUX:
        return False
    if os.environ.get("XDG_RUNTIME_DIR", "").strip() == "":
        return False
    with _SCOPE_PROBE_LOCK:
        if _SCOPE_PROBABLE is not None and (time.monotonic() - _SCOPE_PROBED_AT) < _SCOPE_PROBE_TTL_SECONDS:
            return _SCOPE_PROBABLE
        ok = False
        try:
            import subprocess
            probe_env = dict(os.environ)
            probe_env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
            result = subprocess.run(
                ["systemd-run", "--user", "--scope", "--quiet", "--unit",
                 f"hermes-fg-probe-{os.getpid()}", "--collect", "--", "/bin/true"],
                capture_output=True, timeout=20, env=probe_env)
            ok = result.returncode == 0
        except Exception:
            ok = False
        _SCOPE_PROBABLE = ok
        _SCOPE_PROBED_AT = time.monotonic()
        return ok


def guard_active() -> bool:
    """Should foreground commands run under the guard scope?"""
    return _in_worker_scope_cgroup() and systemd_run_user_scope_usable()


def fg_guard_scope_argv(shell_argv: List[str], *, unit_suffix: str) -> Optional[List[str]]:
    """Wrap *shell_argv* (a full ``bash -c`` argv) in the guard scope; None = no wrap.
    The unit name is unique per spawn: two concurrent foreground commands in the same
    process are routine (parallel tool calls), and a reused --unit would make the
    second systemd-run fail with 'unit already loaded'."""
    if not guard_active():
        return None
    binary = shutil.which("systemd-run")
    if not binary:
        return None
    import uuid

    argv: List[str] = [binary, "--user", "--scope", "--quiet", "--unit",
                       f"hermes-worker-fg-{unit_suffix}-{uuid.uuid4().hex[:8]}", "--collect",
                       "--property", "MemoryAccounting=yes",
                       "--property", f"MemoryMax={_fg_memory_max_bytes()}",
                       "--property", "MemorySwapMax=0"]
    runtime = _fg_runtime_seconds()
    if runtime:
        argv += ["--property", f"RuntimeMaxSec={runtime}"]
    argv += ["--", *shell_argv]
    return argv


def fg_guard_memory_max_mb() -> int:
    return _fg_memory_max_bytes() // (1024 * 1024)


def systemd_scope_oom_kill_line(returncode: int, *, elapsed_seconds: Optional[float] = None) -> Optional[str]:
    """The human/card-readable line for a command killed by the guard scope,
    or None when the exit does not look like a scope kill. ``elapsed_seconds``
    gates the runtime-cap label: a command that actually ran ~to RuntimeMaxSec
    reports the runtime cap even when the kill arrived as a signal (systemd
    escalates SIGTERM to SIGKILL, so 137/143 is ambiguous between a MemoryMax
    OOM-kill and a RuntimeMaxSec expiry — only the elapsed time tells them
    apart). A 124 that exits fast is an ordinary ``timeout`` exit, not claimed."""
    runtime = _fg_runtime_seconds()
    # Positive 137/143: bash survived and reported its killed child.
    # Negative -9/-15: the scope chain itself died by signal (systemd kills
    # the whole cgroup on MemoryMax OOM / RuntimeMaxSec expiry).
    sig = {137: "SIGKILL", 143: "SIGTERM", -9: "SIGKILL", -15: "SIGTERM"}.get(returncode)
    if sig is not None:
        if runtime and elapsed_seconds is not None and elapsed_seconds >= runtime * 0.9:
            return f"[killed: runtime cap {runtime}s (systemd scope RuntimeMaxSec; {sig})]"
        return (f"[killed: memory cap {fg_guard_memory_max_mb()}M "
                f"(systemd scope MemorySwapMax=0; {sig})]")
    if returncode == 124:  # RuntimeMaxSec expiry surfaces as bash exit 124
        if runtime and (elapsed_seconds is None or elapsed_seconds >= runtime * 0.9):
            return f"[killed: runtime cap {runtime}s (systemd scope RuntimeMaxSec)]"
    return None
