"""Foreground command guard (t_fe2b14bc).

A worker's foreground terminal command had no memory ceiling and no runtime
limit: a 1.9 GB test allocation froze the VPS twice on 2026-11-15 (swap
exhausted, load 112, gateway starved). The background spawn path already runs
under ``systemd-run --user --scope`` MemoryMax (#70716); these tests pin the
foreground guard that brings the same isolation — plus ``MemorySwapMax=0`` and
``RuntimeMaxSec`` — to every local foreground command spawned from inside a
``hermes-worker-*.scope`` cgroup, and the card-readable kill line the card
sees when a cap fires.
"""
from __future__ import annotations

import pytest

from tools import fg_guard


@pytest.fixture
def active_guard(monkeypatch):
    """Force the 'running inside a worker scope with a usable user bus' path."""
    monkeypatch.setattr(fg_guard, "_IS_LINUX", True)
    monkeypatch.setattr(fg_guard, "_in_worker_scope_cgroup", lambda: True)
    monkeypatch.setattr(fg_guard, "systemd_run_user_scope_usable", lambda: True)


class TestScopeWrap:
    def test_wraps_argv_in_systemd_scope(self, active_guard):
        argv = fg_guard.fg_guard_scope_argv(
            ["/bin/bash", "-c", "echo hello"], unit_suffix="42")
        assert argv is not None
        assert argv[0].endswith("systemd-run")
        assert "--user" in argv and "--scope" in argv and "--collect" in argv
        properties = [argv[i + 1] for i, v in enumerate(argv[:-1]) if v == "--property"]
        assert "MemoryAccounting=yes" in properties
        assert "MemorySwapMax=0" in properties, (
            "swap escape defeated the MemoryMax cap on 2026-11-15; every guard "
            "scope must set MemorySwapMax=0")
        memory_max = next(v for v in properties if v.startswith("MemoryMax="))
        assert int(memory_max.split("=", 1)[1]) == fg_guard._fg_memory_max_bytes()
        assert any(v.startswith("RuntimeMaxSec=") for v in properties)
        sep = argv.index("--")
        assert argv[sep + 1:] == ["/bin/bash", "-c", "echo hello"]

    def test_unit_name_unique_per_spawn(self, active_guard):
        """Parallel tool calls are routine: a reused --unit would make the second
        systemd-run fail with 'unit already loaded'."""
        units = []
        for _ in range(2):
            argv = fg_guard.fg_guard_scope_argv(["/bin/bash", "-c", "true"], unit_suffix="42")
            units.append(argv[argv.index("--unit") + 1])
        assert units[0] != units[1]
        assert units[0].startswith("hermes-worker-fg-42-")

    def test_no_wrap_outside_worker_scope(self, monkeypatch):
        monkeypatch.setattr(fg_guard, "_IS_LINUX", True)
        monkeypatch.setattr(fg_guard, "_in_worker_scope_cgroup", lambda: False)
        monkeypatch.setattr(fg_guard, "systemd_run_user_scope_usable", lambda: True)
        assert fg_guard.fg_guard_scope_argv(["/bin/bash", "-c", "true"], unit_suffix="1") is None

    def test_no_wrap_when_user_bus_unusable(self, monkeypatch):
        monkeypatch.setattr(fg_guard, "_IS_LINUX", True)
        monkeypatch.setattr(fg_guard, "_in_worker_scope_cgroup", lambda: True)
        monkeypatch.setattr(fg_guard, "systemd_run_user_scope_usable", lambda: False)
        assert fg_guard.fg_guard_scope_argv(["/bin/bash", "-c", "true"], unit_suffix="1") is None

    def test_guard_off_when_runtime_seconds_zero(self, monkeypatch, active_guard):
        monkeypatch.setenv("TERMINAL_FG_RUNTIME_SECONDS", "0")
        argv = fg_guard.fg_guard_scope_argv(["/bin/bash", "-c", "true"], unit_suffix="1")
        properties = [argv[i + 1] for i, v in enumerate(argv[:-1]) if v == "--property"]
        assert not any(v.startswith("RuntimeMaxSec=") for v in properties)


class TestMemoryMax:
    def test_default_floor(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_FG_MEMORY_MAX_MB", raising=False)
        bytes_limit = fg_guard._fg_memory_max_bytes()
        assert bytes_limit >= fg_guard._MIN_FG_MEMORY_MAX_BYTES  # the card's 1200M ceiling
        assert bytes_limit <= fg_guard._FG_MEMORY_MAX_CAP_BYTES

    def test_override_below_floor_is_floored_not_rejected(self, monkeypatch):
        """A tiny override would just re-create the unbounded path; floor it."""
        monkeypatch.setenv("TERMINAL_FG_MEMORY_MAX_MB", "100")
        assert fg_guard._fg_memory_max_bytes() == fg_guard._MIN_FG_MEMORY_MAX_BYTES

    def test_override_cannot_exceed_enclosing_bound(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_FG_MEMORY_MAX_MB", "10000")
        assert fg_guard._fg_memory_max_bytes() <= fg_guard._FG_MEMORY_MAX_CAP_BYTES
        # the safe bound never widens past half of physical RAM (like the worker path)
        import os
        try:
            physical = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, TypeError):
            physical = None
        if physical is not None:
            assert fg_guard._fg_memory_max_bytes() <= max(fg_guard._MIN_FG_MEMORY_MAX_BYTES, physical // 2)


class TestRuntimeSeconds:
    def test_default_has_a_ceiling(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_FG_RUNTIME_SECONDS", raising=False)
        assert fg_guard._fg_runtime_seconds() > 0

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_FG_RUNTIME_SECONDS", "0")
        assert fg_guard._fg_runtime_seconds() == 0

    def test_small_value_floored(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_FG_RUNTIME_SECONDS", "5")
        assert fg_guard._fg_runtime_seconds() == fg_guard._MIN_FG_RUNTIME_SECONDS

    def test_garbage_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_FG_RUNTIME_SECONDS", "banana")
        assert fg_guard._fg_runtime_seconds() == fg_guard._DEFAULT_FG_RUNTIME_SECONDS


class TestKillLine:
    def test_sigkill_names_memory_cap(self, active_guard):
        line = fg_guard.systemd_scope_oom_kill_line(137)
        assert line is not None
        assert line.startswith("[killed: memory cap ")
        assert "MemorySwapMax=0" in line and "SIGKILL" in line

    def test_sigkill_at_runtime_limit_names_runtime_cap(self, active_guard):
        """systemd escalates RuntimeMaxSec SIGTERM to SIGKILL, so a 137 that
        actually ran to the limit must not be mislabeled a memory kill."""
        runtime = fg_guard._fg_runtime_seconds() or 1800
        line = fg_guard.systemd_scope_oom_kill_line(137, elapsed_seconds=runtime)
        assert line is not None and "runtime cap" in line and "SIGKILL" in line

    def test_sigterm_names_memory_cap(self, active_guard):
        line = fg_guard.systemd_scope_oom_kill_line(143)
        assert line is not None and "SIGTERM" in line and "memory cap" in line

    def test_124_at_runtime_limit_names_runtime_cap(self, active_guard):
        runtime = fg_guard._fg_runtime_seconds() or 1800
        line = fg_guard.systemd_scope_oom_kill_line(124, elapsed_seconds=runtime)
        assert line is not None and "runtime cap" in line and "RuntimeMaxSec" in line

    def test_124_short_run_is_not_claimed_as_runtime_kill(self, active_guard):
        """A guarded command that merely exits 124 fast is today's timeout path,
        not the scope's RuntimeMaxSec — no false kill line."""
        assert fg_guard.systemd_scope_oom_kill_line(124, elapsed_seconds=3.0) is None

    def test_124_without_elapsed_reported(self, active_guard):
        assert fg_guard.systemd_scope_oom_kill_line(124) is not None

    def test_ordinary_exit_gets_no_line(self, active_guard):
        for code in (0, 1, 2, 130):
            assert fg_guard.systemd_scope_oom_kill_line(code) is None


class TestCgroupGate:
    def test_in_worker_scope_detection_reads_self_cgroup(self):
        """On this VPS the worker itself runs in hermes-worker-*.scope; outside
        one (dev laptop) the function is False. Either way: no crash, bool."""
        result = fg_guard._in_worker_scope_cgroup()
        assert isinstance(result, bool)

    def test_probe_usable_requires_xdg_runtime_dir(self, monkeypatch):
        monkeypatch.setattr(fg_guard, "_IS_LINUX", True)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        assert fg_guard.systemd_run_user_scope_usable() is False
