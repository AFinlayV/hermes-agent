"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env -- the sanitizer's other decisions (dropped runtime
site-packages, dropped venv markers) stand.

The pin alone still leaves the worker on a bare store interpreter carrying none of
hermes-agent's third-party dependencies: the gateway imports them through its own
PM activation at startup, but this child never runs a bootstrap. The selected dependency
environment's site-packages therefore ride along on PYTHONPATH (gateway-ticked jobs died
on ``from ruamel.yaml import YAML``). Appended after the pin, never prepended: entries
the sanitizer kept keep their place ahead of the dependency tree (its stdlib-adjacent
shadows must not outrank them), and the worker's own stdlib still wins over everything.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _dependency_site_packages(repo_root: Path) -> str | None:
    """Site-packages of PM's selected dependency environment, or ``None``.

    ``committed_venv`` (not ``selected_venv``): the in-tree ``venv``/``.venv`` fallback
    belongs to whichever interpreter built it historically, and loading those packages
    from the store Python mixes ABIs (#122183). The site path is computed from the
    environment record (its own ``pyvenv.cfg`` version), never this process's
    ``sys.version_info``. Built lazily so the import lands only on the spawn path, and
    it never raises: a missing/unreadable record or a vanished tree degrades to the
    pre-existing environment, exactly as before this helper existed.
    """
    try:
        from pm.environments import committed_venv, site_packages

        venv = committed_venv(Path(repo_root))
        if venv is None:
            return None
        site = site_packages(venv)
        if not site.is_dir():
            return None
        return str(site)
    except Exception:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` to the worker env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``. Under a
    source-tree install the selected dependency environment's site-packages are appended
    so the bare store-Python worker can import third-party dependencies at all.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    entries = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    dep_site = _dependency_site_packages(repo_root)
    if dep_site:
        entries.append(dep_site)
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *entries]))
    return worker_env