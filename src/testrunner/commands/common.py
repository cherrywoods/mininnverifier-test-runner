# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Shared helpers and command builders for eval and grad."""

import shlex
from pathlib import Path


DEFAULT_TIMEOUTS = {
    "eval": 60,
    "grad": 60,
    "train": 600,
    "fuzz_eval": 600,
    "fuzz_grad": 600,
}


def get_timeout(config):
    """Return timeout in seconds from config, falling back to per-command defaults."""
    if "timeout" in config:
        return config["timeout"]
    return DEFAULT_TIMEOUTS.get(config.get("command", ""), 60)


def parse_output_paths(stdout):
    """Parse file paths from subprocess stdout.

    Returns (existing_paths, warnings) where warnings list non-existent paths.
    """
    paths = []
    warnings = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.exists():
            paths.append(p)
        else:
            warnings.append(f"ignoring non-existent output path: {line}")
    return paths, warnings


def build_eval_grad_cmd(config, test_dir, output_dir, backend, backend_arg):
    """Shared builder for eval and grad — same argument structure."""
    command = config["command"]
    network = config["network"]
    inputs = config.get("inputs", [])

    if backend == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{test_dir.resolve()}:/data",
            backend_arg,
            command,
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network}",
            *[f"/data/{i}" for i in inputs],
        ]
        return cmd, None
    else:
        cmd = [
            *shlex.split(backend_arg),
            command,
            "--output-dir",
            str(output_dir),
            *[str(test_dir / i) for i in [network, *inputs]],
        ]
        return cmd, None
