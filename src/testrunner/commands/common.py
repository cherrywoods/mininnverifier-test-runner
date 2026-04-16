# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Shared command builder for eval and grad."""

import shlex


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
