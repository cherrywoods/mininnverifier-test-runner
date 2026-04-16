# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Test runner for mininnverifier-compatible implementations.

Usage:
    python -m testrunner <test_dir> docker <image>
    python -m testrunner <test_dir> local "<command>"
    python -m testrunner <test_dir> docker <image> --generate
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from testrunner.commands import COMMANDS, RUNNERS
from testrunner.check import CHECKS, DEFAULT_CHECKS


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single_test(test_dir, backend, backend_arg, generate=False):
    """Run a single test from a directory containing test.json."""
    config = json.loads((test_dir / "test.json").read_text())

    command = config["command"]

    output_dir = test_dir / "actual"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Custom runners handle the full test lifecycle
    if command in RUNNERS:
        return RUNNERS[command](test_dir, config, output_dir, backend, backend_arg)

    if command not in COMMANDS:
        return {"passed": False, "error": f"unknown command: {command}"}

    cmd, cwd = COMMANDS[command](config, test_dir, output_dir, backend, backend_arg)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

    if result.returncode != 0:
        return {
            "passed": False,
            "output_files": [],
            "error": f"command failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    output_files = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.exists():
            output_files.append(p)
        else:
            print(f"warning: ignoring non-existent output path: {line}", file=sys.stderr)

    if generate:
        for out_file in output_files:
            dest = test_dir / f"expected_{out_file.name}"
            shutil.copy2(out_file, dest)
        return {
            "generated": True,
            "output_files": [str(f) for f in output_files],
        }

    check_name = config.get("check", DEFAULT_CHECKS.get(command))
    if check_name is None:
        return {
            "passed": False,
            "error": f"no check specified and no default for command '{command}'",
        }
    if check_name not in CHECKS:
        return {"passed": False, "error": f"unknown check: {check_name}"}

    check_result = CHECKS[check_name](test_dir, config, output_files)
    return {
        "passed": check_result["passed"],
        "output_files": [str(f) for f in output_files],
        "error": check_result["error"],
    }


def run_tests(root_dir, backend, backend_arg, generate=False):
    """Discover and run all tests under root_dir.

    Returns a nested dict mirroring the directory structure from root_dir
    down to each test.json.
    """
    root_dir = Path(root_dir).resolve()
    cwd = Path.cwd().resolve()

    # Discover all test directories
    test_dirs = sorted(p.parent for p in root_dir.rglob("test.json"))

    if not test_dirs:
        return {}

    results = {}
    for test_dir in test_dirs:
        rel = test_dir.relative_to(cwd)
        test_result = run_single_test(test_dir, backend, backend_arg, generate)

        # Nest the result according to the path relative to cwd
        node = results
        parts = rel.parts
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = test_result

    return results


def _any_failed(node):
    """Recursively check whether any test in a nested result dict failed."""
    if "passed" in node:
        return not node["passed"]
    return any(_any_failed(v) for v in node.values() if isinstance(v, dict))


def main():
    parser = argparse.ArgumentParser(description="Run a mininnverifier test.")
    parser.add_argument("test_dir", type=str)
    parser.add_argument("backend", choices=["docker", "local"])
    parser.add_argument("backend_arg", type=str, help="Docker image name or local command")
    parser.add_argument(
        "--generate", action="store_true", help="Generate expected outputs instead of checking"
    )
    args = parser.parse_args()

    results = run_tests(args.test_dir, args.backend, args.backend_arg, args.generate)
    print(json.dumps(results, indent=2))
    if _any_failed(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
