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

from testrunner.commands import COMMANDS, RUNNERS
from testrunner.commands.common import get_timeout, parse_output_paths, run_subprocess
from testrunner.check import CHECKS, DEFAULT_CHECKS
from testrunner.output import CliOutputHandler, JsonOutputHandler


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single_test(test_dir, backend, backend_arg, generate=False,
                    output_handler=None):
    """Run a single test from a directory containing test.json."""
    config = json.loads((test_dir / "test.json").read_text())

    command = config["command"]

    output_dir = test_dir / "actual"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Custom runners handle the full test lifecycle
    if command in RUNNERS:
        return RUNNERS[command](test_dir, config, output_dir, backend, backend_arg,
                                generate=generate, output_handler=output_handler)

    if command not in COMMANDS:
        return {"passed": False, "error": f"unknown command: {command}"}

    cmd, cwd = COMMANDS[command](config, test_dir, output_dir, backend, backend_arg)
    timeout = get_timeout(config)
    try:
        result = run_subprocess(
            cmd, cwd=cwd, timeout=timeout,
            log_file=output_dir / "stdout.log",
            output_handler=output_handler,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output_files": [],
            "error": f"command timed out after {timeout}s",
        }

    if result.returncode != 0:
        return {
            "passed": False,
            "output_files": [],
            "error": f"command failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    output_files, warnings = parse_output_paths(result.stdout)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    if generate:
        for out_file in output_files:
            dest = test_dir / f"expected_{out_file.name}"
            shutil.copy2(out_file, dest)
        return {
            "passed": True,
            "error": None,
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


def run_tests(root_dir, backend, backend_arg, generate=False,
              output_handler=None):
    """Discover and run all tests under root_dir.

    Returns a flat list of (test_path, result_dict) tuples.
    """
    root_dir = Path(root_dir).resolve()

    # Discover all test directories
    test_dirs = sorted(p.parent for p in root_dir.rglob("test.json"))

    if not test_dirs:
        return []

    total = len(test_dirs)
    results = []
    n_passed = 0
    n_failed = 0

    for i, test_dir in enumerate(test_dirs):
        test_path = str(test_dir.relative_to(root_dir))

        if output_handler is not None:
            output_handler.test_starting(test_path, i + 1, total)

        test_result = run_single_test(
            test_dir, backend, backend_arg, generate,
            output_handler=output_handler,
        )
        results.append((test_path, test_result))

        if test_result.get("passed", False):
            n_passed += 1
        else:
            n_failed += 1

        if output_handler is not None:
            output_handler.test_finished(test_path, test_result)

    if output_handler is not None:
        output_handler.all_finished(n_passed, n_failed)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run a mininnverifier test.")
    parser.add_argument("test_dir", type=str)
    parser.add_argument("backend", choices=["docker", "local"])
    parser.add_argument("backend_arg", type=str, help="Docker image name or local command")
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate expected outputs instead of checking",
    )
    parser.add_argument(
        "--output", choices=["cli", "json"], default="cli",
        help="Output mode: cli (default) for interactive display, json for JSONL",
    )
    args = parser.parse_args()

    if args.output == "json":
        handler = JsonOutputHandler()
    else:
        handler = CliOutputHandler()

    results = run_tests(
        args.test_dir, args.backend, args.backend_arg, args.generate,
        output_handler=handler,
    )
    if any(not r["passed"] for _, r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
