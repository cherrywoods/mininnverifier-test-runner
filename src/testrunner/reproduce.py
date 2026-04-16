# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Reproduce a saved fuzz failure.

Usage:
    python -m testrunner.reproduce <backend> <backend_arg> <failure_dir>

Where <failure_dir> is a directory previously saved by the fuzz runner
(containing network.mininn, input .bin files, and metadata.json).
"""

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from testrunner.fuzz.runner import run_and_check


def reproduce(failure_dir, backend, backend_arg):
    """Reproduce a single saved fuzz failure.

    Copies the saved network and inputs into a temporary directory and
    reruns the fuzz checking logic.

    Returns a result dict with ``passed`` and ``error`` keys.
    """
    failure_dir = Path(failure_dir).resolve()

    metadata_path = failure_dir / "metadata.json"
    if not metadata_path.exists():
        return {"passed": False, "error": f"no metadata.json in {failure_dir}"}

    metadata = json.loads(metadata_path.read_text())
    mode = metadata["mode"]
    check_nan_inf = metadata["check_nan_inf"]
    expected_shapes = [tuple(s) for s in metadata["expected_shapes"]]
    input_names = metadata["inputs"]

    network_path = failure_dir / "network.mininn"
    if not network_path.exists():
        return {"passed": False, "error": f"no network.mininn in {failure_dir}"}

    input_paths = []
    for name in input_names:
        p = failure_dir / name
        if not p.exists():
            return {"passed": False, "error": f"missing input file: {name}"}
        input_paths.append(p)

    # Work in a temporary directory so output files don't pollute the failure dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_network = tmp_dir / "network.mininn"
        shutil.copy2(network_path, tmp_network)
        tmp_inputs = []
        for p in input_paths:
            dest = tmp_dir / p.name
            shutil.copy2(p, dest)
            tmp_inputs.append(dest)

        return run_and_check(
            tmp_network,
            tmp_inputs,
            backend,
            backend_arg,
            mode,
            expected_shapes,
            check_nan_inf=check_nan_inf,
        )


def main():
    parser = argparse.ArgumentParser(description="Reproduce a saved fuzz failure.")
    parser.add_argument("backend", choices=["docker", "local"])
    parser.add_argument("backend_arg", type=str, help="Docker image name or local command")
    parser.add_argument("failure_dir", type=str, help="Path to saved failure directory")
    args = parser.parse_args()

    failure_dir = Path(args.failure_dir).resolve()

    _DIM = "\033[2m"
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _RESET = "\033[0m"

    # Print network and inputs before running
    _print_failure_inputs(failure_dir)

    result = reproduce(failure_dir, args.backend, args.backend_arg)

    if result["passed"]:
        print(f"{_GREEN}PASS: failure did not reproduce{_RESET}", file=sys.stderr)
    else:
        print(f"{_RED}FAIL: {result['error']}{_RESET}", file=sys.stderr)
        if result.get("stdout"):
            print(f"\n{_DIM}--- stdout ---{_RESET}", file=sys.stderr)
            print(result["stdout"], file=sys.stderr)
            print(f"{_DIM}--- end stdout ---{_RESET}", file=sys.stderr)
        if result.get("stderr"):
            print(f"\n{_DIM}--- stderr ---{_RESET}", file=sys.stderr)
            print(result["stderr"], file=sys.stderr)
            print(f"{_DIM}--- end stderr ---{_RESET}", file=sys.stderr)
        sys.exit(1)


def _print_failure_inputs(failure_dir):
    """Print the network and input arrays from a saved failure directory."""
    network_path = failure_dir / "network.mininn"
    if network_path.exists():
        print(f"Network ({network_path}):", file=sys.stderr)
        with zipfile.ZipFile(network_path) as z:
            graph = z.read("graph.txt").decode()
        for line in graph.strip().splitlines():
            print(f"  {line}", file=sys.stderr)
        print(file=sys.stderr)

    metadata_path = failure_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        input_names = metadata.get("inputs", [])
    else:
        input_names = sorted(p.name for p in failure_dir.iterdir() if p.suffix == ".bin")

    if input_names:
        print("Inputs:", file=sys.stderr)
        for i, name in enumerate(input_names):
            p = failure_dir / name
            if p.exists():
                arr = np.fromfile(p, dtype=np.float64)
                print(f"  input {i} ({name}):", file=sys.stderr)
                print(f"    shape: {arr.shape}", file=sys.stderr)
                for line in np.array2string(arr, max_line_width=100).splitlines():
                    print(f"    {line}", file=sys.stderr)
                print(file=sys.stderr)

    error_path = failure_dir / "error.txt"
    if error_path.exists():
        print(f"Original error: {error_path.read_text().strip()}", file=sys.stderr)
        print(file=sys.stderr)


if __name__ == "__main__":
    main()
