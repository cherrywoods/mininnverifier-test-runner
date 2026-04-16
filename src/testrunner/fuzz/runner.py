# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Fuzz test runners for eval and grad commands.

Each runner generates random compute graphs, runs the SUT, and checks
for crashes, NaN/Inf, and shape correctness.

Graphs from the "safe" primitive pool (no log/sqrt/reciprocal) are also
checked for NaN/Inf in the output. Graphs from the full pool only check
for crashes and shape correctness, since NaN/Inf are mathematically
expected for certain inputs to those operations.
"""

import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings, HealthCheck, Phase, seed as hypothesis_seed, strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from testrunner.commands.common import get_timeout, parse_output_paths
from .graph_builder import (
    generate_graph, serialize_graph, Graph,
    ALL_PRIMITIVES, SAFE_PRIMITIVES,
)


def run_fuzz_eval(test_dir, config, output_dir, backend, backend_arg, generate=False):
    """Fuzz runner for eval command."""
    if generate:
        import warnings
        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(test_dir, config, output_dir, backend, backend_arg, mode="eval")


def run_fuzz_grad(test_dir, config, output_dir, backend, backend_arg, generate=False):
    """Fuzz runner for grad command."""
    if generate:
        import warnings
        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(test_dir, config, output_dir, backend, backend_arg, mode="grad")


@st.composite
def _graph_with_inputs(draw, primitives=None):
    """Composite strategy that generates a graph and matching input arrays."""
    graph = generate_graph(draw, primitives=primitives)
    inputs = {}
    for var in graph.invars:
        data = draw(np_arrays(
            dtype=np.float64,
            shape=var.shape,
            elements=st.floats(
                min_value=-2.0, max_value=2.0,
                allow_nan=False, allow_infinity=False,
            ),
        ))
        inputs[var.name] = data
    return graph, inputs


def _run_fuzz(test_dir, config, output_dir, backend, backend_arg, mode):
    n_trials = config.get("n_trials", 100)
    timeout = get_timeout(config)
    safe_trials = n_trials // 2
    full_trials = n_trials - safe_trials

    all_results = []

    # Run safe pool (NaN/Inf checked) and full pool (only crash + shape)
    for pool_name, pool_primitives, pool_n_trials, check_nan_inf in [
        ("safe", SAFE_PRIMITIVES, safe_trials, True),
        ("full", ALL_PRIMITIVES, full_trials, False),
    ]:
        if pool_n_trials == 0:
            continue

        results = []

        @given(data=_graph_with_inputs(primitives=pool_primitives))
        @settings(
            max_examples=pool_n_trials,
            database=None,
            phases=[Phase.explicit, Phase.generate],
            suppress_health_check=list(HealthCheck),
        )
        def fuzz_trial(data):
            graph, inputs = data
            result = _run_single_trial(
                graph, inputs, output_dir, backend, backend_arg, mode,
                check_nan_inf=check_nan_inf, timeout=timeout,
            )
            results.append(result)

        if "seed" in config:
            fuzz_trial = hypothesis_seed(config["seed"])(fuzz_trial)

        try:
            fuzz_trial()
        except Exception:
            pass

        for r in results:
            r["pool"] = pool_name
        all_results.extend(results)

    n_passed = sum(1 for r in all_results if r["passed"])
    failures = [
        {"trial": i, "pool": r["pool"], "error": r["error"]}
        for i, r in enumerate(all_results)
        if not r["passed"]
    ]

    return {
        "passed": n_passed == len(all_results),
        "error": None if n_passed == len(all_results) else f"{len(all_results) - n_passed} trial(s) failed",
        "n_trials": len(all_results),
        "n_passed": n_passed,
        "failures": failures,
    }


def _run_single_trial(graph, inputs, output_dir, backend, backend_arg, mode,
                      check_nan_inf=True, timeout=60):
    """Run a single fuzz trial."""
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Write .mininn file
        network_path = tmp_dir / "network.mininn"
        network_path.write_bytes(serialize_graph(graph))

        # Write input .bin files
        input_paths = []
        for var in graph.invars:
            input_path = tmp_dir / f"{var.name}.bin"
            inputs[var.name].tofile(input_path)
            input_paths.append(input_path)

        # Build and run the SUT command
        cmd = _build_cmd(
            mode, network_path, input_paths,
            tmp_dir / "output", backend, backend_arg, tmp_dir,
        )
        (tmp_dir / "output").mkdir()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": f"timed out after {timeout}s",
            }

        if result.returncode != 0:
            return {
                "passed": False,
                "error": f"crash (exit {result.returncode}): {result.stderr.strip()[:500]}",
            }

        # Parse output file paths from stdout
        output_files, _ = parse_output_paths(result.stdout)

        if not output_files:
            return {"passed": False, "error": "no output files produced"}

        # Determine expected shapes
        if mode == "eval":
            expected_shapes = [v.shape for v in graph.outvars]
        else:
            expected_shapes = [v.shape for v in graph.invars]

        if len(output_files) != len(expected_shapes):
            return {
                "passed": False,
                "error": (
                    f"expected {len(expected_shapes)} output file(s), "
                    f"got {len(output_files)}"
                ),
            }

        # Check each output
        for out_file, expected_shape in zip(output_files, expected_shapes):
            data = np.fromfile(out_file, dtype=np.float64)
            expected_size = 1
            for d in expected_shape:
                expected_size *= d

            if data.size != expected_size:
                return {
                    "passed": False,
                    "error": (
                        f"{out_file.name}: expected {expected_size} values "
                        f"(shape {expected_shape}), got {data.size}"
                    ),
                }

            if check_nan_inf:
                if np.any(np.isnan(data)):
                    return {
                        "passed": False,
                        "error": f"{out_file.name}: output contains NaN",
                    }
                if np.any(np.isinf(data)):
                    return {
                        "passed": False,
                        "error": f"{out_file.name}: output contains Inf",
                    }

    return {"passed": True, "error": None}


def _build_cmd(mode, network_path, input_paths, output_dir, backend, backend_arg, test_dir):
    """Build the CLI command for eval or grad."""
    if backend == "docker":
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{test_dir.resolve()}:/data",
            backend_arg,
            mode,
            "--output-dir", f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network_path.relative_to(test_dir)}",
            *[f"/data/{p.relative_to(test_dir)}" for p in input_paths],
        ]
    else:
        cmd = [
            *shlex.split(backend_arg),
            mode,
            "--output-dir", str(output_dir),
            str(network_path),
            *[str(p) for p in input_paths],
        ]
    return cmd
