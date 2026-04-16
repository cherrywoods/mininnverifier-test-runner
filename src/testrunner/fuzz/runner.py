# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Fuzz test runners for eval and grad commands.

Each runner generates random compute graphs, runs the SUT, and checks
for crashes, NaN/Inf, and shape correctness.

The test.json for each fuzz test specifies which primitives to use and
whether to check for NaN/Inf in the output:

    {
        "command": "fuzz_eval",
        "n_trials": 500,
        "seed": 0,
        "primitives": "safe",
        "check_nan_inf": true
    }

``primitives`` can be ``"all"`` (default), ``"safe"`` (excludes
log/sqrt/reciprocal), or an explicit list of primitive names.
``check_nan_inf`` defaults to ``false``.
"""

import json as json_mod
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from hypothesis import (
    given,
    settings,
    HealthCheck,
    Phase,
    seed as hypothesis_seed,
    strategies as st,
)
from hypothesis.extra.numpy import arrays as np_arrays

from testrunner.commands.common import get_timeout, parse_output_paths
from .graph_builder import generate_graph, serialize_graph, ALL_PRIMITIVES, SAFE_PRIMITIVES

PRIMITIVE_SETS = {"all": ALL_PRIMITIVES, "safe": SAFE_PRIMITIVES}


def resolve_primitives(spec):
    """Resolve a primitives spec from test.json to a list of primitive names.

    ``spec`` can be a string key (``"all"``, ``"safe"``) or an explicit
    list of primitive names.  Raises ``ValueError`` on unknown keys or
    invalid primitive names.
    """
    if isinstance(spec, str):
        if spec not in PRIMITIVE_SETS:
            raise ValueError(
                f"unknown primitive set {spec!r}, "
                f"expected one of {sorted(PRIMITIVE_SETS)} or an explicit list"
            )
        return PRIMITIVE_SETS[spec]
    if isinstance(spec, list):
        unknown = set(spec) - set(ALL_PRIMITIVES)
        if unknown:
            raise ValueError(f"unknown primitives: {sorted(unknown)}")
        return spec
    raise ValueError(f"primitives must be a string or list, got {type(spec).__name__}")


def run_fuzz_eval(
    test_dir,
    config,
    output_dir,
    backend,
    backend_arg,
    generate=False,
    output_handler=None,
    closed=False,
):
    """Fuzz runner for eval command."""
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="eval",
        output_handler=output_handler,
        closed=closed,
    )


def run_fuzz_grad(
    test_dir,
    config,
    output_dir,
    backend,
    backend_arg,
    generate=False,
    output_handler=None,
    closed=False,
):
    """Fuzz runner for grad command."""
    if generate:
        import warnings

        warnings.warn("--generate has no effect for fuzz tests")
        return {"passed": True, "error": None, "generated": False}
    return _run_fuzz(
        test_dir,
        config,
        output_dir,
        backend,
        backend_arg,
        mode="grad",
        output_handler=output_handler,
        closed=closed,
    )


@st.composite
def _graph_with_inputs(draw, primitives=None):
    """Composite strategy that generates a graph and matching input arrays."""
    graph = generate_graph(draw, primitives=primitives)
    inputs = {}
    for var in graph.invars:
        data = draw(
            np_arrays(
                dtype=np.float64,
                shape=var.shape,
                elements=st.floats(
                    min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        inputs[var.name] = data
    return graph, inputs


def _run_fuzz(
    test_dir, config, output_dir, backend, backend_arg, mode, output_handler=None, closed=False
):
    n_trials = config.get("n_trials", 100)
    timeout = get_timeout(config)
    primitives = resolve_primitives(config.get("primitives", "all"))
    check_nan_inf = config.get("check_nan_inf", False)

    # For open tests, save up to MAX_SAVED failing cases to disk,
    # deduplicated by error message (one example per distinct error).
    # When multiple trials hit the same error, keep the smallest example
    # (by serialized network length + total input size).
    MAX_SAVED = 10
    save_dir = None
    saved_count = [0]
    # error -> (saved_path, size)
    saved_errors = {}
    if not closed:
        save_dir = output_dir / "fuzz_failures"
        save_dir.mkdir(exist_ok=True)

    results = []
    trial_counter = [0]
    test_path = str(test_dir.name)

    @given(data=_graph_with_inputs(primitives=primitives))
    @settings(
        max_examples=n_trials,
        database=None,
        phases=[Phase.explicit, Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    def fuzz_trial(data):
        graph, inputs = data
        # Always attempt to save when save_dir is set — the dedup
        # logic below will discard or replace as needed.
        trial_save_dir = save_dir
        result = _run_single_trial(
            graph,
            inputs,
            output_dir,
            backend,
            backend_arg,
            mode,
            check_nan_inf=check_nan_inf,
            timeout=timeout,
            save_dir=trial_save_dir,
        )
        if not result["passed"] and result.get("saved_to"):
            trial_size = len(serialize_graph(graph)) + sum(arr.nbytes for arr in inputs.values())
            prev = saved_errors.get(result["error"])
            if prev is not None and trial_size < prev[1]:
                # Smaller example for the same error — replace
                shutil.rmtree(prev[0], ignore_errors=True)
                saved_errors[result["error"]] = (result["saved_to"], trial_size)
            elif prev is not None:
                # Duplicate but not smaller — discard
                shutil.rmtree(result["saved_to"], ignore_errors=True)
                result["saved_to"] = None
            elif saved_count[0] < MAX_SAVED:
                # New error and under the cap — keep
                saved_errors[result["error"]] = (result["saved_to"], trial_size)
                saved_count[0] += 1
            else:
                # New error but already at cap — discard
                shutil.rmtree(result["saved_to"], ignore_errors=True)
                result["saved_to"] = None
        results.append(result)
        trial_counter[0] += 1
        if output_handler is not None:
            output_handler.fuzz_trial_finished(
                test_path, trial_counter[0], n_trials, result["passed"], result["error"]
            )

    if "seed" in config:
        fuzz_trial = hypothesis_seed(config["seed"])(fuzz_trial)

    try:
        fuzz_trial()
    except Exception:
        pass

    n_passed = sum(1 for r in results if r["passed"])
    n_failed = len(results) - n_passed

    result = {
        "passed": n_failed == 0,
        "error": None if n_failed == 0 else f"{n_failed} trial(s) failed",
        "n_trials": len(results),
        "n_passed": n_passed,
    }

    if not closed:
        failures = [
            {"trial": i, "error": r["error"]} for i, r in enumerate(results) if not r["passed"]
        ]
        result["failures"] = failures

        # Build grouped failure summary: error message -> count
        error_counts = Counter(r["error"] for r in results if not r["passed"])
        if error_counts:
            result["failure_summary"] = dict(error_counts.most_common())

        if save_dir is not None and saved_count[0] > 0:
            result["saved_failures_dir"] = str(save_dir)

    return result


def _run_single_trial(
    graph,
    inputs,
    output_dir,
    backend,
    backend_arg,
    mode,
    check_nan_inf=True,
    timeout=60,
    save_dir=None,
):
    """Run a single fuzz trial.

    Args:
        save_dir: If not None and the trial fails, copy the network and
            input files to a subdirectory of save_dir for reproduction.
    """
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

        # Determine expected shapes
        if mode == "eval":
            expected_shapes = [v.shape for v in graph.outvars]
        else:
            expected_shapes = [v.shape for v in graph.invars]

        result = run_and_check(
            network_path,
            input_paths,
            backend,
            backend_arg,
            mode,
            expected_shapes,
            check_nan_inf=check_nan_inf,
            timeout=timeout,
        )

        if not result["passed"] and save_dir is not None:
            result["saved_to"] = _save_failure(
                save_dir,
                network_path,
                input_paths,
                mode=mode,
                check_nan_inf=check_nan_inf,
                expected_shapes=expected_shapes,
                error=result.get("error"),
                stderr=result.get("stderr"),
            )

    return result


def run_and_check(
    network_path,
    input_paths,
    backend,
    backend_arg,
    mode,
    expected_shapes,
    check_nan_inf=True,
    timeout=60,
):
    """Run the SUT on a network/inputs and check the output.

    This is the core fuzz-checking logic, usable both from fuzz trials
    and from reproduction of saved failures.  The network and input files
    must already exist on disk.  An ``output`` subdirectory is created
    next to the network file for SUT output.

    Returns a result dict with at least ``passed`` and ``error`` keys.
    """
    network_path = Path(network_path)
    input_paths = [Path(p) for p in input_paths]
    work_dir = network_path.parent

    cmd = _build_cmd(
        mode, network_path, input_paths, work_dir / "output", backend, backend_arg, work_dir
    )
    (work_dir / "output").mkdir(exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        error = f"timed out after {timeout}s"
        return {"passed": False, "error": error}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        error = f"crash (exit {result.returncode}): {_truncate(stderr)}"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    # Parse output file paths from stdout
    output_files, _ = parse_output_paths(result.stdout)

    if not output_files:
        error = "no output files produced"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    if len(output_files) != len(expected_shapes):
        error = f"expected {len(expected_shapes)} output file(s), got {len(output_files)}"
        return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    # Check each output
    for out_file, expected_shape in zip(output_files, expected_shapes):
        data = np.fromfile(out_file, dtype=np.float64)
        expected_size = 1
        for d in expected_shape:
            expected_size *= d

        if data.size != expected_size:
            error = (
                f"{out_file.name}: expected {expected_size} values "
                f"(shape {expected_shape}), got {data.size}"
            )
            return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

        if check_nan_inf:
            if np.any(np.isnan(data)):
                error = f"{out_file.name}: output contains NaN"
                return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}
            if np.any(np.isinf(data)):
                error = f"{out_file.name}: output contains Inf"
                return {"passed": False, "error": error, "stdout": stdout, "stderr": stderr}

    return {"passed": True, "error": None}


def _save_failure(
    save_dir,
    network_path,
    input_paths,
    mode,
    check_nan_inf,
    expected_shapes,
    error=None,
    stderr=None,
):
    """Copy the network and input files to save_dir for reproduction.

    Returns the path to the saved directory, or None if saving is disabled.
    """
    if save_dir is None:
        return None
    # Use incrementing directory names: 0, 1, 2, ...
    existing = [int(p.name) for p in save_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    idx = max(existing, default=-1) + 1
    dest = save_dir / str(idx)
    dest.mkdir()
    shutil.copy2(network_path, dest / network_path.name)
    for inp in input_paths:
        shutil.copy2(inp, dest / inp.name)
    # Save metadata needed for reproduction
    metadata = {
        "mode": mode,
        "check_nan_inf": check_nan_inf,
        "expected_shapes": [list(s) for s in expected_shapes],
        "inputs": [inp.name for inp in input_paths],
    }
    (dest / "metadata.json").write_text(json_mod.dumps(metadata, indent=2))
    if error:
        (dest / "error.txt").write_text(error)
    if stderr:
        (dest / "stderr.txt").write_text(stderr)
    return str(dest)


def _truncate(text, max_len=500):
    """Truncate text to max_len, keeping the first and last halves."""
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + " ... " + text[-half:]


def _build_cmd(mode, network_path, input_paths, output_dir, backend, backend_arg, test_dir):
    """Build the CLI command for eval or grad."""
    if backend == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{test_dir.resolve()}:/data",
            backend_arg,
            mode,
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            f"/data/{network_path.relative_to(test_dir)}",
            *[f"/data/{p.relative_to(test_dir)}" for p in input_paths],
        ]
    else:
        cmd = [
            *shlex.split(backend_arg),
            mode,
            "--output-dir",
            str(output_dir),
            str(network_path),
            *[str(p) for p in input_paths],
        ]
    return cmd
