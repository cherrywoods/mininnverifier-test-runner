# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the test runner
python -m testrunner {docker,podman,local} <backend_arg> <test_dir>

# Subcommands
python -m testrunner.list <test_dir>        # enumerate all tests
python -m testrunner.show <test_dir>        # visualize graphs/inputs/outputs
python -m testrunner.reproduce <failure>    # rerun a saved fuzz failure

# Run unit tests
pytest

# Run a single test file
pytest tests/test_scoring.py

# Lint
ruff check src/
ruff format src/
```

## Architecture

This is a test orchestration system that discovers `test.json` files under a directory, runs student neural network implementations against them, and scores results.

**Entry point**: `__main__.py:main()` parses CLI args, discovers tests, sorts by cost (eval/grad < train < fuzz < bench), then calls `run_single_test()` for each.

**Backends** (docker, podman, local): Docker/Podman mount the test directory at `/data` and rewrite all paths. All three backends invoke the student implementation with:
```
<impl> <cmd> --output-dir <dir> <network.mininn> <inputs.bin...>
```
The implementation writes output `.bin` files and prints their paths to stdout (one per line).

**Test modes**: `eval`, `grad`, `train`, `fuzz_eval`, `fuzz_grad`, `bench_eval`, `bench_grad`. Commands are registered in `commands/__init__.py`; fuzz runners live in `fuzz/`, benchmark runners in `benchmark/`.

**Network format** (`.mininn`): A ZIP archive containing `graph.txt` (variable shapes and equations) and `*.bin` constant files (flat float64 arrays, no header). Example graph line: `c[2,3] = add{} a b`.

**Scoring** (`scoring.py`): Four pluggable functions configured per test in `test.json` — `binary` (all-or-nothing), `exponential` (accuracy mapped to points via configurable k), `proportional` (passed_trials/total), `speed` (base points + tier bonuses).

**Subprocess management** (`commands/common.py:run_subprocess`): Spawns with `Popen`, drains stderr in a background thread, captures stdout line-by-line, kills on timeout via daemon timer.

**Fuzz testing** (`fuzz/`): Uses Hypothesis to generate random compute graphs (`graph_builder.py`) and runs them through the student implementation; failures are saved to `actual/fuzz_failures/` for reproduction.

**Output** (`output.py`): Two handlers — `CLIOutput` (colored terminal) and `JSONOutput` (JSONL, one object per line) — selected via `--output` flag.

## Test Configuration

Each test directory contains a `test.json`:

```json
{
  "command": "eval",
  "network": "network.mininn",
  "inputs": ["input.bin"],
  "expected_outputs": ["expected_output.bin"],
  "tolerance": 1e-4,
  "points": 10,
  "scoring": {"function": "binary"}
}
```

Runtime outputs land in `<test_dir>/actual/`; fuzz failures in `<test_dir>/actual/fuzz_failures/`.

## Maintenance

### Adding a new command/test mode

1. Create `src/testrunner/commands/mycommand.py` with a builder function `(config, test_dir, output_dir, backend, backend_arg, extra_run_args=()) -> (cmd_list, cwd)` and a runner function returning a result dict with at least `{"passed": bool, "error": str | None}`.
2. Register both in `commands/__init__.py`: add to `COMMANDS` and inside `_init_runners()` add to `RUNNERS`. The lazy `_init_runners()` pattern exists to break circular imports with `fuzz/`.
3. Add the command name to `COMMAND_ORDER` at the appropriate cost position.
4. Add a default timeout in `commands/common.py:DEFAULT_TIMEOUTS` and a default scoring function in `scoring.py:DEFAULT_SCORING`.

### Adding a new scoring function

Define `myscore(max_points, result, **params) -> (score, bonus)` in `scoring.py` and register it in `SCORING_FUNCTIONS`. Params come directly from the `"scoring"` object in `test.json`.

### Adding a new check function

Create `src/testrunner/check/mycheck.py` with `check_mycheck(test_dir, config, output_files, closed=False) -> {"passed": bool, "error": str | None}`. Import and register it in `check/__init__.py:CHECKS`. The `closed` flag suppresses detailed error info shown to students.

### Adding a new fuzz primitive

1. Add the primitive name to `ALL_PRIMITIVES` in `fuzz/graph_builder.py`; also add to `UNSAFE_PRIMITIVES` if it can produce NaN/Inf.
2. Implement an `elif prim_name == "myprim":` branch inside `_try_apply()`: pick inputs from `available`, compute output shape, build an `Equation`, and return `(eqn, out_var, new_consts, var_counter+1, const_counter)` or `None` if inputs are unsuitable.

### Updating dependencies

Edit version constraints in `pyproject.toml`, then run `uv lock` to regenerate `uv.lock`.

### Keeping CLAUDE.md current

Update this file when the architecture changes in ways that aren't obvious from reading the code: new extension points, changed interfaces, new required registration steps, or shifts in the overall data flow. Changes that are self-evident from well-named code (a new file, a renamed variable) don't need to be reflected here.

Sections most likely to go stale: the command/scoring/check/fuzz extension steps in this Maintenance section (interface signatures), and the test mode list in Architecture.

### Dataset cache

Datasets are cached under `$MININNVERIFIER_CACHE_DIR` (default `~/.cache/mininnverifier/datasets`). To force a refresh, delete the relevant subdirectory. Add a new dataset by creating `src/testrunner/datasets/mydata.py` with a `prepare_mydata()` function and registering it in `datasets/__init__.py:DATASETS`.
