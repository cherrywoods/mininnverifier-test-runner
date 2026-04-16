# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .eval import build_eval_cmd
from .grad import build_grad_cmd
from .train import run_train_test

COMMANDS = {"eval": build_eval_cmd, "grad": build_grad_cmd}

# RUNNERS is populated lazily to break a circular import:
# commands/__init__ -> fuzz -> fuzz.runner -> commands.common -> commands/__init__
RUNNERS = {}

# Cheapest commands first. Tests are run in this order so that fast/cheap
# tests fail early before expensive ones are attempted.
# Append new commands at the position that matches their cost.
COMMAND_ORDER = ["eval", "grad", "train", "fuzz_eval", "fuzz_grad", "bench_eval", "bench_grad"]


def command_sort_key(command: str) -> int:
    """Return a numeric priority for *command* (lower = run earlier).

    Unknown commands sort after all known ones, preserving alphabetical
    order among themselves.
    """
    try:
        return COMMAND_ORDER.index(command)
    except ValueError:
        return len(COMMAND_ORDER)


def _init_runners():
    if not RUNNERS:
        from testrunner.fuzz import run_fuzz_eval, run_fuzz_grad
        from testrunner.benchmark import run_bench_eval, run_bench_grad

        RUNNERS.update(
            {
                "train": run_train_test,
                "fuzz_eval": run_fuzz_eval,
                "fuzz_grad": run_fuzz_grad,
                "bench_eval": run_bench_eval,
                "bench_grad": run_bench_grad,
            }
        )
