# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .eval import build_eval_cmd
from .grad import build_grad_cmd
from .train import run_train_test

from testrunner.fuzz import run_fuzz_eval, run_fuzz_grad

COMMANDS = {"eval": build_eval_cmd, "grad": build_grad_cmd}
RUNNERS = {
    "train": run_train_test,
    "fuzz_eval": run_fuzz_eval,
    "fuzz_grad": run_fuzz_grad,
}
