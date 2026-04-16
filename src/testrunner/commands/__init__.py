# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .eval import build_eval_cmd
from .grad import build_grad_cmd
from .train import build_train_cmd, run_train_test

COMMANDS = {"eval": build_eval_cmd, "grad": build_grad_cmd, "train": build_train_cmd}
RUNNERS = {"train": run_train_test}
