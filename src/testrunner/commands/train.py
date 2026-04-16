# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Command builder and custom runner for train.

Expected test.json format (with source_dataset for cached datasets):
    {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": 784,
        "num_classes": 10,
        "source_dataset": "mnist"
    }

Or with explicit paths (no caching):
    {
        "command": "train",
        "dataset": "mnist_mlp",
        "in_size": 784,
        "num_classes": 10,
        "train_inputs": ["resources/train_images.bin", "resources/train_labels.bin"],
        "train_labels": "resources/train_labels.bin",
        "test_inputs": ["resources/test_images.bin"],
        "test_labels": "resources/test_labels.bin"
    }
"""

import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from testrunner.commands.common import build_eval_grad_cmd, get_timeout, parse_output_paths


def build_train_cmd(config, test_dir, output_dir, backend, backend_arg):
    """Build the CLI command for the train step."""
    dataset = config["dataset"]
    train_inputs = config["train_inputs"]

    if backend == "docker":
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{test_dir.resolve()}:/data",
            backend_arg,
            "train",
            "--output-dir",
            f"/data/{output_dir.relative_to(test_dir)}",
            dataset,
            *[f"/data/{i}" for i in train_inputs],
        ]
        return cmd, None
    else:
        resolved_inputs = []
        for i in train_inputs:
            p = Path(i)
            resolved_inputs.append(str(p if p.is_absolute() else test_dir / p))
        cmd = [
            *shlex.split(backend_arg),
            "train",
            "--output-dir",
            str(output_dir),
            dataset,
            *resolved_inputs,
        ]
        return cmd, None


def _eval_accuracy(checkpoint_path, input_bin, labels_bin, eval_batch_size,
                   in_size, num_classes, output_dir, backend, backend_arg,
                   test_dir, batch_paths=None, timeout=60):
    """Evaluate a checkpoint on a dataset and return accuracy.

    If batch_paths is provided, uses those pre-split .bin files directly.
    Otherwise splits the input into batches of eval_batch_size on the fly.
    """
    labels = np.fromfile(labels_bin, dtype=np.float64)
    num_labels = labels.size // num_classes
    labels = labels.reshape(num_labels, num_classes)

    if batch_paths is not None:
        num_batches = len(batch_paths)
        num_samples = num_batches * eval_batch_size
    else:
        images = np.fromfile(input_bin, dtype=np.float64)
        num_samples = images.size // in_size
        images = images.reshape(num_samples, in_size)
        if num_samples % eval_batch_size != 0:
            return None, (
                f"number of samples ({num_samples}) is not divisible "
                f"by eval_batch_size ({eval_batch_size})"
            )
        num_batches = num_samples // eval_batch_size

    all_preds = []

    for batch_idx in range(num_batches):
        if batch_paths is not None:
            batch_input_path = batch_paths[batch_idx]
        else:
            start = batch_idx * eval_batch_size
            end = start + eval_batch_size
            batch_data = images[start:end]
            batch_input_path = output_dir / f"batch_{batch_idx}_input.bin"
            batch_data.tofile(batch_input_path)

        batch_output_dir = output_dir / f"batch_{batch_idx}_output"
        batch_output_dir.mkdir(exist_ok=True)

        # When batch_input_path is from a cached dataset outside test_dir,
        # we can't use build_eval_grad_cmd (which assumes paths relative to
        # test_dir). Build the command directly with absolute paths instead.
        all_under_test_dir = (
            checkpoint_path.is_relative_to(test_dir)
            and batch_input_path.is_relative_to(test_dir)
        )
        if all_under_test_dir:
            eval_config = {
                "command": "eval",
                "network": str(checkpoint_path.relative_to(test_dir)),
                "inputs": [str(batch_input_path.relative_to(test_dir))],
            }
            cmd, _ = build_eval_grad_cmd(
                eval_config, test_dir, batch_output_dir, backend, backend_arg,
            )
        elif backend == "docker":
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{test_dir.resolve()}:/data",
                "-v", f"{batch_input_path.resolve()}:/input/{batch_input_path.name}",
                backend_arg,
                "eval",
                "--output-dir", f"/data/{batch_output_dir.relative_to(test_dir)}",
                f"/data/{checkpoint_path.relative_to(test_dir)}",
                f"/input/{batch_input_path.name}",
            ]
        else:
            cmd = [
                *shlex.split(backend_arg),
                "eval",
                "--output-dir", str(batch_output_dir),
                str(checkpoint_path),
                str(batch_input_path),
            ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, (
                f"eval timed out on {checkpoint_path.name} "
                f"batch {batch_idx} after {timeout}s"
            )
        if result.returncode != 0:
            return None, (
                f"eval failed on {checkpoint_path.name} "
                f"batch {batch_idx}: {result.stderr.strip()}"
            )

        # Parse output file paths from stdout
        output_files, _ = parse_output_paths(result.stdout)

        if not output_files:
            return None, f"eval produced no output for batch {batch_idx}"

        batch_preds = np.fromfile(output_files[0], dtype=np.float64)
        all_preds.append(batch_preds)

    preds = np.concatenate(all_preds).reshape(num_samples, num_classes)
    # Only compare up to the number of samples we evaluated
    accuracy = float(np.mean(
        np.argmax(preds, axis=-1) == np.argmax(labels[:num_samples], axis=-1)
    ))
    return accuracy, None


def run_train_test(test_dir, config, output_dir, backend, backend_arg, generate=False):
    """Custom runner for train tests.

    Runs the train command, evaluates each checkpoint on train and test sets,
    and computes classification accuracy.
    """
    in_size = config["in_size"]
    num_classes = config["num_classes"]

    # Resolve dataset paths from cache if source_dataset is present
    dataset_cache_dir = None
    if "source_dataset" in config:
        from testrunner.datasets import prepare_dataset

        dataset_paths = prepare_dataset(config["source_dataset"])
        dataset_cache_dir = Path(dataset_paths["train_images"]).parent
        config = {
            **config,
            "train_inputs": [dataset_paths["train_images"], dataset_paths["train_labels"]],
            "train_labels": dataset_paths["train_labels"],
            "test_inputs": [dataset_paths["test_images"]],
            "test_labels": dataset_paths["test_labels"],
        }

    # Step 1: Run train command
    timeout = get_timeout(config)
    cmd, cwd = build_train_cmd(config, test_dir, output_dir, backend, backend_arg)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": f"train command timed out after {timeout}s",
        }

    if result.returncode != 0:
        return {
            "passed": False,
            "error": f"train command failed (exit {result.returncode}): {result.stderr.strip()}",
        }

    # Step 2: Parse stdout — first line is eval_batch_size, rest are checkpoint paths
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return {"passed": False, "error": "train command produced no output"}

    first_line = lines[0]
    if not first_line.startswith("eval_batch_size:"):
        return {
            "passed": False,
            "error": f"expected first output line to be 'eval_batch_size: N', got: {first_line}",
        }
    eval_batch_size = int(first_line.split(":", 1)[1].strip())

    checkpoint_paths = []
    for line in lines[1:]:
        p = Path(line)
        if p.exists():
            checkpoint_paths.append(p)
        else:
            print(f"warning: ignoring non-existent checkpoint path: {line}", file=sys.stderr)

    if not checkpoint_paths:
        return {"passed": False, "error": "train produced no checkpoint files"}

    if generate:
        return {
            "passed": True,
            "error": None,
            "generated": True,
            "checkpoints": [str(p) for p in checkpoint_paths],
        }

    # Step 3: Resolve input/label paths
    #   When source_dataset is used, paths are already absolute.
    #   Otherwise they are relative to test_dir.
    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else test_dir / p

    test_images = _resolve(config["test_inputs"][0])
    test_labels = _resolve(config["test_labels"])
    train_images = _resolve(config["train_inputs"][0])
    train_labels = _resolve(config["train_labels"])

    # Prepare cached batch splits if we have a dataset cache
    cached_batches = {}
    if dataset_cache_dir is not None:
        from testrunner.datasets.mnist import prepare_batches

        for split_name, images_path in [("test", test_images), ("train", train_images)]:
            cached_batches[split_name] = prepare_batches(
                dataset_cache_dir, split_name, in_size, eval_batch_size,
            )

    # Step 4: Evaluate each checkpoint on both splits
    checkpoint_results = {}
    best_test_accuracy = 0.0

    for cp_path in checkpoint_paths:
        cp_name = cp_path.name
        cp_result = {}

        for split_name, images_path, labels_path in [
            ("test", test_images, test_labels),
            ("train", train_images, train_labels),
        ]:
            split_output_dir = output_dir / f"eval_{cp_name}_{split_name}"
            split_output_dir.mkdir(parents=True, exist_ok=True)

            accuracy, error = _eval_accuracy(
                cp_path, images_path, labels_path,
                eval_batch_size, in_size, num_classes,
                split_output_dir, backend, backend_arg, test_dir,
                batch_paths=cached_batches.get(split_name),
                timeout=get_timeout({**config, "command": "eval"}),
            )
            if error:
                return {"passed": False, "error": error}

            cp_result[f"{split_name}_accuracy"] = accuracy

        checkpoint_results[cp_name] = cp_result
        if cp_result["test_accuracy"] > best_test_accuracy:
            best_test_accuracy = cp_result["test_accuracy"]

    return {
        "passed": True,
        "error": None,
        "best_test_accuracy": best_test_accuracy,
        "checkpoint_results": checkpoint_results,
    }
