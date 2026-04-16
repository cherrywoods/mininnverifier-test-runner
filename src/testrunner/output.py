# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Output handlers for the test runner.

Two modes:
- JsonOutputHandler: emits one JSON line per test result to stdout (JSONL)
- CliOutputHandler: live progress display with colored pass/fail on stderr
"""

import json
import sys


class JsonOutputHandler:
    """Emit one JSON line per completed test to stdout."""

    def test_starting(self, test_path, index, total):
        pass

    def test_stdout_line(self, line):
        pass

    def test_finished(self, test_path, result):
        print(json.dumps({"test": test_path, **result}), flush=True)

    def all_finished(self, total_passed, total_failed):
        pass


class CliOutputHandler:
    """Live progress display on stderr with colored pass/fail indicators."""

    _GREEN = "\033[32m"
    _RED = "\033[31m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"

    def __init__(self):
        self._tty = sys.stderr.isatty()
        self._live_lines = 0

    def test_starting(self, test_path, index, total):
        self._live_lines = 0
        msg = f"[{index}/{total}] Running {test_path}..."
        if self._tty:
            print(msg, file=sys.stderr, flush=True)
        else:
            print(msg, end="", file=sys.stderr, flush=True)

    def test_stdout_line(self, line):
        if self._tty:
            print(f"  {line}", file=sys.stderr, flush=True)
            self._live_lines += 1

    def test_finished(self, test_path, result):
        passed = result.get("passed", False)
        if self._tty:
            # Move cursor up to overwrite progress line + any stdout lines
            n = self._live_lines + 1
            sys.stderr.write(f"\033[{n}A\033[J")
            if passed:
                print(
                    f"  {self._GREEN}\u2713{self._RESET} {test_path}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                error = result.get("error", "unknown error")
                print(
                    f"  {self._RED}\u2717{self._RESET} {test_path}",
                    file=sys.stderr,
                )
                print(
                    f"    {self._RED}{error}{self._RESET}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            # Non-TTY: simple sequential output
            if passed:
                print(f" PASS", file=sys.stderr, flush=True)
            else:
                error = result.get("error", "unknown error")
                print(f" FAIL: {error}", file=sys.stderr, flush=True)
        self._live_lines = 0

    def all_finished(self, total_passed, total_failed):
        total = total_passed + total_failed
        if total_failed == 0:
            color = self._GREEN
        else:
            color = self._RED
        print(
            f"\n{self._BOLD}{color}{total_passed} passed, {total_failed} failed{self._RESET}"
            f" ({total} total)",
            file=sys.stderr,
            flush=True,
        )
