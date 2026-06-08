# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .bounds_within_range import check_bounds_within_range
from .close_to_expected import check_close_to_expected
from .linear_bounds_within_range import check_linear_bounds_within_range

CHECKS = {
    "close_to_expected": check_close_to_expected,
    "bounds_within_range": check_bounds_within_range,
    "linear_bounds_within_range": check_linear_bounds_within_range,
}
DEFAULT_CHECKS = {
    "eval": "close_to_expected",
    "grad": "close_to_expected",
    "bounds": "close_to_expected",
    "linear_bounds": "linear_bounds_within_range",
}
