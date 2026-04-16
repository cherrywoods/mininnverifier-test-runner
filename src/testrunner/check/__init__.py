# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .close_to_expected import check_close_to_expected

CHECKS = {"close_to_expected": check_close_to_expected}
DEFAULT_CHECKS = {"eval": "close_to_expected", "grad": "close_to_expected"}
