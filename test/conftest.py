"""Pytest configuration for the plctestbench unit test suite.

Ensures the repository root is importable so that ``test._helpers`` and the
``plctestbench`` package resolve no matter where pytest is invoked from.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))