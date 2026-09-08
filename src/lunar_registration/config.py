"""
Configuration loader for the lunar_registration package.

Re-exports the top-level config loader for use within the package.
"""

import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import load_config, get_config_value  # noqa: F401, E402

__all__ = ["load_config", "get_config_value"]
