"""
Configuration loader with defaults and overrides.

Loads the default configuration, then merges method-specific
overrides and any user-provided overrides on top.
"""

from pathlib import Path
from copy import deepcopy

import yaml


# Package root for finding config files
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _PACKAGE_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep-merge override into base dict. Override values take priority.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    Load pipeline configuration.

    Loads default.yaml first, then merges in the specified config
    (if any), then applies any runtime overrides.

    Parameters
    ----------
    config_path : str, Path, or None
        Path to a method-specific YAML config file.
        Can also be a method name: 'sift', 'superpoint', 'loftr'.
    overrides : dict or None
        Additional overrides to apply on top.

    Returns
    -------
    dict
        Merged configuration.
    """
    # Load defaults
    default_path = _CONFIGS_DIR / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Load method-specific config
    if config_path is not None:
        path = Path(config_path)

        # Allow shorthand names: 'sift' → configs/sift.yaml
        if not path.exists() and not path.suffix:
            path = _CONFIGS_DIR / f"{config_path}.yaml"

        if path.exists():
            with open(path) as f:
                method_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, method_config)
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}"
            )

    # Apply runtime overrides
    if overrides:
        config = _deep_merge(config, overrides)

    return config


def get_config_value(config: dict, key_path: str, default=None):
    """
    Get a nested config value using dot-notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key_path : str
        Dot-separated key path, e.g., 'features.detector'.
    default
        Default value if key not found.

    Returns
    -------
    Value at the key path, or default.

    Examples
    --------
    >>> cfg = {'features': {'detector': 'sift'}}
    >>> get_config_value(cfg, 'features.detector')
    'sift'
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current
