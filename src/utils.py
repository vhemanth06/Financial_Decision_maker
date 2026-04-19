import yaml
from pathlib import Path
from typing import Any
import pandas as pd
from typing import Any, Iterable, Optional

def load_config(config_path: Path) -> dict[str, Any]:
    """Load project configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)

def _resolve_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Resolve the first existing column name from a candidate list.

    Args:
        frame: Input dataframe to inspect.
        candidates: Ordered list of potential column names.

    Returns:
        The first matching column name, or None when no match exists.
    """
    existing_columns = set(frame.columns)
    for candidate in candidates:
        if candidate in existing_columns:
            return candidate
    return None

