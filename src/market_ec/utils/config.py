from pathlib import Path
from typing import Any, Dict
import yaml
from .schema import RootCfg

def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f)
    # validate & coerce types
    RootCfg.model_validate(cfg)
    return cfg
