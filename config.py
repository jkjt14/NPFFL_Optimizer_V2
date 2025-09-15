from __future__ import annotations
import yaml
from pathlib import Path

DEFAULTS = {
    "salary_cap": 60000,
    "roster": {"QB":1, "RB":2, "WR":3, "TE":1, "FLEX":1, "DEF":1},
    "scoring": {
        "pass_td": 4,
        "pass_yd_pt_per": 25,
        "pass_bonus_300": 1,
        "pass_bonus_400": 2,
        "rush_yd_pt_per": 10,
        "rush_bonus_100": 1,
        "rush_bonus_200": 2,
        "rec_yd_pt_per": 10,
        "rec_ppr": 0.5,
        "te_rec_ppr": 1.0,
    },
}

def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config and merge with DEFAULTS."""
    p = Path(path)
    if not p.exists():
        return DEFAULTS.copy()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = DEFAULTS.copy()
    # Shallow copy non-nested keys
    for k, v in data.items():
        if k not in ("roster", "scoring"):
            cfg[k] = v
    # Deep-merge nested dicts
    for key in ("roster", "scoring"):
        merged = DEFAULTS[key].copy()
        merged.update(data.get(key) or {})
        cfg[key] = merged
    return cfg

