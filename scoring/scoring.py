from __future__ import annotations
import pandas as pd

def compute_points(row: dict, is_te: bool, cfg: dict) -> float:
    s = cfg["scoring"]
    pts = 0.0

    # Grab with defaults
    pass_yd = float(row.get("pass_yd", 0) or 0)
    pass_td = float(row.get("pass_td", 0) or 0)
    rush_yd = float(row.get("rush_yd", 0) or 0)
    rush_td = float(row.get("rush_td", 0) or 0)
    rec = float(row.get("rec", 0) or 0)
    rec_yd = float(row.get("rec_yd", 0) or 0)
    rec_td = float(row.get("rec_td", 0) or 0)

    # Passing
    pts += pass_td * s["pass_td"]
    pts += pass_yd / s["pass_yd_pt_per"]
    if pass_yd >= 400:
      pts += s["pass_bonus_400"]
    elif pass_yd >= 300:
      pts += s["pass_bonus_300"]

    # Rushing
    pts += rush_td * 6
    pts += rush_yd / s["rush_yd_pt_per"]
    if rush_yd >= 200:
      pts += s["rush_bonus_200"]
    elif rush_yd >= 100:
      pts += s["rush_bonus_100"]

    # Receiving
    ppr = s["te_rec_ppr"] if is_te else s["rec_ppr"]
    pts += rec * ppr
    pts += rec_td * 6
    pts += rec_yd / s["rec_yd_pt_per"]
    if rec_yd >= 200:
      pts += s["rush_bonus_200"]  # same thresholds for rec yards
    elif rec_yd >= 100:
      pts += s["rush_bonus_100"]

    return float(pts)

def apply_custom_scoring(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    pos = df.get("pos")
    if pos is None:
        raise ValueError("DataFrame must include 'pos' column.")
    df["ProjPoints"] = [
        compute_points(r, is_te=(str(r["pos"]).upper()=="TE"), cfg=cfg)
        for _, r in df.iterrows()
    ]
    return df
