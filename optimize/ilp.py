from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

def solve_lineup(players_df: pd.DataFrame, cfg: dict, objective: str = "points",
                 enforce_stack: bool = False, max_from_team: int | None = None) -> dict:
    df = players_df.reset_index(drop=True).copy()
    required_cols = {"player_id","name","team","pos","salary","ProjPoints"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"players_df missing columns: {missing}")
    df["pos"] = df["pos"].astype(str).str.upper()
    positions = df["pos"].tolist()
    model = cp_model.CpModel()
    n = len(df)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    # indices by position
    pos_idxs = {p: [i for i in range(n) if positions[i] == p] for p in ["QB","RB","WR","TE","DEF"]}
    roster = cfg.get("roster", {"QB":1,"RB":2,"WR":3,"TE":1,"FLEX":1,"DEF":1})
    # Base counts
    model.Add(sum(x[i] for i in pos_idxs["QB"]) == roster["QB"])
    model.Add(sum(x[i] for i in pos_idxs["DEF"]) == roster["DEF"])
    # Minimum counts for RB/WR/TE
    model.Add(sum(x[i] for i in pos_idxs["RB"]) >= roster["RB"])
    model.Add(sum(x[i] for i in pos_idxs["WR"]) >= roster["WR"])
    model.Add(sum(x[i] for i in pos_idxs["TE"]) >= roster["TE"])
    # Total slots
    total_slots = sum(roster.values())
    model.Add(sum(x) == total_slots)
    # Salary cap
    salaries = df["salary"].tolist()
    model.Add(sum(salaries[i] * x[i] for i in range(n)) <= cfg["salary_cap"])
    # At least one FLEX across RB/WR/TE
    model.Add(sum(x[i] for i in pos_idxs["RB"]) + sum(x[i] for i in pos_idxs["WR"]) + sum(x[i] for i in pos_idxs["TE"]) >= roster["RB"] + roster["WR"] + roster["TE"] + roster["FLEX"])
    # Objective: maximize ProjPoints
    points = df["ProjPoints"].tolist()
    model.Maximize(sum(int(points[i] * 1000) * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible lineup found.")
    chosen = [i for i in range(n) if solver.BooleanValue(x[i])]
    sel = df.loc[chosen].copy()
    return {
        "players": sel.to_dict(orient="records"),
        "total_salary": int(sel["salary"].sum()),
        "total_proj": float(sel["ProjPoints"].sum()),
        "count": {
            "QB": int((sel["pos"] == "QB").sum()),
            "RB": int((sel["pos"] == "RB").sum()),
            "WR": int((sel["pos"] == "WR").sum()),
            "TE": int((sel["pos"] == "TE").sum()),
            "DEF": int((sel["pos"] == "DEF").sum()),
        },
    }
