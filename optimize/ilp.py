from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

def solve_lineup(players_df: pd.DataFrame, cfg: dict, objective: str = "points",
                 risk_cap: float | None = None, enforce_stack: bool = False) -> dict:
    """
    Minimal ILP:
    - exactly 9 total
    - QB==1, DEF==1, RB>=2, WR>=3, TE>=1
    - salary <= cap
    - FLEX not explicitly modeled yet (this allows extra among RB/WR/TE by requiring RB+WR+TE >= 7)
    Objective: maximize sum(ProjPoints * x)
    """
    df = players_df.reset_index(drop=True).copy()
    required_cols = {"player_id","name","team","pos","salary","ProjPoints"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"players_df missing columns: {missing}")

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(df))]

    # counts
    def count_pos(p):
        return [x[i] for i in range(len(df)) if str(df.loc[i,"pos"]).upper()==p]

    QB = count_pos("QB")
    RB = count_pos("RB")
    WR = count_pos("WR")
    TE = count_pos("TE")
    DEF = count_pos("DEF")

    # constraints
    model.Add(sum(x) == 9)
    model.Add(sum(QB) == 1)
    model.Add(sum(DEF) == 1)
    model.Add(sum(RB) >= 2)
    model.Add(sum(WR) >= 3)
    model.Add(sum(TE) >= 1)
    # allow one extra among RB/WR/TE implicitly:
    model.Add(sum(RB) + sum(WR) + sum(TE) >= 7)

    # salary cap
    salaries = [int(df.loc[i,"salary"]) for i in range(len(df))]
    model.Add(sum(salaries[i] * x[i] for i in range(len(df))) <= int(cfg["salary_cap"]))

    # objective
    points = [float(df.loc[i,"ProjPoints"]) for i in range(len(df))]
    model.Maximize(sum(points[i] * x[i] for i in range(len(df))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible lineup found.")

    chosen_idx = [i for i in range(len(df)) if solver.BooleanValue(x[i])]
    chosen = df.loc[chosen_idx].copy()
    total_salary = int(chosen["salary"].sum())
    total_proj = float(chosen["ProjPoints"].sum())

    return {
        "players": chosen.to_dict(orient="records"),
        "total_salary": total_salary,
        "total_proj": total_proj,
        "count": {
            "QB": int((chosen["pos"].str.upper()=="QB").sum()),
            "RB": int((chosen["pos"].str.upper()=="RB").sum()),
            "WR": int((chosen["pos"].str.upper()=="WR").sum()),
            "TE": int((chosen["pos"].str.upper()=="TE").sum()),
            "DEF": int((chosen["pos"].str.upper()=="DEF").sum()),
        }
    }
