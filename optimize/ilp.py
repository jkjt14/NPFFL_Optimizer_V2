python - <<'PY'
content = r'''
from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

def solve_lineup(
    players_df: pd.DataFrame,
    cfg: dict,
    objective: str = "points",
    enforce_stack: bool = False,
    max_from_team: int | None = None,
) -> dict:
    df = players_df.reset_index(drop=True).copy()
    required = {"player_id", "name", "team", "pos", "salary", "ProjPoints"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"players_df missing columns: {missing}")

    df["pos"] = df["pos"].astype(str).str.upper()
    pos = df["pos"].tolist()
    n = len(df)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

    # position indices
    def idxs(p): return [i for i in range(n) if pos[i] == p]
    I_QB, I_RB, I_WR, I_TE, I_DEF = idxs("QB"), idxs("RB"), idxs("WR"), idxs("TE"), idxs("DEF")

    roster = cfg.get("roster", {"QB":1, "RB":2, "WR":3, "TE":1, "FLEX":1, "DEF":1})
    total_slots = int(sum(roster.values()))

    # exact counts for fixed positions
    model.Add(sum(x[i] for i in I_QB) == roster["QB"])
    model.Add(sum(x[i] for i in I_DEF) == roster["DEF"])

    # minimums for RB/WR/TE
    model.Add(sum(x[i] for i in I_RB) >= roster["RB"])
    model.Add(sum(x[i] for i in I_WR) >= roster["WR"])
    model.Add(sum(x[i] for i in I_TE) >= roster["TE"])

    # one FLEX across RB/WR/TE
    model.Add(
        sum(x[i] for i in I_RB) +
        sum(x[i] for i in I_WR) +
        sum(x[i] for i in I_TE)
        >= roster["RB"] + roster["WR"] + roster["TE"] + roster["FLEX"]
    )

    # total lineup size
    model.Add(sum(x) == total_slots)

    # salary cap
    salaries = df["salary"].fillna(0).astype(int).tolist()
    model.Add(sum(salaries[i] * x[i] for i in range(n)) <= int(cfg.get("salary_cap", 60000)))

    # ---- Phase 2 constraints ----
    teams = df["team"].astype(str).str.upper().tolist()
    unique_teams = sorted(set(teams))

    # cap players per team
    if max_from_team is not None:
        max_from_team = int(max_from_team)
        for t in unique_teams:
            model.Add(sum(x[i] for i in range(n) if teams[i] == t) <= max_from_team)

    # QB stacking: each chosen QB must have >=1 same-team WR/TE
    if enforce_stack:
        for qb in I_QB:
            wr_te_same = [j for j in range(n) if teams[j] == teams[qb] and pos[j] in ("WR", "TE")]
            model.Add(sum(x[j] for j in wr_te_same) >= x[qb])

    # objective (maximize projected points)
    points = df["ProjPoints"].fillna(0).astype(float).tolist()
    model.Maximize(sum(int(round(points[i] * 1000)) * x[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible lineup found.")

    chosen = [i for i in range(n) if solver.BooleanValue(x[i])]
    sel = df.loc[chosen].copy()

    return {
        "players": sel[["pos","name","team","salary","ProjPoints"]].to_dict("records"),
        "total_salary": int(sel["salary"].sum()),
        "total_proj": float(sel["ProjPoints"].sum()),
        "count": {k: int((sel["pos"] == k).sum()) for k in ["QB","RB","WR","TE","DEF"]},
    }
'''
open('dfs_opt/optimize/ilp.py', 'w', encoding='utf-8').write(content)
print("✅ wrote clean dfs_opt/optimize/ilp.py")
PY
