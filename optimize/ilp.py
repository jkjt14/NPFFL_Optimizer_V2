cat > dfs_opt/optimize/ilp.py << 'PY'
from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

def _ints(seq, scale: int = 1000):
    # convert floats to scaled ints for CP-SAT
    out = []
    for v in seq:
        try:
            out.append(int(round(float(v) * scale)))
        except Exception:
            out.append(0)
    return out, scale

def solve_lineup(players_df: pd.DataFrame,
                 cfg: dict,
                 objective: str = "points",           # 'points' | 'floor' | 'ceiling' | 'value'
                 risk_cap: float | None = None,       # average RiskScore <= risk_cap (1-99)
                 enforce_stack: bool = False) -> dict:
    """
    True FLEX (RB/WR/TE), multiple objectives, optional risk cap, optional QB stack.
    Required cols: player_id, name, team, pos, salary, ProjPoints
    Optional cols: ProjFloor, ProjCeiling, RiskScore, Value
    """
    df = players_df.reset_index(drop=True).copy()

    # --- sanity columns
    required = {"player_id","name","team","pos","salary","ProjPoints"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"players_df missing columns: {missing}")

    # normalize types
    df["pos"]  = df["pos"].astype(str).str.upper()
    df["team"] = df["team"].astype(str).str.upper()
    df["salary"] = df["salary"].astype(int)

    # --- index by position
    n = len(df)
    idx = list(range(n))
    pos = df["pos"].tolist()
    def by(p): return [i for i in idx if pos[i] == p]
    I_QB, I_RB, I_WR, I_TE, I_DEF = by("QB"), by("RB"), by("WR"), by("TE"), by("DEF")

    # --- CP-SAT model
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x_{i}") for i in idx]

    # Roster counts with TRUE FLEX
    r = cfg.get("roster", {"QB":1,"RB":2,"WR":3,"TE":1,"FLEX":1,"DEF":1})
    # explicit counts
    m.Add(sum(x[i] for i in I_QB) == r.get("QB",1))
    m.Add(sum(x[i] for i in I_DEF) == r.get("DEF",1))

    # flex choice vars
    rb_flex = m.NewBoolVar("rb_flex")
    wr_flex = m.NewBoolVar("wr_flex")
    te_flex = m.NewBoolVar("te_flex")
    m.Add(rb_flex + wr_flex + te_flex == 1)

    # exact counts (mins + whichever flex)
    m.Add(sum(x[i] for i in I_RB) == r.get("RB",2) + rb_flex)
    m.Add(sum(x[i] for i in I_WR) == r.get("WR",3) + wr_flex)
    m.Add(sum(x[i] for i in I_TE) == r.get("TE",1) + te_flex)

    # total slots (must equal sum roster values)
    total_slots = int(sum(r.values()))  # 1 QB + 2 RB + 3 WR + 1 TE + 1 FLEX + 1 DEF = 9
    m.Add(sum(x) == total_slots)

    # salary cap
    cap = int(cfg.get("salary_cap", 60000))
    salaries = [int(df.loc[i, "salary"]) for i in idx]
    m.Add(sum(salaries[i] * x[i] for i in idx) <= cap)

    # optional risk cap (average RiskScore <= threshold)
    if risk_cap is not None and "RiskScore" in df.columns:
        risks = df["RiskScore"].fillna(50).astype(float).tolist()
        risks_int, scale = _ints(risks, scale=100)  # keep precision
        m.Add(sum(risks_int[i] * x[i] for i in idx) <= int(total_slots * risk_cap * scale))

    # optional QB stack: for each chosen QB, require at least one same-team WR/TE
    if enforce_stack and I_QB:
        teams = df["team"].tolist()
        for qi in I_QB:
            same_team_receivers = [j for j in idx if pos[j] in ("WR","TE") and teams[j] == teams[qi]]
            # If there are no same-team receivers in the pool, the constraint would be impossible;
            # only add it when there is at least one candidate.
            if same_team_receivers:
                m.Add(sum(x[j] for j in same_team_receivers) >= x[qi])

    # objective selection (with graceful fallbacks)
    pts = df["ProjPoints"].fillna(0).astype(float).tolist()
    if objective == "points":
        metric = pts
    elif objective == "floor":
        if "ProjFloor" in df.columns:
            metric = df["ProjFloor"].fillna(0).astype(float).tolist()
        else:
            metric = [0.9 * v for v in pts]
    elif objective == "ceiling":
        if "ProjCeiling" in df.columns:
            metric = df["ProjCeiling"].fillna(0).astype(float).tolist()
        else:
            metric = [1.1 * v for v in pts]
    elif objective == "value":
        if "Value" in df.columns:
            metric = df["Value"].fillna(0).astype(float).tolist()
        else:
            # no salary baselines available: fall back to points
            metric = pts
    else:
        raise ValueError(f"Unknown objective: {objective}")

    weights, w_scale = _ints(metric, scale=1000)
    m.Maximize(sum(weights[i] * x[i] for i in idx))

    # solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible lineup found.")

    chosen = [i for i in idx if solver.BooleanValue(x[i])]
    sel = df.loc[chosen].copy()
    # which flex got used?
    flex_used = "RB" if solver.BooleanValue(rb_flex) else ("WR" if solver.BooleanValue(wr_flex) else "TE")

    return {
        "players": sel.to_dict(orient="records"),
        "total_salary": int(sel["salary"].sum()),
        "total_proj": float(sel.get("ProjPoints", pd.Series([0]*len(sel))).sum()),
        "count": {
            "QB": int((sel["pos"]=="QB").sum()),
            "RB": int((sel["pos"]=="RB").sum()),
            "WR": int((sel["pos"]=="WR").sum()),
            "TE": int((sel["pos"]=="TE").sum()),
            "DEF": int((sel["pos"]=="DEF").sum()),
        },
        "flex_used": flex_used,
        "objective": objective,
        "status": "OPT" if status == cp_model.OPTIMAL else "FEAS"
    }
PY
