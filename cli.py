cat > dfs_opt/cli.py << 'PY'
from __future__ import annotations
import argparse
import pandas as pd
from .config import load_config
from .scoring.scoring import apply_custom_scoring
from .optimize.ilp import solve_lineup

def _demo_pool() -> pd.DataFrame:
    rows = []
    rows += [
        {"player_id":"QB1","name":"Alpha QB","team":"AAA","pos":"QB","salary":8500,"pass_yd":280,"pass_td":2,"rush_yd":20,"rush_td":0,"rec":0,"rec_yd":0,"rec_td":0},
        {"player_id":"QB2","name":"Bravo QB","team":"BBB","pos":"QB","salary":7800,"pass_yd":305,"pass_td":2,"rush_yd":35,"rush_td":0,"rec":0,"rec_yd":0,"rec_td":0},
        {"player_id":"QB3","name":"Charlie QB","team":"CCC","pos":"QB","salary":7200,"pass_yd":240,"pass_td":1,"rush_yd":60,"rush_td":1,"rec":0,"rec_yd":0,"rec_td":0},
    ]
    for i,(sal,ry,rt,rec,ryd) in enumerate([
        (8200,90,1,3,20),(7200,110,0,4,15),(6800,60,1,5,40),
        (6000,45,0,3,25),(5400,100,0,2,10),(4800,75,0,2,15),
    ], start=1):
        rows.append({"player_id":f"RB{i}","name":f"Runner {i}","team":"RBX","pos":"RB","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":ry,"rush_td":rt,"rec":rec,"rec_yd":ryd,"rec_td":0})
    for i,(sal,rec,ryd,td) in enumerate([
        (8800,7,95,1),(7900,6,80,1),(7500,7,70,0),
        (7000,5,85,1),(6400,6,60,0),(5900,4,55,0),
        (5200,5,45,0),(4800,4,40,0),
    ], start=1):
        rows.append({"player_id":f"WR{i}","name":f"Wide {i}","team":"WRX","pos":"WR","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":0,"rush_td":0,"rec":rec,"rec_yd":ryd,"rec_td":td})
    for i,(sal,rec,ryd,td) in enumerate([(6400,6,55,1),(5200,5,40,0),(4200,3,35,0)], start=1):
        rows.append({"player_id":f"TE{i}","name":f"Tight {i}","team":"TEX","pos":"TE","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":0,"rush_td":0,"rec":rec,"rec_yd":ryd,"rec_td":td})
    for i, sal in enumerate([4000, 3600, 3200], start=1):
        rows.append({
            "player_id": f"DEF{i}",
            "name": f"Shield {i}",
            "team": f"D{i}",
            "pos": "DEF",
            "salary": sal,
            "pass_yd": 0, "pass_td": 0, "rush_yd": 0, "rush_td": 0, "rec": 0, "rec_yd": 0, "rec_td": 0
        })
    return pd.DataFrame(rows)

def _normalize_upper(df: pd.DataFrame, cols=("name","team","pos")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
    return df

def cmd_version(_args):
    from . import __version__
    print(f"dfs_opt {__version__}")

def cmd_run_demo(_args):
    cfg = load_config()
    pool = _demo_pool()
    pool = apply_custom_scoring(pool, cfg)
    result = solve_lineup(pool, cfg, objective="points")
    chosen = result["players"]
    df = pd.DataFrame(chosen, columns=["pos","name","team","salary","ProjPoints"]).sort_values(["pos","ProjPoints"], ascending=[True,False])
    print(df.to_string(index=False))
    print("\nTotals:")
    print(f"  Salary: {result['total_salary']}")
    print(f"  Proj:   {result['total_proj']:.2f}")
    print(f"  Counts: {result['count']}")

def cmd_run_csv(args):
    cfg = load_config()

    proj = pd.read_csv(args.proj_csv)
    sal  = pd.read_csv(args.sal_csv)

    # Normalize keys and ensure numeric stat columns exist
    proj = _normalize_upper(proj)
    sal  = _normalize_upper(sal)

    for col in ["pass_yd","pass_td","rush_yd","rush_td","rec","rec_yd","rec_td"]:
        if col not in proj.columns:
            proj[col] = 0

    # Ensure week/year present and aligned
    if "week" not in proj.columns:
        if args.week is None:
            raise SystemExit("projections CSV missing 'week' and --week not provided")
        proj["week"] = args.week
    if "year" not in proj.columns:
        if args.year is None:
            raise SystemExit("projections CSV missing 'year' and --year not provided")
        proj["year"] = args.year

    if "week" not in sal.columns:
        if args.week is None:
            raise SystemExit("salaries CSV missing 'week' and --week not provided")
        sal["week"] = args.week
    if "year" not in sal.columns:
        if args.year is None:
            raise SystemExit("salaries CSV missing 'year' and --year not provided")
        sal["year"] = args.year

    # Score projections with custom rules
    proj_scored = apply_custom_scoring(proj, cfg)

    # Join with salaries and clean
    merged = (proj_scored.merge(sal, on=["name","team","pos","week","year"], how="inner", suffixes=("","_sal"))
                        .rename(columns={"salary":"salary"}))
    if "salary" not in merged.columns:
        raise SystemExit("Joined data has no 'salary' column. Check column names in salaries CSV.")
    merged = merged.dropna(subset=["salary"]).copy()
    merged["salary"] = merged["salary"].astype(int)

    # Require minimal columns
    for col in ["player_id"]:
        if col not in merged.columns:
            # fabricate if missing
            merged[col] = merged["name"] + "|" + merged["team"] + "|" + merged["pos"]

    # Solve
    try:
        result = solve_lineup(merged, cfg, objective="points")
    except Exception as e:
        # Helpful hints if infeasible
        print("Solver failed:", e)
        print("Hints: ensure your merged pool has enough players per position and reasonable salaries.")
        print(merged["pos"].value_counts())
        return

    out_df = pd.DataFrame(result["players"], columns=["pos","name","team","salary","ProjPoints"])
    print(out_df.to_string(index=False))
    print("\nTotals:")
    print(f"  Salary: {result['total_salary']}")
    print(f"  Proj:   {result['total_proj']:.2f}")
    print(f"  Counts: {result['count']}")

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved lineup to {args.out}")

def main():
    ap = argparse.ArgumentParser(prog="dfs_opt")
    sub = ap.add_subparsers(required=True)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=cmd_version)

    p_demo = sub.add_parser("run-demo", help="run a synthetic demo solve")
    p_demo.set_defaults(func=cmd_run_demo)

    p_csv = sub.add_parser("run-csv", help="solve using projections & salaries CSVs")
    p_csv.add_argument("--proj-csv", required=True, help="Path to projections CSV")
    p_csv.add_argument("--sal-csv",  required=True, help="Path to salaries CSV")
    p_csv.add_argument("--year", type=int, help="Override/fill year if missing in CSVs")
    p_csv.add_argument("--week", type=int, help="Override/fill week if missing in CSVs")
    p_csv.add_argument("--out", help="Optional path to write lineup CSV")
    p_csv.set_defaults(func=cmd_run_csv)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
PY
