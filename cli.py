cat > dfs_opt/cli.py << 'PY'
from __future__ import annotations
import argparse
import pandas as pd
from .config import load_config
from .scoring.scoring import apply_custom_scoring
from .optimize.ilp import solve_lineup

def _demo_pool() -> pd.DataFrame:
    # Quick synthetic pool: enough to satisfy constraints
    rows = []
    # QB (3)
    rows += [
        {"player_id":"QB1","name":"Alpha QB","team":"AAA","pos":"QB","salary":8500,"pass_yd":280,"pass_td":2,"rush_yd":20,"rush_td":0,"rec":0,"rec_yd":0,"rec_td":0},
        {"player_id":"QB2","name":"Bravo QB","team":"BBB","pos":"QB","salary":7800,"pass_yd":305,"pass_td":2,"rush_yd":35,"rush_td":0,"rec":0,"rec_yd":0,"rec_td":0},
        {"player_id":"QB3","name":"Charlie QB","team":"CCC","pos":"QB","salary":7200,"pass_yd":240,"pass_td":1,"rush_yd":60,"rush_td":1,"rec":0,"rec_yd":0,"rec_td":0},
    ]
    # RB (6)
    for i,(sal,ry,rt,rec,ryd) in enumerate([
        (8200, 90,1, 3,20),(7200,110,0,4,15),(6800,60,1,5,40),
        (6000,45,0,3,25),(5400,100,0,2,10),(4800,75,0,2,15),
    ], start=1):
        rows.append({"player_id":f"RB{i}","name":f"Runner {i}","team":"RBX","pos":"RB","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":ry,"rush_td":rt,"rec":rec,"rec_yd":ryd,"rec_td":0})
    # WR (8)
    for i,(sal,rec,ryd,td) in enumerate([
        (8800,7,95,1),(7900,6,80,1),(7500,7,70,0),
        (7000,5,85,1),(6400,6,60,0),(5900,4,55,0),
        (5200,5,45,0),(4800,4,40,0),
    ], start=1):
        rows.append({"player_id":f"WR{i}","name":f"Wide {i}","team":"WRX","pos":"WR","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":0,"rush_td":0,"rec":rec,"rec_yd":ryd,"rec_td":td})
    # TE (3)
    for i,(sal,rec,ryd,td) in enumerate([(6400,6,55,1),(5200,5,40,0),(4200,3,35,0)], start=1):
        rows.append({"player_id":f"TE{i}","name":f"Tight {i}","team":"TEX","pos":"TE","salary":sal,"pass_yd":0,"pass_td":0,"rush_yd":0,"rush_td":0,"rec":rec,"rec_yd":ryd,"rec_td":td})
    # DEF (3)  <-- fixed loop + f-string
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

def cmd_version(_args):
    from . import __version__
    print(f"dfs_opt {__version__}")

def cmd_run_demo(_args):
    cfg = load_config()
    pool = _demo_pool()
    # For DEF, projections are zero; that's okay for the demo.
    pool = apply_custom_scoring(pool, cfg)
    result = solve_lineup(pool, cfg, objective="points")
    chosen = result["players"]
    df = pd.DataFrame(chosen, columns=["pos","name","team","salary","ProjPoints"]).sort_values(["pos","ProjPoints"], ascending=[True,False])
    print(df.to_string(index=False))
    print("\nTotals:")
    print(f"  Salary: {result['total_salary']}")
    print(f"  Proj:   {result['total_proj']:.2f}")
    print(f"  Counts: {result['count']}")

def main():
    ap = argparse.ArgumentParser(prog="dfs_opt")
    sub = ap.add_subparsers(required=True)

    p_ver = sub.add_parser("version", help="print version")
    p_ver.set_defaults(func=cmd_version)

    p_demo = sub.add_parser("run-demo", help="run a synthetic demo solve")
    p_demo.set_defaults(func=cmd_run_demo)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
PY
