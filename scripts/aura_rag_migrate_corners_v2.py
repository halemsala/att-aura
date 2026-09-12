"""Aura Fase 1 — Migrador v2: corner_observations_v2 (JSON aninhado).
Modo padrao: DESCOBERTA + DRY-RUN (nao escreve nada no RAG).
"""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "aura_live_learning.sqlite3"

CANON_SEARCH = {
    "home": ["home", "hometeam", "timecasa", "mandante", "timecasa"],
    "away": ["away", "awayteam", "visitante", "timefora"],
    "minute": ["minute", "minuto", "matchminute", "currentminute", "min"],
    "league": ["league", "liga", "competition", "campeonato", "tournament", "torneio", "championship"],
    "corners_home": ["cornershome", "homecorners", "escanteioscasa", "cornerhome"],
    "corners_away": ["cornersaway", "awaycorners", "escanteiosfora", "corneraway"],
    "score_home": ["scorehome", "goalshome", "placarcasa", "homegoals", "hscore"],
    "score_away": ["scoreaway", "goalsaway", "placarfora", "awaygoals", "ascore"],
    "line": ["line", "linha", "ahline", "handicap", "asianline", "linhadeescanteios"],
    "horizon_min": ["horizon", "horizonte", "horizonmin", "janelamin", "window"],
}
PROB_SEARCH = ["prob", "p", "probability", "confidence", "pover", "p5m", "probrecomendada"]


def _norm(s):
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def flatten_leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.update(flatten_leaves(v, p))
            else:
                out[_norm(k)] = v
                out[_norm(p)] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            out.update(flatten_leaves(v, prefix + f"[{i}]"))
    return out


conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
rows = conn.execute(
    "SELECT capture_key, fixture_id, captured_at, payload, analysis "
    "FROM corner_observations_v2").fetchall()
conn.close()
print(f"total rows: {len(rows)}")

payload_keys = Counter()
analysis_keys = Counter()
parsed = 0
mapped = []
for rk, fixture, ts, payload_s, analysis_s in rows:
    try:
        payload = json.loads(payload_s)
        parsed += 1
    except Exception:
        continue
    try:
        analysis = json.loads(analysis_s) if analysis_s else {}
    except Exception:
        analysis = {}
    for k in flatten_leaves(payload):
        payload_keys[k.split(".")[-1]] += 1
    if isinstance(analysis, dict):
        for k in analysis:
            analysis_keys[_norm(k)] += 1

    leaves = flatten_leaves(payload)
    feats = {}
    for canon, aliases in CANON_SEARCH.items():
        for a in aliases:
            if a in leaves and leaves[a] is not None:
                feats[canon] = leaves[a]
                break
    # corners como dict {home, away}
    if "corners" in payload and isinstance(payload.get("corners"), dict):
        ch = payload["corners"].get("home")
        ca = payload["corners"].get("away")
        if ch is not None and "corners_home" not in feats:
            feats["corners_home"] = ch
        if ca is not None and "corners_away" not in feats:
            feats["corners_away"] = ca
    prob = None
    a_leaves = flatten_leaves(analysis)
    for a in PROB_SEARCH:
        if a in a_leaves:
            prob = a_leaves[a]
            break
    mapped.append((rk, fixture, feats, prob, payload, analysis))

print(f"payloads parseados: {parsed}/{len(rows)}")
print("\n=== TOP 30 chaves do payload (ultimo segmento) ===")
for k, n in payload_keys.most_common(30):
    print(f"  {k:28s} {n:5d}  ({100*n//max(parsed,1)}%)")
print("\n=== TOP 15 chaves do analysis ===")
for k, n in analysis_keys.most_common(15):
    print(f"  {k:28s} {n:5d}  ({100*n//max(parsed,1)}%)")

print("\n=== PAYLOAD COMPLETO (amostra 1, ate 1200 chars) ===")
sample = mapped[0][4] if mapped else {}
print(json.dumps(sample, ensure_ascii=False, indent=1)[:1200])
print("\n=== ANALYSIS COMPLETO (amostra 1, ate 600 chars) ===")
print(json.dumps(mapped[0][5] if mapped else {}, ensure_ascii=False, indent=1)[:600])

print("\n=== COBERTURA DO MAPEAMENTO (features canonicas) ===")
total = len(mapped)
for canon in CANON_SEARCH:
    n = sum(1 for _, _, f, _, _, _ in mapped if canon in f)
    print(f"  {canon:14s} {n:5d}/{total}  ({100*n//max(total,1)}%)")
n_prob = sum(1 for _, _, _, p, _, _ in mapped if p is not None)
print(f"  {'prob(analysis)':14s} {n_prob:5d}/{total}  ({100*n_prob//max(total,1)}%)")

print("\n=== AMOSTRAS DE FEATURES MONTADAS (3) ===")
for rk, fixture, f, p, _, _ in mapped[:3]:
    print(f"  fixture={fixture} capture={rk}")
    print(f"    features: {f}")
    print(f"    prob: {p}")

ok = sum(1 for _, _, f, _, _, _ in mapped
         if len(f) >= 3 and ("home" in f or "away" in f) and ("minute" in f or "corners_home" in f))
print(f"\nVEREDITO: {ok}/{total} casos com features suficientes para indexar")
print("(modo dry-run: nada foi escrito no RAG)")
