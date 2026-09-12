"""Aura Fase 2.3 — Higienizacao: move historicos para archive/ (nunca deleta).
Nivel 1: padroes inequivocos -> move. Nivel 2: incertos -> so lista.
Manifesto JSON reversivel em archive/higienizacao_<ts>/manifest.json."""
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTECTED_DIRS = {
    "alfred", "engine", "bridge", "agents", "scripts", "desktop", "config",
    "data", "skills", "tests", "docs", "docs_acceptance", "docs_fonte",
    "venv", ".venv", ".vscode", "backups", "logs", "archive", "Att aura",
    "AuraDashboard", "aura-native", "aura-native-wpf", "aura-native-tests",
    "core", "knowledge", "runtime", "state", "schemas", "templates", "tools",
    "ops", "patches", "training", "evals", "addons", "interface", "extensao",
    "extensao_painel", "halem_control", "instalado", "work_aura",
    "aura-flow-orchestrator", "voice_profiles", ".git", ".continue",
    ".localharness", ".build-cache", ".pytest_temp",
}

DIR_PATTERNS_L1 = [
    r"^aura02_backup_", r"^aura_stage\d+_backup_", r"^backup_\d{8}_",
    r"^backup_autonomia_", r"^backup_etapa\d+_", r"^AURA_Etapa_",
    r"^AURA_HTML_para_WPF_", r"^AURA_Atualizacao_",
    r"^diagnostico_(completo|etapas|rapido)_\d", r"^inventario_total_\d",
    r"^aura_crlf_backup$", r"^e_all_backup_temp$", r"^_etapa\d+",
]

FILE_PATTERNS_L1 = [
    r"\.backup_\d", r"_BACKUP", r"\.bak_", r"^diagnostico_finalizador\.txt",
    r"^diagnostico_\d.*\.zip$", r"^coleta_fontes_harness\.zip$",
    r"^AURA_HTML_para_WPF_WebView2\.zip$",
]
TRASH_FILES = {"cat", "dir", "for", "type", "100MB)", "teste.txt",
               "fsdfsdfswfwhaergfgd.txt"}


def classify_dir(name: str) -> str:
    if name in PROTECTED_DIRS:
        return "protected"
    for pat in DIR_PATTERNS_L1:
        if re.match(pat, name):
            return "l1"
    return "unclassified"


def classify_file(name: str) -> str:
    if name in TRASH_FILES:
        return "l1"
    for pat in FILE_PATTERNS_L1:
        if re.search(pat, name):
            return "l1"
    return "skip"


def dir_size(p: Path) -> int:
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def main(mode: str) -> int:
    real = (mode == "--real")
    print(f"modo: {'REAL (move p/ archive)' if real else 'DRY-RUN (nao move nada)'}\n")
    l1, unclassified = [], []
    for child in sorted(ROOT.iterdir(), key=lambda c: c.name.lower()):
        name = child.name
        if child.is_dir():
            cls = classify_dir(name)
            if cls == "l1":
                l1.append(child)
            elif cls == "unclassified":
                unclassified.append(child)
        else:
            if classify_file(name) == "l1":
                l1.append(child)

    total = 0
    print(f"=== NIVEL 1: {len(l1)} itens inequivocos a mover ===")
    for c in l1:
        s = dir_size(c)
        total += s
        kind = "dir " if c.is_dir() else "file"
        print(f"  [{kind}] {c.name:60s} {fmt_mb(s):>10s}")
    print(f"  TOTAL a organizar: {fmt_mb(total)}\n")

    print(f"=== NIVEL 2: {len(unclassified)} pastas NAO classificadas (nao serao movidas) ===")
    for c in unclassified:
        print(f"  (?)  {c.name}")
    print("  (decida depois; nada foi tocado)\n")

    if not real:
        print("(dry-run: nada movido). Rode com --real apos confirmar.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "archive" / f"higienizacao_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"created": ts, "moved": [], "total_bytes": total,
                "revert": "mover itens de volta de archive/higienizacao_"
                          + ts + " para C:\\aura"}
    for c in l1:
        try:
            shutil.move(str(c), str(dest / c.name))
            manifest["moved"].append({"name": c.name,
                                      "type": "dir" if c.is_dir() else "file"})
            print(f"  movido: {c.name}")
        except Exception as e:
            print(f"  FALHA ao mover {c.name}: {e}")
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nConcluido: {len(manifest['moved'])} itens em {dest}")
    print(f"Espaco organizado: {fmt_mb(total)}")
    print("Reversivel via manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "--dry"))
