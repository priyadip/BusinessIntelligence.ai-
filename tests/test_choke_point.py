"""The semantic gateway is only a choke point if nothing bypasses it."""
import ast, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1] / "casefile"
ALLOWED = {"semantic/gateway.py", "sim/warehouse.py"}   # sim builds the DB; engine may not

def test_no_direct_duckdb():
    bad = []
    for p in ROOT.rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        if rel.replace("\\", "/") in ALLOWED: continue
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import) and any(a.name == "duckdb" for a in n.names):
                bad.append(rel)
            if isinstance(n, ast.ImportFrom) and n.module == "duckdb":
                bad.append(rel)
    assert not bad, f"modules bypassing the semantic gateway: {sorted(set(bad))}"

if __name__ == "__main__":
    test_no_direct_duckdb(); print("PASS: no module bypasses the semantic gateway")
