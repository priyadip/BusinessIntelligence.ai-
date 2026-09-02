#!/usr/bin/env bash
# One command from a fresh clone to a runnable prototype.
set -e
echo "1/3  installing dependencies"
python3 -m pip install -q -r requirements.txt
echo "2/3  generating the synthetic world (deterministic, ~40s)"
cd src && python3 -m casefile.sim.build_all 20260901
echo "3/3  running the four incidents"
python3 run_case.py all --llm-mode off
echo
echo "Done. Open results/workspace.html"
echo "Optional: pip install -r requirements-llm.txt  then  python3 run_case.py all --llm-mode local"
