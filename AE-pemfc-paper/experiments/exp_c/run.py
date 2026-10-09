"""图8四组相变机制入口。"""
from pathlib import Path
import subprocess, sys

root = Path(__file__).parent
for name in ("config.json", "config_kappa2.json", "config_kappa3.json", "config_equilibrium.json"):
    subprocess.run([sys.executable, "-m", "pemfc_coldstart.cli", "simulate", "--config", str(root / name)], check=True)
