"""图9三种产水机制入口。"""
from pathlib import Path
import subprocess, sys

root = Path(__file__).parent
for name in ("config.json", "config_liquid.json", "config_vapor.json"):
    subprocess.run([sys.executable, "-m", "pemfc_coldstart.cli", "simulate", "--config", str(root / name)], check=True)
