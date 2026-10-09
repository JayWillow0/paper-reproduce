"""图5实验入口。"""
from pathlib import Path
import subprocess, sys

subprocess.run([sys.executable, "-m", "pemfc_coldstart.cli", "simulate", "--config", str(Path(__file__).with_name("config.json"))], check=True)
