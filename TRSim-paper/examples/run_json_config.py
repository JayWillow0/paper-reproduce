import json
from pathlib import Path

from moduletr import ModuleConfig, run_simulation


path = Path(__file__).with_name("config.json")
config = ModuleConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
result = run_simulation(config, verbose=True)
print(json.dumps(result.to_dict("summary"), ensure_ascii=False, indent=2))

