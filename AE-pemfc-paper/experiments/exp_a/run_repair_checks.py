"""三项修复的测试、编译和短时积分对照；输出独立诊断目录，不改文章选中运行。"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import time
import numpy as np
from pemfc_coldstart.parameters import ModelParameters, CaseConfig, SolverOptions
from pemfc_coldstart.solver import simulate
from pemfc_coldstart.diagnostics import budgets, output_hashes

root = Path(__file__).resolve().parents[2]
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = root/'experiments/exp_a/runs'/f'{stamp}_adversarial_audit'
output.mkdir()
(output/'diagnostic_only.txt').write_text('Independent repair checks; not a paper reproduction curve.\n')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
tests = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider'], cwd=root,env=env,capture_output=True,text=True)
(output/'pytest.log').write_text(tests.stdout+tests.stderr)
if tests.returncode:
    raise RuntimeError(f'Tests failed, see {output}')
files = sorted(f for directory in ['src','experiments','tests'] for f in (root/directory).rglob('*.py'))
with tempfile.TemporaryDirectory(prefix='pemfc_compile_') as cache:
    for k,file in enumerate(files):
        py_compile.compile(str(file),cfile=str(Path(cache)/f'{k}.pyc'),doraise=True)
(output/'compile.json').write_text(json.dumps({'status':'passed','files':[str(f.relative_to(root)) for f in files]},indent=2))
case = CaseConfig(thermal_mode='coupled',resolution='coarse',t_end_s=5,output_interval_s=.1)
params = ModelParameters()
rows = []
for label,method,rtol,atol in [('bdf','BDF',1e-6,1e-8),('radau','Radau',1e-6,1e-8),('bdf_tight','BDF',1e-7,1e-9)]:
    options = SolverOptions(method=method,rtol=rtol,atol=atol,max_step_s=.05)
    start = time.monotonic()
    result = simulate(case,params,options)
    np.savez_compressed(output/f'{label}.npz',time_s=result.time_s,state=result.state,
                        voltage_v=result.voltage_v,temperature_max_k=result.temperature_max_k,
                        lambda_cathode_mean=result.lambda_cathode_mean,ice_cathode_max=result.ice_cathode_max,
                        cumulative_budgets=result.cumulative_budgets)
    row={'label':label,'options':asdict(options),'wall_s':time.monotonic()-start,'status':result.status,
         'V':result.voltage_v[-1],'T':result.temperature_max_k[-1],'lambda':result.lambda_cathode_mean[-1],
         'ice':result.ice_cathode_max[-1],'budget':budgets(result,case,params)}
    rows.append(row)
    print(label,result.status,flush=True)
(output/'comparison.json').write_text(json.dumps({'case':asdict(case),'rows':rows},indent=2))
source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if '/runs/' not in str(f)}
manifest={'diagnostic_only':True,'command':'.venv/bin/python experiments/exp_a/run_repair_checks.py',
          'source_sha256':source_hashes,'outputs':output_hashes(output),'python':sys.version}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'experiments/repair_checks.json').write_text(json.dumps({'run':str(output.relative_to(root))},indent=2))
print('SAVED',output,flush=True)
