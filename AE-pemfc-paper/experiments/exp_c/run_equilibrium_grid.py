"""平衡分支Jacobian修复后的空间网格对照；不覆盖历史选中运行。"""
from pathlib import Path
from dataclasses import asdict,replace
import json,time,hashlib
import pemfc_coldstart.solver as solver
from pemfc_coldstart.io import load_configuration,create_run_dir,save_result,refresh_manifest,config_hash
from pemfc_coldstart.diagnostics import validate_run,budgets
from pemfc_coldstart.postprocess import plot_run

root=Path(__file__).resolve().parents[2]
config=root/'experiments/exp_c/config_equilibrium.json'
case,params,options,original=load_configuration(config)
index_path=root/'experiments/equilibrium_grid_runs.json'
index=json.loads(index_path.read_text()) if index_path.exists() else {}
for label,method,rtol,atol in [('base_bdf','BDF',1e-6,1e-8),('fine_bdf','BDF',1e-6,1e-8)]:
 case=replace(case,resolution=label.split('_')[0])
 opts=replace(options,method=method,rtol=rtol,atol=atol,max_accepted_steps=5000)
 raw={'case':asdict(case),'solver':asdict(opts),'parameter_overrides':original.get('parameter_overrides',{}),'validation':'phase_preserving_jacobian_v1'}
 if label in index and (root/index[label]["run"]/"manifest.json").is_file():
  old=root/index[label]['run'];manifest=json.loads((old/'manifest.json').read_text())
  same=manifest.get('config_sha256')==config_hash(raw) and all(hashlib.sha256((root/'src/pemfc_coldstart'/f).read_bytes()).hexdigest()==h for f,h in manifest['source_sha256'].items())
  if same and validate_run(old)['valid']:
   print('REUSE',label,flush=True);continue
 run=create_run_dir(config.parent,raw);(run/'config.json').write_text(json.dumps(raw,indent=2)+'\n')
 print('START',label,run.name,flush=True);start=time.time()
 result=solver.simulate(case,params,opts)
 save_result(run,result,case,params,opts,raw,config);plot_run(run);refresh_manifest(run)
 index[label]={'run':str(run.relative_to(root)),'status':result.status,'end_time_s':float(result.time_s[-1]),'wall_s':time.time()-start,'V':float(result.voltage_v[-1]),'ice':float(result.ice_cathode_max[-1]),'lambda':float(result.lambda_cathode_mean[-1]),'budget':budgets(result,case,params),'validation':validate_run(run)}
 index_path.write_text(json.dumps(index,indent=2)+'\n')
 print('DONE',label,json.dumps(index[label]),flush=True)
