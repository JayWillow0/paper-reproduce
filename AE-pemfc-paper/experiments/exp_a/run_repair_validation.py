from pathlib import Path
from dataclasses import replace
import json,time,hashlib
from pemfc_coldstart.io import load_configuration,create_run_dir,save_result,refresh_manifest,config_hash
from pemfc_coldstart.solver import simulate
import pemfc_coldstart.solver as solver_module
class MonitoredBDF(solver_module.BDF):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.last_report=time.monotonic()
 def step(self):
  message=super().step()
  if time.monotonic()-self.last_report>10:
   print('PROGRESS',self.t,'step',self.step_size,'nfev',self.nfev,flush=True)
   self.last_report=time.monotonic()
  return message
solver_module.BDF=MonitoredBDF
from pemfc_coldstart.postprocess import plot_run
from pemfc_coldstart.diagnostics import validate_run,budgets
root=Path(__file__).resolve().parents[2]
selected=json.loads((root/'experiments/selected_runs.json').read_text())
keys=['exp_a_isothermal','exp_a_coupled','exp_d_liquid','exp_c_equilibrium_coarse']
index_path=root/'experiments/repair_runs.json'
index=json.loads(index_path.read_text()) if index_path.exists() else {}
for key in keys:
 old=root/selected[key]
 resolved=json.loads((old/'params_resolved.json').read_text())
 config=Path(resolved['source_config'])
 if not config.is_absolute(): config=root/config
 case,params,options,raw=load_configuration(config)
 # 旧配置若显式携带旧电荷容差，也以新协议的1e-12执行，并写入解析参数。
 options=replace(options,charge_tolerance=1e-12,max_accepted_steps=5000)
 raw={'case':vars(case),'parameter_overrides':raw.get('parameter_overrides',{}),'solver':vars(options),
      'repair_protocol':'full_charge_geometric_v2','baseline_run':selected[key]}
 if key in index:
  existing=root/index[key]['run']
  manifest=json.loads((existing/'manifest.json').read_text())
  hashes=manifest.get('source_sha256',{})
  same_source=all(hashlib.sha256((root/'src/pemfc_coldstart'/name).read_bytes()).hexdigest()==value for name,value in hashes.items()) and bool(hashes)
  if same_source and manifest.get('config_sha256')==config_hash(raw) and validate_run(existing)['valid']:
   print('REUSE',key,index[key]['run'],flush=True)
   continue
 run=create_run_dir(config.parent,raw)
 (run/'config.json').write_text(json.dumps(raw,indent=2,ensure_ascii=False)+'\n')
 start=time.time(); print('START',key,run.name,flush=True)
 result=simulate(case,params,options)
 save_result(run,result,case,params,options,raw,config)
 plot_run(run); refresh_manifest(run)
 validation=validate_run(run)
 index[key]={'run':str(run.relative_to(root)),'baseline':selected[key],
             'elapsed_wall_s':time.time()-start,'status':result.status,'end_time_s':float(result.time_s[-1]),
             'voltage_final_v':float(result.voltage_v[-1]),'ice_max':float(result.ice_cathode_max[-1]),
             'Tmax':float(result.temperature_max_k[-1]),'validation':validation,
             'budget':budgets(result,case,params)}
 (root/'experiments/repair_runs.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
 print('DONE',key,json.dumps(index[key],ensure_ascii=False),flush=True)
