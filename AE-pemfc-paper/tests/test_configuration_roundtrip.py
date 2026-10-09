"""局部加密完整配置可重载且不丢失物性单位与材料类型。"""
from dataclasses import asdict,replace
import json
from pemfc_coldstart.parameters import ModelParameters,Material
from pemfc_coldstart.io import load_configuration
from pemfc_coldstart.mesh import build_mesh

def test_nested_material_configuration_roundtrips(tmp_path):
 p=ModelParameters();mats=tuple(replace(m,cells_base=40) if m.name=='membrane' else m for m in p.materials)
 raw={'case':{'resolution':'base'},'parameter_overrides':{'materials':[asdict(m) for m in mats],'constants':asdict(p.constants)}}
 path=tmp_path/'config.json';path.write_text(json.dumps(raw))
 c,q,o,saved=load_configuration(path)
 assert q.materials==mats
 assert q.constants==p.constants
 assert build_mesh(q,c.resolution).n_cells==128
 assert saved==raw
