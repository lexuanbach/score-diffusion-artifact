#!/usr/bin/env python3
"""Print or execute the complete experiment sequence in an isolated workdir."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',type=Path,default=ROOT/'runs/full');p.add_argument('--device',default='cpu');p.add_argument('--workers',type=int,default=4);p.add_argument('--execute',action='store_true');p.add_argument('--retrain',action='store_true');a=p.parse_args()
c=json.loads((ROOT/'configs/experiments.json').read_text());ds=c['datasets'];mc=[d for d in ds if d not in c['binary_coverage_only']]
commands=[]
if a.retrain:
    commands+=[['train_base.py','--datasets',*ds,'--device',a.device],['train_base.py','--datasets','ogbn-arxiv','--models','GCN','--split','temporal','--device',a.device]]
commands += [['run_cp.py','--datasets',*ds],['run_cp.py','--datasets','Cora','Roman-empire','--models','GCN','--alphas','0.05','--cal-sizes','1000','100','200','500','2000','--tag','_cal'],['csbm.py','--workers',str(a.workers)],['analyze_csbm.py'],['cfgnn.py','--datasets',*mc,'--device',a.device],['labelfree.py'],['run_labelfree_gate.py'],['run_classwise.py'],['run_shift.py','--dataset','ogbn-arxiv','--scenario','temporal'],['run_shift.py','--dataset','Amazon-Computers','--scenario','degree'],['run_shift.py','--dataset','Amazon-ratings','--scenario','degree'],['e0_daps.py']]
for cmd in commands:
    args=[sys.executable,str(ROOT/'scripts/run_stage.py'),'--workdir',str(a.workdir)]+(['--retrain'] if a.retrain else [])+cmd
    print(' '.join(args),flush=True)
    if a.execute:subprocess.run(args,check=True)
