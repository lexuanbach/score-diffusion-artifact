#!/usr/bin/env python3
"""Run an experiment in a separate directory, leaving reference results untouched."""
import argparse, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workdir',type=Path,required=True)
p.add_argument('--retrain',action='store_true',help='Use an independent prediction directory instead of bundled logits')
p.add_argument('script',choices=[f.name for f in (ROOT/'src').glob('*.py') if f.name not in ('render.py','figure1.py')])
p.add_argument('args',nargs=argparse.REMAINDER)
a=p.parse_args();work=a.workdir.resolve()
if work==ROOT or ROOT.is_relative_to(work):p.error('Choose a dedicated child or external working directory')
(work/'results').mkdir(parents=True,exist_ok=True)
shutil.copytree(ROOT/'src',work/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
if not (work/'data').exists():(work/'data').symlink_to(ROOT/'data',target_is_directory=True)
preds=work/'results/preds'
if a.retrain:
    if preds.is_symlink():p.error('Use a new workdir for retraining')
    preds.mkdir(exist_ok=True)
elif not preds.exists():
    preds.symlink_to(ROOT/'results/preds',target_is_directory=True)
if a.script=='train_base.py' and not a.retrain:p.error('Training requires --retrain to protect bundled predictions')
subprocess.run([sys.executable,str(work/'src'/a.script),*a.args],cwd=work,check=True)
print(f'New experiment outputs: {work}/results')
