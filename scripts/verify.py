#!/usr/bin/env python3
"""Check package and optional downloaded data/logits against stored checksums."""
import argparse, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundles',action='store_true');a=p.parse_args()
names=['package-files.json']+(['data-files.json','predictions-files.json'] if a.bundles else [])
count=0
for name in names:
    for rel,meta in json.loads((ROOT/'docs'/name).read_text()).items():
        if re.search(r'(^|[/_.-])(reviews?|audits?|rebuttal|sanity|validation)([/_.-]|$)', rel, re.I):
            raise SystemExit(f'Internal report is not allowed in the artifact: {rel}')
        if Path(rel).suffix.lower() in ('.tex', '.bib', '.sty', '.cls'):
            raise SystemExit(f'LaTeX source is not allowed in the artifact: {rel}')
        path=ROOT/rel
        if not path.is_file():raise SystemExit(f'Missing: {rel}')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if path.stat().st_size!=meta['bytes'] or h.hexdigest()!=meta['sha256']:raise SystemExit(f'Mismatch: {rel}')
        count+=1
print(f'PASS: {count} file checksums')
