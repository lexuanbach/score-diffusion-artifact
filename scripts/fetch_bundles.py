#!/usr/bin/env python3
"""Download release data/logits, verify SHA-256, then safely extract into this artifact."""
import argparse, hashlib, json, tarfile, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--only',choices=['data','predictions','all'],default='all')
a=p.parse_args();info=json.loads((ROOT/'bundles.json').read_text())
for name,meta in info['assets'].items():
    if a.only!='all' and name!=a.only+'.tar.gz':continue
    path=ROOT/'bundles'/name;path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url=f"https://github.com/{info['repository']}/releases/download/{info['release']}/{name}"
        part=path.with_suffix('.part');print(f'Downloading {url}',flush=True)
        urllib.request.urlretrieve(url,part);part.replace(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if path.stat().st_size!=meta['bytes'] or h.hexdigest()!=meta['sha256']:
        raise SystemExit(f'Checksum mismatch: {path}. Remove it and retry.')
    with tarfile.open(path,'r:gz') as t:
        t.extractall(ROOT,filter='data')
    print(f'Extracted verified {name}',flush=True)
