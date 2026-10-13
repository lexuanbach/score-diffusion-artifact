#!/usr/bin/env python3
"""Rebuild readable numerical outputs and plots without manuscript/LaTeX sources."""
import argparse, csv, hashlib, json, os, re, runpy, shutil, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def plain(s):
    s = s.replace('$-$', '-').replace('$+$', '+').replace('\\%', '%')
    s = s.replace('\\pm', ' +/- ').replace('\\textdagger', '†').replace('\\dagger', '†')
    s = s.replace('\\to', ' → ').replace('\\le', '≤').replace('\\ge', '≥')
    s = re.sub(r'\\ci\{([^{}]*)\}', r' +/- \1', s)
    while re.search(r'\\[A-Za-z]+\{[^{}]*\}', s):
        s = re.sub(r'\\[A-Za-z]+\{([^{}]*)\}', r'\1', s)
    s = re.sub(r'\\(aps|daps|cfgnn|naps)\b', lambda m:m[1].upper(), s)
    return s.replace('$', '').replace('{', '').replace('}', '').replace('\\,','').replace('\\ ', ' ').strip()

def table_csv(source, target):
    rows=[]
    for line in source.read_text().splitlines():
        if '&' in line and not line.lstrip().startswith('\\multicolumn'):
            rows.append([plain(c) for c in line.split('\\\\')[0].split('&')])
    with target.open('w',newline='') as f: csv.writer(f, lineterminator='\n').writerows(rows)

def summaries(out):
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory(prefix='score-diffusion-') as tmp:
        os.environ['ARTIFACT_RENDER_DIR']=tmp
        state=runpy.run_path(str(ROOT/'src/render.py'),run_name='__main__')
        raw=state['NUM']
        (out/'numbers.json').write_text(json.dumps(raw,indent=2,sort_keys=True)+'\n')
        with (out/'numbers.csv').open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['key','value']);w.writerows((k,plain(v)) for k,v in sorted(raw.items()))
        for p in Path(tmp).iterdir():
            if p.name.startswith('tab_') and p.suffix=='.tex': table_csv(p,out/(p.stem+'.csv'))
            elif p.suffix in ('.dat','.csv'):shutil.copy2(p,out/p.name)
    expected=json.loads((ROOT/'expected/numbers.json').read_text())
    assert raw==expected, f'Number mismatch: {[k for k in set(raw)|set(expected) if raw.get(k)!=expected.get(k)][:15]}'
    checked=[]
    for p in (ROOT/'expected').iterdir():
        if p.suffix not in ('.dat','.csv'):continue
        q=out/p.name
        if not q.exists():raise AssertionError(f'Missing output {q}')
        x=pd.read_csv(p,sep=r'\s+' if p.suffix=='.dat' else ',')
        y=pd.read_csv(q,sep=r'\s+' if p.suffix=='.dat' else ',')
        pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-10,atol=1e-12)
        checked.append(p.name)
    print(f'PASS: {len(raw)} manuscript number values and {len(checked)} reference data files.')

def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#555555','#009E73','#0072B2','#D55E00']
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.8),layout='constrained')
    d=pd.read_csv(out/'fig_csbm_K2.dat',sep=r'\s+')
    for col,lam,c in zip(['l0','l25','l50','l100'],[0,.25,.5,1],colors):
        axes[0].plot(d.h,d[col],color=c,label=f'λ = {lam}',marker='o',markersize=3)
    axes[0].axvline(.5,color='.65',ls=':');axes[0].set(xlabel='Homophily h',ylabel='Mean set size',title='(a) Set size')
    for tag,lam,c,marker in [('l50',.5,colors[2],'o'),('l10',.1,colors[3],'s')]:
        d=pd.read_csv(out/f'fig_hstar_{tag}.dat',sep=r'\s+')
        axes[1].errorbar(d.dbar,d.hstar,yerr=[d.elo,d.ehi],fmt=marker,color=c,label=f'Data, {lam}',markersize=4)
        axes[1].plot(d.dbar,d.hstar_pred,'--',color=c,label=f'Theory, {lam}')
    axes[1].set(xscale='log',xlabel='Mean degree',ylabel='Crossover h*',title='(b) Crossover')
    axes[1].set_xticks([3,5,10,20,50], labels=['3','5','10','20','50'])
    axes[1].minorticks_off()
    axes[1].axhline(.5,color='.65',ls=':')
    d=pd.read_csv(out/'fig_labelfree.dat',sep=r'\s+')
    for flag,name,c,marker in [(1,'GNN',colors[2],'o'),(0,'MLP',colors[3],'^')]:
        z=d[d.gnn==flag];axes[2].scatter(z.hadj,z.gsoft,label=name,marker=marker,color=c,s=22,alpha=.8)
    axes[2].plot([-.12,.85],[-.12,.85],'--',color='.5',label='Identity')
    axes[2].set(xlabel='Label-based adjusted homophily',ylabel='Label-free estimate',title='(c) Homophily bias')
    for ax in axes:ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(out/'figure2.pdf');fig.savefig(out/'figure2.png',dpi=180);plt.close(fig)
    os.environ['FIGURE1_OUTPUT']=str(out/'figure1.json')
    runpy.run_path(str(ROOT/'src/figure1.py'),run_name='__main__')
    m=json.loads((out/'figure1.json').read_text())
    assert m==json.loads((ROOT/'expected/figure1.json').read_text()),'Figure 1 mismatch'
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.4),layout='constrained')
    axes[0].bar(['Own','Neighbours','Fixed mix'],[float(m['FigStrue']),float(m['FigNbScore']),float(m['FigSdtrue'])],color=[colors[1],colors[3],colors[2]])
    axes[0].set(title='Selected node: true-label score',ylabel='APS score')
    axes[1].bar(['Fixed','Label-based'],[float(m['FigMeanD']),float(m['FigMeanS'])],color=[colors[3],colors[1]])
    axes[1].set(title='Mean test set size',ylabel='Labels per node')
    axes[2].bar(['Overall','Worst class'],[float(m['FigCovMarg']),float(m['FigCovCls'])],color=[colors[1],colors[3]])
    axes[2].axhline(.95,color='.4',ls='--');axes[2].set(ylim=(0,1),title='Coverage audit',ylabel='Coverage')
    fig.savefig(out/'figure1-evidence.pdf');fig.savefig(out/'figure1-evidence.png',dpi=180);plt.close(fig)
    print('PASS: Figure 1 values and run-level checks. Wrote plot PDFs and PNGs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--figures',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'outputs');a=p.parse_args()
    os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True);summaries(a.out)
    if a.figures:figures(a.out)
