"""Aggregate the 3-seed per-component ablation (UBFC-trained)."""
import glob, re, collections, statistics as st
NAME={'canonicalphys_none':'none','canonicalphys_d1':'d1','canonicalphys_d2':'d2',
      'canonicalphys_d3':'d3','canonicalphys':'full'}
LABEL={'none':r'Warp only','d1':r'$+$Lam','d2':r'$+$Cons','d3':r'$+$Dist','full':r'$+$Lam$+$Cons$+$Dist (full)'}
TESTS=['UBFC-rPPG','PURE','OBF','MMPD']
# only count checkpoints trained today (the 3-seed _pfs run) to avoid old single-seed
rows=collections.defaultdict(lambda: collections.defaultdict(dict))
for rt in glob.glob('runs/eval/*/results.txt'):
    t=open(rt).read()
    if 'Training Dataset: UBFC' not in t: continue
    mp=re.search(r'Model Path:\s*(.*)',t); te=re.search(r'Test Dataset:\s*(\S+)',t); mae=re.search(r'FFT MAE \(FFT Label\):\s*([\d\.\-]+)',t)
    if not(mp and te and mae): continue
    m=re.search(r'/([a-z0-9_]+)_best\.pth',mp.group(1))
    name=m.group(1) if m else ''
    if name not in NAME: continue
    # training job id is the last numeric field of the train exp dir
    jm=re.search(r'/exp_\d+_\d+_(\d+)/',mp.group(1))
    if not jm: continue
    jid=int(jm.group(1))
    if not (19465577 <= jid <= 19465591): continue
    rows[NAME[name]][te.group(1)][jid]=float(mae.group(1))

def cell(combo,test):
    v=list(rows[combo].get(test,{}).values())
    if not v: return None
    m=sum(v)/len(v); s=st.pstdev(v) if len(v)>1 else 0.0
    return m,s,len(v)

print(f"{'combo':28s} "+" ".join(f"{t:>14s}" for t in TESTS))
for combo in ['none','d1','d2','d3','full']:
    parts=[]
    for test in TESTS:
        c=cell(combo,test)
        parts.append(f"{c[0]:6.2f}+/-{c[1]:.2f}(n{c[2]})" if c else "     --      ")
    print(f"{LABEL[combo][:28]:28s} "+" ".join(f"{p:>14s}" for p in parts))

print("\n=== LaTeX rows (UBFC PURE OBF MMPD) ===")
for combo in ['none','d1','d2','d3','full']:
    cs=[]
    for test in TESTS:
        c=cell(combo,test)
        cs.append(f"${c[0]:.2f}{{\\scriptstyle\\pm{c[1]:.2f}}}$" if c else "--")
    print(f"{LABEL[combo]:32s} & "+" & ".join(cs)+r" \\")
