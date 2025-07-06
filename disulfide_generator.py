#!/usr/bin/env python
"""
disulfide_generator.py   ·   2025-07-05

build   – CSV of (+)/(–) Cys–Cys pairs with 16 numeric features
train   – FFNN (16→1) + early stopping (auto-epoch)
predict – rank candidate disulfides in a structure
"""

from __future__ import annotations
import argparse, itertools as it, math, random, sys, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import gemmi
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from tqdm import tqdm


# ────────────────────────── geometry helpers ────────────────────────── #
def _find(res: gemmi.Residue, name: str):
    try: return res.find_atom(name, '')
    except TypeError:
        try: return res.find_atom(name, '', gemmi.Element('X'))
        except TypeError:
            try: return res.find_atom(name)
            except Exception:
                for at in res:
                    if at.name.strip() == name.strip(): return at
                return None
def _pos(r,n): at=_find(r,n); return at.pos if at else None
def _N(r):  return _pos(r,'N')
def _C(r):  return _pos(r,'C')
def _CA(r): return _pos(r,'CA')
def _CB(r): p=_pos(r,'CB'); return p if p or r.name!='GLY' else _pos(r,'CA')
def _SG(r): return _pos(r,'SG')
def _dist(a,b): return None if a is None or b is None else a.dist(b)
def _tors(a,b,c,d): return None if None in (a,b,c,d) else math.degrees(gemmi.calculate_dihedral(a,b,c,d))
def _ang(a,b,c):   return None if None in (a,b,c) else math.degrees(gemmi.calculate_angle(a,b,c))
def _phi(p,c): return _tors(_C(p),_N(c),_CA(c),_C(c)) if p else None
def _psi(c,n): return _tors(_N(c),_CA(c),_C(c),_N(n)) if n else None
def _chi1(r): return _tors(_N(r),_CA(r),_CB(r),_SG(r))
def _is_atom(at):   # Gemmi ≤0.4 lacks is_atom()
    el=at.element
    return el.is_atom() if hasattr(el,'is_atom') else getattr(el,'number',0)!=0
def _asa_map(model):
    if hasattr(gemmi,'ShrakeRupley'):
        sr=gemmi.ShrakeRupley(); sr.probe_radius=1.4; sr.set_b_factors(True); sr.compute_b_factors(model)
        return { (ch.name,r.seqid.num): sum(a.b_iso for a in r if _is_atom(a))
                 for ch in model for r in ch}
    return {}
def _bfac(r): vals=[a.b_iso for a in r if _is_atom(a)]; return np.mean(vals) if vals else 0.0
def _safe(x): return 0.0 if x is None else x


# ───────────────────────── dataset builder ──────────────────────────── #
def _parse_one(path: Path):
    try: st=gemmi.read_structure(str(path))
    except Exception: return [],[]
    model=st[0]; asa=_asa_map(model)
    cys={ (ch.name,r.seqid.num):r for ch in model for r in ch if r.name=='CYS'}
    if len(cys)<2: return [],[]

    geom={}
    for ch in model:
        rl=list(ch)
        for i,r in enumerate(rl):
            prev=rl[i-1] if i else None; nxt=rl[i+1] if i<len(rl)-1 else None
            geom[(ch.name,r.seqid.num)]=(
                _phi(prev,r),_psi(r,nxt),_chi1(r),
                _ang(_CA(prev) if prev else None,_CA(r),_CA(nxt) if nxt else None),
                asa.get((ch.name,r.seqid.num),0.0), _bfac(r))

    pos=set()
    for co in getattr(st,'connections',[]):
        typ=getattr(co,'conn_type_id',None) or getattr(co,'type_id','')
        if str(typ).upper().startswith('DI'):
            k1=(co.partner1.asym_id,co.partner1.seq_id.num)
            k2=(co.partner2.asym_id,co.partner2.seq_id.num)
            pos.add(tuple(sorted((k1,k2))))
    for (k1,r1),(k2,r2) in it.combinations(cys.items(),2):
        if _dist(_SG(r1),_SG(r2)) and _dist(_SG(r1),_SG(r2))<=2.3:
            pos.add(tuple(sorted((k1,k2))))

    def row(k1,k2,label):
        (phi1,psi1,chi11,ang1,asa1,b1)=geom[k1]
        (phi2,psi2,chi21,ang2,asa2,b2)=geom[k2]
        cad=_dist(_CA(cys[k1]),_CA(cys[k2])); cbd=_dist(_CB(cys[k1]),_CB(cys[k2]))
        if None in (cad,cbd): return None
        seq=abs(k1[1]-k2[1]); same=int(k1[0]==k2[0])
        return [path.name,k1[0],k1[1],k2[0],k2[1],cad,cbd,seq,same,
                _safe(phi1),_safe(psi1),_safe(chi11),_safe(ang1),asa1,b1,
                _safe(phi2),_safe(psi2),_safe(chi21),_safe(ang2),asa2,b2,label]

    pos_rows,neg_rows=[],[]
    for k1,k2 in pos:
        r=row(k1,k2,1); pos_rows.append(r) if r else None
    cand=[(k1,k2) for (k1,r1),(k2,r2) in it.combinations(cys.items(),2)
          if (k1,k2) not in pos and _dist(_CB(r1),_CB(r2)) and _dist(_CB(r1),_CB(r2))<=8]
    random.Random(0xD15A).shuffle(cand)
    for k1,k2 in cand[:max(100,5*len(pos_rows))]:
        r=row(k1,k2,0); neg_rows.append(r) if r else None
    return pos_rows,neg_rows

def build(args):
    files=(sorted(Path(args.data_dir).rglob('*.cif'))+
           sorted(Path(args.data_dir).rglob('*.mmcif'))+
           sorted(Path(args.data_dir).rglob('*.cif.gz'))+
           sorted(Path(args.data_dir).rglob('*.mmcif.gz'))+
           sorted(Path(args.data_dir).rglob('*.pdb')))
    if not files: print("No structures found.",file=sys.stderr); sys.exit(1)
    hdr=['file','chain1','res1','chain2','res2','ca_dist','cb_dist','seq_sep','same_chain',
         'phi1','psi1','chi1_1','ang1','asa1','b1','phi2','psi2','chi1_2','ang2','asa2','b2','label']
    rows=[]
    with Pool(args.nproc or cpu_count()) as pool:
        for pos,neg in tqdm(pool.imap_unordered(_parse_one,files),total=len(files),desc='Parsing'):
            rows.extend(pos); rows.extend(neg)
    pd.DataFrame(rows,columns=hdr).to_csv(args.out_csv,index=False)
    print(f'Wrote {len(rows):,} rows → {args.out_csv}')


# ──────────────────────────── model & train ─────────────────────────── #
NUMERIC=['ca_dist','cb_dist','seq_sep','same_chain',
         'phi1','psi1','chi1_1','ang1','asa1','b1',
         'phi2','psi2','chi1_2','ang2','asa2','b2']
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16,32),nn.ReLU(),
                               nn.Linear(32,16),nn.ReLU(),
                               nn.Linear(16,1))
    def forward(self,x): return self.net(x)

def train(args):
    df=pd.read_csv(args.dataset)
    for col in ['ca_dist','cb_dist','seq_sep']: df[col]=np.log1p(df[col])
    for col in ['phi1','psi1','chi1_1','ang1','phi2','psi2','chi1_2','ang2']:
        df[col]=df[col]/180.0
    df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=NUMERIC)

    X=df[NUMERIC].values.astype('float32'); y=df['label'].values.astype('float32')
    idx=np.random.default_rng(0xBEEF).permutation(len(df)); split=int(0.8*len(df))
    Xtr,Xva=torch.tensor(X[idx[:split]]),torch.tensor(X[idx[split:]])
    ytr,yva=torch.tensor(y[idx[:split]])[:,None],torch.tensor(y[idx[split:]])[:,None]

    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=1e-3); lossF=nn.BCEWithLogitsLoss()

    # ── Early-stopping settings ──
    min_delta = 5e-5
    patience  = 25
    best_val  = np.inf
    best_ep   = 0
    epochs_no_gain = 0
    t0=time.time()

    for ep in range(1, args.epochs+1):
        net.train(); opt.zero_grad()
        loss=lossF(net(Xtr),ytr); loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            vloss=lossF(net(Xva),yva).item()

        if best_val - vloss > min_delta:
            best_val = vloss; best_ep = ep; epochs_no_gain = 0
            torch.save(net.state_dict(), args.model)
        else:
            epochs_no_gain += 1

        if ep==1 or ep%5==0:
            print(f'Epoch {ep:4d}  train {loss.item():.4f}  val {vloss:.4f}')

        if epochs_no_gain >= patience:
            print(f'⏸️  Early stop at epoch {ep} (no val gain for {patience} epochs).')
            break

    dt=time.time()-t0
    print(f'Best val {best_val:.4f} at epoch {best_ep}  •  {dt/60:.1f} min total')

# ─────────────────────────────── predict ────────────────────────────── #
def _geom_cache(model):
    asa=_asa_map(model); cache={}
    for ch in model:
        rl=list(ch)
        for i,r in enumerate(rl):
            prev=rl[i-1] if i else None; nxt=rl[i+1] if i<len(rl)-1 else None
            cache[(ch.name,r.seqid.num)]=(
                _safe(_phi(prev,r)),_safe(_psi(r,nxt)),_safe(_chi1(r)),
                _safe(_ang(_CA(prev) if prev else None,_CA(r),_CA(nxt) if nxt else None)),
                asa.get((ch.name,r.seqid.num),0.0),_bfac(r))
    return cache

def predict(args):
    net=Net(); net.load_state_dict(torch.load(args.model,map_location='cpu')); net.eval()
    st=gemmi.read_structure(str(args.structure)); model=st[0]; geom=_geom_cache(model)
    res=[((ch.name,r.seqid.num),r) for ch in model for r in ch]

    cand=[]
    for (k1,r1),(k2,r2) in it.combinations(res,2):
        dcb=_dist(_CB(r1),_CB(r2)); cad=_dist(_CA(r1),_CA(r2))
        if dcb is None or cad is None or dcb>args.cutoff: continue
        seq=abs(k1[1]-k2[1]); same=int(k1[0]==k2[0])
        (phi1,psi1,chi11,ang1,asa1,b1)=geom[k1]
        (phi2,psi2,chi21,ang2,asa2,b2)=geom[k2]
        feat=[np.log1p(cad),np.log1p(dcb),np.log1p(seq),same,
              phi1/180,psi1/180,chi11/180,ang1/180,asa1,b1,
              phi2/180,psi2/180,chi21/180,ang2/180,asa2,b2]
        cand.append((k1,k2,feat))
    if not cand: print("No pairs."); return
    with torch.no_grad():
        probs=torch.sigmoid(net(torch.tensor([c[2] for c in cand],dtype=torch.float32))).squeeze(1).numpy()
    rows=[[c[0][0],c[0][1],c[1][0],c[1][1],float(p)] for c,p in zip(cand,probs)]
    rows.sort(key=lambda r:-r[4])
    pd.DataFrame(rows[:args.top_k],columns=['chain1','res1','chain2','res2','prob']
                ).to_csv(args.out,index=False)
    print(f'Wrote {min(args.top_k,len(rows))} → {args.out}')


# ═════════════════════════════ CLI ════════════════════════════════════ #
def main():
    ap=argparse.ArgumentParser(description='Cys-Cys disulfide predictor')
    sub=ap.add_subparsers(dest='cmd',required=True)

    b=sub.add_parser('build');   b.add_argument('--data_dir',required=True); b.add_argument('--out_csv',required=True); b.add_argument('--nproc',type=int); b.set_defaults(func=build)
    t=sub.add_parser('train');   t.add_argument('--dataset',required=True); t.add_argument('--model',default='ss_model.pt'); t.add_argument('--epochs',type=int,default=1000); t.set_defaults(func=train)
    p=sub.add_parser('predict'); p.add_argument('--model',required=True);  p.add_argument('--structure',required=True); p.add_argument('--cutoff',type=float,default=8); p.add_argument('--top_k',type=int,default=25); p.add_argument('--out',default='preds.csv'); p.set_defaults(func=predict)

    args=ap.parse_args(); args.func(args)

if __name__=='__main__':
    main()
