#!/usr/bin/env python
"""
disulfide_predictor.py   ·   2025-07-05

build   – create CSV of (+)/(–) Cys-Cys pairs with 16 features
train   – fit a small FFNN (16→1) using BCEWithLogits
predict – rank candidate disulfides in a new structure
"""

from __future__ import annotations
import argparse, itertools as it, math, random, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import gemmi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ══════════════════════════ geometry helpers ══════════════════════════ #

def _find(res: gemmi.Residue, name: str):
    """Return atom *name* regardless of Gemmi version."""
    try:
        return res.find_atom(name, '')
    except TypeError:
        try:
            return res.find_atom(name, '', gemmi.Element('X'))
        except TypeError:
            try:
                return res.find_atom(name)
            except Exception:
                for at in res:
                    if at.name.strip() == name.strip():
                        return at
                return None

def _pos(res, name):
    at = _find(res, name)
    return at.pos if at else None

def _N(res):  return _pos(res, 'N')
def _C(res):  return _pos(res, 'C')
def _CA(res): return _pos(res, 'CA')
def _CB(res):
    p = _pos(res, 'CB')
    return p if p or res.name != 'GLY' else _pos(res, 'CA')
def _SG(res): return _pos(res, 'SG')

def _dist(p, q):  return None if p is None or q is None else p.dist(q)

def _torsion(a, b, c, d):
    if None in (a, b, c, d): return None
    return math.degrees(gemmi.calculate_dihedral(a, b, c, d))

def _angle(a, b, c):
    if None in (a, b, c): return None
    return math.degrees(gemmi.calculate_angle(a, b, c))

def _phi(prev, cur):  return _torsion(_C(prev), _N(cur), _CA(cur), _C(cur)) if prev else None
def _psi(cur, nxt):   return _torsion(_N(cur), _CA(cur), _C(cur), _N(nxt)) if nxt else None
def _chi1(res):       return _torsion(_N(res), _CA(res), _CB(res), _SG(res))

def _is_atom(at: gemmi.Atom):
    el = at.element
    if hasattr(el, 'is_atom'):
        return el.is_atom()
    return getattr(el, 'number', 0) != 0

def _asa_map(model, probe=1.4):
    """Return ASA per residue; falls back to zeros if ShrakeRupley missing."""
    if hasattr(gemmi, 'ShrakeRupley'):
        sr = gemmi.ShrakeRupley()
        sr.probe_radius = probe
        sr.set_b_factors(True)
        sr.compute_b_factors(model)
        out = {}
        for ch in model:
            for res in ch:
                out[(ch.name, res.seqid.num)] = sum(
                    a.b_iso for a in res if _is_atom(a))
        return out
    return {}

def _bfac(res):
    vals = [a.b_iso for a in res if _is_atom(a)]
    return np.mean(vals) if vals else 0.0

def _safe(x): return 0.0 if x is None else x


# ═════════════════════════ dataset builder ════════════════════════════ #

def _parse_one(path: Path):
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        return [], []

    model = st[0]
    asa   = _asa_map(model)

    # collect cysteines
    cys = { (ch.name, res.seqid.num): res
            for ch in model for res in ch if res.name == 'CYS' }
    if len(cys) < 2:
        return [], []

    # per-residue geometry
    geom = {}
    for ch in model:
        rl = list(ch)
        for i, res in enumerate(rl):
            prev = rl[i-1] if i else None
            nxt  = rl[i+1] if i < len(rl)-1 else None
            geom[(ch.name, res.seqid.num)] = (
                _phi(prev, res), _psi(res, nxt), _chi1(res),
                _angle(_CA(prev) if prev else None, _CA(res),
                       _CA(nxt) if nxt else None),
                asa.get((ch.name, res.seqid.num), 0.0),
                _bfac(res)
            )

    # positives
    pos = set()
    for conn in getattr(st, 'connections', []):
        typ = getattr(conn, 'conn_type_id', None) or getattr(conn, 'type_id', '')
        if str(typ).upper().startswith('DI'):
            k1 = (conn.partner1.asym_id, conn.partner1.seq_id.num)
            k2 = (conn.partner2.asym_id, conn.partner2.seq_id.num)
            pos.add(tuple(sorted((k1, k2))))
    for (k1, r1), (k2, r2) in it.combinations(cys.items(), 2):
        dsg = _dist(_SG(r1), _SG(r2))
        if dsg is not None and dsg <= 2.3:
            pos.add(tuple(sorted((k1, k2))))

    def make_row(k1, k2, label):
        (phi1, psi1, chi11, ang1, asa1, b1) = geom[k1]
        (phi2, psi2, chi21, ang2, asa2, b2) = geom[k2]
        cad = _dist(_CA(cys[k1]), _CA(cys[k2]))
        cbd = _dist(_CB(cys[k1]), _CB(cys[k2]))
        if None in (cad, cbd):     # distances are mandatory
            return None
        seq  = abs(k1[1] - k2[1])
        same = int(k1[0] == k2[0])
        return [path.name, k1[0], k1[1], k2[0], k2[1],
                cad, cbd, seq, same,
                _safe(phi1), _safe(psi1), _safe(chi11), _safe(ang1), asa1, b1,
                _safe(phi2), _safe(psi2), _safe(chi21), _safe(ang2), asa2, b2,
                label]

    pos_rows, neg_rows = [], []
    for k1, k2 in pos:
        r = make_row(k1, k2, 1)
        if r: pos_rows.append(r)

    cand = []
    for (k1, r1), (k2, r2) in it.combinations(cys.items(), 2):
        if (k1, k2) in pos:
            continue
        dcb = _dist(_CB(r1), _CB(r2))
        if dcb is not None and dcb <= 8.0:
            cand.append((k1, k2))
    random.Random(0xD15A).shuffle(cand)
    for k1, k2 in cand[:max(100, 5*len(pos_rows))]:
        r = make_row(k1, k2, 0)
        if r: neg_rows.append(r)

    return pos_rows, neg_rows


def build(args):
    files = (
        sorted(Path(args.data_dir).rglob('*.cif')) +
        sorted(Path(args.data_dir).rglob('*.mmcif')) +
        sorted(Path(args.data_dir).rglob('*.cif.gz')) +
        sorted(Path(args.data_dir).rglob('*.mmcif.gz')) +
        sorted(Path(args.data_dir).rglob('*.pdb'))
    )
    if not files:
        print("No structures found in", args.data_dir, file=sys.stderr)
        sys.exit(1)

    header = ['file', 'chain1', 'res1', 'chain2', 'res2',
              'ca_dist', 'cb_dist', 'seq_sep', 'same_chain',
              'phi1', 'psi1', 'chi1_1', 'ang1', 'asa1', 'b1',
              'phi2', 'psi2', 'chi1_2', 'ang2', 'asa2', 'b2',
              'label']

    rows = []
    with Pool(args.nproc or cpu_count()) as pool:
        for pos, neg in tqdm(pool.imap_unordered(_parse_one, files),
                             total=len(files), desc='Parsing'):
            rows.extend(pos); rows.extend(neg)

    pd.DataFrame(rows, columns=header).to_csv(args.out_csv, index=False)
    print(f'Wrote {len(rows):,} rows → {args.out_csv}')


# ═════════════════════ model + training command ═══════════════════════ #

NUMERIC = [
    'ca_dist','cb_dist','seq_sep','same_chain',
    'phi1','psi1','chi1_1','ang1','asa1','b1',
    'phi2','psi2','chi1_2','ang2','asa2','b2'
]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16,  1)          # logits
        )
    def forward(self, x): return self.m(x)

def train(args):
    df = pd.read_csv(args.dataset)

    # transforms
    for col in ['ca_dist', 'cb_dist', 'seq_sep']:
        df[col] = np.log1p(df[col])
    for col in ['phi1','psi1','chi1_1','ang1','phi2','psi2','chi1_2','ang2']:
        df[col] = df[col] / 180.0

    # drop any row with remaining inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=NUMERIC)

    X = df[NUMERIC].astype('float32').values
    y = df['label'].values.astype('float32')

    rng = np.random.default_rng(0xBEEF)
    idx = rng.permutation(len(df));  split = int(0.8*len(df))
    Xtr,Xva = X[idx[:split]],X[idx[split:]]
    ytr,yva = y[idx[:split]],y[idx[split:]]
    Xtr,Xva = torch.tensor(Xtr),torch.tensor(Xva)
    ytr,yva = torch.tensor(ytr)[:,None],torch.tensor(yva)[:,None]

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossF = nn.BCEWithLogitsLoss()
    best = 9e9

    for ep in range(1, args.epochs+1):
        net.train()
        opt.zero_grad()
        loss = lossF(net(Xtr), ytr)
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            vloss = lossF(net(Xva), yva).item()

        if vloss < best:
            best = vloss
            torch.save(net.state_dict(), args.model)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f'Epoch {ep:3d}/{args.epochs}  train {loss.item():.4f}  val {vloss:.4f}')

    print(f'Best val {best:.4f}  saved → {args.model}')


# ═══════════════════════════ prediction ═══════════════════════════════ #

def _geom_cache(model):
    asa = _asa_map(model)
    cache = {}
    for ch in model:
        rl = list(ch)
        for i, res in enumerate(rl):
            prev = rl[i-1] if i else None
            nxt  = rl[i+1] if i < len(rl)-1 else None
            cache[(ch.name, res.seqid.num)] = (
                _safe(_phi(prev, res)),
                _safe(_psi(res, nxt)),
                _safe(_chi1(res)),
                _safe(_angle(_CA(prev) if prev else None, _CA(res),
                             _CA(nxt) if nxt else None)),
                asa.get((ch.name, res.seqid.num), 0.0),
                _bfac(res)
            )
    return cache

def predict(args):
    net = Net()
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net.eval()

    st = gemmi.read_structure(str(args.structure))
    model = st[0]
    geom  = _geom_cache(model)

    residues = [((ch.name, res.seqid.num), res) for ch in model for res in ch]
    cand = []
    for (k1,r1), (k2,r2) in it.combinations(residues, 2):
        dcb = _dist(_CB(r1), _CB(r2))
        if dcb is None or dcb > args.cutoff: continue
        cad = _dist(_CA(r1), _CA(r2))
        if cad is None: continue

        seq  = abs(k1[1]-k2[1]); same=int(k1[0]==k2[0])
        (phi1,psi1,chi11,ang1,asa1,b1)=geom[k1]
        (phi2,psi2,chi21,ang2,asa2,b2)=geom[k2]
        feat=[np.log1p(cad), np.log1p(dcb), np.log1p(seq), same,
              phi1/180, psi1/180, chi11/180, ang1/180, asa1, b1,
              phi2/180, psi2/180, chi21/180, ang2/180, asa2, b2]
        cand.append((k1,k2,feat))

    if not cand:
        print("No candidate pairs within cutoff."); return

    X = torch.tensor([c[2] for c in cand], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(net(X)).squeeze(1).numpy()

    rows=[[c[0][0],c[0][1],c[1][0],c[1][1],float(p)] for c,p in zip(cand,probs)]
    rows.sort(key=lambda r:-r[4])
    pd.DataFrame(rows[:args.top_k],
                 columns=['chain1','res1','chain2','res2','prob']
                 ).to_csv(args.out,index=False)
    print(f'Top {min(args.top_k,len(rows))} predictions → {args.out}')


# ═════════════════════════════ CLI ════════════════════════════════════ #

def main():
    ap = argparse.ArgumentParser(description='Cys-Cys disulfide predictor')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build')
    b.add_argument('--data_dir', required=True)
    b.add_argument('--out_csv', required=True)
    b.add_argument('--nproc', type=int)
    b.set_defaults(func=build)

    t = sub.add_parser('train')
    t.add_argument('--dataset', required=True)
    t.add_argument('--model', default='ss_model.pt')
    t.add_argument('--epochs', type=int, default=30)
    t.set_defaults(func=train)

    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--structure', required=True)
    p.add_argument('--cutoff', type=float, default=8.0)
    p.add_argument('--top_k', type=int, default=25)
    p.add_argument('--out', default='predictions.csv')
    p.set_defaults(func=predict)

    args = ap.parse_args(); args.func(args)

if __name__ == '__main__':
    main()
