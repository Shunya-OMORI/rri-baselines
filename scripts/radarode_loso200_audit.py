import json, math, numpy as np
from pathlib import Path
from rri_baselines.protocol import loso_folds, trials_of, eval_windows, load_trial, trials_by_subject
R=Path('runs/radarode_ew_wsst200')
lr0, eta=5e-3, 5e-5
def cos_lr(e, T):   # CosineAnnealingLR(閉形式, step を e+1 回した後の lr)
    k=e+1
    return eta+(lr0-eta)*(1+math.cos(math.pi*k/T))/2
print("分割 | エポック数 重複 欠落 | lr のずれ最大(T=100) | 選択: best_epoch vs val 最小 | 学習/val/test の重なり | 予測と暫定の一致 | 窓数の一致")
for f in loso_folds():
    d=R/f"fold{f.index:02d}_test{f.test[0]}"
    h=json.loads((d/'metrics.json').read_text())['history']
    eps=[x['epoch'] for x in h]
    dup=len(eps)-len(set(eps)); miss=sorted(set(range(200))-set(eps))
    lrdev=max(abs(x['lr']-cos_lr(x['epoch'],100)) for x in h)
    lrdev200=max(abs(x['lr']-cos_lr(x['epoch'],200)) for x in h)
    val=[x['val']['metrics']['PPI'][0] for x in h]
    be=json.loads((d/'best_epoch.json').read_text())['epoch']
    vmin=int(np.argmin(val)); tie=[i for i,v in enumerate(val) if v==min(val)]
    cfg=json.loads((d/'config.json').read_text())
    tr,va,te=set(f.train),set(f.val),set(f.test)
    ov=(tr&va)|(tr&te)|(va&te)
    z=np.load(d/'predictions.npz'); zp=np.load(d/'predictions_provisional.npz') if (d/'predictions_provisional.npz').exists() else None
    same=None
    if zp is not None:
        same=all(np.array_equal(z[k],zp[k]) for k in zp.files if k in z.files and k.endswith('_pred_ms'))
        maxd=max(float(np.max(np.abs(z[k]-zp[k]))) for k in zp.files if k in z.files and k.endswith('_pred_ms'))
    nwin=[]
    for t in trials_of(f.test):
        tr_=load_trial(t); n=len(eval_windows(len(tr_['rcg_z']),tr_['peaks'])[1]); nwin.append((t,len(z[f'trial{t}_pred_ms']),n))
    bad=[x for x in nwin if x[1]!=x[2]]
    print(f"{d.name} | {len(eps)} {dup} {miss[:3]} | {lrdev:.1e} (T=200 なら {lrdev200:.1e}) | {be} vs {vmin} 同値{len(tie)} | {sorted(ov)} t_max={cfg['t_max']} 学習{sorted(f.train)} val{f.val} test{f.test} | {same} (最大差 {maxd if zp is not None else '-'}) | 不一致 {bad[:3]}")
