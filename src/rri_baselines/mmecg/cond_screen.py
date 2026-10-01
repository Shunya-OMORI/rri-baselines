# 2026-10-01 の模擬探索(条件 h が使われるか)の記録. 使い方: python -m rri_baselines.mmecg.cond_screen <段数> <案...>  案は input_128 / all_all_nohist / all_all_2stage など
"""条件 h が使われるかの模擬探索. 分割 0 の学習側で短く学習し, val の窓で h を入れ替えたときの損失の悪化を見る."""
import sys, time, numpy as np, torch
import torch.nn.functional as F
import rri_baselines.mmecg.run as R
from rri_baselines.mmecg.model import MMECG, class_value
from rri_baselines.protocol import loso_folds, trials_of
STEPS = int(sys.argv[1]); variants = sys.argv[2:]
fold = loso_folds()[0]; tr = trials_of(fold.train); va = trials_of(fold.val)
data = {t: R.load(t) for t in tr + va}
pool, items = R.Pool(data, tr), R.train_items(data, tr)
vpool, vitems = R.Pool(data, va), R.train_items(data, va)
vb = vpool.batch(vitems[np.random.default_rng(1).choice(len(vitems), 128, replace=False)])
def probe(m, span, nohist=False):
    m.eval()
    with torch.no_grad():
        x, pos, cls = vb
        xp = torch.cat([torch.zeros_like(cls[:, :1], dtype=torch.float32), class_value(cls[:, :-1])], 1)
        xp = xp * 0 if nohist else xp
        h = m.encoder(x, pos); k = 128 if span == "128" else 640
        ce = lambda hh: F.cross_entropy(m.decoder(xp, hh)[:, :, -k:], cls[:, -k:]).item()
        g = torch.Generator(device=h.device).manual_seed(0)
        return ce(h), ce(h[torch.randperm(len(h), generator=g, device=h.device)]) - ce(h), h.std(0).mean().item(), h.std(-1).mean().item()
for v in variants:
    cond_all = "all" in v.split("_")[0]; span = v.split("_")[1]; nohist = "nohist" in v; hdrop = float(v.split("hdrop")[1]) / 100 if "hdrop" in v else 0.0
    torch.manual_seed(42); rng = np.random.default_rng(42)
    m = MMECG(cond_all=cond_all).to(R.DEV); opt = torch.optim.Adam(m.parameters(), lr=R.LR); t0 = time.time()
    for s in range(STEPS + 1):
        if s % (STEPS // 4) == 0:
            c, gap, sw, st = probe(m, span, nohist)
            print(f"[{v}] 段 {s}: val ce {c:.4f}  h を他の窓と入れ替えた悪化 {gap:+.4f}  h の窓間 std {sw:.5f} 時間 std {st:.5f}  ({time.time()-t0:.0f} s)", flush=True)
        if s == STEPS: break
        m.train(); x, pos, cls = pool.batch(items[rng.choice(len(items), R.BATCH, replace=False)])
        stage1 = "2stage" in v and s < STEPS // 2
        if nohist or stage1:
            loss = F.cross_entropy(m(x, pos, torch.zeros(cls.shape, device=cls.device)), cls)
        elif hdrop:
            xp = torch.cat([torch.zeros_like(cls[:, :1], dtype=torch.float32), class_value(cls[:, :-1])], 1)
            xp = xp * (torch.rand_like(xp) >= hdrop)                     # 過去の ECG の入力に dropout(Tacotron 2 の pre-net と同じ考え)
            k = 128 if span == "128" else 640
            loss = F.cross_entropy(m(x, pos, xp)[:, :, -k:], cls[:, -k:])
        else:
            loss = R.tf_loss(m, x, pos, cls, span)
        opt.zero_grad(); loss.backward(); opt.step()

    fb = R.train_mean_rr(tr); med = float(class_value(pool.cls.median()))
    if nohist:      # 過去を使わずに学習した模型は, 過去をゼロにした 1 回の順伝播(非自己回帰)で波形を出す
        from rri_baselines.mmecg.model import mu_law_decode
        from rri_baselines.protocol import eval_windows
        preds, rl = {}, []
        for t in va:
            st, _ = eval_windows(data[t]["n"], data[t]["peaks"]); st = st[::25]
            ha, hb = R.window_conditions(m, data[t], st); h = R.stitch(ha, hb, "switch640").to(R.DEV)
            with torch.no_grad():
                ecg = mu_law_decode(m.decoder(torch.zeros(h.shape[0], h.shape[2], device=R.DEV), h).argmax(1)).cpu().numpy()
            preds[t], r = R.rr_of(ecg, fb); rl.append(r)
        rules = np.concatenate(rl)
    else:
        preds, rules = R.predict(m, data, va, R.DEFAULT, fb, med, 25)
    print(f"[{v}] 生成した ECG の val RR 誤差(窓 5 s おき) {R.score_subsampled(preds, data, 25):.1f} ms  中央をまたぐ組 {np.mean(rules == 0):.0%}  (学習側平均 RR を出すだけなら参考に下の値)", flush=True)
    const = {t: np.full_like(p, fb) for t, p in preds.items()}
    print(f"[{v}] 定数(学習側平均)の val 誤差 {R.score_subsampled(const, data, 25):.1f} ms", flush=True)
