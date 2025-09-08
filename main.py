# main.py
import os, csv, math, time
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from hl_ws import HLFeed, HLMeta  # local file

# ================== Config ==================
COIN            = "ETH"
HORIZON_SEC     = 60.0         # predict +60s mid-return
EMIT_MS         = 200          # feature cadence (ms) ~5 Hz
BATCH_SIZE      = 256
REPLAY_CAP      = 12_000
WARMUP_LABELS   = 1500         # start trading after this many labeled samples
LR              = 1e-3
WD              = 1e-6
CLIP_NORM       = 1.0
DROP_PROB       = 0.10
THRESH_SIGMA    = 0.75         # act when |pred| > 0.75 * rolling_RMSE
RISK_PER_TRADE  = 0.003        # 0.3% equity at risk (scaled by vol)
LEVERAGE        = 25.0
PAPER_EQUITY    = 10_000.0
LOG_PATH        = "paper_fills.csv"
SEED            = 1337
torch.manual_seed(SEED); np.random.seed(SEED)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================ Feature Engine (raw microstructure) ================
class FeatureEngine:
    """
    Builds a 15-dim raw feature vector from L2 + trades at each frame.
    """
    def __init__(self, maxlen:int=90_000):
        self.buf: deque = deque(maxlen=maxlen)
    def update(self, f: Dict[str, Any]):
        self.buf.append(f)
    def _w(self, secs: float) -> List[Dict[str, Any]]:
        now = self.buf[-1]["ts"]; lo = now - secs
        return [x for x in self.buf if x["ts"] >= lo]
    @staticmethod
    def _imbalance(levels_b, levels_a) -> float:
        sb = sum(sz for _, sz, _ in levels_b[:5]); sa = sum(sz for _, sz, _ in levels_a[:5])
        tot = sb + sa;  return 0.0 if tot == 0 else (sb - sa)/tot
    @staticmethod
    def _microprice(b0, a0, sb, sa):
        if sb + sa <= 0:
            mid = 0.5*(b0+a0); return mid, 0.0
        micro = (a0*sb + b0*sa)/(sb+sa); mid = 0.5*(b0+a0)
        return micro, (micro-mid)/max(1e-12, mid)
    def features(self) -> Tuple[np.ndarray, Dict[str, float]]:
        cur = self.buf[-1]; mid = float(cur["mid"]); bb = cur["best_bid"]; ba = cur["best_ask"]
        bids, asks = cur["bids"], cur["asks"]
        spread = (ba - bb)/max(1e-12, mid)
        imb5 = self._imbalance(bids, asks)
        qimb = 0.0
        if bids and asks:
            sb1 = bids[0][1]; sa1 = asks[0][1]; tot = sb1+sa1; qimb = 0.0 if tot==0 else (sb1-sa1)/tot
        sb = sum(sz for _, sz, _ in bids[:5]); sa = sum(sz for _, sz, _ in asks[:5])
        _, mpskew = self._microprice(bb, ba, sb, sa)

        def ofi(win):
            if len(win) < 2: return 0.0
            v = 0.0
            for i in range(1,len(win)):
                b_prev, a_prev = win[i-1]["bids"][0], win[i-1]["asks"][0]
                b_cur,  a_cur  = win[i]["bids"][0],  win[i]["asks"][0]
                v += (b_cur[1]-b_prev[1]) - (a_cur[1]-a_prev[1])
            return v

        w1, w5, w10, w30 = self._w(1.0), self._w(5.0), self._w(10.0), self._w(30.0)
        ofi1, ofi5 = ofi(w1), ofi(w5)

        def signed_vol(win):
            sv = 0.0
            for fr in win:
                for (ts, px, sz, side) in fr.get("trades", []):
                    sv += side*sz
            return sv
        sv1, sv5 = signed_vol(w1), signed_vol(w5)

        def mom(win):
            if len(win) < 2: return 0.0
            return (win[-1]["mid"] - win[0]["mid"])/max(1e-12, win[0]["mid"])
        r1, r5, r15 = mom(w1), mom(w5), mom(self._w(15.0))

        def ewvol(win, alpha):
            if len(win) < 3: return 0.0
            m=0.0; v=0.0; prev=win[0]["mid"]
            for i in range(1,len(win)):
                r = (win[i]["mid"]-prev)/max(1e-12, prev); prev=win[i]["mid"]
                m = (1-alpha)*m + alpha*r
                v = (1-alpha)*v + alpha*(r-m)**2
            return math.sqrt(max(v,0.0))
        vol10, vol30 = ewvol(w10,0.2), ewvol(w30,0.1)

        def slope(levels):
            a = sum(sz for _, sz, _ in levels[:2]) + 1e-9
            b = sum(sz for _, sz, _ in levels[2:5]) + 1e-9
            return (a-b)/(a+b)
        bslope, aslope = slope(bids), slope(asks)

        x = np.array([spread, imb5, qimb, mpskew, ofi1, ofi5, sv1, sv5,
                      r1, r5, r15, vol10, vol30, bslope, aslope], dtype=np.float32)
        return x, {"mid": mid, "vol10": vol10, "ts": cur["ts"]}


# ================ Multi-Scale Exponential Bank ================
class MultiScaleBank:
    """
    Deterministic memory: exponential filters at multiple taus (secs) per feature.
    Produces 15 raw + 15*len(taus) decays + 15 cross-scale slopes = ~120 dims.
    """
    def __init__(self, n_raw:int=15, taus:Tuple[float,...]=(1,2,4,8,16,32)):
        self.n_raw = n_raw
        self.taus = taus
        # states[z][c][k] for channel c and tau index k
        self.z = np.zeros((n_raw, len(taus)), dtype=np.float32)
        self.last_ts: Optional[float] = None

    def update(self, x_raw: np.ndarray, now_ts: float) -> np.ndarray:
        if self.last_ts is None:
            self.last_ts = now_ts
            # first call: seed decays with raw
            for k,_ in enumerate(self.taus):
                self.z[:,k] = x_raw
        dt = max(1e-3, now_ts - self.last_ts); self.last_ts = now_ts

        # update decays with alpha = 1 - exp(-dt/tau)
        for k, tau in enumerate(self.taus):
            alpha = 1.0 - math.exp(-dt / float(tau))
            self.z[:,k] = (1.0 - alpha)*self.z[:,k] + alpha*x_raw

        # cross-scale slope per channel: fast - slow
        slope = self.z[:,0] - self.z[:,-1]  # 1s minus 32s

        # concat: [raw, decays(flat), slope]
        x_full = np.concatenate([x_raw, self.z.reshape(-1), slope], axis=0).astype(np.float32)
        return x_full


# ================ Replay Buffer ================
class Replay:
    def __init__(self, cap:int, dim:int):
        self.x = np.zeros((cap, dim), dtype=np.float32)
        self.y = np.zeros((cap, 1), dtype=np.float32)
        self.n = 0; self.cap = cap; self.i = 0
    def push(self, x: np.ndarray, y: float):
        self.x[self.i] = x; self.y[self.i,0] = y
        self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)
    def sample(self, k:int):
        k = min(k, self.n); idx = np.random.randint(0, self.n, size=k)
        return self.x[idx], self.y[idx]
    def __len__(self): return self.n


# ================ Tiny PyTorch Readout =================
class TinyMLP(nn.Module):
    def __init__(self, in_dim:int, drop:float=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)


# ================ Paper Trader =================
class PaperTrader:
    def __init__(self, equity=PAPER_EQUITY, leverage=LEVERAGE, log_path=LOG_PATH):
        self.eq = equity; self.lev = leverage
        self.pos = 0.0; self.avgpx = 0.0
        self.open_ts: Optional[float] = None
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["ts","event","sz","px","pos","eq","mtm"])
    def _round(self, v: float, step: float): return math.floor(v/step)*step
    def size_by_vol(self, mid: float, vol10: float, qstep: float, risk=RISK_PER_TRADE):
        if vol10 <= 1e-8: return 0.0
        notional = self.eq * risk * self.lev / max(vol10, 1e-6)
        tokens = notional / mid
        return self._round(tokens, qstep)
    def mtm(self, mid: float) -> float: return self.eq + (mid - self.avgpx)*self.pos
    def _log(self, ts, event, sz, px, mid):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, event, sz, px, self.pos, self.eq, self.mtm(mid)])
    def trade(self, side: int, sz: float, px: float, ts: float, mid: float):
        if sz <= 0: return
        new_pos = self.pos + side*sz
        if self.pos == 0.0 or np.sign(new_pos) != np.sign(self.pos):
            self.avgpx = px
        else:
            w_old, w_new = abs(self.pos), abs(side*sz)
            self.avgpx = (w_old*self.avgpx + w_new*px)/(w_old+w_new)
        self.pos = new_pos; self.open_ts = ts if self.pos!=0.0 else None
        self._log(ts, "BUY" if side>0 else "SELL", sz, px, mid)
    def flat(self, px: float, ts: float, mid: float):
        if self.pos == 0.0: return
        pnl = (px - self.avgpx) * self.pos
        self.eq += pnl; self.pos = 0.0; self._log(ts, "FLAT", 0.0, px, mid); self.open_ts=None


# ================ Wire-up & Run ================
def main():
    dev = device()
    print(f"[DEVICE] {dev} | torch {torch.__version__}")
    meta = HLMeta(); px_dec, sz_dec = meta.get_decimals(COIN)
    tick  = 10 ** (-px_dec); qstep = 10 ** (-sz_dec)
    print(f"[META] {COIN}: pxDecimals={px_dec} (tick={tick}), szDecimals={sz_dec} (qty step={qstep})")

    feed = HLFeed(COIN, poll_ms=EMIT_MS); fe = FeatureEngine()
    msb  = MultiScaleBank(n_raw=15, taus=(1,2,4,8,16,32))

    # feature dim: 15 raw + 15*6 decays + 15 slopes = 120
    in_dim = 15 + 15*6 + 15
    rep  = Replay(REPLAY_CAP, in_dim)

    model = TinyMLP(in_dim, drop=DROP_PROB).to(dev)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()

    # error calibrator for gating
    rmse = 1e-4
    trader = PaperTrader()

    fut = deque()  # (t_target, base_mid, x_full)
    last_ts: Optional[float] = None

    def on_frame(f: Dict[str, Any]):
        nonlocal rmse, last_ts
        fe.update(f)
        if len(fe.buf) < 30:
            return

        x_raw, ctx = fe.features()
        mid, vol10, now = ctx["mid"], ctx["vol10"], ctx["ts"]
        if last_ts is None: last_ts = now

        # build multi-scale expanded features
        x_full = msb.update(x_raw, now)

        # create future label handle
        fut.append((now + HORIZON_SEC, mid, x_full.copy()))

        # realize matured labels and train
        while fut and fut[0][0] <= now:
            _, base_mid, xf = fut.popleft()
            y = (mid - base_mid)/max(1e-12, base_mid)  # realized +60s return
            rep.push(xf, y)

        # Online train (light step)
        if len(rep) >= 256:
            xb, yb = rep.sample(BATCH_SIZE)
            xb_t = torch.from_numpy(xb).to(dev)
            yb_t = torch.from_numpy(yb).to(dev)
            model.train(); opt.zero_grad(set_to_none=True)
            pred = model(xb_t)
            loss = lossf(pred, yb_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            # update RMSE EMA
            batch_rmse = float(torch.sqrt(loss).detach().cpu())
            rmse = 0.98*rmse + 0.02*batch_rmse

        # Inference + trading (post warm-up)
        if len(rep) >= WARMUP_LABELS:
            with torch.no_grad():
                yhat = float(model(torch.from_numpy(x_full[None,:]).to(dev))[0,0].cpu())
            thresh = THRESH_SIGMA * max(1e-6, rmse)

            # time stop ≈ horizon
            if trader.open_ts and (now - trader.open_ts) >= HORIZON_SEC * 1.1:
                px = math.floor(mid / tick) * tick
                trader.flat(px, now, mid)

            # simple policy
            if yhat > thresh:
                sz = trader.size_by_vol(mid, vol10, qstep, RISK_PER_TRADE)
                if sz > 0.0:
                    px = math.floor(mid / tick) * tick
                    trader.trade(+1, sz, px, now, mid)
            elif yhat < -thresh:
                sz = trader.size_by_vol(mid, vol10, qstep, RISK_PER_TRADE)
                if sz > 0.0:
                    px = math.floor(mid / tick) * tick
                    trader.trade(-1, sz, px, now, mid)

        # occasional status
        if int(now) % 5 == 0:
            mtm = trader.mtm(mid)
            print(f"t={time.strftime('%H:%M:%S')} mid={mid:.2f} rmse={rmse*1e4:.2f}bp pos={trader.pos:.4f} eq≈{mtm:.2f}")

    feed.subscribe(on_frame)
    feed.start()
    print("[RUN] streaming… Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        feed.stop()
        print("Stopped.")

if __name__ == "__main__":
    main()
