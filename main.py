# eth_cb_meta_paper.py
# Single-file, online ETHUSDT(Perp) paper-trader using Binance USDT-M websockets.
# Novel AI core = coin-betting meta-learner over microstructure "experts" (no traditional ML).
# Predicts ~60s ahead; trades only if predicted edge clears spread/fees + latency buffer.
#
# -------- Quick start --------
#   pip install websockets
#   python eth_cb_meta_paper.py
#
# -------- What it does --------
# - Subscribes: ETHUSDT bookTicker, aggTrade, markPrice; BTCUSDT bookTicker for lead-lag
# - Coalesces to ~10Hz frames; builds features & 5 expert signals in [-1,1]
# - Meta-learner (coin-betting / Hedge) updates weights on *matured* 60s outcomes
# - Paper fills at top-of-book taker prices; fees & slippage applied; tick size 0.1
# - Logs to ./logs/ (ticks, features, predictions, trades); prints rolling summary
#
# -------- Notes --------
# - This is PAPER ONLY. No keys, no orders.
# - ETHUSDT Perp; USDT-M futures endpoint.
# - Default taker fee: 5 bps. Adjust if your actual is different.
# - Latency-aware edge buffer; TTL 60s exits; one position at a time for simplicity.

import asyncio
import json
import math
import os
import csv
import time
import signal
import statistics
from collections import deque, defaultdict
from datetime import datetime, timezone
from typing import Dict, Any, Deque, List, Tuple

import websockets

# =========================
# ---- CONFIGURATIONS -----
# =========================
SYMBOL       = "ethusdt"    # lowercase per Binance stream naming
LEADER_SYM   = "btcusdt"
WS_ENDPOINT  = (
    "wss://fstream.binance.com/stream"
    f"?streams="
    f"{SYMBOL}@bookTicker/"
    f"{SYMBOL}@aggTrade/"
    f"{SYMBOL}@markPrice@1s/"
    f"{LEADER_SYM}@bookTicker"
)

# Market/trading settings
TICK_SIZE          = 0.1         # enforce tick grid
TARGET_HORIZON_SEC = 60.0        # lookahead horizon for realized outcome
FRAME_HZ           = 10          # feature frame frequency (per second)
TAKER_FEE_BPS      = 5.0         # default taker fee (adjust to your tier)
SLIPPAGE_BPS       = 1.0         # extra bps cushion for taker slippage
LATENCY_BUFFER_TICKS = 0.5       # add ~0.5 tick to cost threshold for latency
MAX_SPREAD_TICKS   = 4.0         # don't trade when spread too wide
MAX_POSITION_NOTIONAL = 5000.0   # per trade notional cap (USDT)
LEVERAGE           = 25.0        # isolated 25x (used for risk context only)
TRADE_TTL_SEC      = 60.0        # close positions after ~60s
EDGE_THRESH_P      = 0.25        # min |meta-signal| to consider trading [0..1]

# Risk controls
MAX_CONSEC_LOSERS  = 5
STOP_ON_DRAWDOWN_PCT = 20.0      # stop if peak-to-trough equity drawdown exceeds this %

# Logging / Verbosity
VERBOSE            = True
LOG_DIR            = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID             = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TICKS_CSV          = os.path.join(LOG_DIR, f"ticks_{RUN_ID}.csv")
TRADES_CSV         = os.path.join(LOG_DIR, f"trades_{RUN_ID}.csv")
PRED_CSV           = os.path.join(LOG_DIR, f"pred_{RUN_ID}.csv")

# =========================
# ---- UTILITIES ----------
# =========================
def now_ms() -> int:
    return int(time.time() * 1000)

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"

def round_to_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-9) * tick

def bps(x: float) -> float:
    return x * 1e4

def from_bps(x_bps: float) -> float:
    return x_bps / 1e4

def safe_div(a: float, b: float, default=0.0) -> float:
    return a / b if b != 0 else default

# =========================
# ---- STATE OBJECTS ------
# =========================
class MarketState:
    """Holds last bookTicker, trades window, mid/history for ETH and BTC."""
    def __init__(self):
        self.eth_bid = None
        self.eth_bsz = None
        self.eth_ask = None
        self.eth_asz = None
        self.eth_mid = None
        self.eth_last_update_ms = 0

        self.btc_bid = None
        self.btc_ask = None
        self.btc_mid = None
        self.btc_last_update_ms = 0

        # Trades window (recent seconds)
        self.trade_win_sec = 3.0
        self.trades: Deque[Tuple[int, float, float, int]] = deque()  # (ms, price, qty, sign)

        # Rolling windows for RV/vol and mid path
        self.mid_hist: Deque[Tuple[int, float]] = deque()  # (ms, mid)
        self.window_ms = int(120 * 1000)  # keep ~2 minutes of mids

        # For OFI (using bookTicker deltas)
        self.prev_bsz = None
        self.prev_asz = None

    def on_book_ticker(self, is_eth: bool, bid: float, bsz: float, ask: float, asz: float, ts_ms: int):
        mid = 0.5 * (bid + ask)
        if is_eth:
            self.eth_bid, self.eth_bsz, self.eth_ask, self.eth_asz, self.eth_mid = bid, bsz, ask, asz, mid
            self.eth_last_update_ms = ts_ms
        else:
            self.btc_bid, self.btc_ask, self.btc_mid = bid, ask, mid
            self.btc_last_update_ms = ts_ms

        # record mid history (ETH only)
        if is_eth and mid is not None:
            self.mid_hist.append((ts_ms, mid))
            # prune
            cut = ts_ms - self.window_ms
            while self.mid_hist and self.mid_hist[0][0] < cut:
                self.mid_hist.popleft()

    def on_agg_trade(self, price: float, qty: float, is_buyer_maker: bool, ts_ms: int):
        # Aggressor sign: if buyer is maker -> seller aggressed (down), else buyer aggressed (up)
        sign = -1 if is_buyer_maker else +1
        self.trades.append((ts_ms, price, qty, sign))
        # prune
        cut = ts_ms - int(self.trade_win_sec * 1000)
        while self.trades and self.trades[0][0] < cut:
            self.trades.popleft()

    def get_spread_ticks(self) -> float:
        if self.eth_bid is None or self.eth_ask is None:
            return float("inf")
        return safe_div(self.eth_ask - self.eth_bid, TICK_SIZE, default=float("inf"))

    def rv_60s(self) -> float:
        """Rolling realized volatility proxy for 60s horizon (std of 1s returns last ~2min)."""
        # construct ~1s returns from mid_hist
        if len(self.mid_hist) < 5:
            return 0.0
        # sample per ~1s
        # (simple method: compute returns between ~1s-spaced points)
        ms_now = self.mid_hist[-1][0]
        pts = []
        last_t = ms_now - 60_000
        # gather about 60 points (past minute); fall back to whatever we have
        for t, m in self.mid_hist:
            if t >= last_t:
                pts.append((t, m))
        if len(pts) < 5:
            return 0.0
        rets = []
        for i in range(1, len(pts)):
            p0 = pts[i-1][1]
            p1 = pts[i][1]
            if p0 and p1:
                rets.append((p1 - p0) / p0)
        if len(rets) < 2:
            return 0.0
        try:
            return statistics.pstdev(rets)  # population std
        except statistics.StatisticsError:
            return 0.0

    def btc_ret_2s(self) -> float:
        """Approx 2s BTC return for lead-lag feature."""
        # we only store ETH mid path; for BTC we'll use last and a small buffer
        return 0.0  # will be filled by FeatureBuilder from its own buffer if needed


class FeatureBuilder:
    """Computes 10Hz features & expert signals in [-1,1]."""
    def __init__(self, ms: MarketState):
        self.ms = ms
        self.last_frame_ms = 0
        self.frame_interval_ms = int(1000 / FRAME_HZ)

        # Keep small buffers for deltas
        self.btc_mid_hist: Deque[Tuple[int, float]] = deque(maxlen=50)
        self.eth_l1_hist: Deque[Tuple[int, float, float]] = deque(maxlen=50)  # (ms, bsz, asz)

    def frame_ready(self, ts_ms: int) -> bool:
        if self.last_frame_ms == 0 or ts_ms - self.last_frame_ms >= self.frame_interval_ms:
            self.last_frame_ms = ts_ms
            return True
        return False

    def _norm(self, x: float, a: float) -> float:
        # symmetric squash: scale by a then tanh
        return math.tanh(safe_div(x, a, default=0.0))

    def build(self, ts_ms: int) -> Dict[str, float]:
        ms = self.ms
        feat = {}

        # L1 stats
        if (ms.eth_bid is None) or (ms.eth_ask is None) or (ms.eth_bsz is None) or (ms.eth_asz is None):
            return {}

        mid = ms.eth_mid
        spread = ms.eth_ask - ms.eth_bid
        spd_ticks = safe_div(spread, TICK_SIZE, default=99)

        # Store ETH L1 sizes for OFI
        self.eth_l1_hist.append((ts_ms, ms.eth_bsz, ms.eth_asz))
        if len(self.eth_l1_hist) >= 2:
            _, bsz_prev, asz_prev = self.eth_l1_hist[-2]
            d_bsz = (ms.eth_bsz - bsz_prev)
            d_asz = (ms.eth_asz - asz_prev)
        else:
            d_bsz, d_asz = 0.0, 0.0

        # Trades imbalance last ~3s
        buy_vol = sum(q for (_, _, q, sgn) in ms.trades if sgn > 0)
        sell_vol = sum(q for (_, _, q, sgn) in ms.trades if sgn < 0)
        vol_total = buy_vol + sell_vol
        trade_imb = safe_div((buy_vol - sell_vol), max(vol_total, 1e-9), default=0.0)

        # Microprice gap (weighted mid minus mid) normalized by spread
        # microprice = (ask*bid_size + bid*ask_size) / (bid_size + ask_size)
        denom = (ms.eth_bsz + ms.eth_asz)
        microprice = (ms.eth_ask * ms.eth_bsz + ms.eth_bid * ms.eth_asz) / denom if denom > 0 else mid
        mp_gap = safe_div((microprice - mid), max(spread, TICK_SIZE), 0.0)  # in "spreads"

        # Book pressure slope proxy: (Δ bid_size - Δ ask_size)
        ofi_l1 = d_bsz - d_asz

        # BTC lead-lag 2s: we maintain a small BTC buffer here
        if (self.ms.btc_mid is not None):
            self.btc_mid_hist.append((ts_ms, self.ms.btc_mid))
        # compute ~2s BTC return
        btc_ret_2s = 0.0
        if len(self.btc_mid_hist) >= 2:
            t_now = self.btc_mid_hist[-1][0]
            # find point ~2s ago
            target = t_now - 2000
            older = None
            for t, m in reversed(self.btc_mid_hist):
                if t <= target:
                    older = (t, m); break
            if older:
                m0 = older[1]; m1 = self.btc_mid_hist[-1][1]
                btc_ret_2s = safe_div((m1 - m0), m0, 0.0)

        # Rolling 60s RV (ETH)
        rv60 = self.ms.rv_60s()

        # Normalize/squash to manageable ranges
        feat["spd_ticks"]   = spd_ticks
        feat["trade_imb"]   = self._norm(trade_imb, 0.5)         # already in [-1,1]ish
        feat["mp_gap"]      = self._norm(mp_gap, 0.5)            # gap in spreads
        feat["ofi_l1"]      = self._norm(ofi_l1, 100.0)          # scale OFI
        feat["btc_ret_2s"]  = self._norm(btc_ret_2s, 0.002)      # ~20 bps scale
        feat["rv60"]        = rv60

        return feat

    def experts(self, feat: Dict[str, float]) -> Dict[str, float]:
        """Five bounded experts s_i in [-1,1] (no parameter training)."""
        if not feat:
            return {}

        s = {}
        # 1) OFI expert — pure order flow imbalance at L1
        s["ofi"] = math.tanh( feat["ofi_l1"] )

        # 2) Microprice curvature — follow microprice gap
        s["mp_follow"] = math.tanh( feat["mp_gap"] )

        # 3) Trade-imbalance momentum — follow signed volume
        s["trade_momo"] = math.tanh( feat["trade_imb"] )

        # 4) Spread-aware dampener — prefer following when spread tight
        s["tight_spread_follow"] = math.tanh( (1.5 - min(feat["spd_ticks"], 5.0)) * 0.8 )  # positive when tight

        # 5) BTC -> ETH lead — project BTC move to ETH direction
        s["btc_lead"] = math.tanh( feat["btc_ret_2s"] * 50.0 )  # amplify small btc returns

        return s


class CoinBettingMeta:
    """
    Online coin-betting / Hedge meta-learner over experts.
    - Maintains weights w over experts; predictions p_t = sum w_i * s_i_t in [-1,1]
    - Updates when 60s outcomes mature: reward_i = s_i_t * realized_return (in bps, clipped)
    - Multiplicative update with adaptive eta; normalized to simplex
    """
    def __init__(self, expert_names: List[str]):
        self.names = expert_names
        self.n = len(self.names)
        self.w = [1.0/self.n] * self.n
        self.cum_sq = [1e-6] * self.n  # for adaptivity
        self.eta_cap = 0.5
        self.r_clip = 15.0  # clip per-maturity expert reward in bps

    def predict(self, s: Dict[str, float]) -> float:
        if not s:
            return 0.0
        v = 0.0
        for i, name in enumerate(self.names):
            v += self.w[i] * float(s.get(name, 0.0))
        return max(min(v, 1.0), -1.0)

    def update(self, s: Dict[str, float], realized_bps: float):
        """Update weights on matured outcome; realized_bps is fee-inclusive minute return."""
        if not s:
            return
        # expert rewards
        r = []
        for name in self.names:
            ri = float(s.get(name, 0.0)) * realized_bps
            # clip each reward to control blowups
            ri = max(min(ri, self.r_clip), -self.r_clip)
            r.append(ri)

        # per-expert adaptive learning rate
        for i in range(self.n):
            self.cum_sq[i] += r[i]*r[i]
        # multiplicative-weights update
        new_w = []
        for i in range(self.n):
            eta_i = min(self.eta_cap, 1.0 / math.sqrt(self.cum_sq[i]))
            new_w.append(self.w[i] * math.exp(eta_i * r[i]))
        # normalize
        ssum = sum(new_w)
        if ssum <= 0 or not math.isfinite(ssum):
            self.w = [1.0/self.n] * self.n
            return
        self.w = [wi/ssum for wi in new_w]


class PaperBroker:
    """Very small paper broker with taker fills, fees, TTL exits, and 25x-aware notional cap."""
    def __init__(self, start_equity: float = 10_000.0):
        self.equity = start_equity
        self.cash   = start_equity
        self.pos_qty = 0.0         # positive long, negative short
        self.entry_price = None
        self.entry_time_ms = None
        self.trade_id = 0
        self.max_dd = 0.0
        self.peak_equity = start_equity
        self.consec_losers = 0

        # logging
        self.trades_out = open(TRADES_CSV, "w", newline="", encoding="utf-8")
        self.tw = csv.writer(self.trades_out)
        self.tw.writerow(["utc", "trade_id", "side", "qty", "entry_px", "exit_px",
                          "pnl_usdt", "pnl_bps", "hold_sec"])

    def should_stop(self) -> bool:
        dd = 100.0 * (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_dd = max(self.max_dd, dd)
        return dd >= STOP_ON_DRAWDOWN_PCT or self.consec_losers >= MAX_CONSEC_LOSERS

    def position_open(self) -> bool:
        return self.pos_qty != 0.0

    def _taker_fee_usdt(self, notional: float) -> float:
        return notional * from_bps(TAKER_FEE_BPS)

    def open(self, side: str, price: float, equity_hint: float):
        if self.position_open():
            return False
        # determine qty by notional cap & equity; 25x indicative leverage
        # choose notional = min(cap, 5x equity) to avoid over-sizing
        notional = min(MAX_POSITION_NOTIONAL, max(500.0, 5.0 * equity_hint))
        qty = notional / max(price, 1e-9)
        if side == "SHORT":
            qty = -qty

        # taker fee on entry
        fee = self._taker_fee_usdt(abs(qty) * price)
        self.cash -= fee

        self.pos_qty = qty
        self.entry_price = price
        self.entry_time_ms = now_ms()
        self.trade_id += 1
        if VERBOSE:
            print(f"[OPEN ] id={self.trade_id} side={side} qty={abs(qty):.4f} px={price:.2f} fee={fee:.2f}")
        return True

    def maybe_close_by_ttl(self, last_mid: float):
        if not self.position_open():
            return
        hold = (now_ms() - self.entry_time_ms) / 1000.0
        if hold >= TRADE_TTL_SEC:
            self.close(last_mid)

    def close(self, price: float):
        if not self.position_open():
            return
        side = "LONG" if self.pos_qty > 0 else "SHORT"
        qty = self.pos_qty
        entry_px = self.entry_price
        exit_px = price

        # taker fee on exit
        fee = self._taker_fee_usdt(abs(qty) * price)
        pnl_usdt = (exit_px - entry_px) * qty - fee - self._taker_fee_usdt(abs(qty) * entry_px)
        pnl_bps  = bps(safe_div(pnl_usdt, abs(qty) * entry_px, 0.0))

        self.cash += (entry_px * abs(qty)) + ((exit_px - entry_px) * qty)  # realization at exit
        self.equity += pnl_usdt

        self.peak_equity = max(self.peak_equity, self.equity)
        self.consec_losers = self.consec_losers + 1 if pnl_usdt < 0 else 0

        hold_sec = (now_ms() - self.entry_time_ms) / 1000.0
        self.tw.writerow([utc_ts(), self.trade_id, side, abs(qty), f"{entry_px:.2f}", f"{exit_px:.2f}",
                          f"{pnl_usdt:.4f}", f"{pnl_bps:.2f}", f"{hold_sec:.1f}"])
        self.trades_out.flush()

        if VERBOSE:
            print(f"[CLOSE] id={self.trade_id} side={side} qty={abs(qty):.4f} "
                  f"entry={entry_px:.2f} exit={exit_px:.2f} pnl={pnl_usdt:.2f} ({pnl_bps:.1f} bps) "
                  f"hold={hold_sec:.1f}s")

        # reset
        self.pos_qty = 0.0
        self.entry_price = None
        self.entry_time_ms = None

    def mark_to_market(self, mid: float):
        if not self.position_open() or mid is None:
            return
        # unrealized PnL (ignoring fees) for terminal summary
        unreal = (mid - self.entry_price) * self.pos_qty
        self.peak_equity = max(self.peak_equity, self.equity + unreal)

    def close_all(self, mid: float):
        if self.position_open() and mid is not None:
            self.close(mid)

    def shutdown(self):
        try:
            self.trades_out.close()
        except Exception:
            pass


class PredictorAndTrainer:
    """
    Glue that:
    - builds features each frame
    - emits a prediction p_t in [-1,1]
    - enqueues maturity updates (after 60s) and updates meta-weights on realization
    """
    def __init__(self, fb: FeatureBuilder, meta: CoinBettingMeta):
        self.fb = fb
        self.meta = meta
        # pending predictions: list of (maturity_ms, s_dict, entry_mid)
        self.pending: Deque[Tuple[int, Dict[str,float], float]] = deque()

        # logging
        self.pred_out = open(PRED_CSV, "w", newline="", encoding="utf-8")
        self.pw = csv.writer(self.pred_out)
        self.pw.writerow(["utc","mid","spd_ticks","trade_imb","mp_gap","ofi_l1","btc_ret_2s",
                          "rv60","p_meta","w_ofi","w_mp","w_trade","w_tight","w_btc"])

    def predict(self, ts_ms: int, feat: Dict[str, float], mid: float) -> float:
        s = self.fb.experts(feat)             # experts in [-1,1]
        p = self.meta.predict(s)              # meta signal in [-1,1]

        # enqueue maturity item using current mid
        self.pending.append((ts_ms + int(TARGET_HORIZON_SEC * 1000), s, mid))

        # log
        self.pw.writerow([
            utc_ts(),
            f"{mid:.2f}",
            f"{feat.get('spd_ticks',0):.3f}",
            f"{feat.get('trade_imb',0):.4f}",
            f"{feat.get('mp_gap',0):.4f}",
            f"{feat.get('ofi_l1',0):.4f}",
            f"{feat.get('btc_ret_2s',0):.4f}",
            f"{feat.get('rv60',0):.6f}",
            f"{p:.4f}",
            f"{self.meta.w[0]:.3f}",
            f"{self.meta.w[1]:.3f}",
            f"{self.meta.w[2]:.3f}",
            f"{self.meta.w[3]:.3f}",
            f"{self.meta.w[4]:.3f}",
        ])
        self.pred_out.flush()
        return p

    def mature(self, ts_ms: int, cur_mid: float):
        """Process any matured predictions; update meta-weights with realized 60s bps."""
        while self.pending and self.pending[0][0] <= ts_ms:
            _, s, entry_mid = self.pending.popleft()
            if entry_mid is None or cur_mid is None or entry_mid <= 0:
                continue
            ret = (cur_mid - entry_mid) / entry_mid
            realized_bps = bps(ret)
            # include a small friction to focus on edge that survives costs
            realized_bps -= (TAKER_FEE_BPS + SLIPPAGE_BPS) * 0.25
            self.meta.update(s, realized_bps)

    def shutdown(self):
        try:
            self.pred_out.close()
        except Exception:
            pass


# =========================
# ---- MAIN APP LOOP ------
# =========================
async def run():
    print(f"Connecting to {WS_ENDPOINT}")
    ms = MarketState()
    fb = FeatureBuilder(ms)
    meta = CoinBettingMeta(["ofi","mp_follow","trade_momo","tight_spread_follow","btc_lead"])
    pt = PredictorAndTrainer(fb, meta)
    broker = PaperBroker(start_equity=10_000.0)

    # tick logger
    ticks_out = open(TICKS_CSV, "w", newline="", encoding="utf-8")
    tw = csv.writer(ticks_out)
    tw.writerow(["utc","eth_bid","eth_bsz","eth_ask","eth_asz","eth_mid","btc_mid","spread_ticks"])

    # heartbeat / summary
    last_summary = time.time()
    trades_count = 0
    wins = 0
    losses = 0
    pnl_hist: Deque[float] = deque(maxlen=500)

    async def summary():
        nonlocal last_summary, trades_count, wins, losses
        if time.time() - last_summary >= 5.0:
            last_summary = time.time()
            eq = broker.equity
            dd = broker.max_dd
            wr = safe_div(wins, max(trades_count,1)) * 100.0
            avg = statistics.mean(pnl_hist) if pnl_hist else 0.0
            std = statistics.pstdev(pnl_hist) if len(pnl_hist) > 2 else 0.0
            sharpe = safe_div(avg, std) * math.sqrt(60) if std > 0 else 0.0  # scale ~per hour
            pos = "FLAT"
            if broker.position_open():
                pos = "LONG" if broker.pos_qty > 0 else "SHORT"
            print(f"[SUM ] eq={eq:.2f} dd={dd:.1f}% trades={trades_count} win%={wr:.1f} "
                  f"avgPnL={avg:.2f} std={std:.2f} sh~{sharpe:.2f} pos={pos} w={','.join(f'{x:.2f}' for x in meta.w)}")

    async with websockets.connect(WS_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        print("Connected. Streaming...")
        # graceful stop on Ctrl+C
        stop_flag = False

        def handle_sig(*_):
            nonlocal stop_flag
            stop_flag = True
            print("\n[CTRL-C] Stopping...")

        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)

        while not stop_flag:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("[WARN] websocket timeout; sending ping")
                try:
                    await ws.ping()
                except Exception as e:
                    print(f"[ERR ] ping failed: {e}")
                    break
                continue
            except Exception as e:
                print(f"[ERR ] websocket recv: {e}")
                break

            try:
                obj = json.loads(msg)
                stream = obj.get("stream","")
                data = obj.get("data",{})
                ts_ms = int(data.get("E") or data.get("T") or now_ms())
            except Exception:
                continue

            # --- Route streams ---
            if stream.endswith("@bookTicker"):
                s = data.get("s","")
                bid = float(data.get("b","0"))
                ask = float(data.get("a","0"))
                bsz = float(data.get("B","0"))
                asz = float(data.get("A","0"))

                if s.upper() == "ETHUSDT":
                    ms.on_book_ticker(True, bid, bsz, ask, asz, ts_ms)
                elif s.upper() == "BTCUSDT":
                    ms.on_book_ticker(False, bid, bsz, ask, asz, ts_ms)

                # log ETH ticks
                if ms.eth_mid is not None:
                    tw.writerow([utc_ts(),
                                 f"{ms.eth_bid:.2f}", f"{ms.eth_bsz:.4f}",
                                 f"{ms.eth_ask:.2f}", f"{ms.eth_asz:.4f}",
                                 f"{ms.eth_mid:.2f}",
                                 f"{(ms.btc_mid if ms.btc_mid else 0.0):.2f}",
                                 f"{ms.get_spread_ticks():.2f}"])
                    ticks_out.flush()

            elif stream.endswith("@aggTrade"):
                price = float(data.get("p","0"))
                qty   = float(data.get("q","0"))
                is_buyer_maker = bool(data.get("m", False))
                tms = int(data.get("T") or ts_ms)
                ms.on_agg_trade(price, qty, is_buyer_maker, tms)

            elif stream.endswith("@markPrice@1s"):
                # we don't use it directly for features, but having it is useful if you want to augment later
                pass

            # --- Frame update & decisioning at 10Hz ---
            if ms.eth_mid and fb.frame_ready(ts_ms):
                feat = fb.build(ts_ms)
                if not feat:
                    continue

                spd_ticks = feat["spd_ticks"]
                # maturity updates for 60s outcomes
                pt.mature(ts_ms, ms.eth_mid)

                # prediction
                p = pt.predict(ts_ms, feat, ms.eth_mid)   # [-1,1]
                rv60 = feat["rv60"]

                # Cost model → threshold in bps
                spread_bps  = bps(safe_div(ms.eth_ask - ms.eth_bid, ms.eth_mid, 0.0))
                fee_bps     = TAKER_FEE_BPS
                slip_bps    = SLIPPAGE_BPS
                lat_bps     = bps(LATENCY_BUFFER_TICKS * TICK_SIZE / ms.eth_mid)
                total_cost_bps = spread_bps + fee_bps + slip_bps + lat_bps

                # Expected move magnitude proxy: |p| * rv60 * 1e4 bps-scale * a small gain factor
                exp_bps = abs(p) * bps(rv60) * 1.5

                # Trading logic (one position at a time)
                if not broker.position_open():
                    # Preconditions
                    if spd_ticks <= MAX_SPREAD_TICKS and abs(p) >= EDGE_THRESH_P and exp_bps > total_cost_bps:
                        side = "LONG" if p > 0 else "SHORT"
                        # taker fill at best ask/bid with tick rounding
                        if side == "LONG":
                            px = round_to_tick(ms.eth_ask, TICK_SIZE)
                        else:
                            px = round_to_tick(ms.eth_bid, TICK_SIZE)
                        broker.open(side, px, broker.equity)
                else:
                    # manage open position: TTL exit or flip if strong opposite signal
                    broker.maybe_close_by_ttl(ms.eth_mid)
                    if broker.position_open():
                        # Optional early flip: if strong opposite signal and exp_bps clears cost*1.2
                        if abs(p) >= max(0.8, EDGE_THRESH_P + 0.2) and exp_bps > 1.2 * total_cost_bps:
                            want_side = "LONG" if p > 0 else "SHORT"
                            have_side = "LONG" if broker.pos_qty > 0 else "SHORT"
                            if want_side != have_side:
                                # close then reopen
                                exit_px = round_to_tick(ms.eth_mid, TICK_SIZE)
                                broker.close(exit_px)
                                # ensure we still want the trade
                                if spd_ticks <= MAX_SPREAD_TICKS:
                                    if want_side == "LONG":
                                        px = round_to_tick(ms.eth_ask, TICK_SIZE)
                                    else:
                                        px = round_to_tick(ms.eth_bid, TICK_SIZE)
                                    broker.open(want_side, px, broker.equity)

                # mark-to-market & summary
                broker.mark_to_market(ms.eth_mid)
                await summary()

                # risk kill-switch
                if broker.should_stop():
                    print("[STOP] Risk guard triggered. Flattening and exiting.")
                    break

        # loop end
        print("Closing positions and files...")
        broker.close_all(ms.eth_mid if ms.eth_mid else 0.0)
        broker.shutdown()
        pt.shutdown()
        ticks_out.close()
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
