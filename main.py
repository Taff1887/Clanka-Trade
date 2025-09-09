# main.py — ETHUSDT(Perp) paper trader (Binance USDT-M websockets)
# Online coin-betting meta-learner over microstructure experts (no Torch).
# Directional experts are reweighted online; a separate tight-spread gate scales signal.
# Predicts TARGET_HORIZON_SEC ahead; trades when expected edge > costs.

import asyncio, json, math, os, csv, time, signal, statistics
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Deque, List, Tuple
import websockets

# =========================
# ---- CONFIGURATIONS -----
# =========================
SYMBOL       = "ethusdt"
LEADER_SYM   = "btcusdt"
WS_ENDPOINT  = (
    "wss://fstream.binance.com/stream"
    f"?streams="
    f"{SYMBOL}@bookTicker/"
    f"{SYMBOL}@aggTrade/"
    f"{SYMBOL}@markPrice@1s/"
    f"{LEADER_SYM}@bookTicker"
)

# Trading/test tuning
TICK_SIZE            = 0.01
TARGET_HORIZON_SEC   = 20.0     # fast iteration; try 60 later
FRAME_HZ             = 10
TAKER_FEE_BPS        = 3.0      # adjust to your tier
SLIPPAGE_BPS         = 0.5
LATENCY_BUFFER_TICKS = 0.2
MAX_SPREAD_TICKS     = 8.0
MAX_POSITION_NOTIONAL= 5000.0
LEVERAGE             = 25.0
TRADE_TTL_SEC        = 20.0     # aligned to horizon during tests
EDGE_THRESH_P        = 0.08
RV_BPS_FLOOR         = 6.0      # softer floor than before
GAIN                 = 1.6      # modest boost to expected edge
COST_MULT            = 1.00     # require exp_edge >= full costs

# Risk controls
MAX_CONSEC_LOSERS    = 7
STOP_ON_DRAWDOWN_PCT = 20.0

# Logging / Verbosity
VERBOSE              = True
LOG_DIR              = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
RUN_ID               = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TICKS_CSV            = os.path.join(LOG_DIR, f"ticks_{RUN_ID}.csv")
TRADES_CSV           = os.path.join(LOG_DIR, f"trades_{RUN_ID}.csv")
PRED_CSV             = os.path.join(LOG_DIR, f"pred_{RUN_ID}.csv")

# =========================
# ---- UTILITIES ----------
# =========================
def now_ms() -> int: return int(time.time() * 1000)
def utc_ts() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
def round_down_to_tick(px: float, tick: float) -> float: return math.floor(px / tick + 1e-9) * tick
def round_up_to_tick(px: float, tick: float) -> float:   return math.ceil(px / tick - 1e-9) * tick
def bps(x: float) -> float: return x * 1e4
def from_bps(x_bps: float) -> float: return x_bps / 1e4
def safe_div(a: float, b: float, default=0.0) -> float: return a / b if b else default
def clamp(x, lo, hi): return lo if x < lo else (hi if x > hi else x)

# =========================
# ---- STATE OBJECTS ------
# =========================
class MarketState:
    def __init__(self):
        self.eth_bid = self.eth_bsz = self.eth_ask = self.eth_asz = self.eth_mid = None
        self.eth_last_update_ms = 0
        self.btc_bid = self.btc_ask = self.btc_mid = None
        self.btc_last_update_ms = 0
        self.trade_win_sec = 3.0
        self.trades: Deque[Tuple[int, float, float, int]] = deque()
        self.mid_hist: Deque[Tuple[int, float]] = deque()
        self.window_ms = int(180 * 1000)  # keep ~3min so we can compute arbitrary horizons

    def on_book_ticker(self, is_eth: bool, bid: float, bsz: float, ask: float, asz: float, ts_ms: int):
        mid = 0.5 * (bid + ask)
        if is_eth:
            self.eth_bid, self.eth_bsz, self.eth_ask, self.eth_asz, self.eth_mid = bid, bsz, ask, asz, mid
            self.eth_last_update_ms = ts_ms
            if mid is not None:
                self.mid_hist.append((ts_ms, mid))
                cut = ts_ms - self.window_ms
                while self.mid_hist and self.mid_hist[0][0] < cut:
                    self.mid_hist.popleft()
        else:
            self.btc_bid, self.btc_ask, self.btc_mid = bid, ask, mid
            self.btc_last_update_ms = ts_ms

    def on_agg_trade(self, price: float, qty: float, is_buyer_maker: bool, ts_ms: int):
        sign = -1 if is_buyer_maker else +1
        self.trades.append((ts_ms, price, qty, sign))
        cut = ts_ms - int(self.trade_win_sec * 1000)
        while self.trades and self.trades[0][0] < cut:
            self.trades.popleft()

    def get_spread_ticks(self) -> float:
        if self.eth_bid is None or self.eth_ask is None:
            return float("inf")
        return safe_div(self.eth_ask - self.eth_bid, TICK_SIZE, float("inf"))

    def rv_tau(self, seconds: float) -> float:
        """Std of ~1s returns over the last `seconds` seconds."""
        if len(self.mid_hist) < 6:
            return 0.0
        t_now = self.mid_hist[-1][0]
        window_start = t_now - int(seconds * 1000)
        pts, last_t = [], 0
        for t, m in self.mid_hist:
            if t >= window_start and (not pts or t - last_t >= 900):  # ~1s
                pts.append((t, m)); last_t = t
        if len(pts) < 5: return 0.0
        rets = [(pts[i][1]-pts[i-1][1])/pts[i-1][1] for i in range(1, len(pts)) if pts[i-1][1] > 0]
        if len(rets) < 2: return 0.0
        try: return statistics.pstdev(rets)
        except statistics.StatisticsError: return 0.0

class FeatureBuilder:
    def __init__(self, ms: MarketState):
        self.ms = ms
        self.last_frame_ms = 0
        self.frame_interval_ms = int(1000 / FRAME_HZ)
        self.btc_mid_hist: Deque[Tuple[int, float]] = deque(maxlen=120)
        self.eth_l1_hist: Deque[Tuple[int, float, float]] = deque(maxlen=120)

    def frame_ready(self, ts_ms: int) -> bool:
        if self.last_frame_ms == 0 or ts_ms - self.last_frame_ms >= self.frame_interval_ms:
            self.last_frame_ms = ts_ms
            return True
        return False

    def _norm(self, x: float, a: float) -> float:
        return math.tanh(safe_div(x, a, 0.0))

    def build(self, ts_ms: int) -> Dict[str, float]:
        ms = self.ms
        if (ms.eth_bid is None) or (ms.eth_ask is None) or (ms.eth_bsz is None) or (ms.eth_asz is None):
            return {}

        mid = ms.eth_mid
        spread = ms.eth_ask - ms.eth_bid
        spd_ticks = safe_div(spread, TICK_SIZE, 99)

        # L1 deltas (OFI proxy)
        self.eth_l1_hist.append((ts_ms, ms.eth_bsz, ms.eth_asz))
        if len(self.eth_l1_hist) >= 2:
            _, bsz_prev, asz_prev = self.eth_l1_hist[-2]
            d_bsz = ms.eth_bsz - bsz_prev
            d_asz = ms.eth_asz - asz_prev
        else:
            d_bsz = d_asz = 0.0

        # Trade imbalance (~3s)
        buy_vol  = sum(q for (_, _, q, sgn) in ms.trades if sgn > 0)
        sell_vol = sum(q for (_, _, q, sgn) in ms.trades if sgn < 0)
        vol_total = buy_vol + sell_vol
        trade_imb = safe_div((buy_vol - sell_vol), max(vol_total, 1e-9), 0.0)

        # Microprice gap (in spreads)
        denom = ms.eth_bsz + ms.eth_asz
        microprice = (ms.eth_ask * ms.eth_bsz + ms.eth_bid * ms.eth_asz) / denom if denom > 0 else mid
        mp_gap = safe_div((microprice - mid), max(spread, TICK_SIZE), 0.0)

        # BTC lead (~2s)
        if ms.btc_mid is not None:
            self.btc_mid_hist.append((ts_ms, ms.btc_mid))
        btc_ret_2s = 0.0
        if len(self.btc_mid_hist) >= 2:
            t_now = self.btc_mid_hist[-1][0]
            target = t_now - 2000
            older = None
            for t, m in reversed(self.btc_mid_hist):
                if t <= target: older = (t, m); break
            if older:
                m0, m1 = older[1], self.btc_mid_hist[-1][1]
                btc_ret_2s = safe_div((m1 - m0), m0, 0.0)

        # RV at the prediction horizon
        rv = self.ms.rv_tau(TARGET_HORIZON_SEC)

        return {
            "spd_ticks":  spd_ticks,
            "trade_imb":  self._norm(trade_imb, 0.5),
            "mp_gap":     self._norm(mp_gap, 0.5),
            "ofi_l1":     self._norm((d_bsz - d_asz), 100.0),
            "btc_ret_2s": self._norm(btc_ret_2s, 0.002),
            "rv":         rv,
        }

    # ---------- experts (directional only) + a separate gate ----------
    def experts_directional(self, feat: Dict[str, float]) -> Dict[str, float]:
        if not feat: return {}
        s = {}
        s["ofi"]        = math.tanh(feat["ofi_l1"])
        s["mp_follow"]  = math.tanh(feat["mp_gap"])
        s["trade_momo"] = math.tanh(feat["trade_imb"])
        s["btc_lead"]   = math.tanh(feat["btc_ret_2s"] * 50.0)
        return s

    def tight_spread_gate(self, feat: Dict[str, float]) -> float:
        # Gate in [0,1]: 1 when spread≈1 tick, fades as it widens
        g = math.tanh(max(0.0, (1.8 - min(feat["spd_ticks"], 8.0))) * 0.9)
        return clamp((g + 1.0) * 0.5, 0.0, 1.0)  # map tanh→[0,1]

class CoinBettingMeta:
    def __init__(self, expert_names: List[str]):
        self.names = expert_names
        self.n = len(self.names)
        self.w = [1.0/self.n] * self.n
        self.cum_sq = [1e-4] * self.n
        self.eta_cap = 0.25            # slower updates
        self.r_clip = 6.0              # tighter reward clip (bps)

    def predict(self, s: Dict[str, float]) -> float:
        if not s: return 0.0
        v = 0.0
        for i, name in enumerate(self.names):
            v += self.w[i] * float(s.get(name, 0.0))
        return clamp(v, -1.0, 1.0)

    def update(self, s: Dict[str, float], realized_bps: float):
        if not s: return
        r = []
        for name in self.names:
            ri = float(s.get(name, 0.0)) * realized_bps
            r.append(clamp(ri, -self.r_clip, self.r_clip))
        for i in range(self.n): self.cum_sq[i] += r[i]*r[i]
        new_w = []
        for i in range(self.n):
            eta_i = min(self.eta_cap, 1.0 / math.sqrt(self.cum_sq[i]))
            new_w.append(self.w[i] * math.exp(eta_i * r[i]))
        ssum = sum(new_w)
        self.w = ([wi/ssum for wi in new_w] if (ssum > 0 and math.isfinite(ssum))
                  else [1.0/self.n]*self.n)

class PaperBroker:
    """Taker-only paper broker with taker-side exits and summary stats."""
    def __init__(self, start_equity: float = 10_000.0):
        self.equity = start_equity
        self.cash   = start_equity
        self.pos_qty = 0.0
        self.entry_price = None
        self.entry_time_ms = None
        self.trade_id = 0
        self.max_dd = 0.0
        self.peak_equity = start_equity
        self.consec_losers = 0
        self.num_trades = 0; self.wins = 0; self.losses = 0
        self.pnl_hist: Deque[float] = deque(maxlen=1000)
        self.trades_out = open(TRADES_CSV, "w", newline="", encoding="utf-8")
        self.tw = csv.writer(self.trades_out)
        self.tw.writerow(["utc","trade_id","side","qty","entry_px","exit_px","pnl_usdt","pnl_bps","hold_sec"])

    def _taker_fee(self, notional: float) -> float: return notional * from_bps(TAKER_FEE_BPS)
    def position_open(self) -> bool: return self.pos_qty != 0.0

    def should_stop(self) -> bool:
        dd = 100.0 * (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        self.max_dd = max(self.max_dd, dd)
        return dd >= STOP_ON_DRAWDOWN_PCT or self.consec_losers >= MAX_CONSEC_LOSERS

    def open(self, side: str, price: float, equity_hint: float):
        if self.position_open(): return False
        notional = min(MAX_POSITION_NOTIONAL, max(500.0, 5.0 * equity_hint))
        qty = notional / max(price, 1e-9)
        if side == "SHORT": qty = -qty
        fee = self._taker_fee(abs(qty) * price)
        self.cash -= fee
        self.pos_qty = qty; self.entry_price = price; self.entry_time_ms = now_ms(); self.trade_id += 1
        if VERBOSE:
            print(f"[OPEN ] id={self.trade_id} side={side} qty={abs(qty):.4f} px={price:.2f} fee={fee:.2f}")
        return True

    def taker_exit_px(self, ms: "MarketState") -> float:
        if self.pos_qty > 0:  # close long -> sell at bid
            return round_down_to_tick(ms.eth_bid, TICK_SIZE)
        else:                 # close short -> buy at ask
            return round_up_to_tick(ms.eth_ask, TICK_SIZE)

    def maybe_close_by_ttl(self, ms: "MarketState"):
        if not self.position_open(): return
        if (now_ms() - self.entry_time_ms) / 1000.0 >= TRADE_TTL_SEC:
            self.close(self.taker_exit_px(ms))

    def close(self, price: float):
        if not self.position_open(): return
        side = "LONG" if self.pos_qty > 0 else "SHORT"
        qty, entry_px, exit_px = self.pos_qty, self.entry_price, price
        fee_exit = self._taker_fee(abs(qty) * exit_px)
        pnl_usdt = (exit_px - entry_px) * qty - fee_exit
        pnl_bps  = bps(safe_div(pnl_usdt, abs(qty) * entry_px, 0.0))
        self.cash += (entry_px * abs(qty)) + ((exit_px - entry_px) * qty)
        self.equity += pnl_usdt
        self.peak_equity = max(self.peak_equity, self.equity)
        self.consec_losers = self.consec_losers + 1 if pnl_usdt < 0 else 0
        hold_sec = (now_ms() - self.entry_time_ms) / 1000.0
        self.num_trades += 1; (self.wins if pnl_usdt >= 0 else self.losses).__iadd__(1)
        self.pnl_hist.append(pnl_usdt)
        self.tw.writerow([utc_ts(), self.trade_id, side, abs(qty),
                          f"{entry_px:.2f}", f"{exit_px:.2f}",
                          f"{pnl_usdt:.4f}", f"{pnl_bps:.2f}", f"{hold_sec:.1f}"])
        self.trades_out.flush()
        if VERBOSE:
            print(f"[CLOSE] id={self.trade_id} side={side} qty={abs(qty):.4f} "
                  f"entry={entry_px:.2f} exit={exit_px:.2f} pnl={pnl_usdt:.2f} ({pnl_bps:.1f} bps) "
                  f"hold={hold_sec:.1f}s")
        self.pos_qty = 0.0; self.entry_price = None; self.entry_time_ms = None

    def mark_to_market(self, ms: "MarketState"):
        if not self.position_open() or ms.eth_mid is None: return
        exit_px = self.taker_exit_px(ms)
        unreal = (exit_px - self.entry_price) * self.pos_qty
        self.peak_equity = max(self.peak_equity, self.equity + unreal)

    def open_upnl(self, ms: "MarketState") -> Tuple[float, float]:
        if not self.position_open() or ms.eth_mid is None: return 0.0, 0.0
        exit_px = self.taker_exit_px(ms)
        upnl = (exit_px - self.entry_price) * self.pos_qty
        upnl_bps = bps(safe_div(upnl, abs(self.pos_qty) * self.entry_price, 0.0))
        return upnl, upnl_bps

    def open_hold_sec(self) -> float:
        return 0.0 if not self.position_open() else (now_ms() - self.entry_time_ms) / 1000.0

    def close_all(self, ms: "MarketState"):
        if self.position_open() and ms.eth_mid is not None:
            self.close(self.taker_exit_px(ms))

    def shutdown(self):
        try: self.trades_out.close()
        except Exception: pass

class PredictorAndTrainer:
    def __init__(self, fb: FeatureBuilder, meta: CoinBettingMeta):
        self.fb, self.meta = fb, meta
        self.pending: Deque[Tuple[int, Dict[str,float], float]] = deque()
        self.pred_out = open(PRED_CSV, "w", newline="", encoding="utf-8")
        self.pw = csv.writer(self.pred_out)
        self.pw.writerow(["utc","mid","spd_ticks","trade_imb","mp_gap","ofi_l1","btc_ret_2s",
                          "rv","gate","p_dir","p_meta","w_ofi","w_mp","w_trade","w_btc"])

    def predict(self, ts_ms: int, feat: Dict[str, float], mid: float, gate: float) -> float:
        s = self.fb.experts_directional(feat)
        p_dir = self.meta.predict(s)           # [-1,1]
        p = clamp(gate * p_dir, -1.0, 1.0)     # gated prediction
        self.pending.append((ts_ms + int(TARGET_HORIZON_SEC * 1000), s, mid))
        self.pw.writerow([
            utc_ts(), f"{mid:.2f}",
            f"{feat.get('spd_ticks',0):.3f}",
            f"{feat.get('trade_imb',0):.4f}",
            f"{feat.get('mp_gap',0):.4f}",
            f"{feat.get('ofi_l1',0):.4f}",
            f"{feat.get('btc_ret_2s',0):.4f}",
            f"{feat.get('rv',0):.6f}",
            f"{gate:.3f}",
            f"{p_dir:.4f}",
            f"{p:.4f}",
            f"{self.meta.w[0]:.3f}", f"{self.meta.w[1]:.3f}",
            f"{self.meta.w[2]:.3f}", f"{self.meta.w[3]:.3f}",
        ])
        self.pred_out.flush()
        return p

    def mature(self, ts_ms: int, cur_mid: float):
        while self.pending and self.pending[0][0] <= ts_ms:
            _, s, entry_mid = self.pending.popleft()
            if not entry_mid or not cur_mid: continue
            ret = (cur_mid - entry_mid) / entry_mid
            realized_bps = bps(ret) - (TAKER_FEE_BPS + SLIPPAGE_BPS) * 0.25
            self.meta.update(s, realized_bps)

    def shutdown(self):
        try: self.pred_out.close()
        except Exception: pass

# =========================
# ---- MAIN APP LOOP ------
# =========================
async def run():
    print(f"Connecting to {WS_ENDPOINT}")
    ms = MarketState()
    fb = FeatureBuilder(ms)
    meta = CoinBettingMeta(["ofi","mp_follow","trade_momo","btc_lead"])
    pt = PredictorAndTrainer(fb, meta)
    broker = PaperBroker(start_equity=10_000.0)

    ticks_out = open(TICKS_CSV, "w", newline="", encoding="utf-8")
    tw = csv.writer(ticks_out)
    tw.writerow(["utc","eth_bid","eth_bsz","eth_ask","eth_asz","eth_mid","btc_mid","spread_ticks"])

    last_summary = time.time()
    def print_summary():
        nonlocal last_summary
        if time.time() - last_summary >= 5.0:
            last_summary = time.time()
            eq = broker.equity; dd = broker.max_dd
            trades_count = broker.num_trades; wins = broker.wins
            wr = (wins / trades_count * 100.0) if trades_count else 0.0
            avg = statistics.mean(broker.pnl_hist) if broker.pnl_hist else 0.0
            std = statistics.pstdev(broker.pnl_hist) if len(broker.pnl_hist) > 2 else 0.0
            sharpe = (avg / std * math.sqrt(60)) if std > 0 else 0.0
            pos = "FLAT" if not broker.position_open() else ("LONG" if broker.pos_qty > 0 else "SHORT")
            upnl, upnl_bps = broker.open_upnl(ms)
            hold = broker.open_hold_sec()
            print(f"[SUM ] eq={eq:.2f} dd={dd:.1f}% trades={trades_count} win%={wr:.1f} "
                  f"avgPnL={avg:.2f} std={std:.2f} sh~{sharpe:.2f} pos={pos} "
                  f"openPnL={upnl:.2f}({upnl_bps:.1f}bps) hold={hold:.1f}s "
                  f"w={','.join(f'{x:.2f}' for x in meta.w)}")

    async with websockets.connect(WS_ENDPOINT, ping_interval=20, ping_timeout=20) as ws:
        print("Connected. Streaming...")
        stop_flag = False
        def handle_sig(*_):
            nonlocal stop_flag
            stop_flag = True
            print("\n[CTRL-C] Stopping...")
        signal.signal(signal.SIGINT, handle_sig)
        try: signal.signal(signal.SIGTERM, handle_sig)
        except (AttributeError, ValueError): pass

        dbg_last = 0.0

        while not stop_flag:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print("[WARN] websocket timeout; sending ping")
                try: await ws.ping()
                except Exception as e: print(f"[ERR ] ping failed: {e}"); break
                continue
            except Exception as e:
                print(f"[ERR ] websocket recv: {e}"); break

            try:
                obj = json.loads(msg)
                stream = obj.get("stream","")
                data = obj.get("data",{})
                ts_ms = int(data.get("E") or data.get("T") or now_ms())
            except Exception:
                continue

            if stream.endswith("@bookTicker"):
                s = data.get("s","")
                bid = float(data.get("b","0") or 0.0)
                ask = float(data.get("a","0") or 0.0)
                bsz = float(data.get("B","0") or 0.0)
                asz = float(data.get("A","0") or 0.0)
                if s.upper() == "ETHUSDT": ms.on_book_ticker(True, bid, bsz, ask, asz, ts_ms)
                elif s.upper() == "BTCUSDT": ms.on_book_ticker(False, bid, bsz, ask, asz, ts_ms)
                if ms.eth_mid is not None:
                    tw.writerow([utc_ts(), f"{ms.eth_bid:.2f}", f"{ms.eth_bsz:.4f}",
                                 f"{ms.eth_ask:.2f}", f"{ms.eth_asz:.4f}",
                                 f"{ms.eth_mid:.2f}", f"{(ms.btc_mid if ms.btc_mid else 0.0):.2f}",
                                 f"{ms.get_spread_ticks():.2f}"])
                    ticks_out.flush()

            elif stream.endswith("@aggTrade"):
                price = float(data.get("p","0") or 0.0)
                qty   = float(data.get("q","0") or 0.0)
                is_buyer_maker = bool(data.get("m", False))
                tms = int(data.get("T") or ts_ms)
                ms.on_agg_trade(price, qty, is_buyer_maker, tms)

            # ------- 10Hz frame -------
            if (ms.eth_mid is not None) and fb.frame_ready(ts_ms):
                feat = fb.build(ts_ms)
                if not feat: continue

                # update matured outcomes
                pt.mature(ts_ms, ms.eth_mid)

                # gate + prediction
                gate = fb.tight_spread_gate(feat)          # [0,1]
                p = pt.predict(ts_ms, feat, ms.eth_mid, gate)  # [-1,1]
                rv_bps = max(bps(feat["rv"]), RV_BPS_FLOOR)

                # costs (bps)
                spread_bps  = bps(safe_div(ms.eth_ask - ms.eth_bid, ms.eth_mid, 0.0))
                fee_bps     = TAKER_FEE_BPS
                slip_bps    = SLIPPAGE_BPS
                lat_bps     = bps(LATENCY_BUFFER_TICKS * TICK_SIZE / ms.eth_mid)
                total_cost_bps = spread_bps + fee_bps + slip_bps + lat_bps

                # expected edge proxy
                exp_bps = abs(p) * rv_bps * GAIN

                # periodic debug
                if time.time() - dbg_last > 2.0:
                    dbg_last = time.time()
                    print(f"[DBG ] p={p:+.3f} gate={gate:.2f} spd_ticks={feat['spd_ticks']:.2f} "
                          f"rv_bps={rv_bps:.1f} exp_bps={exp_bps:.1f} cost_bps={total_cost_bps:.1f}")

                # trade logic
                if not broker.position_open():
                    if (feat["spd_ticks"] <= MAX_SPREAD_TICKS and
                        abs(p) >= EDGE_THRESH_P and
                        exp_bps >= total_cost_bps * COST_MULT):
                        side = "LONG" if p > 0 else "SHORT"
                        px = round_up_to_tick(ms.eth_ask, TICK_SIZE) if side=="LONG" else round_down_to_tick(ms.eth_bid, TICK_SIZE)
                        broker.open(side, px, broker.equity)
                else:
                    broker.maybe_close_by_ttl(ms)
                    if broker.position_open():
                        # optional flip if strong opposite edge
                        if abs(p) >= max(0.85, EDGE_THRESH_P + 0.25) and exp_bps >= 1.15*total_cost_bps:
                            want_side = "LONG" if p > 0 else "SHORT"
                            have_side = "LONG" if broker.pos_qty > 0 else "SHORT"
                            if want_side != have_side:
                                broker.close(broker.taker_exit_px(ms))
                                if feat["spd_ticks"] <= MAX_SPREAD_TICKS:
                                    px = round_up_to_tick(ms.eth_ask, TICK_SIZE) if want_side=="LONG" else round_down_to_tick(ms.eth_bid, TICK_SIZE)
                                    broker.open(want_side, px, broker.equity)

                broker.mark_to_market(ms)
                print_summary()

                if broker.should_stop():
                    print("[STOP] Risk guard triggered. Flattening and exiting.")
                    break

        print("Closing positions and files...")
        pt.mature(now_ms(), ms.eth_mid if ms.eth_mid else 0.0)
        broker.close_all(ms)
        broker.shutdown()
        pt.shutdown()
        ticks_out.close()
    print("Done.")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
