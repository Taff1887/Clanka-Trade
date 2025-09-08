# hl_ws.py
import time, threading
from collections import deque
from typing import Callable, Dict, Any, Deque, Optional, List, Tuple

# Try the official SDK; keep classes defined even if SDK is missing
SDK_OK = True
_SDK_ERR: Optional[Exception] = None
try:
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
except Exception as e:
    SDK_OK = False
    _SDK_ERR = e


class HLMeta:
    """
    Tick/size precision helper from exchange metadata.
    tick size = 10^(-pxDecimals), qty step = 10^(-szDecimals)
    """
    def __init__(self, api_url: Optional[str] = None):
        if not SDK_OK:
            raise RuntimeError(f"Install hyperliquid-python-sdk: {_SDK_ERR}")
        self.api_url = api_url or constants.TESTNET_API_URL
        self._meta = None

    def load(self):
        info = Info(self.api_url, skip_ws=True)
        self._meta = info.meta()
        return self._meta

    def get_decimals(self, coin: str = "ETH") -> Tuple[int, int]:
        """
        Return (px_decimals, sz_decimals). Be robust to schema differences:
        - Try multiple common key names.
        - If missing, infer from L2 book values.
        """
        if self._meta is None:
            self.load()

        # 1) Find the asset entry across a few common containers
        def _iter_assets(meta):
            for k in ("universe", "perpMeta", "universePerp", "assets"):
                if k in meta and isinstance(meta[k], list):
                    for a in meta[k]:
                        yield a

        entry = None
        for a in _iter_assets(self._meta):
            name = (a.get("name") or a.get("coin") or a.get("symbol") or "").upper()
            if name == coin.upper():
                entry = a
                break

        # 2) Try direct keys first
        def _pick(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        if entry is not None:
            pxd = _pick(entry, "pxDecimals", "priceDecimals", "pxDecimal")
            szd = _pick(entry, "szDecimals", "sizeDecimals", "szDecimal")
            if pxd is not None and szd is not None:
                return int(pxd), int(szd)

        # 3) Fallback: infer from live L2 book (string precision)
        info = Info(self.api_url, skip_ws=True)
        bids, asks = info.l2_book(coin)

        def _count_decimals(x):
            s = str(x)
            return len(s.split(".", 1)[1]) if "." in s else 0

        # price decimals: use the string precision we see on top-of-book
        px_dec = _count_decimals(bids[0]["px"] if bids else asks[0]["px"])
        # size decimals: use top bid size precision if present; else default 3 (0.001)
        if bids and "sz" in bids[0]:
            sz_dec = _count_decimals(bids[0]["sz"])
        elif asks and "sz" in asks[0]:
            sz_dec = _count_decimals(asks[0]["sz"])
        else:
            sz_dec = 3

        return int(px_dec), int(sz_dec)


class HLFeed:
    """
    Websocket-driven feed. Emits frames at fixed cadence for a simple callback API.
    Frame: {
      'ts','mid','best_bid','best_ask',
      'bids': [(px,sz,n)*5], 'asks': [(px,sz,n)*5],
      'trades': [(ts,px,sz,side)*N]
    }
    """
    def __init__(self, coin: str = "ETH", api_url: Optional[str] = None, poll_ms: int = 200):
        if not SDK_OK:
            raise RuntimeError(f"Install hyperliquid-python-sdk: {_SDK_ERR}")
        self.coin = coin
        self.api_url = api_url or constants.TESTNET_API_URL
        self.poll_ms = poll_ms

        self._bb: Optional[float] = None
        self._ba: Optional[float] = None
        self._bids: List[Tuple[float, float, int]] = []
        self._asks: List[Tuple[float, float, int]] = []
        self._trades: Deque[Tuple[float, float, float, int]] = deque(maxlen=4096)  # (ts, px, sz, side)

        self._subs: Deque[Callable[[Dict[str, Any]], None]] = deque()
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def subscribe(self, on_frame: Callable[[Dict[str, Any]], None]):
        self._subs.append(on_frame)

    def _emit(self, frame: Dict[str, Any]):
        for fn in list(self._subs):
            fn(frame)

    def _ws_loop(self):
        info = Info(self.api_url)  # WS enabled

        def on_bbo(msg):
            try:
                self._bb = float(msg["data"]["bbo"]["bidPx"])
                self._ba = float(msg["data"]["bbo"]["askPx"])
            except Exception:
                pass

        def on_l2(msg):
            try:
                bids = msg["data"].get("bids", [])
                asks = msg["data"].get("asks", [])
                self._bids = [(float(x["px"]), float(x["sz"]), int(x.get("n", 1))) for x in bids[:5]]
                self._asks = [(float(x["px"]), float(x["sz"]), int(x.get("n", 1))) for x in asks[:5]]
            except Exception:
                pass

        def on_trades(msg):
            try:
                now = time.time()
                for t in msg["data"]["trades"]:
                    px = float(t["px"]); sz = float(t["sz"])
                    side = 1 if t.get("side", "B") == "B" else -1
                    self._trades.append((now, px, sz, side))
            except Exception:
                pass

        info.subscribe({"type": "bbo",    "coin": self.coin}, on_bbo)
        info.subscribe({"type": "l2Book", "coin": self.coin}, on_l2)
        info.subscribe({"type": "trades", "coin": self.coin}, on_trades)

        while not self._stop:
            if self._bb is not None and self._ba is not None:
                mid = 0.5 * (self._bb + self._ba)
                frame = {
                    "ts": time.time(),
                    "mid": mid,
                    "best_bid": self._bb,
                    "best_ask": self._ba,
                    "bids": self._bids or [],
                    "asks": self._asks or [],
                    "trades": list(self._trades),
                }
                self._emit(frame)
                self._trades.clear()
            time.sleep(self.poll_ms / 1000.0)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2.0)
