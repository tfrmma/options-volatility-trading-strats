# Deribit WebSocket feed. asyncio + aiohttp.
# Don't use requests in an async context. You know who you are.
#
# Deribit has the better options liquidity for BTC/ETH — primary feed.
# Mark IV from Deribit is what you'll get filled at so use their greeks too.

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import aiohttp

logger = logging.getLogger(__name__)

_DERIBIT_PROD = "wss://www.deribit.com/ws/api/v2"
_DERIBIT_TEST = "wss://test.deribit.com/ws/api/v2"


@dataclass
class TickerData:
    symbol: str
    timestamp: float
    bid: float
    ask: float
    mark_price: float
    index_price: float
    iv: float            # mark IV — Deribit sends as %, we convert to decimal
    delta: float
    gamma: float
    vega: float
    theta: float
    open_interest: float
    volume_24h: float


@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: float
    bids: list[tuple[float, float]]   # (price, size), best first
    asks: list[tuple[float, float]]
    change_id: Optional[int] = None

    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return 0.5 * (b + a) if b and a else None


class DeribitFeed:

    def __init__(
        self,
        on_ticker:    Optional[Callable[[TickerData],        Awaitable[None]]] = None,
        on_orderbook: Optional[Callable[[OrderBookSnapshot], Awaitable[None]]] = None,
        testnet: bool = True,
    ):
        self.on_ticker    = on_ticker
        self.on_orderbook = on_orderbook
        self._url = _DERIBIT_TEST if testnet else _DERIBIT_PROD
        self._ws:      Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self._url, heartbeat=30.0)
        logger.info("connected", extra={"url": self._url})

    async def subscribe_ticker(self, instrument: str) -> None:
        # e.g. 'BTC-25JAN25-50000-C'
        await self._subscribe([f"ticker.{instrument}.100ms"])

    async def subscribe_orderbook(self, instrument: str, depth: int = 10) -> None:
        await self._subscribe([f"book.{instrument}.{depth}.100ms"])

    async def subscribe_index(self, currency: str = "btc_usd") -> None:
        await self._subscribe([f"deribit_price_index.{currency}"])

    async def _subscribe(self, channels: list[str]) -> None:
        await self._ws.send_str(json.dumps({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "public/subscribe",
            "params": {"channels": channels},
        }))

    async def listen(self) -> None:
        # TODO: add reconnect loop with exponential backoff
        # right now a dropped connection kills the process silently
        if not self._ws:
            raise RuntimeError("not connected")

        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._dispatch(json.loads(msg.data))
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                logger.warning("ws disconnected", extra={"type": str(msg.type)})
                break

    async def _dispatch(self, msg: dict) -> None:
        if msg.get("method") != "subscription":
            return

        channel = msg.get("params", {}).get("channel", "")
        data    = msg.get("params", {}).get("data", {})

        if channel.startswith("ticker."):
            ticker = self._parse_ticker(data, channel.split(".")[1])
            if ticker and self.on_ticker:
                await self.on_ticker(ticker)
        elif channel.startswith("book."):
            ob = self._parse_orderbook(data, channel.split(".")[1])
            if ob and self.on_orderbook:
                await self.on_orderbook(ob)

    @staticmethod
    def _parse_ticker(data: dict, instrument: str) -> Optional[TickerData]:
        try:
            g = data.get("greeks", {})
            return TickerData(
                symbol=instrument,
                timestamp=data["timestamp"] / 1000.0,
                bid=data.get("best_bid_price", 0.0),
                ask=data.get("best_ask_price", 0.0),
                mark_price=data.get("mark_price", 0.0),
                index_price=data.get("index_price", 0.0),
                iv=data.get("mark_iv", 0.0) / 100.0,   # % -> decimal
                delta=g.get("delta", 0.0),
                gamma=g.get("gamma", 0.0),
                vega=g.get("vega", 0.0),
                theta=g.get("theta", 0.0),
                open_interest=data.get("open_interest", 0.0),
                volume_24h=data.get("stats", {}).get("volume", 0.0),
            )
        except (KeyError, TypeError) as e:
            logger.warning(f"ticker parse error: {e}")
            return None

    @staticmethod
    def _parse_orderbook(data: dict, instrument: str) -> Optional[OrderBookSnapshot]:
        try:
            def parse_side(raw):
                return [(float(p), float(s)) for _, p, s in raw]
            return OrderBookSnapshot(
                symbol=instrument,
                timestamp=data["timestamp"] / 1000.0,
                bids=parse_side(data.get("bids", [])),
                asks=parse_side(data.get("asks", [])),
                change_id=data.get("change_id"),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"orderbook parse error: {e}")
            return None

    async def disconnect(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("disconnected")
