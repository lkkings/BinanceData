"""PubSub 单元测试"""
import asyncio

import pytest

from src.core.pubsub import PubSub, Subscription

pytestmark = pytest.mark.asyncio


async def test_subscribe_and_publish():
    pubsub = PubSub(queue_maxsize=10)
    sub = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})
    assert pubsub.subscriber_count == 1

    await pubsub.publish("minute", "btcusdt", {"x": 1})
    msg = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
    assert msg.channel == "minute"
    assert msg.symbol == "btcusdt"
    assert msg.payload == {"x": 1}


async def test_filter_by_symbol():
    pubsub = PubSub()
    sub = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})
    await pubsub.publish("minute", "ethusdt", {"v": 1})
    await pubsub.publish("minute", "btcusdt", {"v": 2})
    msg = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
    assert msg.payload == {"v": 2}
    assert sub.queue.empty()


async def test_filter_by_channel():
    pubsub = PubSub()
    sub = pubsub.subscribe(channels={"minute"}, symbols=None)
    await pubsub.publish("trade", "btcusdt", {})
    await pubsub.publish("minute", "btcusdt", {"v": 1})
    msg = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
    assert msg.channel == "minute"
    assert sub.queue.empty()


async def test_empty_symbols_means_all():
    pubsub = PubSub()
    sub = pubsub.subscribe(channels={"minute"}, symbols=None)
    await pubsub.publish("minute", "btcusdt", {})
    await pubsub.publish("minute", "ethusdt", {})
    assert sub.queue.qsize() == 2


async def test_backpressure_drops_oldest():
    pubsub = PubSub(queue_maxsize=3)
    sub = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})

    for i in range(5):
        await pubsub.publish("minute", "btcusdt", {"i": i})

    # 队列最多 3 条；多出的两条触发"丢最旧"
    assert sub.queue.qsize() == 3
    assert sub.dropped == 2

    drained = []
    while not sub.queue.empty():
        drained.append((await sub.queue.get()).payload["i"])
    # 最早的 0、1 被丢弃，剩下 2、3、4
    assert drained == [2, 3, 4]


async def test_unsubscribe_stops_delivery():
    pubsub = PubSub()
    sub = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})
    pubsub.unsubscribe(sub)
    await pubsub.publish("minute", "btcusdt", {})
    assert pubsub.subscriber_count == 0
    assert sub.queue.empty()


async def test_update_subscription_add_remove():
    pubsub = PubSub()
    sub = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})

    pubsub.update_subscription(sub, add_symbols={"ethusdt"})
    await pubsub.publish("minute", "ethusdt", {"v": 1})
    msg = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
    assert msg.symbol == "ethusdt"

    pubsub.update_subscription(sub, remove_symbols={"btcusdt"})
    await pubsub.publish("minute", "btcusdt", {})
    await asyncio.sleep(0.05)
    assert sub.queue.empty()


async def test_multiple_subs_isolated():
    pubsub = PubSub()
    a = pubsub.subscribe(channels={"minute"}, symbols={"btcusdt"})
    b = pubsub.subscribe(channels={"minute"}, symbols={"ethusdt"})
    await pubsub.publish("minute", "btcusdt", {"x": 1})
    await pubsub.publish("minute", "ethusdt", {"x": 2})
    assert a.queue.qsize() == 1
    assert b.queue.qsize() == 1
    assert (await a.queue.get()).symbol == "btcusdt"
    assert (await b.queue.get()).symbol == "ethusdt"
