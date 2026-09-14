"""Robot-to-robot communication interface + an MQTT (paho) implementation.

Every published payload carries robot_id, timestamp, message_type and
payload (per the spec) as JSON today; serialize()/deserialize() are the
single seam to swap in Protocol Buffers later without touching publish()/
subscribe() call sites throughout the swarm/formation layers.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt

    _HAS_PAHO = True
except ImportError:  # pragma: no cover
    _HAS_PAHO = False


def serialize(robot_id: str, message_type: str, payload: dict) -> bytes:
    envelope = {
        "robot_id": robot_id,
        "timestamp": time.time(),
        "message_type": message_type,
        "payload": payload,
    }
    return json.dumps(envelope).encode("utf-8")


def deserialize(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


class CommunicationClient(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def publish(self, topic: str, message_type: str, payload: dict, qos: int = 1) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[str, dict], None], qos: int = 1) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...


class MQTTCommunicationClient(CommunicationClient):
    def __init__(self, robot_id: str, broker_host: str, broker_port: int = 1883, keepalive_s: int = 30):
        self.robot_id = robot_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.keepalive_s = keepalive_s
        self._client: Any = None
        self._callbacks: dict[str, Callable[[str, dict], None]] = {}
        self._connected = False

    def connect(self) -> None:
        if not _HAS_PAHO:
            logger.warning("paho-mqtt not installed; MQTTCommunicationClient running in "
                           "no-op mode. Install paho-mqtt to enable Phase 5 swarm comms.")
            return

        self._client = mqtt.Client(client_id=f"sentinelswarm-{self.robot_id}")
        self._client.on_message = self._on_message
        try:
            self._client.connect(self.broker_host, self.broker_port, self.keepalive_s)
            self._client.loop_start()
            self._connected = True
            logger.info("Connected to MQTT broker %s:%d as %s", self.broker_host, self.broker_port, self.robot_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("MQTT connect failed (%s); running degraded (no swarm comms).", exc)
            self._connected = False

    def publish(self, topic: str, message_type: str, payload: dict, qos: int = 1) -> None:
        if not self._connected or self._client is None:
            return
        try:
            self._client.publish(topic, serialize(self.robot_id, message_type, payload), qos=qos)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MQTT publish to %s failed: %s", topic, exc)

    def subscribe(self, topic: str, callback: Callable[[str, dict], None], qos: int = 1) -> None:
        self._callbacks[topic] = callback
        if self._connected and self._client is not None:
            self._client.subscribe(topic, qos=qos)

    def _on_message(self, _client, _userdata, msg) -> None:
        callback = self._callbacks.get(msg.topic)
        if callback is None:
            return
        try:
            envelope = deserialize(msg.payload)
            callback(msg.topic, envelope)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to handle MQTT message on %s: %s", msg.topic, exc)

    def disconnect(self) -> None:
        if self._client is not None and self._connected:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False
