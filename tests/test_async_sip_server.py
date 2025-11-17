"""Regression tests for AsyncSIPServer call lifecycle management."""

import asyncio

from app.sip_async.async_call import AsyncCall
from app.sip_async.async_sip_server import AsyncSIPServer
from app.sip_async.sip_protocol import SIPMessage, SIPMessageType, SIPMethod


class DummyLifecycleComponent:
    """Simple component that immediately completes run/stop."""

    async def run(self) -> None:
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await asyncio.sleep(0)

    def update_port(self, _new_port: int) -> None:  # pragma: no cover - not used in test
        """Mimic RTPSession API for retries."""
        return


class DummyCallSession:
    """Minimal call session that starts and stops immediately."""

    def __init__(self) -> None:
        self._hangup_handler = None

    def set_hangup_handler(self, handler) -> None:
        self._hangup_handler = handler

    async def start(self) -> None:
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await asyncio.sleep(0)


def _build_invite(call_id: str) -> SIPMessage:
    """Create a minimal INVITE message with SDP."""

    sdp = "\r\n".join(
        [
            "v=0",
            "o=- 0 0 IN IP4 127.0.0.1",
            "s=Test",
            "c=IN IP4 127.0.0.1",
            "t=0 0",
            "m=audio 4000 RTP/AVP 0",
        ]
    )

    return SIPMessage(
        message_type=SIPMessageType.REQUEST,
        method=SIPMethod.INVITE,
        headers={
            "Via": [
                {
                    "type": "SIP/2.0/UDP",
                    "address": ("127.0.0.1", "5060"),
                    "branch": f"z9h{call_id}",
                }
            ],
            "From": {"address": "caller@example.com", "tag": f"from-{call_id}"},
            "To": {"address": "callee@example.com"},
            "Call-ID": call_id,
            "CSeq": {"number": 1, "method": "INVITE"},
        },
        body=sdp,
        remote_addr=("127.0.0.1", 5060),
    )


async def _create_mock_call(server: AsyncSIPServer, call_id: str) -> AsyncCall:
    """Allocate a port and create an AsyncCall with dummy components."""

    invite = _build_invite(call_id)
    local_port = await server.allocate_rtp_port()
    call = AsyncCall(invite, server, server.host, local_port)
    call.rtp_session = DummyLifecycleComponent()
    call.audio_bridge = DummyLifecycleComponent()
    call.call_session = DummyCallSession()
    call.call_session.set_hangup_handler(call.hangup)
    return call


async def _cleanup_calls(server: AsyncSIPServer) -> None:
    """Mirror AsyncSIPServer cleanup logic for tests."""

    async with server._calls_lock:
        ended_ids = [
            call_id
            for call_id, call in list(server.active_calls.items())
            if call.has_completed
        ]
        completed_calls = [server.active_calls.pop(call_id) for call_id in ended_ids]

    for call in completed_calls:
        await server.release_rtp_port(call.local_rtp_port)


def test_sequential_calls_are_not_dropped() -> None:
    """Ensure a new call survives cleanup while it is still starting up."""

    asyncio.run(_exercise_sequential_calls())


async def _exercise_sequential_calls() -> None:
    server = AsyncSIPServer(host="127.0.0.1", port=5070)

    sent_responses: list[str] = []

    async def fake_send_message(data: bytes, _addr: tuple) -> None:
        sent_responses.append(data.decode("utf-8"))

    server.send_message = fake_send_message  # type: ignore[assignment]

    # First call runs to completion and gets cleaned up.
    first_call = await _create_mock_call(server, "call-1")
    async with server._calls_lock:
        server.active_calls[first_call.call_id] = first_call

    await first_call.accept()
    await first_call.run()
    await _cleanup_calls(server)

    assert first_call.call_id not in server.active_calls

    # Second call is accepted while cleanup runs concurrently.
    second_call = await _create_mock_call(server, "call-2")
    async with server._calls_lock:
        server.active_calls[second_call.call_id] = second_call

    await second_call.accept()

    # Cleanup should not remove the new call until run() finishes.
    await _cleanup_calls(server)
    assert second_call.call_id in server.active_calls

    await second_call.run()
    await _cleanup_calls(server)

    assert second_call.call_id not in server.active_calls
    assert len(sent_responses) == 2
    assert any("Call-ID: call-2" in response for response in sent_responses)
