"""Call session lifecycle management: AI connection + audio transport."""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import structlog

from app.bridge.audio_adapter import AudioAdapter
from app.utils.constants import AudioConstants
from app.utils.conversation_mailer import extract_last_call, send_conversation_email
from app.config import config


class CallSession:
    """Manages call session lifecycle: AI connection + audio transport tasks.

    Coordinates AudioAdapter and AI client for a single call.
    """

    def __init__(
        self,
        audio_adapter: AudioAdapter,
        ai_client: any
    ) -> None:
        """Initialize call session.

        Args:
            audio_adapter: AudioAdapter instance
            ai_client: AI duplex client
        """
        self._media = audio_adapter
        self._ai = ai_client

        self._running = False
        self._task_group_task: Optional[asyncio.Task[None]] = None

        self._logger = structlog.get_logger(__name__)
        self._conversation_logger = logging.getLogger("conversation")

        self._call_id: Optional[str] = None
        self._caller: Optional[str] = None
        self._header_logged = False
        self._hangup_handler: Optional[Callable[[], Awaitable[None]]] = None
        self._hangup_requested = False

    def set_call_context(
        self,
        call_id: Optional[str],
        caller: Optional[str] = None
    ) -> None:
        """Store metadata for conversation logging."""

        self._call_id = call_id
        self._caller = caller

    def set_hangup_handler(
        self,
        handler: Callable[[], Awaitable[None]]
    ) -> None:
        """Register coroutine used to send SIP BYE when AI requests hangup."""

        self._hangup_handler = handler

    async def start(self) -> None:
        """Start the call session using asyncio.TaskGroup.

        This starts background tasks and returns immediately without blocking.
        Call stop() to terminate the session.
        """
        if self._running:
            return

        self._running = True

        self._log_conversation_header()

        try:
            # Connect AI client
            await self._ai.connect()
            self._logger.info("AI client connected")

            # Define session runner with TaskGroup
            async def _run_session() -> None:
                self._logger.info("🏁 Session runner starting with TaskGroup...")
                try:
                    async with asyncio.TaskGroup() as tg:
                        self._logger.info("📋 Starting uplink task...")
                        tg.create_task(
                            self._uplink_safe(),
                            name="session-uplink"
                        )

                        self._logger.info("📋 Starting AI receive task...")
                        tg.create_task(
                            self._ai_recv_safe(),
                            name="session-ai-recv"
                        )

                        self._logger.info("📋 Starting health task...")
                        tg.create_task(
                            self._health_safe(),
                            name="session-health"
                        )

                        self._logger.info("✅ All call session tasks started")
                    # TaskGroup exits when all tasks complete or on exception
                    self._logger.info("TaskGroup exited")

                except* asyncio.CancelledError:
                    # Normal cancellation during shutdown
                    self._logger.debug("Call session tasks cancelled (normal shutdown)")

                except* Exception as eg:
                    # Unexpected exceptions
                    self._logger.error(
                        "CallSession TaskGroup exceptions",
                        count=len(eg.exceptions)
                    )
                    for exc in eg.exceptions:
                        self._logger.error(
                            f"Exception: {type(exc).__name__}: {exc}",
                            exc_info=exc
                        )

                finally:
                    self._running = False
                    self._logger.info("Session runner finished")

            # Create background task and store reference
            self._task_group_task = asyncio.create_task(
                _run_session(),
                name="call-session-runner"
            )
            self._logger.info("Call session started - TaskGroup launched")

        except Exception as e:
            self._logger.error(f"Failed to start session: {e}", exc_info=True)
            self._running = False
            raise

    def _log_conversation_header(self) -> None:
        """Write a separator entry for each new call."""

        if self._header_logged:
            return

        call_id = self._call_id or getattr(self._ai, "call_id", None)
        caller = self._caller or getattr(self._ai, "caller", None)

        # Always include the placeholders to keep format consistent
        self._conversation_logger.info(
            "=== NEW CALL === call_id=%s from=%s ===",
            call_id or "unknown",
            caller or "unknown"
        )
        self._header_logged = True

    async def stop(self) -> None:
        """Stop the call session."""
        if not self._running:
            return

        self._logger.info("Stopping call session...")

        self._running = False

        # Cancel the TaskGroup background task
        if self._task_group_task:
            self._task_group_task.cancel()
            try:
                await self._task_group_task
            except asyncio.CancelledError:
                self._logger.info("TaskGroup task cancelled")
            except Exception as e:
                self._logger.error(f"Error during TaskGroup cancellation: {e}")

        # Disconnect AI
        try:
            await self._ai.close()
            self._logger.info("AI client closed")
        except Exception as e:
            self._logger.error(f"Error closing AI client: {e}")

        # Close audio adapter
        try:
            await self._media.close()
            self._logger.info("Audio adapter closed")
        except Exception as e:
            self._logger.error(f"Error closing audio adapter: {e}")

        self._logger.info("Call session stopped")

        self._send_conversation_email()

    def _send_conversation_email(self) -> None:
        """Extract last conversation from log and send via email."""

        log_path = config.system.conversation_log_path
        if not log_path:
            return

        try:
            conversation = extract_last_call(log_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("Failed to extract conversation log", error=str(exc))
            return

        if not conversation:
            return

        call_id = self._call_id or getattr(self._ai, "call_id", "unknown")
        caller = self._caller or getattr(self._ai, "caller", "unknown")
        subject = f"RenTalk conversation summary - call_id={call_id} from={caller}"

        try:
            send_conversation_email(subject, conversation)
            self._logger.info("Conversation email sent", call_id=call_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("Failed to send conversation email", error=str(exc))

    async def request_hangup(self) -> None:
        """Trigger call teardown after AI issues ENDCALL."""

        if self._hangup_requested:
            self._logger.info("Hangup already requested - ignoring duplicate signal")
            return

        self._hangup_requested = True
        self._logger.info("AI requested controlled hangup")

        if self._hangup_handler:
            try:
                await self._hangup_handler()
                self._logger.info("SIP BYE sent successfully")
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.error("Failed to send SIP BYE", error=str(exc))
        else:
            self._logger.warning("No hangup handler registered - cannot send BYE")

        try:
            await self.stop()
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("Error stopping session after hangup", error=str(exc))

    async def _uplink_safe(self) -> None:
        """Safe uplink with proper exception handling and cleanup."""
        frames_processed = 0
        self._logger.info("🚀 Uplink task STARTED")
        try:
            while self._running:
                try:
                    # Timeout per frame to prevent hang
                    async with asyncio.timeout(0.05):
                        frame = await self._media.get_uplink_audio()
                        await self._ai.send_pcm16_8k(frame)

                    frames_processed += 1
                    if frames_processed == 1:
                        self._logger.info("🔊 First frame sent to AI!")

                    if frames_processed % AudioConstants.LOG_INTERVAL_FRAMES == 0:
                        self._logger.info(
                            "🔊 Uplink processed frames",
                            count=frames_processed,
                            direction="SIP → AI"
                        )

                except TimeoutError:
                    # No data available, continue
                    await asyncio.sleep(0.01)
                except Exception as e:
                    self._logger.error(f"Uplink frame error: {e}", exc_info=True)
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            self._logger.info(f"Uplink task cancelled after {frames_processed} frames")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"Uplink task fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 Uplink task STOPPED (processed {frames_processed} frames)")

    async def _ai_recv_safe(self) -> None:
        """Safe AI receive with proper exception handling and cleanup."""
        chunks_received = 0
        self._logger.info("🎧 AI receive task STARTED")
        try:
            async for chunk in self._ai.receive_chunks():
                if not self._running:
                    break

                # Direct passthrough to downlink stream
                await self._media.feed_ai_audio(chunk)

                chunks_received += 1
                if chunks_received % AudioConstants.LOG_INTERVAL_FRAMES == 0:
                    self._logger.info("📢 Received chunks from AI", count=chunks_received)

        except asyncio.CancelledError:
            self._logger.info(f"AI receive task cancelled after {chunks_received} chunks")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"AI receive fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 AI receive task STOPPED (received {chunks_received} chunks)")

    async def _health_safe(self) -> None:
        """Safe health monitoring with proper exception handling."""
        reconnect_attempts = 0
        max_attempts = 3
        health_checks = 0

        self._logger.info("🏥 Health task STARTED")
        try:
            while self._running:
                try:
                    await asyncio.sleep(30)  # Health check interval
                    health_checks += 1

                    # Check AI connection
                    if not await self._ai.ping():
                        self._logger.warning(f"AI connection unhealthy (check #{health_checks})")

                        if reconnect_attempts < max_attempts:
                            self._logger.info(f"Attempting reconnect ({reconnect_attempts + 1}/{max_attempts})")
                            await self._ai.reconnect()
                            reconnect_attempts += 1
                        else:
                            self._logger.error("Max reconnection attempts reached, stopping session")
                            await self.stop()
                            break
                    else:
                        reconnect_attempts = 0  # Reset on success

                except asyncio.CancelledError:
                    raise  # Propagate cancellation
                except Exception as e:
                    self._logger.error(f"Health check error: {e}", exc_info=True)
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            self._logger.info(f"Health task cancelled after {health_checks} checks")
            raise  # Propagate cancellation
        except Exception as e:
            self._logger.error(f"Health task fatal error: {e}", exc_info=True)
        finally:
            self._logger.info(f"🛑 Health task STOPPED (performed {health_checks} checks)")
