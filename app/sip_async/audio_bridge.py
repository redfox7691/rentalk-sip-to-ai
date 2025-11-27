"""Audio bridge between RTP session and AudioAdapter.

Bridges RTP audio (G.711) with AudioAdapter (PCM16) using TaskGroup.
"""

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.bridge.audio_adapter import AudioAdapter
    from app.sip_async.rtp_session import RTPSession

logger = structlog.get_logger(__name__)


class RTPAudioBridge:
    """Bridge RTPSession and AudioAdapter using TaskGroup.

    Data flow:
    - Uplink: RTP → decode G.711 → PCM16 → AudioAdapter → AI
    - Downlink: AI → AudioAdapter → PCM16 → encode G.711 → RTP
    """

    def __init__(self, rtp_session: 'RTPSession', audio_adapter: 'AudioAdapter'):
        """Initialize audio bridge.

        Args:
            rtp_session: RTP session for network audio
            audio_adapter: Audio adapter for AI integration
        """
        self.rtp = rtp_session
        self.adapter = audio_adapter
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Statistics
        self._uplink_frames = 0
        self._downlink_frames = 0

    async def run(self) -> None:
        """Run bidirectional audio bridge with TaskGroup."""
        self._running = True
        self._tasks.clear()

        logger.info("AudioBridge starting")

        try:
            async with asyncio.TaskGroup() as tg:
                # Uplink: RTP → AudioAdapter
                uplink = tg.create_task(
                    self._uplink_task(),
                    name="audiobridge-uplink"
                )
                self._tasks.append(uplink)

                # Downlink: AudioAdapter → RTP
                downlink = tg.create_task(
                    self._downlink_task(),
                    name="audiobridge-downlink"
                )
                self._tasks.append(downlink)

                logger.info("AudioBridge TaskGroup started")

        except* asyncio.CancelledError:
            # Normal cancellation during shutdown
            logger.debug("AudioBridge tasks cancelled (normal shutdown)")

        except* Exception as eg:
            # Unexpected exceptions
            logger.error(
                "AudioBridge TaskGroup exceptions",
                count=len(eg.exceptions)
            )
            for exc in eg.exceptions:
                logger.error(
                    f"Exception: {type(exc).__name__}: {exc}",
                    exc_info=exc
                )
        finally:
            self._running = False
            logger.info(
                "AudioBridge stopped",
                uplink_frames=self._uplink_frames,
                downlink_frames=self._downlink_frames
            )

    async def _uplink_task(self) -> None:
        """Uplink: RTP → AudioAdapter → AI.

        Reads PCM16 audio from RTP and feeds to AudioAdapter.

        Note:
            Runs continuously until cancelled. The async iterator
            will be cancelled when the task is cancelled.
        """
        logger.info("AudioBridge uplink task started")

        try:
            async for pcm_data in self.rtp.receive_audio():
                # Feed PCM16 @ 8kHz to AudioAdapter
                # AudioAdapter expects 320 bytes (160 samples * 2 bytes)
                self.adapter.on_rx_pcm16_8k(pcm_data)

                self._uplink_frames += 1

                # Periodic logging
                if self._uplink_frames % 500 == 0:
                    logger.debug(
                        "AudioBridge uplink stats",
                        frames=self._uplink_frames,
                        direction="RTP → AudioAdapter → AI"
                    )

        except asyncio.CancelledError:
            logger.info("AudioBridge uplink cancelled")
            raise
        except Exception as e:
            logger.error("AudioBridge uplink error", error=str(e), exc_info=True)
            raise
        finally:
            logger.info(
                "AudioBridge uplink stopped",
                frames_processed=self._uplink_frames
            )

    async def _downlink_task(self) -> None:
        """Downlink: AI → AudioAdapter → RTP.

        Gets PCM16 audio from AudioAdapter and sends to RTP.

        Note:
            Runs continuously until cancelled. Does not check _running flag
            to avoid race conditions during startup.
        """
        logger.info("AudioBridge downlink task started")

        try:
            while True:
                # Get PCM16 audio from AudioAdapter downlink (AI → SIP)
                # This blocks until data is available
                pcm_data = await self.adapter.get_downlink_audio()

                # Send to RTP
                await self.rtp.send_audio(pcm_data)

                self._downlink_frames += 1

                # Periodic logging
                if self._downlink_frames % 500 == 0:
                    logger.debug(
                        "AudioBridge downlink stats",
                        frames=self._downlink_frames,
                        direction="AI → AudioAdapter → RTP"
                    )

        except asyncio.CancelledError:
            logger.info("AudioBridge downlink cancelled")
            raise
        except Exception as e:
            logger.error("AudioBridge downlink error", error=str(e), exc_info=True)
            raise
        finally:
            logger.info(
                "AudioBridge downlink stopped",
                frames_sent=self._downlink_frames
            )

    async def stop(self) -> None:
        """Stop the audio bridge."""
        logger.info(
            "AudioBridge stop requested",
            pending_downlink=self.adapter.downlink_size(),
            pending_tx=self.rtp.tx_queue.qsize()
        )

        # Flush any partial greeting so the caller hears the full message
        await self.adapter.flush_pending_downlink()

        # Allow in-flight audio to be delivered before tearing down transport
        await self._drain_queues()

        self._running = False

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

        logger.info("AudioBridge stop completed")

    async def _drain_queues(self, timeout: float = 0.5) -> None:
        """Wait briefly for downlink and RTP queues to empty."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while loop.time() < deadline:
            downlink_pending = self.adapter.downlink_size()
            tx_pending = self.rtp.tx_queue.qsize()

            if downlink_pending == 0 and tx_pending == 0:
                break

            await asyncio.sleep(0.01)
