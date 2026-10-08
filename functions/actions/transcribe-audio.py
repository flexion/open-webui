"""
title: Audio Transcription Action
author: ChatGPT Assistant
version: 0.1.0
license: MIT

This Action Function adds an interactive button to Open WebUI that allows users
to transcribe uploaded audio files directly inside the chat interface.  When
triggered, the function searches the current message for the first attached
audio file (MP3, WAV, FLAC, M4A, etc.), runs OpenAI’s Whisper model locally
to convert the speech to text, and returns the transcript as the assistant’s
message.

The function uses the same parameter configuration (Valves) found in the tool
version: you can adjust the Whisper model size, specify a language hint,
choose between transcription and translation, and tweak the beam size for
decoding.  During execution the function emits real‑time status updates to
inform the user of progress and handles missing dependencies by attempting to
install the Whisper library automatically.
"""

import json
import os
import subprocess
import sys
import asyncio
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field  # type: ignore


class EventEmitter:
    """Helper for sending progress events back to the frontend."""

    def __init__(self, event_emitter: Optional[Callable[[dict], Any]] = None) -> None:
        self.event_emitter = event_emitter

    async def emit(
        self,
        description: str,
        status: str = "in_progress",
        done: bool = False,
    ) -> None:
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Action:
    """An Open WebUI Action Function for audio transcription using Whisper.

    This class adheres to the Action Function structure described in the
    Open WebUI documentation【494419777235562†L100-L116】.  It exposes a single
    asynchronous ``action`` method which is invoked when the user clicks the
    associated button in the chat toolbar.  The function expects the chat
    message to contain an uploaded audio file; it extracts the file path,
    transcribes the audio to text using Whisper, and returns the transcript as
    the assistant’s reply.
    """

    class Valves(BaseModel):
        """Configuration parameters for the Action Function.

        These values are exposed in the Open WebUI settings for this Action so
        administrators can fine‑tune the transcription behaviour.  See the
        documentation for descriptions of each field.
        """

        MODEL: str = Field(
            default="base",
            description=(
                "Size of the Whisper model to load. Options include tiny, base, "
                "small, medium and large. Larger models are more accurate but "
                "consume more memory and compute."
            ),
        )
        LANGUAGE: str = Field(
            default="",
            description=(
                "ISO‑639‑1 code for the language spoken in the audio (e.g. 'en' for "
                "English, 'es' for Spanish). Leave blank to enable automatic "
                "language detection."
            ),
        )
        TASK: str = Field(
            default="transcribe",
            description=(
                "Task to perform: 'transcribe' will convert speech to the same "
                "language as the audio. 'translate' will translate the speech "
                "into English."
            ),
        )
        BEAM_SIZE: int = Field(
            default=5,
            ge=1,
            le=10,
            description=(
                "Beam size used during decoding. Higher values may improve accuracy "
                "slightly at the cost of speed."
            ),
        )

        def to_kwargs(self) -> dict:
            """Convert valves into keyword arguments for whisper.transcribe."""
            kwargs: dict = {
                "task": self.TASK,
                "beam_size": self.BEAM_SIZE,
            }
            if self.LANGUAGE:
                kwargs["language"] = self.LANGUAGE
            return kwargs

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _ensure_whisper_available(self, emitter: EventEmitter) -> Optional[Any]:
        """Ensure the whisper library is installed and importable.

        Attempts to import the ``whisper`` package.  If it's not available,
        automatically installs ``openai-whisper`` via pip.  Any errors during
        installation are reported back to the user via the event emitter.  On
        success, returns the imported module; otherwise ``None``.
        """
        try:
            import whisper  # type: ignore

            return whisper
        except ModuleNotFoundError:
            await emitter.emit(
                description=(
                    "Whisper library not found. Installing openai‑whisper package – "
                    "this may take a while."
                ),
                status="in_progress",
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "openai-whisper==20230918",
                    ],
                    check=True,
                )
                import whisper  # type: ignore  # type: ignore[redefined]

                return whisper
            except Exception as exc:
                await emitter.emit(
                    description=f"Failed to install Whisper: {exc}",
                    status="error",
                    done=True,
                )
                return None

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Handle the Action invocation and return the transcription result.

        Parameters
        ----------
        body: dict
            A dictionary containing message context, including uploaded files.
            The function looks for an entry in ``body['files']`` whose ``type``
            starts with ``'audio'``.  The file must include either a ``path`` or
            ``url`` key pointing to its location on disk.  When no audio file
            is provided, the action returns an instructional message.

        Returns
        -------
        dict
            A dictionary with at least a ``content`` field containing the
            transcript or an error message.  Additional fields like ``files``
            may be included in future enhancements.
        """
        emitter = EventEmitter(__event_emitter__)

        # Locate the first attached audio file
        audio_file: Optional[dict] = None
        for f in body.get("files", []):
            # Accept any file whose type starts with 'audio'
            if str(f.get("type", "")).lower().startswith("audio"):
                audio_file = f
                break

        if not audio_file:
            # No audio file present; instruct the user
            return {"content": "Please attach an audio file to transcribe."}

        # Determine the file path.  Open WebUI typically stores the uploaded
        # file on the server and exposes its local path via the 'path' key.  If
        # the path is not available, fall back to 'url'.
        file_path = audio_file.get("path") or audio_file.get("url") or ""
        if not file_path or not os.path.isfile(file_path):
            return {
                "content": "The attached audio file could not be located on the server."
            }

        # Emit a starting status
        await emitter.emit(f"Starting transcription for: {file_path}")

        # Ensure whisper is available
        whisper = await self._ensure_whisper_available(emitter)
        if whisper is None:
            return {
                "content": "Failed to import the Whisper library. Please check server logs."
            }

        # Load the model
        await emitter.emit(
            description=f"Loading Whisper model '{self.valves.MODEL}'",
            status="in_progress",
        )
        try:
            model = whisper.load_model(self.valves.MODEL)
        except Exception as exc:
            await emitter.emit(
                description=f"Could not load Whisper model: {exc}",
                status="error",
                done=True,
            )
            return {"content": "Error loading Whisper model."}

        # Perform the transcription in a background thread
        await emitter.emit("Transcribing audio…", status="in_progress")
        try:
            loop = asyncio.get_event_loop()
            kwargs_transcribe = self.valves.to_kwargs()
            result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(file_path, **kwargs_transcribe),
            )
        except Exception as exc:
            await emitter.emit(
                description=f"Error during transcription: {exc}",
                status="error",
                done=True,
            )
            return {"content": "An error occurred while transcribing the audio."}

        # Final update
        await emitter.emit(
            description="Transcription completed successfully",
            status="complete",
            done=True,
        )

        transcript_text = result.get("text", "")
        # Return the transcript.  Additional data (e.g. segments) could be
        # returned as part of the message body or as attached files in the
        # future.
        return {
            "content": transcript_text.strip() or "(No speech detected in the audio.)"
        }