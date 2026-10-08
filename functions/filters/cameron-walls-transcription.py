"""
title: Audio Transcription with Whisper
author: GitHub Copilot
author_url: https://github.com/github/copilot
funding_url: https://github.com/sponsors/github
version: 1.0.0
required_open_webui_version: 0.3.8
"""

import os
import tempfile
import json
import base64
import io
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field

# Import Whisper and related libraries
try:
    import whisper
    import torch
    import numpy as np
    from pydub import AudioSegment

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class Filter:
    class Valves(BaseModel):
        """
        Configuration valves for the audio transcription function
        """

        whisper_model: str = Field(
            default="base",
            description="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)",
        )
        max_file_size_mb: int = Field(default=25, description="Maximum file size in MB")
        supported_formats: List[str] = Field(
            default=["mp3", "wav", "m4a", "flac", "ogg", "aac", "wma"],
            description="Supported audio formats",
        )
        language: str = Field(
            default="auto",
            description="Language for transcription (auto for auto-detection, or ISO 639-1 code)",
        )
        temperature: float = Field(
            default=0.0, description="Temperature for Whisper sampling (0.0 to 1.0)"
        )
        include_timestamps: bool = Field(
            default=True, description="Include timestamps in the transcription"
        )
        auto_transcribe: bool = Field(
            default=True,
            description="Automatically transcribe audio files when uploaded",
        )

    def __init__(self):
        # Enable custom file handling
        self.file_handler = True

        # Initialize valves
        self.valves = self.Valves()
        self.whisper_model = None

    def _load_whisper_model(self):
        """Load the Whisper model if not already loaded"""
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper dependencies not available. Please install required packages."
            )

        if self.whisper_model is None:
            try:
                self.whisper_model = whisper.load_model(self.valves.whisper_model)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Whisper model '{self.valves.whisper_model}': {str(e)}"
                )

    def _validate_file(self, file_data: bytes, filename: str) -> bool:
        """Validate the uploaded audio file"""
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > self.valves.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({self.valves.max_file_size_mb}MB)"
            )

        # Check file extension
        file_ext = filename.lower().split(".")[-1] if "." in filename else ""
        if file_ext not in self.valves.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported formats: {', '.join(self.valves.supported_formats)}"
            )

        return True

    def _convert_audio(self, file_data: bytes, filename: str) -> str:
        """Convert audio file to a format suitable for Whisper"""
        try:
            # Create temporary file for input
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{filename.split('.')[-1]}"
            ) as temp_input:
                temp_input.write(file_data)
                temp_input_path = temp_input.name

            # Load audio with pydub
            audio = AudioSegment.from_file(temp_input_path)

            # Convert to wav format for Whisper
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_output:
                temp_output_path = temp_output.name

            # Export as wav with standard settings for Whisper
            audio.export(
                temp_output_path, format="wav", parameters=["-ar", "16000", "-ac", "1"]
            )

            # Clean up input file
            os.unlink(temp_input_path)

            return temp_output_path

        except Exception as e:
            # Clean up files in case of error
            if "temp_input_path" in locals() and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if "temp_output_path" in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            raise RuntimeError(f"Audio conversion failed: {str(e)}")

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """
        Process incoming requests and handle file transcription
        """
        print(f"inlet: Audio Transcription Filter called")

        # Check if auto transcription is enabled
        if not self.valves.auto_transcribe:
            return body

        # Look for files in the request
        files = body.get("files", [])
        if not files:
            return body

        # Process each audio file
        transcriptions = []

        for file_item in files:
            try:
                # Check if it's an audio file
                filename = file_item.get("name", "")
                file_ext = filename.lower().split(".")[-1] if "." in filename else ""

                if file_ext not in self.valves.supported_formats:
                    continue

                print(f"Processing audio file: {filename}")

                if __event_emitter:
                    __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Transcribing {filename}...",
                                "done": False,
                            },
                        }
                    )

                # Get file data
                file_data = file_item.get("data", {})
                content = file_data.get("content", "")

                if not content:
                    continue

                # Transcribe the audio
                transcription = self._transcribe_audio_file(
                    content, filename, __event_emitter
                )
                transcriptions.append(transcription)

            except Exception as e:
                print(f"Error transcribing {filename}: {str(e)}")
                transcriptions.append(f"❌ **Error transcribing {filename}**: {str(e)}")

        # Add transcriptions to the message if any were created
        if transcriptions:
            messages = body.get("messages", [])
            if messages:
                # Find the last user message and append transcriptions
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        current_content = messages[i].get("content", "")
                        transcription_text = "\n\n".join(transcriptions)

                        if current_content:
                            messages[i][
                                "content"
                            ] = f"{current_content}\n\n{transcription_text}"
                        else:
                            messages[i]["content"] = transcription_text
                        break

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Process outgoing responses (no modifications needed for transcription)
        """
        return body

    def _transcribe_audio_file(
        self,
        file_data: str,
        filename: str,
        __event_emitter__: Optional[Callable] = None,
    ) -> str:
        """
        Transcribe an audio file using Whisper

        Args:
            file_data: Base64 encoded audio file data
            filename: Name of the audio file
            __event_emitter__: Event emitter for progress updates (optional)

        Returns:
            Transcribed text with optional timestamps
        """

        try:
            # Decode base64 file data
            try:
                audio_bytes = base64.b64decode(file_data)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {str(e)}")

            # Validate file
            self._validate_file(audio_bytes, filename)

            if __event_emitter:
                __event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": "Loading Whisper model...",
                            "done": False,
                        },
                    }
                )

            # Load Whisper model
            self._load_whisper_model()

            if __event_emitter:
                __event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": "Converting audio format...",
                            "done": False,
                        },
                    }
                )

            # Convert audio to suitable format
            audio_path = self._convert_audio(audio_bytes, filename)

            try:
                if __event_emitter:
                    __event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": "Transcribing audio...",
                                "done": False,
                            },
                        }
                    )

                # Prepare transcription options
                transcribe_options = {
                    "temperature": self.valves.temperature,
                    "word_timestamps": self.valves.include_timestamps,
                }

                # Set language if not auto-detection
                if self.valves.language != "auto":
                    transcribe_options["language"] = self.valves.language

                # Transcribe audio
                result = self.whisper_model.transcribe(audio_path, **transcribe_options)

                # Format output
                output_lines = []
                output_lines.append(f"# Audio Transcription: {filename}")
                output_lines.append("")

                # Add metadata
                if result.get("language"):
                    output_lines.append(f"**Detected Language:** {result['language']}")
                output_lines.append(f"**Model Used:** {self.valves.whisper_model}")
                output_lines.append("")

                # Add transcription
                if self.valves.include_timestamps and "segments" in result:
                    output_lines.append("## Transcription with Timestamps")
                    output_lines.append("")
                    for segment in result["segments"]:
                        start_time = self._format_timestamp(segment["start"])
                        end_time = self._format_timestamp(segment["end"])
                        text = segment["text"].strip()
                        output_lines.append(f"**[{start_time} - {end_time}]** {text}")
                else:
                    output_lines.append("## Transcription")
                    output_lines.append("")
                    output_lines.append(result["text"].strip())

                if __event_emitter:
                    __event_emitter(
                        {
                            "type": "status",
                            "data": {
                                "description": "Transcription completed!",
                                "done": True,
                            },
                        }
                    )

                return "\n".join(output_lines)

            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

        except Exception as e:
            if __event_emitter:
                __event_emitter(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )

            return f"❌ **Transcription Error**\n\n{str(e)}\n\nPlease check your audio file and try again."