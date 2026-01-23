"""
title: Gemini Pipe (Security Hardened)
author_url:Luke Garceau & https://linux.do/u/coker/summary
author:coker
version: 1.2.0-security-fixed
license: MIT
"""
import json
import random
import httpx
from typing import List, AsyncGenerator, Callable, Awaitable, Optional, Union
from pydantic import BaseModel, Field
import re
import time
import asyncio
import base64
import mimetypes

class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEYS: str = Field(
            default="",
            description="API Keys for Google, use , to split",
        )
        BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com/v1beta",
            description="API Base Url",
        )
        OPEN_SEARCH_INFO: bool = Field(
            default=True, description="Open search info show "
        )
        IMAGE_NUM: int = Field(default=2, description="1-4")
        IMAGE_RATIO: str = Field(
            default="16:9", description="1:1, 3:4, 4:3, 16:9, 9:16"
        )
        THINKING_BUDGET: int = Field(
            default=1000, description="Thinking budget, Max 24576"
        )
        VIDEO_RATIO: str = Field(default="16:9", description="16:9, 9:16")
        VIDEO_NUM: int = Field(default=1, description="1-2")
        VIDEO_DURATION: int = Field(default=5, description="5-8")
        VIDEO_NEGATIVE_PROMPT: str = Field(default="", description="Negative prompt")
        MAX_BASE64_SIZE_MB: int = Field(
            default=10, description="Maximum base64 data size in MB"
        )
        ENABLE_SAFETY_FILTERS: bool = Field(
            default=True, description="Enable Google safety filters"
        )
        SAFETY_THRESHOLD: str = Field(
            default="BLOCK_MEDIUM_AND_ABOVE",
            description="Safety threshold: BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH"
        )

    def __init__(self):
        self.type = "manifold"
        self.name = "Google: "
        self.valves = self.Valves()
        self.OPEN_SEARCH_MODELS = ["gemini-2.5-pro-exp-03-25"]
        self.OPEN_THINK_BUDGET_MODELS = ["gemini-2.5-flash-preview-04-17"]
        self.emitter = None
        self.open_search = False
        self.open_image = False
        self.open_think = False
        self.think_first = True

    def _get_headers(self, api_key: str) -> dict:
        """Create secure headers with API key"""
        return {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key
        }

    def _validate_base64_size(self, base64_data: str) -> bool:
        """Validate base64 data size to prevent memory exhaustion"""
        try:
            # Calculate actual size (base64 encoding increases size by ~33%)
            size_bytes = len(base64_data) * 3 / 4
            max_bytes = self.valves.MAX_BASE64_SIZE_MB * 1024 * 1024
            return size_bytes <= max_bytes
        except Exception:
            return False

    def _detect_mime_type(self, base64_data: str) -> str:
        """Detect MIME type from base64 data"""
        try:
            # Decode first few bytes to detect magic numbers
            decoded = base64.b64decode(base64_data[:100])

            # Check magic numbers
            if decoded.startswith(b'\xFF\xD8\xFF'):
                return "image/jpeg"
            elif decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return "image/png"
            elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
                return "image/gif"
            elif decoded.startswith(b'RIFF') and b'WEBP' in decoded[:20]:
                return "image/webp"
            else:
                # Default to jpeg if unknown
                return "image/jpeg"
        except Exception:
            return "image/jpeg"

    def _sanitize_error(self, error_msg: str) -> str:
        """Sanitize error messages to prevent information disclosure"""
        # Remove API keys if accidentally included
        sanitized = re.sub(r'AIza[0-9A-Za-z\-_]{35}', '[REDACTED_API_KEY]', error_msg)
        # Remove sensitive paths
        sanitized = re.sub(r'/[A-Za-z0-9/_\-\.]+/(projects|operations|storage)/[A-Za-z0-9/_\-\.]+', '[REDACTED_PATH]', sanitized)
        # Limit error message length
        if len(sanitized) > 200:
            sanitized = sanitized[:200] + "..."
        return sanitized

    def get_google_models(self) -> List[dict]:
        self.GOOGLE_API_KEY = random.choice(
            self.valves.GOOGLE_API_KEYS.split(",")
        ).strip()
        if not self.GOOGLE_API_KEY:
            return [{"id": "error", "name": f"Error: API Key not found"}]
        try:
            # SECURITY FIX: API key in headers instead of query params
            url = f"{self.valves.BASE_URL}/models"
            headers = self._get_headers(self.GOOGLE_API_KEY)
            response = httpx.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                sanitized_error = self._sanitize_error(response.text)
                raise Exception(f"HTTP {response.status_code}: {sanitized_error}")

            data = response.json()
            models = [
                {
                    "id": model["name"].split("/")[-1],
                    "name": model["name"].split("/")[-1],
                }
                for model in data.get("models", [])
                if (
                    "generateContent" in model.get("supportedGenerationMethods", [])
                    or "predict" in model.get("supportedGenerationMethods", [])
                    or "predictLongRunning"
                    in model.get("supportedGenerationMethods", [])
                )
            ]
            if self.OPEN_SEARCH_MODELS:
                models.extend(
                    [
                        {
                            "id": model + "-search",
                            "name": model + "-search",
                        }
                        for model in self.OPEN_SEARCH_MODELS
                    ]
                )
            if self.OPEN_THINK_BUDGET_MODELS:
                models.extend(
                    [
                        {
                            "id": model + "-thinking",
                            "name": model + "-thinking",
                        }
                        for model in self.OPEN_THINK_BUDGET_MODELS
                    ]
                )
            return models
        except Exception as e:
            sanitized_error = self._sanitize_error(str(e))
            return [{"id": "error", "name": f"Could not fetch models: {sanitized_error}"}]

    async def emit_status(
        self,
        message: str = "",
        done: bool = False,
    ):
        if self.emitter:
            await self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "done": done,
                    },
                }
            )

    def pipes(self) -> List[dict]:
        return self.get_google_models()

    def create_search_link(self, idx, web):
        return f'\n{idx:02d}: [**{web["title"]}**]({web["uri"]})'

    def create_think_info(self, think_info):
        pass

    def _get_safety_settings(self, model: str):
        """Get safety settings with configurable thresholds"""
        if not self.valves.ENABLE_SAFETY_FILTERS:
            # Only disable if explicitly configured
            if model == "gemini-2.0-flash-exp" or model == "gemini-2.0-flash-exp-image-generation":
                threshold = "OFF"
            else:
                threshold = "BLOCK_NONE"
        else:
            # Use configured threshold
            threshold = self.valves.SAFETY_THRESHOLD

        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": threshold},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": threshold},
        ]

    def split_image(self, content):
        # SECURITY FIX: More robust regex to prevent ReDoS
        # Limit capture groups and use atomic grouping where possible
        pattern = r"!\[image\]\(data:([^;]+);base64,([A-Za-z0-9+/=]+)\)"
        matches = re.findall(pattern, content)
        image_data_list = []

        if matches:
            for mime_type, base64_data in matches:
                # SECURITY FIX: Validate base64 size
                if not self._validate_base64_size(base64_data):
                    raise ValueError(f"Base64 data exceeds maximum size of {self.valves.MAX_BASE64_SIZE_MB}MB")

                image_data_list.append({"mimeType": mime_type, "data": base64_data})

            content = re.sub(r"!\[image\]\(data:[^;]+;base64,[A-Za-z0-9+/=]+\)", "", content)

        if not content:
            content = "Please refer to the image content"
        return content, image_data_list

    def convert_message(self, message) -> dict:
        new_message = {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [],
        }
        if isinstance(message.get("content"), str):
            if not message["role"] == "assistant":
                new_message["parts"].append({"text": message["content"]})
                return new_message
            content, image_data_list = self.split_image(message["content"])
            new_message["parts"].append({"text": content})
            if image_data_list:
                for image_data in image_data_list:
                    new_message["parts"].append(
                        {
                            "inline_data": {
                                "mime_type": image_data["mimeType"],
                                "data": image_data["data"],
                            }
                        }
                    )
            return new_message
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content["type"] == "text":
                    new_message["parts"].append({"text": content["text"]})
                elif content["type"] == "image_url":
                    image_url = content["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        image_data = image_url.split(",")[1]

                        # SECURITY FIX: Validate size and detect MIME type
                        if not self._validate_base64_size(image_data):
                            raise ValueError(f"Image data exceeds maximum size of {self.valves.MAX_BASE64_SIZE_MB}MB")

                        mime_type = self._detect_mime_type(image_data)

                        new_message["parts"].append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_data,
                                }
                            }
                        )
        return new_message

    async def do_parts(self, parts):
        res = ""
        if not parts or not isinstance(parts, list):
            return "Error: No parts found"
        for part in parts:
            if "text" in part:
                res += part["text"]
            if "inlineData" in part and part["inlineData"]:
                # SECURITY FIX: Specific exception handling
                try:
                    mime_type = part["inlineData"].get("mimeType", "image/jpeg")
                    data = part["inlineData"].get("data", "")
                    res += f'\n ![image](data:{mime_type};base64,{data}) \n'
                except (KeyError, TypeError) as e:
                    # Log error but continue processing
                    pass
        return res

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__
        self.GOOGLE_API_KEY = random.choice(
            self.valves.GOOGLE_API_KEYS.split(",")
        ).strip()
        self.base_url = self.valves.BASE_URL
        if not self.GOOGLE_API_KEY:
            yield "Error: GOOGLE_API_KEY is not set"
            return

        try:
            model_id = body["model"]
            if "." in model_id:
                model_id = model_id.split(".", 1)[1]

            if "imagen" in model_id:
                await self.emit_status(message="🐎 Generating image...")
                async for res in self.gen_image(body["messages"][-1], model_id):
                    yield res
                return

            if "veo" in model_id:
                await self.emit_status(message="🐎 Generating video...")
                async for res in self.gen_veo(body["messages"][-1], model_id):
                    yield res
                return

            messages = body["messages"]
            stream = body.get("stream", False)

            # Prepare the request payload
            contents = []
            request_data = {
                "generationConfig": {
                    "temperature": body.get("temperature", 0.7),
                    "topP": body.get("top_p", 0.9),
                    "topK": body.get("top_k", 40),
                    "maxOutputTokens": body.get("max_tokens", 8192),
                    "stopSequences": body.get("stop", []),
                },
            }

            for message in messages:
                if message["role"] == "system":
                    request_data["system_instruction"] = {
                        "parts": [{"text": message["content"]}]
                    }
                    continue
                contents.append(self.convert_message(message))

            request_data["contents"] = contents

            if model_id.endswith("-search"):
                model_id = model_id[:-7]
                request_data["tools"] = [{"googleSearch": {}}]
                self.open_search = True
                await self.emit_status(message="🔍 Searching...")
            elif "thinking" in model_id:
                await self.emit_status(message="🧐 Thinking...")
                self.open_think = True
                self.think_first = True
                if model_id.endswith("-thinking"):
                    model_id = model_id[:-9]
                    request_data["generationConfig"]["thinking_config"] = {
                        "thinking_budget": self.valves.THINKING_BUDGET
                    }
            elif model_id.endswith("-image-generation"):
                request_data["generationConfig"]["response_modalities"] = [
                    "Text",
                    "Image",
                ]
                self.open_image = True
            else:
                await self.emit_status(message="🚀 Generating...")

            request_data["safetySettings"] = self._get_safety_settings(model_id)

            # SECURITY FIX: API key in headers, not query params
            headers = self._get_headers(self.GOOGLE_API_KEY)

            if stream:
                url = f"{self.valves.BASE_URL}/models/{model_id}:streamGenerateContent?alt=sse"
            else:
                url = f"{self.valves.BASE_URL}/models/{model_id}:generateContent"

            async with httpx.AsyncClient() as client:
                if stream:
                    async with client.stream(
                        "POST",
                        url,
                        json=request_data,
                        headers=headers,
                        timeout=500,
                    ) as response:
                        if response.status_code != 200:
                            error_content = await response.aread()
                            sanitized_error = self._sanitize_error(error_content.decode('utf-8'))
                            yield f"Error: HTTP {response.status_code}: {sanitized_error}"
                            await self.emit_status(message="❌ Generation failed", done=True)
                            return

                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "candidates" in data and data["candidates"]:
                                        try:
                                            parts = data["candidates"][0]["content"]["parts"]
                                        except (KeyError, IndexError) as e:
                                            if (
                                                "finishReason" in data["candidates"][0]
                                                and data["candidates"][0]["finishReason"] != "STOP"
                                            ):
                                                yield "\n---\n" + "Abnormal termination: " + data["candidates"][0]["finishReason"]
                                                return
                                            else:
                                                continue

                                        text = await self.do_parts(parts)
                                        yield text

                                        try:
                                            if (
                                                self.open_search
                                                and self.valves.OPEN_SEARCH_INFO
                                                and data["candidates"][0]["groundingMetadata"]["groundingChunks"]
                                            ):
                                                yield "\n---------------------------------\n"
                                                groundingChunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
                                                for idx, groundingChunk in enumerate(groundingChunks, 1):
                                                    if "web" in groundingChunk:
                                                        yield self.create_search_link(idx, groundingChunk["web"])
                                        except (KeyError, TypeError):
                                            pass

                                except json.JSONDecodeError:
                                    # Skip malformed JSON lines
                                    pass

                        await self.emit_status(message="🎉 Generation completed", done=True)
                else:
                    response = await client.post(
                        url,
                        json=request_data,
                        headers=headers,
                        timeout=120,
                    )

                    if response.status_code != 200:
                        sanitized_error = self._sanitize_error(response.text)
                        yield f"Error: HTTP {response.status_code}: {sanitized_error}"
                        return

                    data = response.json()
                    res = ""

                    if "candidates" in data and data["candidates"]:
                        parts = data["candidates"][0]["content"]["parts"]
                        res = await self.do_parts(parts)

                        try:
                            if (
                                self.open_search
                                and self.valves.OPEN_SEARCH_INFO
                                and data["candidates"][0]["groundingMetadata"]["groundingChunks"]
                            ):
                                res += "\n---------------------------------\n"
                                groundingChunks = data["candidates"][0]["groundingMetadata"]["groundingChunks"]
                                for idx, groundingChunk in enumerate(groundingChunks, 1):
                                    if "web" in groundingChunk:
                                        res += self.create_search_link(idx, groundingChunk["web"])
                        except (KeyError, TypeError):
                            pass

                        await self.emit_status(message="🎉 Generation completed", done=True)
                        yield res
                    else:
                        yield "No response data"

        except ValueError as e:
            # Handle validation errors
            yield f"Validation Error: {str(e)}"
            await self.emit_status(message="❌ Validation failed", done=True)
        except Exception as e:
            sanitized_error = self._sanitize_error(str(e))
            yield f"Error: {sanitized_error}"
            await self.emit_status(message="❌ Generation failed", done=True)

    async def gen_image(
        self, message: Optional[Union[dict, list]], model: str
    ) -> AsyncGenerator[str, None]:
        content = message.get("content", "")
        if isinstance(content, str):
            prompt = content
        elif isinstance(content, list) and len(content) > 0:
            for msg in content:
                if msg["type"] == "text":
                    prompt = msg["text"]
                    break
        else:
            yield "Error: No prompt found"
            return

        # SECURITY FIX: API key in headers
        url = f"{self.base_url}/models/{model}:predict"
        headers = self._get_headers(self.GOOGLE_API_KEY)

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": self.valves.IMAGE_NUM,
                "personGeneration": "allow_adult",
                "aspectRatio": self.valves.IMAGE_RATIO,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json=request_data, headers=headers, timeout=120
            )

            if response.status_code != 200:
                sanitized_error = self._sanitize_error(response.text)
                yield f"Error: HTTP {response.status_code}: {sanitized_error}"
                await self.emit_status(message="❌ Generation failed", done=True)
                return

            data = response.json()
            await self.emit_status(message="🎉 Generation completed", done=True)

            if "predictions" in data and isinstance(data["predictions"], list):
                yield f"Number of generated images: {len(data['predictions'])}\n\n"
                for index, prediction in enumerate(data["predictions"]):
                    base64_str = prediction.get("bytesBase64Encoded")

                    if base64_str:
                        # SECURITY FIX: Validate size
                        if not self._validate_base64_size(base64_str):
                            yield f"Image {index+1}: Exceeds size limit\n"
                            continue

                        size_bytes = len(base64_str) * 3 / 4
                        if size_bytes >= 1024 * 1024:
                            size = round(size_bytes / (1024 * 1024), 1)
                            unit = "MB"
                        else:
                            size = round(size_bytes / 1024, 1)
                            unit = "KB"

                        yield f"Image {index+1} size: {size} {unit}\n"
                        yield f'![image](data:{prediction["mimeType"]};base64,{base64_str}) \n\n'
                    else:
                        yield "No image data found"

    async def gen_veo(
        self, message: Optional[Union[dict, list]], model: str
    ) -> AsyncGenerator[str, None]:
        content = message.get("content", "")
        img_base64_str = None

        if isinstance(content, str):
            prompt = content
        elif isinstance(content, list) and len(content) > 0:
            for msg in content:
                if msg["type"] == "text":
                    prompt = msg["text"]
                elif msg["type"] == "image_url":
                    if msg["image_url"]["url"].startswith("data:image"):
                        img_base64_str = msg["image_url"]["url"].split(",")[1]

                        # SECURITY FIX: Validate size
                        if not self._validate_base64_size(img_base64_str):
                            yield f"Error: Input image exceeds maximum size of {self.valves.MAX_BASE64_SIZE_MB}MB"
                            return
        else:
            yield "Error: Invalid message format"
            return

        url = f"{self.base_url}/models/{model}:predictLongRunning"

        if not prompt:
            yield "Error: No prompt found"
            return

        request_data = {
            "instances": [
                {
                    "prompt": prompt,
                }
            ],
            "parameters": {
                "aspectRatio": self.valves.VIDEO_RATIO,
                "negativePrompt": self.valves.VIDEO_NEGATIVE_PROMPT,
                "personGeneration": "allow_adult",
                "sampleCount": self.valves.VIDEO_NUM,
                "durationSeconds": self.valves.VIDEO_DURATION,
            },
        }

        if img_base64_str:
            request_data["instances"][0]["image"] = {
                "bytesBase64Encoded": img_base64_str,
                "mimeType": "image/jpeg",
            }
            request_data["parameters"].pop("personGeneration", None)

        # SECURITY FIX: API key in headers
        headers = self._get_headers(self.GOOGLE_API_KEY)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json=request_data, headers=headers, timeout=120
            )

            if response.status_code != 200:
                sanitized_error = self._sanitize_error(response.text)
                yield f"Error: HTTP {response.status_code}: {sanitized_error}"
                await self.emit_status(message="❌ Generation failed", done=True)
                return

            try:
                res_text = response.text
                if res_text.startswith("data: "):
                    res_text = res_text[6:].strip()
                data = json.loads(res_text)
                plan_id = data.get("name", "")

                if not plan_id:
                    yield "Error: No plan ID found"
                    return

                await self.emit_status(message=plan_id, done=False)
                start_time = time.time()
                result_url = []

                while True:
                    if time.time() - start_time > 300:
                        yield "Error: Timeout"
                        return

                    response = await client.get(
                        f"{self.base_url}/{plan_id}",
                        headers=headers,
                        timeout=120,
                    )

                    if response.status_code != 200:
                        sanitized_error = self._sanitize_error(response.text)
                        yield f"Error: HTTP {response.status_code}: {sanitized_error}"
                        return

                    data = response.json()

                    if data.get("done", "") and data["done"]:
                        if "error" in data:
                            error_msg = data["error"].get("message", response.text)
                            sanitized_error = self._sanitize_error(error_msg)
                            yield f"Error: {sanitized_error}"
                            await self.emit_status(message="❌ Generation failed", done=True)
                            return

                        if (
                            "generateVideoResponse" not in data["response"]
                            or "generatedSamples" not in data["response"]["generateVideoResponse"]
                        ):
                            sanitized_error = self._sanitize_error(response.text)
                            yield f"Error: {sanitized_error}"
                            await self.emit_status(message="❌ Generation failed", done=True)
                            return

                        await self.emit_status(message="🎉 Generation completed", done=True)
                        for i in data["response"]["generateVideoResponse"]["generatedSamples"]:
                            result_url.append(i["video"]["uri"].split("?")[0])
                        break
                    else:
                        await self.emit_status(message="Generating video...", done=False)
                        await asyncio.sleep(10)

                if result_url:
                    for idx, url in enumerate(result_url, 1):
                        try:
                            # SECURITY FIX: Use headers for authentication
                            resp = await client.get(
                                url, headers=headers, timeout=120, follow_redirects=True
                            )
                            resp.raise_for_status()
                            video_bytes = resp.content

                            # SECURITY FIX: Validate video size
                            if len(video_bytes) > self.valves.MAX_BASE64_SIZE_MB * 1024 * 1024:
                                yield f"Video {idx}: Exceeds size limit\n"
                                continue

                            b64_video = base64.b64encode(video_bytes).decode("utf-8")
                            yield "\n\n" + "```html\n<video width='350px' height='280px' controls='controls' autoplay='autoplay' loop='loop' preload='auto' src='data:video/mp4;base64,{}'></video>\n```".format(
                                b64_video
                            )
                        except httpx.HTTPError as e:
                            sanitized_error = self._sanitize_error(str(e))
                            yield f"Error downloading video {idx}: {sanitized_error}\n\n"
                            continue

            except json.JSONDecodeError as e:
                yield f"Error: Invalid JSON response"
                await self.emit_status(message="❌ Generation failed", done=True)
                return
            except Exception as e:
                sanitized_error = self._sanitize_error(str(e))
                yield f"Error: {sanitized_error}"
                await self.emit_status(message="❌ Generation failed", done=True)
                return
