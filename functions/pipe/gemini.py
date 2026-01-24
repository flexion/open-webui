"""
title: Gemini Pipe
author_url:Luke Garceau & https://linux.do/u/coker/summary
author:coker
version: 1.2.0
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
    def get_google_models(self) -> List[dict]:
        self.GOOGLE_API_KEY = random.choice(
            self.valves.GOOGLE_API_KEYS.split(",")
        ).strip()
        if not self.GOOGLE_API_KEY:
            return [{"id": "error", "name": f"Error: API Key not found"}]
        try:
            url = f"{self.valves.BASE_URL}/models?key={self.GOOGLE_API_KEY}"
            response = httpx.get(url, timeout=10)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
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
            return [{"id": "error", "name": f"Could not fetch models: {str(e)}"}]
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
        return [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    def split_image(self, content):
        pattern = r"!\[image\]\(data:([^;]+);base64,([^)]+)\)"
        matches = re.findall(pattern, content)
        image_data_list = []
        if matches:
            for mime_type, base64_data in matches:
                image_data_list.append({"mimeType": mime_type, "data": base64_data})
            content = re.sub(r"!\[image\]\(data:[^;]+;base64,[^)]+\)", "", content)
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
                        new_message["parts"].append(
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
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
                try:
                    res += f'\n ![image](data:{part["inlineData"]["mimeType"]};base64,{part["inlineData"]["data"]}) \n'
                except:
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
                self.emit_status(message="🐎 Generating image...")
                async for res in self.gen_image(body["messages"][-1], model_id):
                    yield res
                return
            if "veo" in model_id:
                self.emit_status(message="🐎 Generating video...")
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
            # if len(str(contents)) >1000:
            #     yield "contents :"+str(contents)[:1000]+"......"
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
            params = {"key": self.GOOGLE_API_KEY}
            if stream:
                url = f"{self.valves.BASE_URL}/models/{model_id}:streamGenerateContent"
                params["alt"] = "sse"
            else:
                url = f"{self.valves.BASE_URL}/models/{model_id}:generateContent"
            headers = {"Content-Type": "application/json"}
            async with httpx.AsyncClient() as client:
                if stream:
                    async with client.stream(
                        "POST",
                        url,
                        json=request_data,
                        headers=headers,
                        params=params,
                        timeout=500,
                    ) as response:
                        if response.status_code != 200:
                            error_content = await response.aread()
                            yield f"Error: HTTP {response.status_code}: {error_content.decode('utf-8')}"
                            await self.emit_status(message="❌ Generation failed", done=True)
                            return
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "candidates" in data and data["candidates"]:
                                        try:
                                            parts = data["candidates"][0]["content"][
                                                "parts"
                                            ]
                                        except:
                                            if (
                                                "finishReason" in data["candidates"][0]
                                                and data["candidates"][0][
                                                    "finishReason"
                                                ]
                                                != "STOP"
                                            ):
                                                yield "\n---\n" + "Abnormal termination: " + data[
                                                    "candidates"
                                                ][0]["finishReason"]
                                                return
                                            else:
                                                continue
                                        text = await self.do_parts(parts)
                                        yield text
                                        try:
                                            if (
                                                self.open_search
                                                and self.valves.OPEN_SEARCH_INFO
                                                and data["candidates"][0][
                                                    "groundingMetadata"
                                                ]["groundingChunks"]
                                            ):
                                                yield "\n---------------------------------\n"
                                                groundingChunks = data["candidates"][0][
                                                    "groundingMetadata"
                                                ]["groundingChunks"]
                                                for idx, groundingChunk in enumerate(
                                                    groundingChunks, 1
                                                ):
                                                    if "web" in groundingChunk:
                                                        yield self.create_search_link(
                                                            idx, groundingChunk["web"]
                                                        )
                                        except Exception as e:
                                            pass
                                except Exception as e:
                                    # yield f"Error parsing stream: {str(e)}"
                                    pass
                        await self.emit_status(message="🎉 Generation completed", done=True)
                else:
                    response = await client.post(
                        url,
                        json=request_data,
                        headers=headers,
                        params=params,
                        timeout=120,
                    )
                    if response.status_code != 200:
                        yield f"Error: HTTP {response.status_code}: {response.text}"
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
                                and data["candidates"][0]["groundingMetadata"][
                                    "groundingChunks"
                                ]
                            ):
                                res += "\n---------------------------------\n"
                                groundingChunks = data["candidates"][0][
                                    "groundingMetadata"
                                ]["groundingChunks"]
                                for idx, groundingChunk in enumerate(
                                    groundingChunks, 1
                                ):
                                    if "web" in groundingChunk:
                                        res += self.create_search_link(
                                            idx, groundingChunk["web"]
                                        )
                        except Exception as e:
                            pass
                        await self.emit_status(message="🎉 Generation completed", done=True)
                        yield res
                    else:
                        yield "No response data"
        except Exception as e:
            yield f"Error: {str(e)}"
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
        url = f"{self.base_url}/models/{model}:predict"
        params = {"key": self.GOOGLE_API_KEY}
        headers = {"Content-Type": "application/json"}
        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": self.valves.IMAGE_NUM,  # @param {type:"number", min:1, max:4}
                "personGeneration": "allow_adult",  # @param ["dont_allow", "allow_adult"]
                "aspectRatio": self.valves.IMAGE_RATIO,  # @param ["1:1", "3:4", "4:3", "16:9", "9:16"]
            },
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json=request_data, headers=headers, params=params, timeout=120
            )
            if response.status_code != 200:
                yield f"Error: HTTP {response.status_code}: {response.text}"
                self.emit_status(message="❌ Generation failed", done=True)
                return
            data = response.json()
            self.emit_status(message="🎉 Generation completed", done=True)
            if "predictions" in data and isinstance(data["predictions"], list):
                yield f"Number of generated images: {len(data['predictions'])}\n\n"
                for index, prediction in enumerate(data["predictions"]):
                    base64_str = (
                        prediction["bytesBase64Encoded"]
                        if "bytesBase64Encoded" in prediction
                        else None
                    )
                    if base64_str:
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
                "aspectRatio": self.valves.VIDEO_RATIO,  # @param ["16:9", "9:16"]
                "negativePrompt": self.valves.VIDEO_NEGATIVE_PROMPT,  # @param {type:"string"}
                "personGeneration": "allow_adult",
                "sampleCount": self.valves.VIDEO_NUM,  # @param {type:"number", min:1, max:2}
                "durationSeconds": self.valves.VIDEO_DURATION,  # @param {type:"number", min:5, max:8}
            },
        }
        if img_base64_str:
            request_data["instances"][0]["image"] = {
                "bytesBase64Encoded": img_base64_str,
                "mimeType": "image/jpeg",
            }
            request_data["parameters"].pop("personGeneration", None)
        params = {"key": self.GOOGLE_API_KEY}
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json=request_data, headers=headers, params=params, timeout=120
            )
            if response.status_code != 200:
                yield f"Error: HTTP {response.status_code}: {response.text}"
                self.emit_status(message="❌ Generation failed", done=True)
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
                self.emit_status(message=plan_id, done=False)
                start_time = time.time()
                result_url = []
                while True:
                    if time.time() - start_time > 300:
                        yield "Error: Timeout"
                        return
                    response = await client.get(
                        f"{self.base_url}/{plan_id}",
                        headers=headers,
                        params=params,
                        timeout=120,
                    )
                    if response.status_code != 200:
                        yield f"Error: HTTP {response.status_code}: {response.text}"
                        return
                    data = response.json()
                    if data.get("done", "") and data["done"]:
                        if "error" in data:
                            if "message" in data["error"]:
                                yield f"Error: {data['error']['message']}"
                            else:
                                yield f"Error: {response.text}"
                            self.emit_status(message="❌ Generation failed", done=True)
                            return
                        if (
                            "generateVideoResponse" not in data["response"]
                            or "generatedSamples"
                            not in data["response"]["generateVideoResponse"]
                        ):
                            yield "Error: " + response.text
                            self.emit_status(message="❌ Generation failed", done=True)
                            return
                        self.emit_status(message="🎉 Generation completed", done=True)
                        for i in data["response"]["generateVideoResponse"][
                            "generatedSamples"
                        ]:
                            result_url.append(i["video"]["uri"].split("?")[0])
                        break
                    else:
                        self.emit_status(message="Generating video...", done=False)
                        await asyncio.sleep(10)
                if result_url:
                    params["alt"] = "media"
                    for idx, url in enumerate(result_url, 1):
                        try:
                            resp = await client.get(
                                url, params=params, timeout=120, follow_redirects=True
                            )
                            resp.raise_for_status()
                            video_bytes = resp.content
                            b64_video = base64.b64encode(video_bytes).decode("utf-8")
                            yield "\n\n" + "```html\n<video width='350px' height='280px' controls='controls' autoplay='autoplay' loop='loop' preload='auto' src='data:video/mp4;base64,{}'></video>\n```".format(
                                b64_video
                            )
                        except Exception as e:
                            yield f"Error downloading/encoding video {idx}: {e}" + "\n\n"
                            continue
            except Exception as e:
                yield f"Error: {str(e)}"
                self.emit_status(message="❌ Generation failed", done=True)
                return