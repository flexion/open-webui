"""
title: DRAFT - Flexion Advising Answers
author: bdruth
author_url: https://github.com/bdruth
version: 0.1
"""

from pydantic import BaseModel, Field
import requests
from fastapi import Request

from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion

FLEXION_POLICY_PROMPT = (
    "You are Flexion's helpful employee policy assistant. Your role is to provide accurate, actionable guidance to employees about company policies and procedures.\n\n"
    "INSTRUCTIONS:\n"
    "- Use ONLY the provided context to answer the employee's question\n"
    "- Always include specific URLs from the context when they are available\n"
    "- Provide clear, actionable next steps when possible\n"
    "- If the context has partial information, acknowledge what you know and suggest where to get complete information\n"
    "- Keep answers professional but friendly\n"
    "- If no relevant information is found in the context, clearly state this\n\n"
    "Context:\n{context}\n\n"
    "Employee Question: {question}\n\n"
    "Policy Guidance:"
)


class Pipe:
    class Valves(BaseModel):
        MODEL_ID: str = Field(default="us.meta.llama3-1-8b-instruct-v1:0")
        FLEXION_POLICY_PROMPT: str = Field(default=FLEXION_POLICY_PROMPT)

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
    ) -> str:
        # Logic goes here
        messages = body.get("messages", [])

        if messages:
            question = messages[-1]["content"]
            if "Prompt: " in question:
                question = question.split("Prompt: ")[-1]
            try:
                url = "https://2aengmh5jkmyk4feh32ovm54q40gdket.lambda-url.us-west-2.on.aws/"
                headers = {"Content-Type": "application/json"}
                data = {"query": question, "k": 3}

                response = requests.post(url, headers=headers, json=data)
                results = response.json()

                output = ""
                for i, result in enumerate(results["results"], 1):
                    content = result["chunk"]["content"]
                    output += "```\n"
                    output += content
                    output += "\n```\n\n"

                prompt = self.valves.FLEXION_POLICY_PROMPT.format(
                    context=output, question=question
                )
                messages[-1]["content"] = prompt
                user = Users.get_user_by_id(__user__["id"])
                body["model"] = "us.meta.llama3-1-8b-instruct-v1:0"

                return await generate_chat_completion(__request__, body, user)
            except Exception as e:
                error_msg = f"Error during sequence execution: {str(e)}"
                return {"error": error_msg}
