from typing import AsyncIterator

from ..settings import LlmSettings


class VllmClient:
    def __init__(self, settings: LlmSettings):
        from openai import AsyncOpenAI

        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key="EMPTY",
            max_retries=0,
        )

    async def stream_reply(
        self, messages: list[dict], request_id: str
    ) -> AsyncIterator[str]:
        kwargs = dict(
            model=self._settings.model,
            messages=messages,
            stream=True,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_tokens=self._settings.max_tokens,
        )
        if self._settings.extra_stop:
            kwargs["extra_body"] = {"stop": self._settings.extra_stop}
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content