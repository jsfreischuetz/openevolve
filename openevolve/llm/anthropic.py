"""Anthropic API interface for LLMs"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from openevolve.llm.base import LLMInterface
from openevolve.utils.tracking_utils import log_token_usage

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMInterface):
    """LLM interface using Anthropic-compatible APIs"""

    def __init__(self, model_cfg: Optional[dict] = None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.log_dir = os.environ.get("OPENEVOLVE_LOG_DIR")

        # Set up API client (anthropic client handles retries internally; we manage our own loop retries)
        self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.api_base)

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized Anthropic LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Anthropic uses a separate system field; messages should exclude the system entry
        # Add cache_control to each message block to enable caching.
        transformed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                continue
            transformed_messages.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": transformed_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Optional parameters: Anthropic does not allow temperature and top_p together
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if temperature is not None:
            params["temperature"] = temperature
        elif top_p is not None:
            params["top_p"] = top_p

        if system_message is not None:
            params["system"] = [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                content, usage = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                self._log_usage(usage, kwargs.get("log_context"), kwargs.get("usage_metadata"))
                return content
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    def _log_usage(
        self,
        usage: Optional[Dict[str, Any]],
        log_context: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not usage:
            return
        try:
            log_token_usage(
                model_name=self.model,
                provider="anthropic",
                usage=usage,
                log_context=log_context,
                metadata=metadata,
                log_dir=self.log_dir,
            )
        except Exception:
            logger.debug("Failed to log token usage", exc_info=True)

    def _extract_usage(self, response: Any) -> Optional[Dict[str, Any]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        def _maybe_get(obj: Any, key: str) -> Optional[Any]:
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        input_tokens = _maybe_get(usage, "input_tokens")
        output_tokens = _maybe_get(usage, "output_tokens")
        total_tokens = _maybe_get(usage, "total_tokens")
        cache_creation_input_tokens = _maybe_get(usage, "cache_creation_input_tokens")
        cache_read_input_tokens = _maybe_get(usage, "cache_read_input_tokens")

        if total_tokens is None and isinstance(input_tokens, (int, float)) and isinstance(
            output_tokens, (int, float)
        ):
            total_tokens = input_tokens + output_tokens

        # Anthropic exposes cache read; mirror as cached_tokens for unified logging
        cached_tokens = cache_read_input_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cached_tokens": cached_tokens,
        }

    async def _call_api(self, params: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Make the actual API call"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )
        logger.debug(f"API parameters: {params}")

        contents = []
        for block in response.content:
            if hasattr(block, "text"):
                contents.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                contents.append(block["text"])
        combined = "".join(contents)
        logger.debug(f"API response: {combined}")

        usage = self._extract_usage(response)
        return combined, usage
