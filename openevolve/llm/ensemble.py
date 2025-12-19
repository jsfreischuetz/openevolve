"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.anthropic import AnthropicLLM
from openevolve.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [self._init_model(model_cfg) for model_cfg in models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    async def generate(
        self,
        prompt: str,
        log_context: Optional[str] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(
            prompt,
            log_context=log_context,
            usage_metadata=usage_metadata,
            **kwargs,
        )

    async def generate_with_context(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
        log_context: Optional[str] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.generate_with_context(
            system_message,
            messages,
            log_context=log_context,
            usage_metadata=usage_metadata,
            **kwargs,
        )

    def _init_model(self, model_cfg: LLMModelConfig) -> LLMInterface:
        """Instantiate a model client based on its configured provider"""
        if model_cfg.init_client:
            return model_cfg.init_client(model_cfg)

        model_type = (model_cfg.model_type or "openai").lower()
        if model_type == "anthropic":
            return AnthropicLLM(model_cfg)
        # Default to OpenAI-compatible client
        return OpenAILLM(model_cfg)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(
        self,
        prompt: str,
        n: int,
        log_context: Optional[str] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [
            self.generate(prompt, log_context=log_context, usage_metadata=usage_metadata, **kwargs)
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

    async def parallel_generate(
        self,
        prompts: List[str],
        log_context: Optional[str] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [
            self.generate(prompt, log_context=log_context, usage_metadata=usage_metadata, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
        log_context: Optional[str] = None,
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(
                await model.generate_with_context(
                    system_message,
                    messages,
                    log_context=log_context,
                    usage_metadata=usage_metadata,
                    **kwargs,
                )
            )
        return responses
