"""
Large Language Model processor with context-aware response generation
Handles model configuration and streaming responses
"""

from typing import AsyncGenerator, List, Dict, Optional
import os
import logging
from litellm import acompletion
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Orchestrates LLM interactions with fallback configuration handling"""

    def __init__(self, model_name: str = None):
        """
        Initialize processor with model configuration
        Args:
            model_name: Optional override for default model
        """
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.max_tokens = int(os.getenv("MAX_TOKENS", 6000))
        self.temperature = float(os.getenv("LLM_TEMP", 0.7))
        self.template = None

        # Load model configuration
        self._load_model_config()

    def _load_model_config(self):
        """Attempt to load model configuration from registry"""
        try:
            registry = ModelRegistry()
            logger.info(f"Loading model config for {self.model_name}")
            model_config = registry.get_model_config(self.model_name)
            if model_config:
                self.max_tokens = model_config.get("max_tokens", self.max_tokens)
                self.temperature = model_config.get("temperature", self.temperature)
                self.template = model_config.get("template")
                logger.info(f"Found model config: {model_config}")
                logger.info(f"Using max_tokens={self.max_tokens}, temperature={self.temperature}")
                if self.template:
                    logger.info("Custom template loaded successfully")
        except Exception as e:
            logger.warning(f"Using environment config due to registry error: {str(e)}")

    async def generate_response(self, query: str, context: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM
        Args:
            query: User input question
            context: Retrieved knowledge context
        Yields:
            Response tokens
        """
        logger.debug(f"Generating response for query: {query[:50]}...")
        loop_prevention = ("\n[SAFETY PROTOCOLS]\n"
                           "1. If similar content repeats, summarize key points once\n"
                           "2. Never repeat identical phrases\n"
                           "3. If unsure, ask clarifying questions\n")
        messages = self._build_message(query, context)
        if messages and messages[0]['content']:
            messages[0]['content'] += loop_prevention
        try:
            logger.debug(f"Sending request to LLM: {self.model_name} via {self.base_url}")
            response = await acompletion(
                model=f"ollama/{self.model_name}",
                messages=messages,
                api_base=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True)

            logger.debug(f"LLM response received. Streaming response...")
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            yield "⚠️ Response generation error"

    @staticmethod
    def _format_hybrid_context(context: dict) -> str:
        """Format combined KG and vector results for template"""
        context_lines = []

        if context.get("entities"):
            context_lines.append("## Knowledge Graph Entities:")
            for entity in context["entities"]:
                context_lines.append(f"- {entity['name']}: {entity['description']}")

        if context.get("chunks"):
            context_lines.append("\n## Relevant Content:")
            for chunk in context["chunks"]:
                source = chunk.get("source", "kg")
                context_lines.append(f"({source.upper()} score: {chunk['score']:.2f})\n"
                                     f"{chunk['text']}\n")

        return "\n".join(context_lines)

    def _build_message(self, query: str, context: Optional[dict] = None) -> List[Dict[str, str]]:
        """Handle both string and dict contexts"""
        # Convert string context to dict if needed
        if isinstance(context, str):
            logger.warning("Received string context, converting to structured format")
            context = {
                "chunks": [{"text": context}], "entities": []
                }

        if self.template and context:
            logger.debug("Using custom template with structured context")
            context_str = self._format_hybrid_context(context)
            return [{
                "role": "user", "content": self.template.format(context=context_str, query=query)
                }]

        return self._build_default_message(query, context)

    @staticmethod
    def _build_default_message(query: str, context: Optional[dict] = None) -> List[Dict[str, str]]:
        """Handle structured context"""
        base_prompt = "You're an AI assistant with access to contextual knowledge."

        if not context:
            return [{"role": "user", "content": f"{base_prompt}\n\nQuery: {query}"}]

        context_str = "Contextual information:\n"

        if context.get("entities"):
            context_str += "## Knowledge Graph Entities\n" + "\n".join(
                [f"- {e['name']}: {e['description']}" for e in context["entities"]]) + "\n\n"

        if context.get("chunks"):
            context_str += "## Relevant Document Excerpts\n" + "\n".join(
                [f"- {c['text']}" for c in context["chunks"]]) + "\n\n"

        if context.get("files"):
            context_str += "## Uploaded File Context\n" + "\n".join(context["files"])

        return [{
            "role": "user", "content": f"{base_prompt}\n\n{context_str.strip()}\n\nQuery: {query}"
            }]