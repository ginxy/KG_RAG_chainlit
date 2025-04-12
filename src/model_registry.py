import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, config_path="./config/models.yaml"):
        self.config = self._load_config(config_path)
        if self.config:
            logger.info(f"Model registry initialized with {len(self.config.get('options', []))} models")
            logger.debug(f"Full config content: {self.config}")

        else:
            logger.warning("Failed to initialize model registry config")

    @staticmethod
    def _load_config(path):
        try:
            config_file = Path(__file__).parent.parent / path
            logger.debug(f"Looking for config file at: {config_file}")

            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return {"default": "deepseek-r1:1.5b", "options": []}

            with open(config_file) as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded config: {config}")
                return config.get('models', config)
        except Exception as e:
            logger.error(f"Error loading model config from {path}: {str(e)}")
            # Return a minimal default config to prevent errors
            return {"default": "deepseek-r1:1.5b", "options": []}

    def get_model_config(self, model_name=None):
        try:
            if not self.config:
                logger.warning("No config available, returning None")
                return None

            default_model = self.config.get("default", "llama3")
            model_name = model_name or default_model

            logger.debug(f"Looking for model config: {model_name}, default: {default_model}")

            # Check if we have config options
            if "options" not in self.config or not self.config["options"]:
                logger.warning("No model options found in config")
                return None

            # Find matching model config
            for model in self.config["options"]:
                if model.get("name") == model_name:
                    logger.info(f"Found config for model: {model_name}")
                    return model

            # If no match, try to use default
            logger.warning(f"Model '{model_name}' not found in registry, trying default: {default_model}")
            for model in self.config["options"]:
                if model.get("name") == default_model:
                    logger.info(f"Using default model config: {default_model}")
                    return model

            # No config found at all
            logger.error(f"No configuration found for model '{model_name}' or default '{default_model}'")
            return None

        except Exception as e:
            logger.error(f"Error retrieving model config: {str(e)}")
            return None