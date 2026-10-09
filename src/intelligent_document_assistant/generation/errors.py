class GenerationError(Exception):
    """Base class for answer-generation failures."""


class MissingCredentialsError(GenerationError):
    """The chat model's API key is not configured."""


class ModelCallError(GenerationError):
    """The chat model request failed."""


class ModelOutputError(GenerationError):
    """The model's reply was malformed or cited passages that were not provided."""
