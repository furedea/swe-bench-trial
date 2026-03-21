"""Model name helpers."""


def normalize_model_name(model: str) -> str:
    """Add anthropic/ prefix if not already namespaced.

    Args:
        model (str): Model name with or without provider prefix.

    Returns:
        str: Model name with anthropic/ prefix.
    """
    if "/" in model:
        return model
    return f"anthropic/{model}"
