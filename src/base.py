"""Shared base model for frozen, strict Pydantic models."""

import pydantic


class FrozenModel(pydantic.BaseModel):
    """Immutable Pydantic model: forbids extra fields, disables coercion, validates defaults."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)
