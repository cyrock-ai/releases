"""Validation for the parameter maps bound to a filter's ``$name`` references.

Mirrors the Java ``FilterParameters`` helper. A search filter such as ``visibility = $v`` is parsed
once on the server and every ``$name`` is resolved from this map, so a caller-supplied value is always
compared as a literal and can never change the structure of the predicate - the safe alternative to
formatting the value into the filter string (issue #689).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__: list[str] = []


def copy_filter_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a validated copy of ``parameters``, or an empty dict for ``None``.

    :raises ValueError: if a parameter name is not a non-blank string, or a value or an element of a list
        value is ``None`` (test for an absent value with ``IS NULL`` instead).
    """
    if not parameters:
        return {}
    copied: dict[str, Any] = {}
    for name, value in parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("filter parameter name must be a non-blank string")
        if value is None:
            raise ValueError(
                f"filter parameter ${name} must not be None; use IS NULL to test for an absent value"
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            if any(element is None for element in value):
                raise ValueError(f"filter parameter ${name} must not contain None elements")
            # A list value (an ``IN $name`` operand) is frozen too, so the caller cannot change it afterwards.
            value = tuple(value)
        copied[name] = value
    return copied
