"""One page of a paged listing.

Mirrors ``Page``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["Page"]

T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of a paged listing, such as the graphs or the collections of a project.

    A listing is paged by id: each page holds the next items in ascending id order and its ``next_page_token``
    resumes after the last of them. An item created or deleted between two pages therefore moves none of the
    others: every item that exists from the first page to the last is listed exactly once, and one created or
    deleted meanwhile is listed at most once.

    ``items`` is empty when nothing follows the previous page. ``next_page_token`` fetches the next page and is
    the empty string on the last one; it is opaque, so pass back exactly what the server sent.
    """

    items:           tuple[T, ...]
    next_page_token: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise ValueError(f"items must be a tuple, got {type(self.items).__name__}")
        if not isinstance(self.next_page_token, str):
            raise ValueError(f"next_page_token must be a str, got {type(self.next_page_token).__name__}")

    def has_next_page(self) -> bool:
        """Whether another page follows this one, that is whether ``next_page_token`` is not empty."""
        return self.next_page_token != ""
