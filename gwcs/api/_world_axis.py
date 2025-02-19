from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

__all__ = [
    "WorldAxisClass",
    "WorldAxisClasses",
    "WorldAxisComponent",
    "WorldAxisComponents",
    "WorldAxisConverterClass",
]


class WorldAxisClass(NamedTuple):
    """
    Named tuple for the world_axis_object_classes WCS property
    """

    object_type: type | str
    args: tuple[int | None, ...]
    kwargs: dict[str, Any]


class WorldAxisConverterClass(NamedTuple):
    """
    Named tuple for the world_axis_object_classes WCS property, which have a converter
    """

    object_type: type | str
    args: tuple[int | None, ...]
    kwargs: dict[str, Any]
    converter: Callable[..., Any] | None = None


WorldAxisClasses: TypeAlias = dict[str | int, WorldAxisClass | WorldAxisConverterClass]


class WorldAxisComponent(NamedTuple):
    """
    Named tuple for the world_axis_object_components WCS property
    """

    name: str
    key: str | int
    property_name: str | Callable[[Any], Any]


WorldAxisComponents: TypeAlias = list[WorldAxisComponent]
