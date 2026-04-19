"""Adapters — pluggable memory system implementations.

All adapters satisfy the :class:`mceval.adapters.base.MemoryAdapter` Protocol.
Built-in adapters are importable from here; third-party adapters live in their
own packages and only need to implement the Protocol.
"""

from .base import Memory, MemoryAdapter, Turn

__all__ = ["Memory", "MemoryAdapter", "Turn"]
