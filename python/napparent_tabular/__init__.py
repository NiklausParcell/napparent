"""napparent tabular: effect-feature pipeline (Rust extension)."""

try:
    from napparent_tabular._napparent_tabular import (
        split_batch_xy,
        transform_record_batches,
    )
except ImportError:  # pragma: no cover
    split_batch_xy = None  # type: ignore
    transform_record_batches = None  # type: ignore

__all__ = [
    "split_batch_xy",
    "transform_record_batches",
]
