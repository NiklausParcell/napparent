"""NNM tabular: Psychic Barnacle pipeline (Rust extension)."""

try:
    from nnm_tabular._nnm_tabular import (
        return_barn_record_batches,
        split_batch_xy,
    )
except ImportError:  # pragma: no cover
    return_barn_record_batches = None  # type: ignore
    split_batch_xy = None  # type: ignore

__all__ = [
    "return_barn_record_batches",
    "split_batch_xy",
]
