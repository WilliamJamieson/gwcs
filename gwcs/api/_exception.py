from __future__ import annotations

__all__ = ["GwcsAxesMismatchError", "GwcsFrameMissingError"]


class GwcsFrameMissingError(RuntimeError):
    """An error for when a frame is missing from the pipeline."""

    @classmethod
    def input_frame(cls) -> GwcsFrameMissingError:
        return cls("The input_frame is not defined. ")

    @classmethod
    def output_frame(cls) -> GwcsFrameMissingError:
        return cls("The output_frame is not defined. ")


class GwcsAxesMismatchError(ValueError):
    """An error for when the number of axes mismatch"""

    @classmethod
    def mismatch(cls, naxes: int, value: tuple[int, ...]) -> GwcsAxesMismatchError:
        return cls(
            f"The number of data axes, {naxes}, does not equal the shape {len(value)}."
        )
