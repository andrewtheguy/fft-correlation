from typing import Literal

import numpy as np
import numpy.typing as npt

FULL: Literal["full"]
SAME: Literal["same"]
VALID: Literal["valid"]

def fft_correlate_1d(
    signal: npt.NDArray[np.float32],
    template: npt.NDArray[np.float32],
    mode: Literal["full", "same", "valid"] = "full",
) -> npt.NDArray[np.float32]: ...
