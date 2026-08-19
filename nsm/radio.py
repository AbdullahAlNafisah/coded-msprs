"""The contract an SDR backend must satisfy.

Experiments and DSP talk to this, never to a vendor driver. Adding a radio
means writing one class with these seven methods; nothing in ``scripts/`` or
``nsm/sdr_calibrate.py`` changes.

``nsm.sdr.OTALink`` is the ADALM-Pluto implementation.

Amplitude convention. Samples handed to :meth:`Radio.transmit_receive` already
span the backend's full-scale range, which the backend owns and publishes as
``full_scale``. Callers scale through ``nsm.sdr.build_baseband``; nothing
outside a backend hardcodes a full-scale constant.

Gain convention. ``set_tx_gain_db`` and ``set_rx_gain_db`` take whatever dB
quantity the radio actually implements: on the AD9363 that is TX attenuation
(negative, less power) and RX amplifier gain. A radio with no analog gain
would implement these as digital scaling. The numbers are therefore NOT
comparable across backends, and a calibration measured on one radio does not
transfer to another.

Deliberately absent, because it does not generalise: RSSI, chip temperature,
calibration mode, firmware and hardware version. Those are AD9363 and libiio
specific, are used only by ``scripts/ota_testbed.py``, and stay on the
concrete backend.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Radio(Protocol):
    """One RF link: send a burst, capture it, control gain."""

    full_scale: float

    def transmit_receive(self, samples: np.ndarray, *,
                         overhead: int = 5) -> np.ndarray:
        """Transmit ``samples`` and return the captured receive buffer.

        ``samples`` must already be full-scale-ed. The returned capture holds
        at least one complete burst; framing and sync locate it.
        """

    def receive(self, n_samples: int) -> np.ndarray:
        """Capture ``n_samples`` with no transmit side-effects, stale data flushed."""

    def set_tx_gain_db(self, gain_db: float) -> None: ...

    def set_rx_gain_db(self, gain_db: float) -> None: ...

    def get_rx_gain_db(self) -> float:
        """Current RX gain. Under AGC this is the value the radio chose."""

    def set_gain_mode(self, mode: str) -> None:
        """Select receive gain control. Backends without AGC accept only "manual"."""

    def close(self) -> None: ...
