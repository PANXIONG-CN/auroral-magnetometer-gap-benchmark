"""P1: reconstruction of missing auroral-zone ground magnetometer data.

Planned modules (built in docs/04_task_plan.md order):
    io        downloaders and loaders for IMAGE, Kp, CHAOS; locked-event guard
    reference reference decomposition: main field + station bias + quiet variation
    secs      divergence-free SECS fitting and evaluation
    baselines inverse-distance weighting, ridge regression
    temporal  persistence, pre-gap bias correction, linear state transition
    scenarios scenario generation and hidden-station separation
    metrics   rmse, peak error, peak timing, dB/dt
    stats     event-level bootstrap
"""

__version__ = "0.1.0"
