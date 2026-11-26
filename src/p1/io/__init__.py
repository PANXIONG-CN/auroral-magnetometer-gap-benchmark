"""Data access: downloaders write to data/raw/ and record each file in data/raw/MANIFEST.csv.

Modules:
    manifest  write-once raw storage and the MANIFEST
    http      HTTP GET with retries
    image     IMAGE 1-min magnetometer data and station table (FMI)
    kp        Kp, ap, Ap (GFZ Potsdam)
    dst       hourly Dst (WDC for Geomagnetism, Kyoto)

The locked-event guard (T2) is added to the readers here.
"""
