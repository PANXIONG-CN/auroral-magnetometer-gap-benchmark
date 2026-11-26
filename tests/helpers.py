"""Shared test helpers (tests/ is on sys.path under pytest's default import mode)."""

import pandas as pd

DAY = "2017-08-01"


def image_text(day=DAY, stations=("KEV", "SOD"), missing=(), nrows=1440):
    """Synthetic IMAGE `text` file; `missing` holds (station, minute) pairs written as 99999.9."""
    head = "YYYY MM DD HH MM SS   " + " ".join(
        f"{s} {c}" for s in stations for c in "XYZ"
    )
    lines = [head, "-" * len(head)]
    t0 = pd.Timestamp(day)
    for i in range(nrows):
        t = t0 + pd.Timedelta(minutes=i)
        vals = [
            "99999.9" if (s, i) in missing else f"{10000 + i / 10:.1f}"
            for s in stations
            for _ in "XYZ"
        ]
        lines.append(t.strftime("%Y %m %d %H %M %S") + "    " + " ".join(vals))
    return ("\n".join(lines) + "\n").encode("latin-1")


def store_day(root, day, stations=("KEV", "SOD")):
    """Write a synthetic day file into a temporary raw root, as the downloader would."""
    from p1.io import image, manifest

    manifest.store_raw(
        image.day_relpath(day, stations),
        image_text(day=day, stations=stations),
        source="test",
        product="test",
        version="test",
        url="test",
        root=root,
    )
