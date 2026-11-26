"""CHAOS main field at stations. Needs the cached model files; skipped without them."""

import math

import numpy as np
import pandas as pd
import pytest

from p1 import geo, mainfield
from p1.io import chaos, image, manifest


def cached(version):
    return (manifest.raw_root() / chaos.relpath(version)).exists()


needs_chaos = pytest.mark.skipif(
    not cached("CHAOS-8.6"), reason="CHAOS-8.6 not downloaded"
)


def test_mjd2000():
    assert mainfield.mjd2000(["2000-01-01", "2000-01-02 12:00"]).tolist() == [0.0, 1.5]


@needs_chaos
def test_model_matches_observed_field_at_sod():
    """Core + lithospheric field at SOD (an absolute observatory) against the IMAGE value on
    2017-08-01 00:00: inclination within 0.1 degree, declination within 0.5 degree. A sign or
    component swap fails this; the intensity differs by about 1 % (local crust), so components
    are not compared. The geodetic rotation is checked separately below."""
    st = ["SOD"]
    coords = geo.station_coords(st)
    model = chaos.load("CHAOS-8.6")
    t = pd.Timestamp("2017-08-01")
    m = (
        mainfield.core_xyz(model, coords, [t])["SOD"].iloc[0]
        + mainfield.static_xyz(model, coords).loc["SOD"]
    )
    files = [
        "ABK",
        "AND",
        "IVA",
        "KEV",
        "KIL",
        "KIR",
        "MAS",
        "MUO",
        "OUJ",
        "PEL",
        "SOD",
        "TRO",
    ]
    if not (manifest.raw_root() / image.day_relpath(t, files)).exists():
        pytest.skip("IMAGE day not downloaded")
    obs = image.read_range(t, t + pd.Timedelta(minutes=1), files)["SOD"].iloc[0]
    assert obs["X"] == pytest.approx(11251.1) and obs["Y"] == pytest.approx(2364.6)

    def decl(v):
        return math.degrees(math.atan2(v["Y"], v["X"]))

    def incl(v):
        return math.degrees(math.atan2(v["Z"], math.hypot(v["X"], v["Y"])))

    assert incl(m) == pytest.approx(incl(obs), abs=0.1)
    assert decl(m) == pytest.approx(decl(obs), abs=0.5)


@needs_chaos
def test_geodetic_rotation_matches_wgs84():
    """X and Z are the geocentric components turned by the difference between geodetic and
    geocentric latitude, tan(psi) = (1 - e^2) tan(phi) on the WGS84 ellipsoid; Y is unchanged."""
    from chaosmagpy import coordinate_utils as cu

    lat, lon = 67.37, 26.63
    model = chaos.load("CHAOS-8.6")
    coords = pd.DataFrame({"lat": [lat], "lon": [lon]}, index=["S"])
    gd = mainfield.core_xyz(model, coords, ["2017-08-01"])["S"].iloc[0]
    r, th = cu.gg_to_geo(np.zeros(1), np.array([90.0 - lat]))
    br, bt, bp = model.synth_values_tdep(
        mainfield.mjd2000(["2017-08-01"]), r, th, np.array([lon])
    )
    turn = math.degrees(math.atan2(gd["Z"], gd["X"]) - math.atan2(-br[0], -bt[0]))
    psi = math.degrees(math.atan((1 - 0.00669437999014) * math.tan(math.radians(lat))))
    assert abs(turn) == pytest.approx(lat - psi, abs=1e-4)
    assert gd["Y"] == pytest.approx(bp[0])


@needs_chaos
def test_secular_change_2017_to_2018_at_sod():
    """The core-field change at SOD over the year matches the eastward drift seen at ten stations
    in E004 (Y up 45-60 nT, X down 10-22 nT)."""
    coords = geo.station_coords(["SOD"])
    core = mainfield.core_xyz(
        chaos.load("CHAOS-8.6"), coords, ["2017-08-04", "2018-08-15"]
    )["SOD"]
    d = core.iloc[1] - core.iloc[0]
    assert 35 < d["Y"] < 65 and -30 < d["X"] < 0


@pytest.mark.skipif(not cached("CHAOS-6-x6"), reason="CHAOS-6-x6 not downloaded")
@needs_chaos
def test_causal_release_agrees_with_latest_within_a_few_nt():
    coords = geo.station_coords(["SOD", "PEL", "MUO"])
    times = ["2017-08-04", "2018-08-15"]
    d_new = mainfield.core_xyz(chaos.load("CHAOS-8.6"), coords, times).diff().iloc[1]
    d_old = mainfield.core_xyz(chaos.load("CHAOS-6-x6"), coords, times).diff().iloc[1]
    assert np.abs(d_new - d_old).max() < 5.0
    assert chaos.is_causal("CHAOS-6-x6", "2018-08-25") and not chaos.is_causal(
        "CHAOS-8.6", "2018-08-25"
    )


def test_igrf_causal_version():
    from p1.io import igrf

    assert igrf.causal_version("2015-03-17") == "IGRF-12"
    assert igrf.causal_version("2019-12-31 12:00") == "IGRF-13"
    assert igrf.causal_version("2024-10-10") == "IGRF-13"
    with pytest.raises(ValueError):
        igrf.causal_version("2014-06-01")


@pytest.mark.skipif(
    not cached("CHAOS-8.6")
    or not (manifest.raw_root() / "igrf" / "IGRF-13.txt").exists(),
    reason="CHAOS-8.6 or IGRF-13 not downloaded",
)
def test_igrf13_change_agrees_with_chaos():
    """2017-08 to 2018-08 core change in X and Y (the components P1 uses) from IGRF-13, whose
    2015 and 2020 epochs are definitive, within 4 nT of CHAOS-8.6 (measured: 2.2 and 3.0 nT)."""
    from p1.io import igrf

    coords = geo.station_coords(["SOD", "PEL", "MUO", "AND"])
    t = [pd.Timestamp("2017-08-05T12:00"), pd.Timestamp("2018-08-27T12:00")]
    c = mainfield.core_xyz(chaos.load("CHAOS-8.6"), coords, t)
    g = mainfield.core_xyz(igrf.load("IGRF-13"), coords, t)
    d = (g.iloc[1] - g.iloc[0]) - (c.iloc[1] - c.iloc[0])
    assert d[d.index.get_level_values("component") != "Z"].abs().max() < 4
