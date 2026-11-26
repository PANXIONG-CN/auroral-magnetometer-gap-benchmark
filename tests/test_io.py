"""Offline tests for the raw-data layer: write-once storage, idempotent downloads, parsers."""

import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
from helpers import DAY, image_text

from p1.io import dst, http, image, kp, manifest


class FakeFetch:
    def __init__(self, responses):
        self.responses = responses  # url prefix -> bytes or Exception
        self.calls = []

    def __call__(self, url, params=None):
        self.calls.append((url, params))
        for prefix, resp in self.responses.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp, url
        raise AssertionError(f"unexpected URL {url}")


def raw_files(root):
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.csv"
    )


def test_store_raw_writes_once(tmp_path):
    p = manifest.store_raw(
        "a/b.txt", b"abc", source="s", product="p", version="v", url="u", root=tmp_path
    )
    assert p.read_bytes() == b"abc"
    assert not os.access(p, os.W_OK)
    rows = manifest.read_manifest(tmp_path)
    assert (
        len(rows) == 1
        and rows[0]["sha256"] == manifest.sha256_bytes(b"abc")
        and rows[0]["bytes"] == "3"
    )
    with pytest.raises(FileExistsError):
        manifest.store_raw(
            "a/b.txt",
            b"xyz",
            source="s",
            product="p",
            version="v",
            url="u",
            root=tmp_path,
        )
    assert p.read_bytes() == b"abc"
    assert len(manifest.read_manifest(tmp_path)) == 1
    assert manifest.verify(tmp_path) == []


def test_verify_reports_unlisted_and_changed_files(tmp_path):
    p = manifest.store_raw(
        "a.txt", b"abc", source="s", product="p", version="v", url="u", root=tmp_path
    )
    (tmp_path / "stray.txt").write_bytes(b"x")
    p.chmod(0o644)
    p.write_bytes(b"abd")
    problems = manifest.verify(tmp_path)
    assert (
        "not in MANIFEST: stray.txt" in problems and "hash mismatch: a.txt" in problems
    )


def test_image_download_is_idempotent(tmp_path):
    fetch = FakeFetch({image.DATA_URL: image_text()})
    path, new = image.download_day(DAY, ["SOD", "KEV"], root=tmp_path, fetch=fetch)
    assert new and path.name == "image_20170801_KEV_SOD.txt"
    for _ in range(2):
        _, new = image.download_day(DAY, ["KEV", "SOD"], root=tmp_path, fetch=fetch)
        assert not new
    counts = image.download_days(
        [DAY, DAY], ["KEV", "SOD"], root=tmp_path, fetch=fetch, delay_s=0, log=None
    )
    assert counts == {"downloaded": 0, "cached": 1, "failed_days": []}
    assert len(fetch.calls) == 1
    assert (
        fetch.calls[0][1]["starttime"] == "2017080100"
        and fetch.calls[0][1]["length"] == 1440
    )
    assert raw_files(tmp_path) == ["image/text_60s/2017/image_20170801_KEV_SOD.txt"]
    assert len(manifest.read_manifest(tmp_path)) == 1


def test_image_failed_day_is_skipped_and_refetched(tmp_path):
    day2 = "2017-08-02"
    calls = []

    def flaky(url, params=None):
        calls.append(params["starttime"])
        if params["starttime"] == "2017080100" and len(calls) == 1:
            raise TimeoutError("handshake timed out")
        return image_text(day=pd.Timestamp(params["starttime"][:8])), url

    kw = {"root": tmp_path, "fetch": flaky, "delay_s": 0, "log": None}
    counts = image.download_days([DAY, day2], ["KEV", "SOD"], **kw)
    assert counts["downloaded"] == 1 and counts["failed_days"] == [dt.date(2017, 8, 1)]
    assert raw_files(tmp_path) == ["image/text_60s/2017/image_20170802_KEV_SOD.txt"]
    counts = image.download_days([DAY, day2], ["KEV", "SOD"], **kw)
    assert counts == {"downloaded": 1, "cached": 1, "failed_days": []}
    assert calls == ["2017080100", "2017080200", "2017080100"]
    assert len(manifest.read_manifest(tmp_path)) == 2


def test_image_rejects_incomplete_day(tmp_path):
    fetch = FakeFetch({image.DATA_URL: image_text(nrows=100)})
    with pytest.raises(ValueError):
        image.download_day(DAY, ["KEV", "SOD"], root=tmp_path, fetch=fetch)
    assert raw_files(tmp_path) == [] and manifest.read_manifest(tmp_path) == []


def test_image_rejects_html_error_page(tmp_path):
    fetch = FakeFetch({image.DATA_URL: b"<html><body>Error</body></html>"})
    with pytest.raises(ValueError):
        image.download_day(DAY, ["KEV"], root=tmp_path, fetch=fetch)
    assert raw_files(tmp_path) == []


def test_image_absent_station_is_noted(tmp_path):
    fetch = FakeFetch({image.DATA_URL: image_text()})
    image.download_day(DAY, ["KEV", "PEL", "SOD"], root=tmp_path, fetch=fetch)
    assert manifest.read_manifest(tmp_path)[0]["note"] == "absent from file: PEL"
    v = image.read_range(
        DAY, "2017-08-02", ["KEV", "PEL", "SOD"], validity_only=True, root=tmp_path
    )
    assert not v["PEL"].any().any() and v["KEV"].all().all()


def test_image_missing_values_and_validity(tmp_path):
    missing = {("SOD", 0), ("SOD", 1), ("KEV", 1439)}
    fetch = FakeFetch({image.DATA_URL: image_text(missing=missing)})
    image.download_day(DAY, ["KEV", "SOD"], root=tmp_path, fetch=fetch)
    df = image.read_range(
        "2017-08-01 00:00", "2017-08-01 00:03", ["KEV", "SOD"], root=tmp_path
    )
    assert list(df.index) == list(pd.date_range(DAY, periods=3, freq="min"))
    assert np.isnan(df[("SOD", "X")].iloc[0]) and np.isnan(df[("SOD", "Y")].iloc[1])
    assert df[("KEV", "X")].iloc[2] == pytest.approx(10000.2)
    v = image.read_range(
        DAY, "2017-08-02", ["KEV", "SOD"], validity_only=True, root=tmp_path
    )
    assert v.dtypes.eq(bool).all()
    assert int((~v[("SOD", "X")]).sum()) == 2 and int((~v[("KEV", "Z")]).sum()) == 1


def test_image_read_range_needs_cached_days(tmp_path):
    with pytest.raises(FileNotFoundError):
        image.read_range(DAY, "2017-08-02", ["KEV"], root=tmp_path)


STATION_PAGE = """<table><tr><th colspan=11>Geographical and CGM coordinates</th></tr>
<tr><th></th><th>Code</th><th>Name</th><th>lat</th><th>lon</th><th>cgm lat</th><th>cgm lon</th>
<th>Prov</th><th>Net</th><th>Start</th><th>End</th></tr>
<tr><td>1</td><td>SOD</td><td>Sodankyl&auml;</td><td>67.37</td><td>26.63</td><td>63.92</td><td>107.26</td>
<td>SGO</td><td>I</td><td>1996-01-01</td><td></td></tr>
<tr><td>2</td><td>LRV</td><td>Leirvogur</td><td>64.18</td><td>338.3</td><td>64.97</td><td>67.02</td>
<td>UI</td><td>I</td><td>2022-12-01</td><td></td></tr>
<tr><td>3</td><td>ALT</td><td>Alta</td><td>69.86</td><td>22.96</td><td>66.63</td><td>106.21</td>
<td>FMI</td><td>E</td><td>1982-10-01</td><td>1991-08-30</td></tr></table>"""


def test_station_table_parse():
    t = image.parse_station_table(STATION_PAGE)
    assert list(t.index) == ["SOD", "LRV", "ALT"]
    assert t.loc["SOD", "name"] == "Sodankylä" and t.loc["SOD", "lat"] == 67.37
    assert t.loc["LRV", "lon"] == pytest.approx(-21.7)
    assert pd.isna(t.loc["SOD", "end"]) and t.loc["ALT", "end"] == pd.Timestamp(
        "1991-08-30"
    )


DST_PAGE = """<html><pre class="data">
                                Hourly Equatorial Dst Values (PROVISIONAL)
      unit=nT                                                                                      UT
      1   2   3   4   5   6   7   8    9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24
DAY
 1  -19 -26 -21  -7  -5  -3  -5  -4   -2   0  -3  -4  -6  -5  -4   1    3   6   1   5  12  11  13  10
10   12  10  12  21  16   7   9  16   13  15  13   3  -1   1  10  23   25  66 -33-131-157-277-339-308
11 -313-390-406-395-389-358-320-374 -364-297-318-268-258-279-281-265 -270-248-203-198-190-169-157-167
</pre></html>"""


def test_dst_fixed_width_parse():
    s = dst.parse_month(DST_PAGE, 2024, 5)
    assert len(s) == 31 * 24
    assert s[pd.Timestamp("2024-05-01 00:00")] == -19
    day10 = s["2024-05-10"].to_numpy()
    assert list(day10[16:]) == [25, 66, -33, -131, -157, -277, -339, -308]
    assert s.idxmin() == pd.Timestamp("2024-05-11 02:00") and s.min() == -406
    assert s["2024-05-02"].isna().all()


def test_dst_download_falls_back_and_caches(tmp_path):
    fetch = FakeFetch(
        {
            dst.month_url("final", 2024, 5): http.NotFound("final"),
            dst.month_url("provisional", 2024, 5): DST_PAGE.encode(),
        }
    )
    today = dt.date(2026, 11, 15)
    product, path, new = dst.download_month(
        2024, 5, root=tmp_path, fetch=fetch, today=today
    )
    assert (product, path.name, new) == (
        "provisional",
        "dst_202405_dl20261115.html",
        True,
    )
    product, path2, new = dst.download_month(
        2024, 5, root=tmp_path, fetch=fetch, today=dt.date(2026, 12, 1)
    )
    assert (product, path2, new) == ("provisional", path, False)
    assert len(fetch.calls) == 2 and len(manifest.read_manifest(tmp_path)) == 1


KP_TEXT = """# header line
#YYY MM DD  days  days_m  Bsr dB    Kp1    Kp2    Kp3    Kp4    Kp5    Kp6    Kp7    Kp8  ap1  ap2  ap3  ap4  ap5  ap6  ap7  ap8    Ap  SN F10.7obs F10.7adj D
2015 03 16 30390 30390.5 2477 25  1.667  3.333  3.667  3.000  2.667  2.667  1.333  0.333    6   18   22   15   12   12    5    2    12  46    117.2    116.0 2
2015 03 17 30391 30391.5 2477 26  2.000  4.667  5.667  5.333  7.667  7.667  7.333 -1.000    7   39   67   56  179  179  154   -1   108  38    114.3    113.2 0
"""


def test_kp_parse():
    three, daily = kp.parse(KP_TEXT)
    assert len(three) == 16
    assert three.index[0] == pd.Timestamp("2015-03-16 00:00") and three.index[
        9
    ] == pd.Timestamp("2015-03-17 03:00")
    assert three["kp"].iloc[9] == pytest.approx(4.667) and np.isnan(
        three["kp"].iloc[15]
    )
    assert three["definitive"].iloc[0] and not three["definitive"].iloc[8]
    assert (
        daily.loc["2015-03-17", "kp_max"] == pytest.approx(7.667)
        and daily.loc["2015-03-16", "D"] == 2
    )


def test_kp_download_reuses_cached_copy(tmp_path):
    fetch = FakeFetch({kp.URL: KP_TEXT.encode()})
    p1, new1 = kp.download(root=tmp_path, fetch=fetch, today=dt.date(2026, 11, 15))
    p2, new2 = kp.download(root=tmp_path, fetch=fetch, today=dt.date(2026, 11, 16))
    assert new1 and not new2 and p1 == p2 and len(fetch.calls) == 1
    p3, new3 = kp.download(
        refresh=True, root=tmp_path, fetch=fetch, today=dt.date(2026, 11, 16)
    )
    assert (
        new3 and p3 != p1 and p1.exists() and len(manifest.read_manifest(tmp_path)) == 2
    )


@pytest.mark.parametrize(
    "value, thirds",
    [
        ("2+", 7),
        ("5-", 14),
        ("5o", 15),
        ("5", 15),
        (5, 15),
        (2.333, 7),
        (4.667, 14),
        (0.0, 0),
    ],
)
def test_kp_thirds(value, thirds):
    assert kp.kp_thirds(value) == thirds
