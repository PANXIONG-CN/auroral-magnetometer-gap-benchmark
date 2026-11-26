"""HTTP GET with retries. Downloaders take a `fetch` argument so tests run offline."""

from __future__ import annotations

import time

import requests

USER_AGENT = "p1-polar-missing-station/0.1 (scientific data download)"


class NotFound(Exception):
    """The server answered 404."""


def fetch(
    url: str,
    params: dict | None = None,
    *,
    timeout: float = 120,
    retries: int = 3,
    backoff_s: float = 5.0,
) -> tuple[bytes, str]:
    """GET url and return (body, final URL). Raises NotFound on 404, retries other failures."""
    error: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if r.status_code == 404:
                raise NotFound(r.url)
            r.raise_for_status()
            return r.content, r.url
        except NotFound:
            raise
        except requests.RequestException as e:
            error = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (attempt + 1))
    assert error is not None
    raise error
