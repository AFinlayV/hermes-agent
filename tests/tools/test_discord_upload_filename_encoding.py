"""Discord standalone uploads must hand Discord each filename exactly once.

Field report (2026-11-17 card t_018ff412): the 2026-11-18 music-day composition
MP3 landed on the recipient's disk as
``2026-11-1820The20Occupants20Leave2C20the20Walls20Ring.mp3`` — the on-disk name
``2026-11-18 The Occupants Leave, the Walls Ring.mp3`` with its spaces and comma
percent-encoded. Root cause: the plugin's standalone uploader built a raw
``aiohttp.FormData()``, whose RFC 7578 default (``quote_fields=True``)
percent-encodes the multipart ``filename`` (space -> ``%20``, comma -> ``%2C``),
and Discord stores that quoted form verbatim as the download name. Discord's own
client library never quotes (``discord/http.py`` builds
``aiohttp.FormData(quote_fields=False)``) because Discord treats the multipart
``filename`` as the attachment name verbatim. The tests below snapshot the
actual wire form at the upload call site's POST and assert the filename bytes
the Discord API receives, plus that the audio bytes ride unmodified.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plugins.platforms.discord.adapter import (
    _remember_channel_is_forum,
    _standalone_send,
)

#: The exact human-readable name Alex reported mangled (card t_018ff412).
GOOD = "2026-11-18 The Occupants Leave, the Walls Ring.mp3"
#: The compose contract's audio payload marker (test fixture bytes).
PAYLOAD = b"ID3" + b"\x00" * 32


def _resp(status, json_data=None):
    r = AsyncMock()
    r.status = status
    body = json.dumps(json_data or {}).encode()
    r.json = AsyncMock(return_value=json_data or {})
    r.text = AsyncMock(return_value="")
    r.content = MagicMock()
    r.content.read = AsyncMock(side_effect=[body, b"", b""])
    r.get_encoding = MagicMock(return_value="utf-8")
    return r


def _render_parts(form):
    """(binary headers, value) per multipart part, exactly as the POST transmits.

    Rendered at POST-call time, while the call site's file handles are still
    open — a re-render after the send would only see closed handles.
    """
    parts = []
    for payload, _enc, _te in form()._parts:
        value = payload._value
        if hasattr(value, "read"):
            value.seek(0)
            value = value.read()
        parts.append((bytes(payload._binary_headers), value))
    return parts


def _session_with(responses):
    """Mocked aiohttp.ClientSession recording every POST: (url, json, form, parts)."""
    calls = []
    idx = [0]

    def _post(url, **kwargs):
        form = kwargs.get("data")
        parts = _render_parts(form) if form is not None else None
        calls.append((url, kwargs.get("json"), form, parts))
        r = responses[idx[0]] if idx[0] < len(responses) else responses[-1]
        idx[0] += 1
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=r)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = MagicMock()
    session.post = MagicMock(side_effect=_post)
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx, calls


def _file_parts(call):
    return [(h, v) for h, v in call[3] if b'name="files[' in h]


def _payload_json_of(call):
    for h, v in call[3]:
        if b'name="payload_json"' not in h:
            continue
        return json.loads(v)
    return None


def _pconfig():
    return SimpleNamespace(token="bot-token", extra={})


def _named_file(tmp_path, name, payload=PAYLOAD):
    p = tmp_path / name
    p.write_bytes(payload)
    return str(p)


def test_normal_channel_upload_filename_is_encoded_once(tmp_path):
    """The music-day surface: non-forum POST /channels/{id}/messages multipart.

    The regression asserts on the form the upload call site builds and hands to
    the Discord client (plugins/platforms/discord/adapter.py `_standalone_send`,
    the `files[0]` add_field before session.post) — not on a helper.
    """
    chat_id = "1566982719139876926"
    _remember_channel_is_forum(chat_id, False)  # avoid the live GET probe
    mp3 = _named_file(tmp_path, GOOD)
    try:
        session_ctx, calls = _session_with([_resp(200, {"id": "m1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(), chat_id, "",
                    media_files=[(mp3, False)],
                    caption="the piece",
                )
            )
        assert res["success"] is True
        call = calls[0]
        assert call[0].endswith("/messages")
        parts = _file_parts(call)
        assert len(parts) == 1, f"expected exactly one file part, got {parts!r}"
        cd, value = parts[0]
        # Literal spaces and comma reach the client — encoded exactly once.
        assert b'filename="' + GOOD.encode() + b'"' in cd, (
            f"filename not literal on the wire: {cd!r}")
        assert b"%20" not in cd and b"%2C" not in cd, (
            f"percent-encoded name still reaches the client: {cd!r}")
        # Audio bytes are untouched: the part carries the file's bytes verbatim.
        assert value == PAYLOAD, f"audio bytes were altered: {value!r}"
        # The caption still rides payload_json — content is not the fix surface.
        payload = _payload_json_of(call)
        assert payload is not None and payload.get("content") == "the piece"
    finally:
        os.unlink(mp3)


def test_forum_thread_upload_filenames_encoded_once_per_part(tmp_path):
    """Forum channels: the thread-creation multipart (payload_json + files[N]).

    Two files prove per-index handling — a fix that only cleaned files[0] would
    leave files[1] encoded twice.
    """
    chat_id = "1566982719139876927"
    _remember_channel_is_forum(chat_id, True)
    second = "2026-11-18 Score and Parts, annotated.json"
    mp3 = _named_file(tmp_path, GOOD)
    score = _named_file(tmp_path, second, payload=b"{}")
    try:
        session_ctx, calls = _session_with([_resp(200, {"id": "t1"})])
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            res = asyncio.run(
                _standalone_send(
                    _pconfig(), chat_id, "",
                    media_files=[(mp3, False), (score, False)],
                    caption="the piece",
                )
            )
        assert res["success"] is True
        call = calls[0]
        assert call[0].endswith("/threads")

        # The starter-message attachment meta names each file verbatim.
        payload = _payload_json_of(call)
        assert payload is not None, f"no payload_json part in form: {call!r}"
        meta = payload["message"]["attachments"]
        assert [m["filename"] for m in meta] == [GOOD, second]

        # And every file part carries the literal name on the wire.
        parts = _file_parts(call)
        assert len(parts) == 2, f"expected two file parts, got {len(parts)}"
        for (cd, value), name, expected in zip(
            parts, (GOOD, second), (PAYLOAD, b"{}")
        ):
            assert b'filename="' + name.encode() + b'"' in cd, (
                f"filename not literal on the wire: {cd!r}")
            assert b"%20" not in cd and b"%2C" not in cd, (
                f"percent-encoded name still reaches the client: {cd!r}")
            # Audio bytes are untouched per part, two surfaces at once.
            assert value == expected, f"bytes altered for {name!r}: {value!r}"
    finally:
        os.unlink(mp3)
        os.unlink(score)
