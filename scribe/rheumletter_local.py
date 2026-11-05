#!/usr/bin/env python3
"""
Rheumletter local helper — connects the Rheumletter v3 app to Whisper and Qwen3 on this computer.

  * Serves the app at http://127.0.0.1:8765 (this computer only).
  * POST /api/transcribe  audio from the app's Record button -> faster-whisper (local) -> transcript
  * POST /api/extract     transcript -> Qwen3 via Ollama (local) -> letter-field suggestions
  * GET  /api/status      what is available

Audio is written to a temporary file only while it is transcribed, then deleted.
Nothing is sent off this computer.

Start:  double-click start-rheumletter.bat   (or: python rheumletter_local.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rheumletter_scribe as rs  # noqa: E402  (shared prompt, schema and quote checking)

SERVED = {"/index-v3.html", "/patient-v3.html"}
MAX_AUDIO = 400 * 1024 * 1024
AUDIO_EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/x-wav": ".wav",
             "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "video/webm": ".webm"}

ARGS: argparse.Namespace
_whisper = None
_whisper_lock = threading.Lock()


# ---------------------------------------------------------------- Whisper (local)
def whisper_model():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            last = None
            for ctype in dict.fromkeys([ARGS.compute_type, "default"]):  # fall back if int8 isn't supported
                for local_only in (True, False):  # prefer the model already on this computer
                    try:
                        _whisper = WhisperModel(ARGS.whisper_model, device=ARGS.device, compute_type=ctype,
                                                local_files_only=local_only)
                        return _whisper
                    except Exception as e:
                        last = e
            raise RuntimeError(f"Could not load Whisper model '{ARGS.whisper_model}': {last}")
        return _whisper


def transcribe_file(path: str) -> str:
    model = whisper_model()
    segments, _info = model.transcribe(path, language="en", vad_filter=True, beam_size=5,
                                       initial_prompt=rs.VOCAB)
    out = []
    for seg in segments:
        mm, ss = divmod(int(seg.start), 60)
        out.append(f"[{mm:02d}:{ss:02d}] {seg.text.strip()}")
    return "\n".join(out)


# ---------------------------------------------------------------- Qwen3 via Ollama (local)
def ollama_models() -> list[str]:
    r = requests.get(f"{ARGS.ollama_host}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def chosen_model(models: list[str]) -> str | None:
    if ARGS.model:
        return ARGS.model
    qwen = [m for m in models if m.lower().startswith("qwen3")]
    return qwen[0] if qwen else (models[0] if models else None)


def extract(transcript: str) -> dict:
    fields = rs.load_fields(Path(ARGS.fields))
    schema = rs.build_schema(fields)
    model = chosen_model(ollama_models())
    if not model:
        raise RuntimeError("No Ollama models found. Run: ollama pull qwen3")
    body = {
        "model": model, "stream": False, "format": schema, "think": False,
        "options": {"temperature": 0, "num_ctx": ARGS.ctx},
        "messages": [{"role": "system", "content": rs.SYSTEM_PROMPT},
                     {"role": "user", "content": f"{rs.catalogue_text(fields)}\n\nTRANSCRIPT:\n{transcript}"}],
    }
    r = requests.post(f"{ARGS.ollama_host}/api/chat", json=body, timeout=3600)
    if r.status_code == 400 and "think" in r.text.lower():  # older Ollama without the think option
        body.pop("think")
        r = requests.post(f"{ARGS.ollama_host}/api/chat", json=body, timeout=3600)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    raw = json.loads(content)
    result = rs.verify(raw, fields, transcript)
    return {"type": "rheumletter-scribe", "v": 1,
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "backend": "ollama", "model": model, "whisper_model": ARGS.whisper_model,
            "transcript": transcript, **result}


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, fmt, *args):  # keep the console free of clinical content
        sys.stderr.write(f"{self.log_date_time_string()}  {self.command} {urlparse(self.path).path}\n")

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _same_origin(self) -> bool:
        # Only pages served by this helper may use the API (blocks other websites).
        origin = self.headers.get("Origin")
        return origin is None or origin in (f"http://127.0.0.1:{ARGS.port}", f"http://localhost:{ARGS.port}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_response(302); self.send_header("Location", "/index-v3.html"); self.end_headers(); return
        if path == "/api/status":
            st = {"ok": True, "whisper": {"model": ARGS.whisper_model, "loaded": _whisper is not None},
                  "ollama": {"reachable": False, "models": [], "model": None}}
            try:
                ms = ollama_models()
                st["ollama"] = {"reachable": True, "models": ms, "model": chosen_model(ms)}
            except Exception as e:
                st["ollama"]["error"] = str(e)
            return self._json(200, st)
        if path not in SERVED:
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if not self._same_origin():
            return self._json(403, {"error": "forbidden origin"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            if path == "/api/transcribe":
                if not 0 < length <= MAX_AUDIO:
                    return self._json(413, {"error": "Recording is empty or too large."})
                ctype = (self.headers.get("Content-Type") or "audio/webm").split(";")[0].strip()
                data = self.rfile.read(length)
                fd, tmp = tempfile.mkstemp(suffix=AUDIO_EXT.get(ctype, ".webm"))
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
                    t0 = time.time()
                    text = transcribe_file(tmp)
                    return self._json(200, {"transcript": text, "seconds": round(time.time() - t0, 1)})
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            if path == "/api/extract":
                payload = json.loads(self.rfile.read(length) or b"{}")
                transcript = str(payload.get("transcript", "")).strip()
                if not transcript:
                    return self._json(400, {"error": "The transcript is empty."})
                t0 = time.time()
                out = extract(transcript)
                out["seconds"] = round(time.time() - t0, 1)
                return self._json(200, out)
            return self._json(404, {"error": "not found"})
        except requests.ConnectionError:
            return self._json(503, {"error": "Ollama is not running. Start Ollama from the Start menu and try again."})
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    global ARGS
    p = argparse.ArgumentParser(description="Rheumletter local helper (Whisper + Qwen3 via Ollama)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--whisper-model", default="small", help="faster-whisper model (small, medium, large-v3-turbo…)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--model", help="Ollama model; default: the first qwen3 model installed")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--ctx", type=int, default=16384, help="Context window for the model")
    p.add_argument("--fields", default=str(HERE / "rheumletter_fields.json"))
    p.add_argument("--no-browser", action="store_true")
    ARGS = p.parse_args()
    if not ARGS.ollama_host.startswith("http"):
        ARGS.ollama_host = "http://" + ARGS.ollama_host
    ARGS.ollama_host = ARGS.ollama_host.rstrip("/")

    for f in ("index-v3.html", "rheumletter_fields.json"):
        if not (HERE / f).exists():
            sys.exit(f"Missing {f} next to this script.")
    try:
        ms = ollama_models()
        print(f"Ollama: OK. Using model: {chosen_model(ms)}  (installed: {', '.join(ms) or 'none'})")
    except Exception:
        print("Ollama: not reachable yet. Start Ollama from the Start menu; the app will retry.")
    print(f"Whisper: faster-whisper '{ARGS.whisper_model}' on {ARGS.device} (loads on first recording)")

    url = f"http://127.0.0.1:{ARGS.port}/index-v3.html"
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"\nRheumletter is running at {url}\nKeep this window open while you use it. Close it to stop.\n")
    if not ARGS.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
