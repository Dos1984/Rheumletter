#!/usr/bin/env python3
"""
Rheumletter scribe — record a consultation, transcribe it locally with Whisper,
and extract findings into the Rheumletter v3 letter fields.

Pipeline
  1. record      microphone -> consult.wav            (ALSA `arecord`, Linux)
  2. transcribe  consult.wav -> transcript.txt        (faster-whisper, runs locally)
  3. extract     transcript.txt -> consultation.json  (local LLM via Ollama by default,
                                                       or the Claude API)
  4. Open Rheumletter v3 > Start > "Import a recorded consultation" and load
     consultation.json. Everything arrives as a suggestion with its supporting quote;
     nothing reaches the letter until the clinician confirms it.

Every extracted item must carry a verbatim quote from the transcript. Items whose quote
cannot be found in the transcript are moved to "rejected" and never shown as suggestions.

Usage
  python3 rheumletter_scribe.py record consult.wav
  python3 rheumletter_scribe.py transcribe consult.wav -o transcript.txt
  python3 rheumletter_scribe.py extract transcript.txt -o consultation.json --backend ollama --model qwen2.5:14b
  python3 rheumletter_scribe.py run consult.wav -o consultation.json          # transcribe + extract

Requirements: see README.md. Test data only until your organisation has approved this
for clinical use (consent, DPIA, data-processing terms).
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FIELDS = HERE / "rheumletter_fields.json"

# Words Whisper often mishears in rheumatology clinics. Passed as an initial prompt.
VOCAB = (
    "Rheumatology clinic consultation. Methotrexate, Metoject, folic acid, sulphasalazine, "
    "hydroxychloroquine, Plaquenil, leflunomide, Arava, prednisolone, Depo-Medrone, lignocaine, "
    "naproxen, Vimovo, Arcoxia, etoricoxib, celecoxib, Celebrex, Difene, Solphadeine, amitriptyline, "
    "duloxetine, pregabalin, gabapentin, allopurinol, febuxostat, colchicine, adalimumab, etanercept, "
    "Quantiferon, anti-CCP, rheumatoid factor, ANA, ENA, ANCA, HLA-B27, CRP, ESR, DEXA, DAS28, "
    "MCP, PIP, DIP, MTP, synovitis, enthesitis, dactylitis, psoriasis, Raynaud's, uveitis, iritis, "
    "Sjögren's, haemochromatosis, osteoarthritis, osteoporosis, spondyloarthritis."
)


# ----------------------------------------------------------------------------------------
# 1. Record
# ----------------------------------------------------------------------------------------
def cmd_record(args: argparse.Namespace) -> None:
    if not shutil.which("arecord"):
        sys.exit("arecord not found. Install it with: sudo apt install alsa-utils")
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", args.out]
    if args.device:
        cmd[1:1] = ["-D", args.device]
    if args.seconds:
        cmd[1:1] = ["-d", str(args.seconds)]
    print(f"Recording to {args.out} (16 kHz mono). Press Ctrl+C to stop.", file=sys.stderr)
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
    print(f"Saved {args.out}", file=sys.stderr)


# ----------------------------------------------------------------------------------------
# 2. Transcribe (local)
# ----------------------------------------------------------------------------------------
def transcribe(audio: str, model_name: str, device: str, language: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed. Run: pip install faster-whisper")
    print(f"Loading Whisper model '{model_name}' on {device}…", file=sys.stderr)
    model = WhisperModel(model_name, device=device, compute_type="default")
    segments, info = model.transcribe(
        audio, language=language, vad_filter=True, beam_size=5, initial_prompt=VOCAB
    )
    lines = []
    for seg in segments:
        mm, ss = divmod(int(seg.start), 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg.text.strip()}")
        print(lines[-1], file=sys.stderr)
    return "\n".join(lines)


def cmd_transcribe(args: argparse.Namespace) -> None:
    text = transcribe(args.audio, args.whisper_model, args.device, args.language)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"Transcript saved to {args.out}", file=sys.stderr)


# ----------------------------------------------------------------------------------------
# 3. Extract
# ----------------------------------------------------------------------------------------
def load_fields(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Field catalogue not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_schema(fields: dict) -> dict:
    """JSON schema for the model output. All properties required, no extra keys
    (works for Claude structured outputs and Ollama `format`)."""
    choice_ids = [c["id"] for c in fields["choices"]]
    mg_ids = [m["id"] for m in fields["management"]]
    joint_ids = [j["id"] for j in fields["joints"]]
    S = lambda d="": {"type": "string", "description": d}
    obj = lambda props: {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return obj({
        "summary": S("2–5 sentence history summary in British English, third person, clinical letter style. Empty string if nothing relevant."),
        "findings": {"type": "array", "items": obj({
            "id": {"type": "string", "enum": choice_ids},
            "option": {"type": "integer", "description": "Index of the matching option in the catalogue"},
            "detail": S("Short extra detail, e.g. 'mother', 'both knees'. Empty if none."),
            "evidence": S("Verbatim quote from the transcript, max 25 words"),
        })},
        "medications": {"type": "array", "items": obj({
            "name": S(), "dose": S(), "frequency": S("e.g. once weekly, twice daily"),
            "route": {"type": "string", "enum": ["oral", "subcutaneous", "intramuscular", "intravenous", "topical", "inhaled"]},
            "status": {"type": "string", "enum": ["current", "previous"]},
            "evidence": S("Verbatim quote"),
        })},
        "joints": obj({
            "tender": {"type": "array", "items": {"type": "string", "enum": joint_ids}},
            "swollen": {"type": "array", "items": {"type": "string", "enum": joint_ids}},
            "evidence": S("Verbatim quote(s) of the clinician's examination findings. Empty if none."),
        }),
        "investigations": obj({"bloods": S(), "radiology": S(), "other": S()}),
        "management": {"type": "array", "items": obj({
            "id": {"type": "string", "enum": mg_ids},
            "detail": S("Region, side, drug or other specifics. Empty if none."),
            "evidence": S("Verbatim quote"),
        })},
        "exam_other": S("Other examination findings stated by the clinician. Empty if none."),
    })


SYSTEM_PROMPT = """You turn a rheumatology outpatient consultation transcript into structured fields for a clinic letter.
Rules:
- Record only what is explicitly said in the transcript. Never infer, guess or add likely findings.
- Choose a negative option (e.g. "No family history of …") only when the question was asked and clearly denied.
- Examination findings and joint lists come only from the clinician's own statements during examination.
- Management items: include only actions the clinician says were done, ordered, prescribed or advised today.
- Every finding, medication and management item needs an "evidence" quote copied word for word from the transcript (max 25 words).
- The transcript may contain speech-recognition errors and speakers are not labelled; if unsure, leave the item out.
- Use British English. Do not include names, dates of birth, addresses or identifiers anywhere in your output.
Return JSON that matches the schema exactly."""


def catalogue_text(fields: dict) -> str:
    out = ["CHOICE ITEMS (id | name | options by index):"]
    for c in fields["choices"]:
        opts = " | ".join(f"{i}: {o}" for i, o in enumerate(c["options"]))
        out.append(f"{c['id']} | {c['name']} | {opts}")
    out.append("\nMANAGEMENT ITEMS (id | template no. | name | default wording):")
    for m in fields["management"]:
        out.append(f"{m['id']} | T{m['template_no']} | {m['name']} | {m['default_wording']}")
    out.append("\nJOINT IDS (patient's side; thumb IP = IP1):")
    out.append(", ".join(f"{j['id']}={j['name']}" for j in fields["joints"]))
    return "\n".join(out)


def call_ollama(system: str, user: str, schema: dict, model: str, host: str) -> dict:
    import requests
    r = requests.post(f"{host.rstrip('/')}/api/chat", timeout=1800, json={
        "model": model, "stream": False, "format": schema,
        "options": {"temperature": 0, "num_ctx": 32768},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def call_claude(system: str, user: str, schema: dict, model: str) -> dict:
    try:
        import anthropic
    except ImportError:
        sys.exit("The anthropic package is not installed. Run: pip install anthropic")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=model, max_tokens=8000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if resp.stop_reason == "max_tokens":
        sys.exit("The response was cut off (max_tokens). Try a shorter transcript.")
    if resp.stop_reason == "refusal":
        sys.exit("The model declined this request.")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


# ---- Verification ----------------------------------------------------------------------
def _norm(s: str) -> str:
    s = re.sub(r"\[\d\d:\d\d\]", " ", s.lower())
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_found(quote: str, transcript_norm: str, threshold: float = 0.82) -> bool:
    q = _norm(quote)
    if len(q) < 3:
        return False
    if q in transcript_norm:
        return True
    words, qn = transcript_norm.split(), len(q.split())
    best = 0.0
    for i in range(0, max(1, len(words) - qn + 1)):
        window = " ".join(words[i:i + qn])
        best = max(best, difflib.SequenceMatcher(None, q, window).ratio())
        if best >= threshold:
            return True
    return False


def verify(raw: dict, fields: dict, transcript: str) -> dict:
    tn = _norm(transcript)
    choices = {c["id"]: c for c in fields["choices"]}
    mg_ids = {m["id"] for m in fields["management"]}
    joint_ids = {j["id"] for j in fields["joints"]}
    rejected, findings, meds, mg = [], [], [], []

    for f in raw.get("findings", []):
        c = choices.get(f.get("id"))
        if not c:
            rejected.append({"kind": "finding", "item": f, "reason": "unknown field id"}); continue
        if not isinstance(f.get("option"), int) or not 0 <= f["option"] < len(c["options"]):
            rejected.append({"kind": "finding", "item": f, "reason": "option index out of range"}); continue
        if not quote_found(f.get("evidence", ""), tn):
            rejected.append({"kind": "finding", "item": f, "reason": "quote not found in transcript"}); continue
        f["section"] = c.get("section") or ("dx" if not c["id"].startswith(("h_", "rom_")) else c["id"].split("_")[0])
        f["sentence"] = c["options"][f["option"]]
        findings.append(f)

    for m in raw.get("medications", []):
        if not m.get("name") or not quote_found(m.get("evidence", ""), tn):
            rejected.append({"kind": "medication", "item": m, "reason": "quote not found in transcript"}); continue
        meds.append(m)

    for m in raw.get("management", []):
        if m.get("id") not in mg_ids:
            rejected.append({"kind": "management", "item": m, "reason": "unknown management id"}); continue
        if not quote_found(m.get("evidence", ""), tn):
            rejected.append({"kind": "management", "item": m, "reason": "quote not found in transcript"}); continue
        mg.append(m)

    j = raw.get("joints") or {}
    tender = [x for x in j.get("tender", []) if x in joint_ids]
    swollen = [x for x in j.get("swollen", []) if x in joint_ids]
    if (tender or swollen) and not quote_found(j.get("evidence", ""), tn):
        rejected.append({"kind": "joints", "item": j, "reason": "quote not found in transcript"})
        tender, swollen = [], []

    return {
        "summary": raw.get("summary", ""),
        "findings": findings, "medications": meds,
        "joints": {"tender": tender, "swollen": swollen, "evidence": j.get("evidence", "")},
        "investigations": raw.get("investigations") or {"bloods": "", "radiology": "", "other": ""},
        "management": mg, "exam_other": raw.get("exam_other", ""), "rejected": rejected,
    }


REDACT = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[email]"),
    (re.compile(r"(?:\+353|\b0)\s?8[35679][\s-]?\d{3}[\s-]?\d{4}\b"), "[phone]"),
    (re.compile(r"\b\d{7}[A-W][A-IW]?\b"), "[PPSN]"),
    (re.compile(r"\b[AC-FHKNPRTV-Y]\d{2}\s?[AC-FHKNPRTV-Y0-9]{4}\b"), "[eircode]"),
    (re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born on)\b[^.\n]{0,30}", re.I), "[date of birth]"),
]


def redact(text: str) -> str:
    for rx, rep in REDACT:
        text = rx.sub(rep, text)
    return text


def extract(transcript: str, args: argparse.Namespace) -> dict:
    fields = load_fields(Path(args.fields))
    schema = build_schema(fields)
    if args.redact:
        transcript = redact(transcript)
    user = f"{catalogue_text(fields)}\n\nTRANSCRIPT:\n{transcript}"
    print(f"Extracting with {args.backend}…", file=sys.stderr)
    if args.backend == "ollama":
        raw = call_ollama(SYSTEM_PROMPT, user, schema, args.model or "qwen2.5:14b", args.ollama_host)
    elif args.backend == "claude":
        if not args.redact:
            print("WARNING: sending the transcript to the Claude API. Use --redact, and only with approved data-processing terms.", file=sys.stderr)
        raw = call_claude(SYSTEM_PROMPT, user, schema, args.model or "claude-sonnet-5-5")
    elif args.backend == "replay":
        raw = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    else:
        sys.exit(f"Unknown backend {args.backend}")
    result = verify(raw, fields, transcript)
    out = {
        "type": "rheumletter-scribe", "v": 1,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend, "model": args.model or "",
        "transcript": "" if args.no_transcript else transcript,
        **result,
    }
    n = len(result["findings"]) + len(result["medications"]) + len(result["management"])
    print(f"{n} items extracted, {len(result['rejected'])} rejected (quote not found or invalid).", file=sys.stderr)
    return out


def cmd_extract(args: argparse.Namespace) -> None:
    transcript = Path(args.transcript).read_text(encoding="utf-8")
    out = extract(transcript, args)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.out}. Import it in Rheumletter v3 > Start > Import a recorded consultation.", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> None:
    transcript = transcribe(args.audio, args.whisper_model, args.device, args.language)
    Path(args.out).with_suffix(".transcript.txt").write_text(transcript + "\n", encoding="utf-8")
    out = extract(transcript, args)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.out}", file=sys.stderr)


# ----------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Record, transcribe and extract a consultation for Rheumletter v3.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="Record from the microphone (Ctrl+C to stop)")
    r.add_argument("out"); r.add_argument("--seconds", type=int); r.add_argument("--device", help="ALSA device, e.g. hw:1,0")
    r.set_defaults(fn=cmd_record)

    def whisper_opts(sp):
        sp.add_argument("--whisper-model", default="large-v3-turbo", help="e.g. large-v3-turbo, large-v3, medium.en, small.en")
        sp.add_argument("--device", default="auto", help="auto, cpu or cuda")
        sp.add_argument("--language", default="en")

    def extract_opts(sp):
        sp.add_argument("-o", "--out", default="consultation.json")
        sp.add_argument("--backend", choices=["ollama", "claude", "replay"], default="ollama")
        sp.add_argument("--model", help="Ollama model (default qwen2.5:14b) or Claude model (default claude-sonnet-5-5)")
        sp.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
        sp.add_argument("--fields", default=str(DEFAULT_FIELDS))
        sp.add_argument("--redact", action="store_true", help="Mask emails, phone numbers, PPSNs, Eircodes and DOB mentions first")
        sp.add_argument("--no-transcript", action="store_true", help="Leave the transcript out of the output file")
        sp.add_argument("--replay", help="(testing) JSON file holding a saved model response")

    t = sub.add_parser("transcribe", help="Transcribe audio locally with Whisper")
    t.add_argument("audio"); t.add_argument("-o", "--out", default="transcript.txt"); whisper_opts(t)
    t.set_defaults(fn=cmd_transcribe)

    e = sub.add_parser("extract", help="Extract letter fields from a transcript")
    e.add_argument("transcript"); extract_opts(e)
    e.set_defaults(fn=cmd_extract)

    a = sub.add_parser("run", help="Transcribe and extract in one step")
    a.add_argument("audio"); whisper_opts(a); extract_opts(a)
    a.set_defaults(fn=cmd_run)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
