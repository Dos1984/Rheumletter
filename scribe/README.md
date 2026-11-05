# Rheumletter scribe

Record a consultation on a Linux machine, transcribe it locally with Whisper, and turn
the transcript into suggested entries for the Rheumletter v3 letter. The clinician
reviews and confirms every suggestion in the app before it reaches the letter.

```
microphone ─► consult.wav ─► Whisper (local) ─► transcript.txt
                                                  │
                     rheumletter_fields.json ─────┤  (the letter's fields, exported from the app)
                                                  ▼
                              language model: Ollama (local, default) or Claude API
                                                  │  JSON matching a strict schema
                                                  ▼
                       quote check: every item must quote the transcript word for word
                                                  ▼
                                         consultation.json
                                                  ▼
             Rheumletter v3 › Start › "Import a recorded consultation"
             dashed suggestions with their quotes › clinician confirms › letter
```

It doesn't matter whether the app was built with ChatGPT or Claude. The two connect
through two plain files:

* `rheumletter_fields.json`: the list of fields the letter has (export it from the app
  under Local settings › Export field list).
* `consultation.json`: what the script found, in a fixed format the app can import.

## Files

| File | Purpose |
|---|---|
| `rheumletter_scribe.py` | The script: `record`, `transcribe`, `extract`, `run` |
| `rheumletter_fields.json` | Field catalogue exported from Rheumletter v3 |
| `examples/sample_transcript.txt` | A fictional consultation for testing |
| `examples/sample_model_response.json` | A saved model answer, including one deliberately invented finding |
| `examples/sample_consultation.json` | The verified output; the invented finding has been rejected |

## Setup (Ubuntu or Debian)

```bash
sudo apt install python3-venv alsa-utils ffmpeg
python3 -m venv ~/scribe-env && source ~/scribe-env/bin/activate
pip install faster-whisper requests          # add: pip install anthropic   (only for the Claude API)

# Local language model, so nothing leaves the machine:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b                       # needs about 10 GB; larger models are more accurate
```

The first Whisper run downloads its model (about 1.6 GB for `large-v3-turbo`) and keeps it.

## Use

```bash
# 1. Record (Ctrl+C to stop). List microphones with: arecord -l
python3 rheumletter_scribe.py record consult.wav

# 2 + 3. Transcribe and extract in one go (all local)
python3 rheumletter_scribe.py run consult.wav -o consultation.json

# or step by step
python3 rheumletter_scribe.py transcribe consult.wav -o transcript.txt
python3 rheumletter_scribe.py extract transcript.txt -o consultation.json --backend ollama --model qwen2.5:14b

# Claude API instead of a local model (only with approved data-processing terms)
export ANTHROPIC_API_KEY=...
python3 rheumletter_scribe.py extract transcript.txt -o consultation.json --backend claude --redact

# Try the whole flow without audio or a model:
python3 rheumletter_scribe.py extract examples/sample_transcript.txt -o test.json \
        --backend replay --replay examples/sample_model_response.json
```

Then open Rheumletter v3, go to **Start › Import a recorded consultation** and load `consultation.json`.

## What the app does with it

* History, family history, diagnosis-list and range-of-motion answers appear as dashed
  suggestions, labelled **from recording**, with the quote underneath.
* Medications appear as suggested rows to confirm.
* Joints the clinician called out show as dashed rings on the joint map, with an **Add these joints** button.
* Management items are outlined with their quote. Tick each one, or accept them all, then
  check the wording. Fields such as the review interval and X-ray regions are pre-set where the
  detail allows.
* The summary, investigation results and other examination findings are offered as text to insert.
* **Nothing reaches the letter until you confirm it.** The transcript can be viewed on the Start step.

## Hardware and accuracy

* CPU only works, but a 15-minute consultation can take several minutes with `large-v3-turbo`.
  An NVIDIA GPU (`--device cuda`) is much faster. `small.en` is quicker but less accurate.
* Use a good table or lapel microphone and keep the room quiet. Whisper is given a list of
  rheumatology drug names and terms to improve recognition.
* Speakers are not labelled. If you need "doctor vs patient" separation, add WhisperX with
  pyannote speaker diarisation (it needs a Hugging Face token and a GPU).
* Small local models miss more and invent more than large ones. The quote check removes
  items whose quote isn't in the transcript, but it can't catch a real quote that has been
  misinterpreted. The clinician review is the safety step.

## Before any real patient use

* **Consent:** tell patients the consultation is being recorded and why, and record their consent.
* **Data protection:** health data is special-category data under GDPR. Do a DPIA with your
  hospital's Data Protection Officer, and set how long audio and transcripts are kept
  (ideally deleted once the letter is signed).
* **Where processing happens:** with Whisper plus Ollama, audio and text stay on that machine.
  Sending transcripts to any cloud model (Claude API, ChatGPT/OpenAI) needs an approved
  data-processing agreement. `--redact` only masks obvious identifiers (emails, phone numbers,
  PPSNs, Eircodes, date-of-birth mentions). It cannot reliably remove names.
* **Regulation:** check with your organisation whether an ambient scribe used for clinical
  documentation falls under medical device or AI regulation in your setting.
* Keep the machine encrypted and the recordings out of shared or synced folders.
