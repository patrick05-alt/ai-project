import json
import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Optional ONNX model ---
onnx_session = None
USE_ONNX = False  # Set to True after proper training

try:
    import onnxruntime as ort
    onnx_session = ort.InferenceSession("transcriber_quantized.onnx")
    print("[+] ONNX model loaded.")
except Exception as e:
    print(f"[!] ONNX model not loaded ({e}). Using librosa pitch detection.")


def hz_to_midi(hz: float) -> int:
    return int(round(12 * np.log2(hz / 440.0) + 69))


# ---- Per-connection state (reset on each new WebSocket session) ----
class NoteTracker:
    """
    Strict 3-state machine for reliable monophonic pitch detection:

        SILENCE  →  ATTACK  →  LOCKED
           ↑                      |
           └──────────────────────┘

    Key rule: a new note can ONLY be detected from SILENCE state.
    This completely prevents mid-decay harmonic jumps from registering.
    """

    # Thresholds
    SILENCE_RMS   = 0.006   # Below this → silence
    ATTACK_RMS    = 0.012   # Must exceed this to start ATTACK phase
    LOCK_FRAMES   = 4       # Stable frames needed to lock a note
    HOLD_FRAMES   = 6       # Min frames to stay LOCKED before returning to SILENCE

    # States
    ST_SILENCE = 'silence'
    ST_ATTACK  = 'attack'
    ST_LOCKED  = 'locked'

    def __init__(self):
        self.state        = self.ST_SILENCE
        self.locked_midi  = None
        self.candidates   = []    # MIDI values gathered during ATTACK
        self.hold_count   = 0     # frames spent in LOCKED
        self.prev_rms     = 0.0

    def _fft_peak_midi(self, audio: np.ndarray, sr: int):
        """Return the MIDI note of the strongest FFT peak, or None."""
        n_fft = 4096
        win   = audio * np.hanning(len(audio))
        spec  = np.abs(np.fft.rfft(win, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Restrict to instrument range
        lo = int(27   * n_fft / sr)
        hi = int(4200 * n_fft / sr)
        spec[:lo] = 0
        spec[hi:] = 0

        if spec.max() < 1e-6:
            return None

        # Parabolic interpolation for sub-bin accuracy
        pk = int(np.argmax(spec))
        if 1 <= pk < len(spec) - 1:
            a, b, c = spec[pk-1], spec[pk], spec[pk+1]
            denom = a - 2*b + c
            offset = 0.5 * (a - c) / denom if denom != 0 else 0
            hz = (pk + offset) * sr / n_fft
        else:
            hz = freqs[pk]

        if hz < 27:
            return None

        midi = int(round(12 * np.log2(hz / 440.0) + 69))
        return midi if 21 <= midi <= 108 else None

    def process(self, audio: np.ndarray, sr: int) -> dict:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        self.prev_rms = rms

        # ── SILENCE ──────────────────────────────────────────────────────
        if self.state == self.ST_SILENCE:
            if rms >= self.ATTACK_RMS:
                # Transition to ATTACK
                self.state      = self.ST_ATTACK
                self.candidates = []
                midi = self._fft_peak_midi(audio, sr)
                if midi:
                    self.candidates.append(midi)
            return {"midi": None, "velocity": 0, "onset": False}

        # ── ATTACK ───────────────────────────────────────────────────────
        if self.state == self.ST_ATTACK:
            if rms < self.SILENCE_RMS:
                # Note faded before we could lock — back to silence
                self.state = self.ST_SILENCE
                return {"midi": None, "velocity": 0, "onset": False}

            midi = self._fft_peak_midi(audio, sr)
            if midi:
                self.candidates.append(midi)

            if len(self.candidates) >= self.LOCK_FRAMES:
                # Check stability: all candidates within ±1 semitone
                spread = max(self.candidates) - min(self.candidates)
                if spread <= 2:
                    self.locked_midi = int(round(
                        sorted(self.candidates)[len(self.candidates)//2]  # median
                    ))
                    self.state      = self.ST_LOCKED
                    self.hold_count = 0
                    print(f"[LOCK] midi={self.locked_midi} rms={rms:.3f}")
                    return {
                        "midi":     self.locked_midi,
                        "velocity": min(127, int(rms * 800)),
                        "onset":    True,
                    }
                else:
                    # Unstable — reset candidates, keep in ATTACK
                    self.candidates = self.candidates[-2:]

            return {"midi": None, "velocity": 0, "onset": False}

        # ── LOCKED ───────────────────────────────────────────────────────
        if self.state == self.ST_LOCKED:
            self.hold_count += 1

            if rms < self.SILENCE_RMS and self.hold_count >= self.HOLD_FRAMES:
                # Note has ended — go back to silence
                self.state       = self.ST_SILENCE
                self.locked_midi = None
                self.candidates  = []
                return {"midi": None, "velocity": 0, "onset": False}

            # Keep reporting the locked note through the decay
            return {
                "midi":     self.locked_midi,
                "velocity": min(127, int(rms * 800)),
                "onset":    False,
            }

        # Fallback
        self.state = self.ST_SILENCE
        return {"midi": None, "velocity": 0, "onset": False}


def detect_notes_onnx(audio: np.ndarray, sr: int) -> list:
    """Use the ONNX model to detect notes (only when well-trained)."""
    cqt = np.abs(librosa.cqt(audio, sr=sr, fmin=librosa.note_to_hz('A0'), n_bins=88, bins_per_octave=12))
    feature = cqt[:, :32]
    if feature.shape[1] < 32:
        feature = np.pad(feature, ((0, 0), (0, 32 - feature.shape[1])))
    feature = feature.reshape(1, 1, 88, 32).astype(np.float32)
    inputs = {onnx_session.get_inputs()[0].name: feature}
    probs = onnx_session.run(None, inputs)[0][0]
    max_p = float(np.max(probs))
    active = []
    if max_p >= 0.25:
        for i, p in enumerate(probs):
            if p >= max_p * 0.70:
                active.append({"midi": i + 21, "velocity": int(p * 127)})
    return active


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    sr      = 22050
    buf     = np.zeros(sr, dtype=np.float32)    # 1-second rolling buffer
    tracker = NoteTracker()                      # fresh tracker per client

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" not in message:
                if "text" in message:
                    print(f"[WARN] Got text, expected bytes: {message['text'][:40]}")
                continue

            try:
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
            except Exception as e:
                print(f"[ERROR] bytes parse: {e}")
                continue

            buf = np.roll(buf, -len(chunk))
            buf[-len(chunk):] = chunk

            try:
                if USE_ONNX and onnx_session:
                    raw = detect_notes_onnx(buf, sr)
                    notes  = raw
                    onset  = False
                else:
                    result = tracker.process(buf, sr)
                    midi   = result["midi"]
                    vel    = result["velocity"]
                    onset  = result["onset"]
                    notes  = [{"midi": midi, "velocity": vel}] if midi is not None else []
            except Exception as e:
                import traceback
                print(f"[ERROR] detection: {e}")
                traceback.print_exc()
                notes = []
                onset = False

            payload = {"type": "notes", "notes": notes, "onset": onset}
            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
