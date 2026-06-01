import React, { useEffect, useRef, useState, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { Renderer, Stave, StaveNote, Formatter, Accidental } from 'vexflow';
import './App.css';

const WS_URL = 'ws://localhost:8000/ws';

// ── MIDI helpers ──────────────────────────────────────────────────────────────
const NOTE_NAMES = ['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];
const VEX_NAMES  = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b'];

const midiToLabel  = (m) => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
const midiToOctave = (m) => Math.floor(m / 12) - 1;
const midiToVex    = (m) => {
  const name = VEX_NAMES[m % 12];
  return {
    key:        `${name}/${midiToOctave(m)}`,
    accidental: name.includes('#') ? '#' : null,
  };
};
const msToVexDur = (ms) =>
  ms >= 1500 ? 'w' : ms >= 750 ? 'h' : ms >= 200 ? 'q' : '8';

// ── VexFlow renderer ─────────────────────────────────────────────────────────────
const NOTE_W   = 70;   // px per note slot
const STAVE_H  = 160;  // height of stave area
const LEFT_PAD = 100;  // space for clef + time sig

const drawStave = (container, history, currentMidi) => {
  if (!container) return;
  container.innerHTML = '';

  // Build the full note list
  const items = [...history];
  if (currentMidi !== null) {
    const v = midiToVex(currentMidi);
    items.push({
      keys:        [v.key],
      duration:    'q',
      accidentals: v.accidental ? [{ index: 0, type: v.accidental }] : [],
      isRest:      false,
      isCurrent:   true,
    });
  }

  // Total width grows with number of notes
  const totalW = LEFT_PAD + Math.max(items.length, 4) * NOTE_W + 40;

  const renderer = new Renderer(container, Renderer.Backends.SVG);
  renderer.resize(totalW, STAVE_H);
  const ctx = renderer.getContext();
  ctx.setFont('Arial', 10);

  const stave = new Stave(10, 20, totalW - 20);
  stave.addClef('treble').addTimeSignature('4/4');
  stave.setContext(ctx).draw();

  if (items.length === 0) return;

  try {
    const staveNotes = items.map((item) => {
      const dur = item.isRest ? item.duration + 'r' : item.duration;
      const sn  = new StaveNote({ keys: item.keys, duration: dur, clef: 'treble' });
      if (!item.isRest) {
        item.accidentals.forEach(({ index, type }) =>
          sn.addModifier(new Accidental(type), index)
        );
      }
      if (item.isCurrent) sn.setStyle({ fillStyle: '#92FE9D', strokeStyle: '#92FE9D' });
      return sn;
    });
    Formatter.FormatAndDraw(ctx, stave, staveNotes);
  } catch (err) {
    console.warn('[VexFlow]', err.message);
  }

  // Scroll right so the latest note is always visible
  container.scrollLeft = container.scrollWidth;
};


// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentMidi, setCurrentMidi] = useState(null);
  const [noteHistory, setNoteHistory] = useState([]);
  const [rmsLevel,    setRmsLevel]    = useState(0);    // 0-1 for VU meter

  const containerRef  = useRef(null);
  const audioCtxRef   = useRef(null);
  const workletRef    = useRef(null);
  const streamRef     = useRef(null);
  const wsRef         = useRef(null);

  const prevMidiRef   = useRef(null);
  const noteStartRef  = useRef(Date.now());
  const lastHistMidi  = useRef(null);   // for echo suppression

  // ── WebSocket ────────────────────────────────────────────────────────────────
  const { lastMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: () => true,
    onOpen: (evt) => { wsRef.current = evt.target; },
  });

  const isConnected = readyState === ReadyState.OPEN;

  // ── Commit completed note to history ────────────────────────────────────────
  const commitNote = useCallback((midi, elapsedMs) => {
    if (midi === null || elapsedMs < 400) return;
    // Echo suppression: skip if same note as last history entry
    if (midi === lastHistMidi.current) return;

    const v    = midiToVex(midi);
    const item = {
      keys:        [v.key],
      duration:    msToVexDur(elapsedMs),
      accidentals: v.accidental ? [{index:0, type:v.accidental}] : [],
      isRest:      false,
    };
    lastHistMidi.current = midi;
    setNoteHistory(prev => [...prev, item].slice(-32));  // keep last 32 notes
  }, []);

  // ── Handle server message ────────────────────────────────────────────────────
  useEffect(() => {
    if (!lastMessage) return;
    let data;
    try { data = JSON.parse(lastMessage.data); } catch { return; }
    if (data.type !== 'notes') return;

    const notes  = data.notes  ?? [];
    const onset  = data.onset  ?? false;
    const newMidi = notes.length > 0 ? notes[0].midi : null;

    // Update VU bar (velocity → 0-1)
    const vel = notes.length > 0 ? notes[0].velocity : 0;
    setRmsLevel(vel / 127);

    const prevMidi = prevMidiRef.current;

    // Onset = server confirmed a new pluck of the SAME note
    const noteChanged = newMidi !== prevMidi;
    const newPluck    = onset && newMidi !== null;

    if (noteChanged || newPluck) {
      // Commit whatever was playing
      const elapsed = Date.now() - noteStartRef.current;
      if (noteChanged) commitNote(prevMidi, elapsed);

      prevMidiRef.current  = newMidi;
      noteStartRef.current = Date.now();
      setCurrentMidi(newMidi);
    }
  }, [lastMessage, commitNote]);

  // ── Redraw stave ─────────────────────────────────────────────────────────────
  useEffect(() => {
    drawStave(containerRef.current, noteHistory, currentMidi);
  }, [noteHistory, currentMidi]);

  // Also redraw on resize so the stave width stays correct
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      drawStave(el, noteHistory, currentMidi);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [noteHistory, currentMidi]);

  // ── Microphone ────────────────────────────────────────────────────────────────
  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 22050 });
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);

      const send = (f32) => {
        if (wsRef.current?.readyState === WebSocket.OPEN)
          wsRef.current.send(f32.buffer instanceof ArrayBuffer ? f32.buffer : new Float32Array(f32).buffer);
      };

      let node;
      try {
        await ctx.audioWorklet.addModule('/audio-processor.js');
        const w = new AudioWorkletNode(ctx, 'audio-processor');
        w.port.onmessage = ({ data }) => send(data);
        node = w;
      } catch {
        const sp = ctx.createScriptProcessor(4096, 1, 1);
        sp.onaudioprocess = (ev) => send(new Float32Array(ev.inputBuffer.getChannelData(0)));
        node = sp;
      }

      workletRef.current = node;
      source.connect(node);
      node.connect(ctx.destination);

      prevMidiRef.current  = null;
      noteStartRef.current = Date.now();
      lastHistMidi.current = null;
      setNoteHistory([]);
      setCurrentMidi(null);
      setIsRecording(true);
    } catch (err) {
      alert('Microphone error: ' + err.message);
    }
  };

  const stopMicrophone = () => {
    const elapsed = Date.now() - noteStartRef.current;
    commitNote(prevMidiRef.current, elapsed);
    setCurrentMidi(null);
    setRmsLevel(0);
    workletRef.current?.disconnect();
    audioCtxRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    setIsRecording(false);
  };

  // ── UI ────────────────────────────────────────────────────────────────────────
  const noteName   = currentMidi !== null ? NOTE_NAMES[currentMidi % 12] : null;
  const noteOctave = currentMidi !== null ? midiToOctave(currentMidi)    : null;
  const noteLabel  = currentMidi !== null ? midiToLabel(currentMidi)     : null;

  return (
    <div className="App">
      <div className="glass-container">
        <h1 className="title">
          AudioSpread <span className="subtitle">Real-Time Transcription</span>
        </h1>

        {/* Status row */}
        <div className="status-row">
          <span className="conn-dot" style={{ color: isConnected ? '#92FE9D' : '#FF416C' }}>
            ● {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {/* Controls + Live Note Badge in one row */}
        <div className="controls-note-row">
          {/* Mic button */}
          {!isRecording
            ? <button className="mic-button start" onClick={startMicrophone}>
                🎙 Start Microphone
              </button>
            : <button className="mic-button stop" onClick={stopMicrophone}>
                ⏹ Stop
              </button>
          }

          {/* Live note badge */}
          <div className={`note-badge ${currentMidi !== null ? 'note-badge--active' : ''}`}>
            {currentMidi !== null ? (
              <>
                <span className="note-badge__name">{noteName}</span>
                <span className="note-badge__octave">{noteOctave}</span>
              </>
            ) : (
              <span className="note-badge__idle">—</span>
            )}
          </div>

          {/* VU bar */}
          <div className="vu-track">
            <div className="vu-fill" style={{ height: `${Math.round(rmsLevel * 100)}%` }} />
          </div>
        </div>

        {/* Note label full text */}
        {noteLabel && (
          <p className="note-label-full">{noteLabel}</p>
        )}

        {/* Sheet music */}
        <div ref={containerRef} className="vexflow-wrapper" />
      </div>
    </div>
  );
}
