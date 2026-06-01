# AudioSpread: Real-Time Audio-to-Sheet-Music Architecture

**AudioSpread** is a lightweight, real-time machine learning pipeline that transcribes live audio (e.g., ukulele or piano playing via microphone) directly into standard Western sheet music notation in your browser.

The architecture is optimized to run inference locally on standard consumer hardware (Windows/macOS/Linux) without relying on heavy cloud compute.

## 🏗️ Architecture

The project is split into three main components:

1. **Inference Backend (`server.py`)**: A high-performance FastAPI server running a custom `NoteTracker` state machine. It accepts binary audio streams over WebSockets from the browser and uses FFT (Fast Fourier Transform) with parabolic interpolation and a strict Silence → Attack → Locked state machine to extract fundamental pitches with high accuracy and echo-suppression. (An optional ONNX-based CRNN backend is also included for testing against the MAESTRO dataset).
2. **Frontend UI (`transcriber-ui`)**: A React-based web application with a premium "glassmorphism" design. It captures microphone audio natively in the browser using the modern `AudioWorkletNode` API, streams it to the server, and uses `vexflow` to render the incoming notes onto an infinite-scrolling horizontal musical stave in real-time. It includes a live VU meter and an active note badge.
3. **Training Pipeline (`train.py`)**: An optional, memory-efficient PyTorch implementation that trains a Convolutional Recurrent Neural Network (CRNN) on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro).

---

## 💻 Prerequisites (Windows Native)

Audio capture operates directly within the browser, meaning **WSL is no longer required.** You can run the entire pipeline natively in Windows PowerShell.

- **OS:** Windows 10/11 (PowerShell) or Linux/macOS.
- **Software:** Python 3.10+, Node.js v18+.

---

## 🚀 Setup Guide

### 1. Automated PowerShell Setup
Open Windows PowerShell, navigate to the project directory, and run the automated setup script. This will create your virtual environment and install all Python dependencies:

```powershell
.\setup.ps1
```

If you encounter an Execution Policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

### 2. Frontend Setup
Install the dependencies for the React user interface:

```powershell
cd transcriber-ui
npm install
```

---

## 🎤 Running the Application

To start the real-time pipeline, you need two PowerShell windows.

**Terminal 1: Backend Server**
```powershell
# Inside the music-transcriber directory
.\venv\Scripts\Activate.ps1
uvicorn server:app --port 8000 --reload
```

**Terminal 2: Frontend Visualizer**
```powershell
# Inside the transcriber-ui directory
npm start
```

Navigate to `http://localhost:3000` in your browser. Click "Start Microphone," play an instrument near your mic, and watch the sheet music render in real-time on the scrolling horizontal stave!

---

## 🧠 Optional: Training the Neural Network

The repository includes a script to train a CRNN on the MAESTRO dataset if you wish to swap out the FFT pitch detector for a machine-learning approach.

Ensure the `maestro-v3.0.0` dataset is in the root directory. To launch the interactive menu:

```bash
python train.py
```

Or run it directly via CLI options:

```bash
python train.py --year 2004 --epochs 10 --batch-size 16 --device cuda
```

*Note: The script exports `transcriber_quantized.onnx` at the end of the run. To use it in `server.py`, change `USE_ONNX = True` at the top of the file.*
