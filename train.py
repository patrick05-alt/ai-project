import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf
import mido
import pandas as pd
from tqdm import tqdm

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        # Input: (Batch, 1, 88, 32) -> 88 frequency bins, 32 time frames
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2)) # Output: (16, 44, 16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2)) # Output: (32, 22, 8)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2, 2)) # Output: (64, 11, 4)
        
        # Flatten and FC
        self.fc1 = nn.Linear(64 * 11 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 88)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x)


class MaestroDataset(Dataset):
    def __init__(self, root_dir, year, samples_per_epoch=1000):
        self.root_dir = root_dir
        self.year = str(year)
        self.samples_per_epoch = samples_per_epoch
        self.sample_rate = 22050
        self.context_frames = 32
        self.hop_length = 512
        self.context_samples = self.context_frames * self.hop_length # 16384 samples (~0.743 sec)
        
        csv_path = os.path.join(root_dir, 'maestro-v3.0.0.csv')
        df = pd.read_csv(csv_path)
        
        # Filter by year (the directory usually starts with the year)
        df['year_str'] = df['audio_filename'].apply(lambda x: str(x).split('/')[0])
        self.data_info = df[df['year_str'] == self.year].reset_index(drop=True)
        
        if len(self.data_info) == 0:
            raise ValueError(f"No data found for year/folder: {self.year}")
            
        print(f"Found {len(self.data_info)} tracks for year {self.year}.")
        
        self.midi_cache = {}

    def parse_midi_to_intervals(self, midi_path):
        """Parses MIDI to a list of active note intervals to save memory."""
        mid = mido.MidiFile(midi_path)
        intervals = {pitch: [] for pitch in range(88)}
        active_notes = {}
        current_time = 0.0
        
        for msg in mid:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if 21 <= msg.note <= 108:
                    pitch = msg.note - 21
                    active_notes[pitch] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if 21 <= msg.note <= 108:
                    pitch = msg.note - 21
                    if pitch in active_notes:
                        start_time = active_notes.pop(pitch)
                        intervals[pitch].append((start_time, current_time))
                        
        # Close notes that never got a note_off
        for pitch, start_time in active_notes.items():
            intervals[pitch].append((start_time, current_time))
            
        return intervals

    def get_active_notes_at_time(self, intervals, t):
        """Returns 88-dim binary vector of active notes at time t."""
        active = np.zeros(88, dtype=np.float32)
        for pitch, times in intervals.items():
            for (start, end) in times:
                if start <= t <= end:
                    active[pitch] = 1.0
                    break
        return active

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick a random track
        track_idx = random.randint(0, len(self.data_info) - 1)
        row = self.data_info.iloc[track_idx]
        
        wav_path = os.path.join(self.root_dir, row['audio_filename'])
        midi_path = os.path.join(self.root_dir, row['midi_filename'])
        duration = float(row['duration'])
        
        # Load MIDI intervals if not cached
        if midi_path not in self.midi_cache:
            self.midi_cache[midi_path] = self.parse_midi_to_intervals(midi_path)
            
        intervals = self.midi_cache[midi_path]
        
        # Pick a random time T (at least context_samples/sample_rate into the track)
        min_t = self.context_samples / self.sample_rate
        max_t = duration - 0.1 # slight buffer
        if max_t <= min_t:
            target_t = duration
        else:
            target_t = random.uniform(min_t, max_t)
            
        # Read exact audio chunk using soundfile
        start_frame = int((target_t * self.sample_rate) - self.context_samples)
        start_frame = max(0, start_frame)
        audio_chunk, sr = sf.read(wav_path, frames=self.context_samples, start=start_frame, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.mean(axis=1)
            
        # Resample if needed
        if sr != self.sample_rate:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sample_rate)
            
        # Pad if chunk is too small
        if len(audio_chunk) < self.context_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.context_samples - len(audio_chunk)))
        else:
            audio_chunk = audio_chunk[:self.context_samples]
            
        # Compute CQT
        cqt = np.abs(librosa.cqt(audio_chunk, sr=self.sample_rate, 
                                 fmin=librosa.note_to_hz('A0'), n_bins=88, bins_per_octave=12))
        
        # We want exactly 32 frames. 16384 samples with hop 512 -> 33 frames, we take first 32.
        cqt = cqt[:, :32]
        if cqt.shape[1] < 32:
            cqt = np.pad(cqt, ((0, 0), (0, 32 - cqt.shape[1])))
            
        cqt = cqt.reshape(1, 88, 32).astype(np.float32)
        
        # Get active notes at time target_t
        active_notes = self.get_active_notes_at_time(intervals, target_t)
        
        return torch.tensor(cqt), torch.tensor(active_notes)


def export_onnx(model, device, output_path="transcriber_quantized.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 1, 88, 32).to(device)
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"\n[+] ONNX model successfully exported to {output_path}")

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Training on device: {device}")
    
    root_dir = os.path.join(os.path.dirname(__file__), 'maestro-v3.0.0')
    
    dataset = MaestroDataset(root_dir=root_dir, year=args.year, samples_per_epoch=args.samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = CRNN().to(device)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), "model_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"\n[*] Found existing checkpoint. Loading weights to accumulate progress...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"\n[*] No existing checkpoint found. Starting fresh training...")
        
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch} Average Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Training Complete!")
    
    # Save PyTorch checkpoint for future training sessions
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[+] PyTorch checkpoint saved to {checkpoint_path} for future resuming.")
    
    # Export model for the FastAPI server
    export_onnx(model, device)


def interactive_menu(args, root_dir):
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("="*50)
        print(" MAESTRO DATASET TRAINING MENU ")
        print("="*50)
        
        available_years = []
        if os.path.exists(root_dir):
            for item in sorted(os.listdir(root_dir)):
                if os.path.isdir(os.path.join(root_dir, item)) and item.isdigit():
                    available_years.append(item)
                    
        if not available_years:
            print(f"No valid dataset subfolders found in {root_dir}.")
            sys.exit(1)
            
        print("Available Dataset Subfolders:")
        for i, year in enumerate(available_years):
            print(f"  [{i+1}] {year}")
            
        print("\n  [Q] Quit")
        print("="*50)
        
        choice = input("Select a folder number to train on: ").strip().lower()
        if choice == 'q':
            sys.exit(0)
            
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_years):
                args.year = available_years[choice_idx]
                break
            else:
                input("Invalid selection. Press Enter to try again...")
        except ValueError:
            input("Invalid selection. Press Enter to try again...")
            
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAESTRO CRNN Audio-to-MIDI Trainer")
    parser.add_argument('--year', type=str, required=False, help="Subfolder/Year to process (e.g., '2004'). If omitted, an interactive menu will appear.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size (8 or 16 recommended for laptop)")
    parser.add_argument('--samples-per-epoch', type=int, default=2000, help="Random 0.74s samples to crop per epoch")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help="Device to train on (cuda or cpu)")
    
    args = parser.parse_args()
    
    if not args.year:
        root_dir = os.path.join(os.path.dirname(__file__), 'maestro-v3.0.0')
        args = interactive_menu(args, root_dir)
        
    train(args)
