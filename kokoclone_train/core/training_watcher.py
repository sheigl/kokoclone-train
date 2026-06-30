"""Drop-in training queue and directory watcher."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from kokoclone_train.core.rvc_train import train_voice


class TrainingWatcher:
    """Watches the training/ directory for new voices to train.

    Users drop audio into training/{voice_name}/audio.wav and the system
    automatically queues and trains the voice. Only one voice trains at a time.
    """

    def __init__(
        self,
        training_dir: str = "training",
        voices_dir: str = "voices",
        progress_callback: Optional[Callable] = None,
    ):
        self.training_dir = Path(training_dir)
        self.voices_dir = Path(voices_dir)
        self.progress_callback = progress_callback

        self._queue: list[str] = []
        self._training: Set[str] = set()
        self._trained: Set[str] = set()
        self._partial: Dict[str, dict] = {}  # voice_name -> state info
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the training watcher in a background thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop the training watcher."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def scan_new_voices(self) -> list[str]:
        """Scan training/ directory for new voices to train.

        Returns list of newly discovered voice names.
        """
        new_voices = []
        if not self.training_dir.exists():
            return new_voices

        for voice_dir in sorted(self.training_dir.iterdir()):
            if not voice_dir.is_dir():
                continue

            voice_name = voice_dir.name

            with self._lock:
                # Skip if already training or trained
                if voice_name in self._training or voice_name in self._trained:
                    continue

            # Check for audio files
            audio_files = list(voice_dir.glob("*.wav")) + \
                         list(voice_dir.glob("*.mp3")) + \
                         list(voice_dir.glob("*.flac"))

            if audio_files:
                new_voices.append(voice_name)
                with self._lock:
                    self._queue.append(voice_name)
                    self._training.add(voice_name)

        return new_voices

    def get_queue_status(self) -> dict:
        """Get current training queue status."""
        with self._lock:
            return {
                "queued": [v for v in self._queue if v not in self._trained],
                "training": list(self._training),
                "trained": list(self._trained),
                "partial": dict(self._partial),
            }

    def scan_partial_states(self) -> None:
        """Scan training_temp/ for partial training states."""
        temp_dir = Path("training_temp")
        if not temp_dir.exists():
            return

        for voice_dir in sorted(temp_dir.iterdir()):
            if not voice_dir.is_dir():
                continue

            state_path = voice_dir / "state.json"
            if not state_path.exists():
                continue

            try:
                with open(state_path) as f:
                    state_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            voice_name = voice_dir.name
            can_resume = False
            resume_info = {}

            if state_data.get("stage") == "training":
                current_epoch = state_data.get("current_epoch", 0)
                best_loss = state_data.get("best_loss", None)
                ckpt_file = voice_dir / f"{voice_name}_epoch_{current_epoch}.pth"
                can_resume = ckpt_file.exists()

                if can_resume:
                    resume_info = {
                        "current_epoch": current_epoch,
                        "best_loss": best_loss,
                        "checkpoint": str(ckpt_file),
                    }

            with self._lock:
                self._partial[voice_name] = {
                    "stage": state_data.get("stage", "unknown"),
                    "can_resume": can_resume,
                    **resume_info,
                }

    def _watch_loop(self):
        """Background loop that scans for new voices and trains them."""
        while not self._stop_event.is_set():
            try:
                new = self.scan_new_voices()
                if new:
                    print(f"[TrainingWatcher] Found {len(new)} new voice(s) to train")

                # Process queue (one at a time)
                with self._lock:
                    current = None
                    for v in list(self._queue):
                        if v not in self._trained and v in self._training:
                            current = v
                            break

                if current:
                    self._train_one(current)

            except Exception as e:
                print(f"[TrainingWatcher] Error: {e}")

            self._stop_event.wait(5)  # Check every 5 seconds

    def _train_one(self, voice_name: str):
        """Train a single voice from the queue."""
        audio_dir = self.training_dir / voice_name

        try:
            print(f"[TrainingWatcher] Starting training for '{voice_name}'")

            def progress_callback(msg, percent):
                if self.progress_callback:
                    self.progress_callback(voice_name, msg, percent)

            result = train_voice(
                voice_name=voice_name,
                audio_dir=audio_dir,
                output_dir=self.voices_dir,
                callback=progress_callback,
            )

            print(f"[TrainingWatcher] Training complete for '{voice_name}'")

            # Delete training audio after successful training
            if audio_dir.exists():
                import shutil
                shutil.rmtree(audio_dir)

            with self._lock:
                self._trained.add(voice_name)
                self._queue.remove(voice_name)
                self._training.discard(voice_name)
                self._partial.pop(voice_name, None)

        except Exception as e:
            print(f"[TrainingWatcher] Training failed for '{voice_name}': {e}")
            with self._lock:
                self._training.discard(voice_name)
                # Check if partial state exists for resume capability
                temp_dir = Path("training_temp") / voice_name
                state_path = temp_dir / "state.json"
                if state_path.exists():
                    try:
                        with open(state_path) as f:
                            state_data = json.load(f)
                        self._partial[voice_name] = {
                            "stage": state_data.get("stage", "unknown"),
                            "can_resume": True,
                            "error": str(e),
                        }
                    except (json.JSONDecodeError, IOError):
                        pass
