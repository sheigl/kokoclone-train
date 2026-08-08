"""Whisper transcription client — sends audio chunks to a remote whisper server."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence

logger = logging.getLogger("kokoclone_train.transcribe_client")


def _post_audio(
    url: str,
    wav_path: Path,
    language: Optional[str] = None,
    timeout: int = 300,
) -> dict:
    """Send a single audio file to the whisper server."""
    import urllib.error
    import urllib.request

    data = _encode_multipart(
        {
            "audio": (wav_path.name, wav_path.read_bytes(), "audio/wav"),
        },
        {"language": language} if language else {},
    )

    req = urllib.request.Request(
        url,
        data=data["body"],
        headers={"Content-Type": f'multipart/form-data; boundary="{data["boundary"]}"'},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Server error {e.code}: {body}") from e


# Module-level worker functions for multiprocessing (spawn context requires picklable callables)
_worker_endpoint = ""
_worker_language: Optional[str] = None
_worker_retries = 3
_worker_perm_dir: Path = Path(".")


def _mp_worker_init(endpoint: str, language: Optional[str], max_retries: int, perm_dir_str: str):
    """Initialize worker process globals."""
    global _worker_endpoint, _worker_language, _worker_retries, _worker_perm_dir
    _worker_endpoint = endpoint
    _worker_language = language
    _worker_retries = max_retries
    _worker_perm_dir = Path(perm_dir_str)


def _mp_worker_task(wav_path_str: str) -> dict:
    """Pickle-safe wrapper for _transcribe_single."""
    return _transcribe_single(
        Path(wav_path_str), _worker_endpoint, _worker_language, _worker_retries, _worker_perm_dir, None
    )


def _transcribe_single(
    wav_path: Path,
    endpoint: str,
    language: Optional[str],
    max_retries: int,
    perm_dir: Path,
    callback: Optional[callable],
) -> dict:
    """Worker: transcribe a single WAV file.

    Must be module-level so ThreadPoolExecutor can reference it cleanly.

    Returns dict with keys: stem, success, text, error_msg
    """
    from tqdm import tqdm

    txt_path = wav_path.with_suffix(".txt")
    perm_txt_path = perm_dir / f"{wav_path.stem}.txt"

    # Skip if already exists (check again inside worker in case another thread finished)
    if txt_path.exists() or perm_txt_path.exists():
        return {"stem": wav_path.stem, "success": True, "text": "", "skipped": True}

    for attempt in range(1, max_retries + 1):
        try:
            result = _post_audio(endpoint, wav_path, language)
            text = result.get("text", "").strip()
            txt_path.write_text(text)
            perm_txt_path.write_text(text)
            return {"stem": wav_path.stem, "success": True, "text": text}
        except Exception as e:
            logger.warning(
                "[%s] Attempt %d failed: %s", wav_path.name, attempt, e
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)  # backoff
            else:
                return {"stem": wav_path.stem, "success": False, "text": "", "error_msg": str(e)}

    # Should not reach here, but just in case
    return {"stem": wav_path.stem, "success": False, "text": "", "error_msg": "max retries exceeded"}


def _encode_multipart(
    files: dict[str, tuple],
    fields: dict[str, str],
) -> dict:
    """Minimal multipart/form-data encoder (no external deps)."""
    import uuid

    boundary = uuid.uuid4().hex
    sep = f"--{boundary}\r\n".encode()
    end = f"--{boundary}--\r\n".encode()

    body = b""
    for key, (filename, content, mime) in files.items():
        body += sep
        body += (
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        body += content + b"\r\n"

    for key, value in fields.items():
        body += sep
        body += (
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        ).encode()

    body += end
    return {"body": body, "boundary": boundary}


def transcribe_voice(
    voice_name: str,
    server_url: str,
    temp_dir: Path = Path("training_temp"),
    voices_dir: Path = Path("voices"),
    language: Optional[str] = None,
    max_retries: int = 3,
    concurrency: int = 0,
    callback: Optional[callable] = None,
) -> dict:
    """Transcribe all chunks for a voice via the remote whisper server.

    Transcripts are saved to two locations for safety:
      1. Per-chunk: ``training_temp/{voice_name}/0_gt_wavs/{chunk}.txt``
         (alongside WAVs, consumed by training)
      2. Permanent: ``voices/{voice_name}/transcripts/{chunk}.txt``
         (survives training temp dir resets)

    Args:
        voice_name: Voice name (e.g. "am_jefferson").
        server_url: Server URL with trailing slash (e.g. "http://192.168.1.5:8100/").
        temp_dir: Root of training_temp directory.
        voices_dir: Root of voices directory for permanent storage.
        language: Language code to pass to whisper (e.g. "en").
        max_retries: Retries per chunk on failure.
        concurrency: Number of parallel requests (0 = auto, default cpu_count).
        callback: Optional fn(chunk_name, text, total_done, total) for progress.

    Returns:
        {"total": N, "transcribed": M, "skipped": K, "failed": F}
    """
    wavs_dir = temp_dir / voice_name / "0_gt_wavs"
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"WAV directory not found: {wavs_dir}")

    wav_files = sorted(wavs_dir.glob("*.wav"))
    total = len(wav_files)
    logger.info(
        "Found %d WAV chunks for voice %s in %s", total, voice_name, wavs_dir
    )

    # Permanent storage (survives training temp dir resets)
    perm_dir = voices_dir / voice_name / "transcripts"
    perm_dir.mkdir(parents=True, exist_ok=True)

    if not server_url.endswith("/"):
        server_url += "/"
    endpoint = f"{server_url}transcribe"

    # Health check first
    logger.info("Checking server health at %s", server_url.rstrip("/") + "/health")
    try:
        req = urllib.request.Request(server_url.rstrip("/") + "/health")
        with urllib.request.urlopen(req, timeout=30) as resp:
            health = json.loads(resp.read())
            logger.info("Server OK: %s", health)
    except Exception as e:
        raise RuntimeError(f"Cannot reach whisper server at {server_url}: {e}") from e

    # Pre-scan for existing transcripts to skip them
    skipped_files = []
    pending_files = []
    jsonl_entries = []

    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        perm_txt_path = perm_dir / f"{wav_path.stem}.txt"
        if txt_path.exists() or perm_txt_path.exists():
            # Restore missing local copy from permanent if needed
            if perm_txt_path.exists() and not txt_path.exists():
                txt_path.write_text(perm_txt_path.read_text())
                logger.debug("Restored %s from permanent storage", wav_path.name)
            text = (perm_txt_path if perm_txt_path.exists() else txt_path).read_text().strip()
            jsonl_entries.append({"chunk": wav_path.stem, "text": text})
            skipped_files.append(wav_path)
        else:
            pending_files.append(wav_path)

    logger.info(
        "%d chunks to transcribe, %d already done", len(pending_files), len(skipped_files)
    )

    if not pending_files:
        # Nothing to do — just write JSONL and return
        jsonl_path = wavs_dir.parent / "full_text.jsonl"
        perm_jsonl_path = perm_dir / "full_text.jsonl"
        for path in (jsonl_path, perm_jsonl_path):
            with open(path, "w") as f:
                for entry in jsonl_entries:
                    f.write(json.dumps(entry) + "\n")

        return {
            "total": total,
            "transcribed": 0,
            "skipped": len(skipped_files),
            "failed": 0,
            "jsonl_path": str(jsonl_path),
            "perm_jsonl_path": str(perm_jsonl_path),
        }

    # Determine concurrency level
    max_workers = concurrency if concurrency > 0 else (os.cpu_count() or 1)
    logger.info("Transcribing with %d concurrent workers", max_workers)

    # Thread-safe counters and callback lock
    counter_lock = Lock()
    done_count = 0
    transcribed_count = 0
    failed_count = 0

    def _on_result(result: dict):
        nonlocal done_count, transcribed_count, failed_count
        stem = result["stem"]
        text = result.get("text", "")

        with counter_lock:
            done_count += 1
            current_done = len(skipped_files) + done_count

        if result.get("skipped"):
            return

        jsonl_entries.append({"chunk": stem, "text": text})

        if callback:
            try:
                callback(stem, text, current_done, total)
            except Exception as e:
                logger.warning("Callback error for %s: %s", stem, e)

        if result["success"]:
            with counter_lock:
                transcribed_count += 1
            logger.info("[%d/%d] Saved transcript for %s (%d chars)",
                        current_done, total, stem, len(text))
        else:
            with counter_lock:
                failed_count += 1
            logger.error("[%d/%d] Failed permanently: %s — %s",
                         current_done, total, stem, result.get("error_msg"))

    # Use ProcessPoolExecutor for true parallelism (avoids GIL blocking on urllib socket I/O)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=max_workers,
        initializer=_mp_worker_init,
        initargs=(endpoint, language, max_retries, str(perm_dir)),
    ) as pool:
        pending_strs = [str(p) for p in pending_files]

        # imap_unordered yields results as they complete (non-blocking iteration)
        for result in pool.imap_unordered(_mp_worker_task, pending_strs, chunksize=1):
            _on_result(result)

    skipped_count = len(skipped_files)

    # Write combined JSONL to both locations
    jsonl_path = wavs_dir.parent / "full_text.jsonl"
    perm_jsonl_path = perm_dir / "full_text.jsonl"
    for path in (jsonl_path, perm_jsonl_path):
        with open(path, "w") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry) + "\n")
    logger.info("Wrote %d entries to %s and %s", len(jsonl_entries), jsonl_path, perm_jsonl_path)

    return {
        "total": total,
        "transcribed": transcribed_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "jsonl_path": str(jsonl_path),
        "perm_jsonl_path": str(perm_jsonl_path),
    }
