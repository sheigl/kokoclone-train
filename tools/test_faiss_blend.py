#!/usr/bin/env python3
"""Test different FAISS blend_ratio values for voice conversion."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import time

sys.path.insert(0, str(Path.home() / "code/kokoclone"))
import core.rvc_engine as engine

voice_dir = str(Path.home() / "code/kokoclone-train/voices/am_jefferson")
audio_path = Path.home() / "code/kokoclone-train/training_temp/am_jefferson/0_gt_wavs/0_0.wav"
out_dir = Path.home() / "code/kokoclone-train/tools"

audio_40k, sr = sf.read(str(audio_path), dtype="float32")

voice = engine.load_voice(voice_dir, device_select="cpu")

# Monkey-patch _retrieve_features to use different blend_ratios
orig_retrieve = engine._retrieve_features

def make_retriever(ratio):
    def patched(content, index_path):
        import faiss
        index = faiss.read_index(index_path)
        d = content.shape[1]
        query = content.reshape(-1, d).astype(np.float32)
        n_search = min(256, len(query))
        if n_search > 0:
            _, indices = index.search(query[:n_search], 1)
            retrieved = index.reconstruct_n(int(indices.flatten()[0]), int(n_search))
            blended = query[:n_search] * (1 - ratio) + retrieved * ratio
            if len(query) > n_search:
                return np.concatenate([blended, query[n_search:]])
            return blended
        return content
    return patched

for blend in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    engine._retrieve_features = make_retriever(blend)
    label = f"blend{blend:.1f}"
    print(f"\n[{label}] Running inference...")
    t0 = time.time()
    wav, _ = engine.convert_audio(
        audio_40k.copy(), sr, voice,
        transpose=0, f0_method="harvest",
    )
    elapsed = time.time() - t0
    out_path = out_dir / f"test_tweak_{label}.wav"
    sf.write(str(out_path), wav, sr)
    print(f"  {out_path.name}: {len(wav)/sr:.1f}s, "
          f"mean={wav.mean():+.6f}, std={wav.std():.4f}, "
          f"sat={(np.abs(wav) > 0.99).mean()*100:.1f}%, "
          f"in {elapsed:.0f}s")

# Restore original
engine._retrieve_features = orig_retrieve
