#!/usr/bin/env python3
"""Test TTS with very aggressive FAISS blend ratios."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import time

sys.path.insert(0, str(Path.home() / "code/kokoclone"))
import core.rvc_engine as engine
import faiss

TEXT = "Doors and corners, kid. That's where they get you."
OUT_DIR = Path.home() / "code/kokoclone-train/tools"

# Monkey-patch _retrieve_features to test extreme blend ratios
def make_retriever(k, blend_ratio):
    def patched(content, index_path):
        index = faiss.read_index(index_path)
        query = content.reshape(-1, 768).astype(np.float32)
        if len(query) == 0:
            return content
        D, I = index.search(query, k)
        retrieved = np.zeros_like(query)
        for t in range(len(query)):
            vecs = np.array([index.reconstruct(int(I[t, j])) for j in range(k)])
            retrieved[t] = vecs.mean(axis=0)
        blended = query * (1.0 - blend_ratio) + retrieved * blend_ratio
        return blended
    return patched

for voice_name, blend, k in [("am_jefferson", 0.9, 1), ("am_jefferson", 0.95, 1),
                              ("am_jefferson", 1.0, 3), ("am_welch", 0.9, 1),
                              ("am_welch", 1.0, 3)]:
    label = f"tts_{voice_name}_b{blend}_k{k}"
    print(f"\n[{label}] Generating TTS...")

    from core.cloner import KokoClone
    clone = KokoClone()

    orig_retrieve = engine._retrieve_features
    engine._retrieve_features = make_retriever(k, blend)

    t0 = time.time()
    out_path = OUT_DIR / f"{label}.wav"
    clone.generate(
        text=TEXT, lang="en", voice_name=voice_name,
        output_path=str(out_path),
    )
    elapsed = time.time() - t0

    engine._retrieve_features = orig_retrieve

    wav, sr = sf.read(str(out_path), dtype="float32")
    print(f"  {out_path.name}: {len(wav)/sr:.1f}s, "
          f"mean={wav.mean():+.6f}, std={wav.std():.4f}, "
          f"sat={(np.abs(wav) > 0.99).mean()*100:.1f}%, in {elapsed:.0f}s")
