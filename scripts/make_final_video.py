#!/usr/bin/env python3
"""Build the final submission video: title + seed_7 place + bench cards.

Inputs: out/demo/seed_7.mp4 (256x256@20fps, 15.5s, verified place).
Output: out/demo/MAMBO_VLA_RC_final.mp4 (1280x720@20fps, yuv420p).
CPU-only, no training/eval code touched. Run: ./.venv/bin/python scripts/make_final_video.py
"""
import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "out", "demo")
SEED7 = os.path.join(DEMO, "seed_7.mp4")
OUT = os.path.join(DEMO, "MAMBO_VLA_RC_final.mp4")
W, H, FPS = 1280, 720, 20
BG = (11, 16, 32)
FG = (235, 240, 250)
ACCENT = (255, 176, 32)
DIM = (150, 160, 180)


def resolve_font(bold=False):
    cands = []
    for pat in ("/usr/share/fonts/**/NotoSans*-Bold.ttf",
                "/usr/share/fonts/**/NotoSans*.ttf",
                "/usr/share/fonts/**/DejaVuSans*.ttf"):
        cands += sorted(glob.glob(pat, recursive=True))
    if not bold:
        cands = [c for c in cands if "Bold" not in os.path.basename(c)] + cands
    if not cands:
        raise RuntimeError("no TTF font found")
    print("FONT=" + cands[0], flush=True)
    return cands[0]


def card(lines, path, accent_idx=0):
    """lines: list of (text, size, color)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 120
    fpath = resolve_font()
    for i, (text, size, color) in enumerate(lines):
        f = ImageFont.truetype(fpath, size)
        bb = d.textbbox((0, 0), text, font=f)
        d.text(((W - (bb[2] - bb[0])) / 2, y), text, font=f,
               fill=ACCENT if i == accent_idx else color)
        y += (bb[3] - bb[1]) + 36
    img.save(path)
    print("CARD=" + path, flush=True)


def run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    assert os.path.isfile(SEED7), "missing " + SEED7
    os.makedirs(DEMO, exist_ok=True)
    c1 = os.path.join(DEMO, "_card_title.png")
    c2 = os.path.join(DEMO, "_card_bench.png")
    c3 = os.path.join(DEMO, "_card_method.png")
    c4 = os.path.join(DEMO, "_card_end.png")
    card([("MAMBO-VLA-RC", 84, FG),
          ("Bimanual VLA dinner-table · ACT-52M · 60 demos", 40, DIM),
          ("Closed-loop 3/10 · Intel Xeon + OpenVINO", 40, FG)], c1)
    card([("Intel Xeon E-2386G · OpenVINO 2026.3.0 · CPU-only", 36, DIM),
          ("v3-100k FP32 · latency 40.8 ms / 24.5 FPS", 44, FG),
          ("throughput 27.9 FPS · IR 66.7 MB · parity 1.2e-06", 44, FG),
          ("No NPU/iGPU on host (disclosed)", 34, DIM)], c2)
    card([("Frozen protocol · 10 seeds · negatives filed", 40, FG),
          ("Vision trend 6.6 to 3.3 · intermediate peak at 41ep", 38, DIM),
          ("Honest 3/10 beats fluent-but-hardcoded", 38, DIM)], c3)
    card([("MAMBO-VLA-RC", 72, FG),
          ("AI Infra Summit · Intel Physical AI Challenge", 40, ACCENT)], c4)
    # Upscaled seed_7 with caption, then concat all at 1280x720@20fps.
    fpath = resolve_font()
    seg_clip = os.path.join(DEMO, "_seg_clip.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", SEED7,
         "-vf", f"scale=720:720:flags=neighbor,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0b1020,drawtext=fontfile={fpath}:text='seed 7 — place':fontsize=34:fontcolor=white:x=(w-text_w)/2:y=24",
         "-r", str(FPS), "-pix_fmt", "yuv420p", seg_clip])
    filelist = os.path.join(DEMO, "_concat.txt")
    parts = [(c1, 4), (seg_clip, None), (c2, 5), (c3, 4), (c4, 3)]
    segs = []
    for i, (p, dur) in enumerate(parts):
        if dur is None:
            segs.append(p)  # already a uniform mp4 segment (the clip)
            continue
        seg = os.path.join(DEMO, f"_seg_card{i}.mp4")
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1",
             "-framerate", str(FPS), "-t", str(dur), "-i", p,
             "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", seg])
        segs.append(seg)
    with open(filelist, "w") as fh:
        for s in segs:
            fh.write(f"file '{s}'\n")
    # All segments share codec/params -> stream copy is safe and fast.
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", filelist, "-c", "copy", "-movflags", "+faststart", OUT])
    print("OUT=" + OUT, flush=True)


if __name__ == "__main__":
    sys.exit(main())
