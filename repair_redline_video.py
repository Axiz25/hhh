#!/usr/bin/env python3
"""
repair_redline_video.py
------------------------
Video port of Repair-RedLine.ps1: finds a thin vertical red-line defect
and repairs it, applied consistently across every frame of a video.

Design decisions that differ from a naive "run the photo tool per frame":

1. The defect column is detected ONCE, from a set of frames sampled across
   the whole clip, and that single column is then used to repair every
   frame. A sensor/lens-level line defect sits at a fixed image-space
   column regardless of scene content, so re-detecting per frame would let
   per-frame noise shift the column by a pixel or two -- which reads as
   flicker/wobble on playback. Locking one column guarantees identical
   repair geometry on every frame (a hard requirement here).

2. Frames are streamed through ffmpeg via pipes (decode -> numpy -> encode),
   never written to disk individually. This keeps memory bounded regardless
   of clip length and avoids an extra generation of image compression.

3. Audio is never decoded or touched -- it's stream-copied directly from
   the source file into the output container, so it stays bit-identical
   and in sync (frame count in == frame count out, so timing is preserved).

HONEST LIMITATION: standard video codecs store frames as YUV, not RGB.
Editing pixels means round-tripping every frame through RGB and back to
YUV for re-encoding. That round trip is not perfectly lossless -- it can
shift pixel values by about +/-1 per channel, everywhere in the frame, not
just in the repaired band. This is far below what any lossy video codec's
own quantization introduces, and is imperceptible, but it means the output
is not literally bit-identical outside the repaired region. There is no
way to avoid this while still using a standard YUV-based codec; see
--lossless for the closest practical approach to "as identical as possible".

Requires: ffmpeg + ffprobe on PATH, numpy.
"""

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

MARGIN = 3  # matches the PS1's `x in [3, width-3)` scan bounds


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

def ffprobe_json(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class VideoInfo:
    def __init__(self, path: Path):
        info = ffprobe_json(path)
        vstreams = [s for s in info["streams"] if s["codec_type"] == "video"]
        astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in {path}")
        v = vstreams[0]

        self.width = int(v["width"])
        self.height = int(v["height"])
        self.pix_fmt = v.get("pix_fmt", "yuv420p")
        self.codec_name = v.get("codec_name", "h264")
        self.r_frame_rate = v.get("r_frame_rate", "25/1")
        fr = Fraction(self.r_frame_rate)
        self.fps = float(fr)

        # bit_rate: prefer the video stream's own value, fall back to the
        # container-level bit_rate (minus a rough audio estimate), then None
        br = v.get("bit_rate")
        if br is None:
            fmt_br = info.get("format", {}).get("bit_rate")
            br = fmt_br
        self.bit_rate = int(br) if br else None

        dur = v.get("duration") or info.get("format", {}).get("duration")
        self.duration = float(dur) if dur else None

        nb = v.get("nb_frames")
        if nb is None and self.duration:
            nb = int(round(self.duration * self.fps))
        self.nb_frames = int(nb) if nb else None

        self.has_audio = len(astreams) > 0

        if any(tag in self.pix_fmt for tag in ("10le", "10be", "12le", "12be", "16le", "16be")):
            raise RuntimeError(
                f"Source pixel format is '{self.pix_fmt}' (>8-bit). This script processes "
                "at 8-bit (rgb24) precision, which would quietly reduce bit depth -- exactly "
                "the kind of quality change you asked to avoid. Refusing rather than doing "
                "that silently. Extending to rgb48le for 10/12-bit sources is possible if "
                "you need it -- ask and I'll add it."
            )


# --------------------------------------------------------------------------
# Detection (ported from DetectLineColumn in the .ps1's C#)
# --------------------------------------------------------------------------

def score_frame(frame_rgb: np.ndarray, margin: int = MARGIN):
    """Vectorized port of the PS1's per-column redExcess/localDistance scoring."""
    h, w, _ = frame_rgb.shape
    f = frame_rgb.astype(np.int32)
    xs = np.arange(margin, w - margin)

    center = f[:, xs, :]
    left = f[:, xs - 2, :]
    right = f[:, xs + 2, :]

    interp = (left + right) // 2  # both operands >=0, floor==trunc here

    d = center - interp
    dr, dg, db = d[..., 0], d[..., 1], d[..., 2]

    # C# `(dg+db)/2` truncates toward zero, not floor -- replicate exactly
    half_ge = np.trunc((dg + db).astype(np.float64) / 2.0)
    red_excess = dr - half_ge
    local_distance = np.abs(dr) + np.abs(dg) + np.abs(db)

    score = np.where(red_excess > 4, red_excess - 4, 0.0)
    bonus_mask = (local_distance > 18) & (dr > 0)
    score = score + np.where(bonus_mask, (local_distance - 18) * 0.15, 0.0)

    return xs, score.sum(axis=0)


def extract_frame_at(path: Path, timestamp: float, width: int, height: int) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    frame_size = width * height * 3
    if len(proc.stdout) != frame_size:
        return None
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def detect_line_column(path: Path, info: VideoInfo, sample_frames: int):
    duration = info.duration or (info.nb_frames / info.fps if info.nb_frames else 1.0)
    n = max(1, min(sample_frames, info.nb_frames or sample_frames))
    timestamps = [duration * (i + 0.5) / n for i in range(n)]

    total_scores = None
    xs_ref = None
    per_frame_best = []

    for t in timestamps:
        frame = extract_frame_at(path, t, info.width, info.height)
        if frame is None:
            continue
        xs, scores = score_frame(frame)
        if total_scores is None:
            xs_ref, total_scores = xs, np.zeros_like(scores)
        total_scores += scores
        per_frame_best.append((t, int(xs[np.argmax(scores)]), float(scores.max())))

    if total_scores is None:
        raise RuntimeError("Could not sample any frames for detection.")

    best_idx = int(np.argmax(total_scores))
    return int(xs_ref[best_idx]), float(total_scores[best_idx]), per_frame_best


# --------------------------------------------------------------------------
# Repair (ported from RepairStripe)
# --------------------------------------------------------------------------

def make_repair_geometry(line_x: int, half_width: int, width: int):
    start_x = max(1, line_x - half_width)
    end_x = min(width - 2, line_x + half_width)
    col_count = end_x - start_x + 1
    t = (np.arange(1, col_count + 1, dtype=np.float64) / (col_count + 1))
    return start_x, end_x, t


def repair_frame_inplace(frame_rgb: np.ndarray, start_x: int, end_x: int, t: np.ndarray):
    left_col = frame_rgb[:, start_x - 1, :].astype(np.float64)
    right_col = frame_rgb[:, end_x + 1, :].astype(np.float64)
    blended = left_col[:, None, :] + (right_col[:, None, :] - left_col[:, None, :]) * t[None, :, None]
    frame_rgb[:, start_x:end_x + 1, :] = np.rint(blended).astype(np.uint8)


# --------------------------------------------------------------------------
# Encode settings
# --------------------------------------------------------------------------

def pick_encode_args(info: VideoInfo, lossless: bool, crf_override, vcodec_override):
    out_pix_fmt = info.pix_fmt if info.pix_fmt in ("yuv420p", "yuv422p", "yuv444p") else "yuv420p"

    if vcodec_override:
        vcodec = vcodec_override
    elif info.codec_name in ("hevc", "h265"):
        vcodec = "libx265"
    else:
        vcodec = "libx264"  # safe universal default

    if lossless:
        if vcodec == "libx265":
            quality_args = ["-x265-params", "lossless=1", "-preset", "slow"]
        else:
            quality_args = ["-qp", "0", "-preset", "slow"]
        note = f"lossless ({vcodec}, mathematically lossless mode)"
    elif crf_override is not None:
        quality_args = ["-crf", str(crf_override), "-preset", "slow"]
        note = f"CRF {crf_override} (manual override)"
    elif info.bit_rate:
        br = info.bit_rate
        quality_args = ["-b:v", str(br), "-maxrate", str(int(br * 1.5)),
                         "-bufsize", str(int(br * 2))]
        note = f"matched source bitrate (~{br/1_000_000:.1f} Mbps)"
    else:
        quality_args = ["-crf", "16", "-preset", "slow"]
        note = "source bitrate unavailable -> CRF 16 (near-visually-lossless) fallback"

    return vcodec, out_pix_fmt, quality_args, note


# --------------------------------------------------------------------------
# Main processing pass
# --------------------------------------------------------------------------

def process_video(input_path: Path, output_path: Path, info: VideoInfo,
                   start_x: int, end_x: int, t: np.ndarray,
                   lossless: bool, crf_override, vcodec_override):
    vcodec, out_pix_fmt, quality_args, note = pick_encode_args(
        info, lossless, crf_override, vcodec_override)
    print(f"Encoding: {vcodec}, {note}, pix_fmt={out_pix_fmt}")

    frame_size = info.width * info.height * 3

    decode_cmd = ["ffmpeg", "-v", "error", "-i", str(input_path),
                  "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-fps_mode", "passthrough", "pipe:1"]

    encode_cmd = ["ffmpeg", "-v", "error", "-y",
                  "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-s", f"{info.width}x{info.height}",
                  "-r", info.r_frame_rate,
                  "-i", "pipe:0",
                  "-i", str(input_path),
                  "-map", "0:v:0", "-map", "1:a:0?",
                  "-c:v", vcodec, *quality_args,
                  "-pix_fmt", out_pix_fmt,
                  "-c:a", "copy",
                  "-movflags", "+faststart",
                  str(output_path)]

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            data = decoder.stdout.read(frame_size)
            if not data or len(data) < frame_size:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape(
                info.height, info.width, 3).copy()
            repair_frame_inplace(frame, start_x, end_x, t)
            encoder.stdin.write(frame.tobytes())
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - t0
                total = f"/{info.nb_frames}" if info.nb_frames else ""
                print(f"  frame {frame_count}{total}  ({frame_count/elapsed:.1f} fps)", end="\r")
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
        dec_rc = decoder.wait()
        enc_rc = encoder.wait()

    print()
    if dec_rc != 0:
        raise RuntimeError(f"ffmpeg decode process failed (exit {dec_rc})")
    if enc_rc != 0:
        raise RuntimeError(f"ffmpeg encode process failed (exit {enc_rc})")

    print(f"Done: {frame_count} frames -> {output_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Source video file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                     help="Output path (default: <input>_fixed<ext>)")
    ap.add_argument("--half-width", type=int, default=7,
                     help="Half-width of repaired band in pixels (default: 7 -> 15px band)")
    ap.add_argument("--sample-frames", type=int, default=24,
                     help="Frames sampled across the clip to lock the defect column (default: 24)")
    ap.add_argument("--lossless", action="store_true",
                     help="Mathematically lossless re-encode (qp=0 / x265 lossless). Large files.")
    ap.add_argument("--crf", type=int, default=None,
                     help="Manual CRF override instead of matched-bitrate/lossless")
    ap.add_argument("--vcodec", default=None,
                     help="Force a specific output video codec (default: auto-match source)")
    ap.add_argument("--analyze-only", action="store_true",
                     help="Detect and report the defect column, write nothing")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    info = VideoInfo(args.input)
    print(f"{args.input.name}: {info.width}x{info.height} @ {info.fps:.3f}fps "
          f"({info.r_frame_rate}), {info.codec_name}, pix_fmt={info.pix_fmt}, "
          f"audio={'yes' if info.has_audio else 'no'}, frames~={info.nb_frames}")

    line_x, score, per_frame = detect_line_column(args.input, info, args.sample_frames)
    xs_at = [x for _, x, _ in per_frame]
    spread = max(xs_at) - min(xs_at) if xs_at else 0
    print(f"Detected column x={line_x}  aggregate_score={score:.1f}  "
          f"(sampled {len(per_frame)} frames, per-frame column spread={spread}px)")
    if spread > 2 * max(1, args.half_width):
        print("  WARNING: detected column varies a lot across sampled frames. "
              "This heuristic assumes one fixed-position defect -- if the artifact "
              "actually moves with scene content, a single locked column is the wrong "
              "model and results will be inconsistent.")

    if args.analyze_only:
        return

    output_path = args.output or args.input.with_name(
        args.input.stem + "_fixed" + args.input.suffix)

    start_x, end_x, t = make_repair_geometry(line_x, args.half_width, info.width)
    print(f"Repair band: columns {start_x}-{end_x} ({end_x - start_x + 1}px)")

    process_video(args.input, output_path, info, start_x, end_x, t,
                   args.lossless, args.crf, args.vcodec)


if __name__ == "__main__":
    main()
