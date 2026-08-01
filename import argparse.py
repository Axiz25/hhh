import argparse

import cv2
import numpy as np


def repair_frame(frame, x=None, y=None, width=3, radius=3, method="telea"):
    """Repair a vertical or horizontal dead-pixel line with inpainting."""
    if x is None and y is None:
        raise ValueError("Set either --x for a vertical line or --y for a horizontal line.")

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    half = max(0, width // 2)

    if x is not None:
        x1 = max(0, x - half)
        x2 = min(w, x + half + 1)
        mask[:, x1:x2] = 255
    else:
        y1 = max(0, y - half)
        y2 = min(h, y + half + 1)
        mask[y1:y2, :] = 255

    flags = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    repaired = cv2.inpaint(frame, mask, radius, flags)

    # Feather only the repaired strip so the border does not look pasted in.
    soft_mask = cv2.GaussianBlur(mask, (0, 0), 1.0).astype(np.float32) / 255.0
    soft_mask = soft_mask[..., None]
    return (repaired.astype(np.float32) * soft_mask + frame.astype(np.float32) * (1.0 - soft_mask)).astype(np.uint8)


def process_video(input_path, output_path, x=None, y=None, width=3, radius=3, method="telea"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(repair_frame(frame, x=x, y=y, width=width, radius=radius, method=method))

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Repair a dead-pixel red line in a video.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--x", type=int, help="X coordinate of a vertical dead-pixel line.")
    parser.add_argument("--y", type=int, help="Y coordinate of a horizontal dead-pixel line.")
    parser.add_argument("--width", type=int, default=3, help="Line mask width in pixels. Try 2-5.")
    parser.add_argument("--radius", type=int, default=3, help="Inpaint radius. Try 2-4 for less blur.")
    parser.add_argument("--method", choices=["telea", "ns"], default="telea")
    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        x=args.x,
        y=args.y,
        width=args.width,
        radius=args.radius,
        method=args.method,
    )


if __name__ == "__main__":
    main()