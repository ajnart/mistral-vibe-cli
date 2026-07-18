"""Webcam worker for head-gesture approval — run as ``python -m`` in a subprocess.

Owns the camera and a live OpenCV preview window (both need the process main
thread on macOS, which the parent Textual app can't give it — see head_gesture.py).
Prints one token per line to stdout for the parent to act on:

    yes         nod detected  -> allow once
    no          shake detected -> deny
    ERR_OPENCV  opencv-python not installed
    ERR_CAMERA  camera could not be opened

Exits when the parent terminates it, the preview window is closed, or 'q' is pressed.
"""

from __future__ import annotations

from collections import deque
import sys
import time
from typing import Any

from vibe.cli.textual_ui.head_gesture import _COOLDOWN_S, _WINDOW, classify_gesture

_WINDOW_TITLE = "vibe · nod = allow · shake = deny · (q to close)"
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, inlined to avoid importing cv2 at module load


def _emit(token: str) -> None:
    print(token, flush=True)


def main(camera_index: int = 0) -> int:
    try:
        import cv2
    except ImportError:
        _emit("ERR_OPENCV")
        return 1

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades  # type: ignore[attr-defined]  # valid at runtime, no stub
        + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() or cascade.empty():
        _emit("ERR_CAMERA")
        cap.release()
        return 1

    xs: deque[float] = deque(maxlen=_WINDOW)
    ys: deque[float] = deque(maxlen=_WINDOW)
    last_fire = 0.0
    last_label = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror: natural; axis amplitudes unchanged
            gesture = _track_face(cv2, frame, cascade, xs, ys)

            now = time.monotonic()
            if gesture and (now - last_fire) > _COOLDOWN_S:
                last_fire, last_label = now, gesture
                xs.clear()
                ys.clear()
                _emit(gesture)

            _draw_overlay(cv2, frame, last_label)
            cv2.imshow(_WINDOW_TITLE, frame)
            # waitKey pumps the GUI event loop; also lets 'q' close the window.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Window closed via the title-bar button -> property drops to < 1.
            if cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def _track_face(
    cv2: Any, frame: Any, cascade: Any, xs: deque[float], ys: deque[float]
) -> str | None:
    """Detect the largest face, draw its box, update tracks, return a verdict."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        xs.clear()  # lost the face — don't classify across the gap
        ys.clear()
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    xs.append((fx + fw / 2) / width)
    ys.append((fy + fh / 2) / height)
    return classify_gesture(list(xs), list(ys))


def _draw_overlay(cv2: Any, frame: Any, last_label: str) -> None:
    height = frame.shape[0]
    cv2.putText(
        frame, "nod = ALLOW    shake = DENY", (10, 24), _FONT, 0.6, (255, 255, 255), 2
    )
    if last_label:
        text = "YES -> allow once" if last_label == "yes" else "NO -> deny"
        color = (0, 200, 0) if last_label == "yes" else (0, 0, 255)
        cv2.putText(frame, text, (10, height - 15), _FONT, 0.7, color, 2)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(main(idx))
