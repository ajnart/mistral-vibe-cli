"""Webcam worker for head-gesture approval — run as ``python -m`` in a subprocess.

Owns the camera and a small live OpenCV preview window (both need the process main
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
import json
import os
import subprocess
import sys
import time
from typing import Any

from vibe.cli.textual_ui.head_gesture import _COOLDOWN_S, _WINDOW, classify_gesture

_WINDOW_TITLE = "vibe gesture"
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, inlined to avoid importing cv2 at module load
_DISP_W = 320  # small preview: window autosizes to this, not the raw 640/720p frame
_BORDER = 3  # accent frame thickness (px)
_MARGIN = 24  # gap from the top-right screen corner (px)
_ACCENT = (0, 140, 255)  # BGR ~orange
_BAR = (28, 28, 28)  # header/footer bar
_CAMERA_ENV = (
    "VIBE_GESTURE_CAMERA"  # override with an explicit index if auto-pick is wrong
)


def _emit(token: str) -> None:
    print(token, flush=True)


def _screen_width(default: int = 1440) -> int:
    """Primary-screen width in points via CoreGraphics (ctypes), to hug top-right.

    Uses ctypes rather than tkinter: Tk 9 aborts the process on window creation on
    macOS 26. Non-mac / failure -> a sane default.
    """
    if sys.platform != "darwin":
        return default
    try:
        import ctypes

        class _Size(ctypes.Structure):
            _fields_ = [("w", ctypes.c_double), ("h", ctypes.c_double)]

        class _Rect(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("size", _Size)]

        cg = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayBounds.restype = _Rect
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        width = int(cg.CGDisplayBounds(cg.CGMainDisplayID()).size.w)
        return width if width > 0 else default
    except Exception:
        return default


def _mac_builtin_camera_index() -> int | None:
    """Index of the built-in (FaceTime) camera, so we skip iPhone Continuity Camera.

    Ceiling: assumes ``system_profiler``'s camera order matches OpenCV's AVFoundation
    device index order (it does in practice). If it ever doesn't, set the
    VIBE_GESTURE_CAMERA env var to the right index.
    """
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=6,
        )
        cams = json.loads(out.stdout).get("SPCameraDataType", [])
    except Exception:
        return None

    names = [c.get("_name", "").lower() for c in cams]
    # Built-in Macs name it "MacBook Pro Camera" / "FaceTime HD Camera" / etc.
    builtin = ("macbook", "imac", "mac mini", "mac studio", "facetime", "built-in")
    external = ("iphone", "ipad", "continuity", "desk view")
    for idx, name in enumerate(names):
        if any(k in name for k in builtin):
            return idx
    # Nothing matched a known Mac name — fall back to the first non-iPhone camera.
    for idx, name in enumerate(names):
        if not any(k in name for k in external):
            return idx
    return None


def _resolve_camera_index(requested: int) -> int:
    """Explicit env override > built-in Mac camera > whatever was requested."""
    env = os.environ.get(_CAMERA_ENV, "").strip()
    if env.lstrip("-").isdigit():
        return int(env)
    if sys.platform == "darwin":
        builtin = _mac_builtin_camera_index()
        if builtin is not None:
            return builtin
    return requested


def _open_camera(cv2: Any, requested: int) -> Any:
    """Open the preferred camera; fall back to scanning the first few indices."""
    preferred = _resolve_camera_index(requested)
    for idx in [preferred, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    return None


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
    cap = _open_camera(cv2, camera_index)
    if cap is None or cascade.empty():
        _emit("ERR_CAMERA")
        if cap is not None:
            cap.release()
        return 1

    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(_WINDOW_TITLE, _screen_width() - _DISP_W - _MARGIN, _MARGIN)

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
            frame = cv2.flip(frame, 1)  # mirror: selfie view; axis amplitudes unchanged
            gesture = _track_face(cv2, frame, cascade, xs, ys)

            now = time.monotonic()
            if gesture and (now - last_fire) > _COOLDOWN_S:
                last_fire, last_label = now, gesture
                xs.clear()
                ys.clear()
                _emit(gesture)

            cv2.imshow(_WINDOW_TITLE, _compose(cv2, frame, last_label))
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
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 230, 0), 2)
    xs.append((fx + fw / 2) / width)
    ys.append((fy + fh / 2) / height)
    return classify_gesture(list(xs), list(ys))


def _compose(cv2: Any, frame: Any, last_label: str) -> Any:
    """Shrink to a small preview, add header/footer bars and an accent border."""
    height, width = frame.shape[:2]
    disp = cv2.resize(frame, (_DISP_W, int(_DISP_W * height / width)))
    disp_h = disp.shape[0]

    cv2.rectangle(disp, (0, 0), (_DISP_W, 24), _BAR, -1)
    cv2.putText(
        disp, "nod = ALLOW   shake = DENY", (8, 17), _FONT, 0.45, (240, 240, 240), 1, 16
    )
    if last_label:
        text = "YES - allow once" if last_label == "yes" else "NO - deny"
        color = (0, 220, 0) if last_label == "yes" else (0, 0, 255)
        cv2.rectangle(disp, (0, disp_h - 22), (_DISP_W, disp_h), _BAR, -1)
        cv2.putText(disp, text, (8, disp_h - 6), _FONT, 0.5, color, 1, 16)

    return cv2.copyMakeBorder(
        disp, _BORDER, _BORDER, _BORDER, _BORDER, cv2.BORDER_CONSTANT, value=_ACCENT
    )


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(main(idx))
