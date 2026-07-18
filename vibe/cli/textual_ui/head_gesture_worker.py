"""Webcam worker for head-gesture approval — run as ``python -m`` in a subprocess.

Owns the camera and a small live OpenCV preview window (both need the process main
thread on macOS, which the parent Textual app can't give it — see head_gesture.py).
The user must repeat a gesture twice to confirm; then an outro tick/cross plays and
one token is printed to stdout for the parent to act on:

    yes         nod x2 -> allow once
    no          shake x2 -> deny
    ERR_OPENCV  opencv-python not installed
    ERR_CAMERA  camera could not be opened

Face tracking uses OpenCV's YuNet DNN detector (5 landmarks), which is far more
robust than a Haar cascade and — crucially — lets us read head *rotation* from the
nose vs. eyes, so a natural head-shake ("no") works without sliding your whole head.
Falls back to Haar (box-centroid tracking) if the YuNet model can't be fetched.

Exits when it confirms, the parent terminates it, the window is closed, or 'q'.
"""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Any
import urllib.request

from vibe.cli.textual_ui.head_gesture import _COOLDOWN_S, _WINDOW, classify_gesture

_WINDOW_TITLE = "vibe gesture"
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, inlined to avoid importing cv2 at module load
_DISP_W = 340  # small preview: window autosizes to this, not the raw 640/720p frame
_BORDER = 3  # accent frame thickness (px)
_MARGIN = 24  # gap from the top-right screen corner (px)
_ACCENT = (0, 140, 255)  # BGR ~orange
_BAR = (28, 28, 28)  # header/footer bar
_PANEL = (22, 22, 22)  # command panel background
_CAMERA_ENV = "VIBE_GESTURE_CAMERA"  # override index if the auto-pick is wrong
_PROMPT_ENV = "VIBE_GESTURE_PROMPT"  # what the agent wants to do, shown in the window

_REQUIRED_REPEATS = 2  # do the nod/shake this many times to confirm
_TRAIL_LEN = 24  # how many recent head positions to draw as a trail
_MISS_GRACE = 8  # keep the box/track alive this many frames after losing the face
_RESET_S = 5.0  # forget a half-finished confirmation after this idle gap
_SMOOTH = 0.4  # EMA weight for the drawn box (lower = smoother, less flashy)
_LINE = 16  # cv2.LINE_AA
# Head yaw/pitch (nose vs. eyes, normalized by inter-ocular distance) is a smaller,
# different-scale signal than face-box translation — needs its own amplitude gate.
# Raise to require a bigger head turn/nod, lower if it feels sluggish.
_POSE_AMPLITUDE = 0.16
_MIN_MODEL_BYTES = 10_000  # a truncated/failed YuNet download is smaller than this
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _emit(token: str) -> None:
    print(token, flush=True)


# ---------------------------------------------------------------- camera / screen


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
    """Index of the built-in Mac camera, so we skip iPhone Continuity Camera.

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
    builtin = ("macbook", "imac", "mac mini", "mac studio", "facetime", "built-in")
    external = ("iphone", "ipad", "continuity", "desk view")
    for idx, name in enumerate(names):
        if any(k in name for k in builtin):
            return idx
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


# ------------------------------------------------------------------ face detector


def _yunet_model_path() -> Path:
    """Path to the cached YuNet ONNX, downloading it (~230KB) on first use."""
    cache = Path.home() / ".vibe" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "face_detection_yunet_2023mar.onnx"
    if not path.exists() or path.stat().st_size < _MIN_MODEL_BYTES:
        urllib.request.urlretrieve(_YUNET_URL, path)
    return path


def _load_detector(cv2: Any) -> tuple[str, Any]:
    """Prefer YuNet (robust + landmarks); fall back to a Haar cascade."""
    try:
        det = cv2.FaceDetectorYN.create(
            str(_yunet_model_path()), "", (320, 320), 0.7, 0.3, 50
        )
        return ("yunet", det)
    except Exception:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades  # type: ignore[attr-defined]
            + "haarcascade_frontalface_default.xml"
        )
        return ("haar", cascade)


def _detect(cv2: Any, kind: str, det: Any, frame: Any, prev: Any) -> Any:
    """Return (box, marks) normalized to 0..1, or None. ``marks`` may be None (Haar).

    ``box`` = (x, y, w, h); ``marks`` = {"reye","leye","nose"} of (x, y) points.
    Picks the face nearest the previous track (else largest) to ignore stray faces.
    """
    height, width = frame.shape[:2]
    if kind == "yunet":
        det.setInputSize((width, height))
        _, faces = det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        row = _pick_row(faces, prev, width, height)
        box = (row[0] / width, row[1] / height, row[2] / width, row[3] / height)
        marks = {
            "reye": (row[4] / width, row[5] / height),
            "leye": (row[6] / width, row[7] / height),
            "nose": (row[8] / width, row[9] / height),
        }
        return (box, marks)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))
    if len(faces) == 0:
        return None
    f = max(faces, key=lambda b: b[2] * b[3])
    return ((f[0] / width, f[1] / height, f[2] / width, f[3] / height), None)


def _pick_row(faces: Any, prev: Any, width: int, height: int) -> Any:
    """Choose the YuNet detection nearest the previous face center, else the largest."""
    if prev is not None:
        px, py = prev

        def dist(r: Any) -> float:
            cx = (r[0] + r[2] / 2) / width
            cy = (r[1] + r[3] / 2) / height
            return (cx - px) ** 2 + (cy - py) ** 2

        return min(faces, key=dist)
    return max(faces, key=lambda r: r[2] * r[3])


class _Session:
    """Per-run tracking + confirmation state (kept off `main` to bound its locals)."""

    def __init__(self, pose_mode: bool) -> None:
        self.pose_mode = pose_mode  # True = YuNet yaw/pitch, False = Haar box-centroid
        self.xs: deque[float] = deque(maxlen=_WINDOW)
        self.ys: deque[float] = deque(maxlen=_WINDOW)
        self.trail: deque[tuple[float, float]] = deque(maxlen=_TRAIL_LEN)
        self.smooth: list[float] | None = None  # EMA-smoothed box for drawing
        self.marks: Any = None  # latest landmarks (for drawing)
        self.miss = 0
        self.progress: tuple[str | None, int] = (None, 0)  # (gesture, repeats so far)
        self.last_fire = 0.0

    def center(self) -> tuple[float, float] | None:
        if self.smooth is None:
            return None
        sx, sy, sw, sh = self.smooth
        return (sx + sw / 2, sy + sh / 2)

    def observe(self, detection: Any) -> None:
        if detection is None:
            self.miss += 1
            if self.miss > _MISS_GRACE:  # face really gone — drop the track
                self.xs.clear()
                self.ys.clear()
                self.trail.clear()
                self.smooth = None
                self.marks = None
            return

        box, marks = detection
        self.miss = 0
        self.marks = marks
        self.smooth = (
            list(box)
            if self.smooth is None
            else [
                _SMOOTH * b + (1 - _SMOOTH) * s
                for b, s in zip(box, self.smooth, strict=False)
            ]
        )
        sig_x, sig_y, trail_pt = self._signals(box, marks)
        self.xs.append(sig_x)
        self.ys.append(sig_y)
        self.trail.append(trail_pt)

    def _signals(
        self, box: Any, marks: Any
    ) -> tuple[float, float, tuple[float, float]]:
        """(horizontal, vertical, trail-point) for the classifier and the trail."""
        if self.pose_mode and marks is not None:
            (rx, ry), (lx, ly), (nx, ny) = marks["reye"], marks["leye"], marks["nose"]
            eye_x, eye_y = (rx + lx) / 2, (ry + ly) / 2
            iod = max(0.02, ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5)  # scale ref
            # Yaw (turn L/R = "no"), pitch (nod up/down = "yes"), rotation-invariant
            # to whole-head translation. Trail follows the nose so rotation is visible.
            return ((nx - eye_x) / iod, (ny - eye_y) / iod, (nx, ny))
        bx, by, bw, bh = box
        center = (bx + bw / 2, by + bh / 2)
        return (center[0], center[1], center)

    def step(self, now: float) -> str | None:
        """Advance the confirmation state; return "yes"/"no" once done twice."""
        pending, count = self.progress
        if count and now - self.last_fire > _RESET_S:
            pending, count = None, 0
        amp = _POSE_AMPLITUDE if self.pose_mode else 0.09
        gesture = classify_gesture(list(self.xs), list(self.ys), amplitude=amp)
        if not gesture or now - self.last_fire <= _COOLDOWN_S:
            self.progress = (pending, count)
            return None
        count = count + 1 if gesture == pending else 1
        pending = gesture
        self.last_fire = now
        self.xs.clear()  # fresh buffer for the next repetition
        self.ys.clear()
        self.progress = (pending, count)
        return pending if count >= _REQUIRED_REPEATS else None


# ------------------------------------------------------------------------ drawing


def _bordered(cv2: Any, disp: Any) -> Any:
    return cv2.copyMakeBorder(
        disp, _BORDER, _BORDER, _BORDER, _BORDER, cv2.BORDER_CONSTANT, value=_ACCENT
    )


def _render(cv2: Any, frame: Any, sess: _Session, prompt: str) -> Any:
    """Resized preview with trail, box, landmarks, HUD text and command panel."""
    height, width = frame.shape[:2]
    disp = cv2.resize(frame, (_DISP_W, int(_DISP_W * height / width)))
    _draw_track(cv2, disp, sess)
    _draw_hud(cv2, disp, sess.progress)
    return cv2.vconcat([disp, _prompt_panel(cv2, disp.shape[1], prompt)])


def _draw_track(cv2: Any, disp: Any, sess: _Session) -> None:
    """Draw the motion trail, face box and landmark dots onto the preview."""
    dh, dw = disp.shape[:2]
    trail = list(sess.trail)
    for i in range(1, len(trail)):
        (x0, y0), (x1, y1) = trail[i - 1], trail[i]
        t = i / len(trail)  # older = dim, newer = bright orange
        cv2.line(
            disp,
            (int(x0 * dw), int(y0 * dh)),
            (int(x1 * dw), int(y1 * dh)),
            (int(40 + 60 * t), int(120 * t + 20), int(255 * t)),
            max(1, int(1 + 3 * t)),
            _LINE,
        )

    if sess.smooth is not None:
        nx, ny, nw, nh = sess.smooth
        cv2.rectangle(
            disp,
            (int(nx * dw), int(ny * dh)),
            (int((nx + nw) * dw), int((ny + nh) * dh)),
            (0, 230, 0),
            2,
            _LINE,
        )
    if sess.marks is not None:
        for key, color in (
            ("reye", (255, 220, 0)),
            ("leye", (255, 220, 0)),
            ("nose", (0, 220, 255)),
        ):
            mx, my = sess.marks[key]
            cv2.circle(disp, (int(mx * dw), int(my * dh)), 3, color, -1, _LINE)


def _draw_hud(cv2: Any, disp: Any, progress: tuple[str | None, int]) -> None:
    dh, dw = disp.shape[:2]
    cv2.rectangle(disp, (0, 0), (dw, 24), _BAR, -1)
    cv2.putText(
        disp,
        "nod x2 = ALLOW   shake x2 = DENY",
        (8, 17),
        _FONT,
        0.42,
        (240, 240, 240),
        1,
        _LINE,
    )
    pending, count = progress
    if count and pending:
        label = "NOD" if pending == "yes" else "SHAKE"
        color = (0, 220, 0) if pending == "yes" else (0, 140, 255)
        cv2.rectangle(disp, (0, dh - 22), (dw, dh), _BAR, -1)
        cv2.putText(
            disp,
            f"{label} again  ({count}/{_REQUIRED_REPEATS})",
            (8, dh - 6),
            _FONT,
            0.5,
            color,
            1,
            _LINE,
        )


def _prompt_panel(cv2: Any, width: int, prompt: str) -> Any:
    """A dark strip under the video showing what the agent wants to do."""
    import numpy as np

    lines = ["Agent wants to:"]
    lines += textwrap.wrap(prompt or "(run a tool)", width=52)[:3]
    panel = np.full((16 + 15 * len(lines), width, 3), _PANEL, dtype=np.uint8)
    cv2.putText(panel, lines[0], (8, 14), _FONT, 0.4, (150, 150, 150), 1, _LINE)
    for i, line in enumerate(lines[1:], start=1):
        cv2.putText(
            panel, line, (8, 14 + 15 * i), _FONT, 0.42, (235, 235, 235), 1, _LINE
        )
    return panel


def _draw_symbol(cv2: Any, frame: Any, kind: str, scale: float) -> None:
    dh, dw = frame.shape[:2]
    cx, cy = dw // 2, dh // 2
    r = int(min(dw, dh) * 0.24 * scale)
    color = (0, 200, 0) if kind == "yes" else (0, 0, 255)
    th = max(2, int(r * 0.16))
    cv2.circle(frame, (cx, cy), r, color, 3, _LINE)
    if kind == "yes":
        cv2.line(
            frame,
            (cx - int(r * 0.45), cy),
            (cx - int(r * 0.05), cy + int(r * 0.4)),
            color,
            th,
            _LINE,
        )
        cv2.line(
            frame,
            (cx - int(r * 0.05), cy + int(r * 0.4)),
            (cx + int(r * 0.5), cy - int(r * 0.4)),
            color,
            th,
            _LINE,
        )
    else:
        cv2.line(
            frame,
            (cx - int(r * 0.35), cy - int(r * 0.35)),
            (cx + int(r * 0.35), cy + int(r * 0.35)),
            color,
            th,
            _LINE,
        )
        cv2.line(
            frame,
            (cx - int(r * 0.35), cy + int(r * 0.35)),
            (cx + int(r * 0.35), cy - int(r * 0.35)),
            color,
            th,
            _LINE,
        )
    label = "ALLOWED" if kind == "yes" else "DENIED"
    (tw, _), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
    cv2.putText(frame, label, (cx - tw // 2, cy + r + 26), _FONT, 0.6, color, 2, _LINE)


def _play_outro(cv2: Any, base: Any, kind: str) -> None:
    """Dim the preview and animate a green tick / red cross for ~0.8s."""
    dh, dw = base.shape[:2]
    dim = base.copy()
    cv2.rectangle(dim, (0, 0), (dw, dh), (18, 18, 18), -1)
    cv2.addWeighted(dim, 0.6, base, 0.4, 0, dim)
    for step in range(26):
        frame = dim.copy()
        _draw_symbol(cv2, frame, kind, min(1.0, (step + 1) / 8))
        cv2.imshow(_WINDOW_TITLE, _bordered(cv2, frame))
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break


# --------------------------------------------------------------------------- main


def main(camera_index: int = 0) -> int:
    try:
        import cv2
    except ImportError:
        _emit("ERR_OPENCV")
        return 1

    kind, det = _load_detector(cv2)
    cap = _open_camera(cv2, camera_index)
    if cap is None:
        _emit("ERR_CAMERA")
        return 1

    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(_WINDOW_TITLE, _screen_width() - _DISP_W - _MARGIN, _MARGIN)

    prompt = os.environ.get(_PROMPT_ENV, "")
    sess = _Session(pose_mode=kind == "yunet")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror: selfie view; axis amplitudes unchanged
            sess.observe(_detect(cv2, kind, det, frame, sess.center()))
            result = sess.step(time.monotonic())

            disp = _render(cv2, frame, sess, prompt)
            if result:
                _play_outro(cv2, disp, result)
                _emit(result)
                break
            cv2.imshow(_WINDOW_TITLE, _bordered(cv2, disp))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(main(idx))
