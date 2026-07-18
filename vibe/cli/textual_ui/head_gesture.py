"""Head-gesture approval: nod = yes, shake = no, driven by the webcam via OpenCV.

Two independent pieces:
  - ``classify_gesture`` — pure math over recent face-center positions. No camera,
    fully unit-testable (see ``tests/cli/test_head_gesture.py``).
  - ``HeadGestureDetector`` — a background thread that reads frames, finds the
    largest face with an OpenCV Haar cascade, and feeds its center to the classifier.

Ceiling: Haar-cascade face tracking is cheap but jittery under bad lighting and
loses the face at steep angles. Upgrade path if this ever graduates from
experimental: swap the cascade for a landmark model (e.g. mediapipe FaceMesh) and
classify head *pose* (pitch/yaw) instead of bounding-box centroid drift.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import os
import sys
import threading
import time
from typing import Any

from vibe.core.logger import logger

# macOS calibration: we open the camera from a worker thread, but AVFoundation can
# only *request* camera authorization from the main thread (it tries to spin the
# main run loop and fails otherwise). Skip the in-thread request; the terminal app
# gets camera permission through the normal OS (TCC) prompt on first real use.
# Ceiling: if the terminal has never been granted camera access, capture fails and
# we surface a friendly error instead of a prompt — grant it in System Settings >
# Privacy & Security > Camera, then retry.
if sys.platform == "darwin":
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# All positions are normalized to frame size (0..1), so thresholds are resolution
# independent. Tuned by hand against a 720p webcam at ~15 fps — recalibrate if the
# gesture feels too twitchy or too sluggish.
_WINDOW = 20  # ~1.3s of frames; the rolling buffer of face centers we classify over
_MIN_SAMPLES = 8  # need enough motion history before we trust a verdict
_AMPLITUDE = 0.05  # min peak-to-peak travel of the face center to count as a gesture
_DOMINANCE = 1.3  # the moving axis must out-travel the other by this factor
_JITTER = 0.004  # per-frame deltas smaller than this are noise, not motion
_COOLDOWN_S = 1.5  # ignore new gestures right after one fires (debounce)


def _reversals(vals: list[float]) -> int:
    """Count direction changes in a 1-D track, ignoring sub-jitter wobble.

    A nod ("down then up") or shake ("left then right") shows up as >= 1 reversal.
    """
    signs: list[int] = []
    for a, b in zip(vals, vals[1:], strict=False):
        d = b - a
        if abs(d) < _JITTER:
            continue
        signs.append(1 if d > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def classify_gesture(xs: list[float], ys: list[float]) -> str | None:
    """Return "yes" (nod), "no" (shake), or None from recent face-center tracks.

    xs/ys are normalized horizontal/vertical face-center positions, oldest first.
    """
    if len(ys) < _MIN_SAMPLES:
        return None
    amp_x = max(xs) - min(xs)
    amp_y = max(ys) - min(ys)
    # Vertical oscillation, dominant over horizontal => nod => yes.
    if amp_y > _AMPLITUDE and amp_y > amp_x * _DOMINANCE and _reversals(ys) >= 1:
        return "yes"
    # Horizontal oscillation, dominant over vertical => shake => no.
    if amp_x > _AMPLITUDE and amp_x > amp_y * _DOMINANCE and _reversals(xs) >= 1:
        return "no"
    return None


class HeadGestureDetector:
    """Watches the webcam on a daemon thread and fires ``on_gesture("yes"|"no")``.

    Callbacks run on the detector thread; the caller is responsible for hopping back
    to its own event loop (Textual: ``app.call_from_thread``).
    """

    def __init__(
        self,
        on_gesture: Callable[[str], None],
        on_error: Callable[[str], None] | None = None,
        camera_index: int = 0,
    ) -> None:
        self._on_gesture = on_gesture
        self._on_error = on_error or (lambda _msg: None)
        self._camera_index = camera_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="head-gesture", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Signal the worker to exit without blocking.

        Safe to call from inside an ``on_gesture`` callback: that callback runs on
        the caller's thread (via e.g. Textual's ``call_from_thread``) while the
        worker thread is blocked waiting for it to return, so a ``join`` here would
        deadlock. Use ``stop`` from a different thread to actually reap it.
        """
        self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import cv2
        except ImportError:
            self._on_error(
                "Head-gesture approval needs OpenCV. Install it with: "
                "pip install 'mistral-vibe[gesture]'  (or: pip install opencv-python)"
            )
            return

        cascade_path = (
            cv2.data.haarcascades  # type: ignore[attr-defined]  # valid at runtime, no stub
            + "haarcascade_frontalface_default.xml"
        )
        cascade = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(self._camera_index)
        try:
            if not cap.isOpened() or cascade.empty():
                self._on_error("Head-gesture approval could not open the webcam.")
                return

            xs: deque[float] = deque(maxlen=_WINDOW)
            ys: deque[float] = deque(maxlen=_WINDOW)
            last_fire = 0.0

            while not self._stop.is_set():
                center = _largest_face_center(cv2, cap, cascade)
                if center is None:
                    continue
                xs.append(center[0])
                ys.append(center[1])

                gesture = classify_gesture(list(xs), list(ys))
                now = time.monotonic()
                if gesture and (now - last_fire) > _COOLDOWN_S:
                    last_fire = now
                    xs.clear()
                    ys.clear()
                    self._on_gesture(gesture)
        except Exception:  # camera/driver hiccups shouldn't crash the TUI
            logger.exception("Head-gesture detector crashed")
            self._on_error("Head-gesture approval stopped after an error.")
        finally:
            cap.release()


def _largest_face_center(
    cv2: Any, cap: Any, cascade: Any
) -> tuple[float, float] | None:
    """Read one frame; return the largest face's normalized (x, y) center, or None."""
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        return None
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (fx + fw / 2) / width, (fy + fh / 2) / height
