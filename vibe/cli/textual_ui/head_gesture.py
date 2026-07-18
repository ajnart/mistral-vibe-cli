"""Head-gesture approval: nod = yes, shake = no, driven by the webcam via OpenCV.

Two independent pieces:
  - ``classify_gesture`` — pure math over recent face-center positions. No camera,
    fully unit-testable (see ``tests/cli/test_head_gesture.py``).
  - ``HeadGestureDetector`` — spawns ``head_gesture_worker`` as a subprocess and
    turns the ``yes``/``no`` lines it prints into ``on_gesture`` callbacks.

Why a subprocess and not just a thread: the worker shows a live preview window
(``cv2.imshow``), and OpenCV's HighGUI *and* macOS camera authorization both
require the process's MAIN thread — neither works from a background thread inside
the Textual app. A child process has its own main thread, so it gets both for free.

Ceiling: Haar-cascade face tracking is cheap but jittery under bad lighting and
loses the face at steep angles. Upgrade path if this ever graduates from
experimental: swap the cascade for a landmark model (e.g. mediapipe FaceMesh) and
classify head *pose* (pitch/yaw) instead of bounding-box centroid drift.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import subprocess
import sys
import threading

from vibe.core.logger import logger

_WORKER_MODULE = "vibe.cli.textual_ui.head_gesture_worker"

# All positions are normalized to frame size (0..1), so thresholds are resolution
# independent. Tuned by hand against a 720p webcam at ~15 fps — recalibrate if the
# gesture feels too twitchy or too sluggish. Shared with the worker process.
_WINDOW = 20  # ~1.3s of frames; the rolling buffer of face centers we classify over
_MIN_SAMPLES = 10  # need enough motion history before we trust a verdict
_AMPLITUDE = 0.09  # min peak-to-peak travel of the face center to count as a gesture
_DOMINANCE = 1.4  # the moving axis must out-travel the other by this factor
_JITTER = 0.008  # per-frame deltas smaller than this are noise (Haar box wobble)
_MIN_REVERSALS = 2  # need a real back-and-forth, not a single drift, to fire
_COOLDOWN_S = 1.5  # ignore new gestures right after one fires (debounce)


def _reversals(vals: list[float]) -> int:
    """Count direction changes in a 1-D track, ignoring sub-jitter wobble.

    A deliberate nod/shake oscillates (down-up-down / left-right-left), giving >= 2
    reversals; a slow drift or a single lean gives 0-1, which we reject.
    """
    signs: list[int] = []
    for a, b in zip(vals, vals[1:], strict=False):
        d = b - a
        if abs(d) < _JITTER:
            continue
        signs.append(1 if d > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def classify_gesture(
    xs: list[float], ys: list[float], amplitude: float = _AMPLITUDE
) -> str | None:
    """Return "yes" (nod), "no" (shake), or None from recent motion tracks.

    xs/ys are oldest-first horizontal/vertical signals. With the Haar fallback these
    are face-center positions (frame fractions); with the YuNet landmark path they
    are head yaw/pitch (nose vs. eyes, inter-ocular-normalized) — hence the tunable
    ``amplitude`` threshold, since the two signals live on different scales.
    """
    if len(ys) < _MIN_SAMPLES:
        return None
    amp_x = max(xs) - min(xs)
    amp_y = max(ys) - min(ys)
    # Vertical oscillation, dominant over horizontal => nod => yes.
    if (
        amp_y > amplitude
        and amp_y > amp_x * _DOMINANCE
        and _reversals(ys) >= _MIN_REVERSALS
    ):
        return "yes"
    # Horizontal oscillation, dominant over vertical => shake => no.
    if (
        amp_x > amplitude
        and amp_x > amp_y * _DOMINANCE
        and _reversals(xs) >= _MIN_REVERSALS
    ):
        return "no"
    return None


class HeadGestureDetector:
    """Runs the webcam preview worker and fires ``on_gesture("yes"|"no")``.

    Callbacks run on a reader thread; the caller is responsible for hopping back to
    its own event loop (Textual: ``app.call_from_thread``).
    """

    def __init__(
        self,
        on_gesture: Callable[[str], None],
        on_error: Callable[[str], None] | None = None,
        camera_index: int = 0,
        prompt_text: str = "",
    ) -> None:
        self._on_gesture = on_gesture
        self._on_error = on_error or (lambda _msg: None)
        self._camera_index = camera_index
        self._prompt_text = prompt_text
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if self._proc is not None:
            return
        # Pass what the agent wants to do to the worker so it can show it in-window.
        env = {**os.environ, "VIBE_GESTURE_PROMPT": self._prompt_text[:200]}
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", _WORKER_MODULE, str(self._camera_index)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered so we see each verdict immediately
                env=env,
            )
        except Exception:
            logger.exception("Failed to launch head-gesture worker")
            self._on_error("Head-gesture approval could not start.")
            return
        self._reader = threading.Thread(
            target=self._read_loop, name="head-gesture-reader", daemon=True
        )
        self._reader.start()

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        # The worker emits one token per line: yes | no | ERR_OPENCV | ERR_CAMERA.
        for raw in proc.stdout:
            line = raw.strip()
            if line in {"yes", "no"}:
                self._on_gesture(line)
            elif line == "ERR_OPENCV":
                self._on_error(
                    "Head-gesture approval needs OpenCV. Install it with: "
                    "pip install 'mistral-vibe[gesture]'  (or: pip install opencv-python)"
                )
            elif line == "ERR_CAMERA":
                self._on_error("Head-gesture approval could not open the webcam.")

    def request_stop(self) -> None:
        """Signal the worker to exit without blocking.

        Safe to call from inside an ``on_gesture`` callback: that callback runs on
        the reader thread (via e.g. Textual's ``call_from_thread``, which blocks the
        reader) so we must not join anything here. ``terminate`` just sends a signal.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
