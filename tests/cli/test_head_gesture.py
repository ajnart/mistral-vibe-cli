"""Self-check for the pure gesture classifier (no camera needed).

Run: uv run pytest tests/cli/test_head_gesture.py
"""

from __future__ import annotations

from vibe.cli.textual_ui.head_gesture import classify_gesture
from vibe.cli.textual_ui.head_gesture_worker import _Session


def _nod() -> tuple[list[float], list[float]]:
    # Face parked at center X, head bobs down then back up (vertical oscillation).
    ys = [0.50, 0.50, 0.54, 0.58, 0.62, 0.60, 0.55, 0.50, 0.48, 0.50]
    xs = [0.50] * len(ys)
    return xs, ys


def _shake() -> tuple[list[float], list[float]]:
    # Face parked at center Y, head turns right then back left (horizontal).
    xs = [0.50, 0.50, 0.54, 0.58, 0.62, 0.60, 0.55, 0.50, 0.48, 0.50]
    ys = [0.50] * len(xs)
    return xs, ys


def test_nod_is_yes() -> None:
    assert classify_gesture(*_nod()) == "yes"


def test_shake_is_no() -> None:
    assert classify_gesture(*_shake()) == "no"


def test_still_face_is_none() -> None:
    flat = [0.5] * 10
    assert classify_gesture(flat, flat) is None


def test_tiny_jitter_is_none() -> None:
    # Sub-threshold wobble on both axes must not trigger a false approval.
    xs = [0.500, 0.501, 0.500, 0.499, 0.500, 0.501, 0.500, 0.499, 0.500, 0.501]
    ys = [0.500, 0.499, 0.500, 0.501, 0.500, 0.499, 0.500, 0.501, 0.500, 0.499]
    assert classify_gesture(xs, ys) is None


def test_too_few_samples_is_none() -> None:
    assert classify_gesture([0.5, 0.6, 0.4], [0.5, 0.6, 0.4]) is None


def test_diagonal_ambiguous_is_none() -> None:
    # Equal travel on both axes: not clearly a nod or a shake, so no verdict.
    diag = [0.50, 0.54, 0.58, 0.62, 0.60, 0.55, 0.50, 0.48, 0.50, 0.52]
    assert classify_gesture(diag, list(diag)) is None


def test_single_lean_is_none() -> None:
    # A one-way lean (down, no return) has only 1 reversal — must not fire.
    ys = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.63, 0.64, 0.65, 0.66]
    xs = [0.50] * len(ys)
    assert classify_gesture(xs, ys) is None


def test_slow_drift_is_none() -> None:
    # Big but monotonic drift (e.g. leaning in): amplitude passes, 0 reversals — reject.
    ys = [0.40 + i * 0.02 for i in range(10)]  # 0.40 -> 0.58, no back-and-forth
    xs = [0.50] * len(ys)
    assert classify_gesture(xs, ys) is None


def _feed_nod(sess: _Session, t0: float) -> str | None:
    # One nod worth of frames (box centered at x=0.5, y oscillating), fed to a session.
    ys = [0.50, 0.50, 0.54, 0.58, 0.62, 0.60, 0.55, 0.50, 0.48, 0.50]
    result = None
    for i, y in enumerate(ys):
        sess.observe((0.45, y - 0.05, 0.10, 0.10))
        result = sess.step(t0 + i * 0.05) or result
    return result


def test_confirmation_needs_two_repeats() -> None:
    # A single nod must not confirm; the second (after the cooldown) does.
    # Large base timestamps mimic time.monotonic() so the initial cooldown passes.
    sess = _Session()
    assert _feed_nod(sess, 1000.0) is None
    assert sess.progress == ("yes", 1)
    assert _feed_nod(sess, 1003.0) == "yes"


def test_mixed_gestures_do_not_accumulate() -> None:
    # A shake then a nod are different gestures — the count resets, no confirmation.
    sess = _Session()
    xs = [0.50, 0.50, 0.54, 0.58, 0.62, 0.60, 0.55, 0.50, 0.48, 0.50]
    for i, x in enumerate(xs):
        sess.observe((x - 0.05, 0.45, 0.10, 0.10))
        sess.step(2000.0 + i * 0.05)
    assert _feed_nod(sess, 2003.0) is None
    assert sess.progress == ("yes", 1)


if __name__ == "__main__":
    test_nod_is_yes()
    test_shake_is_no()
    test_still_face_is_none()
    test_tiny_jitter_is_none()
    test_too_few_samples_is_none()
    test_diagonal_ambiguous_is_none()
    test_single_lean_is_none()
    test_slow_drift_is_none()
    test_confirmation_needs_two_repeats()
    test_mixed_gestures_do_not_accumulate()
    print("all head-gesture classifier checks passed")
