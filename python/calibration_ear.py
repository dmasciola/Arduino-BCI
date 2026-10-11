"""
calibration_ear.py

Standalone calibration script to find the optimal EAR thresholds
(ear_close_threshold, ear_open_threshold) for a specific user/webcam,
following the same pattern as the EEG calibration_phase() already in use.

Uses the same MediaPipe FaceLandmarker (Tasks API) as eye_state_module.py -
see the note at the top of that file about why the old mp.solutions API
isn't used.

Usage:
    python calibration_ear.py
    (follow the on-screen instructions: eyes open, then eyes closed)

Produces:
    - printed suggested thresholds
    - eye_calibration.json with the values, ready to be loaded by main_integrated.py
"""

import time
import json
import statistics as stats

import cv2

from eye_state_module import create_face_landmarker, compute_ear_from_frame

CAMERA_INDEX = 0
PROC_SIZE = (320, 240)
PHASE_DURATION_S = 6.0   # duration of each data-collection phase
WARMUP_S = 1.0           # discard the first second of each phase (user/webcam settling time)


def collect_ear_samples(cap, landmarker, duration_s, label, session_start, last_ts_holder):
    """
    last_ts_holder is a single-key dict ({"last": ...}) shared across BOTH
    calibration phases. detect_for_video() requires timestamps that are
    strictly increasing for the *entire lifetime* of the landmarker
    instance, not just within one call to this function - so we can't
    reset the clock to zero when starting the "EYES CLOSED" phase while
    reusing the same landmarker from the "EYES OPEN" phase. Instead every
    timestamp is measured relative to one shared session_start.
    """
    print(f"\n>> Phase '{label}': hold this position for {duration_s:.0f} seconds...")
    time.sleep(1.0)  # time to get ready
    samples = []
    phase_start = time.perf_counter()

    while time.perf_counter() - phase_start < duration_s:
        ok, frame = cap.read()
        if not ok:
            continue

        elapsed_session = time.perf_counter() - session_start
        timestamp_ms = max(int(elapsed_session * 1000), last_ts_holder["last"] + 1)
        last_ts_holder["last"] = timestamp_ms

        ear = compute_ear_from_frame(landmarker, frame, PROC_SIZE, timestamp_ms)
        if ear is None:
            continue

        if time.perf_counter() - phase_start < WARMUP_S:
            continue  # discard the warmup period

        samples.append(ear)

    print(f"   Valid samples collected: {len(samples)}")
    return samples


def main():
    print("EAR (Eye Aspect Ratio) Calibration")
    print("Make sure you're in a stable-light environment with a single face in frame.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        return

    landmarker = create_face_landmarker()
    session_start = time.perf_counter()
    last_ts_holder = {"last": -1}

    try:
        open_samples = collect_ear_samples(cap, landmarker, PHASE_DURATION_S, "EYES OPEN",
                                            session_start, last_ts_holder)
        closed_samples = collect_ear_samples(cap, landmarker, PHASE_DURATION_S, "EYES CLOSED",
                                              session_start, last_ts_holder)
    finally:
        cap.release()
        landmarker.close()

    if len(open_samples) < 10 or len(closed_samples) < 10:
        print("\nToo few valid samples: rerun the calibration with better lighting "
              "or make sure your face is clearly visible.")
        return

    mean_open = stats.mean(open_samples)
    mean_closed = stats.mean(closed_samples)
    std_open = stats.stdev(open_samples)
    std_closed = stats.stdev(closed_samples)

    print(f"\nEAR eyes open:   mean={mean_open:.4f}  std={std_open:.4f}")
    print(f"EAR eyes closed: mean={mean_closed:.4f}  std={std_closed:.4f}")

    if mean_open <= mean_closed:
        print("\nWARNING: EAR with eyes open should be greater than with eyes closed. "
              "Check the webcam lighting/position and rerun the calibration.")
        return

    gap = mean_open - mean_closed
    midpoint = (mean_open + mean_closed) / 2.0
    margin = 0.15 * gap  # width of the hysteresis band around the midpoint

    ear_close_threshold = round(midpoint - margin, 4)
    ear_open_threshold = round(midpoint + margin, 4)

    print("\nSuggested thresholds:")
    print(f"  ear_close_threshold = {ear_close_threshold}")
    print(f"  ear_open_threshold  = {ear_open_threshold}")
    print("\nSuggestion for closed_duration_threshold: 1.0-2.0 s is a good starting "
          "point; increase it if you see false positives on slow, natural blinks.")

    config = {
        "ear_close_threshold": ear_close_threshold,
        "ear_open_threshold": ear_open_threshold,
        "closed_duration_threshold": 1.5,
        "mean_open": mean_open,
        "mean_closed": mean_closed,
        "std_open": std_open,
        "std_closed": std_closed,
    }
    with open("eye_calibration.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved to eye_calibration.json")


if __name__ == "__main__":
    main()