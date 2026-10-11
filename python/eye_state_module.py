"""
eye_state_module.py

Module for detecting PROLONGED eye closures (not physiological blinks)
using MediaPipe Face Landmarker + Eye Aspect Ratio (EAR), running in a
dedicated thread so it doesn't block the main thread (EEG acquisition +
PyQtGraph GUI).

IMPORTANT - MediaPipe API used here
    Starting around mediapipe 0.10.30, Google REMOVED the legacy
    `mp.solutions.face_mesh` API from the pip package entirely (this is a
    breaking change in MediaPipe itself, not something specific to this
    project - see e.g. github.com/google-ai-edge/mediapipe issues #6192,
    #6204, #6206, #6241). This module therefore uses the modern, actively
    maintained Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`)
    instead.
    The Tasks API needs a small model file (face_landmarker.task, a few MB)
    which is downloaded automatically on first run into the same folder as
    this file. That requires an internet connection the first time; after
    that the file is cached locally and no further download happens.
    If you are stuck on an environment without internet access, the
    alternative is pinning an old mediapipe version that still ships
    `solutions` (e.g. `pip install mediapipe==0.10.21`), but the Tasks API
    used here is the future-proof option and is what this module uses.

Communication with the main thread:
    - SharedEyeState: current state (0/1) readable in a thread-safe way,
      meant to be sampled at every EEG sample (no queue to drain).
    - event_queue (optional, queue.Queue): discrete events for completed
      closures (start, end, duration), useful for a separate log/CSV.

PERFORMANCE NOTES
    - MediaPipe can be heavy if run at 30+ fps alongside a high-frequency
      EEG loop in the same process: the GIL means the two threads compete
      for CPU during pure-Python phases, but most of the MediaPipe/OpenCV
      work happens in native code that releases the GIL, so standard
      threading is usually sufficient. If you notice jitter in EEG
      sampling (which is driven by a busy-wait on the serial port, so very
      sensitive to CPU contention), the options are, from simplest to most
      involved:
        1) Lower target_fps (10-15 fps is more than enough for EAR and
           closures of duration >=1s).
        2) Lower the resolution of the frame passed to MediaPipe (already
           done here: resized to 320x240).
        3) Lower the video thread's priority or use more aggressive
           time.sleep() between frames.
        4) If jitter persists, switch from threading to multiprocessing
           (multiprocessing.Process + multiprocessing.Queue instead of
           threading.Thread + queue.Queue): the video process no longer
           shares the GIL with EEG acquisition. The cost is that
           SharedEyeState would need to be replaced with a
           multiprocessing.Value or a multiprocessing.Manager, and
           communication latency increases slightly (typically negligible
           for a signal with second-level granularity).
"""

import os
import time
import threading
import urllib.request
from collections import namedtuple

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# A completed closure event (from the moment it was CONFIRMED as
# "prolonged", until the eyes reopen)
EyeEvent = namedtuple("EyeEvent", ["start_ts", "end_ts", "duration"])

# Indices of the 6 per-eye points on MediaPipe's face mesh topology
# (478 landmarks with the Tasks API; the first 468 use the same layout as
# the old FaceMesh solution), using the classic dlib-style EAR layout.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def _ensure_model_downloaded():
    """Downloads the FaceLandmarker model on first use, then caches it locally."""
    if os.path.exists(_MODEL_PATH):
        return
    print(f"[EyeStateThread] Downloading face landmark model to {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[EyeStateThread] Model downloaded.")


def create_face_landmarker(min_face_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Creates a FaceLandmarker instance in VIDEO mode, ready to call detect_for_video()."""
    _ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_face_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _eye_aspect_ratio(pts_px, idx):
    p1, p2, p3, p4, p5, p6 = [pts_px[i] for i in idx]
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


def compute_ear_from_frame(landmarker, frame_bgr, proc_size, timestamp_ms):
    """
    Runs FaceLandmarker on a single BGR frame and returns the averaged EAR,
    or None if no face was detected. `timestamp_ms` must be monotonically
    increasing across calls for the same landmarker instance (required by
    the VIDEO running mode).
    """
    w, h = proc_size
    small = cv2.resize(frame_bgr, proc_size)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    pts_px = [(p.x * w, p.y * h) for p in lm]
    return (_eye_aspect_ratio(pts_px, LEFT_EYE_IDX) +
            _eye_aspect_ratio(pts_px, RIGHT_EYE_IDX)) / 2.0


class SharedEyeState:
    """
    Thread-safe container for the latest known eye state.
    The main thread reads it (get) whenever it wants to tag an EEG sample;
    there's no need to drain any queue on every iteration.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = 0     # 0 = open/unknown, 1 = prolonged closure in progress
        self._ts = 0.0       # timestamp (perf_counter relative to t0) of the last update
        self._ear = None     # last filtered EAR value (useful for debugging/plots)

    def set(self, state, ts, ear=None):
        with self._lock:
            self._state = state
            self._ts = ts
            self._ear = ear

    def get(self):
        with self._lock:
            return self._state, self._ts, self._ear


class EyeStateThread(threading.Thread):
    """
    Dedicated thread: captures webcam frames, computes EAR with MediaPipe
    FaceLandmarker, applies smoothing + hysteresis + a minimum-duration
    debounce, and reports ONLY prolonged closures (physiological blinks
    never reach closed_duration_threshold and are therefore "naturally"
    ignored, with no need for separate ad-hoc logic).

    Key parameters to calibrate (see calibration_ear.py):
        ear_close_threshold: below this value the eye is considered closed
        ear_open_threshold:  above this value the eye is considered open
                              (must be > ear_close_threshold: the band
                              between the two thresholds is the hysteresis
                              zone that prevents false flips from noise
                              near the boundary)
        closed_duration_threshold: seconds of continuous closure required
                              before reporting a "prolonged closure"
                              (default 1.5s: typical blinks last 0.1-0.4s)
    """

    def __init__(self,
                 shared_state: SharedEyeState,
                 t0: float,
                 event_queue=None,
                 camera_index: int = 0,
                 target_fps: int = 15,
                 proc_width: int = 320,
                 proc_height: int = 240,
                 ear_close_threshold: float = 0.21,
                 ear_open_threshold: float = 0.24,
                 closed_duration_threshold: float = 1.5,
                 smoothing_window: int = 3,
                 stop_event: threading.Event = None):
        super().__init__(daemon=True, name="EyeStateThread")
        self.shared_state = shared_state
        self.t0 = t0
        self.event_queue = event_queue
        self.camera_index = camera_index
        self.target_fps = max(1, target_fps)
        self.proc_size = (proc_width, proc_height)
        self.ear_close_threshold = ear_close_threshold
        self.ear_open_threshold = ear_open_threshold
        self.closed_duration_threshold = closed_duration_threshold
        self.smoothing_window = smoothing_window
        self.stop_event = stop_event if stop_event is not None else threading.Event()

        self._ear_hist = []
        self._closing_since = None
        self._reported_closed = False
        self._closure_start_ts = None

    def _now(self):
        return time.perf_counter() - self.t0

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[EyeStateThread] Could not open webcam (index %s)." % self.camera_index)
            return

        try:
            landmarker = create_face_landmarker()
        except Exception as exc:
            print(f"[EyeStateThread] Could not initialize FaceLandmarker: {exc}")
            cap.release()
            return

        frame_interval = 1.0 / self.target_fps
        last_tick = 0.0
        last_timestamp_ms = -1

        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                wait = frame_interval - (now - last_tick)
                if wait > 0:
                    time.sleep(wait)
                last_tick = time.perf_counter()

                ok, frame = cap.read()
                if not ok:
                    continue

                ts = self._now()
                # detect_for_video requires strictly increasing timestamps
                timestamp_ms = max(int(ts * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                ear = compute_ear_from_frame(landmarker, frame, self.proc_size, timestamp_ms)
                if ear is None:
                    # No face detected: don't touch the state, to avoid
                    # generating false events from a momentarily lost/occluded frame
                    continue

                self._ear_hist.append(ear)
                if len(self._ear_hist) > self.smoothing_window:
                    self._ear_hist.pop(0)
                smoothed_ear = sum(self._ear_hist) / len(self._ear_hist)

                self._update_state(smoothed_ear, ts)
        finally:
            cap.release()
            landmarker.close()

    def _update_state(self, ear, ts):
        if ear < self.ear_close_threshold:
            if self._closing_since is None:
                self._closing_since = ts

            elapsed_closed = ts - self._closing_since
            if elapsed_closed >= self.closed_duration_threshold and not self._reported_closed:
                self._reported_closed = True
                self._closure_start_ts = self._closing_since
                self.shared_state.set(1, ts, ear)
                if self.event_queue is not None:
                    self.event_queue.put(("closure_start", self._closure_start_ts))

        elif ear > self.ear_open_threshold:
            if self._reported_closed:
                duration = ts - self._closure_start_ts
                self.shared_state.set(0, ts, ear)
                if self.event_queue is not None:
                    self.event_queue.put(EyeEvent(self._closure_start_ts, ts, duration))
            else:
                self.shared_state.set(0, ts, ear)
            self._closing_since = None
            self._reported_closed = False
        # else: inside the hysteresis band -> no state change

    def stop(self):
        self.stop_event.set()
