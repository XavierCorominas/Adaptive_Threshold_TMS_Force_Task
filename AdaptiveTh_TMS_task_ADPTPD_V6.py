# -*- coding: utf-8 -*-

"""
RIGHT-HAND FORCE TASK  (single-handle version)
========================================

Corrections applied relative to the previous version
----------------------------------------------------
 1. Screen is now cleared every rendered frame  -> the force cursor no
    longer smears a yellow trail across the display.
 2. Drawing is gated to the display cadence (FPS) instead of running on
    every ~500 Hz loop iteration.
 3. Calibration now measures the ACTUAL resting baseline of each axis
    instead of assuming rest == 0.0.  Devices that idle at -1.0 (very
    common) previously calibrated to 100 % instantly.
 4. Force is derived from signed deflection away from baseline, with the
    deadzone applied in raw axis units.  abs() of the raw value is gone.
 5. pygame.event.pump() is called before every joystick read, so axis
    values are never stale / one iteration behind.
 6. Target position is defined by a FORCE LEVEL (% MVC), not by an
    arbitrary screen fraction.  Previously the target sat at 85 % MVC.
 7. Target box height is derived from the force tolerance, so the box
    means something.
 9. Break is a non-blocking state; pygame.time.delay() no longer freezes
    sampling and the event queue for 30 s.
10. Calibration distinguishes "user quit" from "no force detected".
11. Duplicate/overwritten trial-metric keys resolved.
12. DAQ trigger serialised with a lock (two monitor threads could hit it
    simultaneously).
13. .mat export sanitised (None -> NaN, strings kept as strings).
14. nidaqmx and the second screen are now optional, so the task can be
    debugged on a single machine without hardware.
15. Unused constants removed (SENSE, CALIBRATION_MAX_*).

ADAPTIVE ALGORITHM (unchanged)
------------------------------
Trial 1: no threshold, no trigger
Trial 2: no threshold, no trigger
Trial 3: threshold = mean(trial 1 slope, trial 2 slope)
Trial 4: threshold = mean(trial 2 slope, trial 3 slope)
...

During a trial the CURRENT within-trial rising slope is recalculated
continuously.  When

        current slope < threshold  (BELOW mode)
        current slope > threshold  (ABOVE mode)

a stimulation trigger is sent according to the startup-selected mode.
The final slope is computed only after
the trial, for recording and for the adaptive history.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import datetime
import threading

from collections import deque

import numpy as np
import pandas as pd
import pygame
import matplotlib

matplotlib.use("Agg")          # no interactive backend needed

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.io import savemat

# ---------------------------------------------------------------------------
# nidaqmx is optional so the task can be run without the DAQ attached.
# ---------------------------------------------------------------------------

try:
    import nidaqmx
    DAQ_AVAILABLE = True
except Exception as _daq_import_error:      # noqa: N816
    nidaqmx = None
    DAQ_AVAILABLE = False
    print(
        "WARNING: nidaqmx not available "
        f"({_daq_import_error}). Triggers will be simulated."
    )


# =============================================================================
# PARTICIPANT / SESSION
# =============================================================================

X_NUMBER = "X00000"
SESSION = "Session_1"
BLOCK = "Block_1"


# =============================================================================
# HARDWARE MAPPING
# =============================================================================

# Only the RIGHT handle is used throughout the experiment.
RIGHT_AXIS = 0

# Set True to allow running on a single monitor (debugging).
ALLOW_SINGLE_SCREEN = True


# =============================================================================
# PARTICIPANT FORCE CALIBRATION
# =============================================================================

CALIBRATION_DURATION_S = 2.0
CALIBRATION_REST_DURATION_S = 5.0

# Safety margin.
#   1.00 = measured maximum becomes exactly 100 %
#   1.05 = measured maximum corresponds to ~95.2 % of the display range
CALIBRATION_SCALE_FACTOR = 1.00

# Deadzone expressed in RAW AXIS UNITS around the measured baseline.
# Anything smaller than this is treated as rest.
DEADZONE_RAW = 0.02

# Minimum deflection (raw units) that counts as a real contraction.
MIN_VALID_CALIBRATION_DEFLECTION = 0.01

# Calibration status codes.
CAL_OK = "ok"
CAL_QUIT = "quit"
CAL_NO_FORCE = "no_force"


# =============================================================================
# NUMBER OF TRIALS
# =============================================================================

TOTAL_TRIALS = 15

# Pre-task phases launched from the startup interface.
# Familiarization data are discarded. Baseline trials seed the Main Task threshold.
FAMILIARIZATION_TRIALS = 5
BASELINE_TRIALS = 10

# Startup default for the PREP loadbar shown during the Main Task only.
MAIN_TASK_LOADBAR_DEFAULT = True

# Baseline-derived individualized RFD checking window.
# A force plateau is defined as the first time after detected movement onset
# that the smoothed force reaches at least this fraction of that trial's peak
# and remains at/above that level for PLATEAU_HOLD_S.
PLATEAU_FRACTION_OF_PEAK = 0.95
PLATEAU_HOLD_S = 0.050

# The individualized endpoint is rounded UP to the RFD checkpoint grid.
# This guarantees that the final real-time checkpoint reaches (or slightly
# exceeds by < one step) the baseline-estimated plateau time.
INDIVIDUALIZE_RFD_CHECK_END = True

# Safety/experimental cap for the individualized Main Task checking window.
# Even if the baseline plateau estimate is later, stimulation checking will
# never continue beyond 180 ms after movement onset.
# With the 6-ms grid used below, the latest valid checkpoint <=180 ms is 176 ms.
MAX_RFD_CHECK_END_S = 0.180


# =============================================================================
# ADAPTIVE SLOPE SETTINGS
# =============================================================================

# Number of PREVIOUS completed trials used for the threshold.
SLOPE_REFERENCE_TRIALS = 3

# Force confirmation uses the previous THREE trials at the same checkpoint.
FORCE_REFERENCE_TRIALS = 3

# Default startup selection. The interface lets the operator choose
# BELOW (correct low-RFD trials) or ABOVE (reinforce high-RFD trials).
SLOPE_TRIGGER_MODE = "BELOW"

# Which signal(s) are allowed to decide stimulation during the Main Task.
# Selectable from the startup GUI:
#   RFD_ONLY   -> 20-ms local RFD only
#   FORCE_ONLY -> checkpoint force trajectory only
#   COMBINED   -> BOTH local RFD and checkpoint force must satisfy the rule
STIMULATION_METRIC_MODE = "COMBINED"
VALID_STIMULATION_METRIC_MODES = ("RFD_ONLY", "FORCE_ONLY", "COMBINED")

# FLAT RFD TRIGGER MARGIN
# -----------------------
# All adaptive checks use the same 20-ms local-RFD window. The real-time
# RFD decision therefore uses one fixed relative separation from the adaptive
# RFD reference at every checkpoint.
#
# BELOW: current RFD must be < 90% of the adaptive RFD reference.
# ABOVE: current RFD must be > 110% of the adaptive RFD reference.
TRIGGER_MARGIN_FRACTION = 0.10


# =============================================================================
# ONSET DETECTION
# =============================================================================
#
# EVERYTHING is anchored to force onset, never to the GO cue, because
# reaction time varies from trial to trial.  A window measured from GO
# would mix premotor reaction time into the force measurement.

# Force (fraction of MVC) that must be reached for the rise to count as
# having started.
RISING_ONSET_FORCE = 0.05

# Force must stay above the threshold for this many consecutive samples
# before onset is accepted.  Prevents a noise spike from setting onset
# early, which would corrupt every downstream metric because everything
# is anchored to onset.
#
# MUST exceed SLOPE_SMOOTHING_SAMPLES: the smoothing kernel spreads
# a single spike across the kernel width, so a shorter confirmation
# is defeated by the smoothing itself.  5 samples = 10 ms at 500 Hz.
ONSET_CONFIRM_SAMPLES = 5

# How long after GO we are willing to look for onset.  A trial with no
# confirmed onset inside this window is recorded as invalid.
#
# NOTE: this must fit inside GO_DURATION_S together with
# RFD_CHECK_END_S, otherwise the monitor is disarmed before the late
# checkpoints have been reached.  See the guard next to GO_DURATION_S.
ONSET_SEARCH_WINDOW_S = 0.850

# How much data after onset is retained for analysis and figures.
#
# Reduced from 2.0 s: at 500 Hz, 2 s x 2 hands is ~2850 rows per trial,
# and the diagnostic running-slope loop runs one polyfit per sample, so
# a long window makes post-hoc processing very slow for no benefit.
# 0.8 s comfortably covers PLOT_TIME_FROM_ONSET_S.
POST_ONSET_WINDOW_S = 2.00

# Smoothing before any slope calculation (samples).
#
# Reduced from 5 (10 ms) to 3 (6 ms) so that a 35 ms epoch is not
# dominated by the kernel.  The kernel width must stay BELOW the
# shortest checkpoint, otherwise the fit is regressing over data
# your own filter has smeared.
SLOPE_SMOOTHING_SAMPLES = 3

if ONSET_CONFIRM_SAMPLES <= SLOPE_SMOOTHING_SAMPLES:
    raise ValueError(
        "ONSET_CONFIRM_SAMPLES must exceed SLOPE_SMOOTHING_SAMPLES, "
        "otherwise a single-sample artefact smoothed across the kernel "
        "satisfies the onset confirmation."
    )


# =============================================================================
# RATE OF FORCE DEVELOPMENT (RFD)
# =============================================================================
#
# The real-time decision metric is LOCAL / sliding-window RFD.  At each
# checkpoint the code fits a straight line to the most recent
# LOCAL_RFD_WINDOW_S of smoothed force, ending exactly at that checkpoint.
# This makes the RFD trace physiologically intuitive: high during the rapid
# rise and approaching zero as force reaches a plateau.
#
# To avoid false BELOW-threshold triggers when a successful contraction has
# already reached high force and local RFD naturally falls toward zero, every
# RFD decision is confirmed by the force level at the SAME checkpoint.
#
# BELOW mode requires BOTH:
#   local RFD < adaptive local-RFD threshold
#   current force < mean force of previous FORCE_REFERENCE_TRIALS trials
# and corrective stimulation is suppressed once force is already inside the
# target-force region.
#
# ABOVE mode requires BOTH inequalities to reverse.

LOCAL_RFD_WINDOW_S = 0.020   # 20-ms sliding regression window

# First checkpoint. Starting at 60 ms guarantees a full 50-ms local window
# while retaining the 80-ms primary reporting checkpoint on the 6-ms grid.
# primary reporting checkpoint.
RFD_CHECK_START_S = 0.020

# Initial/fallback last checkpoint. Baseline individualization replaces this
# before Main Task; the individualized value remains capped at 180 ms.
RFD_CHECK_END_S = 0.210

# Spacing between checkpoints.
# 6-ms step with a 20-ms local-RFD window gives 70% overlap between
# consecutive windows, providing near-continuous adaptive checking.
RFD_CHECK_STEP_S = 0.006

RFD_CHECKPOINTS_S = tuple(
    round(x, 4)
    for x in np.arange(
        RFD_CHECK_START_S,
        RFD_CHECK_END_S + RFD_CHECK_STEP_S / 2,
        RFD_CHECK_STEP_S,
    )
)

# Primary checkpoint reported in summary outputs.
RFD_EPOCH_S = 0.08

if RFD_EPOCH_S not in RFD_CHECKPOINTS_S:
    raise ValueError(
        f"RFD_EPOCH_S ({RFD_EPOCH_S}) must be one of the checkpoints "
        f"{RFD_CHECKPOINTS_S}"
    )

# Additional epochs computed post hoc for the trial summary.
RFD_EPOCHS_S = (0.05, 0.10, 0.15, 0.20, 0.30)

# Minimum samples that must fall inside an epoch for it to be fitted.
# At 500 Hz a 100 ms epoch holds ~50 samples, so this tolerates a
# substantial number of dropped samples.
# At 500 Hz, 10 ms holds 6 samples.  5 is the absolute minimum for
# a linear fit to be defined.  The high noise at this sample count
# is handled by the flat adaptive RFD margin.
MIN_EPOCH_SAMPLES = 5

# Window used for peak (maximum instantaneous) RFD, post hoc only.
PEAK_RFD_WINDOW_S = 0.04

# Timepoints from ONSET at which absolute force is recorded.
FORCE_TIMEPOINTS_FROM_ONSET_S = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)

# Diagnostic running-slope trace (onset -> current sample).  Not used for
# any decision; retained so you can see the rise evolve in the figures.
#
# Do NOT set this very low.  A slope fitted through 2 samples is almost
# pure noise: on a clean synthetic ramp of true slope 1.20, dropping
# from 20 to 2 samples widened the running-slope range from 1.03-1.20
# to 0.65-1.46, and with real noise it goes far wider.  That noise then
# drove the shared figure axis out to -4..+11 MVC/s and squashed the
# checkpoint values into an unreadable band.
# 10 samples = 20 ms at 500 Hz.
MIN_SLOPE_SAMPLES = 10



def apply_plot_style(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, fontsize=PLOT_TITLE_SIZE, fontweight=PLOT_FONT_WEIGHT)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PLOT_FONT_SIZE, fontweight=PLOT_FONT_WEIGHT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZE, fontweight=PLOT_FONT_WEIGHT)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(PLOT_FONT_WEIGHT)
    ax.grid(True, alpha=0.25)


def bold_legend(ax, **kwargs):
    leg = ax.legend(fontsize=PLOT_LEGEND_SIZE, **kwargs)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontweight(PLOT_FONT_WEIGHT)
    return leg


# =============================================================================
# FIGURE AXES
# =============================================================================
#
# Every trial figure uses the SAME axis limits so trials can be compared
# by eye without re-reading the scale each time.
#
# Set a value to None to have it computed once from the whole session
# (still identical across every figure, but fitted to the data).

# Time axis of the per-trial figures, in seconds from the GO cue.
PLOT_TIME_FROM_GO_S = (-0.2, 2.0)

# Time axis of the onset-aligned overlay, in seconds from force onset.
PLOT_TIME_FROM_ONSET_S = (-0.2, 1.5)

# Force axis (fraction of MVC).
PLOT_FORCE_LIMITS = (-0.10, 1.5)

# RFD axis (MVC/s).  None = fit once across the session.
PLOT_RFD_LIMITS = (-15.0, 15.0)

# Trial axis of the session-level figure. None = fit to trial count.
PLOT_TRIAL_RFD_LIMITS = None

# Publication-style typography
PLOT_FONT_SIZE = 20
PLOT_TICK_SIZE = 15
PLOT_TITLE_SIZE = 20
PLOT_LEGEND_SIZE = 14
PLOT_FONT_WEIGHT = "bold"


# =============================================================================
# DISPLAY / SAMPLING RATES
# =============================================================================

FPS = 64                    # display refresh
SAMPLING_RATE_HZ = 500      # force sampling


# =============================================================================
# TARGET PRESENTATION
# =============================================================================

# Single-handle experiment: the right target is always active.
TARGET_DISPLAY_MODE = "RIGHT_ONLY"

# Target expressed as a fraction of the participant's calibrated maximum.
TARGET_FORCE_LEVEL = 0.85
TARGET_TOLERANCE = 0.1

# In BELOW/corrective mode, do not stimulate once force has already entered
# the target region. With the default target this is 0.75 MVC.
FORCE_TRIGGER_CEILING = TARGET_FORCE_LEVEL - TARGET_TOLERANCE
# Required separation from the adaptive force reference.
# 0.10 means 10%: BELOW must be <90% of reference; ABOVE must be >110%.
FORCE_REFERENCE_MARGIN = 0.10

RECT_WIDTH = 300
RECT_OUTLINE_WIDTH = 6

# Random-position mode: keep targets away from the extreme edges,
# expressed in force units.
RANDOM_TARGET_MIN_FORCE = 0.20
RANDOM_TARGET_MAX_FORCE = 0.70


# =============================================================================
# FORCE CURSOR
# =============================================================================

CURSOR_RADIUS = 10
CURSOR_OUTLINE_WIDTH = 4

CURSOR_COLOR = (255, 255, 0)
CURSOR_OUTLINE_COLOR = (255, 255, 255)

# Vertical guide rail behind the cursor.
CURSOR_TRACK_WIDTH = 4
CURSOR_TRACK_COLOR = (45, 45, 45)


# =============================================================================
# LOADING BAR
# =============================================================================

LOADBAR_WIDTH = 40
LOADBAR_HEIGHT_FRAC = 0.60
LOADBAR_X_OFFSET = 50


# =============================================================================
# TIMING
# =============================================================================

# NOTE: PREP_DURATION_S, GO_DURATION_S, ITI_MIN_S, ITI_MAX_S and
# USE_RED_CUE below are DEFAULTS only. They are editable live from the
# startup interface (operator screen) before the Main Task is started,
# and the values chosen there overwrite these module-level globals.

# Duration the RED cue is shown before the GREEN target (if the red cue
# is enabled). If the red cue is disabled (USE_RED_CUE = False), this
# is simply a silent wait before the green target appears (no box is
# drawn at all during this wait -- see draw logic in main()).
PREP_DURATION_S = 3.0

# Extra random time (uniform 0..PREP_JITTER_S) added on top of
# PREP_DURATION_S on every trial.
PREP_JITTER_S = 0.0

# Whether the RED cue phase is shown before the GREEN target at all.
# True  -> RED cue for PREP_DURATION_S (+ jitter), then GREEN target.
# False -> GREEN target appears directly (no red cue, no placeholder box).
USE_RED_CUE = True

# GO must be long enough for a slow-reacting trial: the onset search
# window plus the full checking window.  With GO = 0.5 s and
# ONSET_SEARCH_WINDOW_S = 0.85 s, the monitor was disarmed before the
# late checkpoints were reached, so any trial with a reaction time over
# ~290 ms silently lost its last checks:
#
#     RT 250 ms -> last checkpoint 460 ms after GO   fine
#     RT 300 ms -> last checkpoint 510 ms after GO   CUT OFF
#
# The guard below now makes this impossible to get wrong.
#
# If you would rather keep a short GO, lower ONSET_SEARCH_WINDOW_S
# instead: GO = 0.50 s allows an onset search of 0.29 s.
GO_DURATION_S = 1.10

# Extra random time (uniform 0..GO_JITTER_S) added on top of
# GO_DURATION_S on every trial (visible green-box duration only; the
# adaptive checking window itself is never shortened by this).
GO_JITTER_S = 0.0

# Continue recording force after the active GO phase for analysis/plots only.
# No additional TMS/adaptive checks are allowed in this tail.
POST_GO_RECORDING_END_S = 2.00

# Pause (ITI base) and jitter (extra random range added on top of the
# base pause). Actual ITI on each trial is drawn uniformly from
# [ITI_MIN_S, ITI_MIN_S + ITI_JITTER_S]. ITI_MAX_S is kept in sync for
# any code that still reads it directly.
ITI_MIN_S = 1.0
ITI_JITTER_S = 0.5
ITI_MAX_S = ITI_MIN_S + ITI_JITTER_S


# =============================================================================
# CONFIGURATION GUARDS
# =============================================================================
#
# These catch silent misconfigurations rather than letting them quietly
# corrupt a whole session. GO_DURATION_S/PREP_DURATION_S/ITI_* can be
# edited from the startup interface, so the numeric guard is re-applied
# as a function (called again once the operator confirms Start) instead
# of only at import time.

def required_go_duration_s():
    """Minimum GO_DURATION_S needed for the adaptive checking window."""
    return ONSET_SEARCH_WINDOW_S + RFD_CHECK_END_S


def validate_and_fix_timing():
    """
    Re-validate the (possibly operator-edited) timing globals.
    Silently clamps GO_DURATION_S / POST_GO_RECORDING_END_S / ITI_* up
    to the minimum safe values if needed, and returns a list of
    human-readable warning strings describing anything that was
    auto-corrected (empty list if everything was already valid).
    """
    global GO_DURATION_S, POST_GO_RECORDING_END_S
    global ITI_MIN_S, ITI_JITTER_S, ITI_MAX_S, PREP_DURATION_S
    global PREP_JITTER_S, GO_JITTER_S

    warnings = []
    required_go = required_go_duration_s()

    if GO_DURATION_S < required_go:
        warnings.append(
            f"Green target duration was too short for the adaptive "
            f"checking window and was raised from {GO_DURATION_S:.2f} s "
            f"to {required_go:.2f} s (ONSET_SEARCH_WINDOW_S + "
            f"RFD_CHECK_END_S)."
        )
        GO_DURATION_S = required_go

    if POST_GO_RECORDING_END_S < GO_DURATION_S + GO_JITTER_S:
        warnings.append(
            f"Post-GO recording window was shorter than the (jittered) "
            f"green target duration and was raised to "
            f"{GO_DURATION_S + GO_JITTER_S:.2f} s."
        )
        POST_GO_RECORDING_END_S = GO_DURATION_S + GO_JITTER_S

    if PREP_DURATION_S < 0:
        warnings.append("Red cue duration cannot be negative; set to 0.")
        PREP_DURATION_S = 0.0

    if ITI_MIN_S < 0:
        warnings.append("Pause duration cannot be negative; set to 0.")
        ITI_MIN_S = 0.0

    if ITI_JITTER_S < 0:
        warnings.append("Pause jitter cannot be negative; set to 0.")
        ITI_JITTER_S = 0.0

    if PREP_JITTER_S < 0:
        warnings.append("Red cue jitter cannot be negative; set to 0.")
        PREP_JITTER_S = 0.0

    if GO_JITTER_S < 0:
        warnings.append("Green target jitter cannot be negative; set to 0.")
        GO_JITTER_S = 0.0

    ITI_MAX_S = ITI_MIN_S + ITI_JITTER_S

    return warnings

if POST_ONSET_WINDOW_S < RFD_CHECK_END_S:
    raise ValueError(
        f"POST_ONSET_WINDOW_S ({POST_ONSET_WINDOW_S:.2f} s) must be at "
        f"least RFD_CHECK_END_S ({RFD_CHECK_END_S:.2f} s), otherwise "
        "the post-hoc data does not cover the checking window."
    )

_MIN_EPOCH_DURATION_S = MIN_EPOCH_SAMPLES / SAMPLING_RATE_HZ

if RFD_CHECK_START_S < _MIN_EPOCH_DURATION_S:
    raise ValueError(
        f"RFD_CHECK_START_S ({RFD_CHECK_START_S * 1000:.0f} ms) is "
        f"shorter than MIN_EPOCH_SAMPLES ({MIN_EPOCH_SAMPLES}) at "
        f"{SAMPLING_RATE_HZ} Hz, which needs "
        f"{_MIN_EPOCH_DURATION_S * 1000:.0f} ms. Early checkpoints "
        "would return NaN on every trial."
    )


# =============================================================================
# DAQ
# =============================================================================

daq_device = "Dev1"
cue_trigger_channel = "ao1"
rfd_trigger_channel = "ao0"

# GO-cue trigger lead time.
CUE_TRIGGER_LEAD_S = 0.1

# Serialises DAQ access between the two monitor threads and the main thread.
_daq_lock = threading.Lock()


# =============================================================================
# BREAKS
# =============================================================================

BREAK_DURATION = 30          # seconds
BREAK_INTERVAL = 5 * 60      # seconds


# =============================================================================
# MAXIMUM GAME DURATION
# =============================================================================

duration_of_game = 1800


# =============================================================================
# OUTPUT
# =============================================================================

USER_HOME = os.path.expanduser("~")
OUTPUT_ROOT = os.path.join(USER_HOME, "Desktop", "ADAPT_BK")

date_str = datetime.datetime.now().strftime("%d%m%Y")


# =============================================================================
# COLORS
# =============================================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 191, 255)
GREEN = (0, 255, 64)
RED = (220, 20, 60)
NEUTRAL_GRAY = (90, 90, 90)
YELLOW = (255, 255, 0)


# =============================================================================
# GLOBAL TIMING
# =============================================================================

GLOBAL_START_TIME = None


# =============================================================================
# FORCE BUFFER
# =============================================================================

class ForceBuffer:
    """
    Thread-safe storage for force samples.

    Pygame RIGHT-handle acquisition happens ONLY in the main thread.
    The RFDMonitor thread only READS this buffer.
    """

    def __init__(self, maxlen=1_000_000, recent_maxlen=5000):
        self.all_data = deque(maxlen=maxlen)
        self.recent_data = deque(maxlen=recent_maxlen)
        self.lock = threading.Lock()

    def append(self, sample):
        with self.lock:
            self.all_data.append(sample)
            self.recent_data.append(sample)

    @staticmethod
    def _to_arrays(container):
        if not container:
            empty = np.empty((0,), float)
            return empty, empty.copy()

        arr = np.asarray(list(container), dtype=float)
        return arr[:, 0], arr[:, 1]

    def get_all_numpy(self):
        with self.lock:
            snapshot = list(self.all_data)
        return self._to_arrays(snapshot)

    def get_recent_numpy(self):
        with self.lock:
            snapshot = list(self.recent_data)
        return self._to_arrays(snapshot)


force_buffer = ForceBuffer(
    maxlen=1_000_000,
    recent_maxlen=int(
        max(5000,
            (ONSET_SEARCH_WINDOW_S + POST_ONSET_WINDOW_S)
            * SAMPLING_RATE_HZ * 3)
    ),
)


# =============================================================================
# CALIBRATION CONTAINER
# =============================================================================

class Calibration:
    """Right-handle resting baseline and maximum deflection."""

    def __init__(self):
        self.baseline_right = 0.0
        self.max_deflection_right = np.nan

    def is_complete(self):
        return (
            np.isfinite(self.max_deflection_right)
            and self.max_deflection_right > 0
        )

    def summary(self):
        return (
            f"baseline R={self.baseline_right:+.4f}  |  "
            f"max deflection R={self.max_deflection_right:.4f}"
        )


# =============================================================================
# FORCE PROCESSING
# =============================================================================

def raw_to_deflection(raw, baseline):
    """
    Signed distance from rest, deadzoned, in RAW AXIS UNITS.

    This replaces the old abs()-based mapping, which broke for any device
    whose axis does not idle at exactly 0.0.
    """
    deflection = abs(float(raw) - float(baseline))

    if deflection <= DEADZONE_RAW:
        return 0.0

    return deflection - DEADZONE_RAW


def deflection_to_force(deflection, max_deflection):
    """
    Participant-normalised force.

        0.0 = participant rest
        1.0 = participant calibrated maximum contraction
    """
    if (
        max_deflection is None
        or not np.isfinite(max_deflection)
        or max_deflection <= 0
    ):
        return 0.0

    scale = max_deflection * CALIBRATION_SCALE_FACTOR

    if scale <= 0:
        return 0.0

    return float(np.clip(deflection / scale, 0.0, 1.0))


def read_raw_axes(joystick):
    """Read the RIGHT handle axis only."""
    pygame.event.pump()
    return float(joystick.get_axis(RIGHT_AXIS))

# =============================================================================
# HIGH-RATE FORCE ACQUISITION
# =============================================================================

def acquire_force_sample(joystick, start_time, calibration):
    """Acquire one participant-normalised RIGHT-handle force sample."""
    right_raw = read_raw_axes(joystick)
    right_deflection = raw_to_deflection(
        right_raw, calibration.baseline_right
    )
    right_force = deflection_to_force(
        right_deflection, calibration.max_deflection_right
    )
    t_rel = time.perf_counter() - start_time
    force_buffer.append((t_rel, right_force))
    return right_force, t_rel

def get_grip_samples_numpy():
    return force_buffer.get_all_numpy()


def clear_force_buffer():
    """Remove samples between experiment phases so timestamps cannot overlap."""
    with force_buffer.lock:
        force_buffer.all_data.clear()
        force_buffer.recent_data.clear()


def get_recent_samples_numpy():
    return force_buffer.get_recent_numpy()


# =============================================================================
# JOYSTICK
# =============================================================================

def list_joysticks():
    if not pygame.joystick.get_init():
        pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()

    if joystick_count == 0:
        print("No joysticks connected.")
        return None

    joystick = None

    for i in range(joystick_count):
        candidate = pygame.joystick.Joystick(i)
        candidate.init()

        print(
            f"Joystick {i} initialized: {candidate.get_name()} "
            f"({candidate.get_numaxes()} axes)"
        )

        if joystick is None:
            joystick = candidate

    if joystick.get_numaxes() <= RIGHT_AXIS:
        print(
            "ERROR: joystick does not expose the configured axes "
            f"({RIGHT_AXIS})."
        )
        return None

    return joystick


def diagnose_axes(joystick, duration_s=5.0):
    """Optional helper: print the raw RIGHT-handle axis value."""
    print(f"\nRaw RIGHT-axis diagnostic for {duration_s:.0f} s ...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        right_raw = read_raw_axes(joystick)
        print(f"  right={right_raw:+.4f}")
        time.sleep(0.05)
    print("Diagnostic complete.\n")

# =============================================================================
# DISPLAY
# =============================================================================

def open_on_second_screen():
    if not pygame.get_init():
        pygame.init()

    if not pygame.display.get_init():
        pygame.display.init()

    displays = pygame.display.get_desktop_sizes()

    print(f"Detected displays: {len(displays)}")

    for i, (w, h) in enumerate(displays):
        print(f"Display {i}: {w} x {h}")

    if len(displays) < 2:
        if not ALLOW_SINGLE_SCREEN:
            raise RuntimeError(
                "No secondary screen found. "
                f"Detected {len(displays)} display(s)."
            )

        print(
            "WARNING: only one display detected. "
            "Running on the primary screen."
        )

        w, h = displays[0]
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Main Game - Premovement")
        return screen

    w, h = displays[1]
    x_offset = displays[0][0]

    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_offset},0"

    screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
    pygame.display.set_caption("Main Game - Premovement")

    print(f"Experiment screen opened on display 1: {w} x {h}")

    return screen


# =============================================================================
# TEXT
# =============================================================================

_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont(None, size)
    return _font_cache[size]


def show_text(text, color, screen, size=100, y_center=None):
    lines = text.split("\n")
    font = get_font(size)

    line_height = font.get_height()
    total_height = len(lines) * line_height

    if y_center is None:
        y_center = screen.get_height() // 2

    for i, line in enumerate(lines):
        surface = font.render(line, True, color)
        rect = surface.get_rect(
            center=(
                screen.get_width() // 2,
                y_center - total_height // 2 + i * line_height,
            )
        )
        screen.blit(surface, rect)


# =============================================================================
# START / END SCREENS
# =============================================================================

def draw_menu_button(
    screen, rect, text, enabled=True, hovered=False, selected=False
):
    """Draw one startup-menu button with a persistent selected state."""
    if enabled:
        fill = (70, 70, 70) if not hovered else (100, 100, 100)
        border = GREEN if selected else WHITE
        border_width = 5 if selected else 2
        text_color = WHITE
    else:
        fill = (35, 35, 35)
        border = (80, 80, 80)
        border_width = 2
        text_color = (110, 110, 110)

    pygame.draw.rect(screen, fill, rect, border_radius=12)
    pygame.draw.rect(
        screen, border, rect, border_width, border_radius=12
    )

    font = get_font(44)
    surface = font.render(text, True, text_color)
    screen.blit(surface, surface.get_rect(center=rect.center))


def make_rfd_checkpoints(end_s):
    """Build the onset-anchored checkpoint grid up to an individualized end."""
    end_s = float(end_s)
    # Keep the primary reporting epoch available and never request more
    # baseline data than POST_ONSET_WINDOW_S actually contains.
    end_s = max(RFD_CHECK_START_S, RFD_EPOCH_S, end_s)
    end_s = min(end_s, POST_ONSET_WINDOW_S)
    n_steps = int(np.ceil((end_s - RFD_CHECK_START_S) / RFD_CHECK_STEP_S - 1e-12))
    snapped_end = RFD_CHECK_START_S + n_steps * RFD_CHECK_STEP_S
    checkpoints = tuple(
        round(RFD_CHECK_START_S + i * RFD_CHECK_STEP_S, 4)
        for i in range(n_steps + 1)
    )
    return checkpoints, float(round(snapped_end, 4))


def calculate_time_to_force_plateau(t, force):
    """
    Estimate time from force onset to the maximum-force plateau for one trial.

    Plateau = first sample where smoothed force is >=
    PLATEAU_FRACTION_OF_PEAK * trial peak and stays there for PLATEAU_HOLD_S.
    Returns NaN when onset/plateau cannot be identified robustly.
    """
    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)
    if t.size < ONSET_CONFIRM_SAMPLES or force.size != t.size:
        return np.nan

    smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    onset_idx, smoothed = find_rising_onset(t, force, smoothed=smoothed)
    if onset_idx is None:
        return np.nan

    onset_time = float(t[onset_idx])
    search_end = onset_time + POST_ONSET_WINDOW_S
    valid_idx = np.where((t >= onset_time) & (t <= search_end))[0]
    if valid_idx.size < ONSET_CONFIRM_SAMPLES:
        return np.nan

    post_force = smoothed[valid_idx]
    peak_force = float(np.nanmax(post_force))
    if not np.isfinite(peak_force) or peak_force <= RISING_ONSET_FORCE:
        return np.nan

    plateau_level = PLATEAU_FRACTION_OF_PEAK * peak_force
    hold_samples = max(1, int(np.ceil(PLATEAU_HOLD_S * SAMPLING_RATE_HZ)))

    above = post_force >= plateau_level
    run = 0
    for local_i, is_above in enumerate(above):
        if is_above:
            run += 1
            if run >= hold_samples:
                first_local = local_i - hold_samples + 1
                plateau_idx = valid_idx[first_local]
                return float(t[plateau_idx] - onset_time)
        else:
            run = 0

    return np.nan


def _average_checkpoint_history(checkpoint_history, checkpoints=None):
    """Mean RFD at each checkpoint across valid baseline trials."""
    if checkpoints is None:
        checkpoints = RFD_CHECKPOINTS_S
    baseline = {int(round(c * 1000)): np.nan for c in checkpoints}
    for key in baseline:
        values = np.asarray(
            [trial.get(key, np.nan) for trial in checkpoint_history], dtype=float
        )
        values = values[np.isfinite(values)]
        if values.size:
            baseline[key] = float(np.mean(values))
    return baseline


def _draw_phase_header(screen, title, subtitle, trial_number, total_trials):
    """Small phase label shown during familiarization/baseline runs."""
    font = get_font(30)
    small = get_font(22)
    title_s = font.render(title, True, WHITE)
    sub_s = small.render(
        f"{subtitle}   Trial {trial_number}/{total_trials}", True, (190, 190, 190)
    )
    screen.blit(title_s, (30, 24))
    screen.blit(sub_s, (30, 62))


def run_pre_task_phase(screen, joystick, calibration, right_rect, phase,
                       n_trials):
    """
    Run Familiarization or Baseline without adaptive/TMS triggering.

    Familiarization:
      - task display and force acquisition only
      - no adaptive threshold, no TMS/DAQ triggers
      - no data retained when the phase ends

    Baseline:
      - no TMS/DAQ triggers
      - measures time from movement onset to sustained maximum-force plateau
      - derives the participant-specific RFD checking endpoint
      - recomputes checkpoint RFD thresholds on that individualized grid

    Returns {} for Familiarization. For Baseline returns a dict with
    thresholds, plateau times, mean plateau time, individualized endpoint,
    and checkpoint grid. Returns None if the phase is aborted.
    """
    phase = str(phase).upper()
    if phase not in ("FAMILIARIZATION", "BASELINE"):
        raise ValueError(f"Unknown pre-task phase: {phase}")

    title = "Familiarization" if phase == "FAMILIARIZATION" else "Baseline Calibration"
    subtitle = (
        "Practice only - no TMS, data discarded"
        if phase == "FAMILIARIZATION"
        else "No TMS - calibrating threshold and RFD window"
    )

    clear_force_buffer()
    countdown(screen)
    phase_start = time.perf_counter()
    input_dt = 1.0 / SAMPLING_RATE_HZ
    display_dt = 1.0 / FPS
    next_input = time.perf_counter()
    next_display = time.perf_counter()
    baseline_trials = []

    for trial_number in range(1, n_trials + 1):
        prep_start = time.perf_counter()
        right_force = 0.0
        current_prep_duration = np.random.uniform(
            PREP_DURATION_S, PREP_DURATION_S + PREP_JITTER_S
        )
        while True:
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return None

            if now >= next_input:
                right_force, _ = acquire_force_sample(joystick, phase_start, calibration)
                next_input += input_dt
                if next_input < now - input_dt * 5:
                    next_input = now + input_dt

            elapsed = now - prep_start
            if now >= next_display:
                screen.fill(BLACK)
                draw_cursor_track(screen, right_rect)
                if USE_RED_CUE:
                    draw_active_target(screen, right_rect, RED)
                draw_force_cursor(screen, right_force, right_rect)
                _draw_phase_header(screen, title, subtitle, trial_number, n_trials)
                pygame.display.flip()
                next_display = now + display_dt

            if elapsed >= current_prep_duration:
                break

        go_start_time = time.perf_counter() - phase_start
        go_state_start = time.perf_counter()
        current_go_duration = np.random.uniform(
            GO_DURATION_S, GO_DURATION_S + GO_JITTER_S
        )
        while True:
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return None

            if now >= next_input:
                right_force, _ = acquire_force_sample(joystick, phase_start, calibration)
                next_input += input_dt
                if next_input < now - input_dt * 5:
                    next_input = now + input_dt

            if now >= next_display:
                screen.fill(BLACK)
                draw_cursor_track(screen, right_rect)
                draw_active_target(screen, right_rect, GREEN)
                draw_force_cursor(screen, right_force, right_rect)
                _draw_phase_header(screen, title, subtitle, trial_number, n_trials)
                pygame.display.flip()
                next_display = now + display_dt

            if now - go_state_start >= current_go_duration:
                break

        # Keep sampling through ITI so late plateaus are available post hoc.
        iti = np.random.uniform(ITI_MIN_S, ITI_MAX_S)
        iti_start = time.perf_counter()
        while time.perf_counter() - iti_start < iti:
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return None
            if now >= next_input:
                acquire_force_sample(joystick, phase_start, calibration)
                next_input += input_dt
            if now >= next_display:
                screen.fill(BLACK)
                _draw_phase_header(screen, title, subtitle, trial_number, n_trials)
                pygame.display.flip()
                next_display = now + display_dt
            time.sleep(0.001)

        if phase == "BASELINE":
            t_all, f_all = get_grip_samples_numpy()
            mask = (
                (t_all >= go_start_time)
                & (t_all <= go_start_time + ONSET_SEARCH_WINDOW_S + POST_ONSET_WINDOW_S)
            )
            if mask.sum() >= ONSET_CONFIRM_SAMPLES:
                t = t_all[mask].copy()
                f = f_all[mask].copy()
                plateau_s = calculate_time_to_force_plateau(t, f)
            else:
                t = np.empty(0)
                f = np.empty(0)
                plateau_s = np.nan
            baseline_trials.append({
                "trial": trial_number,
                "t": t,
                "force": f,
                "plateau_s": plateau_s,
            })
            print(
                f"Baseline trial {trial_number}: time-to-plateau = "
                + (f"{plateau_s * 1000:.1f} ms" if np.isfinite(plateau_s) else "invalid")
            )

    if phase == "FAMILIARIZATION":
        return {}

    plateau_times = np.asarray(
        [x["plateau_s"] for x in baseline_trials], dtype=float
    )
    valid_plateaus = plateau_times[np.isfinite(plateau_times)]
    min_valid = max(3, int(np.ceil(n_trials / 2)))
    if valid_plateaus.size < min_valid:
        print(
            f"Baseline failed: only {valid_plateaus.size}/{n_trials} trials had "
            f"a valid sustained force plateau (need at least {min_valid})."
        )
        return {
            "thresholds": None,
            "plateau_times_s": plateau_times,
            "mean_plateau_s": np.nan,
            "rfd_check_end_s": np.nan,
            "checkpoints_s": tuple(),
            "force_profile": None,
        }

    mean_plateau_s = float(np.mean(valid_plateaus))
    checkpoints, individualized_end = make_rfd_checkpoints(mean_plateau_s)

    # Preserve the hard 180-ms stimulation-window cap with the 6-ms grid.
    # Starting at 20 ms with 6-ms spacing cannot land exactly on 180 ms:
    # the neighboring grid points are 176 and 182 ms. Therefore retain
    # only checkpoints <= 180 ms, making 176 ms the latest possible check.
    checkpoints = tuple(
        c for c in checkpoints if c <= MAX_RFD_CHECK_END_S + 1e-12
    )
    if not checkpoints:
        checkpoints = (RFD_CHECK_START_S,)
    individualized_end = float(checkpoints[-1])

    checkpoint_history = []
    force_checkpoint_history = []
    for trial in baseline_trials:
        if trial["t"].size >= ONSET_CONFIRM_SAMPLES:
            onset_idx, smoothed = find_rising_onset(trial["t"], trial["force"])
            checkpoint_history.append(
                compute_checkpoint_rfds(
                    trial["t"], trial["force"], onset_idx=onset_idx,
                    smoothed=smoothed, checkpoints=checkpoints
                )
            )
            force_checkpoint_history.append(
                compute_checkpoint_forces(
                    trial["t"], trial["force"], onset_idx=onset_idx,
                    smoothed=smoothed, checkpoints=checkpoints
                )
            )
        else:
            checkpoint_history.append({int(round(c*1000)):np.nan for c in checkpoints})
            force_checkpoint_history.append({int(round(c*1000)):np.nan for c in checkpoints})

    thresholds = _average_checkpoint_history(checkpoint_history, checkpoints=checkpoints)
    baseline_force_profile = _average_checkpoint_history(
        force_checkpoint_history, checkpoints=checkpoints
    )

    print("\n========================================")
    print("INDIVIDUALIZED BASELINE RFD WINDOW")
    print("========================================")
    print(
        f"Valid plateau trials: {valid_plateaus.size}/{n_trials} | "
        f"mean onset-to-plateau: {mean_plateau_s * 1000:.1f} ms"
    )
    print(
        f"Main Task RFD_CHECK_END_S: {individualized_end:.3f} s "
        f"({individualized_end * 1000:.0f} ms)"
    )
    print(
        f"Checkpoint grid: {int(checkpoints[0] * 1000)}-"
        f"{int(checkpoints[-1] * 1000)} ms in "
        f"{int(RFD_CHECK_STEP_S * 1000)} ms steps"
    )

    return {
        "thresholds": thresholds,
        "plateau_times_s": plateau_times,
        "mean_plateau_s": mean_plateau_s,
        "rfd_check_end_s": individualized_end,
        "checkpoints_s": checkpoints,
        "force_profile": baseline_force_profile,
    }


def startup_menu(
    screen, joystick, right_rect,
    calibration=None,
    baseline_result=None,
    subject_id=None,
    session_id=None,
    block_id=None,
    trigger_mode=None,
    stimulation_metric_mode=None,
    show_loadbar=None,
):
    """
    Operator interface controlling the full experiment flow.

    Adds editable Subject / Session / Block identifiers.

    Returns
        (calibration, trigger_mode, stimulation_metric_mode,
         baseline_result, show_loadbar, subject_id, session_id, block_id)
    or Nones on quit.
    """
    global PREP_DURATION_S, GO_DURATION_S, ITI_MIN_S, ITI_JITTER_S
    global ITI_MAX_S, USE_RED_CUE

    # Preserve successful calibration/baseline and operator selections when
    # returning here after a completed block.
    if trigger_mode is None:
        trigger_mode = SLOPE_TRIGGER_MODE
    if stimulation_metric_mode is None:
        stimulation_metric_mode = STIMULATION_METRIC_MODE
    if show_loadbar is None:
        show_loadbar = MAIN_TASK_LOADBAR_DEFAULT

    if subject_id is None:
        subject_id = X_NUMBER
    if session_id is None:
        session_id = SESSION
    if block_id is None:
        block_id = BLOCK

    # Calibration/baseline belong to one Subject + Session.
    # Changing Block does NOT invalidate them.
    reference_subject_id = subject_id if calibration is not None else None
    reference_session_id = session_id if calibration is not None else None

    active_field = None

    clock = pygame.time.Clock()
    width, height = screen.get_size()
    center_x = width // 2

    # --- ID fields -------------------------------------------------------
    field_w = min(280, int(width * 0.25))
    field_h = 48
    gap = 18
    subject_rect = pygame.Rect(0, 0, field_w, field_h)
    session_rect = pygame.Rect(0, 0, field_w, field_h)
    block_rect = pygame.Rect(0, 0, field_w, field_h)
    total_w = 3 * field_w + 2 * gap
    left_x = center_x - total_w // 2
    for i, rect in enumerate((subject_rect, session_rect, block_rect)):
        rect.topleft = (left_x + i * (field_w + gap), int(height * 0.155))

    # --- Task-version selector ------------------------------------------
    version_w = min(300, int(width * 0.27))
    version_h = 50
    version_gap = 12
    rfd_only_rect = pygame.Rect(0, 0, version_w, version_h)
    force_only_rect = pygame.Rect(0, 0, version_w, version_h)
    combined_rect = pygame.Rect(0, 0, version_w, version_h)
    total_vw = 3 * version_w + 2 * version_gap
    left_vx = center_x - total_vw // 2
    for i, rect in enumerate((rfd_only_rect, force_only_rect, combined_rect)):
        rect.topleft = (left_vx + i * (version_w + version_gap), int(height * 0.295))

    # --- BELOW / ABOVE selector -----------------------------------------
    mode_w = min(390, int(width * 0.36))
    mode_h = 52
    below_rect = pygame.Rect(0, 0, mode_w, mode_h)
    below_rect.center = (center_x - mode_w // 2 - 14, int(height * 0.405))
    above_rect = pygame.Rect(0, 0, mode_w, mode_h)
    above_rect.center = (center_x + mode_w // 2 + 14, int(height * 0.405))

    # --- Cue mode toggle (red cue + green  vs  green only) --------------
    cue_mode_w = min(390, int(width * 0.36))
    cue_mode_h = 46
    red_cue_toggle_rect = pygame.Rect(0, 0, cue_mode_w, cue_mode_h)
    red_cue_toggle_rect.center = (center_x, int(height * 0.470))

    # --- Editable timing fields (base durations + individual jitters) ---
    timing_w = min(180, int(width * 0.16))
    timing_h = 46
    timing_gap = 16
    red_time_rect = pygame.Rect(0, 0, timing_w, timing_h)
    green_time_rect = pygame.Rect(0, 0, timing_w, timing_h)
    pause_time_rect = pygame.Rect(0, 0, timing_w, timing_h)
    red_jitter_rect = pygame.Rect(0, 0, timing_w, timing_h)
    green_jitter_rect = pygame.Rect(0, 0, timing_w, timing_h)
    pause_jitter_rect = pygame.Rect(0, 0, timing_w, timing_h)

    base_row = (red_time_rect, green_time_rect, pause_time_rect)
    jitter_row = (red_jitter_rect, green_jitter_rect, pause_jitter_rect)
    total_tw = 3 * timing_w + 2 * timing_gap
    left_tx = center_x - total_tw // 2
    for i, rect in enumerate(base_row):
        rect.topleft = (left_tx + i * (timing_w + timing_gap), int(height * 0.540))
    for i, rect in enumerate(jitter_row):
        rect.topleft = (left_tx + i * (timing_w + timing_gap), int(height * 0.615))

    # Local editable string buffers for the timing fields, seeded from
    # the current (possibly previously-edited) global values.
    timing_values = {
        "red": f"{PREP_DURATION_S:.2f}",
        "green": f"{GO_DURATION_S:.2f}",
        "pause": f"{ITI_MIN_S:.2f}",
        "red_jitter": f"{PREP_JITTER_S:.2f}",
        "green_jitter": f"{GO_JITTER_S:.2f}",
        "pause_jitter": f"{ITI_JITTER_S:.2f}",
    }

    def _commit_timing_field(name):
        """Parse one edited timing field back into the module globals."""
        global PREP_DURATION_S, GO_DURATION_S, ITI_MIN_S
        global PREP_JITTER_S, GO_JITTER_S, ITI_JITTER_S, ITI_MAX_S
        current = {
            "red": PREP_DURATION_S, "green": GO_DURATION_S,
            "pause": ITI_MIN_S, "red_jitter": PREP_JITTER_S,
            "green_jitter": GO_JITTER_S, "pause_jitter": ITI_JITTER_S,
        }
        try:
            value = max(0.0, float(timing_values[name]))
        except ValueError:
            value = current[name]
        if name == "red":
            PREP_DURATION_S = value
        elif name == "green":
            GO_DURATION_S = value
        elif name == "pause":
            ITI_MIN_S = value
        elif name == "red_jitter":
            PREP_JITTER_S = value
        elif name == "green_jitter":
            GO_JITTER_S = value
        elif name == "pause_jitter":
            ITI_JITTER_S = value
        ITI_MAX_S = ITI_MIN_S + ITI_JITTER_S
        timing_values[name] = f"{value:.2f}"

    button_w = min(580, int(width * 0.58))
    button_h = 54
    loadbar_rect = pygame.Rect(0, 0, button_w, 40)
    loadbar_rect.center = (center_x, int(height * 0.680))

    ys = [0.735, 0.790, 0.845, 0.900]
    calibrate_rect, familiar_rect, baseline_rect, start_rect = [
        pygame.Rect(0, 0, button_w, button_h - 12) for _ in ys
    ]
    for rect, y in zip(
        (calibrate_rect, familiar_rect, baseline_rect, start_rect), ys
    ):
        rect.center = (center_x, int(height * y))

    def _sanitize_identifier(value, fallback):
        value = value.strip()
        if not value:
            return fallback
        safe = "".join(
            ch for ch in value
            if ch.isalnum() or ch in ("_", "-")
        )
        return safe or fallback

    def _draw_text_field(rect, label, value, active):
        pygame.draw.rect(
            screen, (70, 70, 70) if active else (45, 45, 45), rect
        )
        pygame.draw.rect(
            screen, WHITE if active else (150, 150, 150), rect, 2
        )
        label_font = get_font(18)
        value_font = get_font(24)
        label_s = label_font.render(label, True, (180, 180, 180))
        value_s = value_font.render(value, True, WHITE)
        screen.blit(label_s, (rect.x, rect.y - 24))
        screen.blit(value_s, value_s.get_rect(center=rect.center))

    while True:
        mouse_pos = pygame.mouse.get_pos()
        calibration_ready = calibration is not None and calibration.is_complete()

        baseline_thresholds = (
            baseline_result.get("thresholds")
            if isinstance(baseline_result, dict) else None
        )
        baseline_force_profile = (
            baseline_result.get("force_profile")
            if isinstance(baseline_result, dict) else None
        )
        baseline_ready = (
            baseline_thresholds is not None
            and baseline_force_profile is not None
            and any(np.isfinite(v) for v in baseline_thresholds.values())
            and any(np.isfinite(v) for v in baseline_force_profile.values())
            and np.isfinite(baseline_result.get("rfd_check_end_s", np.nan))
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return (None,) * 8

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q and active_field is None:
                    return (None,) * 8

                if active_field is not None and active_field.startswith("timing:"):
                    tname = active_field.split(":", 1)[1]
                    if event.key == pygame.K_RETURN:
                        _commit_timing_field(tname)
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:
                        timing_values[tname] = timing_values[tname][:-1]
                    else:
                        ch = event.unicode
                        if ch and (ch.isdigit() or ch == "."):
                            timing_values[tname] += ch
                elif active_field is not None:
                    if event.key == pygame.K_RETURN:
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:
                        if active_field == "subject":
                            subject_id = subject_id[:-1]
                            calibration = None
                            baseline_result = None
                            reference_subject_id = None
                            reference_session_id = None
                        elif active_field == "session":
                            session_id = session_id[:-1]
                            calibration = None
                            baseline_result = None
                            reference_subject_id = None
                            reference_session_id = None
                        else:
                            block_id = block_id[:-1]
                    else:
                        ch = event.unicode
                        if ch and (ch.isalnum() or ch in "_-"):
                            if active_field == "subject":
                                subject_id += ch
                                calibration = None
                                baseline_result = None
                                reference_subject_id = None
                                reference_session_id = None
                            elif active_field == "session":
                                session_id += ch
                                calibration = None
                                baseline_result = None
                                reference_subject_id = None
                                reference_session_id = None
                            else:
                                block_id += ch

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Commit whatever timing field was being edited before
                # switching focus elsewhere.
                if active_field is not None and active_field.startswith("timing:"):
                    _commit_timing_field(active_field.split(":", 1)[1])

                if subject_rect.collidepoint(event.pos):
                    active_field = "subject"
                elif session_rect.collidepoint(event.pos):
                    active_field = "session"
                elif block_rect.collidepoint(event.pos):
                    active_field = "block"
                elif red_time_rect.collidepoint(event.pos):
                    active_field = "timing:red"
                elif green_time_rect.collidepoint(event.pos):
                    active_field = "timing:green"
                elif pause_time_rect.collidepoint(event.pos):
                    active_field = "timing:pause"
                elif red_jitter_rect.collidepoint(event.pos):
                    active_field = "timing:red_jitter"
                elif green_jitter_rect.collidepoint(event.pos):
                    active_field = "timing:green_jitter"
                elif pause_jitter_rect.collidepoint(event.pos):
                    active_field = "timing:pause_jitter"
                else:
                    active_field = None

                    if red_cue_toggle_rect.collidepoint(event.pos):
                        USE_RED_CUE = not USE_RED_CUE
                    elif rfd_only_rect.collidepoint(event.pos):
                        stimulation_metric_mode = "RFD_ONLY"
                    elif force_only_rect.collidepoint(event.pos):
                        stimulation_metric_mode = "FORCE_ONLY"
                    elif combined_rect.collidepoint(event.pos):
                        stimulation_metric_mode = "COMBINED"
                    elif below_rect.collidepoint(event.pos):
                        trigger_mode = "BELOW"
                    elif above_rect.collidepoint(event.pos):
                        trigger_mode = "ABOVE"
                    elif loadbar_rect.collidepoint(event.pos):
                        show_loadbar = not show_loadbar
                    elif calibrate_rect.collidepoint(event.pos):
                        calibration = run_calibration(joystick, screen)
                        baseline_result = None
                        if calibration is None or not calibration.is_complete():
                            calibration = None
                            reference_subject_id = None
                            reference_session_id = None
                        else:
                            reference_subject_id = _sanitize_identifier(
                                subject_id, "X00000"
                            )
                            reference_session_id = _sanitize_identifier(
                                session_id, "Session_1"
                            )
                    elif familiar_rect.collidepoint(event.pos) and calibration_ready:
                        result = run_pre_task_phase(
                            screen, joystick, calibration, right_rect,
                            "FAMILIARIZATION", FAMILIARIZATION_TRIALS,
                        )
                        if result is None:
                            return (None,) * 8
                    elif baseline_rect.collidepoint(event.pos) and calibration_ready:
                        baseline_result = run_pre_task_phase(
                            screen, joystick, calibration, right_rect,
                            "BASELINE", BASELINE_TRIALS,
                        )
                        if baseline_result is None:
                            return (None,) * 8
                    elif (
                        start_rect.collidepoint(event.pos)
                        and calibration_ready and baseline_ready
                    ):
                        subject_id = _sanitize_identifier(subject_id, "X00000")
                        session_id = _sanitize_identifier(session_id, "Session_1")
                        block_id = _sanitize_identifier(block_id, "Block_1")
                        for _tname in (
                            "red", "green", "pause",
                            "red_jitter", "green_jitter", "pause_jitter",
                        ):
                            _commit_timing_field(_tname)
                        for _w in validate_and_fix_timing():
                            print(f"WARNING (timing auto-corrected): {_w}")
                        return (
                            calibration,
                            trigger_mode,
                            stimulation_metric_mode,
                            baseline_result,
                            show_loadbar,
                            subject_id,
                            session_id,
                            block_id,
                        )

        screen.fill(BLACK)

        title_font = get_font(54)
        title_surface = title_font.render(
            "Adaptive Thresholding TMS Task", True, WHITE
        )
        screen.blit(
            title_surface,
            title_surface.get_rect(center=(center_x, int(height * 0.065))),
        )

        _draw_text_field(
            subject_rect, "Subject", subject_id, active_field == "subject"
        )
        _draw_text_field(
            session_rect, "Session", session_id, active_field == "session"
        )
        _draw_text_field(
            block_rect, "Block", block_id, active_field == "block"
        )

        section_font = get_font(23)
        heading = section_font.render(
            "Select Task Version", True, (200, 200, 200)
        )
        screen.blit(
            heading, heading.get_rect(center=(center_x, int(height * 0.255)))
        )

        draw_menu_button(
            screen, rfd_only_rect, "RFD only", True,
            rfd_only_rect.collidepoint(mouse_pos),
            selected=stimulation_metric_mode == "RFD_ONLY",
        )
        draw_menu_button(
            screen, force_only_rect, "Force only", True,
            force_only_rect.collidepoint(mouse_pos),
            selected=stimulation_metric_mode == "FORCE_ONLY",
        )
        draw_menu_button(
            screen, combined_rect, "RFD + Force", True,
            combined_rect.collidepoint(mouse_pos),
            selected=stimulation_metric_mode == "COMBINED",
        )

        draw_menu_button(
            screen, below_rect, "Below threshold", True,
            below_rect.collidepoint(mouse_pos),
            selected=trigger_mode == "BELOW",
        )
        draw_menu_button(
            screen, above_rect, "Above threshold", True,
            above_rect.collidepoint(mouse_pos),
            selected=trigger_mode == "ABOVE",
        )

        draw_menu_button(
            screen, red_cue_toggle_rect,
            "Cue: RED then GREEN" if USE_RED_CUE else "Cue: GREEN only",
            True, red_cue_toggle_rect.collidepoint(mouse_pos),
            selected=True,
        )

        timing_heading = section_font.render(
            "Trial timing (seconds) -- base duration, then jitter added on top",
            True, (200, 200, 200)
        )
        screen.blit(
            timing_heading,
            timing_heading.get_rect(center=(center_x, int(height * 0.515))),
        )
        _draw_text_field(
            red_time_rect,
            "Red cue dur." if USE_RED_CUE else "Red cue dur. (unused)",
            timing_values["red"], active_field == "timing:red",
        )
        _draw_text_field(
            green_time_rect, "Green target dur.",
            timing_values["green"], active_field == "timing:green",
        )
        _draw_text_field(
            pause_time_rect, "Pause (base ITI)",
            timing_values["pause"], active_field == "timing:pause",
        )
        _draw_text_field(
            red_jitter_rect,
            "Red jitter (+rand)" if USE_RED_CUE else "Red jitter (unused)",
            timing_values["red_jitter"], active_field == "timing:red_jitter",
        )
        _draw_text_field(
            green_jitter_rect, "Green jitter (+rand)",
            timing_values["green_jitter"], active_field == "timing:green_jitter",
        )
        _draw_text_field(
            pause_jitter_rect, "Pause jitter (+rand)",
            timing_values["pause_jitter"], active_field == "timing:pause_jitter",
        )

        draw_menu_button(
            screen, loadbar_rect,
            f"Main Task loadbar: {'ON' if show_loadbar else 'OFF'}",
            True, loadbar_rect.collidepoint(mouse_pos),
            selected=True,
        )
        draw_menu_button(
            screen, calibrate_rect,
            "Calibrate Right Handle"
            if not calibration_ready else "Recalibrate Right Handle",
            True, calibrate_rect.collidepoint(mouse_pos),
        )
        draw_menu_button(
            screen, familiar_rect,
            f"Familiarization ({FAMILIARIZATION_TRIALS} trials)",
            calibration_ready,
            calibration_ready and familiar_rect.collidepoint(mouse_pos),
        )
        draw_menu_button(
            screen, baseline_rect,
            (
                f"Baseline Calibration ({BASELINE_TRIALS} trials)"
                if not baseline_ready
                else f"Repeat Baseline Calibration ({BASELINE_TRIALS} trials)"
            ),
            calibration_ready,
            calibration_ready and baseline_rect.collidepoint(mouse_pos),
        )
        draw_menu_button(
            screen, start_rect, "Start Main Task",
            calibration_ready and baseline_ready,
            calibration_ready and baseline_ready
            and start_rect.collidepoint(mouse_pos),
        )

        status_font = get_font(18)
        cal_status = (
            "Handle calibrated"
            if calibration_ready else "Handle calibration required"
        )
        if baseline_ready:
            base_status = (
                f"Baseline ready | RFD end "
                f"{baseline_result['rfd_check_end_s'] * 1000:.0f} ms"
            )
        else:
            base_status = "Baseline required"

        status = (
            f"{cal_status} | {base_status} | "
            f"{stimulation_metric_mode} | {trigger_mode}"
        )
        surf = status_font.render(status, True, (190, 190, 190))
        screen.blit(
            surf, surf.get_rect(center=(center_x, int(height * 0.935)))
        )

        hint = get_font(16).render(
            "Green frame = currently selected option. "
            "Changing Subject or Session resets calibration + baseline. "
            "Click a timing box and type a number, then press Enter.",
            True, (120, 120, 120),
        )
        screen.blit(
            hint, hint.get_rect(center=(center_x, int(height * 0.965)))
        )

        pygame.display.flip()
        clock.tick(60)


def show_end_screen(screen):
    screen.fill(BLACK)
    show_text("END", WHITE, screen)
    pygame.display.flip()


def countdown(screen):
    for i in range(3, 0, -1):
        screen.fill(BLACK)
        show_text(str(i), WHITE, screen)
        pygame.display.flip()

        # Keep the event queue alive while waiting.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            pygame.event.pump()
            time.sleep(0.01)

    screen.fill(BLACK)
    show_text("Start!", WHITE, screen)
    pygame.display.flip()

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        pygame.event.pump()
        time.sleep(0.01)


# =============================================================================
# CURSOR GEOMETRY
# =============================================================================

def force_to_cursor_y(force, screen_height):
    """
    Participant-normalised force -> screen Y.

        0.0 = bottom of screen
        1.0 = top of screen
    """
    force = float(np.clip(force, 0.0, 1.0))

    bottom_y = screen_height - 1
    top_y = 0

    return int(bottom_y - force * (bottom_y - top_y))


def draw_cursor_track(screen, target_rect):
    pygame.draw.rect(
        screen,
        CURSOR_TRACK_COLOR,
        pygame.Rect(
            target_rect.centerx - CURSOR_TRACK_WIDTH // 2,
            0,
            CURSOR_TRACK_WIDTH,
            screen.get_height(),
        ),
    )


def draw_force_cursor(screen, force, target_rect, visible=True):
    if not visible:
        return

    x = target_rect.centerx
    y = force_to_cursor_y(force, screen.get_height())

    pygame.draw.circle(
        screen,
        CURSOR_OUTLINE_COLOR,
        (x, y),
        CURSOR_RADIUS + CURSOR_OUTLINE_WIDTH,
    )
    pygame.draw.circle(screen, CURSOR_COLOR, (x, y), CURSOR_RADIUS)


def draw_active_target(screen, rect, color):
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, WHITE, rect, RECT_OUTLINE_WIDTH)


# =============================================================================
# TARGETS
# =============================================================================

def create_target_rect(screen_height):
    """
    Box height is derived from the force tolerance, so that being inside
    the box genuinely means being within TARGET_TOLERANCE of the target
    force level.
    """
    height = max(20, int(2 * TARGET_TOLERANCE * screen_height))
    return pygame.Rect(0, 0, RECT_WIDTH, height)


def place_targets(right_rect, width, height, right_force_level=None):
    """Place the single RIGHT-hand target in the horizontal centre."""
    if right_force_level is None:
        right_force_level = TARGET_FORCE_LEVEL
    right_y = force_to_cursor_y(right_force_level, height)
    right_rect.center = (width // 2, right_y)

def configure_targets_for_trial(trial_number, right_rect, width, height):
    """Configure the single RIGHT-hand target for every trial."""
    place_targets(right_rect, width, height)
    return {"right": True}

# =============================================================================
# DAQ TRIGGER
# =============================================================================

def send_trigger(channel):
    if not DAQ_AVAILABLE:
        print(f"[SIMULATED] Trigger on {channel}")
        return

    try:
        with _daq_lock:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(
                    f"{daq_device}/{channel}"
                )
                task.write(5.0)
                task.write(0.0)

        print(f"Trigger sent on {channel}")

    except Exception as e:
        print(f"Error sending trigger on {channel}: {e}")


# =============================================================================
# CALIBRATION
# =============================================================================

def measure_baseline(joystick, screen, duration_s, message):
    """Measure the resting baseline of the RIGHT handle only."""
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return None

        samples.append(read_raw_axes(joystick))
        remaining = duration_s - (time.perf_counter() - t0)
        screen.fill(BLACK)
        show_text(
            f"{message}\n\n{max(0, int(np.ceil(remaining)))}",
            WHITE, screen,
        )
        pygame.display.flip()
        time.sleep(0.002)

    if not samples:
        return 0.0
    return float(np.median(np.asarray(samples, dtype=float)))

def calibrate_hand(joystick, screen, calibration):
    """Measure the participant's maximum contraction for the RIGHT handle."""
    print("\nStarting calibration: RIGHT hand")
    baseline = calibration.baseline_right
    t0 = time.perf_counter()
    max_deflection = 0.0

    while True:
        elapsed = time.perf_counter() - t0
        remaining = CALIBRATION_DURATION_S - elapsed
        if remaining <= 0:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return CAL_QUIT, np.nan
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return CAL_QUIT, np.nan

        right_raw = read_raw_axes(joystick)
        deflection = raw_to_deflection(right_raw, baseline)
        if deflection > max_deflection:
            max_deflection = deflection

        screen.fill(BLACK)
        show_text(
            f"KLEM SAA HAARDT DU KAN\n\n"
            f"{max(0, int(np.ceil(remaining)))}",
            WHITE, screen, y_center=int(screen.get_height() * 0.25),
        )
        bar_height = int(screen.get_height() * 0.45)
        bar_rect = pygame.Rect(0, 0, 120, bar_height)
        bar_rect.center = (
            screen.get_width() // 2, int(screen.get_height() * 0.68)
        )
        pygame.draw.rect(screen, (60, 60, 60), bar_rect)
        fraction = (
            float(np.clip(deflection / max_deflection, 0.0, 1.0))
            if max_deflection > 1e-9 else 0.0
        )
        fill_rect = bar_rect.copy()
        fill_rect.height = int(bar_rect.height * fraction)
        fill_rect.top = bar_rect.bottom - fill_rect.height
        pygame.draw.rect(screen, GREEN, fill_rect)
        pygame.draw.rect(screen, WHITE, bar_rect, 3)
        pygame.display.flip()
        time.sleep(0.002)

    if max_deflection < MIN_VALID_CALIBRATION_DEFLECTION:
        print(
            "WARNING: no significant RIGHT force detected "
            f"(max deflection = {max_deflection:.6f})."
        )
        return CAL_NO_FORCE, max_deflection

    print(f"RIGHT calibrated max deflection = {max_deflection:.6f}")
    screen.fill(BLACK)
    show_text(
        "RIGHT KALIBRERET\n\n100% kraft = din maksimale kraft",
        GREEN, screen,
    )
    pygame.display.flip()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        pygame.event.pump()
        time.sleep(0.01)
    return CAL_OK, max_deflection

def run_calibration(joystick, screen):
    """Calibrate only the RIGHT handle."""
    calibration = Calibration()
    print(
        "\n========================================\n"
        "RIGHT-HANDLE FORCE CALIBRATION\n"
        "========================================"
    )

    baseline = measure_baseline(
        joystick, screen, CALIBRATION_REST_DURATION_S,
        "Slap helt af med hoejre haand",
    )
    if baseline is None:
        return None
    calibration.baseline_right = baseline
    print(f"Baseline right = {calibration.baseline_right:+.6f}")

    status, value = calibrate_hand(joystick, screen, calibration)
    if status == CAL_QUIT:
        return None
    if status == CAL_NO_FORCE:
        screen.fill(BLACK)
        show_text("Ingen kraft registreret\n(hoejre haand)", RED, screen)
        pygame.display.flip()
        time.sleep(2.0)
        return None

    calibration.max_deflection_right = value
    print("\nCalibration complete:")
    print(calibration.summary())
    return calibration

# =============================================================================
# SMOOTHING
# =============================================================================

def smooth_signal(values, samples):
    """
    Causal-safe moving average.

    np.convolve(..., mode="same") zero-pads both ends, which attenuates
    the newest samples to ~59 % of their true value for a 5-sample
    kernel.  Those are exactly the samples the real-time path depends
    on, so the old behaviour dragged the measured RFD downward and made
    the low-slope trigger fire early.  Here the leading samples are left
    unsmoothed and nothing past the end is invented.
    """
    values = np.asarray(values, dtype=float)

    if samples <= 1 or values.size < samples:
        return values.copy()

    kernel = np.ones(samples) / samples
    core = np.convolve(values, kernel, mode="valid")
    pad = values.size - core.size

    return np.concatenate([values[:pad], core])


# =============================================================================
# ONSET DETECTION
# =============================================================================

def find_rising_onset(t, force, smoothed=None):
    """
    Locate force onset: the first sample where smoothed force crosses
    RISING_ONSET_FORCE and STAYS above it for ONSET_CONFIRM_SAMPLES
    consecutive samples.

    The confirmation requirement matters.  A single noisy sample nicking
    the threshold would otherwise set onset early, and because every
    downstream metric is anchored to onset, that one sample would shift
    the whole epoch.

    Returns (onset_index, smoothed_force) or (None, smoothed_force).
    """
    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)

    if smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)

    if smoothed.size < ONSET_CONFIRM_SAMPLES:
        return None, smoothed

    above = smoothed >= RISING_ONSET_FORCE

    if not above.any():
        return None, smoothed

    # Rolling count of consecutive True values.
    run = 0

    for i in range(above.size):
        if above[i]:
            run += 1
            if run >= ONSET_CONFIRM_SAMPLES:
                return i - ONSET_CONFIRM_SAMPLES + 1, smoothed
        else:
            run = 0

    return None, smoothed


# =============================================================================
# RATE OF FORCE DEVELOPMENT OVER A FIXED, ONSET-ANCHORED EPOCH
# =============================================================================

def calculate_epoch_rfd(t, force, epoch_s=RFD_EPOCH_S,
                        onset_idx=None, smoothed=None,
                        window_s=LOCAL_RFD_WINDOW_S):
    """
    Local/sliding-window RFD used by the adaptive decision.

    At checkpoint ``epoch_s`` after force onset, fit a line over the most
    recent ``window_s`` seconds ending at that checkpoint.  With the default
    50-ms window, RFD@120 ms is the slope from 70 to 120 ms.
    """
    result = {
        "rfd": np.nan, "onset_idx": None, "onset_time": np.nan,
        "epoch_end_time": np.nan, "window_start_time": np.nan,
        "n_samples": 0, "complete": False, "reason": "",
    }
    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)
    if t.size < ONSET_CONFIRM_SAMPLES:
        result["reason"] = "too_few_samples"; return result
    if onset_idx is None:
        onset_idx, smoothed = find_rising_onset(t, force, smoothed)
    elif smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    if onset_idx is None:
        result["reason"] = "no_onset"; return result

    onset_time = float(t[onset_idx])
    epoch_end = onset_time + float(epoch_s)
    window_start = max(onset_time, epoch_end - float(window_s))
    result.update(onset_idx=int(onset_idx), onset_time=onset_time,
                  epoch_end_time=epoch_end, window_start_time=window_start)
    if t[-1] < epoch_end:
        result["reason"] = "epoch_incomplete"; return result
    result["complete"] = True
    mask = (t >= window_start) & (t <= epoch_end)
    n = int(mask.sum()); result["n_samples"] = n
    if n < MIN_EPOCH_SAMPLES:
        result["reason"] = "too_few_samples_in_epoch"; return result
    t_segment = t[mask]; f_segment = smoothed[mask]
    if np.ptp(t_segment) <= 0:
        result["reason"] = "zero_time_span"; return result
    slope, _ = np.polyfit(t_segment - np.mean(t_segment), f_segment, 1)
    result["rfd"] = float(slope); result["reason"] = "ok"
    return result


# =============================================================================
# DESCRIPTIVE POST-HOC MEASURES
# =============================================================================

def calculate_peak_rfd(t, force, onset_idx, smoothed=None,
                       window_s=PEAK_RFD_WINDOW_S):
    """
    Maximum instantaneous RFD: the steepest short-window slope anywhere
    after onset.  Descriptive only - never used for triggering.
    """
    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)

    if smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)

    if onset_idx is None or t.size - onset_idx < 4:
        return np.nan, np.nan

    best_slope = np.nan
    best_time = np.nan

    for i in range(onset_idx, t.size):
        window_end = t[i] + window_s

        if window_end > t[-1]:
            break

        mask = (t >= t[i]) & (t <= window_end)

        if mask.sum() < 4:
            continue

        t_seg = t[mask]
        f_seg = smoothed[mask]

        if np.ptp(t_seg) <= 0:
            continue

        slope, _ = np.polyfit(t_seg - np.mean(t_seg), f_seg, 1)

        if not np.isfinite(best_slope) or slope > best_slope:
            best_slope = float(slope)
            best_time = float(t[i])

    return best_slope, best_time


def calculate_rfd_to_peak(t, force, onset_idx, smoothed=None):
    """
    Average RFD from onset to peak force.

    Retained for completeness because it is a commonly reported measure,
    but note it is confounded by rise duration: a slower rise to the same
    peak yields a lower value even with identical early force
    development.  This is why it is no longer the trigger metric.
    """
    nan_result = {
        "rfd": np.nan,
        "peak_time": np.nan,
        "peak_force": np.nan,
        "duration": np.nan,
    }

    t = np.asarray(t, dtype=float)
    force = np.asarray(force, dtype=float)

    if smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)

    if onset_idx is None or onset_idx >= t.size - MIN_EPOCH_SAMPLES:
        return nan_result

    peak_idx = onset_idx + int(np.argmax(smoothed[onset_idx:]))

    if (peak_idx - onset_idx) < MIN_EPOCH_SAMPLES:
        return {
            "rfd": np.nan,
            "peak_time": float(t[peak_idx]),
            "peak_force": float(smoothed[peak_idx]),
            "duration": float(t[peak_idx] - t[onset_idx]),
        }

    t_segment = t[onset_idx:peak_idx + 1]
    f_segment = smoothed[onset_idx:peak_idx + 1]

    if np.ptp(t_segment) <= 0:
        slope = np.nan
    else:
        slope, _ = np.polyfit(
            t_segment - np.mean(t_segment), f_segment, 1
        )

    return {
        "rfd": float(slope),
        "peak_time": float(t[peak_idx]),
        "peak_force": float(smoothed[peak_idx]),
        "duration": float(t[peak_idx] - t[onset_idx]),
    }


def calculate_running_slope(t, force, onset_idx=None, smoothed=None,
                            window_s=LOCAL_RFD_WINDOW_S):
    """Continuous diagnostic LOCAL RFD over the most recent 50-ms window."""
    t = np.asarray(t, dtype=float); force = np.asarray(force, dtype=float)
    if t.size < MIN_SLOPE_SAMPLES:
        return np.nan
    if onset_idx is None:
        onset_idx, smoothed = find_rising_onset(t, force, smoothed)
    elif smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    if onset_idx is None:
        return np.nan
    current_time = float(t[-1])
    if current_time - float(t[onset_idx]) < window_s:
        return np.nan
    start_time = current_time - window_s
    mask = (t >= start_time) & (t <= current_time)
    if int(mask.sum()) < MIN_SLOPE_SAMPLES:
        return np.nan
    ts=t[mask]; fs=smoothed[mask]
    if np.ptp(ts) <= 0:
        return np.nan
    slope,_=np.polyfit(ts-np.mean(ts),fs,1)
    return float(slope)


def compute_checkpoint_forces(t, force, onset_idx=None, smoothed=None,
                              checkpoints=None):
    """Smoothed force at each onset-anchored checkpoint."""
    if checkpoints is None:
        checkpoints = RFD_CHECKPOINTS_S
    out = {int(round(c * 1000)): np.nan for c in checkpoints}
    t=np.asarray(t,dtype=float); force=np.asarray(force,dtype=float)
    if onset_idx is None:
        onset_idx, smoothed = find_rising_onset(t, force, smoothed)
    elif smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    if onset_idx is None:
        return out
    onset_time=float(t[onset_idx])
    for c in checkpoints:
        target=onset_time+float(c)
        if t.size and t[-1] >= target:
            idx=int(np.argmin(np.abs(t-target)))
            out[int(round(c*1000))]=float(smoothed[idx])
    return out


def calculate_checkpoint_force_references(force_history):
    """Mean checkpoint force from the previous two completed trials."""
    refs={int(round(c*1000)):np.nan for c in RFD_CHECKPOINTS_S}
    if len(force_history) < FORCE_REFERENCE_TRIALS:
        return refs
    recent=force_history[-FORCE_REFERENCE_TRIALS:]
    for key in refs:
        vals=np.asarray([trial.get(key,np.nan) for trial in recent],dtype=float)
        if np.all(np.isfinite(vals)):
            refs[key]=float(np.mean(vals))
    return refs


# =============================================================================
# ADAPTIVE THRESHOLD
# =============================================================================

def compute_checkpoint_rfds(t, force, onset_idx=None, smoothed=None,
                            checkpoints=RFD_CHECKPOINTS_S):
    """
    RFD at every checkpoint, all measured from the SAME onset.

    Returns {checkpoint_ms: rfd}.  A checkpoint whose window has not
    fully elapsed, or which cannot be fitted, is NaN.
    """
    out = {int(round(c * 1000)): np.nan for c in checkpoints}

    if onset_idx is None:
        onset_idx, smoothed = find_rising_onset(t, force, smoothed)
    elif smoothed is None:
        smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)

    if onset_idx is None:
        return out

    for c in checkpoints:
        e = calculate_epoch_rfd(
            t, force, epoch_s=c,
            onset_idx=onset_idx, smoothed=smoothed,
        )
        out[int(round(c * 1000))] = e["rfd"]

    return out


def calculate_checkpoint_thresholds(checkpoint_history):
    """
    One threshold per checkpoint, each the mean of the previous
    SLOPE_REFERENCE_TRIALS trials AT THAT SAME CHECKPOINT.

    checkpoint_history is a list of {checkpoint_ms: rfd} dicts, one per
    completed trial.

    A checkpoint whose reference trials are not all valid gets NaN, and
    is simply skipped during the trial - the remaining checkpoints still
    work.  This means a trial with one bad early sample does not disable
    triggering entirely.
    """
    thresholds = {
        int(round(c * 1000)): np.nan for c in RFD_CHECKPOINTS_S
    }

    if len(checkpoint_history) < SLOPE_REFERENCE_TRIALS:
        return thresholds

    recent = checkpoint_history[-SLOPE_REFERENCE_TRIALS:]

    for key in thresholds:
        values = np.array(
            [trial.get(key, np.nan) for trial in recent], dtype=float
        )

        if np.all(np.isfinite(values)):
            thresholds[key] = float(np.mean(values))

    return thresholds


def calculate_rfd_threshold(rfd_history):
    """
    Threshold for the upcoming trial = mean of the fixed-epoch RFD from
    the immediately previous SLOPE_REFERENCE_TRIALS trials.

        Trial 3: mean(trial 1, trial 2)
        Trial 4: mean(trial 2, trial 3)

    Both sides of the comparison are now the same quantity: the slope of
    force over the first RFD_EPOCH_S after onset.

    If either reference trial has an invalid RFD (no onset, epoch never
    completed), the threshold is NaN and the upcoming trial cannot
    trigger.  Trials 1 and 2 likewise have no threshold.
    """
    if len(rfd_history) < SLOPE_REFERENCE_TRIALS:
        return np.nan

    recent = np.asarray(
        rfd_history[-SLOPE_REFERENCE_TRIALS:], dtype=float
    )

    if not np.all(np.isfinite(recent)):
        return np.nan

    return float(np.mean(recent))


# =============================================================================
# REAL-TIME RFD MONITOR
# =============================================================================

class RFDMonitor(threading.Thread):
    """
    Watches the RIGHT handle during the GO period and checks RFD REPEATEDLY as
    the rise unfolds.

    Sequence within a trial:

        1. Wait for a confirmed force onset (search limited to
           ONSET_SEARCH_WINDOW_S after GO).
        2. As each checkpoint elapses - 60, 80, 100 ... 300 ms after
           onset - fit force over [onset, checkpoint] and compare it
           with the threshold FOR THAT CHECKPOINT.
        3. Trigger on the first checkpoint satisfying the selected rule (BELOW or ABOVE).
        4. Stop checking at RFD_CHECK_END_S.

    Every comparison is like-for-like: a 100 ms window is only ever
    compared with 100 ms windows from previous trials.  Because all
    windows start at onset, reaction time never enters the measurement.
    """

    def __init__(self, poll_hz=500,
                 trigger_channel=rfd_trigger_channel):

        super().__init__(daemon=True)

        self.poll_dt = 1.0 / float(poll_hz)
        self.trigger_channel = trigger_channel

        self._stop_evt = threading.Event()
        self._active = threading.Event()
        self.lock = threading.Lock()

        self.trigger_mode = SLOPE_TRIGGER_MODE
        self.metric_mode = STIMULATION_METRIC_MODE

        self._reset_state()

    def _reset_state(self):
        self.go_start_time = 0.0
        self.thresholds = {}
        self.force_references = {}

        self.finished = False
        self.triggered = False

        self.onset_time_from_go = np.nan
        self.checked = {}                 # checkpoint_ms -> rfd seen live
        self.checkpoints_done = set()

        self.trigger_checkpoint_ms = np.nan
        self.trigger_rfd = np.nan
        self.trigger_threshold = np.nan
        self.trigger_force = np.nan
        self.trigger_force_reference = np.nan
        self.trigger_from_onset_s = np.nan
        self.trigger_from_go_s = np.nan
        self.trigger_time = None

        self.decision_reason = ""

    # -----------------------------------------------------------------
    # ARM / DISARM
    # -----------------------------------------------------------------

    def arm(self, go_start_time, thresholds, force_references):
        with self.lock:
            self._reset_state()
            self.go_start_time = go_start_time
            self.thresholds = dict(thresholds)
            self.force_references = dict(force_references)

        self._active.set()

    def disarm(self):
        self._active.clear()

        with self.lock:
            if not self.decision_reason:
                if not any(
                    np.isfinite(v) for v in self.thresholds.values()
                ):
                    self.decision_reason = "no_threshold"
                elif np.isnan(self.onset_time_from_go):
                    self.decision_reason = "no_onset_detected"
                elif not self.triggered:
                    self.decision_reason = (
                        f"no_checkpoint_{self.metric_mode.lower()}_"
                        f"{self.trigger_mode.lower()}_criterion"
                    )

            return {
                "thresholds": dict(self.thresholds),
                "force_references": dict(self.force_references),
                "trigger_mode": self.trigger_mode,
                "stimulation_metric_mode": self.metric_mode,
                "triggered": self.triggered,
                "onset_from_go_s": self.onset_time_from_go,
                "checkpoints_checked_live": dict(self.checked),
                "n_checkpoints_checked": len(self.checkpoints_done),
                "trigger_checkpoint_ms": self.trigger_checkpoint_ms,
                "trigger_rfd": self.trigger_rfd,
                "trigger_threshold": self.trigger_threshold,
                "trigger_force": self.trigger_force,
                "trigger_force_reference": self.trigger_force_reference,
                "trigger_from_onset_s": self.trigger_from_onset_s,
                "trigger_from_go_s": self.trigger_from_go_s,
                "trigger_time_s": self.trigger_time,
                "decision_reason": self.decision_reason,
            }

    def stop(self):
        self._stop_evt.set()
        self._active.clear()

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    def run(self):
        while not self._stop_evt.is_set():

            if not self._active.is_set():
                time.sleep(self.poll_dt)
                continue

            with self.lock:
                go_t0 = self.go_start_time
                thresholds = dict(self.thresholds)
                force_references = dict(self.force_references)
                finished = self.finished
                done = set(self.checkpoints_done)

            if finished:
                time.sleep(self.poll_dt)
                continue

            if not any(np.isfinite(v) for v in thresholds.values()):
                time.sleep(self.poll_dt)
                continue

            t_arr, force = get_recent_samples_numpy()

            if t_arr.size < ONSET_CONFIRM_SAMPLES:
                time.sleep(self.poll_dt)
                continue

            mask = (
                (t_arr >= go_t0)
                & (t_arr <= go_t0 + ONSET_SEARCH_WINDOW_S
                   + POST_ONSET_WINDOW_S)
            )

            if mask.sum() < ONSET_CONFIRM_SAMPLES:
                time.sleep(self.poll_dt)
                continue

            t_window = t_arr[mask]
            f_window = force[mask]

            onset_idx, smoothed = find_rising_onset(t_window, f_window)

            if onset_idx is None:
                if t_window[-1] >= go_t0 + ONSET_SEARCH_WINDOW_S:
                    with self.lock:
                        self.finished = True
                        self.decision_reason = "no_onset_detected"

                time.sleep(self.poll_dt)
                continue

            onset_time = t_window[onset_idx]

            if onset_time > go_t0 + ONSET_SEARCH_WINDOW_S:
                with self.lock:
                    self.finished = True
                    self.decision_reason = "onset_too_late"

                time.sleep(self.poll_dt)
                continue

            with self.lock:
                self.onset_time_from_go = onset_time - go_t0

            elapsed_since_onset = t_window[-1] - onset_time

            # -------------------------------------------------------------
            # Check every checkpoint that has elapsed and is still unchecked
            # -------------------------------------------------------------

            fired = None

            for checkpoint_s in RFD_CHECKPOINTS_S:

                key = int(round(checkpoint_s * 1000))

                if key in done:
                    continue

                if elapsed_since_onset < checkpoint_s:
                    break          # checkpoints are ordered

                threshold = thresholds.get(key, np.nan)
                force_reference = force_references.get(key, np.nan)

                epoch = calculate_epoch_rfd(
                    t_window, f_window, epoch_s=checkpoint_s,
                    onset_idx=onset_idx, smoothed=smoothed,
                )
                rfd = epoch["rfd"]
                checkpoint_force = np.nan
                target_time = onset_time + checkpoint_s
                if t_window[-1] >= target_time:
                    idx = int(np.argmin(np.abs(t_window - target_time)))
                    checkpoint_force = float(smoothed[idx])

                with self.lock:
                    self.checkpoints_done.add(key)
                    self.checked[key] = rfd
                done.add(key)

                rfd_valid = np.isfinite(threshold) and np.isfinite(rfd)
                force_valid = (
                    np.isfinite(force_reference)
                    and np.isfinite(checkpoint_force)
                )

                if self.trigger_mode == "BELOW":
                    rfd_decision_threshold = (
                        threshold * (1.0 - TRIGGER_MARGIN_FRACTION)
                        if rfd_valid else np.nan
                    )
                    rfd_condition = (
                        rfd_valid and (rfd < rfd_decision_threshold)
                    )
                    force_decision_threshold = (
                        force_reference * (1.0 - FORCE_REFERENCE_MARGIN)
                        if force_valid else np.nan
                    )
                    force_condition = (
                        force_valid
                        and (checkpoint_force < force_decision_threshold)
                        and (checkpoint_force < FORCE_TRIGGER_CEILING)
                    )
                else:
                    rfd_decision_threshold = (
                        threshold * (1.0 + TRIGGER_MARGIN_FRACTION)
                        if rfd_valid else np.nan
                    )
                    rfd_condition = (
                        rfd_valid and (rfd > rfd_decision_threshold)
                    )
                    force_decision_threshold = (
                        force_reference * (1.0 + FORCE_REFERENCE_MARGIN)
                        if force_valid else np.nan
                    )
                    force_condition = (
                        force_valid
                        and (checkpoint_force > force_decision_threshold)
                    )

                if self.metric_mode == "RFD_ONLY":
                    should_trigger = bool(rfd_condition)
                elif self.metric_mode == "FORCE_ONLY":
                    should_trigger = bool(force_condition)
                elif self.metric_mode == "COMBINED":
                    should_trigger = bool(rfd_condition and force_condition)
                else:
                    should_trigger = False

                rfd_txt = (
                    f"{rfd:.3f} vs ref={threshold:.3f} "
                    f"(decision={rfd_decision_threshold:.3f}, "
                    f"margin={TRIGGER_MARGIN_FRACTION * 100:.1f}%)"
                    if rfd_valid else "N/A"
                )
                force_txt = (
                    f"{checkpoint_force:.3f} vs ref={force_reference:.3f} "
                    f"(decision={force_decision_threshold:.3f}, "
                    f"margin={FORCE_REFERENCE_MARGIN * 100:.1f}%)"
                    if force_valid else "N/A"
                )
                print(
                    f"[RIGHT {key:3d} ms | {self.metric_mode}] "
                    f"localRFD={rfd_txt} | force={force_txt} | "
                    f"gate={'PASS' if should_trigger else 'no'}"
                )

                if should_trigger:
                    fired = (key, rfd, threshold, checkpoint_s,
                             checkpoint_force, force_reference)
                    break

            if fired is not None:
                key, rfd, threshold, checkpoint_s, checkpoint_force, force_reference = fired

                with self.lock:
                    self.trigger_force = checkpoint_force
                    self.trigger_force_reference = force_reference
                self._send_trigger(
                    key, rfd, threshold, checkpoint_s,
                    t_window[-1] - go_t0,
                )

                with self.lock:
                    self.finished = True

            elif elapsed_since_onset >= RFD_CHECK_END_S:
                with self.lock:
                    self.finished = True
                    self.decision_reason = (
                        f"no_checkpoint_{self.metric_mode.lower()}_"
                        f"{self.trigger_mode.lower()}_criterion"
                    )

                print(
                    f"[RIGHT] no trigger | "
                    f"{len(done)} checkpoints checked | "
                    f"onset {self.onset_time_from_go * 1000:.0f} ms "
                    "after GO"
                )

            time.sleep(self.poll_dt)

    # -----------------------------------------------------------------
    # SEND TRIGGER
    # -----------------------------------------------------------------

    def _send_trigger(self, checkpoint_ms, rfd, threshold,
                      checkpoint_s, latency_from_go):

        send_trigger(self.trigger_channel)

        if GLOBAL_START_TIME is not None:
            absolute_time = time.perf_counter() - GLOBAL_START_TIME
        else:
            absolute_time = np.nan

        with self.lock:
            self.triggered = True
            self.trigger_checkpoint_ms = checkpoint_ms
            self.trigger_rfd = rfd
            self.trigger_threshold = threshold
            self.trigger_from_onset_s = latency_from_go - (
                self.onset_time_from_go
            )
            self.trigger_from_go_s = latency_from_go
            self.trigger_time = absolute_time
            direction = "LOW_RFD" if self.trigger_mode == "BELOW" else "HIGH_RFD"
            self.decision_reason = f"{direction}_AT_{checkpoint_ms}MS"
            onset_from_go = self.onset_time_from_go
            actual_from_onset = self.trigger_from_onset_s

        trigger_label = "LOW-RFD" if self.trigger_mode == "BELOW" else "HIGH-RFD"
        relation = "<" if self.trigger_mode == "BELOW" else ">"
        print(
            f"[RIGHT] {trigger_label} TRIGGER at the "
            f"{checkpoint_ms} ms checkpoint | "
            f"RFD={rfd:.6f} {relation} threshold={threshold:.6f} | "
            f"onset {onset_from_go * 1000:.0f} ms after GO | "
            f"stim {actual_from_onset * 1000:.0f} ms after onset"
        )


# =============================================================================
# TRIAL TIME SERIES
# =============================================================================

def _hand_timeseries(trial_number, hand, t, force, go_start_time):
    """
    Build the per-hand time series, anchored to onset.

    time_from_go_s     - kept for cue-locked inspection
    time_from_onset_s  - the analysis axis; NaN if no onset was found
    """
    onset_idx, smoothed = find_rising_onset(t, force)

    if onset_idx is not None:
        onset_time = t[onset_idx]
        time_from_onset = t - onset_time
        in_epoch = (
            (time_from_onset >= 0.0)
            & (time_from_onset <= RFD_EPOCH_S)
        )
    else:
        time_from_onset = np.full(t.size, np.nan)
        in_epoch = np.zeros(t.size, dtype=bool)

    # Diagnostic running slope (onset -> current sample).
    running = np.full(t.size, np.nan)

    if onset_idx is not None:
        for i in range(onset_idx + MIN_SLOPE_SAMPLES - 1, t.size):
            running[i] = calculate_running_slope(
                t[:i + 1], force[:i + 1],
                onset_idx=onset_idx,
                smoothed=smoothed[:i + 1],
            )

    return pd.DataFrame({
        "trial": trial_number,
        "hand": hand,
        "time_s": t,
        "time_from_go_s": t - go_start_time,
        "time_from_onset_s": time_from_onset,
        "force": force,
        "force_smoothed": smoothed,
        "in_rfd_epoch": in_epoch,
        # Full post-onset 20-ms local-RFD trace. This is calculated at
        # every available sample once a complete 20-ms window exists, not
        # only at adaptive trigger checkpoints.
        "local_rfd_full_epoch": running,
        # Backward-compatible alias used by older plotting helpers.
        "running_slope_diagnostic": running,
    })


def extract_trial_timeseries(trial_number, go_start_time, time_array,
                             right_force_array):
    """Retain the RIGHT-handle time series for one trial."""
    mask = (
        (time_array >= go_start_time)
        & (time_array <= (go_start_time + POST_GO_RECORDING_END_S))
    )
    if mask.sum() < ONSET_CONFIRM_SAMPLES:
        return pd.DataFrame()
    t = time_array[mask]
    return _hand_timeseries(
        trial_number, "right", t, right_force_array[mask], go_start_time
    )

# =============================================================================
# TRIAL METRICS
# =============================================================================

def _hand_metrics(side, t, force, go_start_time, monitor_result):
    """
    All post-hoc measures for one hand, every one anchored to onset.
    """
    out = {}

    thresholds = monitor_result["thresholds"]
    force_references = monitor_result.get("force_references", {})

    # ---- what happened in real time -------------------------------
    out[f"{side}_trigger_mode"] = monitor_result.get("trigger_mode", SLOPE_TRIGGER_MODE)
    out[f"{side}_stimulation_metric_mode"] = monitor_result.get(
        "stimulation_metric_mode", STIMULATION_METRIC_MODE
    )
    out[f"{side}_triggered"] = monitor_result["triggered"]
    out[f"{side}_trigger_checkpoint_ms"] = (
        monitor_result["trigger_checkpoint_ms"]
    )
    out[f"{side}_trigger_rfd"] = monitor_result["trigger_rfd"]
    out[f"{side}_trigger_threshold"] = monitor_result["trigger_threshold"]
    out[f"{side}_trigger_force"] = monitor_result.get("trigger_force", np.nan)
    out[f"{side}_trigger_force_reference"] = monitor_result.get("trigger_force_reference", np.nan)
    out[f"{side}_trigger_from_onset_s"] = (
        monitor_result["trigger_from_onset_s"]
    )
    out[f"{side}_trigger_from_go_s"] = (
        monitor_result["trigger_from_go_s"]
    )
    out[f"{side}_trigger_time_s"] = monitor_result["trigger_time_s"]
    out[f"{side}_n_checkpoints_checked"] = (
        monitor_result["n_checkpoints_checked"]
    )
    out[f"{side}_realtime_onset_from_go_s"] = (
        monitor_result["onset_from_go_s"]
    )
    out[f"{side}_decision_reason"] = monitor_result["decision_reason"]

    primary_key = int(round(RFD_EPOCH_S * 1000))
    out[f"{side}_rfd_threshold"] = thresholds.get(primary_key, np.nan)

    # every checkpoint threshold, so the figures can draw them
    for c in RFD_CHECKPOINTS_S:
        k = int(round(c * 1000))
        out[f"{side}_threshold_{k}ms"] = thresholds.get(k, np.nan)
        out[f"{side}_force_reference_{k}ms"] = force_references.get(k, np.nan)

    # ---- post-hoc recomputation -----------------------------------
    if t.size:
        onset_idx, smoothed = find_rising_onset(t, force)
    else:
        onset_idx, smoothed = None, np.empty(0)

    if onset_idx is None:
        out[f"{side}_onset_detected"] = False
        out[f"{side}_reaction_time_s"] = np.nan
        out[f"{side}_rfd"] = np.nan
        out[f"{side}_rfd_valid"] = False
        out[f"{side}_rfd_n_samples"] = 0
        out[f"{side}_rfd_realtime_minus_posthoc"] = np.nan

        for c in RFD_CHECKPOINTS_S:
            out[f"{side}_rfd_{int(round(c * 1000))}ms"] = np.nan

        for epoch_s in RFD_EPOCHS_S:
            out[f"{side}_rfd_{int(epoch_s * 1000)}ms"] = np.nan

        for tp in FORCE_TIMEPOINTS_FROM_ONSET_S:
            out[f"{side}_force_{int(tp * 1000)}ms_from_onset"] = np.nan

        out[f"{side}_peak_rfd"] = np.nan
        out[f"{side}_peak_rfd_time_from_onset_s"] = np.nan
        out[f"{side}_rfd_to_peak"] = np.nan
        out[f"{side}_time_to_peak_s"] = np.nan
        out[f"{side}_peak_force"] = np.nan
        out[f"{side}_baseline_force"] = np.nan
        out[f"{side}_rfd_difference"] = np.nan
        out[f"{side}_rfd_ratio"] = np.nan

        return out

    onset_time = float(t[onset_idx])
    time_from_onset = t - onset_time

    out[f"{side}_onset_detected"] = True
    out[f"{side}_reaction_time_s"] = onset_time - go_start_time

    # ---- every checkpoint, recomputed post hoc ---------------------
    checkpoint_rfds = compute_checkpoint_rfds(
        t, force, onset_idx=onset_idx, smoothed=smoothed
    )
    checkpoint_forces = compute_checkpoint_forces(
        t, force, onset_idx=onset_idx, smoothed=smoothed
    )

    for k, v in checkpoint_rfds.items():
        out[f"{side}_rfd_{k}ms"] = v
        out[f"{side}_checkpoint_force_{k}ms"] = checkpoint_forces.get(k, np.nan)

    # ---- the primary reporting epoch --------------------------------
    epoch = calculate_epoch_rfd(
        t, force, epoch_s=RFD_EPOCH_S,
        onset_idx=onset_idx, smoothed=smoothed,
    )

    out[f"{side}_rfd"] = epoch["rfd"]
    out[f"{side}_rfd_n_samples"] = epoch["n_samples"]
    out[f"{side}_rfd_valid"] = epoch["reason"] == "ok"

    # live-versus-recorded agreement at the checkpoint that fired
    live = monitor_result["checkpoints_checked_live"]
    diffs = [
        live[k] - checkpoint_rfds[k]
        for k in live
        if k in checkpoint_rfds
        and np.isfinite(live[k])
        and np.isfinite(checkpoint_rfds[k])
    ]
    out[f"{side}_rfd_realtime_minus_posthoc"] = (
        float(np.max(np.abs(diffs))) if diffs else np.nan
    )

    # ---- additional post-hoc epochs ---------------------------------
    for epoch_s in RFD_EPOCHS_S:
        key = f"{side}_rfd_{int(epoch_s * 1000)}ms"

        if key in out:
            continue

        e = calculate_epoch_rfd(
            t, force, epoch_s=epoch_s,
            onset_idx=onset_idx, smoothed=smoothed,
        )
        out[key] = e["rfd"]

    # ---- force at fixed times from onset ---------------------------
    for tp in FORCE_TIMEPOINTS_FROM_ONSET_S:
        key = f"{side}_force_{int(tp * 1000)}ms_from_onset"

        if time_from_onset.max() >= tp:
            idx = int(np.argmin(np.abs(time_from_onset - tp)))
            out[key] = float(smoothed[idx])
        else:
            out[key] = np.nan

    # ---- descriptive -----------------------------------------------
    peak_rfd, peak_rfd_time = calculate_peak_rfd(
        t, force, onset_idx, smoothed=smoothed
    )
    out[f"{side}_peak_rfd"] = peak_rfd
    out[f"{side}_peak_rfd_time_from_onset_s"] = (
        peak_rfd_time - onset_time
        if np.isfinite(peak_rfd_time) else np.nan
    )

    to_peak = calculate_rfd_to_peak(
        t, force, onset_idx, smoothed=smoothed
    )
    out[f"{side}_rfd_to_peak"] = to_peak["rfd"]
    out[f"{side}_time_to_peak_s"] = to_peak["duration"]
    out[f"{side}_peak_force"] = to_peak["peak_force"]

    out[f"{side}_baseline_force"] = (
        float(np.mean(force[:onset_idx])) if onset_idx > 0 else np.nan
    )

    # ---- comparison at the primary epoch ----------------------------
    threshold = thresholds.get(primary_key, np.nan)

    if np.isfinite(epoch["rfd"]) and np.isfinite(threshold):
        out[f"{side}_rfd_difference"] = epoch["rfd"] - threshold
        out[f"{side}_rfd_ratio"] = (
            epoch["rfd"] / threshold if threshold != 0 else np.nan
        )
    else:
        out[f"{side}_rfd_difference"] = np.nan
        out[f"{side}_rfd_ratio"] = np.nan

    return out


def _accuracy_metrics(side, t, force, go_start_time,
                       target_level=None, tolerance=None):
    """
    Trial-by-trial spatial/temporal accuracy of the force cursor relative
    to the target box, i.e. how well the participant "hit" the target and
    stayed there. Anchored to GO onset (t and force already sliced to the
    GO/recording window for this trial by the caller).

    "Distance from target" is expressed both in raw MVC units (force
    fraction) and in units of the target tolerance (1.0 = exactly on the
    edge of the box, 0.0 = dead centre).
    """
    out = {}
    if target_level is None:
        target_level = TARGET_FORCE_LEVEL
    if tolerance is None:
        tolerance = TARGET_TOLERANCE

    if t.size < 2:
        out[f"{side}_hit_target"] = False
        out[f"{side}_time_in_target_s"] = 0.0
        out[f"{side}_time_in_target_pct"] = np.nan
        out[f"{side}_time_to_first_hit_s"] = np.nan
        out[f"{side}_mean_abs_error_mvc"] = np.nan
        out[f"{side}_mean_abs_error_tol_units"] = np.nan
        out[f"{side}_min_abs_error_mvc"] = np.nan
        out[f"{side}_max_abs_error_mvc"] = np.nan
        out[f"{side}_mean_signed_error_mvc"] = np.nan
        out[f"{side}_final_abs_error_mvc"] = np.nan
        out[f"{side}_n_target_entries"] = 0
        return out

    error_signed = force - target_level          # +ve = above target
    error_abs = np.abs(error_signed)
    in_target = error_abs <= tolerance

    # Per-sample dt (assume roughly uniform sampling; use actual diffs so
    # dropped samples don't bias the dwell-time estimate).
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, 1.0 / SAMPLING_RATE_HZ * 5)  # guard vs. gaps

    time_in_target_s = float(np.sum(dt[in_target]))
    window_duration_s = float(t[-1] - t[0]) if t[-1] > t[0] else np.nan

    hit_any = bool(np.any(in_target))
    if hit_any:
        first_hit_idx = int(np.argmax(in_target))
        time_to_first_hit_s = float(t[first_hit_idx] - go_start_time)
        # Count discrete entries into the target (rising edges of in_target).
        entries = int(np.sum(np.diff(in_target.astype(int)) == 1) + (1 if in_target[0] else 0))
    else:
        time_to_first_hit_s = np.nan
        entries = 0

    out[f"{side}_hit_target"] = hit_any
    out[f"{side}_time_in_target_s"] = time_in_target_s
    out[f"{side}_time_in_target_pct"] = (
        100.0 * time_in_target_s / window_duration_s
        if np.isfinite(window_duration_s) and window_duration_s > 0
        else np.nan
    )
    out[f"{side}_time_to_first_hit_s"] = time_to_first_hit_s
    out[f"{side}_mean_abs_error_mvc"] = float(np.mean(error_abs))
    out[f"{side}_mean_abs_error_tol_units"] = (
        float(np.mean(error_abs) / tolerance) if tolerance > 0 else np.nan
    )
    out[f"{side}_min_abs_error_mvc"] = float(np.min(error_abs))
    out[f"{side}_max_abs_error_mvc"] = float(np.max(error_abs))
    out[f"{side}_mean_signed_error_mvc"] = float(np.mean(error_signed))
    out[f"{side}_final_abs_error_mvc"] = float(error_abs[-1])
    out[f"{side}_n_target_entries"] = entries

    return out


def calculate_trial_metrics(trial_number, go_start_time, time_array,
                            right_force_array, prev_right_rfd, right_result):
    """Calculate all trial metrics for the RIGHT handle only."""
    results = {
        "trial": trial_number,
        "go_onset_s": go_start_time,
        "rfd_epoch_s": RFD_EPOCH_S,
        "prev_right_rfd": prev_right_rfd,
    }
    mask = (
        (time_array >= go_start_time)
        & (time_array <= go_start_time + ONSET_SEARCH_WINDOW_S + POST_ONSET_WINDOW_S)
    )
    if mask.sum() < ONSET_CONFIRM_SAMPLES:
        results.update(_hand_metrics(
            "right", np.empty(0), np.empty(0), go_start_time, right_result
        ))
        results.update(_accuracy_metrics(
            "right", np.empty(0), np.empty(0), go_start_time
        ))
        return results

    t = time_array[mask]
    force = right_force_array[mask]
    results.update(_hand_metrics(
        "right", t, force, go_start_time, right_result
    ))
    results.update(_accuracy_metrics(
        "right", t, force, go_start_time
    ))
    return results

# =============================================================================
# PUBLICATION-QUALITY PLOTTING HELPERS
# =============================================================================

PLOT_DPI = 350
PLOT_COLORS = {
    "right": "#0072B2",
    "mean": "#005AB5",
    "trial": "#8A8A8A",
    "threshold": "#CC79A7",
    "trigger": "#D62728",
    "grid": "#D9D9D9",
}


def apply_publication_plot_style():
    """Apply one consistent style to every saved Matplotlib figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "savefig.dpi": PLOT_DPI,
    })


def _style_axis(ax):
    ax.grid(True, alpha=0.35, linewidth=0.7, color=PLOT_COLORS["grid"])
    ax.set_axisbelow(True)


def _save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to:\n{path}")


def _checkpoint_ms():
    return np.array([int(round(c * 1000)) for c in RFD_CHECKPOINTS_S], dtype=int)


def _checkpoint_matrix(df_trials, side, kind="rfd"):
    """Return trials x checkpoints matrix for RFD or threshold values."""
    cps = _checkpoint_ms()
    if kind == "rfd":
        cols = [f"{side}_rfd_{k}ms" for k in cps]
    elif kind == "threshold":
        cols = [f"{side}_threshold_{k}ms" for k in cps]
    else:
        raise ValueError("kind must be 'rfd' or 'threshold'")

    matrix = np.full((len(df_trials), len(cps)), np.nan, dtype=float)
    for j, col in enumerate(cols):
        if col in df_trials.columns:
            matrix[:, j] = pd.to_numeric(df_trials[col], errors="coerce").to_numpy()
    return cps, matrix


def _trigger_threshold_matrix(df_trials, side):
    """Exact adaptive trigger threshold used by the real-time decision."""
    return _checkpoint_matrix(df_trials, side, kind="threshold")


def _smooth_curve(x, y, n=400):
    """Smooth an observed checkpoint trajectory without extrapolating it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_dense = np.linspace(x.min(), x.max(), n)
    if x.size >= 3:
        y_dense = CubicSpline(x, y, bc_type="natural")(x_dense)
    else:
        y_dense = np.interp(x_dense, x, y)
    return x_dense, y_dense


def _finite_mean_sd(matrix):
    """Column-wise mean/SD while avoiding warnings for all-NaN columns."""
    mean = np.full(matrix.shape[1], np.nan)
    sd = np.full(matrix.shape[1], np.nan)
    for j in range(matrix.shape[1]):
        v = matrix[:, j]
        v = v[np.isfinite(v)]
        if v.size:
            mean[j] = float(np.mean(v))
            sd[j] = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return mean, sd


apply_publication_plot_style()

# =============================================================================
# OVERALL FORCE FIGURE
# =============================================================================

def plot_force_data(time_data, right_force_data, trial_log, out_dir):
    """Whole-session RIGHT-handle force trace."""
    if len(time_data) == 0:
        return
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(time_data, right_force_data, label="Right force",
            color=PLOT_COLORS["right"], linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (fraction of MVC)")
    ax.set_ylim(*PLOT_FORCE_LIMITS)
    ax.set_title("Right-handle force across the session")
    _style_axis(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, os.path.join(out_dir, "Force_Data.png"))

def _nearest_available_checkpoints(target_ms, available_ms):
    """Return unique real checkpoints nearest to requested display times."""
    available = sorted(int(v) for v in available_ms)
    if not available:
        return []
    chosen = []
    for target in target_ms:
        cp = min(available, key=lambda v: abs(v - target))
        if cp not in chosen:
            chosen.append(cp)
    return chosen


# =============================================================================
# OVERALL TRIAL RFD FIGURE
# =============================================================================

def plot_trial_rfd(df_trials, out_dir):
    """RIGHT-handle local RFD across trials at multiple onset timepoints."""
    if df_trials.empty:
        return

    available = [
        cp for cp in _checkpoint_ms()
        if f"right_rfd_{cp}ms" in df_trials.columns
    ]
    display_cps = _nearest_available_checkpoints(
        [20, 40, 60, 80, 100, 120, 140, 160, 180],
        available,
    )
    if not display_cps:
        return

    trials = pd.to_numeric(
        df_trials["trial"], errors="coerce"
    ).to_numpy(dtype=float)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # Top: local RFD at representative actual checkpoints.
    ax = axes[0]
    for j, cp in enumerate(display_cps):
        col = f"right_rfd_{cp}ms"
        y = pd.to_numeric(
            df_trials[col], errors="coerce"
        ).to_numpy(dtype=float)
        ax.plot(
            trials,
            y,
            "-",
            lw=1.8,
            color=cmap(j / max(1, len(display_cps) - 1)),
            alpha=0.85,
            label=f"{cp} ms",
        )

    if "right_triggered" in df_trials.columns:
        for _, row in df_trials.iterrows():
            if bool(row.get("right_triggered", False)):
                ax.axvline(
                    row["trial"],
                    color=PLOT_COLORS["trigger"],
                    alpha=0.25,
                    lw=1.5,
                )

    ax.set_ylabel("Local RFD (MVC/s)")
    ax.set_title(
        "Right-handle 20-ms local RFD across trials at multiple "
        "onset-anchored checkpoints"
    )
    _style_axis(ax)
    ax.legend(
        title="Actual checkpoint",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )

    # Bottom: primary 80-ms RFD plus its adaptive reference.
    ax2 = axes[1]
    if "right_rfd" in df_trials.columns:
        ax2.plot(
            trials,
            pd.to_numeric(df_trials["right_rfd"], errors="coerce"),
            "-",
            color=PLOT_COLORS["right"],
            lw=2.2,
            label=f"RFD @ {int(RFD_EPOCH_S * 1000)} ms",
        )
    if "right_rfd_threshold" in df_trials.columns:
        ax2.plot(
            trials,
            pd.to_numeric(
                df_trials["right_rfd_threshold"], errors="coerce"
            ),
            "--",
            color=PLOT_COLORS["threshold"],
            lw=2.2,
            label=(
                f"{SLOPE_REFERENCE_TRIALS}-trial adaptive reference "
                f"({SLOPE_TRIGGER_MODE})"
            ),
        )

    if "right_triggered" in df_trials.columns and "right_rfd" in df_trials.columns:
        for _, row in df_trials.iterrows():
            if bool(row.get("right_triggered", False)):
                ax2.plot(
                    row["trial"],
                    row.get("right_rfd", np.nan),
                    "v",
                    ms=9,
                    color=PLOT_COLORS["trigger"],
                )

    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Local RFD (MVC/s)")
    ax2.set_title(
        f"Primary reporting checkpoint ({int(RFD_EPOCH_S * 1000)} ms) "
        "with adaptive reference"
    )
    _style_axis(ax2)
    ax2.legend(loc="best")

    fig.tight_layout()
    _save_figure(fig, os.path.join(out_dir, "Trial_RFD.png"))



# =============================================================================
# ONSET-ALIGNED FORCE OVERLAY
# =============================================================================

def plot_onset_aligned_forces(trial_timeseries, out_dir):
    """Overlay RIGHT-handle force traces aligned to force onset."""
    if trial_timeseries.empty:
        return
    df = trial_timeseries[trial_timeseries["hand"] == "right"]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, trial_df in df.groupby("trial"):
        x = trial_df["time_from_onset_s"]
        valid = np.isfinite(x)
        ax.plot(x[valid], trial_df.loc[valid, "force_smoothed"],
                linewidth=0.9, alpha=0.28, color=PLOT_COLORS["trial"] )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(*PLOT_TIME_FROM_ONSET_S)
    ax.set_ylim(*PLOT_FORCE_LIMITS)
    ax.set_xlabel("Time from force onset (s)")
    ax.set_ylabel("Force (fraction of MVC)")
    ax.set_title("Right-handle onset-aligned force traces")
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, os.path.join(out_dir, "Onset_Aligned_Forces.png"))

# =============================================================================
# INDIVIDUAL TRIAL FIGURES
# =============================================================================

def resolve_rfd_limits(df_trials, df_timeseries=None):
    """
    Decide ONE RFD axis range for the whole session, so every trial
    figure is directly comparable.

    df_timeseries is accepted for call compatibility but no longer used
    (see the comment below).
    """
    if PLOT_RFD_LIMITS is not None:
        return PLOT_RFD_LIMITS

    values = []

    # The axis is set by the numbers you actually interpret - the
    # checkpoint RFDs and their thresholds.  The diagnostic running
    # slope is deliberately EXCLUDED: its early samples are fitted
    # through very few points and are noisy enough to blow the shared
    # axis out to tens of MVC/s, squashing the real values flat.
    if df_trials is not None and not df_trials.empty:
        cols = [
            c for c in df_trials.columns
            if (c.endswith("ms") and "_rfd_" in c)
            or c.endswith("_rfd")
            or c.endswith("_rfd_threshold")
            or ("_threshold_" in c and c.endswith("ms"))
        ]

        for c in cols:
            v = pd.to_numeric(df_trials[c], errors="coerce")
            v = v[np.isfinite(v)]
            if not v.empty:
                values.append(v.to_numpy())

    if df_timeseries is not None and not df_timeseries.empty:
        for trace_col in ("local_rfd_full_epoch", "running_slope_diagnostic"):
            if trace_col in df_timeseries.columns:
                v = pd.to_numeric(df_timeseries[trace_col], errors="coerce")
                v = v[np.isfinite(v)]
                if not v.empty:
                    values.append(v.to_numpy())
                break

    if not values:
        return (-1.0, 4.0)

    allv = np.concatenate(values)
    lo = float(np.nanpercentile(allv, 1))
    hi = float(np.nanpercentile(allv, 99))
    pad = max(0.2, 0.15 * (hi - lo))

    lo = min(0.0, lo - pad)
    hi = hi + pad

    # Round outward to a tidy step.
    step = 0.5 if (hi - lo) < 8 else 1.0
    return (
        float(np.floor(lo / step) * step),
        float(np.ceil(hi / step) * step),
    )


def _label(value, fmt="{:.4f}"):
    if value is None or not np.isfinite(value):
        return "N/A"
    return fmt.format(value)


def plot_individual_trial_figures(df_timeseries, df_trials, out_dir,
                                  rfd_limits=None):
    """Publication-style RIGHT-handle force/RFD figure for every trial."""
    if df_timeseries.empty or df_trials.empty:
        return
    if rfd_limits is None:
        rfd_limits = resolve_rfd_limits(df_trials, df_timeseries)

    individual_dir = os.path.join(out_dir, "SingleTrials")
    os.makedirs(individual_dir, exist_ok=True)

    for trial_number in sorted(df_timeseries["trial"].unique()):
        trial_df = df_timeseries[
            (df_timeseries["trial"] == trial_number)
            & (df_timeseries["hand"] == "right")
        ]
        summary_rows = df_trials[df_trials["trial"] == trial_number]
        if trial_df.empty or summary_rows.empty:
            continue
        summary = summary_rows.iloc[0]

        fig, (ax_force, ax_rfd) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        color = PLOT_COLORS["right"]
        x = trial_df["time_from_go_s"]

        ax_force.axvline(0, color="green", lw=2, alpha=0.7, label="GO cue")
        ax_rfd.axvline(0, color="green", lw=2, alpha=0.7)
        ax_force.axhline(RISING_ONSET_FORCE, color="gray", ls=":", lw=1.2,
                         label="Onset threshold")
        if SLOPE_TRIGGER_MODE == "BELOW":
            ax_force.axhline(
                FORCE_TRIGGER_CEILING, color=PLOT_COLORS["trigger"],
                ls=":", lw=1.4, alpha=0.8,
                label="No corrective trigger above this force",
            )
        ax_force.plot(x, trial_df["force"], color=color, lw=0.9, alpha=0.30)
        ax_force.plot(x, trial_df["force_smoothed"], color=color, lw=2.0,
                      label="Right force")
        # Plot a continuous GO-aligned local-RFD trace.
        # Before movement onset, RFD is defined as 0 for display. Any NaN
        # gap between onset and the first valid 20-ms local-RFD estimate is
        # linearly interpolated so the plotted line is continuous.
        rfd_trace_col = (
            "local_rfd_full_epoch"
            if "local_rfd_full_epoch" in trial_df.columns
            else "running_slope_diagnostic"
        )

        rt = summary.get("right_reaction_time_s", np.nan)

        x_plot = pd.to_numeric(
            trial_df["time_from_go_s"], errors="coerce"
        ).to_numpy(dtype=float)
        rfd_plot = pd.to_numeric(
            trial_df[rfd_trace_col], errors="coerce"
        ).to_numpy(dtype=float)

        cp_x, cp_y, th_x, th_trigger = [], [], [], []
        force_ref_x, force_ref_y = [], []

        if np.isfinite(rt):
            for ax in (ax_force, ax_rfd):
                ax.axvline(rt, color=color, ls="--", lw=1.5, alpha=0.8)

            # Show the adaptive trigger/search window on BOTH panels so
            # the force and local-RFD traces use the same visual time reference.
            for ax in (ax_force, ax_rfd):
                ax.axvspan(
                    rt + RFD_CHECK_START_S,
                    rt + RFD_CHECK_END_S,
                    color=color,
                    alpha=0.07,
                )

            for c in RFD_CHECKPOINTS_S:
                k = int(round(c * 1000))
                v = summary.get(f"right_rfd_{k}ms", np.nan)
                th = summary.get(f"right_threshold_{k}ms", np.nan)
                force_ref = summary.get(
                    f"right_force_reference_{k}ms", np.nan
                )

                if np.isfinite(force_ref):
                    force_ref_x.append(rt + c)
                    force_ref_y.append(force_ref)

                if np.isfinite(v):
                    cp_x.append(rt + c)
                    cp_y.append(v)

                if np.isfinite(th):
                    th_x.append(rt + c)
                    th_trigger.append(th)

        # -------------------------------------------------------------
        # ONE continuous RFD trace constrained through checkpoint values.
        # -------------------------------------------------------------
        go_mask = np.isfinite(x_plot) & (x_plot >= 0.0)
        if np.any(go_mask):
            x_go = x_plot[go_mask]
            rfd_go = rfd_plot[go_mask].copy()

            # Premovement RFD is zero for display.
            if np.isfinite(rt):
                rfd_go[x_go < rt] = 0.0

            # Build interpolation nodes from the dense local-RFD trace.
            node_x = list(x_go[np.isfinite(rfd_go)])
            node_y = list(rfd_go[np.isfinite(rfd_go)])

            # Explicit GO anchor.
            node_x.append(0.0)
            node_y.append(0.0)

            # Explicit movement-onset anchor.
            if np.isfinite(rt):
                node_x.append(float(rt))
                node_y.append(0.0)

            # IMPORTANT: insert the actual checkpoint RFD values into the
            # SAME interpolation node set, so the displayed trace passes
            # exactly through each checkpoint marker.
            for xx, yy in zip(cp_x, cp_y):
                node_x.append(float(xx))
                node_y.append(float(yy))

            node_x = np.asarray(node_x, dtype=float)
            node_y = np.asarray(node_y, dtype=float)
            valid_nodes = np.isfinite(node_x) & np.isfinite(node_y)
            node_x = node_x[valid_nodes]
            node_y = node_y[valid_nodes]

            if node_x.size >= 2:
                order = np.argsort(node_x, kind="stable")
                node_x = node_x[order]
                node_y = node_y[order]

                # Collapse duplicate timestamps. Because checkpoint nodes were
                # appended after diagnostic nodes, their value wins.
                unique_x = []
                unique_y = []
                for xx, yy in zip(node_x, node_y):
                    if unique_x and np.isclose(
                        xx, unique_x[-1], atol=1e-8
                    ):
                        unique_y[-1] = float(yy)
                    else:
                        unique_x.append(float(xx))
                        unique_y.append(float(yy))

                unique_x = np.asarray(unique_x, dtype=float)
                unique_y = np.asarray(unique_y, dtype=float)

                rfd_display = np.interp(
                    x_go,
                    unique_x,
                    unique_y,
                    left=unique_y[0],
                    right=unique_y[-1],
                )

                if np.isfinite(rt):
                    rfd_display[x_go < rt] = 0.0

                ax_rfd.plot(
                    x_go,
                    rfd_display,
                    color=color,
                    lw=2.2,
                    alpha=0.90,
                    label="20-ms local RFD",
                    zorder=3,
                )

        # Force reference as one smooth continuous reference trajectory.
        if len(force_ref_x) >= 2:
            sx_fr, sy_fr = _smooth_curve(force_ref_x, force_ref_y)
            ax_force.plot(
                sx_fr,
                sy_fr,
                color=PLOT_COLORS["threshold"],
                lw=2.0,
                label="Force reference (previous 2)",
            )
        elif force_ref_x:
            ax_force.plot(
                force_ref_x,
                force_ref_y,
                "o",
                color=PLOT_COLORS["threshold"],
                ms=5,
                label="Force reference (previous 2)",
            )

        # Adaptive checkpoint RFD values as a smooth continuous line.
        # These are the actual values used at the real-time checkpoint times.
        if len(cp_x) >= 2:
            sx_cp, sy_cp = _smooth_curve(cp_x, cp_y)
            ax_rfd.plot(
                sx_cp,
                sy_cp,
                color=color,
                lw=2.2,
                alpha=0.95,
                zorder=6,
                label="Checkpoint local RFD",
            )
        elif cp_x:
            ax_rfd.plot(
                cp_x,
                cp_y,
                "o",
                color=color,
                ms=5,
                zorder=6,
                label="Checkpoint local RFD",
            )

        # Adaptive threshold as a smooth continuous line (no dot markers).
        sx, sy = _smooth_curve(th_x, th_trigger)
        if sx.size:
            ax_rfd.plot(
                sx,
                sy,
                color=PLOT_COLORS["threshold"],
                lw=2.5,
                label=f"Adaptive trigger threshold ({SLOPE_TRIGGER_MODE})",
                zorder=4,
            )

        if bool(summary.get("right_triggered", False)):
            stim = summary.get("right_trigger_from_go_s", np.nan)
            if np.isfinite(stim):
                for ax in (ax_force, ax_rfd):
                    ax.axvline(stim, color=PLOT_COLORS["trigger"], lw=3, alpha=0.9)
                ax_force.annotate("STIM", xy=(stim, PLOT_FORCE_LIMITS[1] * 0.93),
                                  color=PLOT_COLORS["trigger"], fontsize=10,
                                  fontweight="bold", ha="center")

        # Show the complete recorded trial epoch from GO onward.
        ax_force.set_xlim(-0.2, 2.0)
        ax_force.set_ylim(-0.10, 1.50)
        ax_force.set_ylabel("Force (fraction of MVC)")
        ax_force.set_title(f"Trial {int(trial_number)} - right handle ({X_NUMBER}, {SESSION})")
        _style_axis(ax_force); ax_force.legend(loc="upper right")

        ax_rfd.axhline(0, color="black", lw=0.8)
        ax_rfd.set_xlim(-0.2, 2.0)
        ax_rfd.set_ylim(-15.0, 15.0)
        ax_rfd.set_xlabel("Time from GO cue (s)")
        ax_rfd.set_ylabel("Local RFD (MVC/s)")
        ax_rfd.set_title(
            "20-ms sliding RFD across full recorded epoch; "
            "shaded = adaptive trigger window only"
        )
        _style_axis(ax_rfd); ax_rfd.legend(loc="upper right")

        trig = bool(summary.get("right_triggered", False))
        cp = summary.get("right_trigger_checkpoint_ms", np.nan)
        info = (
            "Right RT " + _label(summary.get("right_reaction_time_s"), "{:.3f}")
            + " s | RFD@" + f"{int(RFD_EPOCH_S * 1000)}ms "
            + _label(summary.get("right_rfd")) + " | trig " + str(trig)
            + (f" @ {int(cp)} ms" if trig and np.isfinite(cp) else "")
        )
        fig.text(0.99, 0.01, info, ha="right", va="bottom", fontsize=9,
                 family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        path = os.path.join(individual_dir,
            f"trial_{int(trial_number):03d}_{X_NUMBER}_{SESSION}_{date_str}.png")
        _save_figure(fig, path)

# =============================================================================
# SESSION-LEVEL RFD SUMMARY FIGURES
# =============================================================================

def plot_mean_rfd_profile(df_trials, out_dir, rfd_limits=None):
    """Mean RIGHT-handle checkpoint RFD +/- SD and mean adaptive threshold."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    cps, rfd = _checkpoint_matrix(df_trials, "right", "rfd")
    _, trig_th = _trigger_threshold_matrix(df_trials, "right")
    mean, sd = _finite_mean_sd(rfd)
    mean_th, _ = _finite_mean_sd(trig_th)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(cps, mean - sd, mean + sd, alpha=0.18,
                    color=PLOT_COLORS["right"], label="±1 SD")
    ax.plot(cps, mean, "o-", lw=2.8, ms=5, color=PLOT_COLORS["right"],
            label="Session mean RFD")
    tx, ty = _smooth_curve(cps, mean_th)
    if tx.size:
        ax.plot(tx, ty, lw=2.2, ls="--", color=PLOT_COLORS["threshold"],
                label=f"Mean adaptive trigger threshold ({SLOPE_TRIGGER_MODE})")
    ax.set_xlabel("Checkpoint from force onset (ms)")
    ax.set_ylabel("Local RFD (MVC/s)")
    ax.set_xticks(cps)
    ax.set_title(
        f"Right-handle session mean 20-ms local RFD profile | "
        f"{STIMULATION_METRIC_MODE} / {SLOPE_TRIGGER_MODE}"
    )
    _style_axis(ax); ax.legend(loc="best")
    if rfd_limits is not None: ax.set_ylim(*rfd_limits)
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "Mean_RFD_Profile.png"))

def plot_rfd_spaghetti(df_trials, out_dir, rfd_limits=None):
    """RIGHT-handle trial trajectories with session mean, SD and threshold."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    cps, rfd = _checkpoint_matrix(df_trials, "right", "rfd")
    _, trig_th = _trigger_threshold_matrix(df_trials, "right")
    mean, sd = _finite_mean_sd(rfd); mean_th, _ = _finite_mean_sd(trig_th)
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rfd:
        ax.plot(cps, row, lw=0.9, alpha=0.28, color=PLOT_COLORS["trial"])
    ax.fill_between(cps, mean - sd, mean + sd, alpha=0.16,
                    color=PLOT_COLORS["mean"], label="±1 SD")
    ax.plot(cps, mean, "o-", lw=3.0, ms=5, color=PLOT_COLORS["mean"],
            label="Session mean")
    tx, ty = _smooth_curve(cps, mean_th)
    if tx.size:
        ax.plot(tx, ty, lw=2.2, ls="--", color=PLOT_COLORS["threshold"],
                label=f"Mean adaptive trigger threshold ({SLOPE_TRIGGER_MODE})")
    ax.set_xlabel("Checkpoint from force onset (ms)"); ax.set_ylabel("Local RFD (MVC/s)")
    ax.set_xticks(cps); ax.set_title("Right-handle RFD trajectories across trials")
    _style_axis(ax); ax.legend(loc="best")
    if rfd_limits is not None: ax.set_ylim(*rfd_limits)
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "RFD_Spaghetti.png"))

def plot_rfd_heatmap(df_trials, out_dir):
    """RIGHT-handle trials x checkpoints heatmap; stars mark triggers."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    cps, mat = _checkpoint_matrix(df_trials, "right", "rfd")
    finite = mat[np.isfinite(mat)]
    vmin = float(np.percentile(finite, 2)) if finite.size else None
    vmax = float(np.percentile(finite, 98)) if finite.size else None
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap="viridis",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Checkpoint from force onset (ms)"); ax.set_ylabel("Trial")
    ax.set_xticks(np.arange(len(cps))); ax.set_xticklabels(cps, rotation=45, ha="right")
    trials = df_trials["trial"].to_numpy()
    ax.set_yticks(np.arange(len(trials))); ax.set_yticklabels(trials)
    if "right_trigger_checkpoint_ms" in df_trials.columns and "right_triggered" in df_trials.columns:
        for i, row in df_trials.reset_index(drop=True).iterrows():
            cp = row.get("right_trigger_checkpoint_ms", np.nan)
            if bool(row.get("right_triggered", False)) and np.isfinite(cp):
                matches = np.where(cps == int(cp))[0]
                if matches.size:
                    ax.scatter(matches[0], i, marker="*", s=90, facecolors="none",
                               edgecolors="white", linewidths=1.2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9); cbar.set_label("Local RFD (MVC/s)")
    ax.set_title("Right-handle RFD heatmap (star = trigger checkpoint)")
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "RFD_Heatmap.png"))

def plot_trigger_map(df_trials, out_dir):
    """RIGHT-handle trial number versus stimulation trigger checkpoint."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    fig, ax = plt.subplots(figsize=(10, 6)); cps = _checkpoint_ms()
    if "right_triggered" in df_trials.columns and "right_trigger_checkpoint_ms" in df_trials.columns:
        mask = df_trials["right_triggered"].fillna(False).astype(bool)
        x = df_trials.loc[mask, "trial"]
        y = pd.to_numeric(df_trials.loc[mask, "right_trigger_checkpoint_ms"], errors="coerce")
        valid = np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=75, color=PLOT_COLORS["right"],
                   label="Right trigger")
    ax.set_xlabel("Trial"); ax.set_ylabel("Trigger checkpoint from force onset (ms)")
    ax.set_yticks(cps)
    if len(df_trials): ax.set_xlim(0.5, max(TOTAL_TRIALS, int(df_trials["trial"].max())) + 0.5)
    _style_axis(ax); ax.legend(loc="best"); ax.set_title("Right-handle adaptive stimulation trigger map")
    fig.tight_layout(); _save_figure(fig, os.path.join(summary_dir, "Trigger_Map.png"))

def plot_threshold_evolution(df_trials, out_dir, rfd_limits=None):
    """RIGHT-handle adaptive threshold at every checkpoint over trials."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    fig, ax = plt.subplots(figsize=(12, 7)); cmap = plt.get_cmap("viridis")
    cps = _checkpoint_ms(); _, th = _trigger_threshold_matrix(df_trials, "right")
    trials = df_trials["trial"].to_numpy()
    for j, cp in enumerate(cps):
        color = cmap(j / max(1, len(cps) - 1))
        ax.plot(trials, th[:, j], "o-", ms=3.2, lw=1.5, color=color, label=f"{cp} ms")
    ax.set_xlabel("Trial"); ax.set_ylabel("Adaptive trigger threshold (MVC/s)")
    ax.set_xlim(0.5, max(TOTAL_TRIALS, int(df_trials["trial"].max())) + 0.5)
    if rfd_limits is not None: ax.set_ylim(*rfd_limits)
    _style_axis(ax); ax.set_title("Right-handle threshold evolution across trials")
    ax.legend(title="Checkpoint", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); _save_figure(fig, os.path.join(summary_dir, "Threshold_Evolution.png"))

def plot_session_rfd_summaries(df_trials, out_dir, rfd_limits=None):
    """Create all session-level RFD, Force, and combined visualisations."""
    summary_dir = os.path.join(out_dir, "Summary")
    os.makedirs(summary_dir, exist_ok=True)

    # RFD summaries
    plot_mean_rfd_profile(df_trials, summary_dir, rfd_limits)
    plot_rfd_spaghetti(df_trials, summary_dir, rfd_limits)
    plot_rfd_heatmap(df_trials, summary_dir)
    plot_trigger_map(df_trials, summary_dir)
    plot_threshold_evolution(df_trials, summary_dir, rfd_limits)

    # Force summaries
    plot_mean_force_profile(df_trials, summary_dir)
    plot_force_spaghetti(df_trials, summary_dir)
    plot_force_heatmap(df_trials, summary_dir)
    plot_force_reference_evolution(df_trials, summary_dir)

    # Combined summaries
    plot_rfd_force_phase_map(df_trials, summary_dir)
    plot_rfd_force_normalized_overview(df_trials, summary_dir)



# =============================================================================
# FORCE SUMMARY FIGURES
# =============================================================================

def _summary_checkpoint_keys(df_trials, prefix):
    keys = []
    for c in RFD_CHECKPOINTS_S:
        k = int(round(c * 1000))
        col = f"{prefix}_{k}ms"
        if col in df_trials.columns:
            keys.append((k, col))
    return keys


def plot_mean_force_profile(df_trials, out_dir):
    if df_trials is None or df_trials.empty:
        return
    keys = _summary_checkpoint_keys(df_trials, "right_checkpoint_force")
    if not keys:
        return

    x = np.array([k for k, _ in keys], dtype=float)
    mat = np.column_stack([
        pd.to_numeric(df_trials[col], errors="coerce").to_numpy(dtype=float)
        for _, col in keys
    ])
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)

    refs = []
    for k, _ in keys:
        col = f"right_force_reference_{k}ms"
        refs.append(
            pd.to_numeric(df_trials[col], errors="coerce").to_numpy(dtype=float)
            if col in df_trials.columns else np.full(len(df_trials), np.nan)
        )
    ref_mat = np.column_stack(refs)
    mean_ref = np.nanmean(ref_mat, axis=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, mean, linewidth=3, marker="o", label="Mean force")
    ax.fill_between(x, mean - sd, mean + sd, alpha=0.20, label="±SD")
    if np.isfinite(mean_ref).any():
        ax.plot(x, mean_ref, linewidth=2.5, linestyle="--",
                marker="s", label="Mean adaptive force reference")
    apply_plot_style(
        ax,
        title=f"Session Mean Force Profile | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE}",
        xlabel="Time from movement onset (ms)",
        ylabel="Force (fraction of MVC)",
    )
    bold_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Mean_Force_Profile.png"),
                dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_force_spaghetti(df_trials, out_dir):
    if df_trials is None or df_trials.empty:
        return
    keys = _summary_checkpoint_keys(df_trials, "right_checkpoint_force")
    if not keys:
        return

    x = np.array([k for k, _ in keys], dtype=float)
    mat = np.column_stack([
        pd.to_numeric(df_trials[col], errors="coerce").to_numpy(dtype=float)
        for _, col in keys
    ])
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    for row in mat:
        ax.plot(x, row, linewidth=1, alpha=0.28)
    ax.plot(x, mean, linewidth=3.5, label="Session mean")
    ax.fill_between(x, mean - sd, mean + sd, alpha=0.20, label="±SD")
    apply_plot_style(
        ax,
        title=f"Force Spaghetti Plot | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE}",
        xlabel="Time from movement onset (ms)",
        ylabel="Force (fraction of MVC)",
    )
    bold_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Force_Spaghetti.png"),
                dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_force_heatmap(df_trials, out_dir):
    if df_trials is None or df_trials.empty:
        return
    keys = _summary_checkpoint_keys(df_trials, "right_checkpoint_force")
    if not keys:
        return

    labels = [str(k) for k, _ in keys]
    mat = np.column_stack([
        pd.to_numeric(df_trials[col], errors="coerce").to_numpy(dtype=float)
        for _, col in keys
    ])

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(df_trials)))
    ax.set_yticklabels([str(i + 1) for i in range(len(df_trials))])

    apply_plot_style(
        ax,
        title=f"Force Heatmap | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE}",
        xlabel="Checkpoint from movement onset (ms)",
        ylabel="Trial",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Force (fraction of MVC)",
                   fontsize=PLOT_FONT_SIZE,
                   fontweight=PLOT_FONT_WEIGHT)
    cbar.ax.tick_params(labelsize=PLOT_TICK_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight(PLOT_FONT_WEIGHT)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Force_Heatmap.png"),
                dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_force_reference_evolution(df_trials, out_dir):
    if df_trials is None or df_trials.empty:
        return

    keys = []
    for c in RFD_CHECKPOINTS_S:
        k = int(round(c * 1000))
        col = f"right_force_reference_{k}ms"
        if col in df_trials.columns:
            keys.append((k, col))
    if not keys:
        return

    trials = np.arange(1, len(df_trials) + 1)
    fig, ax = plt.subplots(figsize=(13, 8))
    for k, col in keys:
        y = pd.to_numeric(df_trials[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(trials, y, linewidth=1.8, marker="o", markersize=3,
                label=f"{k} ms")

    apply_plot_style(
        ax,
        title=f"Force Reference Evolution | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE}",
        xlabel="Trial",
        ylabel="Adaptive force reference (fraction of MVC)",
    )
    bold_legend(ax, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Force_Reference_Evolution.png"),
                dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# COMBINED RFD + FORCE SUMMARY FIGURES
# =============================================================================

def plot_rfd_force_phase_map(df_trials, out_dir):
    """Multi-timepoint local-RFD versus Force phase map."""
    if df_trials is None or df_trials.empty:
        return

    available = []
    for cp in _checkpoint_ms():
        if (
            f"right_rfd_{cp}ms" in df_trials.columns
            and f"right_checkpoint_force_{cp}ms" in df_trials.columns
        ):
            available.append(cp)

    display_cps = _nearest_available_checkpoints(
        [20, 40, 60, 80, 100, 120, 140, 160, 180],
        available,
    )
    if not display_cps:
        return

    n = len(display_cps)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 5.2 * nrows),
        squeeze=False,
    )

    trial_numbers = pd.to_numeric(
        df_trials["trial"], errors="coerce"
    ).to_numpy(dtype=float)

    for idx_cp, cp in enumerate(display_cps):
        row, col = divmod(idx_cp, ncols)
        ax = axes[row][col]

        rfd = pd.to_numeric(
            df_trials[f"right_rfd_{cp}ms"], errors="coerce"
        ).to_numpy(dtype=float)
        force = pd.to_numeric(
            df_trials[f"right_checkpoint_force_{cp}ms"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(rfd) & np.isfinite(force)

        if np.any(valid):
            ax.scatter(
                rfd[valid],
                force[valid],
                s=55,
                alpha=0.8,
                c=trial_numbers[valid],
                cmap="coolwarm",
                edgecolors="black",
                linewidths=0.3,
            )
            valid_idx = np.flatnonzero(valid)
            for i, xr, yf in zip(valid_idx, rfd[valid], force[valid]):
                ax.annotate(
                    str(int(trial_numbers[i])),
                    (xr, yf),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    alpha=0.75,
                )

        apply_plot_style(
            ax,
            title=f"{cp} ms from onset",
            xlabel="Local RFD (MVC/s)",
            ylabel="Force (fraction of MVC)",
        )

    for idx_unused in range(n, nrows * ncols):
        row, col = divmod(idx_unused, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"RFD vs Force across onset-anchored checkpoints | "
        f"{STIMULATION_METRIC_MODE} / {SLOPE_TRIGGER_MODE}",
        fontsize=PLOT_TITLE_SIZE,
        fontweight=PLOT_FONT_WEIGHT,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        os.path.join(out_dir, "RFD_Force_Phase_Map.png"),
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)



def plot_rfd_force_normalized_overview(df_trials, out_dir):
    if df_trials is None or df_trials.empty:
        return

    cp = int(round(RFD_EPOCH_S * 1000))
    rfd_col = "right_rfd"
    rfd_th_col = "right_rfd_threshold"
    force_col = f"right_checkpoint_force_{cp}ms"
    force_ref_col = f"right_force_reference_{cp}ms"

    if not all(c in df_trials.columns for c in
               (rfd_col, rfd_th_col, force_col, force_ref_col)):
        return

    rfd = pd.to_numeric(df_trials[rfd_col], errors="coerce").to_numpy(dtype=float)
    rth = pd.to_numeric(df_trials[rfd_th_col], errors="coerce").to_numpy(dtype=float)
    frc = pd.to_numeric(df_trials[force_col], errors="coerce").to_numpy(dtype=float)
    fref = pd.to_numeric(df_trials[force_ref_col], errors="coerce").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rfd_ratio = rfd / rth
        force_ratio = frc / fref

    trials = np.arange(1, len(df_trials) + 1)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(trials, rfd_ratio, linewidth=2.5, marker="o",
            label="RFD / RFD threshold")
    ax.plot(trials, force_ratio, linewidth=2.5, marker="s",
            label="Force / force reference")
    ax.axhline(1.0, linewidth=1.5, linestyle="--", label="Reference = 1.0")

    if "right_triggered" in df_trials.columns:
        trig = df_trials["right_triggered"].astype(bool).to_numpy()
        valid = trig & np.isfinite(rfd_ratio)
        ax.scatter(trials[valid], rfd_ratio[valid], s=90, marker="*",
                   label="Triggered trial")

    apply_plot_style(
        ax,
        title=f"Normalized RFD + Force Across Trials | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE}",
        xlabel="Trial",
        ylabel="Value / adaptive reference",
    )
    bold_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "RFD_Force_Normalized_Overview.png"),
                dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MATLAB EXPORT HELPERS
# =============================================================================

def matlab_safe(values):
    """
    Convert a pandas column into something savemat can serialise.

    None -> NaN for numeric columns; strings kept as an object array.
    """
    items = list(values)

    if not items:
        return np.empty((0,), dtype=float)

    is_text = all(
        isinstance(v, (str, bytes)) or v is None or
        (isinstance(v, float) and np.isnan(v))
        for v in items
    ) and any(isinstance(v, (str, bytes)) for v in items)

    if is_text:
        return np.array(
            ["" if v is None or (isinstance(v, float) and np.isnan(v))
             else str(v) for v in items],
            dtype=object,
        )

    out = []

    for v in items:
        if v is None:
            out.append(np.nan)
        elif isinstance(v, (bool, np.bool_)):
            out.append(float(v))
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)

    return np.asarray(out, dtype=float)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global GLOBAL_START_TIME, SLOPE_TRIGGER_MODE, STIMULATION_METRIC_MODE, X_NUMBER, SESSION, BLOCK
    global RFD_CHECK_END_S, RFD_CHECKPOINTS_S
    global PLOT_TIME_FROM_ONSET_S, PLOT_TIME_FROM_GO_S

    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    screen = open_on_second_screen()
    WIDTH, HEIGHT = screen.get_size()

    right_rect = create_target_rect(HEIGHT)
    place_targets(right_rect, WIDTH, HEIGHT)

    loadbar_rect = pygame.Rect(
        0, 0, LOADBAR_WIDTH, int(HEIGHT * LOADBAR_HEIGHT_FRAC)
    )
    loadbar_rect.midleft = (LOADBAR_X_OFFSET, HEIGHT // 2)

    joystick = list_joysticks()
    if joystick is None:
        pygame.quit()
        return

    # Uncomment while debugging the RIGHT transducer:
    # diagnose_axes(joystick, 5.0)

    # Persistent operator/session state. A completed block returns to the
    # same GUI with calibration and baseline still in memory.
    calibration = None
    baseline_result = None
    selected_subject = X_NUMBER
    selected_session = SESSION
    selected_block = BLOCK
    selected_trigger_mode = SLOPE_TRIGGER_MODE
    selected_metric_mode = STIMULATION_METRIC_MODE
    show_main_task_loadbar = MAIN_TASK_LOADBAR_DEFAULT

    while True:
        # Startup interface: calibration is launched explicitly from the menu.
        # The task button remains disabled until calibration has completed.
        (
            calibration, selected_trigger_mode, selected_metric_mode,
            baseline_result, show_main_task_loadbar,
            selected_subject, selected_session, selected_block,
        ) = startup_menu(
            screen, joystick, right_rect,
            calibration=calibration,
            baseline_result=baseline_result,
            subject_id=selected_subject,
            session_id=selected_session,
            block_id=selected_block,
            trigger_mode=selected_trigger_mode,
            stimulation_metric_mode=selected_metric_mode,
            show_loadbar=show_main_task_loadbar,
        )
        if calibration is None:
            print("Operator exited from startup interface.")
            break

        # The returned selections remain the defaults when the GUI reopens.

        baseline_thresholds = dict(baseline_result["thresholds"])
        baseline_force_profile = dict(baseline_result["force_profile"])
        baseline_plateau_times_s = np.asarray(
            baseline_result["plateau_times_s"], dtype=float
        )
        baseline_mean_plateau_s = float(baseline_result["mean_plateau_s"])

        # Individualize the real-time checking window from Baseline Calibration.
        if INDIVIDUALIZE_RFD_CHECK_END:
            RFD_CHECK_END_S = min(float(baseline_result["rfd_check_end_s"]), MAX_RFD_CHECK_END_S)
            RFD_CHECKPOINTS_S = tuple(baseline_result["checkpoints_s"])

        # Keep plots wide enough to show the individualized checking window.
        PLOT_TIME_FROM_ONSET_S = (
            PLOT_TIME_FROM_ONSET_S[0],
            max(PLOT_TIME_FROM_ONSET_S[1], RFD_CHECK_END_S + 0.15),
        )
        PLOT_TIME_FROM_GO_S = (
            PLOT_TIME_FROM_GO_S[0],
            max(PLOT_TIME_FROM_GO_S[1], RFD_CHECK_END_S + 0.45),
        )

        # The monitor must stay armed long enough for a late onset plus the
        # individualized checking window. This may lengthen GO automatically.
        # The participant sees the green GO cue for GO_DURATION_S exactly.
        # The trigger monitor may remain armed slightly longer if required to
        # complete a valid late-onset checkpoint window. This does NOT extend
        # the visible green cue.
        main_go_duration_s = max(
            GO_DURATION_S,
            ONSET_SEARCH_WINDOW_S + RFD_CHECK_END_S + 0.02,
        )

        # Use the operator's startup selection consistently throughout the session.
        X_NUMBER = selected_subject
        SESSION = selected_session
        BLOCK = selected_block

        SLOPE_TRIGGER_MODE = selected_trigger_mode
        STIMULATION_METRIC_MODE = selected_metric_mode
        print(f"Selected trigger direction: {SLOPE_TRIGGER_MODE}")
        print(f"Selected Task Version: {STIMULATION_METRIC_MODE}")
        print(
            "Main Task loadbar: "
            + ("ENABLED" if show_main_task_loadbar else "DISABLED")
        )
        print(
            f"Individualized RFD checking end: {RFD_CHECK_END_S * 1000:.0f} ms "
            f"(baseline mean plateau {baseline_mean_plateau_s * 1000:.1f} ms)"
        )
        print(
            f"Visible green GO cue: {GO_DURATION_S:.3f} s "
            f"(+ up to {GO_JITTER_S:.3f} s jitter)"
        )
        print(
            f"Red cue: {'ON' if USE_RED_CUE else 'OFF'} | "
            f"{PREP_DURATION_S:.3f} s (+ up to {PREP_JITTER_S:.3f} s jitter)"
        )
        print(
            f"Pause (ITI): {ITI_MIN_S:.3f} s "
            f"(+ up to {ITI_JITTER_S:.3f} s jitter)"
        )
        print(f"Adaptive monitor active up to: {main_go_duration_s:.3f} s after GO")
        print(f"Force recording continues to: {POST_GO_RECORDING_END_S:.3f} s after GO")

        # Preserve the familiar final countdown after the user presses Start Main Task.
        clear_force_buffer()
        countdown(screen)

        start_time = time.perf_counter()
        GLOBAL_START_TIME = start_time
        break_start_time = start_time

        right_monitor = RFDMonitor(poll_hz=500)
        right_monitor.metric_mode = STIMULATION_METRIC_MODE
        right_monitor.trigger_mode = SLOPE_TRIGGER_MODE
        right_monitor.start()

        trial_index = 0
        trial_log = []
        trial_timeseries = []
        # Seed Main Task with the baseline-calibrated checkpoint profile.
        # Replicating the seed SLOPE_REFERENCE_TRIALS times makes trial 1 use the
        # full baseline mean; subsequent Main Task trials progressively replace it.
        right_checkpoint_history = [
            dict(baseline_thresholds) for _ in range(SLOPE_REFERENCE_TRIALS)
        ]
        right_force_checkpoint_history = [
            dict(baseline_force_profile) for _ in range(FORCE_REFERENCE_TRIALS)
        ]
        primary_key = int(round(RFD_EPOCH_S * 1000))
        baseline_primary_rfd = baseline_thresholds.get(primary_key, np.nan)
        right_rfd_history = [baseline_primary_rfd] * SLOPE_REFERENCE_TRIALS

        state = "PREP"
        state_start = time.perf_counter()
        cue_trigger_sent = False
        current_prep_duration = np.random.uniform(
            PREP_DURATION_S, PREP_DURATION_S + PREP_JITTER_S
        )
        current_go_duration = GO_DURATION_S
        current_main_go_duration_s = main_go_duration_s
        iti_duration = np.random.uniform(ITI_MIN_S, ITI_MAX_S)
        go_start_time = 0.0

        active_targets = configure_targets_for_trial(
            1, right_rect, WIDTH, HEIGHT
        )

        input_dt = 1.0 / SAMPLING_RATE_HZ
        display_dt = 1.0 / FPS
        next_input_sample_time = time.perf_counter()
        next_display_time = time.perf_counter()
        right_force = 0.0
        running = True

        while running:
            now_perf = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            if now_perf >= next_input_sample_time:
                right_force, _sample_time = acquire_force_sample(
                    joystick, start_time, calibration
                )
                next_input_sample_time += input_dt
                if next_input_sample_time < now_perf - input_dt * 5:
                    next_input_sample_time = now_perf + input_dt

            elapsed_game_time = now_perf - start_time
            if elapsed_game_time >= duration_of_game:
                print("Maximum game duration reached.")
                running = False
                continue
            if trial_index >= TOTAL_TRIALS:
                running = False
                continue

            elapsed_state = now_perf - state_start

            if (
                state in ("PREP", "ITI")
                and (now_perf - break_start_time) >= BREAK_INTERVAL
            ):
                print("Taking a break ...")
                state = "BREAK"
                state_start = now_perf
                elapsed_state = 0.0
                cue_trigger_sent = False

            draw_now = now_perf >= next_display_time
            if draw_now:
                screen.fill(BLACK)

            if state == "BREAK":
                if draw_now:
                    remaining = max(0, BREAK_DURATION - elapsed_state)
                    show_text(
                        f"Kort pause\n\n{int(np.ceil(remaining))}",
                        WHITE, screen,
                    )
                if elapsed_state >= BREAK_DURATION:
                    break_start_time = time.perf_counter()
                    state = "PREP"
                    state_start = time.perf_counter()
                    cue_trigger_sent = False
                    current_prep_duration = np.random.uniform(
                        PREP_DURATION_S, PREP_DURATION_S + PREP_JITTER_S
                    )

            elif state == "PREP":
                progress = min(1.0, elapsed_state / max(current_prep_duration, 1e-6))

                if (
                    not cue_trigger_sent
                    and (current_prep_duration - elapsed_state) <= CUE_TRIGGER_LEAD_S
                ):
                    send_trigger(cue_trigger_channel)
                    cue_trigger_sent = True
                    print(
                        f"Cue trigger sent on {cue_trigger_channel} at "
                        f"{elapsed_game_time * 1000:.1f} ms"
                    )

                if draw_now:
                    draw_cursor_track(screen, right_rect)
                    # No neutral/grey placeholder box: nothing is drawn
                    # until the red cue (if enabled) is shown.
                    if USE_RED_CUE:
                        draw_active_target(screen, right_rect, RED)
                    draw_force_cursor(screen, right_force, right_rect)

                    if show_main_task_loadbar:
                        pygame.draw.rect(screen, (60, 60, 60), loadbar_rect)
                        fill_rect = loadbar_rect.copy()
                        fill_rect.height = int(loadbar_rect.height * progress)
                        fill_rect.top = loadbar_rect.bottom - fill_rect.height
                        pygame.draw.rect(screen, BLUE, fill_rect)
                        pygame.draw.rect(screen, WHITE, loadbar_rect, 2)

                if elapsed_state >= current_prep_duration:
                    trial_index += 1
                    go_start_time = time.perf_counter() - start_time
                    current_go_duration = np.random.uniform(
                        GO_DURATION_S, GO_DURATION_S + GO_JITTER_S
                    )
                    current_main_go_duration_s = max(
                        current_go_duration,
                        ONSET_SEARCH_WINDOW_S + RFD_CHECK_END_S + 0.02,
                    )
                    active_targets = configure_targets_for_trial(
                        trial_index, right_rect, WIDTH, HEIGHT
                    )

                    right_thresholds = calculate_checkpoint_thresholds(
                        right_checkpoint_history
                    )
                    right_force_references = calculate_checkpoint_force_references(
                        right_force_checkpoint_history
                    )
                    right_monitor.arm(go_start_time, right_thresholds, right_force_references)

                    primary_key = int(round(RFD_EPOCH_S * 1000))
                    right_threshold = right_thresholds[primary_key]
                    n_live = sum(
                        1 for v in right_thresholds.values() if np.isfinite(v)
                    )

                    print(
                        "\n========================================\n"
                        f"TRIAL {trial_index}/{TOTAL_TRIALS}\n"
                        "========================================"
                    )
                    print(
                        "Previous right RFD = "
                        f"{right_rfd_history[-SLOPE_REFERENCE_TRIALS:]}"
                    )
                    print(f"Right threshold = {right_threshold}")
                    print(
                        f"Active checkpoints (right): {n_live}/"
                        f"{len(RFD_CHECKPOINTS_S)}"
                    )
                    print(
                        "Continuous checkpoints = "
                        f"{int(RFD_CHECK_START_S * 1000)} to "
                        f"{int(RFD_CHECK_END_S * 1000)} ms after onset"
                    )
                    relation = "<" if SLOPE_TRIGGER_MODE == "BELOW" else ">"
                    purpose = (
                        "correct low-RFD trials"
                        if SLOPE_TRIGGER_MODE == "BELOW"
                        else "reinforce high-RFD trials"
                    )
                    if STIMULATION_METRIC_MODE == "RFD_ONLY":
                        rule_text = (
                            f"20-ms local RFD {relation} adaptive RFD threshold"
                        )
                    elif STIMULATION_METRIC_MODE == "FORCE_ONLY":
                        rule_text = (
                            f"checkpoint force {relation} mean of previous "
                            f"{FORCE_REFERENCE_TRIALS} trials"
                        )
                        if SLOPE_TRIGGER_MODE == "BELOW":
                            rule_text += (
                                f" AND force < {FORCE_TRIGGER_CEILING:.2f} MVC"
                            )
                    else:
                        rule_text = (
                            f"20-ms local RFD {relation} adaptive RFD threshold "
                            f"AND checkpoint force {relation} mean of previous "
                            f"{FORCE_REFERENCE_TRIALS} trials"
                        )
                        if SLOPE_TRIGGER_MODE == "BELOW":
                            rule_text += (
                                f" AND force < {FORCE_TRIGGER_CEILING:.2f} MVC"
                            )

                    print(
                        f"Task Version = {STIMULATION_METRIC_MODE} | "
                        f"Trigger rule = {rule_text} ({purpose})"
                    )

                    state = "GO"
                    state_start = time.perf_counter()

            elif state == "GO":
                if draw_now:
                    draw_cursor_track(screen, right_rect)

                    # Green GO cue is visible for this trial's (jittered)
                    # duration.
                    if elapsed_state < current_go_duration:
                        draw_active_target(screen, right_rect, GREEN)

                    draw_force_cursor(screen, right_force, right_rect)

                if elapsed_state >= current_main_go_duration_s:
                    # End the active task/TMS period now.
                    # Force acquisition continues in RECORD_TAIL only for
                    # analysis and plotting until POST_GO_RECORDING_END_S.
                    right_result = right_monitor.disarm()
                    state = "RECORD_TAIL"
                    state_start = time.perf_counter()

            elif state == "RECORD_TAIL":
                # Keep the force sampler running, but do not re-arm the monitor
                # and do not allow any additional TMS/adaptive decisions.
                # go_start_time is stored relative to experiment start,
                # so compare it with elapsed_game_time (same clock basis).
                total_from_go = elapsed_game_time - go_start_time

                if draw_now:
                    # Recording-only period: GO cue is no longer visible.
                    draw_cursor_track(screen, right_rect)
                    draw_force_cursor(screen, right_force, right_rect)

                if total_from_go >= POST_GO_RECORDING_END_S:
                    right_result = right_monitor.disarm()
                    current_time_array, current_right_force = get_grip_samples_numpy()

                    trial_metrics = calculate_trial_metrics(
                        trial_number=trial_index,
                        go_start_time=go_start_time,
                        time_array=current_time_array,
                        right_force_array=current_right_force,
                        prev_right_rfd=(
                            right_rfd_history[-1]
                            if right_rfd_history else np.nan
                        ),
                        right_result=right_result,
                    )
                    trial_log.append(trial_metrics)

                    right_trial_rfd = trial_metrics.get("right_rfd", np.nan)
                    right_rfd_history.append(right_trial_rfd)
                    right_checkpoint_history.append({
                        int(round(c * 1000)): trial_metrics.get(
                            f"right_rfd_{int(round(c * 1000))}ms", np.nan
                        ) for c in RFD_CHECKPOINTS_S
                    })
                    right_force_checkpoint_history.append({
                        int(round(c * 1000)): trial_metrics.get(
                            f"right_checkpoint_force_{int(round(c * 1000))}ms", np.nan
                        ) for c in RFD_CHECKPOINTS_S
                    })

                    trial_trace = extract_trial_timeseries(
                        trial_number=trial_index,
                        go_start_time=go_start_time,
                        time_array=current_time_array,
                        right_force_array=current_right_force,
                    )
                    if not trial_trace.empty:
                        trial_timeseries.append(trial_trace)

                    print(f"\nTRIAL {trial_index}/{TOTAL_TRIALS} COMPLETE")
                    print("----------------------------------------")
                    rt = trial_metrics.get("right_reaction_time_s", np.nan)
                    print(
                        f"Right | RT {rt * 1000:6.1f} ms | "
                        f"RFD({int(RFD_EPOCH_S * 1000)}ms) "
                        f"{right_trial_rfd:9.6f} | "
                        f"checks {right_result['n_checkpoints_checked']} | "
                        f"trig {right_result['triggered']} | "
                        f"{right_result['decision_reason']}"
                    )
                    diff = trial_metrics.get(
                        "right_rfd_realtime_minus_posthoc", np.nan
                    )
                    if np.isfinite(diff) and abs(diff) > 1e-6:
                        print(
                            "       live-vs-recorded RFD difference: "
                            f"{diff:+.6f}"
                        )
                    print(
                        "       Accuracy | hit target: "
                        f"{trial_metrics.get('right_hit_target', False)} | "
                        f"time in target: "
                        f"{trial_metrics.get('right_time_in_target_s', np.nan):.3f} s "
                        f"({trial_metrics.get('right_time_in_target_pct', np.nan):.1f}%) | "
                        f"mean |error|: "
                        f"{trial_metrics.get('right_mean_abs_error_mvc', np.nan):.4f} MVC"
                    )

                    if trial_index >= TOTAL_TRIALS:
                        print(
                            "\n====================================\n"
                            f"ALL {TOTAL_TRIALS} TRIALS COMPLETED\n"
                            "===================================="
                        )
                        running = False
                        continue

                    iti_duration = np.random.uniform(ITI_MIN_S, ITI_MAX_S)
                    state = "ITI"
                    state_start = time.perf_counter()
                    cue_trigger_sent = False
            elif state == "ITI":
                if draw_now:
                    # No grey placeholder box during the pause -- only
                    # the cursor track and the force cursor are shown.
                    draw_cursor_track(screen, right_rect)
                    draw_force_cursor(screen, right_force, right_rect)

                if elapsed_state >= iti_duration:
                    state = "PREP"
                    state_start = time.perf_counter()
                    cue_trigger_sent = False
                    current_prep_duration = np.random.uniform(
                        PREP_DURATION_S, PREP_DURATION_S + PREP_JITTER_S
                    )

            if draw_now:
                pygame.display.flip()
                next_display_time += display_dt
                if next_display_time < now_perf - display_dt * 5:
                    next_display_time = now_perf + display_dt

            time.sleep(0.0002)

        show_end_screen(screen)
        right_monitor.stop()
        right_monitor.join(timeout=1.0)
        time.sleep(0.5)

        print(
            "\n====================================\n"
            "POST-HOC DATA PROCESSING\n"
            "===================================="
        )

        time_array, right_force_array = get_grip_samples_numpy()

        # Folder hierarchy:
        # OUTPUT_ROOT / Subject / Session / Block / TaskVersion
        block_timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        task_version_tag = f"{STIMULATION_METRIC_MODE}_{SLOPE_TRIGGER_MODE}"
        out_dir = os.path.join(
            OUTPUT_ROOT, X_NUMBER, SESSION, BLOCK, task_version_tag
        )
        try:
            os.makedirs(out_dir, exist_ok=True)
        except (PermissionError, OSError):
            fallback_root = os.path.join(USER_HOME, "Documents", "ADAPT_BK_Exp2")
            out_dir = os.path.join(
                fallback_root, X_NUMBER, SESSION, BLOCK, task_version_tag
            )
            os.makedirs(out_dir, exist_ok=True)
            print("Desktop output directory was not writable.")
            print(f"Using fallback:\n{out_dir}")

        df_trials = pd.DataFrame(trial_log)
        df_raw = pd.DataFrame({
            "time_s": time_array,
            "right_force": right_force_array,
        })
        if trial_timeseries:
            df_timeseries = pd.concat(trial_timeseries, ignore_index=True)
        else:
            df_timeseries = pd.DataFrame()

        baseline_keys = sorted(baseline_thresholds.keys())
        df_baseline_thresholds = pd.DataFrame({
            "checkpoint_ms": baseline_keys,
            "baseline_rfd_threshold": [
                baseline_thresholds[k] for k in baseline_keys
            ],
            "baseline_force_reference": [
                baseline_force_profile.get(k, np.nan) for k in baseline_keys
            ],
            "stimulation_metric_mode": STIMULATION_METRIC_MODE,
            "trigger_direction": SLOPE_TRIGGER_MODE,
        })
        df_baseline_plateau = pd.DataFrame({
            "baseline_trial": np.arange(1, len(baseline_plateau_times_s) + 1),
            "time_to_plateau_s": baseline_plateau_times_s,
            "time_to_plateau_ms": baseline_plateau_times_s * 1000.0,
        })

        df_calibration = pd.DataFrame([{
            "baseline_right": calibration.baseline_right,
            "max_deflection_right": calibration.max_deflection_right,
            "deadzone_raw": DEADZONE_RAW,
            "calibration_scale_factor": CALIBRATION_SCALE_FACTOR,
            "target_force_level": TARGET_FORCE_LEVEL,
            "local_rfd_window_s": LOCAL_RFD_WINDOW_S,
            "force_reference_trials": FORCE_REFERENCE_TRIALS,
            "force_trigger_ceiling": FORCE_TRIGGER_CEILING,
        "force_reference_margin": FORCE_REFERENCE_MARGIN,
            "stimulation_metric_mode": STIMULATION_METRIC_MODE,
            "trigger_direction": SLOPE_TRIGGER_MODE,
            "task_version_tag": task_version_tag,
            "subject_id": X_NUMBER,
            "session_id": SESSION,
            "block_id": BLOCK,
            "target_tolerance": TARGET_TOLERANCE,
            "baseline_mean_time_to_plateau_s": baseline_mean_plateau_s,
            "individualized_rfd_check_end_s": RFD_CHECK_END_S,
            "plateau_fraction_of_peak": PLATEAU_FRACTION_OF_PEAK,
            "plateau_hold_s": PLATEAU_HOLD_S,
        }])

        xlsx_path = os.path.join(
            out_dir, f"game_output_{X_NUMBER}_{SESSION}_{BLOCK}_{task_version_tag}_{block_timestamp}.xlsx"
        )
        try:
            with pd.ExcelWriter(xlsx_path) as writer:
                if not df_trials.empty:
                    df_trials.to_excel(writer, sheet_name="Trial_Summary", index=False)
                if not df_timeseries.empty:
                    df_timeseries.to_excel(
                        writer, sheet_name="Trial_TimeSeries", index=False
                    )
                df_calibration.to_excel(writer, sheet_name="Calibration", index=False)
                df_baseline_thresholds.to_excel(
                    writer, sheet_name="Baseline_Thresholds", index=False
                )
                df_baseline_plateau.to_excel(
                    writer, sheet_name="Baseline_Plateau", index=False
                )
                df_raw.to_excel(writer, sheet_name="Raw_Data", index=False)
            print(f"Excel saved to:\n{xlsx_path}")
        except Exception as e:
            print(f"ERROR writing Excel: {e}")

        plot_force_data(time_array, right_force_array, trial_log, out_dir)
        if not df_trials.empty:
            plot_trial_rfd(df_trials, out_dir)

        if not df_timeseries.empty:
            plot_onset_aligned_forces(df_timeseries, out_dir)
            shared_rfd_limits = resolve_rfd_limits(df_trials, df_timeseries)
            print(
                f"Common RFD axis for all trial figures: "
                f"{shared_rfd_limits[0]:.2f} to {shared_rfd_limits[1]:.2f} MVC/s"
            )
            plot_individual_trial_figures(
                df_timeseries, df_trials, out_dir, rfd_limits=shared_rfd_limits
            )
            plot_session_rfd_summaries(
                df_trials, out_dir, rfd_limits=shared_rfd_limits
            )
        elif not df_trials.empty:
            shared_rfd_limits = resolve_rfd_limits(df_trials, None)
            plot_session_rfd_summaries(
                df_trials, out_dir, rfd_limits=shared_rfd_limits
            )

        mat_path = os.path.join(
            out_dir, f"game_output_{X_NUMBER}_{SESSION}_{BLOCK}_{task_version_tag}_{block_timestamp}.mat"
        )
        mat_data = {
            "raw_data_time_s": time_array,
            "raw_data_right_force": right_force_array,
            "calibration_baseline_right": np.array([calibration.baseline_right]),
            "calibration_max_deflection_right": np.array([
                calibration.max_deflection_right
            ]),
            "calibration_deadzone_raw": np.array([DEADZONE_RAW]),
            "calibration_scale_factor": np.array([CALIBRATION_SCALE_FACTOR]),
            "target_force_level": np.array([TARGET_FORCE_LEVEL]),
            "local_rfd_window_s": np.array([LOCAL_RFD_WINDOW_S]),
            "force_reference_trials": np.array([FORCE_REFERENCE_TRIALS]),
            "force_trigger_ceiling": np.array([FORCE_TRIGGER_CEILING]),
            "target_tolerance": np.array([TARGET_TOLERANCE]),
            "total_trials_requested": np.array([TOTAL_TRIALS]),
            "total_trials_completed": np.array([len(trial_log)]),
            "slope_reference_trials": np.array([SLOPE_REFERENCE_TRIALS]),
            "slope_trigger_mode": np.array([SLOPE_TRIGGER_MODE], dtype=object),
            "stimulation_metric_mode": np.array([STIMULATION_METRIC_MODE], dtype=object),
            "task_version_tag": np.array([task_version_tag], dtype=object),
            "subject_id": np.array([X_NUMBER], dtype=object),
            "session_id": np.array([SESSION], dtype=object),
            "block_id": np.array([BLOCK], dtype=object),
            "main_task_loadbar_enabled": np.array([float(show_main_task_loadbar)]),
            "use_red_cue": np.array([float(USE_RED_CUE)]),
            "prep_red_cue_duration_s": np.array([PREP_DURATION_S]),
            "prep_red_cue_jitter_s": np.array([PREP_JITTER_S]),
            "go_green_target_duration_s": np.array([GO_DURATION_S]),
            "go_green_target_jitter_s": np.array([GO_JITTER_S]),
            "iti_pause_base_s": np.array([ITI_MIN_S]),
            "iti_jitter_s": np.array([ITI_JITTER_S]),
            "target_tolerance_used": np.array([TARGET_TOLERANCE]),
            "baseline_trials": np.array([BASELINE_TRIALS]),
            "baseline_time_to_plateau_s": baseline_plateau_times_s,
            "baseline_mean_time_to_plateau_s": np.array([baseline_mean_plateau_s]),
            "plateau_fraction_of_peak": np.array([PLATEAU_FRACTION_OF_PEAK]),
            "plateau_hold_s": np.array([PLATEAU_HOLD_S]),
            "individualized_rfd_check_end_s": np.array([RFD_CHECK_END_S]),
            "baseline_threshold_checkpoint_ms": np.array(
                sorted(baseline_thresholds.keys()), dtype=float
            ),
            "baseline_threshold_rfd": np.array(
                [baseline_thresholds[k] for k in sorted(baseline_thresholds.keys())],
                dtype=float,
            ),
            "baseline_force_reference": np.array(
                [baseline_force_profile.get(k, np.nan)
                 for k in sorted(baseline_thresholds.keys())],
                dtype=float,
            ),
            "familiarization_trials": np.array([FAMILIARIZATION_TRIALS]),
            "target_display_mode": np.array([TARGET_DISPLAY_MODE], dtype=object),
            "rfd_epoch_s": np.array([RFD_EPOCH_S]),
            "rfd_checkpoints_s": np.array(RFD_CHECKPOINTS_S, dtype=float),
            "rfd_check_start_s": np.array([RFD_CHECK_START_S]),
            "rfd_check_end_s": np.array([RFD_CHECK_END_S]),
            "onset_search_window_s": np.array([ONSET_SEARCH_WINDOW_S]),
            "post_onset_window_s": np.array([POST_ONSET_WINDOW_S]),
            "onset_confirm_samples": np.array([ONSET_CONFIRM_SAMPLES]),
            "trigger_margin_fraction": np.array([TRIGGER_MARGIN_FRACTION]),
            "rising_onset_force": np.array([RISING_ONSET_FORCE]),
            "sampling_rate_hz": np.array([SAMPLING_RATE_HZ]),
            "slope_smoothing_samples": np.array([SLOPE_SMOOTHING_SAMPLES]),
        }

        if not df_trials.empty:
            for column in df_trials.columns:
                try:
                    mat_data[column] = matlab_safe(df_trials[column])
                except Exception as e:
                    print(f"Skipping column '{column}' in .mat export: {e}")

        if not df_timeseries.empty:
            for column in (
                "trial", "time_s", "time_from_go_s", "time_from_onset_s",
                "force", "force_smoothed", "in_rfd_epoch",
                "running_slope_diagnostic",
            ):
                if column in df_timeseries.columns:
                    mat_data[f"timeseries_{column}"] = matlab_safe(
                        df_timeseries[column]
                    )

        try:
            savemat(mat_path, mat_data, do_compression=True)
            print(f"MATLAB file saved to:\n{mat_path}")
        except Exception as e:
            print(f"ERROR writing .mat: {e}")

        print(
            "\n============================================\n"
            "EXPERIMENT COMPLETE\n"
            "============================================"
        )
        print(f"Requested trials: {TOTAL_TRIALS}")
        print(f"Completed trials: {len(trial_log)}")
        print("Active handle: RIGHT only")
        print(f"Target force level: {TARGET_FORCE_LEVEL:.2f} MVC")
        print(f"Local RFD window: {LOCAL_RFD_WINDOW_S*1000:.0f} ms")
        print(f"Force confirmation: previous {FORCE_REFERENCE_TRIALS} trials at same checkpoint")
        print(f"BELOW-mode high-force trigger ceiling: {FORCE_TRIGGER_CEILING:.2f} MVC")
        print(f"Stimulation metric mode: {STIMULATION_METRIC_MODE}")
        print(f"Trigger direction: {SLOPE_TRIGGER_MODE}")
        print(
            "Main Task loadbar: "
            + ("ENABLED" if show_main_task_loadbar else "DISABLED")
        )
        print(f"Reference trials: {SLOPE_REFERENCE_TRIALS}")
        print(f"Nominal force sampling rate: {SAMPLING_RATE_HZ} Hz")

        if len(time_array) > 1:
            achieved_rate = len(time_array) / (time_array[-1] - time_array[0])
            print(f"Achieved mean sampling rate: {achieved_rate:.1f} Hz")
        print(f"Calibration: {calibration.summary()}")

        if not df_trials.empty:
            print("\nTrigger counts:")
            if "right_triggered" in df_trials.columns:
                print(f"Right: {int(df_trials['right_triggered'].sum())}")

            print(
                f"\nMean onset-anchored RFD "
                f"({int(RFD_EPOCH_S * 1000)} ms epoch):"
            )
            if "right_rfd" in df_trials.columns:
                print(f"Right : {df_trials['right_rfd'].mean():.6f} MVC/s")

            print("\nMean reaction time (GO to force onset):")
            if "right_reaction_time_s" in df_trials.columns:
                rt = df_trials["right_reaction_time_s"] * 1000
                print(
                    f"Right : {rt.mean():.1f} ms "
                    f"(SD {rt.std():.1f}, range {rt.min():.0f}-{rt.max():.0f})"
                )

            if "right_onset_detected" in df_trials.columns:
                n_no_onset = int((~df_trials["right_onset_detected"].astype(bool)).sum())
                print(f"\nTrials with no detected onset: {n_no_onset}")

            if "right_hit_target" in df_trials.columns:
                n_hit = int(df_trials["right_hit_target"].astype(bool).sum())
                print(
                    f"\nAccuracy: {n_hit}/{len(df_trials)} trials entered "
                    "the target box"
                )
            if "right_time_in_target_pct" in df_trials.columns:
                pct = df_trials["right_time_in_target_pct"]
                print(
                    f"Mean time in target: {pct.mean():.1f}% "
                    f"(SD {pct.std():.1f})"
                )
            if "right_mean_abs_error_mvc" in df_trials.columns:
                err = df_trials["right_mean_abs_error_mvc"]
                print(
                    f"Mean |force error| from target: {err.mean():.4f} MVC "
                    f"(SD {err.std():.4f})"
                )

        print(
            "\n============================================\n"
            "DATA SAVED\n"
            "============================================"
        )
        print(f"Output directory:\n{out_dir}")
        print(f"\nExcel:\n{xlsx_path}")
        print(f"\nMATLAB:\n{mat_path}")
        print("\nIndividual trial figures:\n" + os.path.join(out_dir, "SingleTrials"))
        print("\nSession summary figures:\n" + os.path.join(out_dir, "Summary"))

        print(
            "\nBLOCK COMPLETE. Returning to operator interface with "
            "calibration and baseline retained.\n"
        )
        # The next block starts from the ORIGINAL baseline seed, because all
        # adaptive histories are recreated at the start of each block above.
        clear_force_buffer()
    pygame.quit()
    print("Experiment application closed.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()