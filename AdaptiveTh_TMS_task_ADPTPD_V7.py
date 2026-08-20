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

# RFD triggering algorithm selectable from the operator GUI.
# SLIDING_CHECKPOINTS = existing repeated local-window checks.
# FIXED_EARLY_SLOPE = one onset-anchored regression, e.g. 0-40 ms, giving
# one comparable slope and one consistent decision time on every trial.
RFD_TRIGGER_ALGORITHM = "SLIDING_CHECKPOINTS"
VALID_RFD_TRIGGER_ALGORITHMS = ("SLIDING_CHECKPOINTS", "FIXED_EARLY_SLOPE")
FIXED_SLOPE_WINDOW_S = 0.040
# Adaptive fixed-slope timing. The operator chooses how much of the predicted
# onset-to-ballistic-peak rise is used for the slope fit. Stimulation, when
# criteria are met, is delivered immediately AFTER that fit window with the
# short fixed hardware/experimental delay below.
#
#   FIT_1Q -> slope 0..1/4 peak latency, TMS at fit end + 5 ms
#   FIT_2Q -> slope 0..2/4 peak latency, TMS at fit end + 5 ms
#   FIT_3Q -> slope 0..3/4 peak latency, TMS at fit end + 5 ms
FIXED_TIMING_MODE = "FIT_3Q"
VALID_FIXED_TIMING_MODES = (
    "FIT_1Q",
    "FIT_2Q",
    "FIT_3Q",
)
FIXED_TIMING_FRACTIONS = {
    "FIT_1Q": 0.25,
    "FIT_2Q": 0.50,
    "FIT_3Q": 0.75,
}
FIXED_SLOPE_MIN_WINDOW_S = 0.010
# Must accommodate 3/4 of the maximum 350-ms ballistic peak-search interval.
FIXED_SLOPE_MAX_WINDOW_S = 0.300
# Stimulation is scheduled this long AFTER the adaptive slope-fit endpoint.
FIXED_TRIGGER_DELAY_S = 0.005

# Baseline review options. AUTO_SAVE writes QC plots/metrics and includes all
# valid trials. INTERACTIVE shows each trial and lets the operator reject it.
BASELINE_REVIEW_MODE = "AUTO_SAVE"
VALID_BASELINE_REVIEW_MODES = ("AUTO_SAVE", "INTERACTIVE")

# Optional participant feedback during the Main Task.
PARTICIPANT_FEEDBACK_ENABLED = False
PARTICIPANT_FEEDBACK_METRIC = "PEAK_FORCE"
VALID_FEEDBACK_METRICS = ("PEAK_FORCE", "ADAPTIVE_RFD", "RFD_80MS", "REACTION_TIME", "TARGET_ERROR")
FEEDBACK_HISTORY_POINTS = 30

# Ballistic peak detection for baseline endpoint calibration.
BALLISTIC_PEAK_SEARCH_S = 0.350
BALLISTIC_DIRECTION_CONFIRM_S = 0.008
BALLISTIC_MIN_PEAK_FORCE = 0.15

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
PLOT_RFD_LIMITS = (-25.0, 25.0)

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


def calculate_sustained_plateau_time(t, force):
    """Legacy sustained-plateau estimate retained for QC/reporting only."""
    t = np.asarray(t, dtype=float); force = np.asarray(force, dtype=float)
    if t.size < ONSET_CONFIRM_SAMPLES or force.size != t.size:
        return np.nan
    smoothed = smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    onset_idx, smoothed = find_rising_onset(t, force, smoothed=smoothed)
    if onset_idx is None: return np.nan
    onset_time = float(t[onset_idx])
    valid_idx = np.where((t >= onset_time) & (t <= onset_time + POST_ONSET_WINDOW_S))[0]
    if valid_idx.size < ONSET_CONFIRM_SAMPLES: return np.nan
    post_force = smoothed[valid_idx]
    peak_force = float(np.nanmax(post_force))
    if not np.isfinite(peak_force) or peak_force <= RISING_ONSET_FORCE: return np.nan
    level = PLATEAU_FRACTION_OF_PEAK * peak_force
    hold_samples = max(1, int(np.ceil(PLATEAU_HOLD_S * SAMPLING_RATE_HZ)))
    run=0
    for local_i, ok in enumerate(post_force >= level):
        run = run + 1 if ok else 0
        if run >= hold_samples:
            idx=valid_idx[local_i-hold_samples+1]
            return float(t[idx]-onset_time)
    return np.nan

def calculate_time_to_force_plateau(t, force):
    """
    Ballistic onset-to-peak time used to individualize the RFD checking window.

    For this task the participant releases after a rapid squeeze, so a sustained
    plateau is not expected. The endpoint is therefore the first robust local
    maximum after onset: force is rising beforehand and the smoothed force-rate
    changes from positive to non-positive for a short confirmation period.
    """
    t=np.asarray(t,dtype=float); force=np.asarray(force,dtype=float)
    if t.size < ONSET_CONFIRM_SAMPLES or force.size != t.size: return np.nan
    smoothed=smooth_signal(force, SLOPE_SMOOTHING_SAMPLES)
    onset_idx, smoothed=find_rising_onset(t, force, smoothed=smoothed)
    if onset_idx is None: return np.nan
    onset_time=float(t[onset_idx])
    end_time=onset_time + min(BALLISTIC_PEAK_SEARCH_S, POST_ONSET_WINDOW_S)
    idxs=np.where((t>=onset_time)&(t<=end_time))[0]
    if idxs.size < max(MIN_EPOCH_SAMPLES, 4): return np.nan
    tf=t[idxs]; ff=smoothed[idxs]
    if float(np.nanmax(ff)) < BALLISTIC_MIN_PEAK_FORCE: return np.nan
    # derivative on the smoothed force; smooth it once more to suppress one-sample sign flips
    rate=np.gradient(ff, tf)
    rate=smooth_signal(rate, max(2, SLOPE_SMOOTHING_SAMPLES))
    confirm=max(1,int(np.ceil(BALLISTIC_DIRECTION_CONFIRM_S*SAMPLING_RATE_HZ)))
    peak_global=float(np.nanmax(ff))
    for j in range(1, len(ff)-confirm):
        if ff[j] < 0.60*peak_global: continue
        before=rate[max(0,j-confirm):j]
        after=rate[j:j+confirm]
        if before.size and after.size and np.nanmean(before)>0 and np.nanmean(after)<=0:
            return float(tf[j]-onset_time)
    # Robust fallback: global maximum within the ballistic search window.
    j=int(np.nanargmax(ff))
    return float(tf[j]-onset_time)

def calculate_fixed_early_slope(t, force, window_s=None, onset_idx=None, smoothed=None):
    """Fit one slope from force onset to a fixed early-ramp endpoint."""
    if window_s is None: window_s=FIXED_SLOPE_WINDOW_S
    t=np.asarray(t,dtype=float); force=np.asarray(force,dtype=float)
    if onset_idx is None:
        onset_idx, smoothed=find_rising_onset(t, force, smoothed)
    elif smoothed is None:
        smoothed=smooth_signal(force,SLOPE_SMOOTHING_SAMPLES)
    if onset_idx is None: return np.nan
    onset=float(t[onset_idx]); end=onset+float(window_s)
    if t.size==0 or t[-1] < end: return np.nan
    mask=(t>=onset)&(t<=end)
    if int(mask.sum()) < MIN_EPOCH_SAMPLES: return np.nan
    ts=t[mask]; fs=smoothed[mask]
    if np.ptp(ts)<=0: return np.nan
    slope,_=np.polyfit(ts-np.mean(ts),fs,1)
    return float(slope)

def _fixed_fit_fraction(mode=None):
    """Return the selected fraction of predicted onset-to-peak used for fitting."""
    mode = FIXED_TIMING_MODE if mode is None else str(mode)
    return float(FIXED_TIMING_FRACTIONS.get(mode, FIXED_TIMING_FRACTIONS["FIT_3Q"]))


def _recent_predicted_peak_latency(peak_latency_history, fallback_peak_s=None):
    """Rolling mean onset-to-peak latency from the most recent valid trials."""
    vals = np.asarray(peak_latency_history, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        vals = vals[-SLOPE_REFERENCE_TRIALS:]
        return float(np.mean(vals))
    if fallback_peak_s is not None and np.isfinite(fallback_peak_s) and fallback_peak_s > 0:
        return float(fallback_peak_s)
    fit_fraction = _fixed_fit_fraction()
    return float(FIXED_SLOPE_WINDOW_S / max(fit_fraction, 1e-6))


def adaptive_fixed_timing(peak_latency_history, fallback_peak_s=None, mode=None):
    """Return adaptive fit window and intended stimulation time for next trial.

    The slope window is a selected fraction of the rolling predicted
    onset-to-ballistic-peak latency. TMS is scheduled FIXED_TRIGGER_DELAY_S
    after the fit endpoint, not at a later quarter boundary.
    """
    predicted_peak_s = _recent_predicted_peak_latency(
        peak_latency_history, fallback_peak_s=fallback_peak_s
    )
    fit_fraction = _fixed_fit_fraction(mode)
    window_s = float(np.clip(
        predicted_peak_s * fit_fraction,
        FIXED_SLOPE_MIN_WINDOW_S,
        FIXED_SLOPE_MAX_WINDOW_S,
    ))
    trigger_target_s = float(window_s + FIXED_TRIGGER_DELAY_S)
    return window_s, trigger_target_s, predicted_peak_s


def adaptive_fixed_slope_window(peak_latency_history, fallback_s=None):
    """Backward-compatible wrapper returning only the adaptive fit window."""
    fallback_peak_s = None
    if fallback_s is not None:
        fit_fraction = _fixed_fit_fraction()
        fallback_peak_s = float(fallback_s) / max(fit_fraction, 1e-6)
    window_s, _, _ = adaptive_fixed_timing(
        peak_latency_history, fallback_peak_s=fallback_peak_s
    )
    return window_s


def fixed_slope_from_reference_trials(reference_trials, window_s, n_trials=None):
    """Mean of the most recent VALID reference slopes over one common window."""
    if n_trials is None:
        n_trials=SLOPE_REFERENCE_TRIALS
    slopes=[]
    for rec in reversed(list(reference_trials)):
        t=np.asarray(rec.get('t',[]),dtype=float); f=np.asarray(rec.get('force',[]),dtype=float)
        if t.size < ONSET_CONFIRM_SAMPLES or f.size != t.size:
            continue
        v=calculate_fixed_early_slope(t,f,window_s=window_s)
        if np.isfinite(v):
            slopes.append(float(v))
        if len(slopes) >= int(n_trials):
            break
    if len(slopes) < int(n_trials):
        return np.nan
    return float(np.mean(slopes))

def fixed_force_from_reference_trials(reference_trials, window_s, n_trials=None):
    """Mean force at the common endpoint from the most recent valid trials."""
    if n_trials is None:
        n_trials=FORCE_REFERENCE_TRIALS
    vals=[]
    for rec in reversed(list(reference_trials)):
        t=np.asarray(rec.get('t',[]),dtype=float); f=np.asarray(rec.get('force',[]),dtype=float)
        if t.size < ONSET_CONFIRM_SAMPLES or f.size != t.size:
            continue
        onset_idx,sm=find_rising_onset(t,f)
        if onset_idx is None:
            continue
        target=float(t[onset_idx])+float(window_s)
        if t[-1] < target:
            continue
        v=float(sm[int(np.argmin(np.abs(t-target)))])
        if np.isfinite(v):
            vals.append(v)
        if len(vals) >= int(n_trials):
            break
    if len(vals) < int(n_trials):
        return np.nan
    return float(np.mean(vals))

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


def _baseline_trial_qc(trial):
    t=np.asarray(trial.get("t",[]),dtype=float); f=np.asarray(trial.get("force",[]),dtype=float)
    onset_idx, sm=find_rising_onset(t,f) if t.size else (None,np.empty(0))
    if onset_idx is None:
        return {"onset":np.nan,"peak_force":np.nan,"peak_time_s":np.nan,"fixed_slope":np.nan,"valid":False}
    peak_time=calculate_time_to_force_plateau(t,f)
    fixed=calculate_fixed_early_slope(t,f,onset_idx=onset_idx,smoothed=sm)
    search=(t>=t[onset_idx])&(t<=t[onset_idx]+BALLISTIC_PEAK_SEARCH_S)
    peak=float(np.nanmax(sm[search])) if search.any() else np.nan
    return {"onset":float(t[onset_idx]),"peak_force":peak,"peak_time_s":peak_time,"fixed_slope":fixed,
            "valid":bool(np.isfinite(peak_time) and np.isfinite(fixed))}

def save_baseline_review_report(baseline_trials, report_dir, included=None):
    os.makedirs(report_dir, exist_ok=True)
    if included is None: included=[True]*len(baseline_trials)
    rows=[]
    for i,(trial,keep) in enumerate(zip(baseline_trials,included),1):
        qc=_baseline_trial_qc(trial); rows.append({"trial":i,"included":bool(keep),**qc})
        t=np.asarray(trial.get("t",[])); f=np.asarray(trial.get("force",[]))
        fig,ax=plt.subplots(figsize=(8,4.2))
        if t.size:
            onset_idx,sm=find_rising_onset(t,f)
            x=(t-(t[onset_idx] if onset_idx is not None else t[0]))*1000
            ax.plot(x,sm,label="Smoothed force")
            ax.axvline(0,linestyle="--",label="Onset")
            if np.isfinite(qc["peak_time_s"]): ax.axvline(qc["peak_time_s"]*1000,linestyle=":",label="Ballistic peak")
            ax.axvline(FIXED_SLOPE_WINDOW_S*1000,linestyle="--",label="Fixed-slope endpoint")
        ax.set_xlabel("Time from onset (ms)"); ax.set_ylabel("Force (MVC)"); ax.set_ylim(0.0,1.0)
        ax.set_title(f"Baseline trial {i} | {'INCLUDED' if keep else 'REJECTED'} | peak={qc['peak_force']:.3f} | peak time={qc['peak_time_s']*1000 if np.isfinite(qc['peak_time_s']) else np.nan:.1f} ms | slope={qc['fixed_slope']:.3f}")
        ax.grid(True,alpha=.25); ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(os.path.join(report_dir,f"baseline_trial_{i:02d}.png"),dpi=160); plt.close(fig)
    qcdf=pd.DataFrame(rows)
    qcdf.to_csv(os.path.join(report_dir,"baseline_trial_qc.csv"),index=False)
    used=qcdf[qcdf["included"] & qcdf["valid"]] if len(qcdf) else qcdf
    summary=pd.DataFrame([{
        "n_trials":len(qcdf), "n_included_valid":len(used),
        "mean_peak_force_mvc":used["peak_force"].mean() if len(used) else np.nan,
        "sd_peak_force_mvc":used["peak_force"].std(ddof=1) if len(used)>1 else np.nan,
        "mean_onset_to_peak_ms":used["peak_time_s"].mean()*1000 if len(used) else np.nan,
        "sd_onset_to_peak_ms":used["peak_time_s"].std(ddof=1)*1000 if len(used)>1 else np.nan,
        "mean_fixed_early_slope_mvc_s":used["fixed_slope"].mean() if len(used) else np.nan,
        "sd_fixed_early_slope_mvc_s":used["fixed_slope"].std(ddof=1) if len(used)>1 else np.nan,
        "fixed_slope_window_ms":FIXED_SLOPE_WINDOW_S*1000,
    }])
    summary.to_csv(os.path.join(report_dir,"baseline_summary.csv"),index=False)
    return rows

def review_baseline_trials(screen, baseline_trials, report_dir):
    """Pygame interactive QC: inspect each baseline trace and include/reject it."""
    included=[_baseline_trial_qc(x)["valid"] for x in baseline_trials]
    i=0; clock=pygame.time.Clock(); w,h=screen.get_size()
    plot_rect=pygame.Rect(20,70,int(w*0.72)-30,h-150)
    prev_r=pygame.Rect(int(w*.74),int(h*.25),int(w*.11),48)
    next_r=pygame.Rect(int(w*.87),int(h*.25),int(w*.11),48)
    toggle_r=pygame.Rect(int(w*.74),int(h*.40),int(w*.24),60)
    finish_r=pygame.Rect(int(w*.74),int(h*.70),int(w*.24),60)
    def trial_surface(k):
        trial=baseline_trials[k]; qc=_baseline_trial_qc(trial); t=np.asarray(trial["t"]); f=np.asarray(trial["force"])
        fig,ax=plt.subplots(figsize=(8,4.5))
        if t.size:
            onset_idx,sm=find_rising_onset(t,f)
            origin=t[onset_idx] if onset_idx is not None else t[0]
            ax.plot((t-origin)*1000,sm)
            ax.axvline(0,linestyle="--")
            if np.isfinite(qc["peak_time_s"]): ax.axvline(qc["peak_time_s"]*1000,linestyle=":")
            ax.axvline(FIXED_SLOPE_WINDOW_S*1000,linestyle="--")
        ax.set_xlim(-100,max(350,FIXED_SLOPE_WINDOW_S*1000+100)); ax.set_ylim(0.0,1.0); ax.set_xlabel("ms from force onset"); ax.set_ylabel("Force (MVC)"); ax.grid(True,alpha=.25)
        fig.tight_layout(); canvas=fig.canvas; canvas.draw(); raw=canvas.buffer_rgba(); size=canvas.get_width_height(); surf=pygame.image.frombuffer(raw,size,"RGBA").copy(); plt.close(fig)
        return pygame.transform.smoothscale(surf,plot_rect.size),qc
    surf,qc=trial_surface(i)
    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return None
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q:return None
                if e.key in (pygame.K_LEFT,pygame.K_a): i=max(0,i-1); surf,qc=trial_surface(i)
                if e.key in (pygame.K_RIGHT,pygame.K_d): i=min(len(baseline_trials)-1,i+1); surf,qc=trial_surface(i)
                if e.key==pygame.K_SPACE: included[i]=not included[i]
                if e.key==pygame.K_RETURN: save_baseline_review_report(baseline_trials,report_dir,included); return included
            if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                if prev_r.collidepoint(e.pos): i=max(0,i-1); surf,qc=trial_surface(i)
                elif next_r.collidepoint(e.pos): i=min(len(baseline_trials)-1,i+1); surf,qc=trial_surface(i)
                elif toggle_r.collidepoint(e.pos): included[i]=not included[i]
                elif finish_r.collidepoint(e.pos): save_baseline_review_report(baseline_trials,report_dir,included); return included
        screen.fill(BLACK); screen.blit(surf,plot_rect)
        font=get_font(28); small=get_font(22)
        screen.blit(font.render(f"Baseline QC: trial {i+1}/{len(baseline_trials)}",True,WHITE),(20,20))
        status="INCLUDE" if included[i] else "REJECT"; col=GREEN if included[i] else RED
        pygame.draw.rect(screen,(55,55,55),toggle_r); pygame.draw.rect(screen,col,toggle_r,4)
        screen.blit(font.render(status,True,col),font.render(status,True,col).get_rect(center=toggle_r.center))
        for r,txt in ((prev_r,"Previous"),(next_r,"Next"),(finish_r,"Use selection")):
            pygame.draw.rect(screen,(55,55,55),r); pygame.draw.rect(screen,WHITE,r,2); ss=small.render(txt,True,WHITE); screen.blit(ss,ss.get_rect(center=r.center))
        metrics=[f"Peak force: {qc['peak_force']:.3f} MVC", f"Peak time: {qc['peak_time_s']*1000 if np.isfinite(qc['peak_time_s']) else np.nan:.1f} ms", f"0-{FIXED_SLOPE_WINDOW_S*1000:.0f} ms slope: {qc['fixed_slope']:.3f} MVC/s"]
        for j,txt in enumerate(metrics): screen.blit(small.render(txt,True,WHITE),(int(w*.74),int(h*.53)+j*30))
        pygame.display.flip(); clock.tick(30)

def run_pre_task_phase(screen, joystick, calibration, right_rect, phase,
                       n_trials, review_mode="AUTO_SAVE", report_dir=None):
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
                draw_active_target(screen, right_rect, RED if USE_RED_CUE else NEUTRAL_GRAY)
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
                draw_cursor_track(screen, right_rect)
                draw_active_target(screen, right_rect, NEUTRAL_GRAY)
                draw_force_cursor(screen, right_force, right_rect)
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

    if report_dir is None:
        report_dir=os.path.join(OUTPUT_ROOT,"Baseline_QC")
    initial_included=[_baseline_trial_qc(x)["valid"] for x in baseline_trials]
    if review_mode == "INTERACTIVE":
        reviewed=review_baseline_trials(screen, baseline_trials, report_dir)
        if reviewed is None: return None
        included=reviewed
    else:
        included=initial_included
        save_baseline_review_report(baseline_trials, report_dir, included)

    plateau_times=np.asarray([x["plateau_s"] if keep else np.nan for x,keep in zip(baseline_trials,included)],dtype=float)
    valid_plateaus=plateau_times[np.isfinite(plateau_times)]
    min_valid=max(3,int(np.ceil(n_trials/2)))
    if valid_plateaus.size < min_valid:
        print(f"Baseline failed: only {valid_plateaus.size}/{n_trials} included trials had a valid ballistic peak (need {min_valid}).")
        return {"thresholds":None,"plateau_times_s":plateau_times,"mean_plateau_s":np.nan,"rfd_check_end_s":np.nan,"checkpoints_s":tuple(),"force_profile":None,"included_trials":np.asarray(included,bool),"trials":baseline_trials,"fixed_slope_mean":np.nan,"fixed_force_mean":np.nan,"report_dir":report_dir}

    mean_plateau_s=float(np.mean(valid_plateaus))
    # Baseline individualizes the early-slope endpoint using the operator's
    # selected quarter scheme.
    global FIXED_SLOPE_WINDOW_S
    _fit_fraction = _fixed_fit_fraction()
    FIXED_SLOPE_WINDOW_S=float(np.clip(
        mean_plateau_s*_fit_fraction,
        FIXED_SLOPE_MIN_WINDOW_S, FIXED_SLOPE_MAX_WINDOW_S
    ))
    baseline_fixed_trigger_target_s = float(FIXED_SLOPE_WINDOW_S + FIXED_TRIGGER_DELAY_S)
    checkpoints,individualized_end=make_rfd_checkpoints(mean_plateau_s)
    checkpoints=tuple(c for c in checkpoints if c <= MAX_RFD_CHECK_END_S+1e-12) or (RFD_CHECK_START_S,)
    individualized_end=float(checkpoints[-1])
    checkpoint_history=[]; force_checkpoint_history=[]; fixed_slopes=[]; fixed_forces=[]
    for trial,keep in zip(baseline_trials,included):
        if not keep: continue
        if trial["t"].size >= ONSET_CONFIRM_SAMPLES:
            onset_idx,smoothed=find_rising_onset(trial["t"],trial["force"])
            checkpoint_history.append(compute_checkpoint_rfds(trial["t"],trial["force"],onset_idx=onset_idx,smoothed=smoothed,checkpoints=checkpoints))
            force_checkpoint_history.append(compute_checkpoint_forces(trial["t"],trial["force"],onset_idx=onset_idx,smoothed=smoothed,checkpoints=checkpoints))
            fs=calculate_fixed_early_slope(trial["t"],trial["force"],onset_idx=onset_idx,smoothed=smoothed)
            fixed_slopes.append(fs)
            target_t=trial["t"][onset_idx]+FIXED_SLOPE_WINDOW_S if onset_idx is not None else np.nan
            if onset_idx is not None and trial["t"][-1]>=target_t:
                fixed_forces.append(float(smoothed[int(np.argmin(np.abs(trial["t"]-target_t)))]))
    thresholds=_average_checkpoint_history(checkpoint_history,checkpoints=checkpoints)
    baseline_force_profile=_average_checkpoint_history(force_checkpoint_history,checkpoints=checkpoints)
    fixed_slopes=np.asarray(fixed_slopes,dtype=float); fixed_forces=np.asarray(fixed_forces,dtype=float)
    fixed_slope_mean=float(np.nanmean(fixed_slopes)) if np.isfinite(fixed_slopes).any() else np.nan
    fixed_force_mean=float(np.nanmean(fixed_forces)) if np.isfinite(fixed_forces).any() else np.nan
    # Re-save after individualization so the final QC plots/CSV show the actual
    # baseline-derived slope endpoint rather than the pre-baseline GUI default.
    save_baseline_review_report(baseline_trials,report_dir,included)
    print("\n========================================\nINDIVIDUALIZED BASELINE BALLISTIC WINDOW\n========================================")
    print(f"Included valid peak trials: {valid_plateaus.size}/{n_trials} | mean onset-to-peak: {mean_plateau_s*1000:.1f} ms")
    print(f"Main Task RFD_CHECK_END_S: {individualized_end:.3f} s ({individualized_end*1000:.0f} ms)")
    print(f"Fixed early slope (0-{FIXED_SLOPE_WINDOW_S*1000:.0f} ms): {fixed_slope_mean:.4f} MVC/s")
    return {"thresholds":thresholds,"plateau_times_s":plateau_times,"mean_plateau_s":mean_plateau_s,"rfd_check_end_s":individualized_end,"checkpoints_s":checkpoints,"force_profile":baseline_force_profile,"included_trials":np.asarray(included,bool),"trials":baseline_trials,"fixed_slope_mean":fixed_slope_mean,"fixed_force_mean":fixed_force_mean,
            "fixed_slope_window_s":FIXED_SLOPE_WINDOW_S,
            "fixed_timing_mode":FIXED_TIMING_MODE,
            "fixed_trigger_delay_s":FIXED_TRIGGER_DELAY_S,
            "fixed_trigger_target_s":baseline_fixed_trigger_target_s,
            "report_dir":report_dir}



def startup_menu(screen, joystick, right_rect, calibration=None, baseline_result=None,
                 subject_id=None, session_id=None, block_id=None, trigger_mode=None,
                 stimulation_metric_mode=None, show_loadbar=None,
                 rfd_algorithm=None, feedback_enabled=None, feedback_metric=None,
                 baseline_review_mode=None):
    """Compact two-column operator GUI designed for wide, short displays."""
    global PREP_DURATION_S,GO_DURATION_S,ITI_MIN_S,ITI_JITTER_S,ITI_MAX_S,USE_RED_CUE
    global PREP_JITTER_S,GO_JITTER_S,TOTAL_TRIALS,FAMILIARIZATION_TRIALS,BASELINE_TRIALS
    global TARGET_FORCE_LEVEL,TARGET_TOLERANCE,FORCE_TRIGGER_CEILING,FIXED_SLOPE_WINDOW_S
    global RFD_TRIGGER_ALGORITHM,PARTICIPANT_FEEDBACK_ENABLED,PARTICIPANT_FEEDBACK_METRIC,BASELINE_REVIEW_MODE,FIXED_TIMING_MODE
    trigger_mode=trigger_mode or SLOPE_TRIGGER_MODE; stimulation_metric_mode=stimulation_metric_mode or STIMULATION_METRIC_MODE
    show_loadbar=MAIN_TASK_LOADBAR_DEFAULT if show_loadbar is None else show_loadbar
    rfd_algorithm=rfd_algorithm or RFD_TRIGGER_ALGORITHM
    feedback_enabled=PARTICIPANT_FEEDBACK_ENABLED if feedback_enabled is None else feedback_enabled
    feedback_metric=feedback_metric or PARTICIPANT_FEEDBACK_METRIC
    baseline_review_mode=baseline_review_mode or BASELINE_REVIEW_MODE
    subject_id=subject_id or X_NUMBER; session_id=session_id or SESSION; block_id=block_id or BLOCK
    active=None; clock=pygame.time.Clock(); w,h=screen.get_size(); margin=18; gap=10
    left_w=int(w*.48); right_x=int(w*.51); right_w=w-right_x-margin
    field_h=max(34,min(44,int(h*.055))); row=max(48,min(62,int(h*.072)))
    def rect(x,y,width,height=field_h): return pygame.Rect(x,y,width,height)
    # fields: three columns per half-screen
    vals={"subject":subject_id,"session":session_id,"block":block_id,
          "familiar":str(FAMILIARIZATION_TRIALS),"baseline":str(BASELINE_TRIALS),"task":str(TOTAL_TRIALS),
          "target":f"{TARGET_FORCE_LEVEL:.2f}","tolerance":f"{TARGET_TOLERANCE:.2f}","fixed_ms":f"{FIXED_SLOPE_WINDOW_S*1000:.0f}",
          "trigger_delay_ms":f"{FIXED_TRIGGER_DELAY_S*1000:.1f}",
          "red":f"{PREP_DURATION_S:.2f}","green":f"{GO_DURATION_S:.2f}","pause":f"{ITI_MIN_S:.2f}",
          "red_j":f"{PREP_JITTER_S:.2f}","green_j":f"{GO_JITTER_S:.2f}","pause_j":f"{ITI_JITTER_S:.2f}"}
    fields={}; fw=(left_w-2*gap)//3
    y0=68
    for j,name in enumerate(("subject","session","block")): fields[name]=rect(margin+j*(fw+gap),y0,fw)
    y1=y0+row
    for j,name in enumerate(("familiar","baseline","task")): fields[name]=rect(margin+j*(fw+gap),y1,fw)
    y2=y1+row
    for j,name in enumerate(("target","tolerance","fixed_ms")): fields[name]=rect(margin+j*(fw+gap),y2,fw)
    y3=y2+row
    fields["trigger_delay_ms"]=rect(margin,y3,fw)
    for j,name in enumerate(("red","green","pause")): fields[name]=rect(margin+j*(fw+gap),y3+row,fw)
    y4=y3+2*row
    for j,name in enumerate(("red_j","green_j","pause_j")): fields[name]=rect(margin+j*(fw+gap),y4,fw)
    # right side controls as paired buttons
    bw=(right_w-gap)//2; bh=max(36,min(48,int(h*.055))); ry=68
    controls={}
    for key in ("algorithm","timing_fraction","direction"):
        controls[key]=(rect(right_x,ry,bw,bh),rect(right_x+bw+gap,ry,bw,bh)); ry+=bh+8
    # Three explicit stimulation-metric choices: RFD, FORCE, or BOTH.
    metric_bw=(right_w-2*gap)//3
    controls["metric"]=(
        rect(right_x,ry,metric_bw,bh),
        rect(right_x+metric_bw+gap,ry,metric_bw,bh),
        rect(right_x+2*(metric_bw+gap),ry,metric_bw,bh),
    )
    ry+=bh+8
    for key in ("cue","feedback","feedback_metric","review","loadbar"):
        controls[key]=(rect(right_x,ry,bw,bh),rect(right_x+bw+gap,ry,bw,bh)); ry+=bh+8
    action_y=max(ry+6,int(h*.70)); aw=(right_w-gap)//2
    calibrate=rect(right_x,action_y,aw,bh); familiar=rect(right_x+aw+gap,action_y,aw,bh)
    baseline=rect(right_x,action_y+bh+8,aw,bh); startb=rect(right_x+aw+gap,action_y+bh+8,aw,bh)
    def sanitize(v,fb):
        x=''.join(c for c in v.strip() if c.isalnum() or c in '_-'); return x or fb
    labels={"subject":"Subject","session":"Session","block":"Block","familiar":"Familiar trials","baseline":"Baseline trials","task":"Task trials",
            "target":"Target level (MVC)","tolerance":"Half-height (MVC)","fixed_ms":"Fixed slope end (ms)","trigger_delay_ms":"TMS delay after fit (ms)","red":"Red/base prep (s)","green":"Green GO (s)","pause":"Pause (s)","red_j":"Red jitter (s)","green_j":"GO jitter (s)","pause_j":"Pause jitter (s)"}
    numeric=set(vals)-{"subject","session","block"}
    def commit(name):
        nonlocal subject_id,session_id,block_id
        global PREP_DURATION_S,GO_DURATION_S,ITI_MIN_S,ITI_JITTER_S,ITI_MAX_S,PREP_JITTER_S,GO_JITTER_S,TOTAL_TRIALS,FAMILIARIZATION_TRIALS,BASELINE_TRIALS,TARGET_FORCE_LEVEL,TARGET_TOLERANCE,FORCE_TRIGGER_CEILING,FIXED_SLOPE_WINDOW_S,FIXED_TRIGGER_DELAY_S
        try:
            if name in ("subject","session","block"): return
            v=float(vals[name])
            if name in ("familiar","baseline","task"): v=max(1,int(round(v)))
            elif name=="target": v=float(np.clip(v,.05,.95))
            elif name=="tolerance": v=float(np.clip(v,.01,.40))
            elif name=="fixed_ms": v=float(np.clip(v,10,200))
            elif name=="trigger_delay_ms": v=float(np.clip(v,0,100))
            else: v=max(0.,v)
        except ValueError: return
        if name=="familiar": FAMILIARIZATION_TRIALS=v
        elif name=="baseline": BASELINE_TRIALS=v
        elif name=="task": TOTAL_TRIALS=v
        elif name=="target": TARGET_FORCE_LEVEL=v
        elif name=="tolerance": TARGET_TOLERANCE=v
        elif name=="fixed_ms": FIXED_SLOPE_WINDOW_S=v/1000.
        elif name=="trigger_delay_ms": FIXED_TRIGGER_DELAY_S=v/1000.
        elif name=="red": PREP_DURATION_S=v
        elif name=="green": GO_DURATION_S=v
        elif name=="pause": ITI_MIN_S=v
        elif name=="red_j": PREP_JITTER_S=v
        elif name=="green_j": GO_JITTER_S=v
        elif name=="pause_j": ITI_JITTER_S=v
        ITI_MAX_S=ITI_MIN_S+ITI_JITTER_S; FORCE_TRIGGER_CEILING=TARGET_FORCE_LEVEL-TARGET_TOLERANCE
    def rebuild_target():
        new=create_target_rect(h); right_rect.size=new.size; place_targets(right_rect,w,h)
    while True:
        calready=calibration is not None and calibration.is_complete()
        bth=baseline_result.get("thresholds") if isinstance(baseline_result,dict) else None
        bfp=baseline_result.get("force_profile") if isinstance(baseline_result,dict) else None
        bready=bth is not None and bfp is not None and any(np.isfinite(v) for v in bth.values()) and np.isfinite(baseline_result.get("rfd_check_end_s",np.nan))
        for e in pygame.event.get():
            if e.type==pygame.QUIT:return (None,)*12
            if e.type==pygame.KEYDOWN:
                if e.key==pygame.K_q and active is None:return (None,)*12
                if active:
                    if e.key==pygame.K_RETURN: commit(active); active=None; rebuild_target()
                    elif e.key==pygame.K_BACKSPACE: vals[active]=vals[active][:-1]
                    else:
                        ch=e.unicode
                        if active in numeric:
                            if ch and (ch.isdigit() or ch=='.'): vals[active]+=ch
                        elif ch and (ch.isalnum() or ch in '_-'): vals[active]+=ch
            if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                if active: commit(active); active=None; rebuild_target()
                hit=False
                for name,r in fields.items():
                    if r.collidepoint(e.pos): active=name; hit=True; break
                if hit: continue
                a,b=controls["algorithm"]
                if a.collidepoint(e.pos): rfd_algorithm="SLIDING_CHECKPOINTS"
                elif b.collidepoint(e.pos): rfd_algorithm="FIXED_EARLY_SLOPE"
                else:
                    a,b=controls["timing_fraction"]
                    if a.collidepoint(e.pos):
                        k=VALID_FIXED_TIMING_MODES.index(FIXED_TIMING_MODE)
                        FIXED_TIMING_MODE=VALID_FIXED_TIMING_MODES[(k-1)%len(VALID_FIXED_TIMING_MODES)]
                    elif b.collidepoint(e.pos):
                        k=VALID_FIXED_TIMING_MODES.index(FIXED_TIMING_MODE)
                        FIXED_TIMING_MODE=VALID_FIXED_TIMING_MODES[(k+1)%len(VALID_FIXED_TIMING_MODES)]
                    else:
                        a,b=controls["direction"]
                        if a.collidepoint(e.pos): trigger_mode="BELOW"
                        elif b.collidepoint(e.pos): trigger_mode="ABOVE"
                        else:
                            a,b,c=controls["metric"]
                            if a.collidepoint(e.pos): stimulation_metric_mode="RFD_ONLY"
                            elif b.collidepoint(e.pos): stimulation_metric_mode="FORCE_ONLY"
                            elif c.collidepoint(e.pos): stimulation_metric_mode="COMBINED"
                            else:
                                a,b=controls["cue"]
                                if a.collidepoint(e.pos): USE_RED_CUE=True
                                elif b.collidepoint(e.pos): USE_RED_CUE=False
                                else:
                                    a,b=controls["feedback"]
                                    if a.collidepoint(e.pos): feedback_enabled=True
                                    elif b.collidepoint(e.pos): feedback_enabled=False
                                    else:
                                        a,b=controls["feedback_metric"]
                                        if a.collidepoint(e.pos):
                                            k=VALID_FEEDBACK_METRICS.index(feedback_metric); feedback_metric=VALID_FEEDBACK_METRICS[(k-1)%len(VALID_FEEDBACK_METRICS)]
                                        elif b.collidepoint(e.pos):
                                            k=VALID_FEEDBACK_METRICS.index(feedback_metric); feedback_metric=VALID_FEEDBACK_METRICS[(k+1)%len(VALID_FEEDBACK_METRICS)]
                                        else:
                                            a,b=controls["review"]
                                            if a.collidepoint(e.pos): baseline_review_mode="AUTO_SAVE"
                                            elif b.collidepoint(e.pos): baseline_review_mode="INTERACTIVE"
                                            else:
                                                a,b=controls["loadbar"]
                                                if a.collidepoint(e.pos): show_loadbar=True
                                                elif b.collidepoint(e.pos): show_loadbar=False
                                                elif calibrate.collidepoint(e.pos):
                                                    calibration=run_calibration(joystick,screen); baseline_result=None
                                                elif familiar.collidepoint(e.pos) and calready:
                                                    if run_pre_task_phase(screen,joystick,calibration,right_rect,"FAMILIARIZATION",FAMILIARIZATION_TRIALS) is None:return (None,)*12
                                                elif baseline.collidepoint(e.pos) and calready:
                                                    sid=sanitize(vals['subject'],'X00000'); ses=sanitize(vals['session'],'Session_1')
                                                    rep=os.path.join(OUTPUT_ROOT,sid,ses,"Baseline_QC_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                                                    baseline_result=run_pre_task_phase(screen,joystick,calibration,right_rect,"BASELINE",BASELINE_TRIALS,review_mode=baseline_review_mode,report_dir=rep)
                                                    if baseline_result is None:return (None,)*12
                                                    # Reflect the baseline-derived adaptive slope endpoint immediately in the GUI.
                                                    vals["fixed_ms"]=f"{FIXED_SLOPE_WINDOW_S*1000:.1f}"
                                                    vals["trigger_delay_ms"]=f"{FIXED_TRIGGER_DELAY_S*1000:.1f}"
                                                elif startb.collidepoint(e.pos) and calready and bready:
                                                    for n in numeric: commit(n)
                                                    rebuild_target(); subject_id=sanitize(vals['subject'],'X00000'); session_id=sanitize(vals['session'],'Session_1'); block_id=sanitize(vals['block'],'Block_1')
                                                    for warning in validate_and_fix_timing(): print('WARNING:',warning)
                                                    RFD_TRIGGER_ALGORITHM=rfd_algorithm; PARTICIPANT_FEEDBACK_ENABLED=feedback_enabled; PARTICIPANT_FEEDBACK_METRIC=feedback_metric; BASELINE_REVIEW_MODE=baseline_review_mode
                                                    return calibration,trigger_mode,stimulation_metric_mode,baseline_result,show_loadbar,subject_id,session_id,block_id,rfd_algorithm,feedback_enabled,feedback_metric,baseline_review_mode
        screen.fill(BLACK); title=get_font(max(32,min(48,int(h*.06)))); ts=title.render("Adaptive Thresholding TMS Task",True,WHITE); screen.blit(ts,(margin,12))
        lf=get_font(max(15,min(18,int(h*.023)))); vf=get_font(max(18,min(23,int(h*.03))))
        for name,r in fields.items():
            pygame.draw.rect(screen,(70,70,70) if active==name else (42,42,42),r); pygame.draw.rect(screen,WHITE if active==name else (130,130,130),r,2)
            screen.blit(lf.render(labels[name],True,(185,185,185)),(r.x,r.y-18)); ss=vf.render(vals[name],True,WHITE); screen.blit(ss,ss.get_rect(center=r.center))
        def pair(key,l1,l2,sel1=False,sel2=False):
            a,b=controls[key]
            for r,txt,sel in ((a,l1,sel1),(b,l2,sel2)):
                pygame.draw.rect(screen,(70,70,70),r); pygame.draw.rect(screen,GREEN if sel else (135,135,135),r,4 if sel else 2); q=lf.render(txt,True,WHITE); screen.blit(q,q.get_rect(center=r.center))
        pair('algorithm','Sliding RFD','Fixed early slope',rfd_algorithm=='SLIDING_CHECKPOINTS',rfd_algorithm=='FIXED_EARLY_SLOPE')
        _timing_label={
            "FIT_1Q":"Fit 1/4 -> TMS +5ms",
            "FIT_2Q":"Fit 2/4 -> TMS +5ms",
            "FIT_3Q":"Fit 3/4 -> TMS +5ms",
        }[FIXED_TIMING_MODE]
        pair('timing_fraction','< '+_timing_label,_timing_label+' >')
        pair('direction','Trigger BELOW','Trigger ABOVE',trigger_mode=='BELOW',trigger_mode=='ABOVE')
        # Stimulation metric has three explicit choices; Combined is never hidden behind a toggle.
        ma,mb,mc=controls['metric']
        for rr,txt,sel in ((ma,'RFD only',stimulation_metric_mode=='RFD_ONLY'),
                           (mb,'Force only',stimulation_metric_mode=='FORCE_ONLY'),
                           (mc,'Combined',stimulation_metric_mode=='COMBINED')):
            pygame.draw.rect(screen,(70,70,70),rr)
            pygame.draw.rect(screen,GREEN if sel else (135,135,135),rr,4 if sel else 2)
            qq=lf.render(txt,True,WHITE); screen.blit(qq,qq.get_rect(center=rr.center))
        pair('cue','Red cue ON','Grey -> Green',USE_RED_CUE,not USE_RED_CUE)
        pair('feedback','Feedback ON','Feedback OFF',feedback_enabled,not feedback_enabled)
        pair('feedback_metric','< '+_feedback_gui_label(feedback_metric),_feedback_gui_label(feedback_metric)+' >')
        pair('review','QC auto-save','QC interactive',baseline_review_mode=='AUTO_SAVE',baseline_review_mode=='INTERACTIVE')
        pair('loadbar','Loadbar ON','Loadbar OFF',show_loadbar,not show_loadbar)
        for r,txt,en in ((calibrate,'Calibrate grip',True),(familiar,f'Familiarize ({FAMILIARIZATION_TRIALS})',calready),(baseline,f'Baseline ({BASELINE_TRIALS})',calready),(startb,f'Start task ({TOTAL_TRIALS})',calready and bready)):
            pygame.draw.rect(screen,(65,65,65) if en else (28,28,28),r); pygame.draw.rect(screen,GREEN if en and r==startb else (150,150,150),r,2); q=lf.render(txt,True,WHITE if en else (90,90,90)); screen.blit(q,q.get_rect(center=r.center))
        status=f"Calibration: {'READY' if calready else 'needed'}   Baseline: {'READY' if bready else 'needed'}"
        screen.blit(lf.render(status,True,GREEN if bready else (210,210,210)),(right_x,min(h-24,action_y+2*(bh+8)+4)))
        pygame.display.flip(); clock.tick(30)

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
    """Place target; reserve the right side for optional participant feedback."""
    if right_force_level is None:
        right_force_level = TARGET_FORCE_LEVEL
    right_y = force_to_cursor_y(right_force_level, height)
    task_width = int(width * 0.70) if PARTICIPANT_FEEDBACK_ENABLED else width
    right_rect.center = (task_width // 2, right_y)

def configure_targets_for_trial(trial_number, right_rect, width, height):
    """Configure the single RIGHT-hand target for every trial."""
    place_targets(right_rect, width, height)
    return {"right": True}

def _feedback_value(metrics, metric):
    if not metrics:
        return np.nan
    if metric == "PEAK_FORCE":
        return metrics.get("right_peak_force", np.nan)
    if metric == "ADAPTIVE_RFD":
        # This is the actual onset->adaptive-window regression slope used by
        # FIXED_EARLY_SLOPE on that trial.
        return metrics.get("right_fixed_early_slope", np.nan)
    if metric == "RFD_80MS":
        # Legacy/reporting RFD: one regression from onset to 80 ms.
        return metrics.get("right_rfd", np.nan)
    if metric == "REACTION_TIME":
        return metrics.get("right_reaction_time_s", np.nan) * 1000.0
    if metric == "TARGET_ERROR":
        return metrics.get("right_mean_abs_error_mvc", np.nan)
    return np.nan


def _feedback_axis_info(metric):
    """Human-readable participant-feedback axis label and display unit."""
    info = {
        "PEAK_FORCE": ("Peak force", "MVC"),
        "ADAPTIVE_RFD": ("Adaptive RFD", "MVC/s"),
        "RFD_80MS": ("RFD 0-80 ms", "MVC/s"),
        "REACTION_TIME": ("Reaction time", "ms"),
        "TARGET_ERROR": ("Target error", "MVC"),
    }
    return info.get(metric, (metric.replace("_", " ").title(), ""))


def _feedback_gui_label(metric):
    """Compact GUI label for participant-feedback metric selection."""
    return {
        "PEAK_FORCE": "Peak force",
        "ADAPTIVE_RFD": "Adaptive RFD",
        "RFD_80MS": "RFD 0-80 ms",
        "REACTION_TIME": "Reaction time",
        "TARGET_ERROR": "Target error",
    }.get(metric, metric.replace("_", " ").title())


def draw_participant_feedback(screen, trial_log, metric):
    """On-screen trial history with explicit axis labels and physical units."""
    if not PARTICIPANT_FEEDBACK_ENABLED:
        return

    w, h = screen.get_size()
    panel = pygame.Rect(int(w*.72), int(h*.10), int(w*.26), int(h*.78))
    pygame.draw.rect(screen, (20,20,20), panel)
    pygame.draw.rect(screen, (120,120,120), panel, 2)

    font = get_font(max(18, min(26, int(h*.035))))
    small = get_font(max(14, min(20, int(h*.026))))
    tick_font = get_font(max(12, min(17, int(h*.022))))
    y_name, unit = _feedback_axis_info(metric)

    title = font.render(f"Feedback: {y_name}", True, WHITE)
    screen.blit(title, (panel.x+10, panel.y+8))

    vals = np.asarray(
        [_feedback_value(m, metric) for m in trial_log[-FEEDBACK_HISTORY_POINTS:]],
        dtype=float
    )
    finite = np.isfinite(vals)

    # Extra left/bottom room is reserved for tick values and axis units.
    plot = pygame.Rect(panel.x+68, panel.y+58, panel.w-88, panel.h-130)
    pygame.draw.line(screen, (130,130,130),
                     (plot.left,plot.bottom),(plot.right,plot.bottom),2)
    pygame.draw.line(screen, (130,130,130),
                     (plot.left,plot.top),(plot.left,plot.bottom),2)

    # Explicit axes: x is always trial number; y depends on selected metric.
    x_label = small.render("Trial number", True, (210,210,210))
    screen.blit(x_label, x_label.get_rect(center=(plot.centerx, plot.bottom+31)))

    y_label_text = f"{y_name} ({unit})" if unit else y_name
    y_label = tick_font.render(y_label_text, True, (210,210,210))
    y_label = pygame.transform.rotate(y_label, 90)
    screen.blit(y_label, y_label.get_rect(center=(panel.x+17, plot.centery)))

    if finite.any():
        v = vals[finite]
        lo = float(np.min(v))
        hi = float(np.max(v))
        pad = max((hi-lo)*.15, abs(hi)*.05, 1e-6)
        lo -= pad
        hi += pad
        if hi <= lo:
            hi = lo + 1.0

        coords = []
        inds = np.where(finite)[0]
        for idx,val in zip(inds,v):
            x = plot.left + (idx/max(1,len(vals)-1))*plot.w
            y = plot.bottom - (val-lo)/(hi-lo)*plot.h
            coords.append((int(x),int(y)))

        if len(coords)>1:
            pygame.draw.lines(screen, WHITE, False, coords, 2)
        for pt in coords:
            pygame.draw.circle(screen, GREEN, pt, 5)

        # Y-axis min/mid/max tick labels carry the same unit as the axis title.
        for frac, value in ((0.0, lo), (0.5, (lo+hi)/2.0), (1.0, hi)):
            y = plot.bottom - frac*plot.h
            pygame.draw.line(screen,(130,130,130),(plot.left-5,int(y)),
                             (plot.left,int(y)),1)
            lab = tick_font.render(f"{value:.2f}", True, (190,190,190))
            screen.blit(lab, lab.get_rect(midright=(plot.left-8,int(y))))

        # X-axis tick labels show the actual task trial numbers represented.
        n_shown = len(vals)
        first_trial = max(1, len(trial_log)-n_shown+1)
        tick_positions = sorted(set([0, max(0,(n_shown-1)//2), max(0,n_shown-1)]))
        for idx in tick_positions:
            x = plot.left + (idx/max(1,n_shown-1))*plot.w
            pygame.draw.line(screen,(130,130,130),(int(x),plot.bottom),
                             (int(x),plot.bottom+5),1)
            trial_no = first_trial + idx
            lab = tick_font.render(str(trial_no), True, (190,190,190))
            screen.blit(lab, lab.get_rect(midtop=(int(x),plot.bottom+7)))

        last = v[-1]
        suffix = f" {unit}" if unit else ""
        window_note = ""
        if metric == "ADAPTIVE_RFD":
            # Report the exact adaptive window used by the most recent finite
            # trial, so the participant/operator knows what the slope means.
            finite_logs = [
                m for m in trial_log[-FEEDBACK_HISTORY_POINTS:]
                if np.isfinite(_feedback_value(m, metric))
            ]
            if finite_logs:
                w_s = finite_logs[-1].get("right_fixed_slope_window_s", np.nan)
                if np.isfinite(w_s):
                    window_note = f" (0-{w_s*1000:.0f} ms)"
        screen.blit(
            small.render(f"Last: {last:.3f}{suffix}{window_note}", True, WHITE),
            (panel.x+10,panel.bottom-34)
        )
        if len(v)>1:
            prev=v[-2]
            delta=last-prev
            good_delta=-delta if metric in ("REACTION_TIME","TARGET_ERROR") else delta
            txt=("Better" if good_delta>0 else "Worse" if good_delta<0 else "Same")
            txt += f" ({delta:+.3f}{suffix})"
            screen.blit(
                small.render(txt,True,GREEN if good_delta>=0 else RED),
                (panel.x+150,panel.bottom-34)
            )
    else:
        screen.blit(
            small.render("Feedback appears after trial 1",True,(180,180,180)),
            (plot.left,plot.centery)
        )

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
        self.rfd_algorithm = RFD_TRIGGER_ALGORITHM

        self._reset_state()

    def _reset_state(self):
        self.go_start_time = 0.0
        self.thresholds = {}
        self.force_references = {}
        self.fixed_slope_threshold = np.nan
        self.fixed_force_reference = np.nan
        self.fixed_slope_window_s = float(FIXED_SLOPE_WINDOW_S)
        self.fixed_trigger_delay_s = float(FIXED_TRIGGER_DELAY_S)

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

    def arm(self, go_start_time, thresholds, force_references, fixed_slope_threshold=np.nan, fixed_force_reference=np.nan,
            fixed_slope_window_s=None, fixed_trigger_delay_s=None):
        with self.lock:
            self._reset_state()
            self.go_start_time = go_start_time
            self.thresholds = dict(thresholds)
            self.force_references = dict(force_references)
            self.fixed_slope_threshold = fixed_slope_threshold
            self.fixed_force_reference = fixed_force_reference
            self.fixed_slope_window_s = float(FIXED_SLOPE_WINDOW_S if fixed_slope_window_s is None else fixed_slope_window_s)
            self.fixed_trigger_delay_s = float(FIXED_TRIGGER_DELAY_S if fixed_trigger_delay_s is None else fixed_trigger_delay_s)

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
                "rfd_trigger_algorithm": self.rfd_algorithm,
                "fixed_timing_mode": FIXED_TIMING_MODE,
                "fixed_slope_threshold": self.fixed_slope_threshold,
                "fixed_slope_window_s": self.fixed_slope_window_s,
                "fixed_trigger_delay_s": self.fixed_trigger_delay_s,
                "fixed_trigger_target_s": self.fixed_slope_window_s + self.fixed_trigger_delay_s,
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

            if self.rfd_algorithm == "FIXED_EARLY_SLOPE":
                slope_window_s=float(self.fixed_slope_window_s)
                trigger_target_s=slope_window_s+float(self.fixed_trigger_delay_s)
                slope_key=int(round(slope_window_s*1000))
                trigger_key=int(round(trigger_target_s*1000))
                # Wait until the intended stimulation latency. The slope itself
                # is frozen at the earlier 0 -> slope_window_s interval.
                if elapsed_since_onset >= trigger_target_s and trigger_key not in done:
                    rfd=calculate_fixed_early_slope(t_window,f_window,window_s=slope_window_s,onset_idx=onset_idx,smoothed=smoothed)
                    threshold=float(self.fixed_slope_threshold); target_time=onset_time+slope_window_s
                    idx=int(np.argmin(np.abs(t_window-target_time))); checkpoint_force=float(smoothed[idx]); force_reference=float(self.fixed_force_reference)
                    with self.lock: self.checkpoints_done.add(trigger_key); self.checked[slope_key]=rfd
                    done.add(trigger_key); rfd_valid=np.isfinite(rfd) and np.isfinite(threshold); force_valid=np.isfinite(checkpoint_force) and np.isfinite(force_reference)
                    if self.trigger_mode=="BELOW":
                        rfd_decision_threshold=threshold*(1-TRIGGER_MARGIN_FRACTION) if rfd_valid else np.nan; rfd_condition=rfd_valid and rfd<rfd_decision_threshold
                        force_decision_threshold=force_reference*(1-FORCE_REFERENCE_MARGIN) if force_valid else np.nan; force_condition=force_valid and checkpoint_force<force_decision_threshold and checkpoint_force<FORCE_TRIGGER_CEILING
                    else:
                        rfd_decision_threshold=threshold*(1+TRIGGER_MARGIN_FRACTION) if rfd_valid else np.nan; rfd_condition=rfd_valid and rfd>rfd_decision_threshold
                        force_decision_threshold=force_reference*(1+FORCE_REFERENCE_MARGIN) if force_valid else np.nan; force_condition=force_valid and checkpoint_force>force_decision_threshold
                    should_trigger=rfd_condition if self.metric_mode=="RFD_ONLY" else force_condition if self.metric_mode=="FORCE_ONLY" else (rfd_condition and force_condition)
                    print(f"[RIGHT FIXED slope 0-{slope_key} ms -> stim {trigger_key} ms] slope={rfd:.3f} ref={threshold:.3f} | force@{slope_key}ms={checkpoint_force:.3f} ref={force_reference:.3f} | gate={'PASS' if should_trigger else 'no'}")
                    if should_trigger: fired=(trigger_key,rfd,threshold,trigger_target_s,checkpoint_force,force_reference)
                    else:
                        with self.lock: self.finished=True; self.decision_reason=f"fixed_slope_{self.trigger_mode.lower()}_criterion_not_met"
                # Fixed algorithm has exactly one decision point; skip sliding checks.
            else:
                pass

            for checkpoint_s in (() if self.rfd_algorithm == "FIXED_EARLY_SLOPE" else RFD_CHECKPOINTS_S):

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

            elif self.rfd_algorithm != "FIXED_EARLY_SLOPE" and elapsed_since_onset >= RFD_CHECK_END_S:
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
    out[f"{side}_rfd_trigger_algorithm"] = monitor_result.get("rfd_trigger_algorithm", RFD_TRIGGER_ALGORITHM)
    out[f"{side}_fixed_timing_mode"] = monitor_result.get("fixed_timing_mode", FIXED_TIMING_MODE)
    out[f"{side}_fixed_slope_threshold"] = monitor_result.get("fixed_slope_threshold", np.nan)
    out[f"{side}_fixed_slope_window_s"] = monitor_result.get("fixed_slope_window_s", FIXED_SLOPE_WINDOW_S)
    out[f"{side}_fixed_trigger_delay_s"] = monitor_result.get("fixed_trigger_delay_s", FIXED_TRIGGER_DELAY_S)
    out[f"{side}_fixed_trigger_target_s"] = monitor_result.get("fixed_trigger_target_s", np.nan)
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
        out[f"{side}_fixed_early_slope"] = np.nan
        out[f"{side}_fixed_force"] = np.nan
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
        out[f"{side}_ballistic_peak_latency_s"] = np.nan
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

    _trial_fixed_window=float(monitor_result.get("fixed_slope_window_s",FIXED_SLOPE_WINDOW_S))
    out[f"{side}_fixed_early_slope"] = calculate_fixed_early_slope(t, force, window_s=_trial_fixed_window, onset_idx=onset_idx, smoothed=smoothed)
    _fixed_target_time = onset_time + _trial_fixed_window
    out[f"{side}_fixed_force"] = (float(smoothed[int(np.argmin(np.abs(t-_fixed_target_time)))])
                                  if t[-1] >= _fixed_target_time else np.nan)
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
    out[f"{side}_ballistic_peak_latency_s"] = calculate_time_to_force_plateau(t, force)
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
    """Publication-style RIGHT-handle force/RFD figure for every trial.

    Plot representation follows the trigger algorithm actually used:
      * SLIDING_CHECKPOINTS -> local 20-ms RFD trace + checkpoint threshold curve.
      * FIXED_EARLY_SLOPE   -> onset-anchored linear fit over that trial's adaptive
        window + a threshold-slope comparison. No sliding-RFD threshold is shown.
    """
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
        ].copy()
        summary_rows = df_trials[df_trials["trial"] == trial_number]
        if trial_df.empty or summary_rows.empty:
            continue
        summary = summary_rows.iloc[0]

        algorithm = str(summary.get(
            "right_rfd_trigger_algorithm", RFD_TRIGGER_ALGORITHM
        ))
        is_fixed = algorithm == "FIXED_EARLY_SLOPE"

        fig, (ax_force, ax_rfd) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        color = PLOT_COLORS["right"]
        x = pd.to_numeric(trial_df["time_from_go_s"], errors="coerce").to_numpy(float)
        force_raw = pd.to_numeric(trial_df["force"], errors="coerce").to_numpy(float)
        force_sm = pd.to_numeric(trial_df["force_smoothed"], errors="coerce").to_numpy(float)
        rt = float(summary.get("right_reaction_time_s", np.nan))

        # --------------------------- force panel ---------------------------
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
        ax_force.plot(x, force_raw, color=color, lw=0.9, alpha=0.30)
        ax_force.plot(x, force_sm, color=color, lw=2.0, label="Right force")

        if np.isfinite(rt):
            for ax in (ax_force, ax_rfd):
                ax.axvline(rt, color=color, ls="--", lw=1.5, alpha=0.8,
                           label="Force onset" if ax is ax_force else None)

        if is_fixed:
            # =============================================================
            # FIXED / ADAPTIVE EARLY-SLOPE REPRESENTATION
            # =============================================================
            window_s = float(summary.get(
                "right_fixed_slope_window_s", FIXED_SLOPE_WINDOW_S
            ))
            delay_s = float(summary.get(
                "right_fixed_trigger_delay_s", FIXED_TRIGGER_DELAY_S
            ))
            reference_slope = float(summary.get(
                "right_fixed_slope_threshold", np.nan
            ))
            actual_slope = float(summary.get(
                "right_fixed_early_slope", np.nan
            ))
            trigger_target_s = window_s + delay_s

            if SLOPE_TRIGGER_MODE == "BELOW":
                decision_slope = (reference_slope * (1.0 - TRIGGER_MARGIN_FRACTION)
                                  if np.isfinite(reference_slope) else np.nan)
            else:
                decision_slope = (reference_slope * (1.0 + TRIGGER_MARGIN_FRACTION)
                                  if np.isfinite(reference_slope) else np.nan)

            if np.isfinite(rt):
                fit_end_go = rt + window_s
                planned_stim_go = rt + trigger_target_s

                # Highlight only the actual adaptive regression epoch.
                for ax in (ax_force, ax_rfd):
                    ax.axvspan(rt, fit_end_go, color=color, alpha=0.08)
                    ax.axvline(fit_end_go, color=color, ls=":", lw=2.0,
                               alpha=0.9)
                    ax.axvline(planned_stim_go, color=PLOT_COLORS["trigger"],
                               ls="--", lw=1.8, alpha=0.75)

                # Reconstruct and draw the actual linear regression used for
                # the trial, directly on the FORCE curve.
                fit_mask = (
                    np.isfinite(x) & np.isfinite(force_sm)
                    & (x >= rt) & (x <= fit_end_go)
                )
                if fit_mask.sum() >= MIN_EPOCH_SAMPLES:
                    x_rel = x[fit_mask] - rt
                    y_fit_data = force_sm[fit_mask]
                    slope_fit, intercept_fit = np.polyfit(x_rel, y_fit_data, 1)
                    xx_rel = np.linspace(0.0, window_s, 100)
                    xx_go = rt + xx_rel
                    yy_fit = intercept_fit + slope_fit * xx_rel
                    ax_force.plot(
                        xx_go, yy_fit, lw=3.0, color=PLOT_COLORS["mean"],
                        label=(f"Actual linear fit: {slope_fit:.2f} MVC/s")
                    )

                    # Convert the decision threshold (a single slope number)
                    # into a line anchored at the SAME fitted intercept. This
                    # makes the visual comparison genuinely slope-vs-slope.
                    if np.isfinite(decision_slope):
                        yy_thr = intercept_fit + decision_slope * xx_rel
                        ax_force.plot(
                            xx_go, yy_thr, lw=2.6, ls="--",
                            color=PLOT_COLORS["threshold"],
                            label=(f"Decision-slope threshold: "
                                   f"{decision_slope:.2f} MVC/s")
                        )

                # Lower panel: one slope measurement, one decision threshold.
                # Deliberately no local-RFD trace or checkpoint threshold curve.
                if np.isfinite(actual_slope):
                    ax_rfd.hlines(
                        actual_slope, rt, fit_end_go, color=PLOT_COLORS["mean"],
                        lw=3.2, label=f"Adaptive-window slope: {actual_slope:.2f} MVC/s"
                    )
                    ax_rfd.plot(fit_end_go, actual_slope, "o",
                                color=PLOT_COLORS["mean"], ms=7)
                if np.isfinite(reference_slope):
                    ax_rfd.hlines(
                        reference_slope, rt, fit_end_go,
                        color=PLOT_COLORS["threshold"], lw=1.8, ls=":",
                        label=f"Previous-trial reference: {reference_slope:.2f} MVC/s"
                    )
                if np.isfinite(decision_slope):
                    ax_rfd.hlines(
                        decision_slope, rt, fit_end_go,
                        color=PLOT_COLORS["threshold"], lw=2.8, ls="--",
                        label=f"Trigger decision threshold: {decision_slope:.2f} MVC/s"
                    )

            ax_rfd.set_ylabel("Fixed early slope (MVC/s)")
            ax_rfd.set_title(
                "Adaptive fixed early-slope decision: linear fit from onset to "
                "trial-specific window endpoint"
            )

        else:
            # =============================================================
            # SLIDING-CHECKPOINT REPRESENTATION (existing algorithm)
            # =============================================================
            rfd_trace_col = (
                "local_rfd_full_epoch"
                if "local_rfd_full_epoch" in trial_df.columns
                else "running_slope_diagnostic"
            )
            rfd_plot = pd.to_numeric(
                trial_df[rfd_trace_col], errors="coerce"
            ).to_numpy(dtype=float)

            cp_x, cp_y, th_x, th_trigger = [], [], [], []
            force_ref_x, force_ref_y = [], []

            if np.isfinite(rt):
                for ax in (ax_force, ax_rfd):
                    ax.axvspan(
                        rt + RFD_CHECK_START_S,
                        rt + RFD_CHECK_END_S,
                        color=color, alpha=0.07,
                    )

                for c in RFD_CHECKPOINTS_S:
                    k = int(round(c * 1000))
                    v = summary.get(f"right_rfd_{k}ms", np.nan)
                    th = summary.get(f"right_threshold_{k}ms", np.nan)
                    force_ref = summary.get(f"right_force_reference_{k}ms", np.nan)
                    if np.isfinite(force_ref):
                        force_ref_x.append(rt + c); force_ref_y.append(force_ref)
                    if np.isfinite(v):
                        cp_x.append(rt + c); cp_y.append(v)
                    if np.isfinite(th):
                        th_x.append(rt + c); th_trigger.append(th)

            go_mask = np.isfinite(x) & (x >= 0.0)
            if np.any(go_mask):
                x_go = x[go_mask]
                rfd_go = rfd_plot[go_mask].copy()
                if np.isfinite(rt):
                    rfd_go[x_go < rt] = 0.0
                node_x = list(x_go[np.isfinite(rfd_go)])
                node_y = list(rfd_go[np.isfinite(rfd_go)])
                node_x.append(0.0); node_y.append(0.0)
                if np.isfinite(rt):
                    node_x.append(float(rt)); node_y.append(0.0)
                for xx, yy in zip(cp_x, cp_y):
                    node_x.append(float(xx)); node_y.append(float(yy))
                node_x = np.asarray(node_x, dtype=float)
                node_y = np.asarray(node_y, dtype=float)
                valid_nodes = np.isfinite(node_x) & np.isfinite(node_y)
                node_x, node_y = node_x[valid_nodes], node_y[valid_nodes]
                if node_x.size >= 2:
                    order = np.argsort(node_x, kind="stable")
                    node_x, node_y = node_x[order], node_y[order]
                    unique_x, unique_y = [], []
                    for xx, yy in zip(node_x, node_y):
                        if unique_x and np.isclose(xx, unique_x[-1], atol=1e-8):
                            unique_y[-1] = float(yy)
                        else:
                            unique_x.append(float(xx)); unique_y.append(float(yy))
                    unique_x = np.asarray(unique_x); unique_y = np.asarray(unique_y)
                    rfd_display = np.interp(x_go, unique_x, unique_y,
                                            left=unique_y[0], right=unique_y[-1])
                    if np.isfinite(rt):
                        rfd_display[x_go < rt] = 0.0
                    ax_rfd.plot(x_go, rfd_display, color=color, lw=2.2,
                                alpha=0.90, label="20-ms local RFD", zorder=3)

            if len(force_ref_x) >= 2:
                sx_fr, sy_fr = _smooth_curve(force_ref_x, force_ref_y)
                ax_force.plot(sx_fr, sy_fr, color=PLOT_COLORS["threshold"],
                              lw=2.0, label="Force reference")
            elif force_ref_x:
                ax_force.plot(force_ref_x, force_ref_y, "o",
                              color=PLOT_COLORS["threshold"], ms=5,
                              label="Force reference")

            if len(cp_x) >= 2:
                sx_cp, sy_cp = _smooth_curve(cp_x, cp_y)
                ax_rfd.plot(sx_cp, sy_cp, color=color, lw=2.2, alpha=0.95,
                            zorder=6, label="Checkpoint local RFD")
            elif cp_x:
                ax_rfd.plot(cp_x, cp_y, "o", color=color, ms=5,
                            zorder=6, label="Checkpoint local RFD")

            sx, sy = _smooth_curve(th_x, th_trigger)
            if sx.size:
                ax_rfd.plot(sx, sy, color=PLOT_COLORS["threshold"], lw=2.5,
                            label=f"Adaptive trigger threshold ({SLOPE_TRIGGER_MODE})",
                            zorder=4)

            ax_rfd.set_ylabel("Local RFD (MVC/s)")
            ax_rfd.set_title(
                "20-ms sliding RFD across full recorded epoch; "
                "shaded = adaptive trigger window only"
            )

        # ------------------------- stimulation marker ----------------------
        if bool(summary.get("right_triggered", False)):
            stim = summary.get("right_trigger_from_go_s", np.nan)
            if np.isfinite(stim):
                for ax in (ax_force, ax_rfd):
                    ax.axvline(stim, color=PLOT_COLORS["trigger"], lw=3, alpha=0.9)
                ax_force.annotate("STIM", xy=(stim, PLOT_FORCE_LIMITS[1] * 0.93),
                                  color=PLOT_COLORS["trigger"], fontsize=10,
                                  fontweight="bold", ha="center")

        ax_force.set_xlim(-0.2, 2.0)
        ax_force.set_ylim(-0.10, 1.50)
        ax_force.set_ylabel("Force (fraction of MVC)")
        ax_force.set_title(
            f"Trial {int(trial_number)} - right handle ({X_NUMBER}, {SESSION})"
        )
        _style_axis(ax_force); ax_force.legend(loc="upper right")

        ax_rfd.axhline(0, color="black", lw=0.8)
        ax_rfd.set_xlim(-0.2, 2.0)
        if is_fixed:
            vals = [summary.get("right_fixed_early_slope", np.nan),
                    summary.get("right_fixed_slope_threshold", np.nan)]
            vals = np.asarray([v for v in vals if np.isfinite(v)], float)
            if vals.size:
                pad = max(1.0, 0.25 * max(1.0, np.nanmax(np.abs(vals))))
                lo = min(0.0, float(np.nanmin(vals) - pad))
                hi = float(np.nanmax(vals) + pad)
                if hi <= lo: hi = lo + 1.0
                ax_rfd.set_ylim(lo, hi)
        else:
            ax_rfd.set_ylim(-25.0, 25.0)
        ax_rfd.set_xlabel("Time from GO cue (s)")
        _style_axis(ax_rfd); ax_rfd.legend(loc="upper right")

        trig = bool(summary.get("right_triggered", False))
        cp = summary.get("right_trigger_checkpoint_ms", np.nan)
        if is_fixed:
            win_ms = 1000.0 * float(summary.get(
                "right_fixed_slope_window_s", FIXED_SLOPE_WINDOW_S))
            info = (
                "Right RT " + _label(summary.get("right_reaction_time_s"), "{:.3f}")
                + " s | fixed slope 0-" + f"{win_ms:.1f}ms "
                + _label(summary.get("right_fixed_early_slope"))
                + " | trig " + str(trig)
                + (f" @ {int(cp)} ms" if trig and np.isfinite(cp) else "")
            )
        else:
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
        path = os.path.join(
            individual_dir,
            f"trial_{int(trial_number):03d}_{X_NUMBER}_{SESSION}_{date_str}.png"
        )
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

def _actual_trigger_ms(row):
    """Return the recorded stimulation latency from force onset in milliseconds."""
    trig_s = row.get("right_trigger_from_onset_s", np.nan)
    if np.isfinite(trig_s):
        return float(trig_s) * 1000.0
    cp = row.get("right_trigger_checkpoint_ms", np.nan)
    return float(cp) if np.isfinite(cp) else np.nan


def _heatmap_trigger_x(row, checkpoint_ms):
    """Continuous heatmap x-position for a recorded TMS trigger.

    Uses the actual trigger latency from force onset when available. This is
    essential for FIXED_EARLY_SLOPE, whose trigger does not necessarily land
    on the legacy checkpoint grid.
    """
    cps = np.asarray(checkpoint_ms, dtype=float)
    if cps.size == 0:
        return np.nan

    trigger_ms = _actual_trigger_ms(row)
    if not np.isfinite(trigger_ms):
        return np.nan

    if cps.size == 1:
        return 0.0 if abs(trigger_ms - cps[0]) < 1e-9 else np.nan

    if trigger_ms < cps[0] or trigger_ms > cps[-1]:
        return np.nan

    return float(np.interp(trigger_ms, cps, np.arange(cps.size, dtype=float)))


def _row_uses_fixed_rfd(row):
    return str(row.get("right_rfd_trigger_algorithm", "")) == "FIXED_EARLY_SLOPE"


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
    if "right_triggered" in df_trials.columns:
        for i, row in df_trials.reset_index(drop=True).iterrows():
            if bool(row.get("right_triggered", False)):
                star_x = _heatmap_trigger_x(row, cps)
                if np.isfinite(star_x):
                    ax.scatter(star_x, i, marker="*", s=110, facecolors="none",
                               edgecolors="white", linewidths=1.4, zorder=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9); cbar.set_label("Local RFD (MVC/s)")
    ax.set_title("Right-handle RFD heatmap (star = actual TMS time)")
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "RFD_Heatmap.png"))

def plot_trigger_map(df_trials, out_dir):
    """Trial number versus actual stimulation latency from force onset."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    fig, ax = plt.subplots(figsize=(10, 6))

    if "right_triggered" in df_trials.columns:
        xs, ys = [], []
        for _, row in df_trials.reset_index(drop=True).iterrows():
            if bool(row.get("right_triggered", False)):
                trig_ms = _actual_trigger_ms(row)
                if np.isfinite(trig_ms):
                    xs.append(float(row.get("trial", len(xs)+1)))
                    ys.append(trig_ms)
        if ys:
            ax.scatter(xs, ys, s=75, color=PLOT_COLORS["right"],
                       label="Right trigger")

    ax.set_xlabel("Trial")
    ax.set_ylabel("Actual TMS latency from force onset (ms)")
    if len(df_trials):
        ax.set_xlim(0.5, max(TOTAL_TRIALS, int(df_trials["trial"].max())) + 0.5)
    _style_axis(ax)
    ax.legend(loc="best")
    ax.set_title("Right-handle stimulation timing across trials")
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "Trigger_Map.png"))



def plot_threshold_evolution(df_trials, out_dir, rfd_limits=None):
    """Adaptive RFD threshold evolution; representation follows the algorithm used."""
    if df_trials.empty:
        return
    summary_dir = os.path.join(out_dir, "Summary")
    fixed_mask = (
        df_trials.get("right_rfd_trigger_algorithm", pd.Series("", index=df_trials.index))
        .astype(str).eq("FIXED_EARLY_SLOPE")
    )

    if fixed_mask.any() and (~fixed_mask).sum() == 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        trials = pd.to_numeric(df_trials["trial"], errors="coerce").to_numpy(dtype=float)
        ref = pd.to_numeric(df_trials.get("right_fixed_slope_threshold", np.nan),
                            errors="coerce").to_numpy(dtype=float)
        if SLOPE_TRIGGER_MODE == "BELOW":
            decision = ref * (1.0 - TRIGGER_MARGIN_FRACTION)
        else:
            decision = ref * (1.0 + TRIGGER_MARGIN_FRACTION)
        ax.plot(trials, ref, "o-", lw=2.2, label="Rolling reference slope")
        ax.plot(trials, decision, "s--", lw=2.2,
                label=f"Decision threshold ({SLOPE_TRIGGER_MODE})")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Adaptive early-slope threshold (MVC/s)")
        ax.set_title("Fixed Early Slope threshold evolution")
        ax.set_xlim(0.5, max(TOTAL_TRIALS, int(np.nanmax(trials))) + 0.5)
        if rfd_limits is not None:
            ax.set_ylim(*rfd_limits)
        _style_axis(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, os.path.join(summary_dir, "Threshold_Evolution.png"))
        return

    # Sliding/checkpoint representation.
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("viridis")
    cps = _checkpoint_ms()
    _, th = _trigger_threshold_matrix(df_trials, "right")
    trials = df_trials["trial"].to_numpy()
    for j, cp in enumerate(cps):
        color = cmap(j / max(1, len(cps) - 1))
        ax.plot(trials, th[:, j], "o-", ms=3.2, lw=1.5,
                color=color, label=f"{cp} ms")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Adaptive trigger threshold (MVC/s)")
    ax.set_xlim(0.5, max(TOTAL_TRIALS, int(df_trials["trial"].max())) + 0.5)
    if rfd_limits is not None:
        ax.set_ylim(*rfd_limits)
    _style_axis(ax)
    ax.set_title("Sliding RFD threshold evolution across trials")
    ax.legend(title="Checkpoint", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    _save_figure(fig, os.path.join(summary_dir, "Threshold_Evolution.png"))



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

    # Mark every recorded stimulation at its actual onset-relative time.
    # The star may lie between checkpoint columns in Fixed Early Slope mode.
    checkpoint_ms = np.asarray([k for k, _ in keys], dtype=float)
    if "right_triggered" in df_trials.columns:
        for i, row in df_trials.reset_index(drop=True).iterrows():
            if bool(row.get("right_triggered", False)):
                star_x = _heatmap_trigger_x(row, checkpoint_ms)
                if np.isfinite(star_x):
                    ax.scatter(star_x, i, marker="*", s=110, facecolors="none",
                               edgecolors="white", linewidths=1.4, zorder=8)

    apply_plot_style(
        ax,
        title=f"Force Heatmap | {STIMULATION_METRIC_MODE}/{SLOPE_TRIGGER_MODE} | star = actual TMS time",
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
    """Normalized task-decision variables across trials, algorithm-aware."""
    if df_trials is None or df_trials.empty:
        return

    n = len(df_trials)
    trials = pd.to_numeric(df_trials["trial"], errors="coerce").to_numpy(dtype=float)
    rfd_ratio = np.full(n, np.nan, dtype=float)
    force_ratio = np.full(n, np.nan, dtype=float)

    for i, (_, row) in enumerate(df_trials.reset_index(drop=True).iterrows()):
        if _row_uses_fixed_rfd(row):
            slope = row.get("right_fixed_early_slope", np.nan)
            ref = row.get("right_fixed_slope_threshold", np.nan)
            if np.isfinite(slope) and np.isfinite(ref) and ref != 0:
                rfd_ratio[i] = float(slope) / float(ref)

            # Fixed mode records the force/reference actually used at the decision.
            frc = row.get("right_trigger_force", np.nan)
            fref = row.get("right_trigger_force_reference", np.nan)
            if np.isfinite(frc) and np.isfinite(fref) and fref != 0:
                force_ratio[i] = float(frc) / float(fref)
        else:
            cp = int(round(RFD_EPOCH_S * 1000))
            rfd = row.get("right_rfd", np.nan)
            rth = row.get("right_rfd_threshold", np.nan)
            frc = row.get(f"right_checkpoint_force_{cp}ms", np.nan)
            fref = row.get(f"right_force_reference_{cp}ms", np.nan)
            if np.isfinite(rfd) and np.isfinite(rth) and rth != 0:
                rfd_ratio[i] = float(rfd) / float(rth)
            if np.isfinite(frc) and np.isfinite(fref) and fref != 0:
                force_ratio[i] = float(frc) / float(fref)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(trials, rfd_ratio, linewidth=2.5, marker="o",
            label="RFD / adaptive reference")
    ax.plot(trials, force_ratio, linewidth=2.5, marker="s",
            label="Force / adaptive reference")
    ax.axhline(1.0, linewidth=1.5, linestyle="--", label="Reference = 1.0")

    if "right_triggered" in df_trials.columns:
        trig = df_trials["right_triggered"].fillna(False).astype(bool).to_numpy()
        # Always mark triggered trials, even when one normalized series is NaN.
        for i in np.flatnonzero(trig):
            candidates = [rfd_ratio[i], force_ratio[i]]
            candidates = [v for v in candidates if np.isfinite(v)]
            if candidates:
                y = candidates[0]
            else:
                # Place a visible star on the reference line rather than silently
                # omitting a confirmed trigger because a reporting ratio is absent.
                y = 1.0
            ax.scatter(trials[i], y, s=110, marker="*",
                       color=PLOT_COLORS["trigger"], zorder=8,
                       label="Triggered trial" if i == np.flatnonzero(trig)[0] else None)

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
    global RFD_TRIGGER_ALGORITHM, PARTICIPANT_FEEDBACK_ENABLED, PARTICIPANT_FEEDBACK_METRIC, BASELINE_REVIEW_MODE
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
    selected_rfd_algorithm=RFD_TRIGGER_ALGORITHM; selected_feedback_enabled=PARTICIPANT_FEEDBACK_ENABLED
    selected_feedback_metric=PARTICIPANT_FEEDBACK_METRIC; selected_review_mode=BASELINE_REVIEW_MODE

    while True:
        # Startup interface: calibration is launched explicitly from the menu.
        # The task button remains disabled until calibration has completed.
        (
            calibration, selected_trigger_mode, selected_metric_mode,
            baseline_result, show_main_task_loadbar,
            selected_subject, selected_session, selected_block,
            selected_rfd_algorithm, selected_feedback_enabled,
            selected_feedback_metric, selected_review_mode,
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
            rfd_algorithm=selected_rfd_algorithm, feedback_enabled=selected_feedback_enabled,
            feedback_metric=selected_feedback_metric, baseline_review_mode=selected_review_mode,
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
        RFD_TRIGGER_ALGORITHM=selected_rfd_algorithm; PARTICIPANT_FEEDBACK_ENABLED=selected_feedback_enabled
        PARTICIPANT_FEEDBACK_METRIC=selected_feedback_metric; BASELINE_REVIEW_MODE=selected_review_mode
        # target rectangle may have changed in the GUI
        right_rect.size=create_target_rect(HEIGHT).size; place_targets(right_rect,WIDTH,HEIGHT)
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
        right_monitor.rfd_algorithm = RFD_TRIGGER_ALGORITHM
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
        baseline_fixed_slope=float(baseline_result.get("fixed_slope_mean",np.nan)); baseline_fixed_force=float(baseline_result.get("fixed_force_mean",np.nan))
        baseline_fixed_window=float(baseline_result.get("fixed_slope_window_s",FIXED_SLOPE_WINDOW_S))
        # Peak-latency history starts from the accepted baseline MEAN so Trial 1
        # uses the baseline-derived selected quarter endpoint. Main Task trials then
        # replace these seeds one-by-one in the rolling three-trial mean.
        right_peak_latency_history=[baseline_mean_plateau_s]*SLOPE_REFERENCE_TRIALS
        # Keep raw accepted baseline trials so previous slopes/forces can be
        # recomputed over the SAME adaptive window selected for each new trial.
        _base_trials=baseline_result.get("trials",[])
        _base_keep=np.asarray(baseline_result.get("included_trials",[]),dtype=bool)
        right_fixed_reference_trials=[
            {"t":np.asarray(tr["t"],dtype=float).copy(),"force":np.asarray(tr["force"],dtype=float).copy()}
            for tr,keep in zip(_base_trials,_base_keep) if keep
        ]
        current_fixed_window_s=baseline_fixed_window

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
                    draw_active_target(screen, right_rect, RED if USE_RED_CUE else NEUTRAL_GRAY)
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
                    current_fixed_window_s,current_fixed_trigger_target_s,current_predicted_peak_s=adaptive_fixed_timing(
                        right_peak_latency_history,
                        fallback_peak_s=baseline_mean_plateau_s,
                        mode=FIXED_TIMING_MODE
                    )
                    # Compare current slope against previous trials recomputed
                    # over exactly this same adaptive window.
                    fixed_thr=fixed_slope_from_reference_trials(
                        right_fixed_reference_trials,current_fixed_window_s,SLOPE_REFERENCE_TRIALS
                    )
                    fixed_force_ref=fixed_force_from_reference_trials(
                        right_fixed_reference_trials,current_fixed_window_s,FORCE_REFERENCE_TRIALS
                    )
                    right_monitor.arm(
                        go_start_time,right_thresholds,right_force_references,
                        fixed_thr,fixed_force_ref,
                        fixed_slope_window_s=current_fixed_window_s,
                        fixed_trigger_delay_s=max(0.0,current_fixed_trigger_target_s-current_fixed_window_s)
                    )

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
                    if RFD_TRIGGER_ALGORITHM == "FIXED_EARLY_SLOPE":
                        print(
                            f"Adaptive fixed slope = 0-{current_fixed_window_s*1000:.1f} ms | "
                            f"predicted peak = {current_predicted_peak_s*1000:.1f} ms | "
                            f"planned TMS = {current_fixed_trigger_target_s*1000:.1f} ms after onset | "
                            f"timing mode = {FIXED_TIMING_MODE} (TMS = fit end + {FIXED_TRIGGER_DELAY_S*1000:.0f} ms)"
                        )
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
                    draw_active_target(screen, right_rect, GREEN if elapsed_state < current_go_duration else NEUTRAL_GRAY)

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
                    # Recording-only period: neutral target remains visible.
                    draw_cursor_track(screen, right_rect)
                    draw_active_target(screen, right_rect, NEUTRAL_GRAY)
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
                    _peak_lat=trial_metrics.get("right_ballistic_peak_latency_s",np.nan)
                    if np.isfinite(_peak_lat) and _peak_lat>0:
                        right_peak_latency_history.append(float(_peak_lat))
                    else:
                        # Preserve trial order: invalid timing prevents a bad trial
                        # from silently redefining the next adaptive window.
                        right_peak_latency_history.append(np.nan)
                    _trial_mask=(current_time_array>=go_start_time)&(current_time_array<=go_start_time+ONSET_SEARCH_WINDOW_S+POST_ONSET_WINDOW_S)
                    right_fixed_reference_trials.append({
                        "t":current_time_array[_trial_mask].copy(),
                        "force":current_right_force[_trial_mask].copy()
                    })
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
                    draw_cursor_track(screen, right_rect)
                    draw_active_target(screen, right_rect, NEUTRAL_GRAY)
                    draw_force_cursor(screen, right_force, right_rect)

                if elapsed_state >= iti_duration:
                    state = "PREP"
                    state_start = time.perf_counter()
                    cue_trigger_sent = False
                    current_prep_duration = np.random.uniform(
                        PREP_DURATION_S, PREP_DURATION_S + PREP_JITTER_S
                    )

            if draw_now:
                draw_participant_feedback(screen, trial_log, PARTICIPANT_FEEDBACK_METRIC)
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
            "rfd_trigger_algorithm": np.array([RFD_TRIGGER_ALGORITHM], dtype=object),
            "fixed_slope_window_s": np.array([FIXED_SLOPE_WINDOW_S]),
            "participant_feedback_enabled": np.array([float(PARTICIPANT_FEEDBACK_ENABLED)]),
            "participant_feedback_metric": np.array([PARTICIPANT_FEEDBACK_METRIC], dtype=object),
            "baseline_review_mode": np.array([BASELINE_REVIEW_MODE], dtype=object),
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
            "baseline_endpoint_definition": np.array(["ballistic force-rate turning point / local maximum"], dtype=object),
            "baseline_included_trials": np.asarray(baseline_result.get("included_trials",[]),dtype=float),
            "baseline_fixed_slope_mean": np.array([baseline_result.get("fixed_slope_mean",np.nan)]),
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
