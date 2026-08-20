# **Adaptive Thresholding TMS Grip-Force Task**

## *Short repository overview*

A Python-based ballistic grip-force task for delivering adaptive, movement-locked TMS stimulation based on real-time force development.

The task records right-hand grip force at **500 Hz**, detects movement onset, and compares the participant's developing contraction with adaptive references derived from baseline and recent trials.

Two alternative RFD triggering algorithms are available:

1. **Sliding RFD** — repeated local RFD evaluations at onset-relative checkpoints.
2. **Adaptive Fixed Early Slope** — one onset-anchored linear slope fitted over an individualized portion of the ballistic force rise.

For either algorithm, stimulation can be based on **RFD**, **force**, or a **combination of both**.


---

# **How the task works**

### Force calibration

The input device is calibrated to each participant so force is represented relative to maximum voluntary contraction (**MVC**).

The program first measures the actual resting position of the grip device and then determines the participant's maximum force deflection.

Force during the experiment is therefore expressed as a fraction of calibrated MVC.

---

### Familiarization

An optional familiarization phase allows participants to practice the ballistic grip contraction before baseline acquisition.

Familiarization trials are not used to calculate adaptive thresholds.

The number of familiarization trials can be configured from the startup GUI.

---

### Baseline

Baseline consists of no-stimulation trials used to initialize the adaptive algorithms.

The number of baseline trials is configurable from the GUI.

For every baseline contraction, the task:

- detects movement onset;
- identifies the ballistic force peak;
- calculates onset-to-peak latency;
- calculates early force-development slopes;
- extracts checkpoint RFD and force values;
- stores the force trajectory for subsequent adaptive calculations.

Baseline trials can additionally be reviewed before the final calibration is accepted.

Two review modes are available:

**QC auto-save**

All automatically valid baseline trials are retained and a baseline QC report is saved.

**QC interactive**

Each baseline trial is displayed individually and the experimenter can manually **include or reject** it before the final baseline metrics are calculated.

The baseline review plots use a fixed **0–1 MVC force axis**, allowing trials to be compared directly.

---

# **Movement onset**

All adaptive measurements are aligned to detected **force onset**, not to the visual GO cue.

Force is smoothed over **3 samples**.

Movement onset is accepted when force reaches:

**0.05 MVC**

and remains at or above that level for:

**5 consecutive samples**.

Once detected, movement onset becomes:

**t = 0**

for the adaptive force-development analysis.

This prevents reaction-time variability from shifting the RFD measurement window.

---

# **Ballistic peak detection**

The task is designed for a **ballistic squeeze-and-release contraction**. Participants rapidly increase grip force and then release rather than maintaining a prolonged force plateau.

For this reason, V7 does **not** use a sustained force plateau as the main timing endpoint.

Instead, the program estimates the **ballistic force peak**.

After movement onset, the program searches the early force trajectory and examines the smoothed force-rate signal.

A candidate peak is identified when:

- force has reached a substantial proportion of the trial peak;
- force rate is positive before the candidate;
- force rate becomes non-positive after the candidate.

In other words, the algorithm detects the point where the force trajectory changes from:

**rising → falling**

which corresponds to the ballistic force maximum.

The search is limited to the early post-onset interval.

If a robust direction change cannot be identified, the maximum force within the ballistic search window is used as a fallback.

The resulting:

**onset → peak latency**

is used to individualize adaptive timing.

---

# **Two real-time RFD algorithms**

The operator can select between two different adaptive RFD algorithms from the startup GUI.

## **1. Sliding RFD**

The Sliding RFD algorithm performs repeated evaluations during the early force rise.

Adaptive checking begins:

**20 ms after detected movement onset**

and repeats every:

**6 ms**.

At every checkpoint, local RFD is estimated using a linear regression over the preceding:

**20 ms**

of smoothed force.

Because checkpoints are separated by 6 ms while the RFD window is 20 ms, consecutive windows overlap substantially.

This provides near-continuous monitoring of the rapidly developing force trajectory while avoiding slopes calculated from only a few highly noise-sensitive samples.

The baseline onset-to-peak estimate is also used to individualize how long sliding checking is allowed to continue.

The checking period is capped at:

**180 ms after movement onset**.

---

## **2. Adaptive Fixed Early Slope**

The Fixed Early Slope algorithm uses a fundamentally different strategy.

Instead of repeatedly calculating local RFD at many checkpoints, the task fits **one linear regression from movement onset to an adaptive early-ramp endpoint**.

The endpoint is determined from the participant's predicted onset-to-peak latency.

The operator can select one of three timing modes:

### FIT 1/4

Slope is fitted from onset to:

**25% of predicted onset-to-peak latency**

### FIT 2/4

Slope is fitted from onset to:

**50% of predicted onset-to-peak latency**

### FIT 3/4

Slope is fitted from onset to:

**75% of predicted onset-to-peak latency**

For example, if predicted onset-to-peak latency is:

**60 ms**

then:

- FIT 1/4 → slope window = **0–15 ms**
- FIT 2/4 → slope window = **0–30 ms**
- FIT 3/4 → slope window = **0–45 ms**

The stimulation decision is made after this slope has been calculated.

TMS is scheduled at:

**slope-fit endpoint + configurable delay**

The default delay is:

**5 ms**

but this value can be changed from the startup GUI.

Therefore, with a 60-ms predicted peak:

**FIT 2/4**

0–30 ms slope fit  
→ TMS approximately **35 ms after onset**

and:

**FIT 3/4**

0–45 ms slope fit  
→ TMS approximately **50 ms after onset**

with the default 5-ms delay.

Importantly, stimulation is **not delayed until the next quarter boundary**.

---

# **Adaptive Fixed-window timing across trials**

The Fixed Early Slope window is adaptive.

During baseline, the mean valid:

**movement onset → ballistic peak latency**

is calculated.

The initial Main Task slope window is then:

**mean baseline onset-to-peak latency × selected fraction**

For example:

mean baseline peak latency = 60 ms

FIT 3/4 selected

→ initial slope window = **45 ms**

During the Main Task, the predicted peak latency is updated using the most recent valid trials.

By default, the most recent:

**3 trials**

are used.

Therefore, if the participant's ballistic rise becomes faster or slower, the slope window also changes.

The slope reference from previous trials is recomputed using the **same current adaptive window**.

This ensures that the current trial and previous trials are always compared using equivalent onset-relative slope intervals.

---

# **Adaptive threshold**

The adaptive reference is based on recent completed trials.

By default:

**Previous 3 trials**

are used.

A **10% relative margin** is applied around the adaptive reference.

For RFD or Fixed Early Slope:

### BELOW

Current value must be:

**< 90% of adaptive reference**

### ABOVE

Current value must be:

**> 110% of adaptive reference**

Force uses the same **10% relative margin**.

---

# **Stimulation metric modes**

Three stimulation modes are available from the GUI.

### RFD ONLY

The RFD criterion alone determines whether stimulation occurs.

Depending on the selected RFD algorithm, this means either:

- Sliding local RFD, or
- Adaptive Fixed Early Slope.

### FORCE ONLY

The force trajectory relative to the adaptive force reference determines stimulation.

### COMBINED

Both the RFD and force criteria must be satisfied.

In Sliding mode, RFD and force are evaluated at the same checkpoint.

In Fixed Early Slope mode, the slope is calculated over the adaptive onset-to-fit window and force is evaluated at the corresponding adaptive endpoint.

COMBINED mode can therefore help prevent stimulation based on an RFD difference that is not accompanied by a corresponding difference in the participant's force trajectory.

<img width="975" height="390" alt="image" src="https://github.com/user-attachments/assets/f0653c4e-e2a8-49b3-a3be-968f311d2ab7" />

---

# **Trigger timing**

Only **one adaptive stimulation trigger** can be delivered during a trial.

### Sliding RFD

The first eligible checkpoint satisfying the selected criterion produces the stimulation trigger.

### Fixed Early Slope

Only one adaptive decision is made.

The task waits until:

**adaptive slope endpoint + configured TMS delay**

and then evaluates the adaptive criterion.

If the criterion is satisfied, stimulation is delivered.

If the criterion is not satisfied, no later Sliding-style checks are performed for that Fixed trial.

---

# **Target display**

The target is defined directly in MVC units.

Both its:

- vertical force position;
- vertical tolerance / box height

can be configured from the startup GUI.

The default target is:

**0.85 MVC ± 0.10 MVC**

The target box remains visible during the normal trial sequence.

### With red cue enabled

**Grey → Red → Green → Grey**

### Without red cue

**Grey → Green → Grey**

The green target indicates the GO period.

The force cursor and target use the same MVC-based vertical scale.

---

# **Participant performance feedback**

Optional real-time participant feedback can be enabled from the GUI.

When enabled, part of the right side of the participant screen is reserved for a trial-history plot.

A new point is added after each completed trial.

The operator can select the feedback metric.

Available metrics include:

### Peak force

**Unit:** MVC

### Adaptive RFD

For Fixed Early Slope trials, this is the actual slope calculated over that trial's adaptive onset-to-window interval.

**Unit:** MVC/s

### RFD 0–80 ms

A reporting RFD calculated from movement onset to 80 ms.

**Unit:** MVC/s

### Reaction time

**Unit:** ms

### Target error

**Unit:** MVC

The plot explicitly displays the x- and y-axis units.

The feedback panel also indicates whether the most recent performance improved or worsened relative to the previous valid trial.

---

# **Key values**

| Parameter | Default |
|---|---:|
| Sampling rate | 500 Hz |
| Display rate | 64 FPS |
| Smoothing | 3 samples (~6 ms) |
| Onset threshold | 0.05 MVC |
| Onset confirmation | 5 samples |
| Onset search window | 850 ms after GO |
| Adaptive history | Previous 3 trials |
| RFD / force margin | ±10% |
| Sliding RFD window | 20 ms |
| First Sliding checkpoint | 20 ms after onset |
| Sliding checkpoint spacing | 6 ms |
| Maximum Sliding checking window | 180 ms |
| Ballistic peak search | 350 ms |
| Ballistic direction confirmation | 8 ms |
| Minimum ballistic peak | 0.15 MVC |
| Fixed timing options | 1/4, 2/4, 3/4 peak latency |
| Fixed TMS delay | 5 ms default, GUI configurable |
| Target | 0.85 ± 0.10 MVC |
| Familiarization trials | 5 default, GUI configurable |
| Baseline trials | 10 default, GUI configurable |
| Main Task trials | 15 default, GUI configurable |
| Feedback history | Up to 30 trials |

---

# **Why movement-onset locking?**

The adaptive clock starts from detected force onset rather than from the visual GO cue.

Reaction time can vary substantially between trials.

If RFD were calculated at a fixed time after the GO cue, the same analysis window could represent very different physiological phases of the contraction.

Movement-onset locking instead allows equivalent portions of the force rise to be compared across trials.

This is particularly important for the Adaptive Fixed Early Slope algorithm because both:

**slope window**

and

**stimulation timing**

are defined relative to the actual beginning of the ballistic contraction.

---

# **Startup GUI**

The startup interface allows the experimenter to configure the task without modifying the source code.

Configurable parameters include:

- participant / session / block identifiers;
- number of familiarization trials;
- number of baseline trials;
- number of Main Task trials;
- target force level;
- target tolerance / box height;
- PREP, GO and pause durations;
- timing jitter;
- Sliding vs Fixed Early Slope algorithm;
- FIT 1/4, FIT 2/4 or FIT 3/4 timing;
- TMS delay after the Fixed slope fit;
- BELOW vs ABOVE triggering;
- RFD_ONLY, FORCE_ONLY or COMBINED stimulation;
- red cue on/off;
- participant feedback on/off;
- feedback metric;
- baseline QC auto-save vs interactive review;
- Main Task loadbar.

The interface uses a wide two-column layout so it is better suited to a secondary monitor with substantial horizontal width but limited vertical height.

---

# **Main Python dependencies**

NumPy • Pandas • SciPy • Pygame • Matplotlib • NI-DAQmx

NI-DAQmx is optional when running the task without the stimulation/DAQ hardware.

---

# **Output Results and Figures**

## *Short guide to the files, metrics, and plots generated after each block*

After a Main Task block, the program performs post-hoc processing and saves the results inside a:

**participant / session / block / task-version**

folder.

Outputs include:

- Excel workbook;
- MATLAB `.mat` file;
- individual-trial figures;
- session-level summary figures;
- baseline quality-control reports;
- raw and processed force information.

---

# **Excel workbook**

The Excel workbook provides trial-level, time-series, calibration, baseline, and raw-data outputs for subsequent analysis.

### Trial_Summary

One row per trial.

Contains trial-level information including:

- movement onset;
- reaction time;
- peak force;
- RFD metrics;
- adaptive Fixed Early Slope;
- checkpoint RFD values;
- checkpoint force values;
- adaptive references;
- trigger decision;
- actual stimulation timing;
- target-performance metrics;
- algorithm and timing information.

### Trial_TimeSeries

Sample-by-sample trial data used for detailed visualization and custom analysis.

Includes force, smoothed force and timing information relative to the GO cue and movement onset.

### Calibration

Contains participant/device calibration values and relevant task configuration.

### Baseline outputs

Baseline results contain the measurements used to initialize the adaptive task, including:

- included/rejected trial information;
- ballistic onset-to-peak latency;
- adaptive checking information;
- RFD reference profiles;
- force reference profiles;
- Fixed Early Slope calibration information.

### Raw_Data

Continuous recorded time and right-hand force for the block.

---

# **Baseline QC reports**

Baseline quality-control results are additionally saved to a dedicated report folder.

The report contains trial-by-trial QC plots and summary metrics.

In interactive mode, these outputs reflect the experimenter's final include/reject decisions.

Baseline plots display force on a common:

**0–1 MVC y-axis**

so contractions can be compared visually without changes in axis scaling.

---

# **MATLAB output**

A matching `.mat` file is generated for MATLAB-based analysis.

It contains the recorded force/time arrays, calibration information, adaptive settings, baseline information and trial-level results.

This provides a direct route for custom signal processing and statistical analysis outside Python.

---

# **Individual trial figures**

A publication-style PNG is generated for every completed Main Task trial in the `SingleTrials` folder.

The exact RFD representation depends on the algorithm used.

## Sliding RFD trials

The RFD panel shows the continuous/local 20-ms Sliding RFD trace together with checkpoint values and adaptive threshold information.

## Fixed Early Slope trials

The figure instead represents the actual:

**movement onset → adaptive endpoint**

linear fit.

The adaptive slope reference and decision threshold correspond to this Fixed fit rather than to a Sliding checkpoint threshold.

The adaptive fit endpoint and actual stimulation timing are displayed when applicable.

This prevents Fixed Early Slope trials from being visually misinterpreted as Sliding RFD trials.

---

# **Main session-level figures**

The `Summary` folder provides plots for understanding behavior across the complete block.

### Mean_RFD_Profile.png

Summarizes the average onset-relative RFD behavior across the session.

### RFD_Spaghetti.png

Displays trial-to-trial RFD trajectories and variability.

### RFD_Heatmap.png

Trials × onset-relative time/checkpoint representation of RFD.

Triggered trials are marked with stars.

For Fixed Early Slope trials, trigger markers use the **actual recorded stimulation latency**, allowing triggers that occur between the legacy Sliding checkpoint columns to be displayed correctly.

### Force_Heatmap.png

Equivalent heatmap representation for force.

Trigger markers likewise use the actual stimulation timing.

### Trigger_Map.png

Displays the **actual TMS latency from movement onset** for triggered trials.

It is therefore not restricted to the Sliding checkpoint grid.

### Threshold_Evolution.png

The representation depends on the selected RFD algorithm.

For Sliding RFD, it shows checkpoint-specific adaptive thresholds across trials.

For Fixed Early Slope, it shows the rolling Fixed slope reference and corresponding BELOW/ABOVE decision threshold.

### Force summary figures

Equivalent session-level summaries describe force behavior and adaptive force references.

### Combined RFD–Force figures

Additional figures jointly visualize the RFD and force components used by COMBINED triggering.

### RFD_Force_Normalized_Overview.png

Displays RFD and force relative to their adaptive references.

For Fixed trials, the RFD quantity is the adaptive Fixed Early Slope rather than a legacy Sliding/checkpoint RFD value.

Confirmed stimulation trials are marked even when a legacy reporting quantity is unavailable.

---

# **Additional overview figures**

### Trial_RFD.png

Provides trial-by-trial RFD information.

The task retains the conventional onset-to-80-ms RFD as a reporting metric, while the adaptive Fixed Early Slope is stored separately.

These quantities should not be interpreted as the same metric.

### Onset_Aligned_Forces.png

Overlays smoothed force trajectories after shifting every trial so:

**detected movement onset = 0**

This allows the shape and timing of the ballistic force rise to be compared independently of reaction-time differences.

---

# **How to interpret the outputs**

- Use **Trial_Summary** to determine exactly why and when stimulation occurred.

- Use **Trial_TimeSeries** when the complete shape of a force trajectory is required.

- Use **SingleTrials** for visual quality control of onset detection, ballistic force development, adaptive slope/RFD behavior, threshold crossing and stimulation timing.

- Use the **baseline QC report** to verify the trials used to initialize the adaptive algorithms.

- Use **RFD_Heatmap**, **Force_Heatmap** and **Trigger_Map** to examine where stimulation occurred across trials and onset-relative time.

- Use **Threshold_Evolution** to verify that the adaptive reference changes as new trials enter the rolling history.

- When using **Fixed Early Slope**, interpret the adaptive Fixed slope rather than the conventional 80-ms RFD as the real-time RFD decision variable.

- When using **COMBINED** stimulation, inspect both the RFD/slope and force components because satisfying only one criterion is insufficient to trigger.

- Use **actual onset-relative trigger time** when interpreting Fixed Early Slope stimulation rather than assuming stimulation occurred on the Sliding RFD checkpoint grid.

---

# **Output folder structure**

ADAPT_BK/

└── Subject/

    └── Session/

        └── Block/

            └── TaskVersion/

                ├── game_output_....xlsx

                ├── game_output_....mat

                ├── Trial_RFD.png

                ├── Onset_Aligned_Forces.png

                ├── RFD_Force_Normalized_Overview.png

                ├── SingleTrials/

                │   └── trial_001_....png

                └── Summary/

                    ├── Mean_RFD_Profile.png

                    ├── RFD_Spaghetti.png

                    ├── RFD_Heatmap.png

                    ├── Force_Heatmap.png

                    ├── Trigger_Map.png

                    ├── Threshold_Evolution.png

                    └── Force / combined summary figures

Baseline quality-control reports are additionally saved during the baseline phase.
