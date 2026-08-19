# **Adaptive Thresholding TMS Force Task**

## *Short repository overview*

A Python-based force-production task for delivering adaptive, movement-locked TMS stimulation based on real-time force development. The task records grip-hand force at 500 Hz, detects movement onset, and compares the developing force contraction, force rate development or both with specific time-references derived from baseline and recent trials.
 <img width="1020" height="279" alt="image" src="https://github.com/user-attachments/assets/d3b70ec3-835e-47f4-be1c-00a0e674ef6f" />

## *How the task works*

•	Force calibration — The input device is calibrated to the participant so force is represented relative to maximum voluntary contraction (MVC).

•	Baseline — Ten no-stimulation baseline trials establish the initial RFD and force profiles and help individualize the useful early stimulation window.

•	Movement onset — Force is smoothed over 3 samples. Onset is detected when force reaches 0.05 MVC and remains there for 5 consecutive samples.

•	Real-time monitoring — Adaptive checking starts 20 ms after detected onset and repeats every 6 ms. At each checkpoint, local RFD is calculated from a linear regression over the preceding 20 ms of smoothed force.

•	Adaptive decision — Current RFD and/or force are compared with checkpoint-specific references from the previous 3 profiles. The operator can select BELOW or ABOVE behavior, and RFD_ONLY, FORCE_ONLY, or COMBINED decision logic. COMBINED logic is recommended to avoid false positives.

•	Trigger — The first eligible checkpoint satisfying the selected rule sends the stimulation trigger. The individualized checking window is capped at 180 ms after onset.

 <img width="1005" height="447" alt="image" src="https://github.com/user-attachments/assets/09be02ba-f254-4751-832d-04c5dc37b1d7" />

## *Adaptive threshold*

The adaptive reference is calculated independently at each checkpoint. V6 uses a 10% margin around that reference: in BELOW mode, current RFD must fall below 90% of the reference; in ABOVE mode, it must exceed 110%. Force uses the same 10% relative margin. COMBINED mode requires both RFD and force criteria to pass at the same checkpoint.
 <img width="975" height="390" alt="image" src="https://github.com/user-attachments/assets/f0653c4e-e2a8-49b3-a3be-968f311d2ab7" />

## *Key values*

Sampling rate	500 Hz

Smoothing	3 samples (~6 ms)

Onset threshold	0.05% MVC

Local RFD window	20 ms

First RFD checkpoint	20 ms after onset

Checkpoint spacing	6 ms

Adaptive history	Previous 3 trials

RFD / force trigger margin	±10%

Maximum checking window	180 ms after onset

## *Why movement-onset locking?*
The adaptive clock starts from detected force onset rather than from the visual GO cue. This prevents reaction-time variability from shifting the RFD measurement window and allows equivalent phases of force development to be compared across trials.

## *Main Python dependencies*
NumPy • Pandas • SciPy • Pygame • Matplotlib • NI-DAQmx (optional for hardware triggering)





