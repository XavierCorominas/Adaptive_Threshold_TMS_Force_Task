# **Adaptive Thresholding TMS Grip-Force Task**

## *Short repository overview*

A Python-based force-production task for delivering adaptive, movement-locked TMS stimulation based on real-time force development. The task records grip-hand force at 500 Hz, detects movement onset, and compares the developing force contraction, force rate development or both with specific time-references derived from baseline and recent trials.
 <img width="1020" height="279" alt="image" src="https://github.com/user-attachments/assets/d3b70ec3-835e-47f4-be1c-00a0e674ef6f" />

## *How the task works*

•	Force calibration — The input device is calibrated to the participant so force is represented relative to maximum voluntary contraction (MVC).

•	Baseline — Ten no-stimulation baseline trials establish the initial RFD and force profiles and help individualize the useful early stimulation window.

•	Movement onset — Force is smoothed over 3 samples. Onset is detected when force reaches 0.05 MVC and remains there for 5 consecutive samples.

•	Real-time monitoring — Adaptive checking begins 20 ms after detected movement onset and is repeated every 6 ms. At each checkpoint, local RFD is estimated by linear regression over the preceding 20 ms of smoothed force. Because the checkpoint interval (6 ms) is shorter than the RFD window (20 ms), consecutive windows overlap by 14 ms, or 70%. This provides frequent, near-continuous monitoring of the rapidly changing early force trajectory while each RFD estimate still uses a substantially longer 20-ms window rather than relying on only a few highly noise-sensitive samples.

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

 .
 .
 .
 .



## *Output Results and Figures*

Short guide to the files, metrics, and plots generated after each block

### What the task produces

After a Main Task block, the program performs post-hoc processing and saves the results inside a participant/session/block/task-version folder. The main outputs are an Excel workbook, a MATLAB .mat file, individual-trial PNG figures, and session-level summary PNG figures. The task also prints a short block summary to the console, including achieved sampling rate, trigger count, mean 80-ms RFD, reaction time, onset failures, target accuracy, time in target, and mean force error.

## Excel workbook

The Excel workbook is the most convenient output for inspecting and analysing the experiment. It separates the data into several worksheets:

### Sheet	What it contains

Trial_Summary	One row per trial. Contains trial-level RFD, reaction time, onset status, checkpoint RFD/force values, adaptive references/thresholds, trigger information and target-performance metrics.

Trial_TimeSeries	Sample-by-sample trial data, including force and smoothed force, time relative to GO/onset, and the local-RFD trace used for detailed visualization.

Calibration	Calibration values and key task settings, including resting baseline, maximum deflection, target force, RFD window, trigger mode and individualized checking endpoint.

Baseline_Thresholds	Baseline RFD threshold and force reference at each adaptive checkpoint.

Baseline_Plateau	Time-to-plateau for each baseline trial, used to determine the individualized adaptive checking window.
Raw_Data	Continuous recorded time and right-handle force for the block.

### MATLAB output

A matching .mat file is saved for MATLAB-based analysis. It contains the raw force/time arrays, calibration values, baseline profiles, adaptive checkpoint settings, task configuration, and trial-level result columns. This provides a direct route for custom signal processing or statistical analysis outside Python.

### Individual trial figures

A separate publication-style PNG is generated for every completed trial in the SingleTrials folder. Each figure contains a force panel and a local-RFD panel on a common GO-aligned time axis.

### Force panel

The force panel shows the raw/light force trace together with the smoothed force trajectory. It marks the GO cue, the 0.05-MVC onset threshold, and—when using BELOW mode—the force ceiling above which corrective stimulation is suppressed. Adaptive force references and stimulation timing are also displayed when available.

### RFD panel

The RFD panel shows the continuous 20-ms sliding local-RFD trace across the recorded epoch, while visually distinguishing the shorter interval in which adaptive stimulation was actually allowed. Checkpoint RFD values and the adaptive trigger threshold are overlaid. If stimulation occurred, its timing is marked. A text summary reports reaction time, the primary (arbitrary time reference normally in the middle of the force curve) 80-ms RFD, whether stimulation occurred, and the trigger checkpoint.

### Main session-level figures

The Summary folder provides plots for understanding behavior across the whole block rather than inspecting trials individually.

## Figure	Interpretation

Mean_RFD_Profile.png	Mean local RFD across checkpoints, ±1 SD, together with the mean adaptive trigger threshold. Shows the typical early RFD profile for the session.

RFD_Spaghetti.png	Plots every trial's checkpoint RFD trajectory, plus the session mean, variability and threshold. Useful for seeing trial-to-trial variability.

RFD_Heatmap.png	Trials × checkpoints heatmap of local RFD. Trigger checkpoints are marked with stars, making temporal patterns easy to identify.

Trigger_Map.png	Shows which trials triggered and the onset-relative checkpoint at which stimulation occurred.
Threshold_Evolution.png	Shows how the adaptive RFD trigger threshold at each checkpoint changes across trials as the rolling reference is updated.

Force summary figures	Equivalent session summaries are generated for checkpoint force: mean profile, individual trajectories, heatmap and force-reference evolution.

Combined RFD–Force figures	Additional plots jointly visualize RFD and force, allowing the two components of COMBINED triggering to be interpreted together.

### Additional overview figures

Trial_RFD.png summarizes the primary reporting checkpoint (80 ms) across trials together with its adaptive reference. Onset_Aligned_Forces.png overlays all smoothed force traces after shifting each trial so detected movement onset is t = 0. This is useful for evaluating the consistency of the force rise independently of reaction-time differences.

### How to interpret the outputs

•	Use Trial_Summary for trial-level statistics and determining exactly why/when stimulation occurred.

•	Use Trial_TimeSeries when the full shape of an individual force or RFD trace is required.

•	Use the SingleTrials figures for quality control: onset detection, force trajectory, RFD behavior, threshold crossing and stimulation timing can be inspected visually.

•	Use Mean_RFD_Profile and RFD_Spaghetti to understand the overall RFD behavior of the participant.

•	Use RFD_Heatmap and Trigger_Map to identify where stimulation tends to occur across trials and onset-relative time.

•	Use Threshold_Evolution to verify that the adaptive reference changes appropriately as new trials enter the rolling history.

•	Use the force and combined summaries when interpreting COMBINED stimulation, because an RFD threshold crossing alone does not necessarily imply that the full trigger criterion was satisfied.

### Output folder structure

ADAPT_BK/

└── Subject/

    └── Session/
    
        └── Block/
        
            └── TaskVersion/
            
                ├── game_output_....xlsx
                
                ├── game_output_....mat
                
                ├── Trial_RFD.png
                
                ├── Onset_Aligned_Forces.png
                
                ├── SingleTrials/
                
                │   └── trial_001_....png
                
                └── Summary/
                
                    ├── Mean_RFD_Profile.png
                    
                    ├── RFD_Spaghetti.png
                    
                    ├── RFD_Heatmap.png
                    
                    ├── Trigger_Map.png
                    
                    ├── Threshold_Evolution.png
                    
                    └── Force / combined summary figures
                    





