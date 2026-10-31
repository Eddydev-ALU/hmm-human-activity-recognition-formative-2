# Modeling Human Activity States Using Hidden Markov Models

Formative 2: human activity recognition from smartphone accelerometer and gyroscope data with a **Hidden Markov Model implemented from scratch in numpy** (Viterbi and Baum-Welch).

**Use case:** a phone or wearable monitor for older adults living alone. It infers the wearer's activity (standing, walking, jumping, still) and alerts a caregiver on long inactivity (*still*), or on sudden, violent, high-energy movement (proxied by *jumping*) that could be a fall or a seizure.

## Author and devices

Individual assignment by **Eddy Irasetsa**. A friend (not a group member) shared recordings from their own phone, used only as extra unseen test data.

| Person | Phone | App | Configured rate | Detected rate (timestamps) | Harmonised to |
|---|---|---|---|---|---|
| Eddy Irasetsa (all training data and test sequences) | iPhone XS (iOS 18.7.5) | Sensor Logger 1.68 | 50 Hz (`sampleRateMs = 20`) | 49.76–49.77 Hz | 50 Hz (linear interpolation) |
| A friend, not a group member (unseen test data only) | Samsung SM-A235F (Android) | Sensor Logger 1.68 | 100 Hz (`sampleRateMs = 10`) | 102.07 Hz | 50 Hz (20 Hz anti-aliasing low-pass, then linear interpolation) |

## Data collection videos (one per activity)

| Activity | Video |
|---|---|
| Standing | https://drive.google.com/file/d/1f4_p_-ujCiJZu97DpZ5USixrpc7iwrEp/view?usp=sharing |
| Walking | https://drive.google.com/file/d/1jgML2SbKMkt49ShfATg1yWFIp1IBQ9B9/view?usp=sharing |
| Jumping | https://drive.google.com/file/d/1U_uXRKpqcJjFT1HDZSg_uy9OPgarakPx/view?usp=sharing |
| Still | https://drive.google.com/file/d/1k5gLUf_np0WNJqVYsKyLhyXlgU_nX-zk/view?usp=sharing |

## Results (unseen test data)

Unseen data comes from two sources: (a) 3 new multi-activity sequences recorded on the training phone in a later session, and (b) 4 single-activity files shared by a friend (not a group member) from a different phone (Samsung, 100 Hz).

| State (Activity) | Number of Samples (windows) | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|---|
| Standing | 23 | 0.870 | 0.929 | 0.917 |
| Walking | 34 | 1.000 | 0.932 | 0.954 |
| Jumping | 32 | 0.938 | 1.000 | 0.982 |
| Still | 19 | 0.684 | 1.000 | 0.944 |
| **All (multi-class)** | **108** | 0.873 (macro) | 0.966 (macro) | **0.898** |

| Unseen source | Windows | Accuracy |
|---|---|---|
| same phone (iPhone XS), sequences | 88 | 0.875 |
| second phone (SM-A235F), single-activity | 20 | 1.000 |

Training-set accuracy is 0.984. Baum-Welch converged in 6 iterations. See `report/report.docx` for the full analysis.

## Repository structure

```
data/
  raw/train/<label>_<NN>-<timestamp>/   52 Sensor Logger exports (13 per activity, about 13.6 s each)
  raw/test/<sequence>-<timestamp>/      3 unseen multi-activity recordings, training phone (never used for training)
  raw/test/<Label><NN>-<timestamp>/     4 unseen single-activity recordings, second phone (Samsung SM-A235F, 100 Hz)
  labels/test_segments.csv              hand-labelled activity boundaries of the test recordings (seconds from recording start)
  processed/
    train/<label>_<NN>.csv              cleaned recordings: merged acc+gyro, 50 Hz, 2 s trimmed each end, labelled
    test/<sequence>.csv                 cleaned test recordings with per-sample labels
    features_train.csv / features_test.csv   per-window feature tables
    scaler.json                         Z-score mean/std (fitted on training windows only)
    hmm_params.npz                      learned pi, A, emission means/variances
    results.json                        every number used in the report
notebooks/hmm_activity_recognition.ipynb   full pipeline (data → features → HMM → evaluation)
figures/                                all plots (PNG), also shown in the notebook
report/
  report.docx / report.pdf              5-page report
requirements.txt
```

Raw files are never modified. All cleaning happens in the notebook.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/hmm_activity_recognition.ipynb   # or Run All in Jupyter
```

## Method summary

- **Pre-processing:** merge accelerometer and gyroscope on the nearest timestamp, drop NaNs, resample to a uniform 50 Hz grid (with an anti-aliasing low-pass filter for faster phones), trim 2 s at each end.
- **Windowing:** 2 s windows (`int(2.0 * FS)` = 100 samples) with 50 % overlap, built within each recording only.
- **Features (17):** 12 time-domain (log RMS, per-axis log std, log SMA, axis correlations) and 5 frequency-domain (dominant frequency, spectral energy, spectral entropy), followed by Z-score normalisation fitted on train only.
- **HMM:** 4 hidden states, diagonal-Gaussian emissions with a variance floor, log-space forward/backward, Viterbi, and multi-sequence Baum-Welch (stops when the log-likelihood gain is below 1e-4). It is initialised from labelled windows. States are mapped to labels with the Hungarian algorithm.
- **Evaluation:** Viterbi decoding of the unseen recordings, per-class sensitivity, specificity and accuracy, a confusion matrix, and decoded-sequence plots.
