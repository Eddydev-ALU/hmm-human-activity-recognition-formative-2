# Modeling Human Activity States Using Hidden Markov Models

Formative 2: human activity recognition from smartphone accelerometer and gyroscope data with a **Hidden Markov Model implemented from scratch in numpy** (Viterbi and Baum-Welch).

**Use case:** a phone or wearable monitor for older adults living alone. It infers the wearer's activity (standing, walking, jumping, still) and alerts a caregiver on long inactivity (*still*), or on sudden, violent, high-energy movement (proxied by *jumping*) that could be a fall or a seizure.

## Group members and devices

| Member | Phone | App | Configured rate | Detected rate (timestamps) | Harmonised to |
|---|---|---|---|---|---|
| [TODO: member name] | iPhone XS (iOS 18.7.5) | Sensor Logger 1.68 | 50 Hz (`sampleRateMs = 20`) | 49.76–49.77 Hz | 50 Hz (linear interpolation) |

## Data collection videos (one per activity)

| Activity | Video |
|---|---|
| Standing | [TODO: video link] |
| Walking | [TODO: video link] |
| Jumping | [TODO: video link] |
| Still | [TODO: video link] |

## Results (unseen test data)

| State (Activity) | Number of Samples (windows) | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|---|
| Standing | 18 | 0.889 | 0.914 | 0.909 |
| Walking | 29 | 1.000 | 0.932 | 0.955 |
| Jumping | 27 | 0.926 | 1.000 | 0.977 |
| Still | 14 | 0.571 | 1.000 | 0.932 |
| **All (multi-class)** | **88** | 0.847 (macro) | 0.962 (macro) | **0.886** |

Training-set accuracy is 0.985. Baum-Welch converged in 6 iterations. See `report/report.docx` for the full analysis.

## Repository structure

```
data/
  raw/train/<label>_<NN>-<timestamp>/   48 Sensor Logger exports (12 per activity, about 13.6 s each)
  raw/test/<sequence>-<timestamp>/      3 unseen multi-activity recordings (never used for training)
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

- **Pre-processing:** merge accelerometer and gyroscope on timestamp, drop NaNs, resample to a uniform 50 Hz grid, trim 2 s at each end.
- **Windowing:** 2 s windows (`int(2.0 * FS)` = 100 samples) with 50 % overlap, built within each recording only.
- **Features (17):** 12 time-domain (log RMS, per-axis log std, log SMA, axis correlations) and 5 frequency-domain (dominant frequency, spectral energy, spectral entropy), followed by Z-score normalisation fitted on train only.
- **HMM:** 4 hidden states, diagonal-Gaussian emissions with a variance floor, log-space forward/backward, Viterbi, and multi-sequence Baum-Welch (stops when the log-likelihood gain is below 1e-4). It is initialised from labelled windows. States are mapped to labels with the Hungarian algorithm.
- **Evaluation:** Viterbi decoding of the unseen recordings, per-class sensitivity, specificity and accuracy, a confusion matrix, and decoded-sequence plots.
