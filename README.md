# Harmonic Tide Prediction 

 **Demo only — not for navigation/operations.**

This script Illustrates harmonic tide prediction with standard Python tools.
a non-profissional harmonic prediction tool for sea level that generates a short animation.
The script  reads an hourly time series in CSV format, converts levels from >millimeters to
meters, splits the record into two halves (training and test), fits a tidal harmonic model
by least squares on the first half, and, on the second half, animates sliding 7-day windows
comparing observed vs. predicted levels. It also computes and plots 4-week moving averages
over the full period.

This software is not for profissional use. It was created using vibe programming to
illustrate  how harmonic prediction can be done using  standard libraries in Python.

**Do not use for navigation or operational decisions.**

## Examples

Tide Predition vs Colleted Data for Salvador-Brazil

![Salvador-Brazil](https://github.com/SmaniaD/Tide-Prediction/blob/main/Salvador-Brazil_prediction.gif)



Average sea level along 100 years for Honolulu


![Honolulu](https://github.com/SmaniaD/Tide-Prediction/blob/main/Honolulu_average_level_4_weeks.png)

## What it does
- Reads hourly CSV `year,month,day,hour,level_mm`
- Converts **mm → m**, removes `-32767` (missing)
- Splits data: **train** (1st half) / **test** (2nd half)
- Fits ~37 tidal constituents (NOS/CO-OPS) by least squares
- Animates a **sliding 7-day** window (observed × predicted)
- Plots **4-week averages** + linear trend (m/yr)
- Saves **MP4** (ffmpeg) and **PNG**; logs to `tide.log`

## Requirements
- Python ≥ 3.8 · `numpy`, `pandas`, `matplotlib`
- `ffmpeg` on PATH (for MP4)

Setup:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib
# install ffmpeg via apt/brew/choco/winget, etc.
```

## Data
Sample CSVs in `./data/`: **Honolulu**, **Fortaleza**, **Salvador**  
Source: University of Hawai‘i Sea Level Center (UHSLC) — the data keep UHSLC terms.  
**Cite if used:**  
Caldwell, Merrifield, Thompson (2015), *Sea level measured by tide gauges…* NOAA NCEI, doi:10.7289/V5V40S7W.

CSV format (no header): `year,month,day,hour,level_mm` (UTC).  
Minimum **100** records after filtering.

## Quick start
Interactive:
```bash
python tide.py
```
Example:
```bash
python tide.py --csv_file ./data/Honolulu.csv --initial_year 1993 --final_year 2020   --local "Honolulu" --animation_weeks 4 --animation_speed 1.0 --week_seed 42
```

## Key options
- `--csv_file PATH` · `--initial_year INT` · `--final_year INT`
- `--week_seed INT` (start position reproducible)
- `--animation_weeks INT` (duration, in weeks)
- `--local "Label"` (titles/filenames)
- `--animation_speed FLOAT` (use `1.0`)

## Outputs
- MP4 animation: `{local_lower}_weekly_average.mp4`  
  or `{csv_basename}_weekly_average_{YYYYMMDD_HHMM}.mp4`
- PNG averages: `{csv_basename}_average_level_4_weeks_{YYYYMMDD_HHMM}.png`
- Log: `tide.log`

## Notes / limitations
- Educational code; no nodal corrections or QA beyond `-32767`
- Frequencies may be ill-conditioned on some datasets
- Model trained on first half only

## License
**Code:** MIT.  
**Data:** UHSLC terms (see site above).
