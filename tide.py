#!/usr/bin/env python3
"""
Harmonic Tide prediction  .
– Reads CSV "year, month, day, hour, level_mm"
– Converts to meters, splits into two halves
– Fits 8 harmonics on the 1st half
– Animates weekly windows on the 2nd half (real × predicted)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from pathlib import Path
import sys
import random
import argparse
import logging
from datetime import datetime
from matplotlib.ticker import MultipleLocator

# Set modern theme
plt.style.use('dark_background')

# ------------------------------------------------------------------
# 1) LOAD DATA (format 'year,month,day,hour,level_mm')
# ------------------------------------------------------------------

# Parse command line arguments
parser = argparse.ArgumentParser(description='Harmonic prediction for Fortaleza')
parser.add_argument('--csv_file', type=str, help='CSV data file')
parser.add_argument('--initial_year', type=int, help='Initial year of the period')
parser.add_argument('--final_year', type=int, help='Final year of the period')
parser.add_argument('--week_seed', type=int, default=42, help='Seed for random initial week position. Default:42')
parser.add_argument('--animation_weeks', type=int, default=4, help='Animation duration in weeks. Default: 4')
parser.add_argument('--local', type=str, help='Name of the local station', default='')
parser.add_argument('--animation_speed', type=float, default=1.0, help='Animation speed (multiplier). Default: 1.0')

args = parser.parse_args()

# ------------------------------------------------------------------
# CONFIGURE LOGGING
# ------------------------------------------------------------------

# Log the exact command line used to call this script
command_line = f"python {' '.join(sys.argv)}"

# Configure logging
log_file = Path('tide.log')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler(log_file),
		logging.StreamHandler(sys.stdout)
	]
)

logger = logging.getLogger(__name__)

# Log start of execution
start_time = datetime.now()
logger.info(f"Starting execution of tide.py")
logger.info(f"CSV file: {args.csv_file}")
logger.info(f"Parameters: initial_year={args.initial_year}, final_year={args.final_year}, week_seed={args.week_seed}, animation_weeks={args.animation_weeks}")

if not args.csv_file:
	# List CSV files in the current directory
	current_dir = Path('./data/')
	csv_files = list(current_dir.glob('*.csv'))
	
	if not csv_files:
		logger.error("Error: No CSV files found in the current directory.")
		sys.exit(1)
	
	print("\nCSV files found:")
	for i, csv_file in enumerate(csv_files, 1):
		print(f"{i}. {csv_file.name}")
	
	while True:
		try:
			choice = int(input(f"\nChoose a file (1-{len(csv_files)}): "))
			if 1 <= choice <= len(csv_files):
				args.csv_file = str(csv_files[choice - 1])
				logger.info(f"Selected file: {args.csv_file}")
				break
			else:
				print(f"Please choose a number between 1 and {len(csv_files)}")
		except ValueError:
			print("Please enter a valid number")

csv = Path(args.csv_file)
if not csv.exists():
	logger.error(f"Error: File '{csv}' not found.")
	sys.exit(1)

cols = ["year", "month", "day", "hour", "level_mm"]
df = pd.read_csv(csv, names=cols)

# build datetime column (UTC)
df["time"] = pd.to_datetime(
	dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
	utc=True
)
df["level"] = df.level_mm / 1000.0         # mm  →  m
df = df[["time", "level"]].set_index("time").sort_index()

# Check available year range
year_min, year_max = df.index.year.min(), df.index.year.max()
print(f"Data available from {year_min} to {year_max}")

# Check if both years were not specified
if not args.initial_year and not args.final_year:
	# Ask user for initial and final years interactively
	print(f"\nData available from {year_min} to {year_max}")
	while True:
		try:
			initial_year = int(input(f"Enter initial year ({year_min}-{year_max}): "))
			if initial_year < year_min or initial_year > year_max:
				print(f"Year must be between {year_min} and {year_max}")
				continue
			break
		except ValueError:
			print("Please enter a valid number")

	while True:
		try:
			final_year = int(input(f"Enter final year ({initial_year}-{year_max}): "))
			if final_year < initial_year or final_year > year_max:
				print(f"Year must be between {initial_year} and {year_max}")
				continue
			break
		except ValueError:
			print("Please enter a valid number")

	# Apply the selected year range
	print(f"Filtering data for period: {initial_year}-{final_year}")
	df = df[(df.index.year >= initial_year) & (df.index.year <= final_year)]
elif args.initial_year or args.final_year:
		# Use provided command line arguments
		initial_year = args.initial_year if args.initial_year else year_min
		final_year = args.final_year if args.final_year else year_max
		
		# Validate the year range
		if initial_year < year_min:
			logger.error(f"Error: Initial year {initial_year} is before available data starts ({year_min})")
			sys.exit(1)
		
		if final_year > year_max:
			logger.error(f"Error: Final year {final_year} is after available data ends ({year_max})")
			sys.exit(1)
		
		if initial_year > final_year:
			logger.error(f"Error: Initial year {initial_year} is after final year {final_year}")
			sys.exit(1)
		
		# Apply the year range filter
		print(f"Filtering data for period: {initial_year}-{final_year}")
		df = df[(df.index.year >= initial_year) & (df.index.year <= final_year)]
		
		# Check if we have data after filtering
		if len(df) == 0:
			logger.error(f"Error: No data available for the specified period {initial_year}-{final_year}")
			sys.exit(1)
		
		logger.info(f"Data filtered to period {initial_year}-{final_year}: {len(df)} records")
# ------------------------------------------------------------------
# Discard missing data (-32767)
# ------------------------------------------------------------------
logger.info(f"Data before filtering missing values: {len(df)} records")
initial_count = len(df)

# Filter -32767 values (missing data)
df = df[df['level'] != -32.767]  # -32767 mm = -32.767 m

missing_count = initial_count - len(df)
if missing_count > 0:
	logger.info(f"Removed {missing_count} records with missing data (-32767)")
	logger.info(f"Data after filtering: {len(df)} records")
else:
	logger.info("No missing data (-32767) found")

# Check if we still have enough data
if len(df) < 100:
	logger.error("Error: Insufficient data after filtering. Minimum 100 records required.")
	sys.exit(1)

# Set seed for random position
SEED = args.week_seed
print(f"Using seed: {SEED}")

# ------------------------------------------------------------------
# 2) TRAIN (1st half)  ×  TEST (2nd half)
# ------------------------------------------------------------------
mid = len(df) // 2
obs, tst = df.iloc[:mid], df.iloc[mid:]

offset = obs['level'].mean()      # provisional offset
obs['level'] -= offset
tst['level'] -= offset

# Show period used for training
print(f"Training period: {obs.index[0]} to {obs.index[-1]}")
print(f"Total training data: {len(obs)} records")
print(f"Test period: {tst.index[0]} to {tst.index[-1]}")
print(f"Total test data: {len(tst)} records")

# ------------------------------------------------------------------
# 37 standard NOAA constituents (name → speed °/h)
# ref. NOS/CO‑OPS "standard suite"
# ------------------------------------------------------------------
SPEEDS_DPH = {          # deg per hour
	"M2":   28.9841042, "S2":   30.0000000, "N2":   28.4397295,
	"K2":   30.0821373, "K1":   15.0410686, "O1":   13.9430356,
	"P1":   14.9589314, "Q1":   13.3986609,

	# shallow water harmonics
	"M4":   57.9682084,               # 2·M2
	"MS4":  58.9841042,               # M2+S2
	"MN4":  57.4238337,               # M2+N2
	"M6":   86.9523126,               # 3·M2
	"M8":  115.9364168,               # 4·M2

	# long-period
	"SA":    0.0410686,               # annual (≈ 365 d)
	"SSA":   0.0821373,               # semi-annual
	"MF":    1.0980331,               # lunar fortnightly
	"MM":    0.5443750,               # lunar monthly
	"MSF":   1.0158958,               # solar-lunar fortnightly

	# extra diurnal/lunisolar
	"J1":   15.5854433, "M1":   14.4920521, "OO1":  16.1391017,
	"RHO":  13.4715145, "2Q1":  12.8542862, "S1":   15.0000000,

	# secondary semidiurnals
	"2N2":  27.8953548, "MU2":  27.9682084, "NU2":  28.5125831,
	"L2":   29.5284789, "T2":   29.9589333, "R2":   30.0410667,
	"LAM2": 29.4556253, "2SM2": 31.0158958,

	# additional harmonics and compounds
	"MK3":  44.0251729, "2MK3": 73.0092771, "SK3":  45.0410686,
	"S4":   60.0000000, "S6":   90.0000000,
	"M3":   43.4761563, "MS4":  58.9841042, # already included above
}

# ------------------------------------------------------------------
# derive periods (h) and frequencies rad/s
# ------------------------------------------------------------------
periods_h = [360.0 / v for v in SPEEDS_DPH.values()]
ω = 2*np.pi / (np.array(periods_h)*3600)     # rad/s
t0 = obs.index[0]

def tsec(idxs):            # seconds since t0
	return (idxs - t0).total_seconds().values

def design_matrix(t):
	# 1 | cos(ω₁t)… | sin(ω₁t)…
	cos = [np.cos(w*t) for w in ω]
	sin = [np.sin(w*t) for w in ω]
	return np.column_stack([np.ones_like(t), *cos, *sin])

# least squares
coef, *_ = np.linalg.lstsq(
	design_matrix(tsec(obs.index)), obs.level.values, rcond=None
)

# ------------------------------------------------------------------
# 4) PREDICTION ON THE SECOND HALF
# ------------------------------------------------------------------
t_test = tsec(tst.index)
pred   = design_matrix(t_test) @ coef
real   = tst.level.values

# ------------------------------------------------------------------
# 5) ANIMATION – SLIDING 7-DAY WINDOW
# ------------------------------------------------------------------
dt_s        = (tst.index[1] - tst.index[0]).total_seconds()
win_days    = 7
win_samples = int(win_days*24*3600/dt_s)

step_h      = 1 / args.animation_speed        # shift 1 h/frame adjusted by speed
step_samples= int(step_h*3600/dt_s)

# CHOOSE RANDOM INITIAL POSITION FOR THE WEEK
max_start = len(tst) - win_samples

random.seed(SEED)
random_start = random.randint(0, max_start)

# ANIMATION DURATION PARAMETER IN WEEKS
animation_weeks = args.animation_weeks
if animation_weeks:
	print(f"Animation duration: {animation_weeks} weeks")

# Calculate number of frames based on specified duration
if animation_weeks is not None:
	animation_samples = int(animation_weeks * 7 * 24 * 3600 / dt_s)
	max_end = min(random_start + animation_samples, len(tst) - win_samples)
	frames = range(random_start, max_end, step_samples)
	print(f"Animation limited to {animation_weeks} weeks ({len(frames)} frames)")
else:
	frames = range(random_start, len(tst)-win_samples, step_samples)
	total_weeks = (len(frames) * step_samples * dt_s) / (7 * 24 * 3600)
	print(f"Animation with total available duration: {total_weeks:.1f} weeks ({len(frames)} frames)")

print(f"Starting animation at random position: {random_start}")
print(f"Initial week date: {tst.index[random_start]}")
print(f"Final week date: {tst.index[random_start + win_samples - 1]}")

interval_ms = 2000*step_h/24                  # ~83 ms/frame

# Modern figure setup
fig, ax = plt.subplots(figsize=(14, 8), facecolor='black')
ax.set_facecolor('#0a0a0a')

# Vibrant modern colors
color_real = '#00ffff'    # Electric cyan
color_pred = "#eaff00"    # Vibrant magenta pink

# Simple lines
line_r, = ax.plot([], [], '-', lw=2, color=color_real, 
				 label="Observed")
line_p, = ax.plot([], [], '--', lw=2, color=color_pred, 
				 label="Predicted")

# Graph styling
ax.set_ylabel("Level (m)", fontsize=14, color='white', fontweight='bold')
ax.set_xlabel("Time", fontsize=14, color='white', fontweight='bold')

# Subtle grid
ax.grid(True, alpha=0.2, color='#333333', linewidth=0.5)

# Configure spines (borders)
for spine in ax.spines.values():
	spine.set_color('#444444')
	spine.set_linewidth(1.5)

# Configure ticks
ax.tick_params(colors='white', labelsize=11, width=1.2)
ax.tick_params(axis='x', rotation=15)

# Styled legend
legend = ax.legend(loc='upper left', fontsize=12, 
				  frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('#1a1a1a')
legend.get_frame().set_edgecolor('#444444')
legend.get_frame().set_alpha(0.9)
for text in legend.get_texts():
	text.set_color('white')

def init():
	line_r.set_data([], [])
	line_p.set_data([], [])
	return line_r, line_p

def update(i):
	i0, i1 = i, i + win_samples
	x = tst.index[i0:i1]
	y_real = real[i0:i1]
	y_pred = pred[i0:i1]
	
	y_real_norm = y_real 
	y_pred_norm = y_pred 
	
	line_r.set_data(x, y_real_norm)
	line_p.set_data(x, y_pred_norm)
	ax.set_xlim(x[0], x[-1])
	
	# Adjust y scale based on normalized data in current window
	y_min = min(y_real_norm.min(), y_pred_norm.min())
	y_max = max(y_real_norm.max(), y_pred_norm.max())
	margin = (y_max - y_min) * 0.1
	ax.set_ylim(y_min - margin, y_max + margin)
	
	# Main title with location
	base_filename = Path(args.csv_file).stem
	main_title = f"{args.local.upper()} - HARMONIC PREDICTION" if args.local else "HARMONIC PREDICTION- " + base_filename.upper()
	
	# Subtitle with detailed info
	period_str = f"{year_min}-{year_max}" if 'year_min' in locals() else f"{df.index[0].year}-{df.index[-1].year}"
	train_period = f"{obs.index[0].strftime('%Y-%m-%d')} to {obs.index[-1].strftime('%Y-%m-%d')}"
	n_harmonics = len(SPEEDS_DPH)
	
	subtitle = f"Data: {period_str} | Training: {train_period} | {n_harmonics} harmonics"
	
	fig.suptitle(f"{main_title}\n7-Day Moving Window\n{subtitle}", 
				 y=0.95, fontsize=14, color='white', fontweight='bold',
				 ha='center')
	return line_r, line_p

ani = anim.FuncAnimation(fig, update, frames=frames,
						 init_func=init, interval=interval_ms,
						 blit=False, repeat=False)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# ------------------------------------------------------------------
# 6) PLOT OF 4-WEEK AVERAGE OBSERVED LEVEL OVER THE PERIOD
# ------------------------------------------------------------------

# Set parameters for 4-week averages
weeks_4_days = 28  # 4 weeks = 28 days
samples_4w = int(weeks_4_days * 24 * 3600 / dt_s)

# Calculate 4-week moving averages for the entire data period
all_data = df.copy()
all_data['level'] -= offset  # Apply same offset used in training

# Calculate moving averages
dates_4w = []
means_4w = []

for i in range(0, len(all_data) - samples_4w, samples_4w // 4):  # Step of 1 week
	window_data = all_data.iloc[i:i + samples_4w]
	if len(window_data) == samples_4w:
		dates_4w.append(window_data.index[samples_4w // 2])  # Midpoint of window
		means_4w.append(window_data['level'].mean())

# Create plot of 4-week averages
fig_means, ax_means = plt.subplots(figsize=(16, 10), facecolor='black')
ax_means.set_facecolor('#0a0a0a')

# Plot averages
ax_means.plot(dates_4w, means_4w, '-o', linewidth=2, markersize=4, 
			  color='#00ffaa', alpha=0.8, label='Average Level (4 weeks)')

# Add trend line
if len(dates_4w) > 1:
	# Convert dates to numbers for regression
	dates_num = [(d - dates_4w[0]).total_seconds() / (365.25 * 24 * 3600) for d in dates_4w]
	trend_coef = np.polyfit(dates_num, means_4w, 1)
	trend_line = np.polyval(trend_coef, dates_num)
	ax_means.plot(dates_4w, trend_line, '--', linewidth=4, 
				  color='#ff6600', alpha=0.7, label=f'Trend ({trend_coef[0]:.3f} m/year)', 
				  zorder=10)
	# Increase trend legend font size
	legend_means = ax_means.legend(loc='upper left', fontsize=22, 
								   frameon=True, fancybox=True, shadow=True)
	legend_means.get_frame().set_facecolor('#1a1a1a')
	legend_means.get_frame().set_edgecolor('#444444')
	legend_means.get_frame().set_alpha(0.9)
	for text in legend_means.get_texts():
		text.set_color('white')

# Styling
ax_means.set_ylabel("Average Level (m)", fontsize=28, color='white', fontweight='bold')
ax_means.set_xlabel("Decade", fontsize=28, color='white', fontweight='bold')
ax_means.grid(True, alpha=0.3, color='#333333', linewidth=0.5)

# Configure spines
for spine in ax_means.spines.values():
	spine.set_color('#444444')
	spine.set_linewidth(1.5)

# Configure x-axis formatting to show decades
from matplotlib import dates as mdates
from datetime import datetime

# Custom locator for decades (every 10 years)
class DecadeLocator(mdates.YearLocator):
	def __init__(self):
		super().__init__(base=10)

# Custom formatter for decades
class DecadeFormatter(mdates.DateFormatter):
	def __init__(self):
		super().__init__('%Y')
	
	def __call__(self, x, pos=None):
		year = mdates.num2date(x).year
		# Show decades as "1990s", "2000s", etc.
		decade = (year // 10) * 10
		return f"{decade}s"

ax_means.xaxis.set_major_locator(DecadeLocator())
ax_means.xaxis.set_major_formatter(DecadeFormatter())
ax_means.xaxis.set_minor_locator(mdates.YearLocator(base=5))  # Minor ticks every 5 years

# Larger, whiter ticks
ax_means.tick_params(axis='x', colors='white', labelsize=22, width=2.5, length=10, direction='out')
ax_means.tick_params(axis='y', colors='white', labelsize=22, width=2.5, length=10, direction='out')

# Legend
legend_means = ax_means.legend(loc='upper left', fontsize=18, 
							   frameon=True, fancybox=True, shadow=True)
legend_means.get_frame().set_facecolor('#1a1a1a')
legend_means.get_frame().set_edgecolor('#444444')
legend_means.get_frame().set_alpha(0.9)
for text in legend_means.get_texts():
	text.set_color('white')

# Title
base_filename = Path(args.csv_file).stem
period_full = f"{all_data.index[0].year}-{all_data.index[-1].year}"
title_means = f"{args.local.upper()} - OBSERVED AVERAGE LEVEL (4 WEEKS)" if args.local else "OBSERVED AVERAGE LEVEL (4 WEEKS)- " + base_filename.upper()
fig_means.suptitle(f"{title_means}\nPeriod: {period_full} | Total: {len(dates_4w)} periods of 4 weeks", 
				   fontsize=20, color='white', fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Save plot
base_filename = Path(args.csv_file).stem
# Generate filename with current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M")
fig_means.savefig(f'{base_filename}_average_level_4_weeks_{current_time}.png', dpi=300, bbox_inches='tight', 
				  facecolor='black', edgecolor='none')

# Log statistics
logger.info(f"Calculated {len(dates_4w)} 4-week averages")
logger.info(f"Overall average level: {np.mean(means_4w):.3f} m")
logger.info(f"Standard deviation of averages: {np.std(means_4w):.3f} m")
if len(dates_4w) > 1:
	logger.info(f"Trend: {trend_coef[0]:.3f} m/year")

# optional:
# Generate filename based on location
if args.local:
	video_filename = f"{args.local.lower()}_weekly_average.mp4"
else:
	# Extract base filename without extension and path
	base_filename = Path(args.csv_file).stem
	# Generate filename with current date and time
	current_time = datetime.now().strftime("%Y%m%d_%H%M")
	video_filename = f"{base_filename}_weekly_average_{current_time}.mp4"
	video_filename_gif = f"{base_filename}_weekly_average_{current_time}.gif"

ani.save(video_filename, writer="ffmpeg", fps=1000/interval_ms)
ani.save(video_filename_gif, writer="pillow", fps=1000/interval_ms)
logger.info(f"Animation saved as GIF: {video_filename_gif}")
logger.info(f"Animation saved as: {video_filename}")
