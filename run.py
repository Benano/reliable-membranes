"""Script to perform parameter search."""
# Importing
import yaml
import os
import datetime
import main as main_sbi
import shutil
from pathlib import Path

# Loading Parameters
curr_dir = os.getcwd()
with open(curr_dir + '/config.yml', 'r') as f:
    params = \
        yaml.load(f, Loader=yaml.FullLoader)
run_params = params['run_params']
sbi_params = params['sbi_params']
sim_params = params['sim_params']
neuron_params = params['neuron_params']
plot_params = params['plot_params']
data_params = params['data_params']

# Checking data folder
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = Path(os.path.join(script_dir, "recordings/ChangeDetectionConflict"))
assert data_dir.exists(), (
    "No data found in recordings folder."
    "Make sure the data folder is named ChangeDetectionConflict")

# %% Creating Save Folder
cwd = os.getcwd()
save_dir = cwd + '/results_archive/'
data_folder = cwd + '/results_current'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

now = datetime.datetime.now()
time_format_code = '%H\ua789%M\ua789%S'
day_format_code = '%d-%m-%y'
day = now.strftime(day_format_code) + '/'
hms = now.strftime(time_format_code) + '_' + run_params['note']
if not os.path.isdir(save_dir+day):
    os.mkdir(save_dir+day)
title = hms

# Creating Run folder
save_folder = save_dir+day+title
if run_params['saving']:
    print("saving results to folder")
    os.mkdir(save_folder)
    shutil.copy('config.yml', save_folder)
else:
    print("YOU ARE NOT SAVING TO UNIQUE FOLDER")

# Save to data folder
shutil.copy('config.yml', data_folder)

# Perform SBI
main_sbi.main(run_params, sbi_params, sim_params, data_params,
              neuron_params, plot_params, save_folder, data_folder)
