
===========================
CONFIGURING THE ENVIRONMENT
===========================
Setup a new environment in Python 3.8.
Install NEST 3.2. For info: https://nest-simulator.readthedocs.io/en/stable/installation/conda_forge.html#conda-forge-install
Install remaining packages via: pip install -r requirements.txt

====================
DOWNLOADING THE DATA
====================
1. Navigate to https://search.kg.ebrains.eu/instances/ba41de6c-1b30-4f48-89c4-a6eb47b2c549
2. Go to "Get Data" -> "Browse Files" -> "ChangeDetectionDataset" -> "CHDET"
3. Download the "ChangeDetectionConflict" folder
4. Place the zip file into the \recordings folder und unzip.
5. Rename the file to "ChangeDetectionConflict"

================
RUNNING ANALYSIS
================
You can configure which parts of the code you would like to execute in the config.yml file.
By default the script will run the full analysis. 
Running the script: 

	python run.py

Plotting and analysis:

	python analyse_and_plot.py









