# -*- coding: utf-8 -*-
'''
Set up experimental, recording, preprocessing and classification parameters.

Initialization of paths for system scripts, subjects directory, and data directory.
'''

import os

###### EXPERIMENTAL VARIABLES (experimentFunctions) ######
subjID = '00' 
expDay = '1'  # Neurofeedback day
monitor_size = [1536, 824]

# Stimuli presentation times (in Hz)
frameRate = 60
probeTime = frameRate * 6 # Frames to display probe word
fixTime = frameRate * 2 # Frames to display fixation cross
stimTime = frameRate # Frames for presenting each image

# Experiment structure and length
numRuns = 6 # Number of neurofeedback runs
numBlocks = 8 # Number of blocks within each run
blockLen = 50 # Number of trials within each block

###### EEG SAMPLING (runSystem) ###### 
samplingRate = 500 
samplingRateResample = 100
baselineTime = -0.1 # Baseline for each epoch (i.e. before stimuli onset) (seconds)
epochTime = 0.800 # Seconds (i.e. after stimuli onset) (seconds)

# Parameters for continuous data streaming
NUM_TRIALS = 10  # Number of trials for testing, set to None for infinite loop
CONTINUOUS_STREAM = True if NUM_TRIALS is None else False

maxBufferData = 2 # Maximum amount of data to store in the buffer (seconds)

# Channels for EPOC X
channelNames = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# If manually preselecting channels
rejectChannels = True
channelNamesSelected = channelNames
channelNamesExcluded = []

###### EEG preprocessing (realtimeFunctions) ###### 
highpass = 0  # Hz
lowpass = 40 # Hz
detrend = True # Temporal detrending of EEG signal (linear)
filterPhase = 'zero-double'
montage = 'standard_1020' 
SSP = True # Whether to apply SSP artifact correction
thresholdSSP = 0.1 # SSP variance threshold for rejection of SSP projections

###### EEG real-time classification (realtimeFunctions) ###### 
from sklearn.linear_model import LogisticRegression # Import and specify classifier of interest
classifier = LogisticRegression(solver='saga', C=1, random_state=1, penalty='l1', max_iter=100)

# For saving first time model
config = {"is_model_generated": False}
config_score = {"best_score": 0.0}

###### DIRECTORY STRUCTURE ###### 
def base_dir_init(): # Base directory for ClosedLoop GitHub
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return base_dir

def script_path_init():
    script_path = base_dir_init() + '/Scripts'
    return script_path

def data_path_init(): # Data (images) storage directory
    data_path = base_dir_init() + '/imageStimuli'
    return data_path

def subject_path_init(): # Subjects directory, for storing EEG data 
    subject_path = base_dir_init() + '/subjectsData'
    return subject_path

if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current directory ======')
    print(base_dir)