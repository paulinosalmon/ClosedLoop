'''
The script finds the markers (trigger points for stimuli onset) from the experimental script and samples EEG epochs a specified number of seconds pre-stimuli onset (settings.baselineTime)
to a specified number of seconds (settings.epochTime) post-stimuli onset. Based on a specified number of epochs, the system changes between the following states: 
'stable' for recording EEG, 'train' for training a real-time decoding model, and 'feedback' for recording and providing neurofeedback in real-time).
'''

# Imports
import numpy as np
import time
import os
from pylsl import StreamOutlet, StreamInfo, resolve_byprop, local_clock
import settings
import sys

# Paths
script_path = settings.script_path_init()
subject_path = settings.subject_path_init()
os.chdir(script_path)

import realtimeFunctions
import streamFunctions

# Extract experimental categories 
y = realtimeFunctions.extractCat(subject_path + '/' + settings.subjID + '/createIndices_' + settings.subjID + '_day_' + str(settings.expDay) + '.csv')
y = np.array([int(x) for x in y])

#%% Tracking of print outputs. Saved as a log file to the subject ID folder.

# Check if directory exists
dir_path = os.path.join(subject_path, settings.subjID)
print(f"Attempting to create directory: {dir_path}")

if not os.path.exists(dir_path):
    print(f"Directory {dir_path} does not exist. Creating...")
    os.makedirs(dir_path)
else:
    print(f"Directory {dir_path} already exists.")

log_file_path = os.path.join(dir_path, 'stream_logfile_subject_' + settings.subjID + '_' + time.strftime('%m-%d-%y_%H-%M') + '.log')
print(f"Attempting to create log file: {log_file_path}\n")

sys.stdout = open(log_file_path, 'w')

# sys.stdout = open(subject_path + '/' + settings.subjID + '/stream_logfile_subject_' + settings.subjID + '_' + time.strftime('%m-%d-%y_%H-%M') + '.log')

#%% Look for a recent stream from the Psychopy experimental script (will localize the experimental stream after runSystem is started).

marker_not_present = 1 # Whether to look for a marker stream from other scripts
timeout = 1 # Time to look for stream in seconds 

while marker_not_present:
    t = local_clock()
    streams = resolve_byprop('type', 'Markers', timeout=timeout)
    for i in range (len(streams)):
        lsl_created = (streams[i].created_at())
        if np.abs(lsl_created - t) < 20: # Make sure that the stream has been created within the last 20 seconds
            marker_not_present = 0
            print("Stream from PsychoPy experimental script found")

#%% Start sampling, preprocessing and analysis of EEG data real-time in three states:
# 1) Stable (EEG data during stable blocks used for training the decoding classifier. EEG data size: 600 most recent stable trials)
# 2) Train (training of the decoding classifier)
# 3) Feedback (preprocessing and classification of EEG data for neurofeedback)

# Read EEG data stream and experimental script marker stream
inlet_EEG,store_EEG = streamFunctions.read_EEG_stream(fs=settings.samplingRate, max_buf=settings.maxBufferData) # EEG stream
inlet_marker, store_marker = streamFunctions.read_marker_stream(stream_name ='PsychopyExperiment') # Experimental marker stream
info_outlet = StreamInfo('alphaStream', 'Markers', 1, 0, 'float32', 'myuniquesourceid23441') # Outlet for alpha values from the trained classifier. Used for updating the experimental stimuli
outlet_alpha = StreamOutlet(info_outlet)

# Variables for sampling of EEG and classifier training/feedback
n_time = int((settings.epochTime - settings.baselineTime) * settings.samplingRateResample)
n_channels = len(settings.channelNamesSelected)

# MNE info structures for EEG data
info_fs_orig = realtimeFunctions.createInfoMNE(channel_names=settings.channelNames, sfreq=settings.samplingRate) 
info_fs_resample = realtimeFunctions.createInfoMNE(channel_names=settings.channelNamesSelected, sfreq=settings.samplingRateResample)

# Clear EEG and experiment streams to ensure a clean start
streamFunctions.clear_stream(inlet_marker)
streamFunctions.clear_stream(inlet_EEG)

n_runs = settings.numRuns + 1 # No. runs (total)
n_stable = int((settings.numBlocks + settings.numBlocks/2) * settings.blockLen) # No. stable epochs used for training (a full run and 1/2 run)
n_stable_epochs = int(settings.numBlocks/2 * settings.blockLen) # No. stable epochs in each run (after first run)
n_feedback_epochs = int(settings.numBlocks/2 * settings.blockLen) # No. feedback epochs in each run (after first run)
n_epochs = n_feedback_epochs + n_stable_epochs

# Initializations
state = 'stable' # Initialization state
look_for_trigger = 1
n_run = 0
n_trials = n_stable*n_runs 

excess_EEG = []
excess_EEG_time = []
excess_marker = []
excess_marker_time = []
stable_blocks = np.zeros((n_stable, n_channels, n_time))
stable_blocks0 = np.zeros((n_stable_epochs, n_channels, n_time)) 
marker_all = []
y_run = y[0:n_stable]
marker = [0]
alphaAll = []
epochs_fb = np.zeros((n_feedback_epochs, n_channels, n_time))
fb_starts = [t*n_feedback_epochs for t in range(n_runs)]
y_feedback = np.concatenate(([y[(r+1)*n_epochs+n_stable_epochs:(r+2)*n_epochs] for r in range(n_runs)])) # Experimental categories (y) for neurofeedback blocks


# Start sampling (continues as long as the marker from the experimental script is below the number of total trials)
while marker is not None and len(marker) > 0 and marker[0]+1 < n_trials:
    if state == 'stable':
        # Extract epoch
        epoch, state, marker, excess_EEG, excess_EEG_time,\
        excess_marker, excess_marker_time, look_for_trigger = streamFunctions.get_epoch(inlet_EEG, inlet_marker,\
                                                                                        store_EEG, store_marker, \
                                                                                        settings.subjID, excess_EEG, \
                                                                                        excess_EEG_time, excess_marker, \
                                                                                        excess_marker_time, state=state, \
                                                                                        look_for_trigger=look_for_trigger, \
                                                                                        tmax=settings.epochTime, fs=settings.samplingRate)
        
        # Preprocess one epoch at a time and append to stable_blocks array
        if epoch is not None and len(epoch):
            epoch = realtimeFunctions.preproc1epoch(epoch, info_fs_orig, SSP=settings.SSP, reject_chs=settings.rejectChannels)
            
            stable_blocks[marker[0]-n_run*n_epochs,:,:] = epoch
            marker_all.append(marker)
            
    elif state == 'train':
        # Test if stable block epochs are missing
        ss = np.sum(np.sum(np.abs(stable_blocks),axis=2), axis=1)
        epochs0_idx = np.where(ss==0)[0]
        if len(epochs0_idx):
            rep = 0
            while rep < 30:
                print('WARNING missing ' + str(len(epochs0_idx)) + ' stable epochs\n')
                rep += 1
            epochs_non0_avg = np.mean(np.delete(stable_blocks,epochs0_idx,axis=0), axis=0)
            stable_blocks[epochs0_idx] = epochs_non0_avg
        
        # Compute SSP projectors based on stable blocks (training EEG data) 
        projs, stable_blocksSSP = realtimeFunctions.applySSP(stable_blocks, info_fs_resample, threshold=settings.thresholdSSP)

        # Average two consecutive trials over a moving window 
        stable_blocksSSP = realtimeFunctions.averageStable(stable_blocksSSP)

        print('Training classifier on SSP corrected EEG data')
        
        # Train classifier for other runs than the first run and estimate classifier bias for offset correction
        if n_run > 0:
            # Test if neurofeedback epochs are missing
            ss_fb = np.sum(np.sum(np.abs(epochs_fb),axis=2), axis=1)
            epochs0_idx = np.where(ss_fb==0)[0]
            if len(epochs0_idx):
                rep = 0
                while rep < 30:
                    print('WARNING missing ' + str(len(epochs0_idx)) + ' neurofeedback epochs\n')
                    rep += 1
                    epochs_non0_avg = np.mean(np.delete(epochs_fb, epochs0_idx,axis=0), axis=0)
                    epochs_fb[epochs0_idx] = epochs_non0_avg
            
            # Append SSP corrected neurofeedback blocks to array with SSP corrected stable blocks (stable_blocksSSP_append)
            stable_blocksSSP_append = np.append(stable_blocksSSP, epochs_fb, axis=0)
            fb_start = (n_run-1)*n_feedback_epochs
            y_run_append = np.append(y_run,y_feedback[fb_start:fb_start+n_feedback_epochs])
            print('About to train classifier for n_run > 0')
            clf, offset = realtimeFunctions.trainClassifiersifierCV2(stable_blocksSSP_append, y_run_append, settings.classifier)
        
        # Train classifier for the first run and estimate classifier bias for offset correction 
        else:
            clf, offset = realtimeFunctions.trainClassifierCV(stable_blocksSSP, y_run, settings.classifier)
        
        # Limit offset correction to abs(0.125)
        offset = (np.max([np.min([offset,0.25]),-0.25]))/2
        print('Offset: ' + str(offset))
        
        # Prepare stable_blocks for next run (remove first 200 trials and pre-initialize to 600 trials)
        stable_blocks = stable_blocks[n_stable_epochs:,:,:]
        stable_blocks = np.concatenate((stable_blocks,stable_blocks0), axis=0)
        y_run = np.concatenate((y_run[n_stable_epochs:], y[n_epochs*(n_run+2):n_stable_epochs+n_epochs*(n_run+2)]))
        
        
        print('Classifier training finished, commencing onto neurofeedback')
        state = 'feedback'
        n_run += 1
        epochs_fb = np.zeros((n_feedback_epochs, n_channels, n_time))


    elif state == 'feedback':
        epoch, state, marker, excess_EEG, excess_EEG_time, \
        excess_marker, excess_marker_time, look_for_trigger = streamFunctions.get_epoch(inlet_EEG, inlet_marker,\
                                                                                        store_EEG, store_marker, settings.subjID, \
                                                                                        excess_EEG, excess_EEG_time, excess_marker, \
                                                                                        excess_marker_time, state=state, \
                                                                                        look_for_trigger=look_for_trigger, tmax=settings.epochTime, \
                                                                                        fs=settings.samplingRate)
        
        # Test which number epoch is currently sampled and about to be preprocessed
        t_test = marker-n_stable_epochs-n_epochs*n_run 
        if len(epoch):
            epoch = realtimeFunctions.preproc1epoch(epoch, info_fs_orig, projs=projs, SSP=settings.SSP, reject_chs=settings.rejectChannels)
            if t_test > 0:    
                epoch = (epoch+epoch_prev)/2
            
            epochs_fb[t_test,:,:] = epoch
            print('Epoch number: ' + str(t_test))
            
            # Apply classifier for real-time decoding
            pred_prob = realtimeFunctions.testEpoch(clf, epoch)
            
            # Apply offset correction for both binary categories. Limit prediction probability to a value between 0 and 1
            pred_prob[0][0,0] = np.min([np.max([pred_prob[0][0,0]+offset,0]),1]) 
            pred_prob[0][0,1] = np.min([np.max([pred_prob[0][0,1]-offset,0]),1])
            clf_output = pred_prob[0][0,int(y[marker[0]])]-pred_prob[0][0,int(y[marker[0]]-1)]
            
            # Compute alpha from the corrected classifier output using a sigmoid transfer function. Alpha value used for updating experimental stimuli.
            alpha = realtimeFunctions.sigmoid(clf_output)
            marker_alpha = [marker[0][0], alpha]
            print('Marker number: ' + str(marker_alpha)) 
            
            # Push alpha value to an outlet for use in experimental script
            outlet_alpha.push_sample([alpha])
            epoch_prev = epoch
