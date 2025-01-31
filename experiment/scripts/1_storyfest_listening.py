# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: January 31, 2025
# Description: This script is used to run the listening portion of the StoryFest experiment.

#--------------------------------- Import ---------------------------------#
from __future__ import absolute_import, division, print_function
import psychopy
import pandas as pd
import os
import time
import psychtoolbox as ptb
from psychopy import visual, core, event, iohub, data, gui, logging, sound
from psychopy.iohub import launchHubServer
from psychopy.iohub.util import hideWindow, showWindow
from psychopy.sound import Sound, Microphone
from psychopy.hardware.keyboard import Keyboard
from psychopy.constants import PLAYING, PAUSED
from pylsl import StreamInfo, StreamOutlet

# Length of stim in seconds
# first_half_length = 1320 
# second_half_length = 1320

# psychopy.useVersion('2022.2.5')

#--------------------------------- Toggle ---------------------------------#

# =========================================================
# Toggle tracker: 
# 1=Eyelink, 2=Mouse (with calibration), 0=Skip calibration
# =========================================================
ET = 1

# ====================================
# Toggle kill switch:
# 0=No kill switch, 1=Yes kill switch
# ====================================
KS = 1

#------------------------------- Initialize -------------------------------#

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = 'v2022.2.5'
expName = 'storyfest'
expInfo = {'participant': '', 'group': ''}  

# Get participant ID via dialog
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit() # If user pressed cancel
expInfo['date'] = data.getDateStr() # Add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Setup the Window
win = visual.Window(
    size = [1920, 1080], fullscr=True,
    # screen=1,
    units='pix',
    allowGUI=False,
    colorSpace='rgb255',
    monitor='55w_60dist', 
    color="gray"
)
win.mouseVisible = False

# Setup central circle
crossCentralBlack = visual.TextStim(
    win=win, 
    name='crossCentralBlack',
    text='+',
    font='Arial',
    pos=[0, 0],
    height=50,
    bold=True,
    color='black',
    units='pix',
    colorSpace='rgb'
)

# ==================
# Data save settings
# ==================

# Path to save data
path = os.path.join(_thisDir, '..', 'data')

# Data file name
filename = os.path.join(path, '%s_%s_%s_%s' % (expInfo['participant'], expInfo['group'], expName, expInfo['date']))

# Save a log file for detail verbose info
logFile = logging.LogFile(filename + '.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)

# Store participant ID and timestamps 
df = {'id': [expInfo['participant']]}
dfTimeStamps = pd.DataFrame(df)  

# Save df to csv
dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)

#------------------------- Experiment Settings ----------------------------#

# ================
# Eyetracker setup
# ================

devices_config = {}

## getting iohub startup error when eyetracker isn't connected yet still running using ET=1

if ET == 1:
    TRACKER = 'eyelink'
    eyetracker_config = dict(name='tracker')
    #devices_config = {}
   
    eyetracker_config['model_name'] = 'EYELINK 1000 DESKTOP'
    eyetracker_config['runtime_settings'] = dict(sampling_rate=500, track_eyes='RIGHT')
    devices_config['eyetracker.hw.sr_research.eyelink.EyeTracker'] = eyetracker_config
    eyetracker_config['calibration'] = dict(type='FIVE_POINTS')
    
    win.setMouseVisible(False)

io = launchHubServer(window=win, **devices_config)

# ===========
# iohub setup
# ===========
keyboard = io.getDevice('keyboard')
kb = Keyboard(waitForStart=True) # JP: clock?
tracker = io.getDevice('tracker')

# =====================
# Quit experiment setup
# =====================

keys = kb.getKeys(['p'])
if "p" in keys:
    core.quit()

#------------------------------- Functions --------------------------------#

# Start ET calibration
def start_calibration():
    
    if ET == 1:
        # Calibration instructions
        calibration = visual.TextStim(
            win=win, 
            text='Now we will calibrate the eyetracker. \n\nPlease keep your head as still as possible and follow the circle on the next screen. \n\n Please wait for the experimenter to start.',
            font='Arial',
            pos=[0, 0], height=36,color='black', units='pix', colorSpace='named',
            wrapWidth=win.size[0] * .9
        )
        
        # Draw text stimulus and wait until "enter" is pressed
        calibration.draw()
        win.flip()
        event.waitKeys(keyList=["return"])
        
        # Print calibration result (if calibration was passed)
        hideWindow(win)
        result = tracker.runSetupProcedure()
        print("Calibration returned: ", result)
        showWindow(win)
    
    # No calibration in debug mode
    elif ET == 0:
        core.wait(0)
    
    return


# Adjust Volume
def adjust_volume(win, audio_file_path, start_time=5, duration=10):
    """
    Allows participants to test and adjust the volume before starting the experiment.
    Plays a segment of the actual audio files.

    Args:
        win: PsychoPy window object.
        audio_file_path (str): Path to the audio file for volume testing.
        start_time (float): Start time (in seconds) for the audio segment.
        duration (float): Duration (in seconds) to play the audio segment.
    """
    # Instruction for volume adjustment
    volumeInstructions = visual.TextStim(
        win=win,
        text="Adjust the volume using the system's controls. You will hear a short segment of the audio.\n\n"
             "When you're satisfied with the volume, press ENTER to proceed.\n\n"
             "If you can't hear the audio, please notify the experimenter.",
        font='Arial',
        pos=[0, 0],
        height=36,
        color='black',
        units='pix',
        wrapWidth=win.size[0] * 0.9,
    )

    # Verify the file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file '{audio_file_path}' not found! Skipping volume adjustment.")
        return

    # Load the audio file
    test_sound = sound.Sound(audio_file_path, stereo=True, startTime=start_time, stopTime=start_time + duration)

    # Display instructions and play the test sound in a loop
    while True:
        volumeInstructions.draw()
        win.flip()

        # Play the audio segment
        test_sound.play()
        core.wait(duration)  # Wait for the duration of the audio segment
        test_sound.stop()

        # Check for key press to proceed
        keys = event.getKeys(keyList=["return"])
        if "return" in keys:
            break

#----------------------------- Instructions -------------------------------#

# Instruction: Welcome
startInstructions = visual.TextStim(
    win=win, 
    name = 'instrStart',
    text='Welcome! Press ENTER to start the experiment.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Wait
waitInstructions = visual.TextStim(
    win=win, 
    name = 'instrWait',
    text='Please wait for the experimenter.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Video intro
storyfestIntroInstructions = visual.TextStim(
    win=win,
    name='instrVideoIntro',
    text="In this study, you will be listening to 6 short stories. \n Each story is between 4-12 minutes long, so it should take about 50 minutes total. \n Following the stories, you will be asked to recount what you heard. \n\n\n\n\n\nPress ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

# Instruction: Video ready
storyfestReadyInstructions1 = visual.TextStim(
    win=win, 
    name='instrVideoReady',
    text='While listening to each story,\nplease try to keep your head as still as possible and refrain from moving. \nPlease stare at the cross in the center of the screen as you listen to the stories.\n\n\n Press ENTER to continue.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

storyfestReadyInstructions2 = visual.TextStim(
    win=win, 
    name='instrVideoReady',
    text='The six stories will be split into two segments, each around 25 minutes long.\nYou will be given a break in between the segments.\n\n\n Press ENTER to continue.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Video ready
storyfestBeginInstructions = visual.TextStim(
    win=win, 
    name='instrVideoBegin',
    text='Now, we will start the story listening part of the experiment. \n\n\n Press ENTER when you are ready to begin.',
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='named',
    wrapWidth=win.size[0] * .9
)

# Instruction: Between stories
betweenInstructions = visual.TextStim(
    win=win,
    name='instrBreak',
    text="You are done with the first listening segment.\n Please feel free to take a short break. \n\n\nWhenever you are ready, press ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.7
)

# Instruction: Break
breakInstructions = visual.TextStim(
    win=win,
    name='instrBreak',
    text="You are done with the listening portion of the experiment.\n Please feel free to take a short break. \n\n\nWhenever you are ready, press ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.7
)

#------------------------------ Stimuli -----------------------------------#

# Get file names sheet
stim_path = os.path.join(_thisDir, '..', 'stimuli')
df = pd.read_excel(stim_path + '/file_names.xlsx')

# Get which video order to play (3 different configs of stimuli)

# Order 1
if int(expInfo['group']) == 1: 
    link1 = df.at[0,"VideoList"] 
    link2 = df.at[1, "VideoList"] 


# Order 2
elif int(expInfo['group']) == 2: 
    link1 = df.at[2,"VideoList"] 
    link2 = df.at[3, "VideoList"] 


# Order 3
elif int(expInfo['group']) == 3: 
    link1 = df.at[4,"VideoList"] 
    link2 = df.at[5, "VideoList"] 

# Set up stimuli in order
story1 = sound.Sound(
    link1,
    stereo=True
)

story2 = sound.Sound(
    link2, 
    stereo=True
)

#--------------------------- Start Experiment -----------------------------#


# Prep for ET calibration
startInstructions.draw()
win.flip()
event.waitKeys(keyList=['return'])

# ==================
# Run ET Calibration
# ==================
if ET == 1:
    start_calibration()

adjust_volume(win, audio_file_path=link1, start_time=0, duration=10)

# Show instruction: Wait
waitInstructions.draw()
win.flip()
event.waitKeys(keyList=["return"])

# This marks the start of the main experiment. 
mainExpClock = core.Clock() # Timer for tracking time stamps

# Show instruction: Video intro
storyfestIntroInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# Show instruction: Video ready
storyfestReadyInstructions1.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

storyfestReadyInstructions2.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# Show instruction: Video begin
storyfestBeginInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# ==============================
# Send story start message to ET
# ==============================
if ET == 1:
    tracker.sendMessage("ALL_STORY_START")
    dfTimeStamps.loc[0, 'startETallstory'] = mainExpClock.getTime()

# ==============
# Begin stimulus
# ==============

# Show the black cross
crossCentralBlack.draw()
win.flip()

# Record start time
dfTimeStamps.loc[0, "storyStart_1"] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data

# Play audio
story1.play()
paused = False
pause_count = 0
    
# Keep the dot until the audio finishes playing
while story1.status == PLAYING or paused:

    # Check for key presses
    keys = event.getKeys(keyList=['k', 'p'])

    # To end audio
    if 'k' in keys: # K for kill
        if KS == 1:
            story1.stop(reset=True)
            break
    elif 'p' in keys: # P for pause
        if not paused:
            # Pause the audio
            story1.pause()
            dfTimeStamps.loc[pause_count, 'storyPause_1'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data
            paused = True
        else:
            story1.play()
            dfTimeStamps.loc[pause_count, 'storyRestart_1'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data
            paused = False
            pause_count +=1
            
    # Redraw the dot to keep on screen
    crossCentralBlack.draw()
    win.flip()

    # Keep window open until audio finishes playing 
    core.wait(0.1)

# ============================
# Send story end message to ET
# ============================
if ET == 1:
    tracker.sendMessage("STORY_1_END")
    dfTimeStamps.loc[0, 'endETstory_1'] = mainExpClock.getTime()

# record end time
dfTimeStamps.loc[0, "storyEnd_1"] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # save partial data

# show instruction: between
betweenInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# ============================
# Send story start message to ET
# ============================
if ET == 1:
    tracker.sendMessage("STORY_2_START")
    dfTimeStamps.loc[0, 'startETstory_2'] = mainExpClock.getTime()

# Record start time
dfTimeStamps.loc[0, "storyStart_2"] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data

# start second audio
story2.play()
paused = False
pause_count = 0

# Keep the dot until the audio finishes playing
while story2.status == PLAYING or paused:

    # Check for key presses
    keys = event.getKeys(keyList=['k', 'p'])

    # To end audio
    if 'k' in keys: # K for kill
        if KS == 1:
            story2.stop(reset=True)
            break
    elif 'p' in keys: # P for pause
        if not paused:
            # Pause the audio
            story2.pause()
            dfTimeStamps.loc[pause_count, 'storyPause_2'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data
            paused = True
        else:
            story2.play()
            dfTimeStamps.loc[pause_count, 'storyRestart_2'] = mainExpClock.getTime()
            dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data
            paused = False
            pause_count +=1
            
    # Redraw the dot to keep on screen
    crossCentralBlack.draw()
    win.flip()

    # Keep window open until audio finishes playing 
    core.wait(0.1)

# ============================
# Send story end message to ET
# ============================
if ET == 1:
    tracker.sendMessage("STORY_2_END")
    dfTimeStamps.loc[0, 'endETstory_2'] = mainExpClock.getTime()
    
# Record end time
dfTimeStamps.loc[0, "storyEnd_2"] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_encoding_timestamps.csv', index=False)  # Save partial data

# show instruction: break
breakInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# --------------------- Post-Experiment Settings --------------------------#
# Stop recording ET
if ET == 1:
    tracker.setRecordingState(False)

win.close()

# Disconnect ET
if ET == 1:
    tracker.setConnectionState(False)

# Quit IO Hub
io.quit()

# Explort participant ET data
if ET == 1:
    edf_root = ''
    edf_file = edf_root + '/' + filename + '.EDF'
    os.rename('storyfest_eyetracker_encoding.EDF', edf_file)

# Close experiment
core.quit()




