# Authors: Kruthi Gollapudi (kruthig@uchicago.edu), Jadyn Park (jadynpark@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: January 31, 2025
# Description: This script is used to run the recall portion of the StoryFest experiment.

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

# =============================
# Toggle voice recording:
# 0=Don't record, 1=Record
# =============================
REC = 1

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
dfTimeStamps.to_csv(filename + '_recall_timestamps.csv', index=False)

#------------------------- Experiment Settings ----------------------------#

# =================
# Microphone setup
# =================

if REC == 1: 
    recordingDevicesList = Microphone.getDevices()
    device=recordingDevicesList[0] # Double check that this corresponds to the external mic!
    mic = Microphone(streamBufferSecs=3000.0,
                        sampleRateHz=48000,
                        device=recordingDevicesList[0],
                        channels=1,
                        maxRecordingSize=300000 * 2,
                        audioRunMode=0
    )


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

# Establish connection with eye tracker
if ET == 1:
    tracker.setConnectionState(True)

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

# Instruction: Record intro
recordIntroInstructions = visual.TextStim(
    win=win,
    name='instrRecordIntro',
    text="Now, we would like you to recount, in your own words, \nthe events of the stories in the original order they were experienced in, with as much detail as possible. \n\nSpeak for at least 20 min if possible -- but the longer the better. \n\nPlease verbally indicate when you are finished by saying, for example, \"I'm done.\" \n\n\n Press ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

recordReadyInstructions = visual.TextStim(
    win=win,
    name='instrRecordReady',
    text="Completeness and detail are more important than temporal order. \n\nIf at any point you realize that you missed something, feel free to return to it. \n\n\n Press ENTER to continue.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

recordBeginInstructions = visual.TextStim(
    win=win,
    name='instrRecordBegin',
    text="When you press ENTER to begin the recording portion of the experiment, the microphone will automatically turn on. \n\nPlease do NOT touch/move the microphone. \n\nThere will be a black cross on the screen, keep your eyes on it during recording. \n When you are finished speaking, press ENTER again to stop recording. \n\n\nPress ENTER to begin.",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
)

# Instruction: Finish
finishInstructions = visual.TextStim(
    win=win,
    name='instrFinish',
    text="Thank you for your participation! \n\n\n\n\n Please let the experimenter know you have finished!",
    font='Arial',
    pos=[0, 0], height=36, color='black', units='pix', colorSpace='rgb',
    wrapWidth=win.size[0] * 0.9
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

# show instruction: record intro
recordIntroInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# show instruction: record ready
recordReadyInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# show instruction: record begin
recordBeginInstructions.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# ==================================
# Send recording start message to ET
# ==================================
if ET == 1:
    tracker.sendMessage("REC_START")
    tracker.setRecordingState(True)
    dfTimeStamps.loc[0, 'startETVoiceRec'] = mainExpClock.getTime()

# start recording
mic.start()

# record start time
dfTimeStamps.loc[0,'recordStart'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_recall_timestamps.csv', index=False)  # save partial data

# show central white dot during recording
crossCentralBlack.draw()
win.flip()
keys = event.waitKeys(keyList=["return"])

# stop recording
mic.stop()

# record end time
dfTimeStamps.loc[0,'recordEnd'] = mainExpClock.getTime()
dfTimeStamps.to_csv(filename + '_recall_timestamps.csv', index=False)  # save data

# ==================================
# Send recording end message to ET
# ==================================
if ET == 1:
    tracker.sendMessage("REC_END")
    dfTimeStamps.loc[0, 'endETVoiceRec'] = mainExpClock.getTime()

# save audio
audioClip = mic.getRecording()
audioClip.save(filename + '.wav')

# show instruction: finish
finishInstructions.draw()
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
    edf_source = os.path.join(_thisDir, 'et_data.EDF')
    edf_target = os.path.join(_thisDir, '..', 'data', 'storyfest_eyetracker_recall.EDF')
    
    edf_path = os.path.dirname(edf_target)
    if not os.path.exists(edf_path):
        os.makedirs(edf_path)
    
    if os.path.exists(edf_source):
        os.rename(edf_source, edf_target)


# Close experiment
core.quit()




