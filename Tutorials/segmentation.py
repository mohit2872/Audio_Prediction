# Open the python terminal at the root directory

import librosa
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pydub import AudioSegment

# Loading the audio from 30 to 35 secs in the 'y'
y, sr = librosa.load("./Tutorials/sample.wav", offset=30, duration=5.0)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
actual_frames = librosa.frames_to_time(onset_frames, sr=sr)

# For getting the first onset from the audio
t1 = 30 * 1000 # start time
t2 = (30+actual_frames[0]) * 1000 # end time
newAudio = AudioSegment.from_wav("./Tutorials/sample.wav")
newAudio = newAudio[t1:t2]
newAudio.export('./Tutorials/newSample.wav', format="wav")

# Getting the MFCC
(rate,sig) = wav.read("./Tutorials/newSample.wav")
# we want 13*4 ,hence we are taking out all the further rows
mfcc_feat = mfcc(signal=sig, samplerate=rate, nfft=2048, ceplifter=4)[:4, :]