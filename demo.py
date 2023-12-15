import torch 
import torchaudio
import torch.nn as nn 
from dfadetect.models.raw_net2 import RawNet
from experiment_config import RAW_NET_CONFIG, feature_kwargs
from copy import deepcopy
from dfadetect.datasets import *
import os

# def list_files(directory):
#     try:
#         # List all files in the directory
#         files = os.listdir(directory)
        
#         # Filter out directories, if any
#         files = [file for file in files if os.path.isfile(os.path.join(directory, file))]
#         absolute_paths = [os.path.join(directory, file) for file in files if os.path.isfile(os.path.join(directory, file))]

#         return absolute_paths
#     except OSError as e:
#         print(f"Error reading directory {directory}: {e}")
#         return None

# # Replace 'your_directory_path' with the path of the directory you want to list files from
# directory_path = 'your_directory_path'

# file_names = list_files(directory_path)





# model= RawNet(deepcopy(RAW_NET_CONFIG), 'cpu').to('cpu')
current_model = RawNet(deepcopy(RAW_NET_CONFIG), 'cpu')
current_model.load_state_dict(torch.load('final/mfcc/raw_net/model_with_new/ckpt.pth'))
current_model = current_model.to('cpu')


oldpath_generated = 'dataset/genOld/ljspeech_hifiGAN/LJ001-0067_generated.wav'
oldpath_real ='../../LJSpeech-1.1/wavs/LJ001-0015.wav'
newpath_real = 'new_data_set/real/9.wav'
newpath_fake = 'new_data_set/generated/p1/1.wav'


waveform, sample_rate = torchaudio.load(oldpath_real)


with torch.no_grad():
    output = current_model(waveform)

print(output)
# model.load.state_dict(torch.load('newdata/mfcc/raw_net/home_group09-f2023_WaveFake_DeepMl_FakeDetectv_data_generated_p1/ckpt.pth'))
# model.eval()
