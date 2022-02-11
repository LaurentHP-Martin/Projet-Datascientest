# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:09:26 2022

@author: Laurent Martin
"""

###################################################################
import os
import re
import ffmpeg
import glob
import librosa
import sys
import youtube_dl
import subprocess
import pandas as pd
import numpy as np
import glob
import time
import pathlib
import librosa
import matplotlib.pyplot as plt
import torchaudio
import torch
from sklearn.metrics import classification_report
from IPython.display import Audio
import swifter
import streamlit as st
from PIL import Image
import cv2
import librosa
import librosa.display
import scipy.signal as signal
from scipy.io.wavfile import write
from IPython.display import Audio
import random
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, UpSampling2D, Cropping2D
from tensorflow.keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Deconv2D
from progressbar import ProgressBar
from pytube import YouTube
from stat import ST_CTIME
from keras import backend as K
from progressbar import ProgressBar

####################################################################
###################################################################

########################## Fonctions utiles ########################

# On ajoute @st.cache aux fonctions de sorte à ne les charger qu'une seule fois

####### POUR AFFICHAGE & ECOUTE ##############################################
####### MORCEAU ORIGINAL

@st.cache
def affiche_tempo_original(signal): # Pour afficher le signal temporel du morceau original youtube
    
    plt.plot(signal)
    plt.xlabel('Pas de temps')
    plt.ylabel('Amplitude')
    plt.savefig("tempo_original.png",dpi=300)

@st.cache
def affiche_spectro_original(signal): # Pour afficher le spectrogramme du morceau original youtube
    
    n_fft = 1024 
    hop_length = 256
    D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    plt.figure(figsize=(17,10))
    DB = librosa.amplitude_to_db(D, ref=np.min)
    librosa.display.specshow(DB, hop_length=hop_length, x_axis='time', y_axis='mel', htk=True, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_original.png",dpi=300)
    
        
@st.cache
def get_sound(audio,f): # Pour générer le fichier audio du morceau original
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio.wav', f,scaled)
    

####### POUR AFFICHAGE & ECOUTE ##############################################
####### MORCEAU resample à 11025

# Une fonction qui rajoute un zéro toutes les 4 valeurs d'une liste donnée:



@st.cache
def get_sound_11025(audio,f): # Pour générer le fichier audio du morceau original
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_11025.wav', f,scaled)

@st.cache
def affiche_tempo_11025(signal_original): # Pour afficher le signal temporel du morceau original youtube
    
    plt.plot(signal_original)
    
    
    plt.xlabel('Pas de temps')
    plt.ylabel('Amplitude')
    plt.savefig("tempo_11025.png",dpi=300)

@st.cache
def affiche_spectro_11025(signal): # Pour afficher le spectrogramme du morceau original youtube
    
    n_fft = 1024 
    hop_length = 256
    D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    plt.figure(figsize=(17,10))
    DB = librosa.amplitude_to_db(D, ref=np.min)
    librosa.display.specshow(DB, hop_length=hop_length, x_axis='time', y_axis='mel', htk=True, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_11025.png",dpi=300)
    
    
 ####### POUR AFFICHAGE & ECOUTE ##############################################
####### MORCEAU resample et restrict  à 11025  
    
    
@st.cache
def get_sound_restrict(audio,f): # Pour générer le fichier audio du morceau original
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_restrict.wav', f,scaled)

@st.cache
def affiche_tempo_restrict(signal_original): # Pour afficher le signal temporel du morceau original youtube
    
    plt.plot(signal_original)
    
    plt.xlabel('Pas de temps')
    plt.ylabel('Amplitude')
    plt.savefig("tempo_restrict.png",dpi=300)

@st.cache
def affiche_spectro_restrict(signal): # Pour afficher le spectrogramme du morceau original youtube
    
    n_fft = 1024 
    hop_length = 256
    D = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    plt.figure(figsize=(17,10))
    DB = librosa.amplitude_to_db(D, ref=np.min)
    librosa.display.specshow(DB, hop_length=hop_length, x_axis='time', y_axis='mel', htk=True, cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_restrict.png",dpi=300)    
    
    
    
    
    
    
####### POUR AFFICHAGE & ECOUTE ##############################################
####### SELON " Voix seule via VI "  
 

   
            
@st.cache    
def affiche_spectro_approche_VI(signal): # Pour afficher le spectrogramme du vocal selon approche VI
    plt.figure(figsize=(17,10))
    librosa.display.specshow(10*np.log((np.abs(signal)**2+1e-12)/1e-12), cmap='coolwarm')
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_projet_VI.png",dpi=300)
    
    
@st.cache
def get_sound_VI(audio,f):
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_vox_VI.wav', f,scaled)    


####### POUR AFFICHAGE & ECOUTE ##############################################
####### SELON " Voix seule via VAD + VI "  

@st.cache    
def affiche_spectro_approche_VAD_VI(signal): # Pour afficher le spectrogramme du vocal selon approche VI
    plt.figure(figsize=(17,10))
    librosa.display.specshow(10*np.log((np.abs(signal)**2+1e-12)/1e-12), cmap='coolwarm')
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_projet_VAD_VI.png",dpi=300)
    
    
@st.cache
def get_sound_VAD_VI(audio,f):
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_vox_VAD_VI.wav', f,scaled)  


####### POUR AFFICHAGE & ECOUTE ##############################################
####### SELON " Voix seule via UNET "  

@st.cache    
def affiche_spectro_approche_UNET(signal): # Pour afficher le spectrogramme du vocal selon approche VI
    plt.figure(figsize=(17,10))
    librosa.display.specshow(10*np.log((np.abs(signal)**2+1e-12)/1e-12), cmap='coolwarm')
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_projet_UNET.png",dpi=300)
    
    
@st.cache
def get_sound_UNET(audio,f):
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_vox_UNET.wav', f,scaled)  


####### POUR AFFICHAGE & ECOUTE ##############################################
####### SELON " Voix seule via Generator VI "  

@st.cache    
def affiche_spectro_approche_Generator_VI(signal): # Pour afficher le spectrogramme du vocal selon approche VI
    plt.figure(figsize=(17,10))
    librosa.display.specshow(10*np.log((np.abs(signal)**2+1e-12)/1e-12), cmap='coolwarm')
    
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel ('time', fontsize=13)
    plt.ylabel ('mels', fontsize=13)
    plt.savefig("spectro_projet_Generator_VI.png",dpi=300)
    
    
@st.cache
def get_sound_Generator_VI(audio,f):
    
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    write('audio_vox_Generator_VI.wav', f,scaled)  

 

############################################################################################
############################################################################################


########################### A L'AFFICHE - HEADER ##########################

logo = Image.open('logo.jpg')

st.image(logo,use_column_width = False)

st.title('Neural Net_Vox')

title = st.text_input('lien Youtube')



########################### A L'AFFICHE - Choix  ###################

add_selectbox = st.selectbox(
    "Que souhaitez-vous afficher",
    ("Morceau initial","Morceau resample à 11025 hz","Voix seule via VI", "Voix seule via VAD + VI", "Voix seule via UNET",'Voix seule via Generator VI' )
)


########################### CACHE - CODE  ###################

#########
######### RECUP MORCEAU YOUTUBE
#########

@st.cache
def get_muz(title):
    video_url = title

    video_info = youtube_dl.YoutubeDL().extract_info(url = video_url,download=False)

    ydl_opts = {
        'verbose': True,
        'format': 'bestaudio/best',
        'outtmpl': 'Z.%(ext)s',
        'noplaylist' : True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            }],
        }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_info['webpage_url']])


    path = 'C:/Users/Utilisateur/Desktop/Soutenance_final'
    liste = glob.glob(path+'/*.wav', recursive=True) # récupérer les fichiers audio du dossier soutenance


    source_mixture,freq = librosa.load (liste[-1], sr=None)
    source_mixture = source_mixture[400000:1200000]
    return source_mixture,freq

source_mixture,freq = get_muz(title)
    
# PREPA Affichage morceau youtube    
visuel_tempo = affiche_tempo_original(source_mixture) # Pour générer le png du signal temporel du morceau youtube
visuel_spectro = affiche_spectro_original(source_mixture) # Pour générer le png du spectrogram du morceau youtube

audio = source_mixture # Pour générer le signal audio du morceau youtube
#scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
#write('audio.wav', freq,scaled)    

get_sound(audio,freq)

#########
######### Morceau à 11025  ##########################################################
#########

@st.cache
def resample(muz,f):

    muz_11025 = librosa.resample(muz, f, 11025)
    
    return muz_11025
    
audio_11025 = resample(source_mixture,freq)



    
get_sound_11025(audio_11025,11025)    

affiche_tempo_11025(source_mixture)
affiche_spectro_11025(audio_11025)   


#########
######### Morceau à 11025hz et restriction aux 400 premièresfrequences  ##########################################################
#########

@st.cache
def resample_and_restrict(muz,f):

    muz_11025 = librosa.resample(muz, f, 11025)
    f, t, Zxx1 = signal.stft(muz_11025, fs=11025, nperseg=1024)
    
    restriction = 400
    
    Zxx1_restrict = Zxx1[:,:]
    
    signal_restrict = signal.istft(Zxx1_restrict, fs=11025)
    
    return signal_restrict

audio_restrict = np.array(resample_and_restrict(source_mixture,freq))[1,:]
    
get_sound_restrict(audio_restrict,11025)    

affiche_tempo_restrict(source_mixture)
affiche_spectro_restrict(audio_restrict)


#########
######### APPROCHE VI ##########################################################
#########


def custom_mse(y_true, y_pred):
    mask1 = y_true
    mask2 = 1 - y_true
    lambda1 = 1.5 # (y_true = 1 et y_pred = 0) renforce l'erreur 
    lambda2 = 1
    return K.mean(K.square(y_pred - y_true)*mask1*lambda1 + K.square(y_pred - y_true)*mask2*lambda2, axis=-1)

def a1(y_true, y_pred):
    somme = K.sum(y_true)
    if somme == 0 :
      return 0.0
    else:
      return K.sum(y_true * K.round(y_pred)) / somme

def a2(y_true, y_pred):
    return K.sum((1-y_true) * (1-K.round(y_pred))) / K.sum(1-y_true)

@st.cache
def get_predVI(): # le code du DS21-P10-C-VI

    taille =10

    loaded_model = tf.keras.models.load_model('model_saved_VI.h5',custom_objects={'custom_mse':custom_mse,'a1':a1,'a2':a2}) # On charge le modele
    
    source_mixture_ech = librosa.resample(source_mixture, freq, 11025)
    
    # Passage dans le domaine spectral
    f, t, Zxx1 = signal.stft(source_mixture_ech, fs=11025, nperseg=1024)

    pbar = ProgressBar()

    Zxx1_norm = 10*np.log((np.abs(Zxx1)**2+1e-12)/1e-12)
    Zxx1_norm = Zxx1_norm/np.max(Zxx1_norm)

    a, b = Zxx1.shape

    pred_VI = np.zeros((513,b-taille))
    
    

    for j in pbar(range(0,b-taille,1)):   
        Zxx1_i_VI = Zxx1_norm[:,0+j:taille+1+j]
        Zxx1_i_VI = Zxx1_i_VI[np.newaxis,:,:,np.newaxis]
        
        

        prediction_VI = loaded_model.predict(Zxx1_i_VI)
        prediction_VI_r = np.round(prediction_VI)

        pred_VI[:,j] = Zxx1[:,int(taille/2)+j]*prediction_VI_r[0,:]
        
              
    
    pred_VI = np.concatenate((np.zeros((513,int(taille/2))),pred_VI,np.zeros((513,int(taille/2)))),axis=1)
        
    return pred_VI



pred_VI = get_predVI()


signal_reconstruit0 = signal.istft(pred_VI, fs=11025)


#### PREPA Affichage morceau youtube selon VI

spectro = affiche_spectro_approche_VI(pred_VI) # le spectrogramme du mask*mix

audio_bis = np.array(signal_reconstruit0)[1,:]
 
get_sound_VI(audio_bis,11025)


############################################################################

#########
######### APPROCHE VAD + VI ##########################################################
#########


@st.cache
def get_pred_VAD_VI(): # le code du DS21-P10-C-VI

    taille =10
    restriction = 400

    model = keras.models.load_model('model_saved_VAD_VI_VAD.h5')

    model2 = keras.models.load_model('model_saved_VAD_VI_VI.h5', custom_objects={"custom_mse": custom_mse, "a1" : a1, "a2" : a2})
    
    source_mixture_ech = librosa.resample(source_mixture, freq, 11025)
    
    # Passage dans le domaine spectral
    f, t, Zxx1 = signal.stft(source_mixture_ech, fs=11025, nperseg=1024)

    pbar = ProgressBar()

    Zxx1_norm = 10*np.log((np.abs(Zxx1)**2+1e-12)/1e-12)
    Zxx1_norm = Zxx1_norm/np.max(Zxx1_norm)

    a, b = Zxx1.shape
    pred_VAD = np.zeros((b-taille))
    pred_VI = np.zeros((513,b-taille))
    
    

    for j in pbar(range(0,b-taille,1)):
        
        Zxx1_i_VAD = Zxx1_norm[0:restriction,0+j:taille+1+j]
        Zxx1_i_VAD = Zxx1_i_VAD[np.newaxis,:,:,np.newaxis]

        Zxx1_i_VI = Zxx1_norm[:,0+j:taille+1+j]
        Zxx1_i_VI = Zxx1_i_VI[np.newaxis,:,:,np.newaxis]
    
        prediction_VAD = model.predict(Zxx1_i_VAD)
        prediction_VAD = np.round(prediction_VAD)
    
        pred_VAD[j] = prediction_VAD

        if prediction_VAD == 1:
            prediction_VI = model2.predict(Zxx1_i_VI)
            prediction_VI_r = np.round(prediction_VI)

        pred_VI[:,j] = Zxx1[:,int(taille/2)+j]*prediction_VI_r[0,:]

    pred_VI = np.concatenate((np.zeros((513,int(taille/2))),pred_VI,np.zeros((513,int(taille/2)))),axis=1)
        
    return pred_VI  
        
        
pred_VAD_VI = get_pred_VAD_VI()


signal_reconstruit1 = signal.istft(pred_VAD_VI, fs=11025)


#### PREPA Affichage morceau youtube selon VAD_VI

spectro = affiche_spectro_approche_VAD_VI(pred_VAD_VI) # le spectrogramme 

audio_ter = np.array(signal_reconstruit1)[1,:]
 
get_sound_VAD_VI(audio_ter,11025)        
        
        
############################################################################

#########
######### APPROCHE UNET ##########################################################
#########        
        
import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(3,3), stride=(2, 2), padding=2) # ré-essayer de doubler / kernel size de 3,3 plutôt
        self.conv_bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.conv_bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.conv_bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.conv_bn4 = torch.nn.BatchNorm2d(64)
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.conv_bn5 = torch.nn.BatchNorm2d(128)
        self.conv6 = torch.nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.conv_bn6 = torch.nn.BatchNorm2d(256)
      
        self.deconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.deconv_bn1 = torch.nn.BatchNorm2d(128)
        self.dropout1 = torch.nn.Dropout2d(0.2)
        self.deconv2 = torch.nn.ConvTranspose2d(256, 64, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.deconv_bn2 = torch.nn.BatchNorm2d(64)
        self.dropout2 = torch.nn.Dropout2d(0.2)
        self.deconv3 = torch.nn.ConvTranspose2d(128, 32, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.deconv_bn3 = torch.nn.BatchNorm2d(32)
        self.dropout3 = torch.nn.Dropout2d(0.2)
        self.deconv4 = torch.nn.ConvTranspose2d(64, 16, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.deconv_bn4 = torch.nn.BatchNorm2d(16)
        self.deconv5 = torch.nn.ConvTranspose2d(32, 8, kernel_size=(3,3), stride=(2, 2), padding=2)
        self.deconv_bn5 = torch.nn.BatchNorm2d(8)
        self.deconv6 = torch.nn.ConvTranspose2d(16, 1, kernel_size=(3,3), stride=(2, 2), padding=2)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv_bn1(self.conv1(x)))
        h2 = F.leaky_relu(self.conv_bn2(self.conv2(h1)))
        h3 = F.leaky_relu(self.conv_bn3(self.conv3(h2)))
        h4 = F.leaky_relu(self.conv_bn4(self.conv4(h3)))
        h5 = F.leaky_relu(self.conv_bn5(self.conv5(h4)))
        h = F.leaky_relu(self.conv_bn6(self.conv6(h5)))

        h = self.dropout1(F.relu(self.deconv_bn1(self.deconv1(h, output_size = h5.size()))))
        h = torch.cat((h, h5), dim=1)
        h = self.dropout2(F.relu(self.deconv_bn2(self.deconv2(h, output_size = h4.size()))))
        h = torch.cat((h, h4), dim=1)
        h = self.dropout3(F.relu(self.deconv_bn3(self.deconv3(h, output_size = h3.size()))))
        h = torch.cat((h, h3), dim=1)
        h = F.relu(self.deconv_bn4(self.deconv4(h, output_size = h2.size())))
        h = torch.cat((h, h2), dim=1)
        h = F.relu(self.deconv_bn5(self.deconv5(h, output_size = h1.size())))
        h = torch.cat((h, h1), dim=1)
        h = F.sigmoid(self.deconv6(h, output_size = x.size()))
        return h
       
        
model_UNET = Net()
model_UNET.load_state_dict(torch.load('model_UNET.pth',map_location=torch.device('cpu')))
model_UNET.eval() 


def get_pred_UNET(): # le code du DS21-P10-C-VI

    taille =128
    

    
    
    source_mixture_ech = librosa.resample(source_mixture, freq, 11025)
    
    # Passage dans le domaine spectral
    f, t, Zxx1 = signal.stft(source_mixture_ech, fs=11025, nperseg=1024)

    pbar = ProgressBar()

    Zxx1 = 10*np.log((np.abs(Zxx1)**2+1e-12)/1e-12)/10/np.log((np.max(np.abs(Zxx1))**2+1e-12)/1e-12)

    a, b = Zxx1.shape
    b2 = np.floor(b/taille)
    outputs = np.zeros((513,b))
    outputs = np.array(outputs)
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    

    device = torch.device(dev)

# Lancement du modèle
    for j in range(b2.astype(int)):
   
        inputs = Zxx1[:,j*taille:(j+1)*taille]

        inputs = torch.tensor(inputs) 
        inputs = torch.unsqueeze(inputs,0)
        inputs = torch.unsqueeze(inputs,0)
        inputs = inputs.to(device)

        outputs_net = model_UNET(inputs.float()).detach()
        outputs_net = outputs_net * inputs.float()

        outputs_net = torch.squeeze(outputs_net,0)
        outputs_net = torch.squeeze(outputs_net,0)

        outputs[:,j*taille:(j+1)*taille] = outputs_net.cpu().detach().numpy()
    

    phase  = np.angle(np.array(Zxx1[:,0:b]))
    stft = np.sqrt(np.exp(outputs*10*np.log((np.max(np.abs(Zxx1))**2+1e-12)/1e-12)/10)*1e-12-1e-12) * np.exp(1j * phase)
        
    return stft

pred_UNET = get_pred_UNET()

signal_reconstruit3 = signal.istft(pred_UNET, fs=11025)


#### PREPA Affichage morceau youtube selon VAD_VI

spectro = affiche_spectro_approche_UNET(pred_UNET) # le spectrogramme 

audio_quart = np.array(signal_reconstruit3)[1,:]
 
get_sound_UNET(audio_quart,11025) 



############################################################################

#########
######### APPROCHE Generator VI ##########################################################
#########  


@st.cache
def get_pred_Generator_VI(): # le code du DS21-P10-C-VI

    taille =10

    model_gen = tf.keras.models.load_model('model_saved_VI_generator.h5',custom_objects={'custom_mse':custom_mse,'a1':a1,'a2':a2}) # On charge le modele
    
    source_mixture_ech = librosa.resample(source_mixture, freq, 11025)
    
    # Passage dans le domaine spectral
    f, t, Zxx1 = signal.stft(source_mixture_ech, fs=11025, nperseg=1024)

    pbar = ProgressBar()

    Zxx1_norm = 10*np.log((np.abs(Zxx1)**2+1e-12)/1e-12)
    Zxx1_norm = Zxx1_norm/np.max(Zxx1_norm)

    a, b = Zxx1.shape

    pred_VI = np.zeros((513,b-taille))
    
    

    for j in pbar(range(0,b-taille,1)):   
        Zxx1_i_VI = Zxx1_norm[:,0+j:taille+1+j]
        Zxx1_i_VI = Zxx1_i_VI[np.newaxis,:,:,np.newaxis]
        
        

        prediction_VI = model_gen.predict(Zxx1_i_VI)
        prediction_VI_r = np.round(prediction_VI)

        pred_VI[:,j] = Zxx1[:,int(taille/2)+j]*prediction_VI_r[0,:]
        
              
    
    pred_VI = np.concatenate((np.zeros((513,int(taille/2))),pred_VI,np.zeros((513,int(taille/2)))),axis=1)
        
    return pred_VI



pred_gen_VI = get_pred_Generator_VI()


signal_reconstruit4 = signal.istft(pred_gen_VI, fs=11025)


#### PREPA Affichage morceau youtube selon VI

spectro = affiche_spectro_approche_Generator_VI(pred_gen_VI) # le spectrogramme du mask*mix

audio_cinq = np.array(signal_reconstruit4)[1,:]
 
get_sound_Generator_VI(audio_cinq,11025)



########################### A L'AFFICHE - Show result  ###################
# PREPA Affichage

Header_col_gauche =0
Header_col_centre =0
taille = 10


if add_selectbox == "Morceau initial":
    
    image_gauche = "tempo_original.png"
    image_centre = "spectro_original.png"
    
    Header_col_gauche = "Signal Temporel"
    Header_col_centre = "Spectrogram"
    
    AUDIO = 'audio.wav'
    
    
    
if add_selectbox == "Voix seule via VI":
    image_gauche = "spectro_original.png"
    image_centre = "spectro_projet_VI.png"
    
    Header_col_gauche = "Spectrogram morceau original"
    Header_col_centre = "Spectrogram voix seule reconstruite"
    
    AUDIO = 'audio_vox_VI.wav'
    


if add_selectbox == "Voix seule via VAD + VI":
    image_gauche = "spectro_original.png"
    image_centre = "spectro_projet_VAD_VI.png"
    
    Header_col_gauche = "Spectrogram morceau original"
    Header_col_centre = "Spectrogram voix seule reconstruite"
    
    AUDIO = 'audio_vox_VAD_VI.wav'



if add_selectbox == "Voix seule via UNET":
    image_gauche = "spectro_original.png"
    image_centre = "spectro_projet_UNET.png"
    
    Header_col_gauche = "Spectrogram morceau original"
    Header_col_centre = "Spectrogram voix seule reconstruite"
    
    AUDIO = 'audio_vox_UNET.wav'


if add_selectbox == "Voix seule via Generator VI":
    image_gauche = "spectro_original.png"
    image_centre = "spectro_projet_Generator_VI.png"
    
    Header_col_gauche = "Spectrogram morceau original"
    Header_col_centre = "Spectrogram voix seule reconstruite"
    
    AUDIO = 'audio_vox_Generator_VI.wav'


if add_selectbox == "Morceau resample à 11025 hz":
    image_gauche = "tempo_11025.png"
    image_centre = "spectro_11025.png"
    
    Header_col_gauche = "Signal Temporel"
    Header_col_centre = "Spectrogram"
    
    AUDIO = 'audio_11025.wav'

#if add_selectbox == "Morceau resample à 11025hz et restriction de fréquences":
   # image_gauche = "tempo_restrict.png"
    #image_centre = "spectro_restrict.png"
    
   # Header_col_gauche = "Signal Temporel"
   # Header_col_centre = "Spectrogram "
    
   # AUDIO = 'audio_restrict.wav'




#  Affichage


with st.expander(Header_col_gauche):
     visu_image_gauche = Image.open(image_gauche)
     st.write("""
         
     """)
     st.image(visu_image_gauche)

    
with st.expander(Header_col_centre):
     visu_image_centre = Image.open(image_centre)
     st.write("""
         
     """)
     st.image(visu_image_centre)

with st.expander("Ecoute"):
     audio_file = open(AUDIO,'rb')
     audio_bytes = audio_file.read()
     st.write("""
         
     """)
     st.audio(audio_bytes, format='wav')
     
     
     
st.balloons()
st.success("Prêt pour écoute !")




















