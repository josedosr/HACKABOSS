#!/usr/bin/env python
# coding: utf-8

# # Exploración Data Sets

# In[10]:


import os
import numpy as np
import pandas as pd
import tqdm


# In[11]:


[x for x in dir(os) if 'path' in x]


# In[12]:


path = 'Data/'


# # Péptidos 

# In[4]:


peptidos = pd.read_csv(f'{path}peptidos.csv', sep = ';')
peptidos.head()


# In[5]:


peptidos.shape


# In[6]:


diccionario_pep = {}

for pep in tqdm.tqdm(peptidos['sequence']):
    
    for char in pep:
        
        if char in diccionario_pep:
            
            diccionario_pep[char] += 1
            
        else:
            diccionario_pep[char] = 1
            
diccionario_pep


# In[7]:


peptidos.shape


# In[8]:


peptidos['sequence'].apply(lambda x: len(x)).value_counts()


# # Spotify Songs 2020

# In[40]:


import numpy as np
import pandas as pd

import requests
from pprint import pprint
import json

from dotenv import load_dotenv
import os

import base64

from time import sleep
import pickle

load_dotenv()


# In[3]:


CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')


# In[4]:


def get_token():
    auth_string = CLIENT_ID + ':' + CLIENT_SECRET
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')

    url = 'https://accounts.spotify.com/api/token'

    headers = {
        'Authorization': 'Basic ' + auth_base64,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {'grant_type': 'client_credentials'}
    result = requests.post(url, headers = headers, data = data)
    json_result = json.loads(result.content)
    token = json_result['access_token']
    return token


# In[5]:


token = get_token()
print(token)


# In[6]:


def get_auth_header(token):
    return {'Authorization': 'Bearer ' + token}


# In[127]:


def search(token, search):
    
    endpoint = 'https://api.spotify.com/v1/search'
    
    query = f'?q={str(search)}&type=artist&limit=1'
    
    headers = get_auth_header(token)
    
    query_url = endpoint + query
    result = requests.get(query_url, headers = headers)
    json_result = json.loads(result.content)
    
    return json_result

def search_playlist(token, search):
    
    endpoint = 'https://api.spotify.com/v1/search'
    
    query = f'?q={str(search)}&type=playlist&limit=20'
    
    headers = get_auth_header(token)
    
    query_url = endpoint + query
    result = requests.get(query_url, headers = headers)
    json_result = json.loads(result.content)
    
    return json_result

def search_playlist_v2(token, search):
    
    endpoint = 'https://api.spotify.com/v1/search'
    
    query = f'?q={str(search)}&type=playlist&limit=50'
    
    headers = get_auth_header(token)
    
    query_url = endpoint + query
    result = requests.get(query_url, headers = headers)
    json_result = json.loads(result.content)
    
    return json_result


def get_songs(token, artist_id):
    
    endpoint = f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=ES'
    
    headers = get_auth_header(token)

    result = requests.get(endpoint, headers = headers)
    json_result = json.loads(result.content)
    
    return json_result


# In[117]:


def get_songs_from_playlist(token, playlist_id):
    
    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    
    headers = get_auth_header(token)

    result = requests.get(endpoint, headers = headers)
    json_result = json.loads(result.content)
    
    return json_result


# In[14]:


songs = get_songs_from_playlist(token, '2PqDpi1qID9Zx4x9GdnjH6')


# In[204]:


spotify.head(1)


# In[272]:


spotify['playlist_genre'].value_counts().index


# In[20]:


songs['items']


# In[34]:


songs['items'][0]['track'].keys()


# In[230]:


songs['items'][0]['track']['album'].keys()


# In[271]:


songs['items'][0]['track']


# In[244]:


[x['name'] for x in songs['items'][0]['track']['artists']]


# In[251]:


results = []

for song in songs['items']:
    
    #track data
    track = song['track']
    track_id = track['id']
    track_name = track['name']
    track_artists = [artist['name'] for artist in track['artists']]
    track_popularity = track['popularity']
    
    #album data
    album = song['track']['album']
    track_album_id = album['id']
    track_album_name = album['name']
    track_album_release_date = album['release_date']
    
    #playlist data
    playlist = song[]
    
    print(track_album_release_date)
    
    break


# In[135]:


canciones = get_songs(token, artist_id)


# In[165]:


artista = search(token, 'Bizarrap')


# In[180]:


playlist = search_playlist(token, 'Bizarrap')


# In[259]:


playlist['playlists']['items'][0].keys()


# In[263]:


playlist['playlists']['items'][0]


# In[129]:


token = get_token()

genres = ['edm', 'rap', 'pop', 'r&b', 'latin', 'rock']

results = []

none_counter = 0

for genre in genres:
    
    playlists = search_playlist_v2(token, genre)
    playlists = playlists['playlists']['items']
    
    songs_counter = 0
    
    for playlist in tqdm.tqdm(playlists):
        
        #playlist data
        playlist_id = playlist['id']
        playlist_name = playlist['name']
        playlist_genre = genre
        
        songs = get_songs_from_playlist(token, playlist_id)
        songs = songs['items']
        
        for idx, song in enumerate(songs):
        
            try: 
                #track data
                track = song['track']
                
                """if track == None:
                    none_counter += 1
                    print(none_counter)"""

                if track is not None or track != None :

                    track_id = track['id']
                    track_name = track['name']
                    track_artists = [artist['name'] for artist in track['artists']]
                    track_popularity = track['popularity']

                    #album data
                    album = track['album']
                    track_album_id = album['id']
                    track_album_name = album['name']
                    track_album_release_date = album['release_date']

                    results.append([track_id, track_name, track_artists, track_popularity,track_album_id, track_album_name, track_album_release_date,playlist_name, playlist_id, playlist_genre])
            
            except:
                print(f'Ha fallado la cancion #{idx} dela playlist {playlist_name} - {playlist_id}')
            
        songs_counter += len(songs)    
            
        sleep(0.5) #Este sleep es para bajar el ritmo de llamadas a la API para sacar las canciones de cada playlist encontrado
        
    print(f'La API ha buscado datos de {genre}, ha sacado {len(playlists)} playlists y {songs_counter} canciones')    
        
    sleep(1.5) #Este sleep es para bajar el ritmo de llamadas a la API para sacar los playlists de cada genero
    
df = pd.DataFrame(results, columns = ['track_id', 'track_name', 'track_artist', 'track_popularity', 'track_album_id', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_genre'])
df.to_csv('Data/spotifyAPI_v2.csv', index = False)
df.to_pickle('Data/spotifyAPI_v2.pkl')


# In[128]:


search_playlist_v2(token, 'edm')


# In[136]:


test = pd.DataFrame(results, columns = ['track_id', 'track_name', 'track_artist', 'track_popularity', 'track_album_id', 'track_album_name', 'track_album_release_date', 'playlist_name', 'playlist_id', 'playlist_genre'])


# In[137]:


test


# In[119]:


spotify = pd.read_csv(f'{path}spotify.csv')
spotify.head(3)


# In[135]:


spotify.groupby('playlist_genre')['playlist_name'].nunique()


# ## About Dataset
# Almost 30,000 Songs from the Spotify API. See the readme file for a formatted data dictionary table.
# 
# ### Data Dictionary:
# 
# variable class description
# 
# track_id character Song unique ID
# 
# track_name character Song Name
# 
# track_artist character Song Artist
# 
# track_popularity double Song Popularity (0-100) where higher is better
# 
# track_album_id character Album unique ID
# 
# track_album_name character Song album name
# 
# track_album_release_date character Date when album released
# 
# playlist_name character Name of playlist
# 
# playlist_id character Playlist ID
# 
# playlist_genre character Playlist genre
# 
# playlist_subgenre character Playlist subgenre
# 
# danceability double Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
# 
# energy double Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
# 
# key double The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
# 
# loudness double The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
# 
# mode double Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
# 
# speechiness double Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
# 
# acousticness double A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
# 
# instrumentalness double Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
# 
# liveness double Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
# 
# valence double A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
# 
# tempo double The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
# 
# duration_ms double Duration of song in milliseconds

# In[14]:


spotify['playlist_genre'].value_counts(normalize = True).sort_values(ascending = False)


# In[15]:


spotify['playlist_genre'].value_counts().sort_values(ascending = False)


# # Pistachio types detection

# In[16]:


pistachio = pd.read_csv(f'{path}pistachio.csv')
pistachio.head(3)


# In[17]:


pistachio['Class'].unique()


# In[18]:


pistachio.shape


# # Avocado Price / Classification

# This data was downloaded from the Hass Avocado Board website in May of 2018 & compiled into a single CSV. Here's how the Hass Avocado Board describes the data on their website:
# 
# The table below represents weekly 2018 retail scan data for National retail volume (units) and price. Retail scan data comes directly from retailers’ cash registers based on actual retail sales of Hass avocados. Starting in 2013, the table below reflects an expanded, multi-outlet retail data set. Multi-outlet reporting includes an aggregation of the following channels: grocery, mass, club, drug, dollar and military. The Average Price (of avocados) in the table reflects a per unit (per avocado) cost, even when multiple units (avocados) are sold in bags. The Product Lookup codes (PLU’s) in the table are only for Hass avocados. Other varieties of avocados (e.g. greenskins) are not included in this table.
# 
# #### Some relevant columns in the dataset:
# 
# Date - The date of the observation
# 
# AveragePrice - the average price of a single avocado
# 
# type - conventional or organic
# 
# year - the year
# 
# Region - the city or region of the observation
# 
# Total Volume - Total number of avocados sold
# 
# 4046 - Total number of avocados with PLU 4046 sold
# 
# 4225 - Total number of avocados with PLU 4225 sold
# 
# 4770 - Total number of avocados with PLU 4770 sold

# In[19]:


avocado = pd.read_csv(f'{path}avocado.csv')
avocado.head(5)


# In[20]:


avocado['type'].unique()


# In[21]:


avocado.shape


# In[22]:


avocado['region'].unique()


# In[23]:


avocado.columns


# # Glassdoor Job Reviews NLP
# 
# #### About Dataset
# 
# This large dataset contains job descriptions and rankings among various criteria such as work-life balance, income, culture, etc. The data covers the various industry in the UK. Great dataset for multidimensional sentiment analysis.
# 
# #### Glassdoor Reviews
# 
# Glassdoor produces reports based upon the data collected from its users, on topics including work–life balance, CEO pay-ratios, lists of the best office places and cultures, and the accuracy of corporate job searching maxims. Data from Glassdoor has also been used by outside sources to produce estimates on the effects of salary trends and changes on corporate revenues. Glassdoor also puts the conclusions of its research of other companies towards its own company policies. In 2015, Tom Lakin produced the first study of Glassdoor in the United Kingdom, concluding that Glassdoor is regarded by users as a more trustworthy source of information than career guides or official company documents.
# 
# #### Features
# 
# The columns correspond to the date of the review, the job name, the job location, the status of the reviewers, and the reviews. Reviews are divided in s sub-categories Career Opportunities, Comp & Benefits, Culture & Values, Senior Management, and Work/Life Balance. In addition, employees can add recommendations on the firm, the CEO, and the outlook.
# 
# #### Other information
# 
# Ranking for the recommendation of the firm, CEO approval, and outlook are allocated categories v, r, x, and o, with the following meanings:
# v - Positive, r - Mild, x - Negative, o - No opinion
# 
# #### Some examples of the textual data entries
# 
# ##### MCDONALD-S
# I don't like working here,don't work here
# 
# Headline: I don't like working here,don't work here
# 
# Pros: Some people are nice,some free food,some of the managers are nice about 95% of the time
# 
# Cons: 95% of people are mean to employees/customers,its not a clean place,people barely clean their hands of what i see,managers are mean,i got a stress rash because of this i can't get rid of it,they don't give me a little raise even though i do alot of crap there for them
# 
# Rating: 1.0

# In[24]:


gd = pd.read_csv(f'{path}glassdoor_reviews.csv')
gd.head(5)


# In[25]:


gd['firm'].value_counts(ascending = False).head(10)


# In[26]:


gd['firm'].value_counts(ascending = False).head(10).sum()


# In[27]:


gd.shape


# In[28]:


gd.columns


# In[29]:


gd['ceo_approv'].value_counts()


# In[30]:


[x for x in dir(pd.DataFrame) if 'na' in x]


# In[31]:


gd.select_dtypes('number').columns


# In[32]:


gd['rating'] = gd[['work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt']].mean(axis = 1)


# In[33]:


gd[['overall_rating', 'rating']]


# In[34]:


gd[['work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt']].isna().sum().sort_values()


# In[35]:


df1 = gd.drop('diversity_inclusion', axis = 1).copy()


# In[36]:


df1 = df1.dropna(subset = ['work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt'])


# In[37]:


df1.shape, gd.shape


# In[38]:


df2[['work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt']].isna().sum().sort_values()


# In[ ]:


df1.head(1)


# In[ ]:




