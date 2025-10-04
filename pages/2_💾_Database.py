import streamlit as st 
import pickle 
import yaml 
import pandas as pd 
cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
PKL_PATH = cfg['PATH']["PKL_PATH"]
st.set_page_config(layout="wide")

#Load databse 
with open(PKL_PATH, 'rb') as file:
    database = pickle.load(file)

Index, MoodleID, Name, Image  = st.columns([0.6,0.9,3,3])

# Headers
with Index:
    st.markdown("**Index**")
with MoodleID:
    st.markdown("**Moodle ID**")
with Name:
    st.markdown("**Name**")
with Image:
    st.markdown("**Image**")

for idx, person in database.items():
    with Index:
        st.write(idx)
    with MoodleID: 
        st.write(person['id'])
    with Name:     
        st.write(person['name'])
    with Image:     
        st.image(person['image'],width=200)

