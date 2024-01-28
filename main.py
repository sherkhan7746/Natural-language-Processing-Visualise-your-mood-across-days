import pandas as pd
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path
import streamlit as st
import plotly.express as px

notes = []
file_paths = glob.glob("diary/*.txt")
dates = []
for file_path in file_paths:
    # Extracting Dates from file Paths
    path = Path(file_path)
    dates.append(path.stem)
    # Appending the text from files into the list
    with open(file_path, "r") as file:
        notes.append(file.read())

# Calculating the Mood Score
analyzer = SentimentIntensityAnalyzer()
sentiments = []
for words in notes:
    sentiments.append(analyzer.polarity_scores(words))

# Generating the dataframe
df = pd.DataFrame(data=sentiments)
df["Date"] = dates
# Generating new columns and renaming it
df["Negativity"] = df["neg"]
df["Positivity"] = df["pos"]

st.title("Diary Tone")

# render positivity chart
st.subheader("Positivity")
figure = px.line(df, x="Date", y="Positivity")
st.plotly_chart(figure)
# Render negativity Chart
st.subheader("Negativity")
figure = px.line(df, x="Date", y="Negativity")
st.plotly_chart(figure)
