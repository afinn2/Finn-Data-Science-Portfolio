import streamlit as st
import pandas as pd

df = pd.read_csv("data/penguins.csv")

# Convert 'id' to string
df['id'] = df['id'].astype(str)

# User selection
penguin_id = st.selectbox("Select a penguin ID", df["id"])

# Filter based on user selection
filtered_penguin = df[df["id"] == penguin_id]

# Display the selected penguin's data
st.write(f"Data for Penguin {penguin_id}:")
st.dataframe(filtered_penguin)