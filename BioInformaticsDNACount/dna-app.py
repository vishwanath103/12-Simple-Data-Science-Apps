# Import Libraries
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

# Page Title
image = Image.open('dna-logo.jpg')
st.image(image, use_column_width=True)
st.write("""
# DNA Nucleotide Count Web App

This app counts the nucleotide composition of quaery DNA!

***
""")

# Input Text Box
st.header('Enter DNA sequence')

sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

sequence = st.text_area("Sequence input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:]
sequence = ''.join(sequence)

st.write("""
***
""")

# Prints the input DNA query
st.header('INPUT (DNA Query)')
sequence

# DNA Nucleotide count
st.header('OUTPUT (DNA Nucleotide Count)')

## 1. Print dictionary
st.subheader('1. Print Dictionary')
def DNA_nucleotide_count(seq):
    d = dict([
        ('A',seq.count('A')),
        ('T',seq.count('T')),
        ('G',seq.count('G')),
        ('C',seq.count('C'))
        ])
    return d
X = DNA_nucleotide_count(sequence)
X

## 2. Print text
st.subheader('2. Print text')
st.write('There are ' + str(X['A']) + ' adenine (A)')
st.write('There are ' + str(X['T']) + ' thymine (T)')
st.write('There are ' + str(X['G']) + ' guanine (G)')
st.write('There are ' + str(X['C']) + ' cytosine (C)')

## 3. Display Dataframe
st.subheader('3. Display Dataframe')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0: 'Count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns={'index':'nucleotide'})
st.write(df)

## 4. Display Bar chart using Altair
st.subheader('4. Display Bar Chart')
p = alt.Chart(df).mark_bar().encode(
        x='nucleotide',
        y='Count'
        )
p = p.properties(
        width=alt.Step(80)
        )
st.write(p)
