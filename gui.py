# Import libraries 
import tkinter as tk
from tkinter import ttk
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import argmax
from keras import layers
from keras.models import load_model
from keras.layers import TextVectorization
from keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.saving import register_keras_serializable

# ------------------------------
# Register Custom Layer (For Spanish Model)
# ------------------------------
@register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions

# ------------------------------
# English to Spanish Translation Setup
# ------------------------------
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "").replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

# Load English vectorization layer
with open('eng_vectorization_config.json') as json_file:
    eng_vectorization_config = json.load(json_file)

eng_vectorization = TextVectorization(
    max_tokens=eng_vectorization_config['max_tokens'],
    output_mode=eng_vectorization_config['output_mode'],
    output_sequence_length=eng_vectorization_config['output_sequence_length']
)
eng_vectorization.standardize = custom_standardization

# Load Spanish vectorization layer
with open('spa_vectorization_config.json') as json_file:
    spa_vectorization_config = json.load(json_file)

spa_vectorization = TextVectorization(
    max_tokens=spa_vectorization_config['max_tokens'],
    output_mode=spa_vectorization_config['output_mode'],
    output_sequence_length=spa_vectorization_config['output_sequence_length'],
    standardize=custom_standardization
)

# Load English & Spanish vocab
with open('eng_vocab.json') as json_file:
    eng_vocab = json.load(json_file)
    eng_vectorization.set_vocabulary(eng_vocab)

with open('spa_vocab.json') as json_file:
    spa_vocab = json.load(json_file)
    spa_vectorization.set_vocabulary(spa_vocab)

# Load Spanish model
try:
    transformer = load_model('transformer_model.keras', custom_objects={"PositionalEmbedding": PositionalEmbedding})
except Exception as e:
    print(f"Error loading Spanish model: {e}")
    transformer = None

spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sentence(input_sentence):
    if transformer is None:
        return "Error: Spanish model not loaded."
    
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    
    for _ in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = tf.argmax(predictions[0, -1, :]).numpy().item()
        sampled_token = spa_index_lookup.get(sampled_token_index, "")
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    
    return decoded_sentence.replace("[start]", "").replace("[end]", "").strip()

# ------------------------------
# English to French Translation Setup
# ------------------------------
try:
    model = load_model('english_to_french_model.keras')
except Exception as e:
    print(f"Error loading French model: {e}")
    model = None

with open('english_tokenizer.json') as f:
    english_tokenizer = tokenizer_from_json(json.load(f))

with open('french_tokenizer.json') as f:
    french_tokenizer = tokenizer_from_json(json.load(f))

with open('sequence_length.json') as f:
    max_length = json.load(f)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    if model is None:
        return "Error: French model not loaded."

    english_sentence = re.sub(r"[.!?,]", "", english_sentence.lower())
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    english_sentence = english_sentence.reshape((-1, max_length))

    french_sentence = model.predict(english_sentence)[0]
    french_sentence = [np.argmax(word) for word in french_sentence]
    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    return french_sentence.strip()

# ------------------------------
# GUI Application
# ------------------------------
def translate_text():
    selected_language = language_var.get()
    english_sentence = text_input.get("1.0", "end-1c").strip()

    if not english_sentence:
        translation_output.delete("1.0", "end")
        translation_output.insert("end", "Error: No text entered.")
        return

    if selected_language == "French":
        translation = translate_to_french(english_sentence)
    elif selected_language == "Spanish":
        translation = decode_sentence(english_sentence)
    else:
        translation = "Error: No language selected."

    translation_output.delete("1.0", "end")
    translation_output.insert("end", f"{selected_language} translation:\n{translation}")

# Setup GUI window
root = tk.Tk()
root.title("Language Translator")
root.geometry("550x600")

# Font configuration
font_style = "Times New Roman"
font_size = 14

# Frame for input
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# Heading for input
input_heading = tk.Label(input_frame, text="Enter the text to be translated", font=(font_style, font_size, 'bold'))
input_heading.pack()

# Text input for English sentence
text_input = tk.Text(input_frame, height=5, width=50, font=(font_style, font_size))
text_input.pack()

# Language selection
language_var = tk.StringVar()
language_label = tk.Label(root, text="Select the language to translate to", font=(font_style, font_size, 'bold'))
language_label.pack()
language_select = ttk.Combobox(root, textvariable=language_var, values=["French", "Spanish"], font=(font_style, font_size), state="readonly")
language_select.pack()

# Submit button
submit_button = ttk.Button(root, text="Translate", command=translate_text)
submit_button.pack(pady=10)

# Frame for output
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

# Heading for output
output_heading = tk.Label(output_frame, text="Translation: ", font=(font_style, font_size, 'bold'))
output_heading.pack()

# Text output for translations
translation_output = tk.Text(output_frame, height=10, width=50, font=(font_style, font_size))
translation_output.pack()

# Running the application
root.mainloop()
