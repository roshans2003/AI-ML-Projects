#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))

nltk.download('punkt')


# In[19]:


file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)


# In[20]:


vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = [] 
patterns = [] 
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# In[21]:


def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# In[22]:


counter = 0


# In[24]:


def main():
    global counter
    st.title("Intents based Chatbot using NLP by Roshanoos")

    menu = ["Home", "Conversation History" , "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot made by Roshan ;-)  TYPE MESSAGE AND PRESS ENTER TO CHAT")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv','w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You: ", key=f"user_input_{counter}")

        if user_input:

            user_input_str = str(user_input)
            st.text_area("chatbot:", value=response, height= 120, max_chars = None, key=f"chatbot{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodboy', 'bye']:
                st.write("Thank you for chatting with me. Have a great day! :-) ")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv','r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot using Natural Language Processing (NLP) that acts like a friendly companion engaging users in casual, fun, or meaningful conversations.")

        st.subheader("Project Overview")

        st.write("""
        -To develop a chatbot that can understand and generate human-like responses.
        -To Ensure the responses are friendly, engaging, and contextually relevant.
        -To Build an interactive, user-friendly interface for smooth and enjoyable conversations.
        """)
        st.subheader("Dataset:")
        st.write("""
        -The dataset is formatted as a JSON file containing a dictionary of key-value pairs.
        -Each key represents a specific category, such as "greeting," "about," or "budget."
        -The values hold relevant data for the corresponding key, such as phrases, descriptions, or questions.
        -For example, the "greeting" key might store phrases like "Hello!" or "Hi, how can I help you?" while the "budget" key might include financial queries or details.
        -This structure is particularly useful in chatbot development for defining response categories or in NLP tasks for organizing data to train models.
        -The JSON format is lightweight, easy to read, and straightforward to parse, modify, and integrate into applications.
        -It is both scalable and flexible, making it suitable for dynamic and evolving projects.
        """)
        st.subheader("Streamlit Chatbot Interface:")
        st.write("Streamlit is an open-source Python library for building interactive, data-driven web applications quickly and easily.")
        st.subheader("Conclusion:")
        st.write("This Intent-Based Chatbot project showcases the power of AI and NLP to create a friendly, engaging, and interactive conversational companion.")
        


# In[ ]:




