import streamlit as st
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)


def ai_chatbot_page():
    st.title("AI Chatbot for Stock Insights")
    st.info("Ask questions about the stock analysis you just completed.")

    # Ensure analysis data exists in session state
    if "gathered_data" not in st.session_state:
        st.error("No stock analysis data found. Please run the Stock Analysis page first.")
        return
    
    analysis_data = st.session_state["gathered_data"]
    st.write(f"Chatbot is now trained with data for {analysis_data['Company']} ({analysis_data['Ticker']}).")

    # Initialize chat session state
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = []

    user_input = st.text_input("Enter your query:")

    if user_input:
        st.session_state["chatbot_messages"].append({"role": "user", "content": user_input})

        max_messages = 3  # Adjust the maximum number of messages as needed
        if len(st.session_state["chatbot_messages"]) > max_messages:
            st.session_state["chatbot_messages"] = st.session_state["chatbot_messages"][-max_messages:]

        # Construct the context for the chatbot
        chat_context = [
            {"role": "system", "content": "You are an AI assistant analyzing stock data based on gathered insights."},
            {"role": "system", "content": f"Analysis Data: {json.dumps(analysis_data)}"},
        ] + st.session_state["chatbot_messages"]

        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with your preferred model
            messages=chat_context
        )

        assistant_response = response.choices[0].message.content
        st.session_state["chatbot_messages"].append({"role": "assistant", "content": assistant_response})

        if st.button("Clear Chat"):
            st.session_state["chatbot_messages"] = []
            st.success("Chat history cleared!")

        for msg in st.session_state["chatbot_messages"]:
            if msg["role"] == "user":
                st.write(f"**You:** {msg['content']}")
            else:
                st.write(f"**AI Assistant:** {msg['content']}")



