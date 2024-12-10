import streamlit as st
from stock_page import stock_page
from chatbot_page import ai_chatbot_page


st.set_page_config(page_title="Stock Market and AI Chatbot", layout="wide", page_icon="📈")

# Initialize session state for stock analysis completion
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

# Navigation between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Stock Analysis", "AI Chatbot"],
    index=0 if not st.session_state["analysis_complete"] else 1,  # Default to stock analysis if incomplete
)

# Render the selected page
if page == "Stock Analysis":
    stock_page()
elif page == "AI Chatbot":
    if st.session_state["analysis_complete"] ==True:
        ai_chatbot_page()
    else:
        st.error("You must complete the Stock Analysis before accessing the AI Chatbot.")



