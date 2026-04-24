import streamlit as st
import requests
import json

# Page config
st.set_page_config(page_title="CodeDocsAI", layout="wide")
st.title("🚀 CodeDocsAI - Intelligent Documentation Assistant")

# Sidebar configuration
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
use_cache = st.sidebar.checkbox("Use Cache", value=True)

# Main interface
st.write("Ask questions about code documentation and get instant, context-aware answers.")

# Query input
question = st.text_area(
    "Enter your question:",
    placeholder="e.g., How do I create a REST API in FastAPI?",
    height=100
)

if st.button("🔍 Search"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        try:
            # Call backend
            response = requests.post(
                f"{api_url}/query",
                json={"question": question, "use_cache": use_cache}
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data["response"]
                
                # Display results
                st.success("✅ Results Found")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("📖 Answer")
                    st.write(result.get("answer", "No answer"))
                    
                    if result.get("code_example"):
                        st.subheader("💻 Code Example")
                        st.code(result["code_example"], language="python")
                    
                    if result.get("key_points"):
                        st.subheader("📌 Key Points")
                        for point in result["key_points"]:
                            st.write(f"• {point}")
                
                with col2:
                    st.metric("Latency", f"{data['latency_ms']:.0f}ms")
                    st.metric("Sources", len(data['sources']))
                    
                    with st.expander("📚 Sources"):
                        for source in data['sources']:
                            st.write(f"• {source}")
            else:
                st.error(f"Error: {response.text}")
        
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")

# Footer with stats
st.divider()
if st.button("📊 View Stats"):
    try:
        stats = requests.get(f"{api_url}/stats").json()
        st.json(stats)
    except:
        st.error("Could not fetch stats")
