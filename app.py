# app.py

import streamlit as st
import agent as ag

st.set_page_config(page_title="Vision Agent", layout="centered")

st.title("🔍 Vision Agent - Object Detection")

st.markdown("""
This app uses AI to analyze images and detect objects using YOLOv8.

**Example queries:**
- "What objects are in the images folder?"
- "Analyze all images"
- "What's in image1.jpg?"
""")

# Input box
query = st.text_area("Enter your question:", height=100, placeholder="e.g., What objects are in the images?")

# Button
if st.button("🚀 Analyze", type="primary"):
    if query.strip():
        with st.spinner("🤔 Analyzing images..."):
            try:
                result = ag.run_query(query)
                
                st.success("✅ Analysis Complete")
                st.markdown("### Result")
                st.write(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 and LangChain*")
