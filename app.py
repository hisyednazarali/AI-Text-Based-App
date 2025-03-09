import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load model and tokenizer (forcing redownload if needed)
@st.cache_resource()
def load_model():
    model_name = "gpt2"
    
    # Ensure model and tokenizer are downloaded
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# Streamlit UI
st.title("🚀 AI Based Text Generation App")
st.write("Enter a prompt, and let AI generate text for you!")

# User input
user_input = st.text_area("Enter a prompt:", "Once upon a time...")

# Text Generation Button
if st.button("Generate Text"):
    if user_input.strip():
        with st.spinner("Generating..."):
            output = generator(user_input, max_length=500, num_return_sequences=1)
            st.subheader("Generated Text:")
            st.write(output[0]['generated_text'])
    else:
        st.warning("Please enter a valid prompt.")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Transformers")
