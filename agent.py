import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
import docxpy
import tempfile
import shutil
from io import StringIO
import torch

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
FILE_DIR: str = os.path.join(ABS_PATH, "file_dir")

if "response" not in st.session_state:
    st.session_state["response"] = ""

st.session_state.analysis = ""
st.session_state.source = ""


def model_res_generator(model, messages):
    if torch.cuda.is_available():
        # Set the global PyTorch device to GPU
        device = torch.device("cuda")
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        # Use CPU if no GPU available
        device = torch.device("cpu")

    response = model.predict(messages)
    return response


def format_data(url, data):
    data_to_export = url + "\n\n" + data
    return data_to_export

def save_temp_copy(uploaded_file):
    # Create a temporary directory
    temp = tempfile.NamedTemporaryFile()
    temp.write(uploaded_file)
    # Copy the uploaded file to the temporary directory
    temp_file_path = shutil.copy(temp.name, FILE_DIR)
    
    return temp_file_path

def process_uploaded_file(file):
    print(f"uploaded file is {file}")
    
    # Decode the file if it's a .txt file
    if file.name.endswith('.txt'):
        text = StringIO(file.getvalue().decode("utf-8"))

    # Process if a PDF file
    if file.name.endswith('.pdf'):
        print(f"pdf file is {file}")
        pdf_path = save_temp_copy(file.getvalue())
        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

    # Process if a Word .docx file
    if file.name.endswith('.docx'):
        docx_path = save_temp_copy(file.getvalue())
        text = docxpy.process(docx_path)
    
    return text


def analyse_data(data_type):
    if data_type == "file":
        st.session_state.analysis = "file"
    elif data_type == "url":
        st.session_state.analysis = "url"
    else:
        st.session_state.analysis = ""
    print(f"analysis: {st.session_state.analysis}")

def main():

    prompt = """

    ================================================================================
    Based on the above pasted text:

    Assume the role of expert prompt engineer, expert healthcare business analyst, 
    expert market researcher, and expert business developer.
    - Summarize text in 5 bullet points. 
    - Give 5 keywords you can find from the pasted text.
    - Give 5 business opportunities from the knowledge you studied from the text.
    """

    ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
    st.session_state.downloaded = False

    data_to_export = ""
    # Set the title and subtitle of the app
    st.title('🧐 Knowledge Discovery Agent')

    pasted_text = ""

    with st.expander("Use File upload"):
        file_placeholder = st.empty()
        uploaded_file = file_placeholder.file_uploader("Choose a file", key="file_uploader_1", type=["pdf", "txt", "docx"])
        if st.button("Analyze File", type="primary", on_click=analyse_data, args=["file"]):
            if uploaded_file is not None:
                st.session_state.source = str(uploaded_file)
                text = process_uploaded_file(uploaded_file)
                pasted_text = text

    with st.expander("Insert URL"):
        url_placeholder = st.empty()
        url_value = url_placeholder.text_input("Insert The website URL", key="url_key_1")
        if st.button("Analyze URL", type="primary", on_click=analyse_data, args=["url"]):
            if url_value != "":
                st.session_state.source = url_value
                # Load data from the specified URL
                print(f"url: {url_value}")
                loader = WebBaseLoader(url_value)
                data = loader.load()

                pasted_text = data[0].page_content


    if pasted_text != "":
        
        # Use a llama3 llm from Ollama
        llm = Ollama(model="llama3")

        full_prompt = f"{pasted_text}:\n {prompt}"
        messages = model_res_generator(llm, full_prompt)
        st.session_state["response"] = messages

        st.write(st.session_state["response"])

        if "response" in st.session_state:
            data_to_export = format_data(st.session_state.source, st.session_state["response"])
        if data_to_export != "":
            st.download_button('Download Results', data_to_export, file_name="results.txt", mime='text/csv')


if __name__ == '__main__':
    main()