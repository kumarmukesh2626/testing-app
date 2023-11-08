import time
import requests
import json
from io import BytesIO
import json
import os
import pandas as pd
import openai
# import pdftotext
import math
from langchain import PromptTemplate, LLMChain
import time
import textract
import pandas as pd
from flask import Flask, jsonify , request
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings, CohereEmbeddings, \
    HuggingFaceInstructEmbeddings
from langchain import FAISS
from core.settings import settings


openai.api_key  = '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'

if os.getenv("API_KEY") is not None:
    api_key_set = True
else:
    api_key_set = False
if os.getenv("EMBEDDINGS_KEY") is not None:
    embeddings_key_set = True
else:
    embeddings_key_set = False

if settings.API_KEY is not None:
    api_key_set = True
else:
    api_key_set = False
if settings.EMBEDDINGS_KEY is not None:
    embeddings_key_set = True
else:
    embeddings_key_set = False

if os.getenv("EMBEDDINGS_NAME") is not None:
    embeddings_choice = os.getenv("EMBEDDINGS_NAME")
else:
    embeddings_choice = "openai_text-embedding-ada-002"

def get_completion(text,data, model="gpt-3.5-turbo"):
    message = data[:4090] + "..." if len(data) > 4090 else data
    text = text[:4090] + "..." if len(text) > 4090 else text

    prompt = f"""Please read the text carefully and Give me output in json format for all these questions {message} ,Q1,Q2,Q3,Q4 give me answer in json format like A1,A2,A3,A4.
    ```{text}```
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



def download_pdf(url, save_folder, filename):
    # Create the save_folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        filepath = os.path.join(save_folder, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"PDF file downloaded and saved at: {filepath}")
    else:
        print("Failed to download the PDF file.")

def check_docs(data):
    if data["docs"].split("/")[0] == "local":
        return {"status": 'exists'}
    vectorstore = "vectors/" + data["docs"]
    base_path = 'https://raw.githubusercontent.com/arc53/DocsHUB/main/'
    if os.path.exists(vectorstore) or data["docs"] == "default":
        return {"status": 'exists'}
    else:
        r = requests.get(base_path + vectorstore + "index.faiss")

        if r.status_code != 200:
            return {"status": 'null'}
        else:
            if not os.path.exists(vectorstore):
                os.makedirs(vectorstore)
            with open(vectorstore + "index.faiss", "wb") as f:
                f.write(r.content)

            # download the store
            r = requests.get(base_path + vectorstore + "index.pkl")
            with open(vectorstore + "index.pkl", "wb") as f:
                f.write(r.content)

        return {"status": 'loaded'}






app = Flask(__name__)

@app.route('/v1/src/', methods=['POST'])
def pipeline():
    data = request.json
    if not api_key_set:
        api_key = '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'
    else:
        api_key = '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'
    if not embeddings_key_set:
        embeddings_key = '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'
    else:
        embeddings_key = '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'

    pdf_url = data['url']

    folder_path = 'indexes/local'  # Specify the folder path where you want to save the PDF file
    filename = 'downloaded_pdf.pdf'  # Specify the desired filename

    download_pdf(pdf_url, folder_path, filename)

    # pdf_content = base64.b64decode(pdf_content_base64)

    # Saving PDF in the system
    pdf_path = "indexes/local/{}".format(filename)
    print(pdf_path)

    try:
        # Check if the vectorstore is set
        if "url" in data:
            if pdf_path.split("/")[1] == "local":
                vectorstore = "vectors/local"  # Point to the directory where the vector data is stored
                if pdf_path.split("/")[1] == "default":
                    vectorstore = ""
                else:
                    vectorstore += "/" + filename
            else:
                vectorstore = "vectors/local/" + filename

            if data['url'] == "default":
                vectorstore = ""
        else:
            vectorstore = ""
        print("Vectorstore", vectorstore)

        # Call the check_docs function to ensure vectorstore is available
        docs_status = check_docs(data)
        print("Docs Status", docs_status)

        if embeddings_choice == "openai_text-embedding-ada-002":
            # gpt-3.5-turbo
            # if embeddings_choice == "gpt-3.5-turbo":
            docsearch = FAISS.load_local(vectorstore, OpenAIEmbeddings(openai_api_key=embeddings_key))
        elif embeddings_choice == "huggingface_sentence-transformers/all-mpnet-base-v2":
            docsearch = FAISS.load_local(vectorstore, HuggingFaceHubEmbeddings())
        elif embeddings_choice == "huggingface_hkunlp/instructor-large":
            docsearch = FAISS.load_local(vectorstore, HuggingFaceInstructEmbeddings())
        elif embeddings_choice == "cohere_medium":
            docsearch = FAISS.load_local(vectorstore, CohereEmbeddings(cohere_api_key=embeddings_key))
        # Parsing PDF and fetching relevant information
        print(docsearch)
        text = textract.process(pdf_path, method='tesseract', encoding='utf-8')
        text = text.decode('utf-8')
        # print(text)

        response = get_completion(text, data)

        content_details = response.split('\n')
        print(response)

        return response
    except Exception as e:
        print("Error", e)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5008, debug=True)
