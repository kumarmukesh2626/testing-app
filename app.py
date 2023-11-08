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
import time
import textract
import pandas as pd
from flask import Flask, jsonify , request



openai.api_key  = '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'


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

app = Flask(__name__)

@app.route('/v1/src/',methods=['POST'])
def pipeline():
    data = request.json

    pdf_url = data['url']

    folder_path = 'indexes/local'  # Specify the folder path where you want to save the PDF file
    filename = 'downloaded_pdf.pdf'  # Specify the desired filename

    download_pdf(pdf_url, folder_path, filename)

    # pdf_content = base64.b64decode(pdf_content_base64)

    # Saving PDF in the system
    pdf_path = "indexes/local/{}".format(filename)
    print(pdf_path)

    try:
        # check if the vectorstore is set
        if "url" in data:
            if pdf_path.split("/")[1] == "local":
                if pdf_path.split("/")[1] == "default":
                    vectorstore = ""
                else:
                    vectorstore = "indexes/local/" + filename
            else:
                vectorstore = "vectors/local/" + filename
            if data['url'] == "default":
                vectorstore = ""
        else:
            vectorstore = ""
        print("Vectorstore",vectorstore)

        text = textract.process(pdf_path, method='tesseract', encoding='utf-8')
        text = text.decode('utf-8')
        # print(text)
  
        response = get_completion(text,data)
        
        content_details = response.split('\n')
        print(response)
        
        return jsonify({"result": response})  # Return the response as JSON

    except Exception as e:
        print("Error", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5004, debug=True)
