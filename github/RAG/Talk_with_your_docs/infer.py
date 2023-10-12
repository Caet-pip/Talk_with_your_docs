from flask import Flask, request, render_template, jsonify
# from unstructured.partition.pdf import partition_pdf
import re
from chromadb.utils import embedding_functions
import openai
import chromadb 
from chromadb import Client
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from haystack.nodes import PromptNode
from getpass import getpass

HF_TOKEN = "HF_TOKEN"


def generate_response(question: str,context: str):
    pn = PromptNode(model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",  # instruct fine-tuned model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
                max_length=800,
                api_key=HF_TOKEN)
    
    out=pn(f"""You are a information provider answer this {question} based on this information {context} """)

    return out[0]

from pdfminer.high_level import extract_text
def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text

a = extract_text_from_pdf('American_Cancer_Society_Insurance.pdf')

def string_to_dataframe(input_string):

    sentences = input_string.split('.')
    sentences = [sentence.strip().replace('\n', '') for sentence in sentences]
    sentences = [sentence.strip().replace('\'', '') for sentence in sentences]

    df = pd.DataFrame({'text': sentences})
    
    return df
f = string_to_dataframe(a)
df = f

numeric_pattern = r'^\d+$'

df = df[~df['text'].str.match(numeric_pattern)]

df['word_count'] = df['text'].str.split().str.len()
df = df.sort_values(by='word_count', ascending=False)

df['text'] = df['text'].astype(str)
dfa = df.reset_index(drop=True)
dfb = dfa[dfa['word_count']> 0] 
dfb

documents = [ ]
metadatas = [ ]
for t, m in zip(dfb['text'], dfb['word_count']):
    documents.append(t)
    metadatas.append({'word_count':m,'file_name':1})


client = chromadb.Client()
collection = client.create_collection("insurance4")

collection.add(
    ids=[str(i) for i in range(0, len(documents))],  
    documents=documents,
    metadatas=metadatas,
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/query", methods=['POST'])
def query():
    question = request.form.get('query')

    if not question:
        return render_template('index.html', error="Missing query")
    
    try:
        results = collection.query(
            query_texts=[f"{question}"],
            n_results=3
        )

        if not results or not results['documents']:
            return jsonify({"error": "No documents found"}), 404
        c1 = results['documents'][0][0] + results['documents'][0][1] + results['documents'][0][2]
        context = c1
        input_text = question
        outputs = generate_response(input_text, context)
        answer = outputs
        return render_template('index.html', result=answer, context=context)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
