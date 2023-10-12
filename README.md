# Talk with your docs

A simple Python Flask server application that allows yoru to talk with your documents using an LLM.

I used ChromaDB as the vector store database and a document about Insurance as a toy example.

The Template file has all the HTML and CSS scripts for the web page.

Code can be edited to include an LLM and also other features. For my use case I have used a vanilla implementation of Mistral 7B model. I used a HuggingFace Token and you can supply your own token if you are going to use a model.

To start server: 

1) Run infer.py
2) Go to the webpage (in this case it is localhost:5050
3) Run your query and retrieve in contect answer form the LLM
