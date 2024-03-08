import openai
import langchain
from flask import render_template,request,Flask,redirect
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI
app = Flask(__name__)

# # Function to read documents from directory
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Chunk data function
def chunk_data(docs, chunk_size=500, chunk_overlap=5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=' your api key')

# Initialize OpenAI client
client = OpenAI(api_key='ypur api key')

# Load documents
doc = read_doc('documents')
documents = chunk_data(docs=doc)
                
# Create FAISS index from documents
db = FAISS.from_documents(documents, embeddings)
# db.save_local("faiss_index")
                
                # Perform similarity search
                
# db.load_local("C:\\Users\\Vinayk\\flask\\new_proj\\faiss_index",embeddings,allow_dangerous_deserialization=True)
db = FAISS.load_local("C:\\Users\\Vinayk\\flask\\new_proj\\faiss_index",embeddings,allow_dangerous_deserialization=True)
# Main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Perform similarity search
        docs = db.similarity_search(user_input)
        retriever = db.as_retriever()
        docs = retriever.invoke(user_input)

        # Get document text for OpenAI completion
        doc_text = documents[0].page_content 
        
        # Get completion from OpenAI
        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "system", "content": doc_text},
                                                            {"role": "user", "content": user_input}])

        # Get reply from completion
        reply = completion.choices[0].message.content

        return render_template('index.html', reply=reply, user_input=user_input)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
