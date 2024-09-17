import requests
import base64
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from flask import Flask, request
from flask_restful import Api, Resource
from dotenv import load_dotenv

load_dotenv()  # Cargar las variables del archivo .env
app = Flask(__name__)
api = Api(app)

class GitHubData:
    def __init__(self, username, repo_name, openai_key, github_token):
        self.username = username
        self.repo_name = repo_name
        self.openai_key = openai_key
        self.github_token = github_token
        self.repo_data = None
        self.file_content_list = None
        self.vectorstore = None

    def api_auth_headers(self):
       
        headers = {"Authorization": f"token {self.github_token}"}
        return headers

    def get_repo_data(self):
        repo_url = f"https://api.github.com/users/{self.username}/repos"
        self.repo_data = requests.get(repo_url, headers=self.api_auth_headers()).json()
        return self.repo_data

    def get_repo_files(self):
        repos = []
        repo_url_contents = f"https://api.github.com/repos/{self.username}/{self.repo_name}/contents/"
        contents = requests.get(repo_url_contents, headers=self.api_auth_headers()).json()

        repo_files = []
        for content in contents:
            if content["download_url"] is not None:
                file_url = content["download_url"]
                file_name = content["name"]

                repo_files.append({
                    "name": file_name,
                    "url": file_url
                })

        repos.append({"repo_name": self.repo_name, "files": repo_files})
        return repos

    def github_read_file(self, repository_name, file_path):
        headers = self.api_auth_headers()
        url = f'https://api.github.com/repos/{self.username}/{repository_name}/contents/{file_path}'
        print(url)
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
        file_content = data['content']
        file_content_encoding = data.get('encoding')
        if file_content_encoding == 'base64':
            file_content = base64.b64decode(file_content).decode()
        else:
            print("Content is not base64")
        return file_content

    def get_list_of_files(self, repo):
        file_content_list = []
        for file_info in repo[0]['files']:
            file_content = self.github_read_file(self.repo_name, file_info['name'])
            file_content_list.append({"name": file_info['name'], "contenido": file_content})
        self.file_content_list = file_content_list
        return file_content_list

    def update_repo_with_content(self, repo):
        for file_info in repo[0]['files']:
            file_content = self.github_read_file(repo[0]['repo_name'], file_info['name'])
            file_info['contenido'] = file_content
        return repo

    def create_document(self, file):
        return Document(page_content=file['contenido'], metadata={"source": file['name']})

    def split_text_into_chunks(self, doc):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents([doc])

    def add_documents_to_vector_store(self, chunks):
        embeddings = OpenAIEmbeddings()
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            self.vectorstore.add_documents(chunks)

    def ask_question(self, question):
        llm = OpenAI(api_key=self.openai_key)
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=self.vectorstore.as_retriever())
        return qa_chain.run(question)

    def process_files(self):
        if self.file_content_list is None:
            print("No file content list found. Please run get_list_of_files first.")
            return

        # Initialize vectorstore
        self.vectorstore = None

        for file_data in self.file_content_list:
            print("Procesando archivo...")
            document = self.create_document(file_data)
            text_chunks = self.split_text_into_chunks(document)
            self.add_documents_to_vector_store(text_chunks)
    
    def main(self, question):

        # Get repository data and file contents
        self.get_repo_data()
        repo_files = self.get_repo_files()
        file_content_list = self.get_list_of_files(repo_files)

        # Update repo with file content
        self.update_repo_with_content(repo_files)

        # Process files and build vectorstore

        self.process_files()

        answer = self.ask_question(question)

        return answer



class GetRepoData(Resource):

    def post(self):
 
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        OPENAI_KEY = os.getenv('OPENAI_KEY')
       
        data = requests.get_json()

        username = data['username']
        repo_name = data['repo_name']
        question = data['question']

        github_data = GitHubData(username, repo_name, OPENAI_KEY, GITHUB_TOKEN)
        answer = github_data.main(question)

        return answer



api.add_resource(GetRepoData, '/ask-repo')

if __name__ == '__main__':
    app.run(debug=True)



