import base64
import io
import logging
import mimetypes
import os
import pickle
import requests
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS  # Importar CORS
from flask_restful import Api, Resource
import firebase_admin
from firebase_admin import credentials, db, firestore, storage
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings as OpenAIEmbeddingsAlt
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from Vectordbhandler import VectorDBHandler


# Configura logging
log_stream = io.StringIO()  # Crea un flujo de texto en memoria
logging.basicConfig(stream=log_stream, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # Cargar las variables del archivo .env
app = Flask(__name__)
CORS(app)  # Habilitar CORS en toda la aplicación
api = Api(app)

cred = credentials.Certificate("service_account_gitragbot.json")

firebase_admin.initialize_app(cred,{
'databaseURL' : 'https://github-rag-app.firebaseio.com',
'storageBucket':"gs://github-rag-app.appspot.com"
})
db = firestore.client()


class GitHubData:
    def __init__(self, openai_key, github_token, pineconekey):
        self.openai_key = openai_key
        self.github_token = github_token
        self.repo_data = None
        self.file_content_list = None
        self.file_not_relevant = None
        self.vectorstore = None
        self.repo_name = None
        self.qa_chain = None
        self.pineconekey = pineconekey
                # Lista de extensiones de lenguajes de programación más comunes y relevantes
        self.RELEVANT_EXTENSIONS = {
            '.js', '.jsx', '.py', '.java', '.cpp', '.h', '.hpp', '.cs', '.php', 
            '.ts', '.tsx', '.swift', '.kt', '.kts', '.rb', '.go', '.rs', '.dart', 
            '.scala', '.pl', '.pm', '.r', '.m', '.sh', '.bash', '.sql', '.vba', 
            '.lua', '.hs', '.ex', '.exs', '.fs', '.fsi', '.fsx', '.c', '.erl', 
            '.clj', '.cljs', '.jl', '.groovy', '.f', '.f90', '.for', '.pas', '.cob', 
            '.cbl', '.ada', '.ps1', '.vb', '.rkt', '.d', '.pro', '.scm', '.lisp', 
            '.ml', '.mli', '.zig', '.nim', '.vhd', '.vhdl', '.v', '.sv', '.sas', 
            '.sol', '.abap', '.txt','.md','.html', '.yaml','.json', '.gradle','.ipynb','.css'
        }

    def api_auth_headers(self):
  
        return {"Authorization": f"token {self.github_token}"}

    def get_repo_data(self, username):
        repo_url = f"https://api.github.com/users/{username}/repos"
        self.repo_data = requests.get(repo_url, headers=self.api_auth_headers()).json()

        return self.repo_data

    def get_repo_files(self, repo_name, path=''):

        repo_url_contents = f"https://api.github.com/repos/{repo_name}/contents/{path}"
        
        response = requests.get(repo_url_contents, headers=self.api_auth_headers())
        
        # Verificar si la respuesta es correcta y contiene JSON
        if response.status_code == 200:
            try:
                contents = response.json()  # Asegúrate de que es un JSON válido
            except ValueError: 
                logging.error("Error al obtener la solicitud HTTP. No se pueden obtener los ficheros del repositorio.")
                return [], []  # Devuelve listas vacías si la respuesta no es válida
            
            # Verifica que 'contents' sea una lista
            if not isinstance(contents, list):
                
                return [], []
            
            files = []
            not_relevant_files = []
            
            repo_size = 0
            # El límite de memoria que puedes procesar (512 MB en bytes)
            MAX_MEMORY = ( 512 * 1024 * 1024 ) - 1000  # 512 MB = 536870912 bytes

            for content in contents:

                if repo_size <= MAX_MEMORY:

                    if content.get('type') == 'file':  # Usa get() para evitar KeyError
                        repo_size = repo_size + int(content['size'])

                        if self.is_relevant_file(content['name']):
                            print('File -> ', content)
                            files.append({
                                "name": content['path'],
                                "url": content['download_url'],
                                "size": content['size']
                            })
                        else:
                            print('File -> Not relevant. Format:', content['name'])
                            not_relevant_files.append({
                                "name": content['path'],
                                "url": content['download_url'],
                                "size": content['size']
                            })
                    elif content.get('type') == 'dir':
                        print('Dir type file -> ', content)
                        dir_files, dir_not_relevant = self.get_repo_files(repo_name, content['path'])
                        files.extend(dir_files)
                        not_relevant_files.extend(dir_not_relevant)

                #memoria maxima alcanzada.
                else:
                    print(logging.error(f"memoria máxima alcanzada: {str(MAX_MEMORY)}, no se indexarán todos los archivos"))
                    break 


            
            return files, not_relevant_files
        
        else:
            
            logging.error(f"Error: La solicitud a la API falló con el código {str(response.status_code)}")
            return [], []


    def is_text_file(self, file_info):
        mime_type, _ = mimetypes.guess_type(file_info['name'])
        return mime_type and mime_type.startswith('text')

    def github_read_file(self, repo_name, file_info):
        url = f'https://api.github.com/repos/{repo_name}/contents/{file_info["name"]}'
        r = requests.get(url, headers=self.api_auth_headers())
        r.raise_for_status()
        data = r.json()

        if self.is_text_file(file_info):
            file_content = base64.b64decode(data['content']).decode('utf-8')
            return file_content
        else:
            return f"Binary file: {file_info['name']} (Size: {file_info['size']} bytes)"

    def get_list_of_files(self, repo_name):
        files, not_relevant_files = self.get_repo_files(repo_name)
        file_content_list = []

        for file_info in files:
            try:
                file_content = self.github_read_file(repo_name, file_info)
                file_content_list.append({"name": file_info['name'], "content": file_content, "url":file_info['url']})
            except Exception as e:
                print(f"Error reading file {file_info['name']}: {str(e)}")
        self.file_content_list = file_content_list
        self.file_not_relevant = not_relevant_files

        return file_content_list 

    
    def get_git_userdata(self):

        user_url = "https://api.github.com/user"
        headers = {"Authorization": f"Bearer {self.github_token}", "User-Agent":"Flutter Api Call"}
        
        try:
            response = requests.get(user_url, headers=headers)
            response.raise_for_status()  # Verifica si hubo errores en la solicitud
            
            # Devuelve la respuesta en formato JSON (datos del usuario)
            user_data = response.json()
            return user_data
        
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Manejo de errores HTTP
            return None
        except Exception as err:
            print(f"Other error occurred: {err}")  # Manejo de otros errores
            return None
        
    def get_username_from_email(self, email):
        url = "https://api.github.com/search/users?q={email}"
        response = requests.get(url, headers=self.api_auth_headers())
        return response.json()
    
    def is_relevant_file(self,file_name):
        """
        Verifica si el archivo tiene una extensión relevante para el análisis de código.
        """

        return any(file_name.endswith(ext) for ext in self.RELEVANT_EXTENSIONS)
    

    def index_and_process_files(self, repo_name, vdb_instance) -> bool:

        print(f"Getting repository data for  {repo_name}")
        username = repo_name.split('/')[0]
        self.get_repo_data(username)

        print("Getting list of files")
        self.get_list_of_files(repo_name)

        print("Entering Processing files")
        print("Len of content_list", len(self.file_content_list))
        print("repo name:", repo_name)
        if vdb_instance.go_index_files(self.file_content_list, repo_name) == False:
            return False

        print("RAG Bot is ready. You can start asking questions.")
        return True


    def initialize(self, repo_name) -> bool:

        try:
            vdb = VectorDBHandler(openai_api_key = self.openai_key, pinecone_api_key = self.pineconekey, pinecone_environment = 'us-east-1')
            index_exist = vdb.check_index_existence(repo_name)
            
            #if index not exists in Pinecone, index files, else not.
            if index_exist == True:

                return True 
            
            else:

                ok_indexed = self.index_and_process_files(repo_name=repo_name, vdb_instance= vdb)
                return ok_indexed
        
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            return False

class ReprocessIndexFiles(Resource):

    def post(self):
        data = request.get_json()
        GITHUB_TOKEN = data['github']
        OPENAI_KEY = data['openai']
        PINECONE_KEY = data['pinecone']
        repo_name = data['repo_name']
 
        vdb = VectorDBHandler(openai_api_key = OPENAI_KEY, pinecone_api_key = PINECONE_KEY, pinecone_environment = 'us-east-1')
        github_data = GitHubData(OPENAI_KEY, GITHUB_TOKEN, PINECONE_KEY)

        ok_indexed = github_data.index_and_process_files(repo_name=repo_name, vdb_instance= vdb)
        
        if ok_indexed:
            return {"repo_name":repo_name,"not_relevant": github_data.file_not_relevant, "relevant":github_data.file_content_list, "messages":log_stream.getvalue()}
        else:
            return {"repo_name":repo_name,"not_relevant": github_data.file_not_relevant, "relevant":github_data.file_content_list, "messages":log_stream.getvalue()}



class InitializeRepo(Resource):
    def post(self):
        data = request.get_json()
        GITHUB_TOKEN = data['github']
        OPENAI_KEY = data['openai']
        PINECONE_KEY = data['pinecone']
        repo_name = data['repo_name']
        

        try:
            github_data = GitHubData(OPENAI_KEY, GITHUB_TOKEN, PINECONE_KEY)
            init = github_data.initialize(repo_name)       

            if init == True:
                return {"repo_name":repo_name, "not_relevant": github_data.file_not_relevant, "relevant":github_data.file_content_list, "messages":log_stream.getvalue()}

            else:
           
                return {"repo_name":"", "not_relevant":github_data.file_not_relevant, "messages":log_stream.getvalue()}
            
        except Exception as e:

            logging.error(f'An error occurred during init repo {str(e)}')
            
            return {'repo_name': "", "not_relevant":[], "messages":log_stream.getvalue()}  
        
class AskRepo(Resource):
    def post(self):
        data = request.get_json()
        reponame = data['repo_name']
        question = data['question']
        
        OPENAI_KEY = data['openai']
        PINECONE_KEY = data['pinecone']
 
        try:
            
            handler = VectorDBHandler(OPENAI_KEY, PINECONE_KEY, 'us-east-1')
            responses = handler.similarity_search(question, reponame)
            
            two_best_resp = ''
            other_responses = ''

            for idx, resp in enumerate(responses):
                documento = resp[idx].page_content
                
                # Combinar los dos primeros documentos en `two_best_resp`
                if idx < 2:
                    two_best_resp += '\n' + documento
                    
 
            best_response = handler.get_improved_response(question, two_best_resp)

            return {"answer": str(best_response), "statusCode": 200}
        except Exception as e:
            logging.error(f'An error occurred while processing the question: {str(e)}')
            return {"statusCode": 500, "answer": str(e)}
        
class GetUsername(Resource):
    
    def post(self):

        data = request.get_json()
        GITHUB_TOKEN = data['github']
        email = data['email']
        github_data = GitHubData("", GITHUB_TOKEN)
        username_response = github_data.get_username_from_email(email)
        return username_response
    

class GetReposUser(Resource):

    def post(self):

        data = request.get_json()
        GITHUB_TOKEN = data['github']
        username = data['username']
        github_data = GitHubData("", GITHUB_TOKEN)
        repos = github_data.get_repo_data(username)
        return repos


class GetUserData(Resource):

    def post(self):
        
        data = request.get_json()
        GITHUB_TOKEN = data['github']

        github_data = GitHubData("", GITHUB_TOKEN)
        userdata = github_data.get_git_userdata()
        return userdata
        



api.add_resource(GetUsername, '/get-username/')
api.add_resource(GetReposUser,'/get-repos-user/')
api.add_resource(GetUserData,'/get-user-data-github/')
api.add_resource(InitializeRepo, '/initialize-repo/')
api.add_resource(AskRepo, '/ask-repo/')
api.add_resource(ReprocessIndexFiles, '/reprocess_index_files/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True )



