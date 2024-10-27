 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import logging
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
import re
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
import faiss
import time
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import logging
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec

class VectorDBHandler:
    def __init__(self, openai_api_key, pinecone_api_key, pinecone_environment):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_instance = None
  

    def initialize_pinecone(self):
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        pc = Pinecone(api_key=self.pinecone_api_key)
        self.pinecone_instance = pc
        print("Pinecone instance ok")
    
    def go_index_files(self, file_content_list , index_name):
        print("Init pinecone")
        self.initialize_pinecone()
        print("Finish init pinecone")
        is_success = self.process_files(file_content_list, index_name)
        print("¿Is success?", is_success)
        return is_success
        
    
    def convert_reponame_toindex_format(self, string):
 
        # Reemplazar barra inclinada por guion
        string = string.replace('/', '-')
        
        # Eliminar cualquier carácter que no sea letra o guion
        string = re.sub(r'[^a-zA-Z-]', '', string)
        
        # Convertir a minúsculas
        string = string.lower()
        
        return string
    
    def add_answer_tovector_pinecone(self, answer, question, index_name):

        index_name = self.convert_reponame_toindex_format(index_name)
        document_1 = Document(page_content=answer, metadata={"question":question})
        vector_store = self.add_documents_to_store(document_1, OpenAIEmbeddings(), index_name)
        print("Reindexed the vectorstore with the answer!")
        


    def process_files(self, file_content_list, index_name):
        #todo, mecesito hacer textsplits
        try:
            index_name = self.convert_reponame_toindex_format(index_name)
            self.create_index(index_name)
        
            documents = []
            ids = []
            id = 0
            for file_data in file_content_list:
                try:
                    content = file_data['content']
                    doc = Document(page_content=content, metadata={"source": file_data['name']})
                    documents.append(doc)

                    id = id + 1
                    idstr = str(id)
                    ids.append(idstr)
                    logging.info(f"Successfully processed: {file_data['name']}")
                except Exception as e:
                    logging.error(f"Error processing file {file_data['name']}: {str(e)}")
                    continue

            if not documents:
                logging.error("No documents to process. Check if files were read correctly.")
                return False

            
    
            # Split the extensive documents into smaller chunks
            docs = self.split_documents(documents)

            # Add the split documents to the Pinecone store
            vector_store = self.add_documents_to_store(docs, OpenAIEmbeddings(), index_name)

            print("added docs to vectorstore")
            return True

             
        except Exception as e:
            logging.error(f"Error during document processing: {str(e)}")
            print(f"Error during document processing: {str(e)}")
            return False
    
    def check_index_existence(self, index_name) -> bool:
        self.initialize_pinecone()
        index_name = self.convert_reponame_toindex_format(index_name)
        print("Checking existence of index:", index_name)
        existing_indexes = [index_info["name"] for index_info in self.pinecone_instance.list_indexes()]

        if index_name in existing_indexes:
            print("Index exists")
            return True
        else:
            print("Index not exists")
            return False

    
    def create_index(self, index_name):

        #

        #existing_indexes = [index_info["name"] for index_info in self.pinecone_instance.list_indexes()]

        #if index_name not in existing_indexes:
        print("Creating index..")
        self.pinecone_instance.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self.pinecone_instance.describe_index(index_name).status["ready"]:
            time.sleep(1)
        
        print("index created!")

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)
   
    # Add documents to Pinecone vector store
    def add_documents_to_store(self,docs, embeddings, index_name):
        return PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

    def get_improved_response(self, question, provided_answer):
    
        if self.openai_api_key is None:
            return "No API key provided"
        
        client = OpenAI(api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an advanced AI assistant specialized in software development.
                        Your role is to provide accurate, concise, and practical coding assistance,
                        helping users solve technical issues in code, improve performance, and follow best practices.
                        
                        Detect the language (e.g., English, Spanish, etc.) in the user's question and respond in the same language.
                        Do not answer in multiple languages, just the language you detected previosuly.
                        
                        When applicable, give clear explanations but always focus on delivering a solution in the detected language.
                    """
                },
                {
                    "role": "user",
                    "content": f"Based on the user's question: '{question}'\n"
                            f"Rewrite the answer to offer the user a more technical approach: '{provided_answer}'"
                }
            ]
        )

        return response.choices[0].message.content


    def similarity_search(self, query, indexname):

        if self.openai_api_key == None or self.pinecone_api_key == None:
            return "not api key provided"
        else: 
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            os.environ["PINECONE_API_KEY"] = self.pinecone_api_key

        index_name = self.convert_reponame_toindex_format(indexname)
        embeddings = OpenAIEmbeddings()

        vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        answer = vectorstore.similarity_search(query, k = 5 )
        #devuelve una lista de documentos
        return answer
        #return str(answer[0].page_content
