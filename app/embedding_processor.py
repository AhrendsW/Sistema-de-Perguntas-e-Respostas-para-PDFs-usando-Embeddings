# embedding_processor.py

from dotenv import load_dotenv
import openai
import os
import tiktoken
import pickle

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o tokenizador com o modelo especificado
tokenizer = tiktoken.get_encoding("cl100k_base")

def split_text_into_token_chunks(text, max_tokens=2000):
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])

def process_text_embeddings(text):
    embeddings = []
    for chunk in split_text_into_token_chunks(text, max_tokens=2000):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        chunk_embedding = response['data'][0]['embedding']
        embeddings.append(chunk_embedding)
    return embeddings  # Retorna todos os embeddings dos chunks como uma lista

# Funções de salvar e carregar embeddings usando Pickle
def save_embeddings_to_file(embeddings, filename):
    filepath = os.path.join("data", "embeddings", filename)
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_file(filename):
    filepath = os.path.join("data", "embeddings", filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None
