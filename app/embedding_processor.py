from dotenv import load_dotenv
import openai
import os
import tiktoken
import pickle

load_dotenv()  # Carrega variáveis de ambiente, incluindo a chave da API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o tokenizador para o modelo cl100k_base
tokenizer = tiktoken.get_encoding("cl100k_base")

# Divide o texto em pedaços com base no número de tokens, para evitar limites do modelo
def split_text_into_token_chunks(text, max_tokens=4000):  # Ajustado para 3000 tokens
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])

# Processa o texto e gera embeddings, dividindo-o em pedaços conforme a necessidade
def process_text_embeddings(text):
    embeddings = []
    for chunk in split_text_into_token_chunks(text, max_tokens=4000):  # Ajustado para 3000 tokens
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        chunk_embedding = response['data'][0]['embedding']  # Cada embedding tem 1536 dimensões
        embeddings.append(chunk_embedding)
    return embeddings

# Função para salvar embeddings em um arquivo usando Pickle
def save_embeddings_to_file(embeddings, filename):
    filepath = os.path.join("data", "embeddings", filename)
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)

# Função para carregar embeddings salvos, se o arquivo existir
def load_embeddings_from_file(filename):
    filepath = os.path.join("data", "embeddings", filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None
