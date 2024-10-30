# main.py

import os
from .pdf_reader import extract_text_from_pdf
from .embedding_processor import process_text_embeddings, save_embeddings_to_file, load_embeddings_from_file, split_text_into_token_chunks
from .qa_interface import answer_question
from .cli import ask_question

def main():
    pdf_folder = "data/pdfs"
    all_embeddings = {}
    all_texts = {}

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processando {filename}...")

            # Extrai o texto do PDF e o divide em blocos de tokens
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            paragraphs = list(split_text_into_token_chunks(text, max_tokens=2000))  # Blocos de 2000 tokens
            all_texts[filename] = paragraphs

            # Carrega ou gera embeddings para cada bloco de tokens
            embeddings = load_embeddings_from_file(filename + ".pkl")
            if embeddings is None or len(embeddings) != len(paragraphs):
                embeddings = [process_text_embeddings(p) for p in paragraphs]
                save_embeddings_to_file(embeddings, filename + ".pkl")
            
            all_embeddings[filename] = embeddings

    # Loop para perguntas
    while True:
        question = ask_question()
        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando o sistema de perguntas.")
            break
        
        # Passa a pergunta, todos os embeddings e os textos dos documentos
        answer = answer_question(question, all_embeddings, all_texts)
        print(f"Resposta: {answer}")

