import os
from colorama import Fore, Style, init
from rich.console import Console
from rich.progress import track
from .pdf_reader import extract_text_from_pdf
from .embedding_processor import process_text_embeddings, save_embeddings_to_file, load_embeddings_from_file, split_text_into_token_chunks
from .qa_interface import answer_question
from .cli import ask_question

# Inicializa colorama e rich console para melhorar a experiência visual no terminal
init(autoreset=True)
console = Console()

def main():
    pdf_folder = "data/pdfs"
    all_embeddings = {}
    all_texts = {}

    # Exibe cabeçalho inicial para indicar o começo do processamento de PDFs
    console.print("-" * 30, style="bold yellow")
    console.print("Iniciando Processamento dos PDFs", style="bold cyan")
    console.print("-" * 30, style="bold yellow")

    # Processa cada arquivo PDF na pasta com barra de progresso
    for filename in track(os.listdir(pdf_folder), description="Processando PDFs..."):
        if filename.endswith(".pdf"):
            console.print(f"✔️ [cyan]Processando[/cyan] {filename}...")

            # Extrai o texto do PDF e o divide em blocos menores (chunking) de tokens
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            paragraphs = list(split_text_into_token_chunks(text, max_tokens=2000))
            all_texts[filename] = paragraphs

            # Carrega embeddings salvos ou gera novos se não existirem ou estiverem incompletos
            embeddings = load_embeddings_from_file(filename + ".pkl")
            if embeddings is None or len(embeddings) != len(paragraphs):
                embeddings = [process_text_embeddings(p) for p in paragraphs]
                save_embeddings_to_file(embeddings, filename + ".pkl")
            
            all_embeddings[filename] = embeddings

    # Exibe cabeçalho para a seção de perguntas e respostas
    console.print("\n" + "=" * 30, style="bold yellow")
    console.print("Perguntas e Respostas", style="bold magenta")
    console.print("=" * 30 + "\n", style="bold yellow")

    # Loop para captura de perguntas do usuário até ele optar por sair
    while True:
        question = ask_question()
        if question.lower() in ["sair", "exit", "quit"]:
            console.print("❌ [red]Encerrando o sistema de perguntas.[/red]")
            break
        
        # Separador visual antes de cada pergunta e resposta
        console.print("-" * 50, style="bold yellow")

        # Recebe e exibe a pergunta, busca a resposta e exibe a resposta gerada
        console.print(f"❓ [yellow]Pergunta recebida:[/yellow] {question}")
        answer = answer_question(question, all_embeddings, all_texts)
        console.print(f"✔️ [green]Resposta:[/green] {answer}")
