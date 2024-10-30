import fitz  # Biblioteca PyMuPDF para manipulação de PDFs

# Extrai texto de um arquivo PDF, página por página
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()  # Concatena o texto de cada página
    return text
