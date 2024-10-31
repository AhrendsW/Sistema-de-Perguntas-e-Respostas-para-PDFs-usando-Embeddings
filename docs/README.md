# Sistema de Perguntas e Respostas para PDFs usando Embeddings

Este projeto é um sistema de perguntas e respostas desenvolvido em Python que permite a extração de conteúdo de arquivos PDF, processamento de embeddings para consultas eficientes e geração de respostas usando IA. Ele é adequado para consultas contextuais com PDFs, como artigos acadêmicos, relatórios e documentos textuais, e possui uma interface de linha de comando (CLI).

## Funcionalidades

- **Extração de texto de PDFs**: Suporte para PDFs com conteúdo digital. PDFs que contenham apenas imagens (como escaneamentos) precisam de OCR adicional, o qual pode ser integrado com `pytesseract` e Tesseract OCR.
- **Processamento de embeddings**: Utiliza o modelo `text-embedding-ada-002` da OpenAI para transformar o conteúdo do PDF em embeddings, possibilitando consultas eficazes.
- **Consultas baseadas em similaridade de cosseno**: Responde perguntas usando embeddings, comparando a similaridade entre a pergunta e o conteúdo do PDF.
- **Interface de usuário CLI**: Permite que o usuário faça perguntas diretamente pela linha de comando.

## Requisitos

Certifique-se de que o Python 3.11.9 esteja instalado.

Crie e ative o seu ambiente virtual através do seguinte comando:
 - No Windows:
 ```bash
 python -m venv .venv
 .venv\Scripts\activate
```
- No macOS ou Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Dependências

Instale as bibliotecas necessárias executando:

```bash
pip install -r requirements.txt
```

## Arquivo .env

Para processar embeddings com o modelo da OpenAI, crie um arquivo `.env` com sua chave de API:

```plaintext
OPENAI_API_KEY=your_openai_api_key
```

## Estrutura do Projeto

<pre>
├── app/
│   ├── __init__.py              # Inicializa o pacote principal
│   ├── main.py                  # Arquivo principal para execução do sistema
│   ├── cli.py                   # Interface de perguntas e respostas CLI
│   ├── pdf_reader.py            # Módulo para extração de texto de PDFs
│   ├── embedding_processor.py   # Processamento e salvamento de embeddings
│   └── qa_interface.py          # Interface para gerar respostas com base na similaridade de embeddings
├── data/
│   ├── pdfs/                    # Diretório para armazenar arquivos PDF
│   └── embeddings/              # Diretório para armazenar embeddings processados
├── run.py                       # Script de execução do sistema
├── requirements.txt             # Lista de dependências do projeto
├── docs/
│   ├── README.md                # Documentação do projeto
</pre>

## Uso

Para iniciar o sistema, execute o seguinte comando:

```bash
python run.py
```

O sistema irá processar todos os PDFs armazenados no diretório data/pdfs, gerar embeddings para cada um e, em seguida, permitir que você faça perguntas sobre o conteúdo desses documentos. Digite sair para encerrar o sistema.

Exemplo de Pergunta
Coloque o arquivo PDF desejado em data/pdfs.

Execute o comando python run.py.

Digite sua pergunta:

```plaintext
Digite sua pergunta: Qual é o tema principal do documento sobre tecnologias de informação em Angola?
```

## Observações
Este projeto foi desenvolvido para lidar apenas com PDFs baseados em texto. PDFs que contenham apenas imagens (como escaneamentos) precisam de OCR adicional, o qual pode ser integrado com pytesseract e Tesseract OCR.
O modelo text-embedding-ada-002 é usado para transformar cada trecho do PDF em um embedding. Este projeto foi testado com um threshold de similaridade ajustável para respostas mais precisas.
## Melhorias Futuras
- Suporte para OCR: Implementar OCR automático para PDFs baseados em imagem.
- Interface Web: Criar uma interface web para acesso mais fácil e dinâmico.
- Suporte a múltiplos idiomas: Expansão para permitir perguntas e respostas em outros idiomas além do português.
