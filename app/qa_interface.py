import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_processor import process_text_embeddings
import openai

# Gera uma resposta dinâmica usando o modelo GPT-3.5-turbo com base no contexto relevante do PDF
def generate_dynamic_response(question, context):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é uma IA que responde perguntas com base exclusivamente no conteúdo fornecido do PDF. Não inclua informações externas ao conteúdo fornecido."},
                {"role": "user", "content": f"{question}\n\nContexto:\n{context}"}
            ],
            max_tokens=500,  # Limita a resposta a 300 tokens
            temperature=0.4   # Define o nível de criatividade como baixo
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao gerar resposta dinâmica: {str(e)}"

# Função principal de resposta a perguntas
def answer_question(question, all_embeddings, all_texts, similarity_threshold=0.8):
    if len(question.split()) < 3:
        return "Por favor, faça uma pergunta mais específica."

    # Calcula o embedding da pergunta para comparar com os documentos
    question_embedding = np.array(process_text_embeddings(question)[0])  # Garante que o vetor tenha 1536 dimensões

    best_responses = []  # Lista para armazenar as melhores respostas de cada documento

    for doc_name, doc_embeddings in all_embeddings.items():
        best_similarity_doc = -1
        best_text_doc = "Desculpe, não encontrei uma resposta relevante para este documento."

        for i, emb in enumerate(doc_embeddings):
            # Converte o embedding em uma matriz para cálculo de similaridade
            emb = np.array(emb[0]).reshape(1, -1) if isinstance(emb, list) else np.array(emb).reshape(1, -1)
            similarity = cosine_similarity([question_embedding], emb)[0][0]

            # Se a similaridade é maior que a anterior e atende ao limiar, armazena como a melhor até agora
            if similarity > best_similarity_doc and similarity >= similarity_threshold:
                best_similarity_doc = similarity
                best_text_doc = all_texts[doc_name][i]

        best_responses.append((best_similarity_doc, best_text_doc, doc_name))

    # Ordena para encontrar a melhor resposta de todos os documentos
    best_responses.sort(key=lambda x: x[0], reverse=True)
    if best_responses[0][0] >= similarity_threshold:
        best_text, best_doc_name = best_responses[0][1], best_responses[0][2]
        return generate_dynamic_response(question, best_text)

    return "Desculpe, não encontrei uma resposta relevante."
