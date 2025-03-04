from flask import Flask, render_template, request, session, redirect, url_for
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import logging


logging.basicConfig(level=logging.INFO,    
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
                    )


app = Flask(__name__)
app.secret_key = '59d1dbc9-f780-461f-ac9a-3ab06ccb1822' 


collection_name="qa_collection"
batch_size = 200

# Инициализация моделей и клиента
def initialize_services():
    client = QdrantClient("http://qdrant:6333")  
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')
    return client, embedding_model, cross_encoder

# Создание коллекции и загрузка данных
def load_data_to_qdrant(client, model, json_file):
    # Проверка существования коллекции
    if client.collection_exists(collection_name):
        logging.info(f"Коллекция  {collection_name} уже существует")    
        return
    logging.info(f"Загрузка коллекции  {collection_name}")
    # Чтение JSON файла
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    
    # Создание коллекции
    client.recreate_collection(
        collection_name="qa_collection",
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )
    # Разбиваем данные на пакеты
    total_points = len(data)
    for i in range(0, total_points, batch_size):
        batch = data[i:i + batch_size]
        
        points = [
            models.PointStruct(
                id=idx + i,
                vector=model.encode(item['q']).tolist(),
                payload={
                    "character": item['character'],
                    "answer": item['a'],
                    "question": item['q']
                }
            )
            for idx, item in enumerate(batch)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points
        )
        logging.info(f"Загружено {min(i + batch_size, total_points)}/{total_points} записей")    


# Поиск и реранкинг
def search_with_reranking(client, model, cross_encoder, question, character):
    # Векторизация вопроса
    query_vector = model.encode(question).tolist()
    
    # Поиск с фильтром
    search_results = client.search(
        collection_name="qa_collection",
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="character",
                match=models.MatchValue(value=character)
            )]
        ),
        limit=10
    )
    
    # Реранкинг с Cross-Encoder
    pairs = [(question, hit.payload['question']) for hit in search_results]
    scores = cross_encoder.predict(pairs)
    
    # Комбинирование и сортировка результатов
    reranked_results = sorted(
        zip(scores, search_results),
        key=lambda x: x[0],
        reverse=True
    )
    
    return reranked_results


qdrant_client, emb_model, ce_model = initialize_services()

# Загрузка данных 
load_data_to_qdrant(qdrant_client, emb_model, './data/dialogs.json')


#Получение уникальных персонажей из коллекции
def get_unique_characters(client):
    try:
        # Получаем все записи из коллекции
        records, _ = client.scroll(
            collection_name="qa_collection",
            limit=20000,
            with_payload=True,
            with_vectors=False
        )
        
        # Извлекаем уникальных персонажей
        characters = {record.payload.get('character') for record in records}
        return sorted([c for c in characters if c])  # Фильтрация None и сортировка
    
    except Exception as e:
        print(f"Ошибка при получении персонажей: {e}")
        return []
    
    
characters = get_unique_characters(qdrant_client) 


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_character = request.form.get('character')
        if selected_character in characters:
            session['character'] = selected_character
            session['chat_history'] = [{
                'sender': selected_character,
                'message': f"Hi! I'm {selected_character}. Welcome!"
            }]
            return redirect(url_for('chat'))
    return render_template('index.html', characters=characters)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'character' not in session:
        return redirect(url_for('index'))
    
    current_character = session['character']
    
    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            # Сообщение пользователя
            session['chat_history'].append({
                'sender': 'Вы',
                'message': user_message
            })
            
            logging.info(f'сообщение пользователя - {user_message}')

            # Поиск
            results = search_with_reranking(
                qdrant_client,
                emb_model,
                ce_model,
                question=user_message,
                character= current_character
            )
            print(f'результат поиска - {results[0]}')
            _, best_hit = results[0]
            # Ответ бота            
            bot_response = best_hit.payload['answer']
            
            session['chat_history'].append({
                'sender': current_character,
                'message': bot_response
            })
            
            session.modified = True
    
    return render_template('chat.html', 
                         character=current_character,
                         chat_history=session['chat_history'])

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':    
    logging.info('Start service')
    app.run(host="0.0.0.0", port=5000,debug=True)