from flask import Flask, render_template, request, session
import json
import logging
from openai import OpenAI


client = OpenAI(
    base_url='http://llamaserver:8000/v1',
    api_key='not-needed',
)

logging.basicConfig(
    level=logging.INFO,    
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

app = Flask(__name__)
app.secret_key = '59d1dbc9-f780-461f-ac9a-3ab06ccb1821' 


@app.route('/', methods=['GET', 'POST'])
def index():
    # Инициализируем chat_history, если его нет
    if 'chat_history' not in session:
        session['chat_history'] = []
             
    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            logging.info(f'сообщение пользователя - {user_message}')

            # Базовый system prompt (роль: system)
            history = [
                {
                    "role": "system",
                    "content": "You are Chandler from friends."
                }
            ]

            # Пример: добавляем последние три сообщения из истории, если хотите учитывать контекст
            last_messages = session['chat_history'][-3:]
            for msg in last_messages:
                role = "user" if msg['sender'] == "Вы" else "assistant"
                history.append({
                    "role": role,
                    "content": msg['message']
                })

            # Добавляем текущее сообщение пользователя
            history.append({"role": "user", "content": user_message})

            # Посмотрим в логах, что именно уходит на сервер:
            logging.info(f'запрос к модели (messages) - {history}')


            completion = client.chat.completions.create(
                model="local-model",
                messages=history,
                temperature=0.5,
                max_tokens=25
            )
            completion = client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": "Hello"}]
            )
            bot_response = completion.choices[0].message.content    
            
            

            logging.info(f'ответ модели - {bot_response}')

            # Сохраняем реплики в session['chat_history']
            session['chat_history'].append({
                'sender': 'Вы',
                'message': user_message
            })
            session['chat_history'].append({
                'sender': 'Chandler',
                'message': bot_response
            })
            
            session.modified = True
    
    return render_template('index.html', 
                           chat_history=session['chat_history'])


if __name__ == '__main__':
    logging.info('Start service')
    app.run(host="0.0.0.0", port=5000, debug=True)
