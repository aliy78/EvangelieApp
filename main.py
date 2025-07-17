from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import os

app = Flask(__name__)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 📖 Загрузка Библии
def load_bible():
    with open('biblie.json', encoding='utf-8') as f:
        data = json.load(f)
    verses, references = [], []
    for book in data['Books']:
        name = book['Book']
        for chapter in book['Chapters']:
            num = chapter['Chapter']
            for verse in chapter['Verses']:
                verses.append(verse['Text'])
                ref = f"{name} {num}:{verse['Verse']}"
                references.append(ref)
    return verses, references

verses, references = load_bible()
embeddings = model.encode(verses, convert_to_tensor=True)

# 🏠 Главная страница
@app.route('/')
def home():
    return render_template('index.html')

# 🔍 Поиск стиха по смыслу
@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'Запрос не указан'}), 400
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = int(scores.argmax())
    return jsonify({
        'verse': verses[top_idx],
        'reference': references[top_idx]
    })

# 🚀 Запуск сервера
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
