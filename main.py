
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Загружаем Библию из JSON-файла
with open('biblie.json', encoding='utf-8') as f:
    data = json.load(f)

# Собираем список стихов и сохраняем ссылки на них
verses = []
references = []

for book in data['Books']:
    book_name = book['Book']
    for chapter in book['Chapters']:
        chapter_num = chapter['Chapter']
        for verse in chapter['Verses']:
            verses.append(verse['Text'])
            ref = f"{book_name} {chapter_num}:{verse['Verse']}"
            references.append(ref)

# Строим эмбеддинги
embeddings = model.encode(verses, convert_to_tensor=True)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'Нет запроса'}), 400

    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = int(scores.argmax())

    return jsonify({
        'verse': verses[top_idx],
        'reference': references[top_idx],
        'score': float(scores[top_idx])
    })

if __name__ == '__main__':
    app.run(debug=True)
