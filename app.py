from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import sqlite3
import os
import numpy as np
import requests
import json
app = Flask(__name__)
app.secret_key = 'test_key'

# 모델 및 벡터화 도구 로드
cur_dir = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'spam_classifier.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'tfidf_vectorizer.pkl'), 'rb'))
db_path = os.path.join(cur_dir, 'spam_ham.sqlite')

# Ollama API 설정
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Ollama API 기본 URL
LLAMA_MODEL = "llama3"  # 사용할 모델 이름 (llama3로 대체 가능)

def query_llama(message):
    """Ollama API 호출 및 스트리밍 응답 처리"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "user", "content": message}]
    }
    try:
        # 스트리밍 요청
        with requests.post(OLLAMA_API_URL, json=payload, headers=headers, stream=True) as response:
            if response.status_code == 200:
                content = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:  # 빈 줄은 무시
                        try:
                            # 각 줄을 JSON으로 파싱
                            data = json.loads(line)
                            # "done" 플래그가 참이면 응답 끝
                            if data.get("done"):
                                break
                            # assistant의 응답 내용 추가
                            content += data["message"]["content"]
                        except json.JSONDecodeError:
                            # JSON 파싱 실패 시 무시
                            continue
                return content
            else:
                return f"Error: {response.status_code} - {response.text}"
    except requests.ConnectionError:
        return "Error: Unable to connect to Llama API. Make sure the server is running."


# LLM 채팅 상위 URL
@app.route('/llmchat')
def llm_chat_home():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('llm_chat.html', chat_history=session['chat_history'])

@app.route('/llmchat/response', methods=['POST'])
def llm_chat_response():
    if request.method == 'POST':
        user_message = request.form['user_message']
        llama_response = query_llama(user_message)

        # 채팅 기록 업데이트
        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append({'sender': 'user', 'message': user_message})
        session['chat_history'].append({'sender': 'llama', 'message': llama_response})
        session.modified = True

        return redirect(url_for('llm_chat_home'))
    
def classify_review(review):
    """리뷰를 벡터화하고 모델로 예측"""
    X = vectorizer.transform([review])
    label = {0: 'Ham', 1: 'Spam'}
    prediction = model.predict(X)[0]
    probability = np.max(model.predict_proba(X))
    return label[prediction], probability

def train_model(review, label):
    """리뷰를 학습 데이터로 추가하여 모델을 점진적으로 학습"""
    X = vectorizer.transform([review])
    model.partial_fit(X, [label])  # 점진적 학습

    # 학습된 모델을 다시 저장
    with open(os.path.join(cur_dir, 'pkl_objects', 'spam_classifier.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)

    # 데이터베이스에 저장
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, label) VALUES (?, ?)", (review, label))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

# 스팸 분류기 상위 URL
@app.route('/spam')
def spam_home():
    return render_template('spam_classifier.html')

@app.route('/spam/result', methods=['POST'])
def spam_result():
    if request.method == 'POST':
        review = request.form['review']
        prediction, probability = classify_review(review)
        return render_template('spam_result.html', review=review, prediction=prediction, probability=round(probability * 100, 2))

@app.route('/spam/feedback', methods=['POST'])
def spam_feedback():
    """사용자 피드백을 수집하여 모델 재학습"""
    review = request.form['review']
    correct_label = request.form['feedback']  # 선택된 정답
    label = 1 if correct_label == 'Spam' else 0

    # 모델 학습
    train_model(review, label)

    return redirect(url_for('spam_home'))  # 스팸 분류 페이지로 리다이렉트

if __name__ == '__main__':
    app.run(debug=True)
