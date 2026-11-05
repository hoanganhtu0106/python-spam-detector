from flask import Flask, request, render_template, redirect, url_for, Response
from naive_bayes import NaiveBayesClassifier
from data_preprocessing import preprocess, load_and_prepare_data
from datetime import datetime
import os
import sqlite3
import csv
import io
import math

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'spam_classifier.db')
MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.pkl')


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            predicted_result TEXT NOT NULL,
            actual_label TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def get_history():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT message, result, timestamp FROM history ORDER BY id DESC'
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_to_history(message, result):
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO history (message, result, timestamp) VALUES (?, ?, ?)',
        (message, result, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()


def get_feedback():
    conn = get_db_connection()
    rows = conn.execute(
        'SELECT message, predicted_result, actual_label, timestamp FROM feedback ORDER BY id DESC'
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_feedback(message, predicted_result, actual_label):
    if not message or not predicted_result:
        return
    if actual_label == 'correct':
        actual_label = predicted_result
    conn = get_db_connection()
    conn.execute(
        'INSERT INTO feedback (message, predicted_result, actual_label, timestamp) VALUES (?, ?, ?, ?)',
        (message, predicted_result, actual_label, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()


def compute_confidence(model, words, predicted_label):
    if not words:
        return 0.5

    scores = {}
    for label in [0, 1]:
        score = math.log(model.class_probs.get(label, 1e-9))
        for word in words:
            if word in model.vocab:
                score += math.log(model.word_probs.get(label, {}).get(word, 1e-9))
        scores[label] = score

    max_score = max(scores.values())
    exp_scores = {label: math.exp(value - max_score) for label, value in scores.items()}
    total = sum(exp_scores.values())
    if total == 0:
        return 0.5
    return exp_scores.get(predicted_label, 0) / total


# Load the trained model
if os.path.exists(MODEL_PATH):
    model = NaiveBayesClassifier.load_model(MODEL_PATH)
else:
    train_data, _ = load_and_prepare_data(split_ratio=1.0)
    model = NaiveBayesClassifier()
    model.train(train_data)
    model.save_model(MODEL_PATH)

init_db()


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    message_display = None
    error = None
    confidence = None

    if request.method == 'POST':
        try:
            # Handle text input
            if 'message' in request.form and request.form['message'].strip():
                message = request.form['message'].strip()
                tokens = preprocess(message)
                prediction = model.predict(tokens)
                result = 'Spam' if prediction == 1 else 'Ham'
                confidence = compute_confidence(model, tokens, prediction)
                message_display = message
                add_to_history(message, result)

            # Handle file upload
            elif 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    if file.filename.endswith('.txt'):
                        content = file.read().decode('utf-8')
                        tokens = preprocess(content)
                        prediction = model.predict(tokens)
                        result = 'Spam' if prediction == 1 else 'Ham'
                        confidence = compute_confidence(model, tokens, prediction)
                        message_display = content
                        add_to_history(content, result)

                    elif file.filename.endswith('.csv'):
                        import csv as csv_module
                        content = file.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        results = []

                        for line in lines[1:]:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                tokens = preprocess(line)
                                prediction = model.predict(tokens)
                                result_type = 'Spam' if prediction == 1 else 'Ham'
                                results.append((line[:50], result_type))
                                add_to_history(line, result_type)
                            except Exception:
                                pass

                        result = f"Classified {len(results)} messages from file"
                        message_display = f"File: {file.filename}"
                    else:
                        error = "Please upload a .txt or .csv file"
                else:
                    error = "Please select a file"
            else:
                error = "Please enter a message or upload a file"

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        'index.html',
        result=result,
        message_display=message_display,
        error=error,
        confidence=confidence
    )


@app.route('/history')
def history():
    retrain_message = request.args.get('retrain_message')
    return render_template('history.html', history=get_history(), retrain_message=retrain_message)


@app.route('/feedback', methods=['POST'])
def feedback():
    message = request.form.get('message', '').strip()
    predicted_result = request.form.get('predicted_result', '').strip()
    actual_label = request.form.get('actual_label', '').strip()

    if not message or not predicted_result:
        return redirect(url_for('home'))

    add_feedback(message, predicted_result, actual_label)
    return redirect(url_for('history'))


@app.route('/retrain', methods=['POST'])
def retrain():
    feedback_rows = get_feedback()
    if not feedback_rows:
        return redirect(url_for('history', retrain_message='No feedback yet. Label some messages first.'))

    train_data, _ = load_and_prepare_data(split_ratio=1.0)
    additional_data = []
    for row in feedback_rows:
        text = row['message'].strip()
        if not text:
            continue
        label = 1 if row['actual_label'] == 'Spam' else 0
        additional_data.append((preprocess(text), label))

    if not additional_data:
        return redirect(url_for('history', retrain_message='No valid feedback samples found.'))

    combined_data = train_data + additional_data
    updated_model = NaiveBayesClassifier()
    updated_model.train(combined_data)
    updated_model.save_model(MODEL_PATH)

    global model
    model = updated_model

    return redirect(url_for('history', retrain_message=f'Model retrained with {len(additional_data)} feedback samples.'))


@app.route('/clear-history', methods=['POST'])
def clear_history():
    conn = get_db_connection()
    conn.execute('DELETE FROM history')
    conn.commit()
    conn.close()
    return redirect(url_for('history'))


@app.route('/export-history')
def export_history():
    history_rows = get_history()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['message', 'result', 'timestamp'])
    for row in history_rows:
        writer.writerow([row['message'], row['result'], row['timestamp']])

    response = Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=spam_history.csv'}
    )
    return response


if __name__ == '__main__':
    app.run(debug=True)