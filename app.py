from flask import Flask, request, render_template, redirect, url_for
from naive_bayes import NaiveBayesClassifier
from data_preprocessing import preprocess
from datetime import datetime
import os

app = Flask(__name__)

# Load the trained model
model = NaiveBayesClassifier.load_model('spam_model.pkl')

# In-memory history storage
classification_history = []

def add_to_history(message, result):
    """Add a classification to history"""
    classification_history.append({
        'message': message,
        'result': result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    message_display = None
    error = None
    
    if request.method == 'POST':
        try:
            # Handle text input
            if 'message' in request.form and request.form['message'].strip():
                message = request.form['message'].strip()
                tokens = preprocess(message)
                prediction = model.predict(tokens)
                result = 'Spam' if prediction == 1 else 'Ham'
                message_display = message
                add_to_history(message, result)
            
            # Handle file upload
            elif 'file' in request.files:
                file = request.files['file']
                if file and file.filename:
                    if file.filename.endswith('.txt'):
                        # Text file
                        content = file.read().decode('utf-8')
                        tokens = preprocess(content)
                        prediction = model.predict(tokens)
                        result = 'Spam' if prediction == 1 else 'Ham'
                        message_display = content
                        add_to_history(content, result)
                    
                    elif file.filename.endswith('.csv'):
                        # CSV file - classify each line
                        import csv
                        content = file.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        results = []
                        
                        for line in lines[1:]:  # Skip header if present
                            try:
                                tokens = preprocess(line)
                                prediction = model.predict(tokens)
                                result_type = 'Spam' if prediction == 1 else 'Ham'
                                results.append((line[:50], result_type))
                                add_to_history(line, result_type)
                            except:
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
    
    return render_template('index.html', result=result, message_display=message_display, error=error)

@app.route('/history')
def history():
    return render_template('history.html', history=classification_history)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    global classification_history
    classification_history = []
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)