# app.py
from flask import Flask, request, jsonify, render_template
import json
from processing_RAG import process_RAG 
from processing_classification import predict_text

app = Flask(__name__)

CLASSIFICATION_DICTIONARY = {
    0: "No Symptoms Detected (0)",
    1: "Symptoms Detected and Context is Relevant (1)",
    -1: "Symptoms Detected but Context is Irrelevant (-1)"
}

# Home route to serve the front end
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_api():
    try:
        # Expect a JSON body with key "texts" as an array of strings
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "Invalid input. Expecting JSON with key 'texts'."}), 400

        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list."}), 400
        print(texts[0])
        results = []
        predicted_label = int(predict_text(texts[0]))
        print(predicted_label)
        # If predicted label is "0", return only text and label
        if predicted_label == 0:
            results.append({
                "text": texts[0],
                "predicted_label": CLASSIFICATION_DICTIONARY[predicted_label]
            })
        # Otherwise, process the text using process_RAG and include annotation
        else:
            annotation = process_RAG(texts[0])
            results.append({
                "text": texts[0],
                "predicted_label": CLASSIFICATION_DICTIONARY[predicted_label],
                "annotation": annotation
                })
        return jsonify({"results": results})
        print(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)