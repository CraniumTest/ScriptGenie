from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize Flask app and SocketIO for real-time collaboration
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load pre-trained model and tokenizer from the transformers library
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Example function to generate script draft
def generate_script(prompt, max_length=250):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    length = data.get('length', 250)
    script = generate_script(prompt, length)
    return jsonify({'generated_script': script})

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000)
