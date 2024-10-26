from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import request

from llmmodels.config import MODEL_CONFIGS
from llmmodels.ConversationManagerService import ConversationManager


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

manager = ConversationManager()

@app.route('/')
def home():
    return jsonify({"message": "Call Multiple LLM"})

@app.route('/getResponse')
def analyze_text():  
    query = request.args.get('query')
    model = request.args.get('model')  
    if model not in MODEL_CONFIGS:
        model="gpt-3.5-turbo"
    print("Using Model :"+model)        
    response = manager.ask(query, model)
    return response


if __name__ == '__main__':
    app.run(debug=False)


