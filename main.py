from flask import *
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from rag import RAGAgent
import os

load_dotenv()
app = Flask(__name__)
CORS(app)


@app.route("/")
def upload():
    return ""


@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        f = request.files["file"]
        file_path = "uploads/" + f.filename
        f.save(file_path)
        rag = RAGAgent(file_path, request.form.get("jobDesc"))
        try:
            resp = rag.rag_task(request.form.get("taskType"))
            os.remove(file_path)
            return make_response(resp, 200)
        except:
            return make_response("Invalid task type", 400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
