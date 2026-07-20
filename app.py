from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq
import os, traceback
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import pdfplumber
import io

load_dotenv()
app = Flask(__name__)
CORS(app)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def extract_text_from_file(file):
    filename = file.filename
    file_bytes = file.read()
    if filename.lower().endswith(('png','jpg','jpeg','webp')):
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image, lang='eng+tel')
        return f"User uploaded an image. Text found: {text}. Describe and answer."
    elif filename.lower().endswith('.pdf'):
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return f"User uploaded PDF. Content: {text[:3000]}. Summarize."
    elif filename.lower().endswith('.txt'):
        return file_bytes.decode('utf-8')
    else:
        return "File type not supported"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    q = request.form.get('message', '').strip()
    user_lang = request.form.get('lang', 'en-IN')
    file = request.files.get('file')

    try:
        final_prompt = q
        if file:
            file_content = extract_text_from_file(file)
            final_prompt = f"{file_content}\n\nUser question: {q}"
        if not final_prompt:
            final_prompt = "Describe the uploaded file"

        system_prompt = """You are My Talkie, a friendly voice assistant.
        RULE: Reply in the same language as user. Default English.
        Keep answers short for voice. If user asks 'what is your name' reply 'My name is My Talkie'."""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        answer = chat_completion.choices[0].message.content
        return jsonify({"reply": answer, "lang": user_lang})
    except Exception as e:
        print("ERROR:", traceback.format_exc())
        return jsonify({"reply": f"Error: {str(e)}", "lang": "en-IN"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
