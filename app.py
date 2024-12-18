from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

# Function to generate a clean response
def generate_response(question):
    # Explicitly handle the case for "Hi"
    
    """
    Generate a clean response for a given input question.
    """
    # Ensure the question ends with a question mark if not present
    if not question.endswith('?'):
        question += '?'
# Specific handling for "Hi"
    if question.strip().lower() in ["hi", "hi?", "hello", "hey"]:
        return "Hi, Good day!"
    # Explicitly structure the input to guide the model's response
    input_text = f"Question: {question} Answer briefly:"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=50, truncation=True)

    # Generate response
    bad_words_ids = [[tokenizer.convert_tokens_to_ids('[SEP]')]]
    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=70,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        bad_words_ids=bad_words_ids,
        temperature=0.7,
        top_p=0.9,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process to clean the response:
    # Post-process to remove repeated question, artifacts, and prefix
    response = response.replace(question, "").replace("Answer briefly:", "").strip()
    response = response.split("[SEP]")[0]
    response = response.split("_______________________________________________")[0]
    response = response.split("@")[0]  # Remove email-like artifacts
    response = response.strip()
    # 1. Remove the question and prompt (e.g., "Question: education? Answer briefly:")
    response = re.sub(r"^Question:\s*", "", response).strip()
    response = response.replace("Answer briefly:", "").strip()


    # 2. Remove unwanted tokens like [SEP] if present
    response = response.replace("[SEP]", "").strip()

    # 3. Ensure we return only the first clean sentence
    response = response.split(".")[0].strip() + "."

    # Grammar correction: Convert lowercase "i" to uppercase "I"
    response = re.sub(r'\bi\b', 'I', response)  # Use word boundaries to match "i" as a pronoun

    return response.split(".")[0] + "."


# Flask Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    # Generate a clean response based on the provided question
    response = generate_response(question)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
