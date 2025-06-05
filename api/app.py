from flask import Flask, request, jsonify
from rag import retrieve_context
from conversation import get_history, update_history, format_history
from prompt_filter import is_relevant_prompt, filter_manglish_only
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Constants
ADAPTER_PATH = "./qlora_adapter"
QWEN_MODEL = "Qwen/Qwen1.5-0.5B"

# Device assignment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer & models globally to avoid reloading on every request
try:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Model loading failed: {e}")
    tokenizer = None
    model = None

def qwen_generate(prompt, max_new_tokens=128):
    if tokenizer is None or model is None:
        return "Sorry boss, model not available right now. Please try again later."
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "Sorry boss, got problem with model inference."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    user_id = request.remote_addr  # Can use session or other user ID system

    if not is_relevant_prompt(user_input):
        return jsonify({"response": ["Sorry boss, I only answer questions about Sibu, Sarawak."]})

    history = get_history(user_id)
    formatted_history = format_history(history)
    retrieved_docs = retrieve_context(user_input)

    context_section = f"Context: {retrieved_docs}\n" if retrieved_docs and "No relevant documents found." not in retrieved_docs else ""

    instruction = (
        "IMPORTANT: YOU MUST answer ONLY in Malaysian English (Manglish). "
        "DO NOT use Bahasa Melayu, Chinese, or formal English. "
        "If you don't know, just try your best lah. Answer like this example:\n"
        "Example Manglish: 'Aiya, Sibu there ah, got plenty of nice food one, and the people super friendly lor. "
        "If you go, must try the kampua mee and kompia, confirm you like wan!'\n"
        "Remember: Use Manglish for your whole answer. Don't use BM or Chinese at all, ok bro?\n"
    )

    prompt = (
        instruction +
        context_section +
        f"Conversation so far:\n{formatted_history}\n"
        f"User: {user_input}\n"
        "Bot:"
    )

    response = qwen_generate(prompt)
    update_history(user_id, user_input, response)

    response_lines = [line.strip() for line in response.split('\n') if line.strip()]
    response_lines = filter_manglish_only(response_lines)
    return jsonify({"response": response_lines})

if __name__ == "__main__":
    print(f"Starting Flask server on {DEVICE}…")
    app.run(host="127.0.0.1", port=5000, debug=True)