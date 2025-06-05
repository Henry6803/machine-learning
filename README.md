# Sibu Chatbot Backend

## Getting Started

1. **Install dependencies:**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare RAG index:**
   - Put `.txt` files about Sibu, Sarawak in `data/rag_corpus/docs/`.
   - Run `python build_faiss_index.py` in `data/rag_corpus/`.

3. **Prepare QLoRA and classifier data:**
   - Place your fine-tuning and classifier data in `data/`.
   - Run `prepare_data.py`, `train_qlora.py`, and `classifier_train.py` as needed.

4. **Run the API:**
   ```bash
   python api/app.py
   ```

5. **Test the API:**
   ```bash
   curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "What are the best foods in Sibu?"}'
   ```

## Directory Structure

```
sibu-chatbot/
├── api/
├── finetune/
├── data/
│   ├── rag_corpus/
```

## Notes

- All models run locally, no internet needed after setup.
- Only answers questions about Sibu, Sarawak (prompt filter will block others).