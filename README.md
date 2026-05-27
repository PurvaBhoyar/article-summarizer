
# 📰 AI Article Summarizer

An AI-powered article summarizer built with **Hugging Face Transformers** and **Gradio**.

## Features
- Accepts long articles as input
- Generates concise, accurate summaries using Facebook's BART model
- Interactive web UI with adjustable summary length
- Built-in sample article for quick testing

## Model
Uses [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) — a state-of-the-art abstractive summarization model fine-tuned on CNN/DailyMail news articles.

## Controls
| Parameter | Description |
|---|---|
| **Max Length** | Maximum number of words in the summary (50–500) |
| **Min Length** | Minimum number of words in the summary (10–200) |

## Tech Stack
- 🤗 Hugging Face Transformers
- 🎨 Gradio (Web UI)
- 🔥 PyTorch

## How to Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/article-summarizer
cd article-summarizer
pip install -r requirements.txt
python app.py
```
Then open http://localhost:7860 in your browser.
