import gradio as gr
from transformers import pipeline

# Load the summarization model (downloads automatically on first run)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

def summarize_article(article_text, max_length, min_length):
    """
    Takes an article and returns a concise summary.
    """
    if not article_text.strip():
        return "⚠️ Please paste an article to summarize."

    if len(article_text.split()) < 30:
        return "⚠️ Article is too short to summarize. Please paste a longer article."

    if min_length >= max_length:
        return "⚠️ Min length must be less than Max length."

    try:
        result = summarizer(
            article_text,
            max_length=int(max_length),
            min_length=int(min_length),
            do_sample=False
        )
        return result[0]["summary_text"]

    except Exception as e:
        return f"❌ Error during summarization: {str(e)}"


# ── Sample articles for quick testing ──────────────────────────────────────────
SAMPLE_ARTICLE = """Artificial intelligence (AI) is transforming every industry at an unprecedented pace. 
From healthcare to finance, education to transportation, AI systems are being deployed to automate tasks, 
improve decision-making, and unlock new capabilities that were once considered science fiction.

In healthcare, AI algorithms can now detect certain cancers from medical images with accuracy matching or 
exceeding that of experienced radiologists. In finance, machine learning models identify fraudulent 
transactions in milliseconds, protecting millions of customers. Self-driving cars, powered by deep neural 
networks, are being tested on public roads across multiple countries.

The technology behind modern AI — particularly large language models (LLMs) — has seen explosive growth 
since 2017, when the Transformer architecture was introduced. Models like GPT-4, Claude, and Gemini can 
understand and generate human-like text, write code, analyze data, and even engage in complex reasoning tasks.

However, this rapid advancement raises significant concerns. Job displacement is a major worry — economists 
estimate that AI could automate up to 40% of current jobs within the next decade. Privacy and surveillance 
risks are mounting as AI-powered facial recognition becomes more widespread. Bias in AI systems can 
perpetuate and amplify existing social inequalities if training data is not carefully curated.

Governments around the world are scrambling to regulate AI. The European Union passed the landmark AI Act 
in 2024, the first comprehensive AI law globally. The United States, China, and other major powers are 
developing their own regulatory frameworks, though international coordination remains elusive.

Despite the challenges, most experts agree that AI will continue to advance rapidly. The key question 
is not whether AI will transform society, but how humanity will manage that transformation responsibly — 
ensuring benefits are broadly shared while minimizing harms to individuals and communities."""


# ── Gradio Interface ────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Article Summarizer",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 📰 AI Article Summarizer
        ### Powered by Facebook BART (bart-large-cnn)
        Paste any long article below and get a concise, accurate summary in seconds.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            article_input = gr.Textbox(
                label="📄 Article Text",
                placeholder="Paste your article here...",
                lines=15,
                max_lines=30,
            )

            with gr.Row():
                max_len = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=150,
                    step=10,
                    label="Max Summary Length (words)",
                )
                min_len = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Min Summary Length (words)",
                )

            with gr.Row():
                summarize_btn = gr.Button("✨ Summarize", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", size="lg")

        with gr.Column(scale=1):
            summary_output = gr.Textbox(
                label="📝 Summary",
                lines=15,
                interactive=False,
                placeholder="Your summary will appear here...",
            )

    gr.Markdown("---")
    gr.Markdown("### 🧪 Try a Sample Article")
    sample_btn = gr.Button("Load Sample Article", variant="secondary")

    # ── Event handlers ──────────────────────────────────────────────────────────
    summarize_btn.click(
        fn=summarize_article,
        inputs=[article_input, max_len, min_len],
        outputs=summary_output,
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[article_input, summary_output],
    )

    sample_btn.click(
        fn=lambda: SAMPLE_ARTICLE,
        inputs=[],
        outputs=article_input,
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - Increase **Max Length** for more detailed summaries
        - Decrease **Min Length** for shorter, punchier summaries
        - Works best with articles 200–2000 words long
        """
    )

if __name__ == "__main__":
    demo.launch()