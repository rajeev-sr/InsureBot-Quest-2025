# Veena: Multilingual Voice-Based Insurance Call Assistant

**Veena** is an intelligent voice assistant built to simulate and automate real insurance customer service calls for **ValuEnable Life Insurance**. She handles multilingual conversations, follows a strict branching calling script, supports natural voice interaction, and retrieves context-aware answers using Gemini Pro and LanceDB.

---

## Features

- Voice-based two-way communication via microphone and speaker
- Automatically detects customer language and switches to it (supports Hindi, Marathi, Gujarati, English)
- Human-like conversation with support for interruption and mid-sentence turn-taking
- Gracefully handles unrecognized audio or silence without raising errors
- Strictly follows a branching insurance service call script
- Uses Gemini LLM and Gemini Embeddings with RAG via LanceDB for dynamic, context-aware replies

---

## Tech Stack

| Layer             | Technology                        |
|------------------|------------------------------------|
| LLM               | Google Gemini Pro (LangChain)      |
| Embeddings        | Google GenerativeAI Embeddings     |
| Vector Store      | LanceDB                            |
| Speech-to-Text    | `speech_recognition` with Google ASR |
| Text-to-Speech    | `gTTS` with `pydub` playback        |
| UI                | Streamlit                          |
| Preprocessing     | AssemblyAI Audio Transcript Loader |

---

## Pipeline Overview

### 1. Data Preprocessing

- Call recordings (`.wav`) are first slowed down and transcribed using `AssemblyAIAudioTranscriptLoader`.
- These transcriptions are reconstructed into complete, well-structured text using Gemini LLM with a custom prompt.
- The output text is also translated (Hindi/Marathi/Gujarati) using Gemini with language-specific prompts.
- Each version (English + translation) is stored in LanceDB in chunks with a `source` tag (e.g., `call_transcript`, `knowledge_base`).

### 2. RAG Integration

- Transcripts and KB chunks are embedded using Gemini Embeddings.
- All vectors are indexed in LanceDB for efficient similarity search.
- On receiving a customer voice input:
  - The speech is transcribed
  - Language is detected from the first user utterance
  - Top-k relevant context is retrieved from LanceDB using vector similarity
  - Context + calling script + conversation history is passed to Gemini LLM to generate an appropriate response

### 3. Interactive Voice Flow

- The conversation starts with the predefined script:  
  `Veena: Hello and very Good Morning Sir, May I speak with {policy_holder_name}?`
- Speech Recognition handles the customer's reply.
- Veena's TTS auto-stops if the customer interrupts mid-sentence.
- If the customer speaks in a different language (from their first input), Veena continues in that language.
- If the response is empty or unrecognizable, Veena repeats:  
  `"I'm sorry, I didn't understand that. Could you please repeat?"`  
  This loop continues until a clear response is received.

---

## Performance Summary

- Average audio transcription latency: ~2.5 sec
- Average Gemini response generation latency: ~1.8 sec
- Vector search latency (LanceDB): ~40 ms
- End-to-end response time per turn: ~4.5 sec (including TTS playback)

---

Here's the updated **"How to Run"** section of the `README.md` with `.env` setup, environment creation, and requirements installation instructions:

---

````markdown
## How to Run

1. **Clone the repository**

```bash
git clone <url>
cd InsureBot-Quest-2025
````

2. **Create a virtual environment**

```bash
python3 -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

3. **Configure environment variables**

* Copy the `.env.example` to `.env` and update with your actual keys:

```bash
cp .env.example .env
```
* or add these to your .env 
```bash
GOOGLE_API_KEY="<GOOGLE_API_KEY>"
lancedb_uri="<lancedb_uri>"
lancedb_api_key="<lancedb_api_key>"
lancedb_region="<lancedb_region>"
ASSEMBLYAI_API_KEY="<ASSEMBLYAI_API_KEY>"
```

* Fill in your Gemini API key and other configs in `.env`.

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Prepare data**

* Place `.wav` call recordings and `.txt` knowledge base files into the appropriate folders.
* Run the preprocessing script to transcribe, translate, and store chunked embeddings into LanceDB:

```bash
python preprocess_and_store.py
```

6. **Launch the Streamlit interface**

```bash
streamlit run app.py
```

7. **Use the interface**

* Click `Start` to initiate the voice conversation.
* Click `End` to stop the session.

```