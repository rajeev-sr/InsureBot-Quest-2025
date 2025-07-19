import uuid
from gemini_model import gemini_embedding
from lancedb_connection import table
from text_cleaning import text_cleaning
from text_translate import text_translate
from chunks import split_text
import time

def add_to_vector_db(tb,embeddings,text, language, doc_type, source):
    embedding = embeddings.embed_query(text)

    tb.add([{
        "id": str(uuid.uuid4()),
        "text": text,
        "embedding": embedding,
        "language": language,
        "type": doc_type,
        "source": source
    }])
    print("Added to vector DB")

def add_to_lancedb():
    audio_text=text_cleaning()
    time.sleep(60)
    translated_text=text_translate(audio_text)
    tb = table()
    embeddings = gemini_embedding()

    for text in audio_text:
        chunk=split_text(text)
        if "संशोधित" in chunk[0] :
            chunk=chunk[1:]

        for t in chunk:
            add_to_vector_db(tb,embeddings,t, "Hindi", "conversation", "call transcript hindi")
            
    for text in translated_text:
        chunk=split_text(text)
        if "translat" in chunk[0] :
            chunk=chunk[1:]

        for t in chunk:
            add_to_vector_db(tb,embeddings,t, "English", "conversation", "call transcript english")

    with open("/home/rajeev-kumar/Desktop/InsureBot-Quest-2025/src/components/Knowledge Base.txt") as f:
        kb_text = f.read()

    f.close()
    chunk=split_text(kb_text)
    for t in chunk:
        add_to_vector_db(tb,embeddings,t, "English", "knowledge base", "Scenario Based Talking Points")


if __name__ == "__main__":
    add_to_lancedb()

