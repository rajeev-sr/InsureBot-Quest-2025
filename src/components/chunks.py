from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=800, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks
