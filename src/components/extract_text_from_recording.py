import os
import assemblyai as aai
from langchain_community.document_loaders.assemblyai import AssemblyAIAudioTranscriptLoader, TranscriptFormat
from dotenv import load_dotenv
load_dotenv()



def audio_to_text():
    ass_api_key=os.getenv("ASSEMBLYAI_API_KEY")

    os.environ["ASSEMBLYAI_API_KEY"]=ass_api_key

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        entity_detection=True,
        language_code="hi"  # explicitly set to Hindi
    )

    audio_text=[]
    directory_path = "/home/rajeev-kumar/Desktop/InsureBot-Quest-2025/src/components/output"

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
    
            # Initialize the loader with config and desired transcript format
            loader = AssemblyAIAudioTranscriptLoader(
                file_path=full_path,
                transcript_format=TranscriptFormat.TEXT,
                config=config,
                api_key=ass_api_key
            )
            # Load the transcription document(s)
            docs4 = loader.load()

            audio_text.append(docs4[0].page_content)
            # print(docs4[0].page_content)
    return audio_text

if __name__ == "__main__":
    audio_to_text()