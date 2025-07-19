from langchain_core.prompts import ChatPromptTemplate
from extract_text_from_recording import audio_to_text
from gemini_model import gemini_model



def text_cleaning():
    cleaned_text=[]

    audio_text=audio_to_text()
    model=gemini_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert Hindi conversation editor, speaker identifier, and natural dialogue writer.

                Your task is to carefully correct and clean the following Hindi conversation transcription.

                **Rules:**
                - Fix grammar, punctuation, and broken phrases.
                - If any part is incomplete or garbled, intelligently complete it based on the conversation’s context.
                - Classify each line as either **'एजेंट:'** (agent) or **'ग्राहक:'** (customer).
                - Make the conversation sound polite, friendly, and natural — like a real human phone conversation in India.
                - Use commonly spoken English words naturally in-between Hindi where appropriate, such as **'रिवाइज'**, **'रिगार्डिंग'**, **'कॉल बैक'**, **'अपडेट'**, **'पेमेंट'**, **'पॉलिसी'**, **'ट्रांजैक्शन आईडी'** etc.
                - Avoid very formal Hindi words like **'पुनर्जीवित'** or **'संबंध'**; replace them with natural spoken alternatives like **'रिवाइव'**, **'रिगार्डिंग'**, **'कॉल'**, **'कन्वीनियंट'**, etc.
                - Use polite fillers like **'जी'**, **'ठीक है'**, **'धन्यवाद'**, **'ओके'**, etc., to keep the tone friendly and natural.
                - Do not add unrelated extra information.
                - Preserve the meaning and intent exactly.
                - Keep the entire conversation in clean, polite, meaningful, human-like Hindi with natural conversational English words mixed-in where typically used.
                - don't write "यहाँ संशोधित और साफ की गई बातचीत है:"
                """
            ),
            (
                "human",
                """
                Here is the transcription text:
                {text}

                Please rewrite this conversation in clean, polite, meaningful, natural human-like Hindi conversation style, clearly labeling each line with **'एजेंट:'** or **'ग्राहक:'**.
                """
            ),
        ]
    )

    chain=prompt | model

    for text in audio_text:
        res=chain.invoke({"text": text})
        cleaned_text.append(res.content)

    return cleaned_text

if __name__ == "__main__":
    print(text_cleaning() )
