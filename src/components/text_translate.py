from langchain_core.prompts import ChatPromptTemplate
from text_cleaning import text_cleaning
from gemini_model import gemini_model



def text_translate(audio_text):
    translated_text=[]
    model=gemini_model()
    translation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a professional conversation translator.

                Your job is to carefully translate the following polite, human phone conversation from Hindi to clear, grammatically correct English.

                **Rules:**
                - Translate each sentence faithfully, preserving its meaning and tone.
                - Classify each sentence as either **'Agent:'** or **'Customer:'**.
                - Keep the conversation polite, friendly, and natural — like a real human phone call in English.
                - Do not omit, guess, or add any extra information.
                - Retain any financial, insurance-related or technical terms as-is (like 'premium', 'policy number', 'fund value', 'transaction ID').
                - Use polite conversational fillers like 'Sure', 'Okay', 'Thank you', 'Alright' where appropriate.
                """
            ),
            (
                "human",
                """
                Here is the cleaned Hindi conversation:
                {text}

                Please translate this conversation into polite, meaningful, natural human-like English, clearly labeling each line as **'Agent:'** or **'Customer:'**.
                """
            ),
        ]
    )

    chain=translation_prompt | model

    for text in audio_text:
        res=chain.invoke({"text": text})
        translated_text.append(res.content)

    return translated_text

if __name__ == "__main__":
    audio_text=text_cleaning() 
    print(text_translate(audio_text) )
