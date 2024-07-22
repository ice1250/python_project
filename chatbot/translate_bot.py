import gradio as gr
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# model
prompt1 = ChatPromptTemplate.from_template("어떤 질문이던지 친절하게 답변을 해주세요. <Question>: {input}")
prompt2 = ChatPromptTemplate.from_template("Translate to english. <Question>: {input2}")
prompt3 = ChatPromptTemplate.from_template("Translate to chinese. <Question>: {input3}")

chain_korean = (
        prompt1
        | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        | StrOutputParser()
)  # 순서가 중요!

chain_english = (
        {"input2": chain_korean}
        | prompt2
        | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        | StrOutputParser()
)

chain_chinese = (
        {"input3": chain_korean}
        | prompt3
        | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        | StrOutputParser()
)


def answer_question(input):
    korean_answer = chain_korean.invoke({"input": input})
    english_answer = chain_english.invoke({"input": input})
    chinese_answer = chain_chinese.invoke({"input": input})
    return korean_answer, english_answer, chinese_answer


with gr.Blocks() as app:
    gr.Interface(fn=answer_question, inputs=gr.Textbox(lines=2, placeholder='Enter your question here...'),
                 outputs=[gr.Textbox(label='Korean Answer'), gr.Textbox(label='English Answer'), gr.Textbox(label='Chinese Answer')])

app.launch(debug=True)
