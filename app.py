# import gradio as gr
# from chatbot.document_bot import create_document_bot_tab
# from chatbot.schedule_bot import create_schedule_bot_tab
# from chatbot.voice_bot import create_voice_bot_tab
#
# with gr.Blocks() as app:
#     create_voice_bot_tab()
#     create_document_bot_tab()
#     create_schedule_bot_tab()
#
# app.launch(debug=True, share=True)


# from langchain_core.callbacks import StreamingStdOutCallbackHandler
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
#
# # model
# prompt1 = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
# prompt2 = ChatPromptTemplate.from_template("Translate english to chinese. <Question>: {english_input}")
#
# chain = (
#         prompt1
#         | ChatOpenAI(model="gpt-4o")
#         | StrOutputParser()
# )  # 순서가 중요!
# # chain.invoke({ "input": "What is the largest planet in the solar system?" })
#
# chain2 = (
#         {"english_input": chain}
#         | prompt2
#         | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
#         | StrOutputParser()
# )
#
# chain2.invoke({"input": "What is the largest planet in the solar system?"})

import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt = ChatPromptTemplate.from_messages("어떤 질문이던지 친절하게 답변을 해주세요. <Question>: {input}")


def load_memory(input):
    print('input:', input)
    return memory.load_memory_variables({})['chat_history']


chain_korean = (
        prompt
        | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        | StrOutputParser()
        | memory

)  # 순서가 중요!


def answer_question(message, history):
    print('history:', history)
    response = chain_korean.stream({"input": message})
    partial_message = ""
    for chunk in response:
        if chunk is not None:
            partial_message = partial_message + chunk
            yield partial_message


def predict(message, history):
    from openai import OpenAI
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = OpenAI().chat.completions.create(model='gpt-3.5-turbo',
                                                messages=history_openai_format,
                                                temperature=1.0,
                                                stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


gr.ChatInterface(answer_question).launch()
