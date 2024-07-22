# 스트리밍 방식
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


def call_chat_1():
    return chatgpt.invoke("서울의 인기 관광장소는?")
