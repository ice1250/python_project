from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

chatgpt = ChatOpenAI(
    model="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
messages = [
    SystemMessage(content="당신은 파이썬 프로그래머 입니다."),
    HumanMessage(content="파이썬에서 문자의 길이를 알려 주는 명령어는?"),
]


def call_chat_model_1():
    result = chatgpt.invoke(messages)
    print(result.content)
