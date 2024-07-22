from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain_openai import OpenAI, ChatOpenAI


def call_prompt_1():
    prompt_template = PromptTemplate.from_template("{goods}의 성분에 대해 알려줘")
    prompt_result = prompt_template.format(goods="고양이")

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    result = llm.invoke(prompt_result)
    print(result)


def call_prompt_2():
    chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo")
    system_message = "당신은 {language}선생님입니다. {language}로 답변해 주세요"
    system_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_template = "{text}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    message = chat_prompt.format_messages(language="영어", text="대한민국 수도는?")

    result = chatgpt.invoke(message)
    print(result.content)
