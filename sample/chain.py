from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


# 멀티 체인
def call_chain_1():
    llm = OpenAI(temperature=0.7)
    template = """당신은 신문기자입니다. 제목과 같은 기사를 작성해 주세요.
    제목: {title}
    """
    prompt_template = PromptTemplate(
        input_variables=["title"], template=template
    )

    article_chain = prompt_template | llm

    llm = OpenAI(temperature=0.7)
    template = """해당 신문기사를 짧게 줄여 주세요.
    신문기사:
    {article}
    """
    prompt_template = PromptTemplate(
        input_variables=["article"], template=template
    )

    review_chain = prompt_template | llm

    overall_chain = article_chain | review_chain

    review = overall_chain.invoke({"title": "평화로운 대한민국"})
    print(review)


# 무조건 한국어로 말하도록 하는 봇
def call_chain_2():
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # model
    prompt1 = ChatPromptTemplate.from_template(
        "You are an expert in astronomy. Answer the question. <Question>: {input}")
    prompt2 = ChatPromptTemplate.from_template("Translate english to korean. <Question>: {english_input}")

    chain = (
            prompt1
            | ChatOpenAI(model="gpt-4o")
            | StrOutputParser()
    )  # 순서가 중요!
    # chain.invoke({ "input": "What is the largest planet in the solar system?" })

    chain2 = (
            {"english_input": chain}
            | prompt2
            | ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
            | StrOutputParser()
    )

    chain2.invoke({"input": "What is the largest planet in the solar system?"})
