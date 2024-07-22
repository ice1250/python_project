# ConversationBufferMemory
def call_history_1():
    """
    ConversationBufferMemory 는 대화 내용을 그대로 저장합니다.
    """
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory()

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
    )

    conversation.predict(input="안녕하세요 제 이름은 art 입니다.")
    conversation.predict(input="제 이름을 이용해 멋진 별명을 만들어주세요.")


# ConversationBufferMemory
def call_history_2():
    """
    ConversationBufferMemory 는 지정한 개수만큼의 대화만 기억합니다.
    """
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "안녕, 나이 직업은 헤이마트 주인이야."},
                        {"output": "헤이마트 주인님 반갑습니다."})
    memory.save_context({"input": "나는 서울에서 살고 있어"},
                        {"output": "참 좋은 곳에서 사시는군요"})

    some = memory.load_memory_variables({})

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
    )

    conversation.predict(input="내 직업이 뭐라고 했지?")


# ConversationSummaryBufferMemory
def call_history_3():
    """
    ConversationSummaryBufferMemory 는 대화 내용을 요약해서 저장합니다.
    """
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0)

    sale_info = ("헤이마트에서 세일 상품입니다. \
    신선한 과일을 20% 할인 판매합니다. \
    채소류는 15% 할인 판매합니다. \
    그 외 다양한 제품을 할인 판매하니 방문해 주세요.")

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "헤이마트"}, {"output": "안녕하세요"})
    memory.save_context({"input": "궁금한게 있어"}, {"output": "무엇을 알려드릴까요?"})
    memory.save_context({"input": "할인 상품 정보를 알려줘."}, {"output": f"{sale_info}"})
    memory.load_memory_variables({})
