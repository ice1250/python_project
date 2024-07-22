from langchain.agents import AgentExecutor, create_openai_functions_agent, initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import YouTubeSearchTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI


def call_agent_1():
    llm = ChatOpenAI(model="gpt-4")
    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name="google_search",
            description="Search Google for useful results.",
            func=search.run,
        ),
        Tool(
            name="llm-math",
            description="수학계산",
            func=search.run,
        )
    ]

    agent = create_openai_functions_agent(llm, tools, ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]))
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke(
        {"input": "미국대통령 나이를 10으로 나눈 값을 알려주세요"}
    )
    print(f'답변: {response["output"]}')


def call_agent_2():
    llm = ChatOpenAI(model="gpt-4")

    tools = load_tools(["ddg-search"]) + [YouTubeSearchTool()]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    result = agent.invoke({"input": "영화 토토로의 주제곡을 찾아 유튜브 링크를 알려 주세요."})
    print(result)


def call_agent_3():
    from langchain import hub
    prompt = hub.pull("artryu/ai-test1")
    llm = ChatOpenAI()
    search = GoogleSearchAPIWrapper()

    # tools = [
    #     Tool(
    #         name="google_search",
    #         description="Search Google for useful results.",
    #         func=search.run,
    #     ),
    #     Tool(
    #         name="llm-math",
    #         description="수학계산",
    #         func=search.run,
    #     )
    # ]

    # tools = load_tools(["ddg-search", "llm-math"], llm=llm)
    tools = load_tools(["ddg-search", "llm-math"], llm=llm)

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke(
        {"input": "미국 대통령의 나이를 10으로 나눈 값은 무엇인가요?"}
    )
    print(f'답변: {response["output"]}')
