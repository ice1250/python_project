from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAI

ex_qa = [
    {
        "question": "홍길동에 대해서 알려 줘",
        "answer": "나이 : 31, 키 : 150, 사는 곳 : 대한민국"
    }, {
        "question": "헐크에 대해서 알려 줘",
        "answer": "나이 : 40, 키 : 180, 사는 곳 : 미국"
    }
]

ex_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}\n",
)


def call_few_shot_1():
    prompt = FewShotPromptTemplate(
        examples=ex_qa,
        example_prompt=ex_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    prompt_result = prompt.format(input="아이언맨에 대해서 알려줘")
    result = llm.invoke(prompt_result)
    print(result)  # Answer: 나이 : 45, 키 : 183, 사는 곳 : 미국
