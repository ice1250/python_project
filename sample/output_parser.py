"""
언어 모델을 사용하다 보면 내가 원하는 자료형 형태로 출력돼야 하는 경우들이 있습니다.
예를 들어, 리스트 형태의 결과 JSON 형태의 결과로 출력해야 하는 경우입니다.
아웃풋 파서는 출력 형태를 지정한 형태로 출력할 수 있도록 해 줍니다.

- Format instructions : 원하는 출력의 형태를 LLM에 전달
- Parser : 결괏값을 원하는 지정한 형태로 추출
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

prompt = PromptTemplate(
    template="{subject} 5개를 추천해 줘.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

llm = OpenAI()

prompt_result = prompt.format(subject="책")
output = llm.invoke(prompt_result)


def call_output_parser_1():
    print(output_parser.parse(output))
    print(output)
