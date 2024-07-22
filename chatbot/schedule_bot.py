import os.path

import gradio as gr
import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

system_schedule_ko = """일정 관리에 대한 사용자 입력이 주어집니다.
'schedule_type,' 'schedule_content,' 'schedule_content_detail,' 'year,'
'month,' 'day,' 'hour,' 'min' 과 같은 구성 요소가 있습니다.
'schedule_type'은 조회, 삭제, 추가, 업데이트 중 하나일 수 있습니다.
'year,' 'month,' 'day,' 'hour,' 'min' 에 대한 값은 숫자여야 합니다.
'schedule_content,' 'schedule_content_detail,' 'year,' 'month,' 'day,' 'hour,' 'min' 에 대한 입력은 json 문자열 형식으로 이루어져야 합니다.
"""
system_setting_schedule = SystemMessagePromptTemplate.from_template(system_schedule_ko)

# 프롬프트 설정
schedule_prompt = ChatPromptTemplate.from_messages([
    system_setting_schedule,
    MessagesPlaceholder(variable_name="HeyMate_schedule"),
    HumanMessagePromptTemplate.from_template("{master_user}"),
])

# 메모리 설정
schedule_memory = ConversationBufferWindowMemory(
    memory_key="HeyMate_schedule",
    ai_prefix="AI 인공지능 비서 HeyMate 일정 관리",
    human_prefix="사용자",
    return_messages=True,
    k=10
)

# llm 모델 정의
chatgpt = ChatOpenAI(
    temperature=0.3,
    max_tokens=2048,
    model_name="gpt-3.5-turbo"
)

# llmchain 정의
conversation_schedule = LLMChain(
    prompt=schedule_prompt,
    memory=schedule_memory,
    llm=chatgpt,
    verbose=True
)

# 처음 파일 생성
initial_df = pd.DataFrame(
    columns=['schedule_type', 'schedule_content', 'schedule_content_detail', 'year', 'month', 'day', 'hour', 'min'])
excel_file_path = "schedule.xlsx"
initial_df.to_excel(excel_file_path, index=False)
print(f"데이터를 저장할 {excel_file_path} 파일이 생성되었습니다.")


def schedule_bot_save(submit_file):
    temp_excel_file = pd.read_excel(submit_file)
    temp_excel_file.to_excel(excel_file_path, index=False)


def schedule_bot_chat(message, chat_history):
    answer = conversation_schedule({"master_user": message})
    ai_answer = answer['text']

    try:
        import json
        schedule_dic = json.loads(ai_answer)
        if schedule_dic['schedule_type'] == '추가':
            schedule_df = pd.read_excel(excel_file_path)
            schedule_df = pd.concat([schedule_df, pd.DataFrame([schedule_dic])], ignore_index=True)
            schedule_df.to_excel(excel_file_path, index=False)
            chat_history.append([message, f"{schedule_dic['schedule_content']}_일정이 추가되었습니다."])
        elif schedule_dic['schedule_type'] == '조회':
            schedule_df = []
            if os.path.isfile(excel_file_path):
                schedule_df = pd.read_excel(excel_file_path)
            chat_history.append([message, "전체 일정을 말씀드리겠습니다."])

            for idx, event in schedule_df.iterrows():
                chat_history.append(
                    [None,
                     f"{idx + 1}. 일정: {event['schedule_content']}, 일정 시간: {event['year']}년 {event['month']}월 {event['day']}일 {event['hour']}시 {event['min']}분, 일정 내용: {event['schedule_content_detail']}"]
                )
    except:
        chat_history.append([message, ai_answer])
    return "", chat_history


def create_schedule_bot_tab():
    with gr.Tab("일정 관리봇"):
        # 1
        gr.Markdown(
            value="""
                    # <center>일정 관리봇</center>
                    <center>AI 인공지능 비서 HeyMate입니다. 일정 관리를 위한 봇입니다.</center>
                    """
        )
        chatbot_schedule = gr.Chatbot(
            value=[
                [None, "안녕하세요, 일정 이름, 시간, 일정 설명으로 일정을 추가할 수 있습니다.\n\
                        예시: 크리스미스, 2023년 12월 25일 12시 00분, 올해의 크리스마스 일정 추가해 줘\n\
                        전체 일정이 보고 싶다면 전체 일정 보여 줘 라고 말해 주세요"]
            ],
            label="일정 관리"
        )
        with gr.Row():
            msg_schedule = gr.Textbox(
                label="채팅",
                lines=1,
                placeholder="채팅 입력 창",
                scale=8
            )
            cb_schedule_submit_btn = gr.Button(
                value="보내기",
                scale=2,
                variant="primary"
            )
            cb_schedule_submit_btn.click(
                fn=schedule_bot_chat,
                inputs=[msg_schedule, chatbot_schedule],
                outputs=[msg_schedule, chatbot_schedule]
            )
            msg_schedule.submit(
                fn=schedule_bot_chat,
                inputs=[msg_schedule, chatbot_schedule],
                outputs=[msg_schedule, chatbot_schedule]
            )
        schedule_file = gr.File(
            label="일정 파일을 업로드해 주세요",
            scale=8,
            height=100
        )
        schedule_file.change(
            fn=schedule_bot_save,
            inputs=[schedule_file]
        )
