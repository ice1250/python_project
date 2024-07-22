import gradio as gr
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)


# 상담봇 - 채팅 및 답변
def counselling_bot_chat(message, chat_history):
    if message == "":
        return "", chat_history
    else:
        result_message = ""
        if len(chat_history) <= 1:
            messages = [
                SystemMessage(content="당신은 헤이마트의 상담원 입니다. 마트 상품과 관련되지 않은 질문에는 정중히 거절하세요."),
                AIMessage(content="안녕하세요, 헤이마트입니다. 상담을 도와드리겠습니다."),
                HumanMessage(content=message)
            ]
            result_message = conversation.predict(input=messages)
        else:
            result_message = conversation.predict(input=message)

        chat_history.append([message, result_message])
        return "", chat_history


# 상담봇 - 되돌리기
def counselling_bot_undo(chat_history):
    if len(chat_history) > 1:
        chat_history.pop()
    return chat_history


# 상담봇 - 초기화
def counselling_bot_reset(chat_history):
    chat_history = [[None, "안녕하세요, 헤이마트입니다. 상담을 도와드리겠습니다."]]


# 번역봇
def translate_bot(output_conditions, output_language, input_text):
    if input_text == "":
        return ""
    else:
        if output_conditions == "":
            output_conditions = ""
        else:
            output_conditions = "번역할 떄의 조건은 다음과 같습니다 : " + output_conditions

        system_content = """user가 입력한 내용을 번역해 주세요. 입력한 언어를 다른 설명 없이 바로 {0}로 번역해서 알려주세요. {1} """.format(
            output_language, output_conditions)

        print(system_content)

        completion = llm.invoke([
            {"role": "system",
             "content": system_content},
            {"role": "user", "content": input_text},
        ])
        return completion.content


def translate_bot_text_upload(files):
    loader = TextLoader(files)
    document = loader.load()
    return document[0].page_content


def translate_bot_pdf_upload(files):
    loader = PyPDFLoader(files)
    document = loader.load()
    return document[0].page_content


# 소설봇
def novel_bot(model, temperature, detail):
    client = ChatOpenAI(temperature=temperature, model_name=model)
    completion = client.invoke([
        {"role": "system", "content": "당신은 소설가입니다. 요청하는 조건에 맞춰 소설을 작성해 주세요."},
        {"role": "user", "content": detail},
    ])
    return completion.content


with gr.Blocks(theme="freddyaboulton/test-blue") as app:
    with gr.Tab("상담봇"):
        gr.Markdown(
            value="""
            # <center>상담봇</center>
            <center>헤이마트 상담봇입니다. 마트에서 판매하는 상품과 관련된 질문에 답변드립니다.</center>
            """
        )
        cb_chatbot = gr.Chatbot(
            value=[[None, "안녕하세요, 헤이마트입니다. 상담을 도와드리겠습니다."]],
            show_label=False
        )
        with gr.Row():
            cb_user_input = gr.Text(
                lines=1,
                placeholder="입력 창",
                container=False,
                scale=9
            )
            cb_send_button = gr.Button(
                value="보내기",
                scale=1,
                variant="primary",
                icon="images/send.png",
            )
            # 보내기
            cb_send_button.click(fn=counselling_bot_chat, inputs=[cb_user_input, cb_chatbot],
                                 outputs=[cb_user_input, cb_chatbot])
            cb_user_input.submit(fn=counselling_bot_chat, inputs=[cb_user_input, cb_chatbot],
                                 outputs=[cb_user_input, cb_chatbot])
        with gr.Row():
            gr.Button(
                value="되돌리기"
            ).click(fn=counselling_bot_undo, inputs=cb_chatbot, outputs=cb_chatbot)
            gr.Button(
                value="초기화"
            ).click(fn=counselling_bot_reset, inputs=cb_chatbot, outputs=cb_chatbot)
        pass
    with gr.Tab("번역봇"):
        # 1
        gr.Markdown(
            value="""
# <center>번역봇</center>
<center>헤이마트 번역봇입니다. 다양한 언어로 번역해드립니다.</center>
            """
        )
        with gr.Row():
            # 2
            tb_output_conditions = gr.Text(
                label="번역 조건",
                placeholder="예시 : 자연스럽게",
                lines=1,
                max_lines=3
            )
            # 3
            tb_output_language = gr.Dropdown(
                label="출력 언어",
                choices=["한국어", "영어", "일본어", "중국어"],
                value="한국어",
                allow_custom_value=True,
                interactive=True
            )
        # 4
        tb_submit = gr.Button(
            value="번역하기",
            variant="primary"
        )
        with gr.Row():
            # 5
            tb_input_text = gr.Text(
                placeholder="번역할 내용을 적어 주세요.",
                lines=10,
                max_lines=20,
                show_copy_button=True,
                show_label=False,
            )
            # 6
            tb_output_text = gr.Text(
                lines=10,
                max_lines=20,
                show_copy_button=True,
                label="",
                interactive=False
            )
        pass
        with gr.Row():
            tb_TXTupload = gr.File(label="TXT 파일 업로드")
            tb_PDFupload = gr.File(label="PDF 파일 업로드")

        # 번역봇 내용 보내기
        tb_submit.click(fn=translate_bot, inputs=[tb_output_conditions, tb_output_language, tb_input_text],
                        outputs=[tb_output_text])
        # Text 파일 업로드
        tb_TXTupload.upload(fn=translate_bot_text_upload, inputs=tb_TXTupload, outputs=tb_input_text)

        # PDF 파일 업로드
        tb_PDFupload.upload(fn=translate_bot_pdf_upload, inputs=tb_PDFupload, outputs=tb_input_text)
    with gr.Tab("소설봇"):
        # 1
        gr.Markdown(
            value="""
# <center>소설봇</center>
<center>소설을 생성해 주는 봇입니다.</center>
            """
        )
        with gr.Accordion(label="사용자 설정"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 2
                    nb_model = gr.Dropdown(
                        label="모델 선택",
                        choices=["gpt-3.5-turbo", "gpt-4-turbo"],
                        value="gpt-3.5-turbo",
                        interactive=True
                    )
                    # 3
                    nb_temperature = gr.Slider(
                        label="온도",
                        info="숫자가 높을 수록 창의적",
                        minimum=0,
                        maximum=2,
                        step=0.1,
                        value=1,
                        interactive=True
                    )
                # 4
                nb_detail = gr.Text(
                    container=False,
                    placeholder="소설의 세부적인 설정을 작성합니다.",
                    lines=8,
                    scale=4
                )
        # 5
        nb_submit = gr.Button(
            value="생성하기",
            variant="primary"
        )
        # 6
        nb_output = gr.Text(
            label="",
            placeholder="이곳에 소설의 내용이 출력됩니다.",
            lines=10,
            max_lines=200,
            show_copy_button=True,
        )
        # 소설봇 내용 보내기
        nb_submit.click(fn=novel_bot, inputs=[nb_model, nb_temperature, nb_detail], outputs=[nb_output])
        pass
    pass

app.launch()
