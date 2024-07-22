import gradio as gr
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

system_ai_ko = "당신은 인공지능 비서야, pdf 읽기, 문서 요약하기, 일정 관리, 날씨, 최단경로 검색, 웹 검색 등 다양한 내용에 답변할 수 있어야 해"
system_setting = SystemMessagePromptTemplate.from_template(system_ai_ko)

# 프롬프트 설정
voice_bot_prompt = ChatPromptTemplate.from_messages([
    system_setting,
    MessagesPlaceholder(variable_name="HeyMate"),
    HumanMessagePromptTemplate.from_template("{master_user}")
])

voice_bot_memory = ConversationBufferWindowMemory(
    memory_key="HeyMate",
    ai_prefix="AI 비서 HeyMate",
    human_prefix="사용자",
    return_messages=True,
    k=10
)

chatgpt = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",
    max_tokens=2048
)

conversation = LLMChain(
    prompt=voice_bot_prompt,
    memory=voice_bot_memory,
    llm=chatgpt,
    verbose=True
)


def voice_bot_handle_audio(audio_record):
    from pydub import AudioSegment

    save_file_path = "../voice.wav"
    frame_rate = audio_record[0]
    audio_data = audio_record[1].tobytes()
    sample_width = audio_record[1].dtype.itemsize
    audio = AudioSegment(
        audio_data,
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=1
    )
    audio.export(save_file_path, format="wav")


def voice_bot_create_stt():
    from openai import OpenAI
    client = OpenAI()
    file_path = "../voice.wav"
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcript.text


def voice_bot_create_audio(text):
    from openai import OpenAI
    client = OpenAI()
    response = client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    speech_file_path = "../output.wav"
    # response.stream_to_file(speech_file_path)
    return speech_file_path


def voice_bot_chat(message, cb_user_input_audio, chat_history):
    if cb_user_input_audio:
        message = voice_bot_create_stt()
    ai_answer = conversation({"master_user": message})['text']
    chat_history.append([message, ai_answer])
    audio_file = voice_bot_create_audio(ai_answer)
    return "", audio_file, chat_history


def voice_bot_undo(chat_history):
    if len(chat_history) > 1:
        chat_history.pop()
    return chat_history


def voice_bot_reset(chat_history):
    global vocie_bot_memory
    chat_init = [[None, "안녕하세요, AI 인공지능 비서 HeyMate입니다. 무엇이든 시켜만 주세요."]]
    voice_bot_memory.clear()
    return "", chat_init


def create_voice_bot_tab():
    with gr.Tab("음성 인식봇"):
        with gr.Column():
            # 1
            gr.Markdown(
                value="""
            # <center>음성 인식봇</center>
            <center>AI 인공지능 비서 HeyMate입니다. 음성으로 묻거나, 문서 요약, 일정 관리를 할 수 있습니다.</center>
            """
            )
            # 2
            cb_chatbot = gr.Chatbot(
                value=[[None, "안녕하세요, AI 인공지능 비서 HeyMate입니다. 무엇이든 시켜만 주세요."]],
                show_label=False
            )
        with gr.Row():
            # 3
            cb_user_input = gr.Textbox(
                lines=1,
                placeholder="입력 창",
                container=False,
                scale=7
            )
            # 4
            cb_audio_record = gr.Audio(
                sources=["microphone"],
                format="wav",
                scale=1,
                min_width=200,
                label="음성을 입력해 주세요"
            )
            # 음성출력
            cb_audio_chatbot = gr.Audio(
                autoplay=True,
                visible=False
            )
            # 5
            cb_submit_btn = gr.Button(
                value="보내기",
                scale=1,
                variant="primary",
                icon="images/send.png"
            )
            cb_audio_record.stop_recording(
                fn=voice_bot_handle_audio,
                inputs=[cb_audio_record]
            )
            cb_submit_btn.click(
                fn=voice_bot_chat,
                inputs=[cb_user_input, cb_audio_record, cb_chatbot],
                outputs=[cb_user_input, cb_audio_chatbot, cb_chatbot]
            )
            cb_user_input.submit(
                fn=voice_bot_chat,
                inputs=[cb_user_input, cb_audio_record, cb_chatbot],
                outputs=[cb_user_input, cb_audio_chatbot, cb_chatbot]
            )
        with gr.Row():
            # 6
            gr.Button(value="되돌리기").click(
                fn=voice_bot_undo,
                inputs=[cb_chatbot],
                outputs=[cb_chatbot]
            )
            # 7
            gr.Button(value="초기화").click(
                fn=voice_bot_reset,
                inputs=[cb_chatbot],
                outputs=[cb_chatbot]
            )
