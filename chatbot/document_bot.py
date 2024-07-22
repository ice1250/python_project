import gradio as gr
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_loader(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pdf_doc = loader.load()
    return pdf_doc


def pdf_bot_chatbot(pdf_path):
    pages_content = pdf_loader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000,
        chunk_overlap=0,
    )
    # pages_content 내용 분할
    split_docs = text_splitter.split_documents(pages_content)

    # 분할된 문서의 수
    print("분할된 문서의 수 : ", len(split_docs))

    # map template 설정, {pages_content} 분할된 내용이 입력
    map_template = """ 다음은 문서 중 일부 내용입니다.
    {pages_content}
    이 문서의 주요 내용을 요약해 주세요.
    """

    # map 기본 프롬프트
    map_prompt = PromptTemplate.from_template(map_template)

    # 문서 내용이 길 수 있기 때문에 model을 pgt-3.5-turbo-16k 설정
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # reduce 단계에서 처리할 프롬프트 정의
    reduce_template = """ 다음은 문서 요약의 집합입니다.
    {summaries}
    이 내용을 바탕으로 통합된 문서 요약을 작성해 주세요.
    """

    # Reduce 프롬프트
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # reduce에서 수행할 LLMChain 정의
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains import ReduceDocumentsChain

    # 문서 목록 통합 체인 설정
    combine_doc_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="summaries",  # reduce 프롬프트에 대입되는 변수
    )

    # 분할된 문서 순차적으로 Reduce 처리
    reduce_doc_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_doc_chain,
        collapse_documents_chain=combine_doc_chain,
        token_max=4000,  # 토큰 최대 개수 설정
    )

    # 최종 체인 연결
    final_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,  # 각 문서 맵핑
        reduce_documents_chain=reduce_doc_chain,
        document_variable_name="pages_content",
        return_intermediate_steps=False,
    )

    # 최종 결과 실행
    result_summary = final_chain(split_docs)
    # 요약 결과 출력
    return result_summary["output_text"]


def create_document_bot_tab():
    with gr.Tab("PDF 문서 요약봇"):
        # 1
        gr.Markdown(
            value="""
                    # <center>문서 요약봇</center>
                    <center>AI 인공지능 비서 HeyMate입니다. PDF를 업로드하면 내용을 번역해 줄 수 있습니다.</center>
                    """
        )
        pdf_input = gr.File()
        summary_btn = gr.Button(value="문서 요약하기")
        summary = gr.Textbox(
            label="PDF 요약",
            lines=8,
            placeholder="PDF 요약 내용입니다.",
            scale=8
        )
        summary_btn.click(fn=pdf_bot_chatbot, inputs=[pdf_input], outputs=[summary])
