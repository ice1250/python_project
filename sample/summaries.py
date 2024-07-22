# 문서요약

def call_summaries_call_1():
    from langchain.chains import MapReduceDocumentsChain
    from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.chains.llm import LLMChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    def pdf_loader(pdf_path):
        loader = PyPDFLoader(pdf_path)
        pdf_doc = loader.load()
        return pdf_doc

    path = "files/hey_pdf_2.pdf"
    pages_content = pdf_loader(path)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000,
        chunk_overlap=0,
    )
    split_docs = text_splitter.split_documents(pages_content)

    # map 기본 프롬프트
    map_prompt = PromptTemplate.from_template(""" 다음은 문서 중 일부 내용입니다.
        {pages_content}
        이 문서의 주요 내용을 요약해 주세요.
        """)
    # 문서 내용이 길 수 있기 때문에 model을 pgt-3.5-turbo-16k 설정
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

    # map_chain = map_prompt | llm
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce 프롬프트
    reduce_prompt = PromptTemplate.from_template(""" 다음은 문서 요약의 집합입니다.
    {summaries}
    이 내용을 바탕으로 통합된 문서 요약을 작성해 주세요.
    """)

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_doc_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="summaries",
    )

    # 분할된 문서를 순차적으로 reduce 처리
    reduce_doc_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_doc_chain,
        collapse_documents_chain=combine_doc_chain,
        token_max=4000,
    )

    final_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_doc_chain,
        document_variable_name="pages_content",
        return_intermediate_steps=False,
    )

    result_summary = final_chain.run(split_docs)

    print(result_summary)
