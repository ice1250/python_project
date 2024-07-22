"""
RAG는 외부 소스에서 검색하거나 가져온 정보를 LLM 모델에 입력하여 답변을 생성하는 방법입니다.
"""

from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


# 문서
def call_retrieval_1():
    loader = TextLoader("files/hey.txt")
    document = loader.load()
    print(document)


# CSV
def call_retrieval_2():
    csv_loader = CSVLoader("files/hey_csv.csv")
    document = csv_loader.load()
    print(document)


# PDF
def call_retrieval_3():
    pdf_loader = PyPDFLoader("files/hey_pdf.pdf")
    document = pdf_loader.load()
    print(document)


# Document Transformer
def call_retrieval_4():
    text_loader = TextLoader("files/hey.txt")
    document = text_loader.load()
    document_content = document[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )

    texts = text_splitter.create_documents([document_content])
    print(len(texts))
    # print(texts)


# Text Embedding Models
def call_retrieval_5():
    openai_embedding = OpenAIEmbeddings()

    embeddings = openai_embedding.embed_documents(
        texts=[
            "안녕하세요",
            "무엇을 도와드릴까요?",
            "어서오세요?",
            "도움이 필요해요",
        ]
    )

    print("임베딩 수", len(embeddings))
    print("임베딩 차원", len(embeddings[0]))
    print("임베딩 차원", len(embeddings[1]))
    print("embeddings[0]", embeddings[0])

    embeddings = openai_embedding.embed_query("안녕")
    print(embeddings[:5])

