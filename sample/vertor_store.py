from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def call_vector_store_1():
    openai_embedding = OpenAIEmbeddings()

    db = Chroma.from_texts(
        texts=[
            "안녕",
            "안녕하세요",
            "반갑습니다.",
            "반가워요",
        ],
        embedding=openai_embedding,
    )

    similar_texts = db.similarity_search("안녕")
    print(similar_texts)
