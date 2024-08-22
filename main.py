import os

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain

from llm import CustomLLM

from utils import show_image

if __name__ == '__main__':
    file_path = './pdf/test_01.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # documents = []
    # for i in os.listdir('./pdf'):
    #     file_path = os.path.join('./pdf', i)
    #     loader = PyPDFLoader(file_path)
    #     documents.append(loader.load())
    # loader = DirectoryLoader('./pdf', glob="*.pdf", show_progress=True)
    # documents = loader.load()

    embedding_model_path = '/home/liquid/download/models--hkunlp--instructor-xl'
    embedding_model = HuggingFaceInstructEmbeddings(model_name=embedding_model_path)

    vectorstore = FAISS.from_documents(documents=documents,
                                       embedding=embedding_model)

    llm_model = CustomLLM()
    chain = ConversationalRetrievalChain.from_llm(llm=llm_model,
                                                  retriever=vectorstore.as_retriever(),
                                                  return_source_documents=True, )

    # query = "conclusion中说了什么？"
    query = "openai公司的组织架构？"
    chat_history = []
    result = chain({"question": query,
                    "chat_history": chat_history},
                   return_only_outputs=True)
    print(result)
    page = list(result["source_documents"][0])[1][1]["page"]
    img = show_image(file_path, page)
    img.save('test.jpg')