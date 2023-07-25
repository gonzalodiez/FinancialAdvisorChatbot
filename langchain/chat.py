from langchain.memory import ConversationBufferMemory
from vertexai.preview.language_models import TextGenerationModel

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='application_default_credentials.json'
os.environ['GOOGLE_CLOUD_PROJECT'] ='starlit-factor-393301'


llm = TextGenerationModel.from_pretrained("text-bison@001")


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=db.as_retriever(),
    return_source_documents=False,
    memory=memory,
    chain_type="stuff"
)