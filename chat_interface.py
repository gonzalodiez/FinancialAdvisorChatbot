import os
import chainlit as cl
from vertexai.preview.language_models import TextGenerationModel
from utils.chroma import chroma
from chromadb.utils import embedding_functions

# Code to run chainlit: chainlit run chat_interface.py -w

#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='$HOME/.config/gcloud/application_default_credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='application_default_credentials.json'
os.environ['GOOGLE_CLOUD_PROJECT'] ='starlit-factor-393301'

# Contexto para testear
'''
context = """
The turn of the 20th century marked a pivotal moment for science fiction. 
The rapid development of science and technology inspired a new wave of writers and visionaries.
 One of the genre's pioneers was H.G. Wells, whose iconic novels like "The War of the Worlds" (1898) 
 and "The Time Machine" (1895) captured the public's imagination and set the stage for what was to come. 
 Science fiction magazines, like Amazing Stories (1926), 
 emerged, providing a platform for aspiring authors to share their speculative tales with a growing audience.
"""
'''

# Levanto el llm
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

# Levanto Chroma
path='/home/gonza/gonza/Financial_Chatbot_Advisor/git/FinancialAdvisorChatbot/langchain/chroma'
chroma = chroma(path=path)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")
chroma.collection(collection='chroma',embedding_function=sentence_transformer_ef)

#Version sin contexto
'''
@cl.on_message
async def main(user_message: str):
    # Your custom logic goes here...
 
    prompt = f"""Q: {user_message}\n
            A:
         """
    
    answer=generation_model.predict(
        prompt,
        max_output_tokens=100,
        temperature=0.5,
    ).text
    # Send a response back to the user
    await cl.Message(
        content=f"{answer}",
    ).send()
'''

#Version con contexto:
@cl.on_message
async def main(message: str):
    
    n=3
    user_question=message
    docs=chroma.retriever(query=user_question, n_results=n)

    #docs['ids']
    #docs['metadatas']
    context=docs['documents'][0]
    #docs['distances']

    prompt = f"""Answer the question given in the context below and explain why:
    Context: {context}\n 
    Question: {user_question} \n
    Answer:
    """

    answer=generation_model.predict(
        prompt, temperature=0.99
    ).text

    # Send a response back to the user
    await cl.Message(
        content=f"{answer}",
    ).send()


