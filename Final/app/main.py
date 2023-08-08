import chainlit as cl
from agent import create_agent

# Code to run chainlit: chainlit run main.py -w

@cl.on_chat_start
def start_chat():
    agent=create_agent()
    cl.user_session.set("agent", agent)



@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    answer=agent.run(message)
    await cl.Message(content=answer).send()
    
    