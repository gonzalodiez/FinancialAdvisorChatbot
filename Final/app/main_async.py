import chainlit as cl
from agent_async import acreate_agent

# Code to run chainlit: chainlit run chat_interface2.py -w

@cl.on_chat_start
def start_chat():
    agent=acreate_agent()
    cl.user_session.set("agent", agent)
    '''cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )'''


@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    answer=await agent.run(message)
    #message_history = cl.user_session.get("message_history")
    #message_history.append({"role": "user", "content": message})
    
    #answer=openai.ChatCompletion.create(
    #    engine=model_name,
    #    messages=message_history 
    #)
    #msg = cl.Message(content="")
    #msg.stream_token(answer.choices[0].message.content)
     
    
    #message_history.append({"role": "assistant", "content": msg.content})
    await cl.Message(content=answer).send()
    
    