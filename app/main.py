# Importing chainlit library
import chainlit as cl

# Importing the function to create an agent from a custom module
from agent import create_agent

# Importing the make_async function from the chainlit library
from chainlit import make_async

# Command to run chainlit application
# Code to run chainlit: chainlit run main.py -w

# Handler function that runs when the chat starts
@cl.on_chat_start
def start_chat():
    # Creating an agent using the custom function
    agent = create_agent()
    # Storing the agent in the user session for later retrieval
    cl.user_session.set("agent", agent)

# Asynchronous handler function that runs when a message is received
@cl.on_message
async def main(message: str):
    # Retrieving the agent stored in the user session
    agent = cl.user_session.get("agent")

    # Getting the run method from the agent object
    chat = agent.run

    # Creating an asynchronous version of the chat function
    async_function = make_async(chat)

    # Awaiting the asynchronous function with the provided message and a special callback
    answer = await async_function(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Awaiting the sending of the answer as a chat message
    await cl.Message(content=answer).send()
