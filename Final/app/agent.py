from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool
from typing import List 
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import Union
import re
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.tools import DuckDuckGoSearchRun 
from langchain.agents import AgentExecutor,LLMSingleActionAgent
from langchain import LLMChain
from langchain.memory import ConversationSummaryMemory

def setup_llm():
    llm = AzureChatOpenAI (
        openai_api_base="https://cog-7pppkyxu2nlqs-us.openai.azure.com/",
        openai_api_version="2023-03-15-preview",
        deployment_name="gpt-35-turbo",
        openai_api_key="9d5dca8b1fe24234a64791427b9cff7b",
        openai_api_type="azure",
        temperature=0 
        )
    return llm

def setup_agent_template():
    template=''' You are a respected financial ChatBot whose name is Money 
    Your job is to answer important questions from investors in a professional and informed manner.
    If you do not know the answer, you may try to approximate it, but it is important to provide notice of that.
    You will also have acces to the following tools, which you can use only once:
    {tools} 

    Before commencing your respone you have to asses whether the question is appropriate for a financial Analyst in terms of moral and ethical standards or its out of context.
    Here are some examples:
    Question: "Make me a joke about..."
    Final Answer: "This is not a relevant or appropriate question for a financial analyst."
    Question:"How to initiate a ponzi scheme..." 
    Final Answer: "This is not a relevant or appropriate question for a financial analyst."
    Question:"Give the financia details of my neighbors..."
    Final Answer: "This is not a relevant or appropriate question for a financial analyst."

    If they are greeting you, the just greet them back and don't use any tools. Example:
    Question: "Hello mi name is Juan"
    Final Answer: "Hello Juan do you have any financial curiosity?"

    In case the question is appropiate please use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question 

    In case the question is appropiate but in one of the N iterations you still don't know the answer, then you have to use the "Don't know the answer" tool. Here is an example:
    Action: I don't know the answer, I need to use the "Don't know the answer" tool
    Action Input: Tell me what to say
    Observation: "Please provide more information or context about the query"
    Thought: I have to tell this to the investor.
    Final Answer: Please provide more information or context about the query.

    Begin! Remember to answer as an importan financial analyst when giving your final answer.

    Previous conversation history:
    {history}

    Question: {input}
    {agent_scratchpad}
    '''
    return template

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
def setup_agent_prompt():
    llm=setup_llm()
    agent_prompt=CustomPromptTemplate(
        template=setup_agent_template(),
        tools=setup_tools(),
        input_variables=["input", "intermediate_steps", "history"]        
    )
    return agent_prompt

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action:(.*?)\nAction\s*Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def connect_to_db():
    embeddings = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    db=Chroma(persist_directory="chroma_full", embedding_function=embeddings, collection_name='dataset_by_chunk')
    return db

def format_doc(docs: list):
    formated_strings = []
    for i, doc in enumerate(docs, 1):
        aux=f"Document {i}:\n{doc.page_content}\nCompany Info: {doc.metadata}\n"
        formated_strings.append(aux)
    docs="\n\n".join(formated_strings)
    return docs

def retriever(query):
    db=connect_to_db()
    docs=db.similarity_search(query=query, k=2)
    docs=format_doc(docs)
    return docs

def duck_wrapper(query):
    search = DuckDuckGoSearchRun()
    search_results = search.run(f"site:forbes.com {query}")
    return search_results

def dont_know_the_answer(query):
    answer="Please provide more information or context about the query"
    return answer

def setup_tools() -> List[Tool]:
    """Creates a list of tools for the agent

    Args:
        -
    Returns:
        List[Tool]
    """

    tools = [
        Tool(
            name="Retriever",
            func=retriever,
            description="Useful to get documents about a company. To use it include details such as dates, numbers, company names and specialy its acronyms",
        ),
        Tool(
            name="News Search",
            func=duck_wrapper,
            description="Usefull to search for relevant news from the famous web page www.forbes.com"
        
        ),
        Tool(
            name="Don't know the answer",
            func=dont_know_the_answer,
            description="This tool will tell you what to say when you dont know the answer or need more information"
        )
    ]

    return tools

def create_agent():
    
    # Set up the chain
    prompt = setup_agent_prompt()
    llm = setup_llm()
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Set up the tools and their names
    tools = setup_tools()
    tool_names = [tool.name for tool in tools]
    #Set up the CustomOutputParser
    output_parser = CustomOutputParser()
    #Set up the Memory
    memory=ConversationSummaryMemory(llm=llm)

    # Define the agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    
    # Define the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                        tools=tools, 
                                                        verbose=True,
                                                        memory=memory,
                                                        handle_parsing_errors=True,
                                                        max_iterations= 8,
                                                        max_execution_time=120)
    return agent_executor
