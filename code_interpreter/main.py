from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.python import PythonREPL
from langchain.agents import load_tools, initialize_agent

load_dotenv()


def main():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    tool = PythonREPLTool()
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    python_agent_executor = create_python_agent(
        llm=llm, tool=tool, agent_type=agent_type, verbose=True
    )
    python_agent_executor.invoke(
        {
            "input": "In current working directory, generate and save 15 QR codes that point to https://rutambhagat.carrd.co"
        }
    )


if __name__ == "__main__":
    main()
