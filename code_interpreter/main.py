from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType
from langchain.python import PythonREPL
from langchain.agents import load_tools, initialize_agent
import os

load_dotenv()


def main():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    tool = PythonREPLTool()
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION

    python_agent_executor = create_python_agent(
        llm=llm, tool=tool, agent_type=agent_type, verbose=True
    )

    # python_agent_executor.invoke(
    #     {
    #         "input": "In current working directory, generate and save 15 QR codes that point to https://rutambhagat.carrd.co"
    #     }
    # )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, "episode_info.csv")

    print("CSV file path: ", csv_file_path)
    csv_agent = create_csv_agent(
        llm=llm,
        path=csv_file_path,
        agent_type=agent_type,
        verbose=True,
    )

    # csv_agent.invoke(
    #     {"input": "How many columns are there in the episode-info.csv file?"}
    # )

    # csv_agent.invoke(
    #     {
    #         "input": "Which write wrote the least episodes? And how many episodes did he write? there can be multiple writers for the same episode, make sure you properly split them before counting"
    #     }
    # )

    # csv_agent.invoke({"input": "Which season has the most episodes?"})

    csv_agent.invoke(
        {
            "input": "Print seasons in ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main()
