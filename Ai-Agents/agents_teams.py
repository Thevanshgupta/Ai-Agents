from phi.agent import Agent
# from phi.model.openai import OpenAIChat
# from openai import OpenAI
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from phi.model.groq import Groq
load_dotenv()

web_agent=Agent(
    name="web-agent",
    # model=OpenAIChat(id="gpt-4o-mini"),
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    # model=OpenAI(id="gpt-4o-mini"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,)

finance_agent=Agent(
    name="finance-agent",   
    # model=OpenAIChat(id="gpt-4o-mini"),
    # model=OpenAI(id="gpt-4o-mini"),
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=  [YFinanceTools(stock_price=True,analyst_recommendations=True,company_info=True)],
    instructions=["Use tables to display thr data."],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Use the web-agent to find information and the finance-agent to analyze it.","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("summarize and compare analyst recommendations and fundamentals for TSLA and NVDA", stream=True)