from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools
load_dotenv()

agent=Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, 
                         analyst_recommendations=True
                         )],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display thr data."],
    debug_mode=True,
)

agent.print_response("summarize and compare analyst recommendations and fundamentals for TSLA and NVDA")