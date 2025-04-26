import os
import traceback
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP
from run_agents import run_browser_agent as _run_browser_agent
from typing import Any
import argparse
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from mcp.server.sse import SseServerTransport
from starlette.routing import Route, Mount

mcp = FastMCP("browser_use")
# 这里也要设置以下，防止browser API 报错
os.environ["OPENAI_API_KEY"] = ""

# 需要启动chrome并监听端口 "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 -headless

@mcp.tool()
async def run_browser_agent(task: str) -> str:
    """
    Runs a browser agent task synchronously and waits for the result.
    :param task: 用户输入
    :return: 查询结果
    """
    try:
        (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
        ) = await _run_browser_agent(
            agent_type=os.getenv("MCP_AGENT_TYPE", "org"),
            llm_model_name=os.getenv("MCP_MODEL_NAME", "deepseek-chat"),
            llm_base_url=os.getenv("MCP_BASE_URL", "https://api.deepseek.com"),
            llm_api_key=os.getenv("MCP_API_KEY", ""),
            use_own_browser=True,
            keep_browser_open=True,
            headless=False,
            disable_security=True,
            window_w=1280,
            window_h=720,
            save_recoding_path="./recording",
            save_agent_history_path="./tmp/agent_history",
            save_trace_path="./tmp/trace",
            enable_recording=False,
            task=task,
            # add_infos=add_infos,
            max_steps=100,
            use_vision=False,
            max_actions_per_step=5,
            tool_calling_method="auto",
            chrome_cdp=os.getenv("CHROME_CDP", "http://localhost:9222"),
            max_input_tokens=128000
        )

        if any(error is not None for error in errors):
            return f"Task failed: {errors}\n\nResult: {final_result}"
        else:
            return final_result
    except Exception as e:
        return f"Error during task execution: {str(e)}"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )


if __name__ == '__main__':
    mcp_server = mcp._mcp_server
    parser = argparse.ArgumentParser(description='启动 MCP 服务')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to listen on')
    args = parser.parse_args()


    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)