import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import llm_timeout, mcp_timeout
from errors import MalformedToolResponse


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Support both environment-variable names.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """
    Preserve the current Windows/Conda environment and add MCP API keys.
    """

    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
    timeout=llm_timeout(),  # ChatGroq has no request timeout by default
)


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },

        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },

        "weather": {
            "transport": "stdio",

            # Run the server with the same Python that runs this app.
            "command": sys.executable,

            # The weather MCP server script in this project folder.
            "args": [
                str(WEATHER_SERVER_PATH),
            ],

            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one tool from one MCP server.

    This prevents a broken weather or AviationStack server from
    crashing an unrelated Tavily request.
    """

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the travel environment, and run "
                "`uvx --version`."
            )

    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )

        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    # Important: load only the requested MCP server.
    tools = await asyncio.wait_for(
        client.get_tools(
            server_name=server_name,
        ),
        timeout=mcp_timeout(),
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    # By default, langchain-mcp-adapters returns an MCP error as normal text
    # ("Error executing tool ..."), so agents would treat failures as real data.
    # Raising an exception instead lets each agent's fallback handle it.
    tool.handle_tool_error = False

    return tool


async def _invoke_tool(server_name: str, tool_name: str, tool_args: dict[str, Any]):
    """Load one MCP tool and call it, with a timeout on each step."""

    tool = await _get_server_tool(server_name, tool_name)

    try:
        return await asyncio.wait_for(
            tool.ainvoke(tool_args),
            timeout=mcp_timeout(),
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"MCP tool '{server_name}/{tool_name}' timed out "
            f"after {mcp_timeout():.0f}s"
        ) from exc


# =========================================================
# MCP result helpers
# =========================================================

def mcp_result_text(result: Any) -> str:
    """
    Flatten an MCP tool result to plain text.

    langchain-mcp-adapters returns a list of content blocks such as
    [{"type": "text", "text": "..."}]; some tools may return a plain string.
    """

    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        parts = []

        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)

        return "\n".join(part for part in parts if part).strip()

    return str(result)


def mcp_result_json(result: Any) -> Any:
    """Parse an MCP tool result whose text is JSON (e.g. the weather tools)."""

    text = mcp_result_text(result)

    if not text:
        raise MalformedToolResponse("The tool returned an empty response.")

    try:
        return json.loads(text)
    except ValueError as exc:
        raise MalformedToolResponse(
            "The tool response was not valid JSON."
        ) from exc


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """
    Test every MCP server independently.

    One failed server will not stop the remaining tests.
    """

    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )

            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )

            print(
                f"{server_name}: OK -> {tool_names}"
            )

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    return await _invoke_tool(
        "tavily",
        "tavily_search",
        {
            "query": query,
        },
    )


# =========================================================
# AviationStack MCP
# =========================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    return await _invoke_tool(
        "aviationstack",
        tool_name,
        tool_args or {},
    )


# =========================================================
# Weather MCP
# =========================================================

async def weather_mcp_search(city: str):
    return await _invoke_tool(
        "weather",
        "get_current_weather",
        {
            "city": city,
        },
    )


async def forecast_mcp_search(city: str):
    return await _invoke_tool(
        "weather",
        "get_forecast",
        {
            "city": city,
        },
    )


# =========================================================
# Destination extractor
# =========================================================

def extract_destination(query: str) -> str:
    prompt = f"""
Extract only the destination city or country from the travel request.

Travel request:
{query}

Return only the destination name.
Do not add any explanation.
"""

    response = llm.invoke(prompt)

    # The model is asked for just the city name. Keep only the first line and
    # remove quotes and punctuation, so extra text never reaches the weather
    # API or the cache key.
    lines = str(response.content).strip().splitlines()
    destination = (lines[0] if lines else "").strip().strip("\"'`.,;: ")[:80]

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination