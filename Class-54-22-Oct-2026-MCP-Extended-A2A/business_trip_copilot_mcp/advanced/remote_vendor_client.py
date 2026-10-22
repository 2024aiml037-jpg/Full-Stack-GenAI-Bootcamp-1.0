import asyncio
import os

import httpx2
from dotenv import load_dotenv
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

URL = os.environ["REMOTE_MCP_URL"]
TOKEN = os.environ.get("REMOTE_MCP_TOKEN", "").strip()


async def main():
    headers = {}

    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    async with httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(
            30.0,
            read=300.0,
        ),
    ) as http_client:

        transport = streamable_http_client(
            URL,
            http_client=http_client,
        )

        async with Client(transport) as client:
            print("Connected:", client.server_info)

            tools = await client.list_tools()

            print("\nVENDOR MCP TOOLS")
            for tool in tools.tools:
                print("-", tool.name)


if __name__ == "__main__":
    asyncio.run(main())
