import httpx
from typing import Optional, Dict, Any


class MCPClient:
    def __init__(self, host: str, auth_token: str):
        self.base_url = f"{host}/mcp"
        self.headers = {"X-API-Key": auth_token}

    async def execute_tool(self, tool_name: str, params: dict) -> Dict[str, Any]:
        """Updated to handle progress context"""
        async with httpx.AsyncClient() as client:
            try:
                # Add mock progress context parameter
                params["ctx"] = {"task_id": "123"}  # Example task ID

                response = await client.post(f"{self.base_url}/tools/{tool_name}", json=params, headers=self.headers,
                    timeout=30)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP error: {e.response.text}"}
            except Exception as e:
                return {"error": str(e)}