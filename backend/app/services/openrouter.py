import os
import requests
from typing import Dict, List, Optional

class OpenRouterClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-site-url.com",  # Required by OpenRouter
            "X-Title": "Multimodal RAG App"  # Optional but recommended
        }
    
    def chat_completion(self, messages: List[Dict], model: str = "anthropic/claude-3-sonnet", **kwargs):
        """Get chat completion from OpenRouter"""
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")