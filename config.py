import os


# LLM API config (OpenAI-compatible endpoint)
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "temperature": 0.7,
    "max_tokens": 8192,
    "extra_body": {"thinking": {"type": "disabled"}},
}

# Mem0 API config (Long-term memory)
MEM0_CONFIG = {
    "api_key": os.getenv("MEM0_API_KEY", ""),
}
