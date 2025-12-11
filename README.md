# Token Optimiser v3
Multi-source context engine integrating Redis memory (MCP) + RAG + Token Gating.

# Pre-Setup Installations
1. Install LM Studio
   https://lmstudio.ai/download
2. Install Docker
   https://www.docker.com
3. Set up Redis
   https://redis.io

## Setup
1. Run Redis locally
   docker run -d --name redis -p 6379:6379 redis:latest
2. Start Redis
   docker start redis
3. Install requirements
   pip install -r requirements.txt
4. Add your OpenAI key:
   export OPENAI_API_KEY="sk-..."

## Running the Frontend
1. Make sure you have completed the setup steps above.
2. Run the Streamlit application:
   streamlit run Frontend/frontend.py
