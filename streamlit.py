# api/streamlit.py
from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
async def run_streamlit():
    # Run the Streamlit app
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8000", "--server.headless", "true"])
    return {"message": "Streamlit app started"}
