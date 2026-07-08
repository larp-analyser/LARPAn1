import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db.mongo import MongoDB
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing PSI-09 Core Engine v2...")
    MongoDB.connect()
    yield
    # Shutdown
    print("Shutting down gracefully...")
    MongoDB.disconnect()

fastapi_app = FastAPI(title="PSI-09 Core Engine", lifespan=lifespan)

# Add CORS for any web-based bridges
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our core engine routes (e.g. POST /psi09)
fastapi_app.include_router(router)

# Create a Dummy Gradio Interface to bypass the Hugging Face Docker requirement
def engine_status():
    return "LARPAn1 Engine is actively running in the background. The core API is fully accessible at /psi09"

demo = gr.Interface(
    fn=engine_status,
    inputs=[],
    outputs="text",
    title="LARPAn1 Monitor",
    description="This UI exists to satisfy the Hugging Face Space free tier. The true agentic engine runs via API."
)

# Mount the Gradio UI at the root path, while keeping the API endpoints intact
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860)
