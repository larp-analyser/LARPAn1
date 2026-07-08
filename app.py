import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db.mongo import MongoDB
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing PSI-09 Core Engine v2...")
    MongoDB.connect()
    yield
    print("Shutting down gracefully...")
    MongoDB.disconnect()

fastapi_app = FastAPI(title="PSI-09 Core Engine", lifespan=lifespan)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app.include_router(router)

def engine_status():
    return "LARPAn1 Engine is active. API is running perfectly."

# Use a minimalist blocks setup to avoid Gradio 6 transpiler/SSR hooks doing weird things
with gr.Blocks() as demo:
    gr.Markdown("# LARPAn1 Monitor\nThe backend API is actively running.")
    btn = gr.Button("Check Status")
    out = gr.Textbox()
    btn.click(fn=engine_status, inputs=[], outputs=out)

# Mount the Gradio UI at the root path, while keeping the API endpoints intact
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
