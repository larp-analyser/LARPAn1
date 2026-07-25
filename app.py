import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db.mongo import MongoDB
from contextlib import asynccontextmanager
import spaces # Required for ZeroGPU environments

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing AN1 ...")
    MongoDB.connect()
    yield
    print("Shutting down gracefully...")
    MongoDB.disconnect()

# The variable MUST be named 'app' so Hugging Face's internal server can find it
app = FastAPI(title="AN1 Neural Core", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

# --- The Fix: Add the ZeroGPU Decorator ---
@spaces.GPU
def dummy_inference(text):
    return "Engine Status: Online."

with gr.Blocks() as iface:
    gr.Markdown("# AN1 Engine Core")
    btn = gr.Button("Ping Diagnostics")
    btn.click(fn=dummy_inference, inputs=gr.Textbox(), outputs=gr.Textbox())

# Mount Gradio at a subpath so FastAPI's routes remain intact
app = gr.mount_gradio_app(app, iface, path="/ui")

# 🛑 DO NOT add uvicorn.run() or iface.launch() here.
# Hugging Face's SDK will automatically boot the 'app' variable on port 7860.
