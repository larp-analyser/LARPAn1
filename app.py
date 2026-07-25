import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spaces

from app.api.routes import router
from app.db.mongo import MongoDB

# --- 1. The Stealth Injection (Monkeypatch) ---
original_init = FastAPI.__init__

def custom_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Inject routes and database connections exactly once
    if not getattr(self, "_an1_injected", False):
        self.include_router(router)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.add_event_handler("startup", MongoDB.connect)
        self.add_event_handler("shutdown", MongoDB.disconnect)
        self._an1_injected = True

# Overwrite FastAPI's core initialization before Gradio builds its server
FastAPI.__init__ = custom_init


# --- 2. The ZeroGPU Decoy ---
@spaces.GPU
def dummy_inference(text):
    return "Engine Status: Online."

# Name this 'demo'. DO NOT expose any variable named 'app'
with gr.Blocks() as demo:
    gr.Markdown("# AN1 Engine Core")
    btn = gr.Button("Ping Diagnostics")
    btn.click(fn=dummy_inference, inputs=gr.Textbox(), outputs=gr.Textbox())
