import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spaces

from app.api.routes import router
from app.db.mongo import MongoDB

# --- 1. Eagerly Initialize Database ---
# Connects Mongo and starts the keepalive thread immediately when the app imports
MongoDB.connect()

# --- 2. Stealth Route Injection ---
original_init = FastAPI.__init__

def custom_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Inject routes and CORS middleware exactly once
    if not getattr(self, "_an1_injected", False):
        self.include_router(router)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._an1_injected = True

# Overwrite FastAPI's initialization before Gradio constructs its internal App
FastAPI.__init__ = custom_init


# --- 3. ZeroGPU Decoy ---
@spaces.GPU
def dummy_inference(text):
    return "Engine Status: Online."

# Expose 'demo' so Hugging Face handles port binding automatically
with gr.Blocks() as demo:
    gr.Markdown("# AN1 Engine Core")
    btn = gr.Button("Ping Diagnostics")
    btn.click(fn=dummy_inference, inputs=gr.Textbox(), outputs=gr.Textbox())
    
# Add this exact line to keep the server awake and bind to the ZeroGPU proxy
demo.launch()
