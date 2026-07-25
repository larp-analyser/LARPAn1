import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spaces

from app.api.routes import router
from app.db.mongo import MongoDB

# Connects Mongo and starts the keepalive thread immediately when the app imports
MongoDB.connect()

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


@spaces.GPU
def dummy_inference():
    return "Engine Status: Online."

# Expose 'demo' so Hugging Face handles port binding automatically
with gr.Blocks() as demo:
    gr.Markdown("# LARPAn1")
    
    btn = gr.Button("Ping Diagnostics")
    status_output = gr.Textbox(label="System Status")
    btn.click(fn=dummy_inference, inputs=None, outputs=status_output)

demo.launch()
