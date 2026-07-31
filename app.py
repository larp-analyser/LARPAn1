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
    if not getattr(self, "_an1_injected", False):
        self.include_router(router)
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize DB safely inside the worker process
        @self.on_event("startup")
        def startup_db():
            MongoDB.connect()
            
        self._an1_injected = True

# Overwrite FastAPI's initialization before Gradio constructs its internal App
FastAPI.__init__ = custom_init


@spaces.GPU
def dummy_inference():
    pass

with gr.Blocks() as demo:
    gr.Markdown("# LARPAn1")
    
    hidden_btn = gr.Button("Hidden", visible=False)
    hidden_btn.click(fn=dummy_inference, inputs=None, outputs=None)

demo.launch()
