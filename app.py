import os
os.environ["OPENAI_MAX_RETRIES"] = "0"
os.environ["LITELLM_NUM_RETRIES"] = "0"
os.environ["LITELLM_MAX_RETRIES"] = "0"

from contextlib import asynccontextmanager
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import spaces
import logging

from app.api.routes import router
from app.db.mongo import MongoDB

# --- CONFIGURE GLOBAL LOGGING HERE ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.info("LARPAn1 Neural Core Booting Sequence Initiated...")
# --------------------------------------

original_init = FastAPI.__init__

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    logger.info("Lifespan Context: Establishing MongoDB Connection...")
    MongoDB.connect()
    yield
    logger.info("Lifespan Context: Tearing down connections...")
    MongoDB.disconnect()

def custom_init(self, *args, **kwargs):
    if "lifespan" not in kwargs:
        kwargs["lifespan"] = app_lifespan
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
    pass

with gr.Blocks() as demo:
    gr.Markdown("# LARPAn1")
    
    hidden_btn = gr.Button("Hidden", visible=False)
    hidden_btn.click(fn=dummy_inference, inputs=None, outputs=None)

demo.launch()
