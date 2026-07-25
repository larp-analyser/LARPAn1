import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.db.mongo import MongoDB
from contextlib import asynccontextmanager
import spaces

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing AN1 ...")
    MongoDB.connect()
    yield
    print("Shutting down gracefully...")
    MongoDB.disconnect()

app = FastAPI(title="AN1 Neural Core", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

def dummy_inference(text):
    return "Status: Online."

with gr.Blocks() as iface:
    gr.Markdown("# AN1 ")
    btn = gr.Button("Ping Diagnostics")
    btn.click(fn=dummy_inference, inputs=gr.Textbox(), outputs=gr.Textbox())

app = gr.mount_gradio_app(app, iface, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
