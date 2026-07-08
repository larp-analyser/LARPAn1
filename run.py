import uvicorn
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

app = FastAPI(title="PSI-09 Core Engine", lifespan=lifespan)

# Add CORS for any web-based bridges
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    # Start the server
    uvicorn.run("run:app", host="0.0.0.0", port=7860, reload=True)
