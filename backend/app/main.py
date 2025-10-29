from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Accent Detection API",
    version="0.1.0",
    description="A small API that accepts an audio file and returns a mock regional-accent detection result.",
)

# Allow all origins for now (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect-accent")
async def detect_accent():
    """Accept an uploaded audio file and return a mock accent detection response.

    You will later add the real detection logic here.
    """
    # Inspect filename and content-type for debugging / validation if you like

    return {"None"}
