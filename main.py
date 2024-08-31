# from fastapi import FastAPI
# from routes.router import router

# app = FastAPI()

# app.include_router(router, prefix="/predict")

# if _name_ == "_main_":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.router import router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, specify allowed origins.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
    expose_headers=["X-Predictions"]  # Explicitly expose X-Predictions header
)

app.include_router(router, prefix="/predict")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
