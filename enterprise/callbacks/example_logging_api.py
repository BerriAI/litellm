# this is an example endpoint to receive data from litellm
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()


@app.post("/log-event")
async def log_event(request: Request):
    try:
        print("Received /log-event request")
        # Assuming the incoming request has JSON data
        data = await request.json()
        print("Received request data:")
        print(data)

        # Your additional logic can go here
        # For now, just printing the received data

        return {"message": "Request received successfully"}
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
