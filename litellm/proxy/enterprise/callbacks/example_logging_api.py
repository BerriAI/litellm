# # this is an example endpoint to receive data from litellm
# from fastapi import FastAPI, HTTPException, Request

# app = FastAPI()


# @app.post("/log-event")
# async def log_event(request: Request):
#     try:
#         # Assuming the incoming request has JSON data
#         data = await request.json()
#         # Your additional logic can go here
#         # For now, just printing the received data

#         return {"message": "Request received successfully"}
#     except Exception as e:
#         import traceback

#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Internal Server Error")


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)
