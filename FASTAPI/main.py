from fastapi import FastAPI

app = FastAPI()
# path
# POST, PUT, DELETE, GET
@app.get("/")
async def root():
    return {"message":"Hello World from FASTAPI"}

@app.get("/demo")
def demo_func():
    return {"message":"This is output from demo function"}

@app.post("/post_demo")
def demo_post():
    return {"message":"This is output from post demo function"}
# POST: to create data.
# GET: to read data.
# PUT: to update data.
# DELETE: to delete data.

# @app.post()
# @app.put()
# @app.delete()
