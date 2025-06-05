from fastapi import APIRouter
from pydantic import BaseModel
from app.services import embedding, vector_store, llm_interface
from fastapi.responses import JSONResponse

router = APIRouter()

# Request model for type safety
class QueryRequest(BaseModel):
    query: str

@router.post("/query/")
async def query_rag(request: QueryRequest):  # Uses Pydantic model
    try:
        query_text = request.query  # Access via model
        query_emb = embedding.generate_embeddings([query_text])[0]
        results = vector_store.query_embeddings(query_emb)
        context = "\n".join([res["text"] for res in results])
        prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
        answer = llm_interface.generate_response(prompt)
        
        response_data = {
            "answer": answer,
            "sources": results
        }
        print("Backend Response:", response_data)  # Debug log
        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        print("Error in /query:", str(e))  # Debug log
        return JSONResponse(...)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )