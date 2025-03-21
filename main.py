"""
This module provides a FastAPI application that uses sequence classification
to rank documents based on their similarity to a given query.

The application accepts POST requests to the '/api/v1/rerank'
endpoint, which takes in a RequestData object containing the query and
a list of Document objects. It then constructs pairs of query and
document texts for scoring. The ranked documents with their
corresponding similarity scores are returned as a ResponseData object.

"""

import os
import logging
import time
import uuid
from uuid import UUID
from typing import List, Union, Optional, Dict, Any
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator  # Update import
import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove duplicate imports
# from typing import Optional
# from pydantic import Field

port = int(os.getenv("PORT", "8787"))
max_length=int(os.getenv("MAX_LENGTH", "512"))
model_name=os.getenv("MODEL", "BAAI/bge-reranker-v2-m3")
device=os.getenv("DEVICE")

logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')
logging.info("port: %d", port)
logging.info("max_length: %d", max_length)
logging.info("model: %s", model_name)
logging.info("device: %s", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()

app = FastAPI()
# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Document(BaseModel):
    """
    A model representing a document with an ID and text.

    Attributes:
        id Union[int, str, UUID]: The unique ID of the document.
        text (str): The text content of the document.
    """

    id: Union[int, str, UUID]
    text: str

class RequestData(BaseModel):
    model: Optional[str] = "reranker"
    query: Optional[str] = None
    input: Optional[str] = None
    documents: Optional[List[Document]] = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def extract_query_and_docs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """处理各种可能的输入格式"""
        if isinstance(values, dict):
            # 记录原始请求
            logging.info(f"Original request: {values}")

            # 处理query/input字段
            if not values.get('query') and values.get('input'):
                values['query'] = values['input']

            # 确保documents字段存在
            if not values.get('documents'):
                values['documents'] = []

        return values

    def construct_pairs(self):
        """构建查询-文档对"""
        query = self.query or self.input or ""
        if not self.documents:
            return []
        return [[query, doc.text] for doc in self.documents]

@app.post("/v1/rerank")
async def rerank_documents(request: RequestData):
    # Enhanced logging
    logging.info(f"Processed request: {json.dumps(request.dict(exclude_unset=True), default=str)}")

    # 如果没有文档，返回空结果
    if not request.documents:
        return {
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {"total_tokens": 0}
        }

    response = []
    pairs = request.construct_pairs()
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True,
                         return_tensors="pt", max_length=max_length).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # 生成带索引的结果
        sorted_docs = sorted(
            zip(request.documents, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        # 构建OpenAI兼容格式
        results = [
            {
                "index": idx,
                "document": {"id": str(doc.id), "text": doc.text},
                "score": score
            }
            for idx, (doc, score) in enumerate(sorted_docs)
        ]

    return {
        "object": "list",
        "data": results,
        "model": request.model,
        "usage": {  # ✅ 修正后的正确结构
            "total_tokens": (inputs.attention_mask.sum().item()
                             if inputs.attention_mask is not None
                             else inputs.input_ids.numel())
        }
    }

# Add this after the existing rerank_documents function

@app.post("/v1/completions")
async def completions_compat(
    request: Dict[str, Any] = Body(...)
):
    """Compatibility endpoint for Dify"""
    logging.info(f"Completions request: {json.dumps(request, default=str)}")

    # Extract query and documents from the request
    query = request.get("prompt", "") or request.get("query", "")
    documents = []

    # Try to parse documents from the request
    if "documents" in request:
        doc_list = request.get("documents", [])
        for i, doc in enumerate(doc_list):
            if isinstance(doc, dict) and "text" in doc:
                doc_id = doc.get("id", i)
                documents.append(Document(id=doc_id, text=doc["text"]))
            elif isinstance(doc, str):
                documents.append(Document(id=i, text=doc))

    # Create a RequestData object
    req_data = RequestData(
        model=request.get("model", "reranker"),
        query=query,
        documents=documents
    )

    # Process the request
    if not req_data.documents:
        # Return Dify-compatible empty response
        return {
            "id": "rerank-" + str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": req_data.model,
            "choices": [],
            "usage": {"total_tokens": 0}
        }

    # Fix empty query issue
    if not req_data.query:
        req_data.query = " "  # Use space instead of empty string

    try:
        pairs = req_data.construct_pairs()
        with torch.no_grad():
            # Fix BatchEncoding issue by ensuring device is not None
            device_to_use = device if device else "cpu"

            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length
            )
            # Move to device after tokenization
            inputs = {k: v.to(device_to_use) for k, v in inputs.items()}

            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

            # Normalize scores to 0-1 range
            normalized_scores = torch.sigmoid(scores).tolist()

            # Generate results with indices
            sorted_docs = sorted(
                zip(req_data.documents, normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )

            # Build Dify-compatible format
            choices = [
                {
                    "index": idx,
                    "text": doc.text,
                    "document": {"id": str(doc.id), "text": doc.text},
                    "score": score,
                    "relevance_score": score,
                    "finish_reason": "stop"
                }
                for idx, (doc, score) in enumerate(sorted_docs)
            ]

        # Return Dify-compatible response
        token_count = sum(len(inputs["input_ids"][i]) for i in range(len(inputs["input_ids"])))

        return {
            "id": "rerank-" + str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": req_data.model,
            "choices": choices,
            "usage": {
                "total_tokens": token_count
            },
            "results": [
                {
                    "index": idx,
                    "document": {"id": str(doc.id), "text": doc.text},
                    "score": score,
                    "relevance_score": score
                }
                for idx, (doc, score) in enumerate(sorted_docs)
            ]
        }
    except Exception as e:
        logging.error(f"Error in completions_compat: {str(e)}")
        return {
            "id": "rerank-" + str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": req_data.model,
            "choices": [],
            "error": str(e),
            "usage": {"total_tokens": 0}
        }

@app.post("/v1/completions/rerank")
async def v1_completions_rerank(request: Dict[str, Any] = Body(...)):
    """Additional endpoint for Dify compatibility"""
    logging.info(f"v1_completions_rerank request: {json.dumps(request, default=str)}")
    return await completions_compat(request)

@app.post("/completions/rerank")
async def completions_rerank(request: Dict[str, Any] = Body(...)):
    """Additional endpoint for Dify compatibility"""
    logging.info(f"completions_rerank request: {json.dumps(request, default=str)}")
    return await completions_compat(request)

@app.post("/rerank")
async def simple_rerank(request: Dict[str, Any] = Body(...)):
    """Simple rerank endpoint for Dify compatibility"""
    logging.info(f"simple_rerank request: {json.dumps(request, default=str)}")
    return await completions_compat(request)

# Add a raw endpoint to capture exactly what Dify is sending
@app.post("/v1/raw_rerank")
async def raw_rerank(request: Request):
    """Debug endpoint to log raw request body"""
    try:
        body = await request.json()
        logging.info(f"Raw request body: {json.dumps(body, default=str)}")
        return {"status": "logged"}
    except Exception as e:
        logging.error(f"Error parsing request: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 修改前（仅监听本地）
    # uvicorn.run(app, host="127.0.0.1", port=port)

    # 修改后（允许外部访问）
    uvicorn.run(app, host="0.0.0.0", port=port)
