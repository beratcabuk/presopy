from dataclasses import field

from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import httpx
import json
import numpy as np
import hashlib
import logging
from typing import List, Dict, Optional
from pydantic_settings import BaseSettings
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
from collections import defaultdict


# Configuration
class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    vector_dim: int = 1536  # text-embedding-3-small dimension
    similarity_threshold: float = 0.9
    openai_api_base: str = "https://api.openai.com/v1"
    cache_ttl: int = 86400  # 24 hours in seconds
    request_timeout: int = 10  # seconds
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 3600  # window in seconds
    service_api_keys: List[str] = field()  # List of valid API keys for the service

    class Config:
        env_file = ".env"


settings = Settings()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cache-service")


# Simple rate limiter using Redis
class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.window = settings.rate_limit_window
        self.max_requests = settings.rate_limit_requests

    async def check_rate_limit(self, api_key: str) -> bool:
        key = f"ratelimit:{api_key}"
        pipe = self.redis.pipeline()
        now = datetime.now(timezone.utc).timestamp()

        try:
            # Cleanup old entries and add new request
            await pipe.zremrangebyscore(key, 0, now - self.window)
            await pipe.zadd(key, {str(now): now})
            await pipe.zcard(key)
            await pipe.expire(key, self.window)
            results = await pipe.execute()

            request_count = results[2]
            return request_count <= self.max_requests
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open for MVP


# Metrics tracking
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.last_reset = datetime.now(timezone.utc)

    async def track_request(self, cache_hit: bool, api_key: str):
        self.metrics["total_requests"] += 1
        self.metrics[f'cache_{"hit" if cache_hit else "miss"}'] += 1
        self.metrics[f"requests_{api_key}"] += 1

    async def get_metrics(self):
        return dict(self.metrics)

    async def reset_metrics(self):
        self.metrics.clear()
        self.last_reset = datetime.now(timezone.utc)


metrics = MetricsCollector()


class RedisClient:
    def __init__(self):
        self.client: Optional[Redis] = None
        self.index_name = "prompt_cache_idx"

    async def get_client(self) -> Redis:
        if not self.client or not await self.client.ping():
            await self.init()
        return self.client

    async def init(self):
        self.client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
            encoding="utf-8",
        )
        await self.init_index()

    async def init_index(self):
        try:
            # Drop existing index if it exists
            try:
                await self.client.ft(self.index_name).dropindex()
                logger.info("Dropped existing index")
            except:
                pass

            schema = (
                TextField("messages"),
                TextField("response"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": settings.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )

            await self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(
                    prefix=["cache:"], index_type=IndexType.HASH
                ),
            )
            logger.info("Successfully created index")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise e

    async def execute_with_retry(self, operation, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = await self.get_client()
                return await operation(client, *args, **kwargs)
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

    async def close(self):
        if self.client:
            await self.client.close()


redis_client = RedisClient()


# API key validation
api_key_header = APIKeyHeader(name="X-API-Key")


async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key not in settings.service_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# FastAPI startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await redis_client.init()
    yield
    # Shutdown
    await redis_client.close()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
async def get_embedding(messages: List[Dict], auth_header: str) -> List[float]:
    """Get embedding for messages using OpenAI API"""
    text = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.openai_api_base}/embeddings",
                headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                },
                json={"input": text, "model": "text-embedding-3-small"},
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except httpx.HTTPError as e:
            logger.error(f"Error getting embedding: {e}")
            raise HTTPException(status_code=response.status_code, detail=str(e))


def generate_cache_key(messages: List[Dict]) -> str:
    """Generate a cache key for the messages"""
    msg_str = json.dumps(messages, sort_keys=True)
    return f"cache:{hashlib.sha256(msg_str.encode()).hexdigest()}"


# Endpoints
@app.get("/health")
async def health():
    status = {"status": "healthy", "timestamp": time.time(), "components": {}}

    try:
        await redis_client.client.ping()
        status["components"]["redis"] = "healthy"
    except Exception as e:
        status["components"]["redis"] = f"unhealthy: {str(e)}"
        status["status"] = "degraded"

    return status


@app.get("/cache/stats")
async def cache_stats(api_key: str = Depends(validate_api_key)):
    try:
        info = await redis_client.client.info()
        metrics_data = await metrics.get_metrics()

        return {
            "used_memory": info["used_memory"],
            "used_memory_human": info["used_memory_human"],
            "total_keys": await redis_client.client.dbsize(),
            "connected_clients": info["connected_clients"],
            "metrics": metrics_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache(api_key: str = Depends(validate_api_key)):
    """Clear the entire cache"""
    try:
        await redis_client.client.flushdb()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/{cache_key}")
async def invalidate_cache_entry(
    cache_key: str, api_key: str = Depends(validate_api_key)
):
    """Invalidate specific cache entry"""
    try:
        await redis_client.client.delete(f"cache:{cache_key}")
        return {"status": "success", "message": f"Cache key {cache_key} invalidated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(validate_api_key)):
    """Get service metrics"""
    return await metrics.get_metrics()


@app.post("/chat/completions")
async def completions(request: Request, api_key: str = Depends(validate_api_key)):
    """Main endpoint for OpenAI API passthrough with caching"""
    try:
        # Rate limiting check
        rate_limiter = RateLimiter(redis_client.client)
        if not await rate_limiter.check_rate_limit(api_key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        auth = request.headers.get("Authorization")
        if not auth:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        data = await request.json()
        if "messages" not in data:
            # Pass through non-chat requests directly
            return await openai_passthrough(auth, data)

        messages = data["messages"]
        messages_str = json.dumps(messages)

        # Try exact cache first
        cached_response = await redis_client.client.get(messages_str)
        if cached_response is not None:
            logger.info("Exact cache hit")
            await metrics.track_request(True, api_key)
            return json.loads(cached_response)

        # Try semantic cache
        # cache_hit = False
        try:
            embedding = await get_embedding(messages, auth)
            cache_key = generate_cache_key(messages)

            # Prepare vector search
            query = (
                Query("*=>[KNN 1 @embedding $vec_param AS score]")
                .sort_by("score")
                .return_fields("score", "messages", "response")
                .dialect(2)
            )
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # Search for similar prompts
            results = await redis_client.client.ft(redis_client.index_name).search(
                query, query_params={"vec_param": embedding_bytes}
            )

            if results.total:
                similarity_score = 1 - float(results.docs[0].score)
                if similarity_score >= settings.similarity_threshold:
                    logger.info(
                        f"Semantic cache hit with similarity {similarity_score}"
                    )
                    # cache_hit = True
                    await metrics.track_request(True, api_key)
                    return json.loads(results.docs[0].response)

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            # Continue to API call on error

        # Cache miss - call OpenAI API
        response = await openai_passthrough(auth, data)
        await metrics.track_request(False, api_key)

        # Store in caches
        try:
            await redis_client.client.set(
                messages_str, json.dumps(response), ex=settings.cache_ttl
            )

            cache_data = {
                "messages": messages_str,
                "response": json.dumps(response),
                "embedding": embedding_bytes,
            }
            await redis_client.client.hset(cache_key, mapping=cache_data)
            await redis_client.client.expire(cache_key, settings.cache_ttl)

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            # Continue even if caching fails

        return response

    except Exception as e:
        logger.error(f"Unhandled error in completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def openai_passthrough(auth: str, data: dict) -> dict:
    """Pass through requests to OpenAI API"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.openai_api_base}/chat/completions",
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                },
                json=data,
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise HTTPException(
                status_code=response.status_code if hasattr(e, "response") else 500,
                detail=str(e),
            )


# Error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500, content={"error": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
