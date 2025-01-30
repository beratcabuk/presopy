Do your users send ONE BILLION requests to your AI agent asking for the same thing? Introducing:
```
+-----------------------------------------+
|    ____                       ______  __|
|   / __ \________  _________  / __ \ \/ /|
|  / /_/ / ___/ _ \/ ___/ __ \/ /_/ /\  / |
| / ____/ /  /  __(__  ) /_/ / ____/ / /  |
|/_/   /_/   \___/____/\____/_/     /_/   |
+-----------------------------------------+
```
# PresoPy

A minimalistic prompt caching service using FastAPI and Redis Stack. Designed for speed and efficiency, Preso-Py caches responses to reduce redundant computations and enhance performance.

## Features

- **FastAPI** backend for handling API requests
- **Redis Stack** as the caching layer
- **Simple API key authentication** for controlled access
- **Dockerized** for easy deployment

## Installation

### Prerequisites

- Docker & Docker Compose

### Quick Start

```sh
docker-compose up -d --build
```

This will:

- Build and start the FastAPI service
- Start a Redis Stack container
- Mount a persistent Redis volume

## Configuration

Environment variables (defined in `docker-compose.yml`):

- `REDIS_HOST` (default: `redis`)
- `REDIS_PORT` (default: `6379`)
- `REDIS_DB` (default: `0`)
- `SERVICE_API_KEYS` (list of authorized API keys)

## API Usage

### Store a response

```sh
curl -X POST "http://127.0.0.1:8000/store" \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"key": "example", "value": "cached response"}'
```

### Retrieve a cached response

```sh
curl -X GET "http://127.0.0.1:8000/get/example" \
     -H "Authorization: Bearer YOUR_API_KEY"
```

## Development

```sh
docker-compose down  # Stop containers
docker-compose up --build  # Rebuild and start
```

### Running without Docker

```sh
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## License

MIT

