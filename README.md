# 🚀 LLM Core Proxy Server

A production-ready, extensible **LLM proxy server** built with **Node.js + Express**, supporting multiple language model providers via a unified **OpenAI-compatible API** interface.

This service allows developers to call `/v1/chat/completions`, `/v1/embeddings`, and `/v1/images/generations` using the **same request/response format** used by OpenAI, while internally routing calls to different LLM providers like OpenAI, Azure OpenAI, Anthropic, or Hugging Face.

---

## ✅ Features

- 🔗 **OpenAI-Compatible Endpoints**  
  Easily integrate OpenAI clients (e.g. `openai`, `langchain`, `llamaindex`, etc.) with any supported provider.

- 🔌 **Multi-Provider Support**  
  Route requests to:
  - `openai/gpt-4`
  - `anthropic/claude-3-sonnet`
  - `azure/gpt-35-deployment`
  - `huggingface/tiiuae/falcon-7b-instruct`

- 📤 **Streaming Support**  
  Fully supports `stream: true` for OpenAI and Azure-compatible streaming APIs.

- 🧠 **Embeddings + Image Generation**  
  Generate vector embeddings or DALL·E-style images using a consistent API.

- 🔧 **Zero Middleware Opinionation**  
  No rate limiting, caching, or key management logic — delegate all that to your cloud infrastructure.

---

## 📦 Supported Providers

| Provider     | Chat Completions | Embeddings | Image Generation |
|--------------|------------------|------------|------------------|
| `openai`     | ✅                | ✅          | ✅               |
| `azure`      | ✅                | ✅          | ❌             |
| `anthropic`  | ✅                | ❌          | ❌               |
| `huggingface`| ✅                | ✅          | ❌               |

> **Note**: Azure support for image generation depends on your resource configuration.

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/llm-core-proxy
cd llm-core-proxy
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and populate with your keys:

```bash
cp .env.example .env
```

**`.env.example` contents:**

```env
OPENAI_API_KEY=
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_KEY=
PORT=4000
```

Fill in your actual API keys and Azure endpoint URL.

---

### 4. Start the server

```bash
node server.js
```

Server will start at:

```
http://localhost:4000
```

---

## 📚 Usage

### 🔹 Chat Completions

```http
POST /v1/chat/completions
```

#### Example Request

```json
{
  "model": "openai/gpt-3.5-turbo",
  "messages": [
    { "role": "user", "content": "Tell me a joke." }
  ],
  "stream": false
}
```

#### Example Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Why don’t skeletons fight each other? They don’t have the guts."
      },
      "finish_reason": "stop"
    }
  ]
}
```

---

### 🔹 Embeddings

```http
POST /v1/embeddings
```

#### Example Request

```json
{
  "model": "openai/text-embedding-ada-002",
  "input": "The quick brown fox jumps over the lazy dog."
}
```

---

### 🔹 Image Generation

```http
POST /v1/images/generations
```

#### Example Request

```json
{
  "prompt": "A fantasy castle on a cliff during sunset",
  "n": 1,
  "size": "512x512",
  "model": "openai/dall-e"
}
```

---

### 🔹 Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok",
  "timestamp": "2025-04-08T12:00:00Z"
}
```

---

## 🧩 Model Routing Syntax

Models are prefixed with the provider name to determine which backend to route the request to:

| Format                        | Description                         |
|-------------------------------|-------------------------------------|
| `openai/gpt-4`               | Routes to OpenAI API                |
| `azure/gpt35-deployment`    | Routes to Azure OpenAI deployment   |
| `anthropic/claude-3-sonnet` | Routes to Claude via Anthropic API  |
| `huggingface/falcon-7b`     | Routes to HuggingFace Inference API |

---

## 📌 Notes

- **Streaming:** Supported for `openai`, `azure`, and `anthropic`.
- **Prompt formatting** is provider-specific (e.g., Anthropic requires `Human:`/`Assistant:` roles and is handled internally).
- **No rate limits / API keys / auth management** — rely on cloud service limits or reverse proxy (e.g., Cloudflare, NGINX).
- **Production ready**: deploy behind load balancers or edge gateways.

---

## 📞 Contributions

Feel free to connect, contact, or collaborate on the project.
