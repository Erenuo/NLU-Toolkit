# NLU Toolkit API

An API for common Natural Language Understanding tasks like sentiment analysis and entity extraction.

## Quick Start

1.  **Clone and install:**

    ```bash
    git clone https://github.com/Erenuo/NLU_API.git
    cd NLU_API
    pip install -r requirements.txt
    ```

2.  **Run the server:**

    ```bash
    python app.py
    ```

    The API will be available at `http://127.0.0.1:5000`.

## Example Usage

To get the sentiment of a piece of text, send a `POST` request to the `/sentiment` endpoint.

**Request:**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "This API is fantastic!"}' \
  http://127.0.0.1:5000/sentiment
```

**Response:**

```json
{
  "sentiment": "positive",
  "score": 0.98
}

```

You can run python client.py file to test the API
-----
