# GPT Book Parser

A Python script that scrapes book information from books.toscrape.com using OpenAI's GPT-4 model. The script extracts book titles, prices, and ratings, then saves the data in JSON format.

## Features

- Web scraping of book data
- GPT-4 powered information extraction
- Token usage tracking and cost calculation
- JSON output format
- UTF-8 encoding support

## Requirements

- OpenAI API key (store in `.env` file)
- Python packages:
  - openai
  - tiktoken
  - requests
  - python-dotenv

## Output

Results are saved to `parsed_books.json` with the following structure:
```json
{
  "books": [
    {
      "book": "Book Title",
      "price": 0.00,
      "rating": 0
    }
  ]
}