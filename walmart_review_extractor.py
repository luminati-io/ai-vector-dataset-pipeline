import json
import logging
import os
import time

import google.generativeai as genai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def scrape_walmart_reviews(product_id="609040889", start_page=1, max_pages=None):
    """Scrape Walmart reviews using Bright Data and Gemini AI."""
    base_url = f"https://www.walmart.com/reviews/product/{product_id}"
    output_file = "extracted_walmart_reviews.json"

    # Configure Gemini AI
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.0-flash", generation_config={"response_mime_type": "application/json"}
    )

    # Initialize output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f)

    current_page = start_page
    consecutive_failures = 0
    total_reviews = 0

    while True:
        # Failure handling
        if consecutive_failures >= 3:
            logger.error("Stopping after 3 consecutive failures")
            break

        # Max pages check
        if max_pages and current_page > start_page + max_pages - 1:
            logger.info(f"Reached maximum configured pages: {max_pages}")
            break

        logger.info(f"Processing page {current_page}")

        # Fetch page content
        markdown = fetch_page(base_url, current_page)
        if not markdown:
            logger.warning(f"Failed to retrieve page {current_page}")
            consecutive_failures += 1
            current_page += 1
            continue

        # Extract and save reviews
        page_reviews = extract_reviews(markdown, model)
        if page_reviews:
            total_reviews += len(page_reviews)
            save_reviews(page_reviews, output_file)
            logger.info(
                f"Page {current_page}: Added {len(page_reviews)} reviews (Total: {total_reviews})"
            )
            consecutive_failures = 0
        else:
            logger.warning(f"No reviews extracted from page {current_page}")
            consecutive_failures += 1

        # Pagination check
        if f"page={current_page + 1}" not in markdown:
            logger.info("Reached final page of reviews")
            break

        current_page += 1
        time.sleep(5)  # Rate limiting

    logger.info(f"Scraping complete. Total reviews collected: {total_reviews}")


def fetch_page(base_url, page_num):
    """Fetch page content using Bright Data web unlocker."""
    url = f"{base_url}?entryPoint=viewAllReviewsBottom&page={page_num}"

    for attempt in range(1, 4):
        try:
            logger.debug(f"Fetching page {page_num} (attempt {attempt})")

            response = requests.post(
                "https://api.brightdata.com/request",
                headers={
                    "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "zone": "web_unlocker2",
                    "url": url,
                    "format": "raw",
                    "data_format": "markdown",
                },
                timeout=60,
            )

            if response.status_code == 200 and len(response.text) > 1000:
                return response.text

            logger.warning(
                f"Invalid response from page {page_num} (status: {response.status_code}, "
                f"length: {len(response.text)})"
            )

        except Exception as e:
            logger.warning(f"Fetch attempt {attempt} failed: {str(e)}")

        time.sleep(3)

    return None


def extract_reviews(markdown, model):
    """Extract and structure reviews using Gemini AI."""
    prompt = f"""
Extract all customer reviews from this Walmart product page content.
Return a JSON array of review objects with the following structure:
{{
  "reviews": [
    {{
      "date": "YYYY-MM-DD or original date format if available",
      "title": "Review title/headline",
      "description": "Review text content",
      "rating": <integer from 1-5>
    }}
  ]
}}

Rules:
- Include all reviews found on the page
- Use null for any missing fields
- Convert ratings to integers (1-5)
- Extract the full review text, not just snippets
- Preserve original review text without summarizing

Here's the page content:
{markdown}
"""

    try:
        response = model.generate_content(prompt)
        logger.debug(f"Partial Gemini response: {response.text[:200]}...")

        # Parse the JSON response
        result = json.loads(response.text)
        reviews = result.get("reviews", [])

        # Validate and clean reviews
        validated_reviews = []
        for review in reviews:
            # Skip reviews without description
            if (
                not review.get("description")
                or review.get("description", "").strip().lower() == "null"
            ):
                continue

            # Ensure rating is an integer
            rating = review.get("rating")
            if rating is not None:
                try:
                    rating = int(rating)
                    if rating < 1 or rating > 5:
                        rating = None
                except (ValueError, TypeError):
                    rating = None

            validated_reviews.append(
                {
                    "date": review.get("date"),
                    "title": review.get("title"),
                    "description": review.get("description", "").strip(),
                    "rating": rating,
                }
            )

        return validated_reviews

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.error(f"Response text: {response.text[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return []


def save_reviews(new_reviews, output_file):
    """Append new reviews to JSON output file."""
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            all_reviews = json.load(f)

        all_reviews.extend(new_reviews)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_reviews, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Failed to save reviews: {str(e)}")


if __name__ == "__main__":
    try:
        scrape_walmart_reviews()
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        exit(1)