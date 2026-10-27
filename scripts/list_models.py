"""Lists the Gemini models your API key can use for text generation.

Run it from the project folder with the virtual environment active:
    python scripts/list_models.py

Copy one of the printed names (without the "models/" prefix) into GEMINI_MODEL in your .env file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allows "import config" from scripts/

from config import get_gemini_api_key  # noqa: E402


def main() -> int:
    api_key = get_gemini_api_key()
    if not api_key:
        print("GEMINI_API_KEY is not set. Add it to your .env file first (see .env.example).")
        return 1

    from google import genai

    client = genai.Client(api_key=api_key)
    print("Models that support generateContent. Use one of these names for GEMINI_MODEL:")
    found = 0
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"  {model.name.replace('models/', '', 1)}")
            found += 1
    if found == 0:
        print("No text-generation models were returned. Check the key and your quota in Google AI Studio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
