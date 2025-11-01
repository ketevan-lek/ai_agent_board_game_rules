# <script async src = "https://cse.google.com/cse.js?cx=10ba722503d954a97" >
# </script >
# <div class = "gcse-search" > </div >
"AIzaSyBQPQnVK-BnNzGss0JSsESxPr_OoPhYLoI"


import requests
from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()


def query_google(board_game):
    API_KEY = os.getenv("GOOGLE_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_CX_KEY")
    query = f"{board_game} rules"

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    try:
        r = requests.get(url, params=params)
    except Exception as e:
        raise (f"{e}")

    results = r.json().get("items", [])

    for i, item in enumerate(results, 1):
        url = item["link"]
        if url.split(".")[-1] == "pdf":
            save_pdf(url)
            break


def save_pdf(url, save_dir="pdfs"):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if url.lower().endswith(".pdf"):
        filename = save_dir.joinpath(os.path.basename(url.split("?")[0]))
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"⚠️ Failed: {url} ({e})")
