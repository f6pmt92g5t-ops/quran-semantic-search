"""Opens the website like a visitor and checks that the app really loads.

- If the app is asleep, presses the wake-up button and waits for it.
- Succeeds only when the app's search box is on the page.
- Otherwise saves a screenshot and the page text, and fails (GitHub then emails the repository owner).
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL") or "https://albahith-quran.streamlit.app/"
if not URL.startswith("http"):
    URL = f"https://{URL}.streamlit.app/"
APP_ELEMENT = '[data-testid="stSelectbox"]'   # the search box: exists only once the app itself has loaded
WAKE = ("get this app back up", "Yes, get this app back up")
TIMEOUT = int(os.environ.get("TIMEOUT", "300"))


def page_text(page):
    parts = []
    for frame in page.frames:
        try:
            parts.append(frame.inner_text("body", timeout=3000))
        except Exception:
            pass
    return "\n".join(parts)


def main():
    print("Checking", URL)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        for attempt in range(3):
            try:
                page.goto(URL, wait_until="domcontentloaded", timeout=120000)
                break
            except Exception as e:
                print(f"Could not open the page (attempt {attempt + 1}): {e}".splitlines()[0])
                if attempt == 2:
                    print("FAILED: the site did not answer.")
                    browser.close()
                    return 1
                time.sleep(20)
        start, woke = time.time(), False
        while time.time() - start < TIMEOUT:
            loaded = False
            for frame in page.frames:
                try:
                    loaded = loaded or frame.locator(APP_ELEMENT).count() > 0
                except Exception:
                    pass
            if loaded:
                print(f"OK: app loaded after {time.time() - start:.0f}s" + (" (was asleep, woken)" if woke else ""))
                browser.close()
                return 0
            for frame in page.frames:
                for label in WAKE:
                    btn = frame.get_by_role("button", name=label)
                    try:
                        if btn.count():
                            print("App was asleep - pressing the wake-up button")
                            btn.first.click()
                            woke = True
                    except Exception:
                        pass
            time.sleep(5)
        page.screenshot(path="keepalive_failure.png", full_page=True)
        print("FAILED: the app did not load. Page text:\n" + page_text(page)[:2000])
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
