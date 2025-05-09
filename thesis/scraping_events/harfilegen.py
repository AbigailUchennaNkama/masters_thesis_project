# import json
# import pandas as pd
# from playwright.sync_api import sync_playwright
# from datetime import datetime
# import time

# def capture_eventbrite_har():
#     har_path = '/home/nkama/eventbrite_events.har'
    
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)
#         context = browser.new_context(
#             record_har_path=har_path,
#             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
#             viewport={"width": 1920, "height": 1080}
#         )
#         page = context.new_page()

#         try:
#             # Navigate to Eventbrite's US events page
#             page.goto("https://www.eventbrite.com/d/united-states/all-events/", wait_until="networkidle")
            
#             # Scroll to load events using map-based loading
#             for _ in range(10):  # Increased scroll attempts
#                 page.mouse.wheel(0, 15000)
#                 time.sleep(3)
#                 page.wait_for_selector('div[data-testid="search-event"]', timeout=5000)
                
#             # Wait for final network requests
#             time.sleep(5)

#         finally:
#             context.close()
#             browser.close()
    
#     return har_path

# def process_har(har_path):
#     try:
#         with open(har_path, 'r') as f:
#             har_data = json.load(f)
#     except Exception as e:
#         print(f"Error loading HAR file: {e}")
#         return pd.DataFrame()

#     events = []
    
#     for entry in har_data.get("log", {}).get("entries", []):
#         try:
#             # Focus on map-based search endpoint (per search results)
#             if "/api/v3/destination/search/" in entry["request"]["url"]:
#                 response = json.loads(entry["response"]["content"]["text"])
                
#                 for event in response.get("events", {}).get("results", []):
#                     # Extract core event data
#                     event_info = {
#                         "name": event.get("name"),
#                         "url": event.get("url"),
#                         "start_utc": event.get("start_date"),
#                         "end_utc": event.get("end_date"),
#                         "venue_id": event.get("venue_id"),
#                         "event_id": event.get("id"),
#                         "latitude": event.get("latitude"),
#                         "longitude": event.get("longitude"),
#                         "price": event.get("ticket_availability", {}).get("minimum_price")
#                     }
                    
#                     # Add venue details if available
#                     if "venue" in event:
#                         event_info.update({
#                             "venue_name": event["venue"].get("name"),
#                             "venue_address": event["venue"].get("address"),
#                             "venue_capacity": event["venue"].get("capacity")
#                         })
                    
#                     events.append(event_info)
                
#         except Exception as e:
#             continue

#     return pd.DataFrame(events)

# if __name__ == "__main__":
#     print("Starting HAR capture...")
#     har_file = capture_eventbrite_har()
    
#     print("Processing HAR file...")
#     df = process_har(har_file)
    
#     if not df.empty:
#         df["scraped_date"] = datetime(2025, 3, 16).strftime("%Y-%m-%d")
#         csv_path = '/home/nkama/eventbrite_events.csv'
#         df.to_csv(csv_path, index=False)
#         print(f"Successfully saved {len(df)} events to {csv_path}")
#         print(df.head())
#     else:
#         print("No events found. Possible reasons:")
#         print("1. Eventbrite's API structure changed")
#         print("2. Network requests not captured properly")
#         print("3. Events marked as private (per search result [1])")
#         print("4. Regional restrictions or anti-bot measures")





import json
import pandas as pd
from playwright.sync_api import sync_playwright
from datetime import datetime
import time

def capture_eventbrite_har():
    har_path = 'thesis/eventbrite_events.har'
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            record_har_path=har_path,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        page = context.new_page()

        try:
            # Navigate through multiple pages using URL parameters
            for page_num in range(1, 10):  # 500 pages
                url = f"https://www.eventbrite.com/d/united-states/all-events/?page={page_num}"
                page.goto(url, wait_until="networkidle")
                
                # Scroll to load all events on page
                for _ in range(5):
                    page.mouse.wheel(0, 15000)
                    time.sleep(3)
                    page.wait_for_selector('div[data-testid="search-event"]', timeout=5000)
                
                print(f"Scraped page {page_num}/500")
                time.sleep(3)  # Respect rate limits

        finally:
            context.close()
            browser.close()
    
    return har_path

def process_har(har_path):
    try:
        with open(har_path, 'r') as f:
            har_data = json.load(f)
    except Exception as e:
        print(f"Error loading HAR file: {e}")
        return pd.DataFrame()

    events = []
    
    for entry in har_data.get("log", {}).get("entries", []):
        try:
            if "/api/v3/destination/search/" in entry["request"]["url"]:
                response = json.loads(entry["response"]["content"]["text"])
                
                for event in response.get("events", {}).get("results", []):
                    event_info = {
                        "page_number": entry["request"]["url"].split("page=")[-1].split("&")[0],
                        "name": event.get("name"),
                        "url": event.get("url"),
                        "start_date": event.get("start_date"),
                        "end_date": event.get("end_date"),
                        "venue": event.get("venue", {}).get("name"),
                        "location": f"{event.get('venue', {}).get('address', {}).get('city')}, "
                                  f"{event.get('venue', {}).get('address', {}).get('region')}",
                        "price": event.get("ticket_availability", {}).get("minimum_price"),
                        "organizer": event.get("primary_organizer", {}).get("name"),
                        "category": event.get("category", {}).get("name")
                    }
                    events.append(event_info)
                
        except Exception as e:
            continue

    return pd.DataFrame(events)

if __name__ == "__main__":
    print("Starting multi-page HAR capture...")
    har_file = capture_eventbrite_har()
    
    print("Processing HAR file...")
    df = process_har(har_file)
    
    if not df.empty:
        df["scraped_date"] = datetime(2025, 3, 16).strftime("%Y-%m-%d")
        csv_path = '/home/nkama/eventbrite_events.csv'
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved {len(df)} events from {df['page_number'].nunique()} pages")
        print(df.head())
    else:
        print("No events found. Potential issues:")
        print("- Website structure changed")
        print("- Anti-scraping mechanisms detected")
        print("- Network requests not captured properly")
