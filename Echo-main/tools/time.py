# tools/time.py
from langchain.tools import tool
from datetime import datetime
import pytz

@tool
def get_time(city: str = "toronto") -> str:
    """Returns the current time in a given city. If no city is specified, defaults to Toronto. Use this when user asks for time in a specific city."""
    try:
        city_timezones = {
            "toronto": "America/Toronto", # added toronto and made it the defualt (17/11/25)
            "new york": "America/New_York",
            "london": "Europe/London",
            "tokyo": "Asia/Tokyo",
            "sydney": "Australia/Sydney"
        }
        city_key = city.lower()
        if city_key not in city_timezones:
            return f"Sorry, I don't know the timezone for {city}."
        timezone = pytz.timezone(city_timezones[city_key])
        current_time = datetime.now(timezone).strftime("%I:%M %p")
        return f"The current time in {city.title()} is {current_time}."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Returns the current date, day of the week, and local time in Toronto EST. Use this when user asks what day it is, what date it is, or what time it is without specifying a different city (17/11/25)"""
    toronto_tz = pytz.timezone("America/Toronto")
    now = datetime.now(toronto_tz)
    return now.strftime("Today is %A, %B %d, %Y. The current time is %I:%M %p EST.")