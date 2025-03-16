import requests
import json
import datetime
import os
from geopy.distance import geodesic
import openai
from dotenv import load_dotenv


class EventAnalysisService:
    def __init__(self, api_key_events, api_key_openai):
        """Initialize the service with necessary API keys."""
        self.api_key_events = api_key_events
        self.api_key_openai = api_key_openai

    def get_nearby_events(self, latitude, longitude, radius_km=5, days_ahead=7):
        """
        Fetch nearby events using the Ticketmaster Discovery API.

        Args:
            latitude: Current latitude
            longitude: Current longitude
            radius_km: Search radius in kilometers
            days_ahead: Number of days to look ahead for events

        Returns:
            List of events with relevant details
        """
        # Calculate date range for events
        start_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = (
            datetime.datetime.now() + datetime.timedelta(days=days_ahead)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Set up API request
        base_url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {
            "apikey": self.api_key_events,
            "latlong": f"{latitude},{longitude}",
            "radius": radius_km,
            "unit": "km",
            "startDateTime": start_date,
            "endDateTime": end_date,
            "size": 20,  # Number of events to return
            "sort": "date,asc",
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract relevant event information
            events = []
            if (
                "page" in data
                and "totalElements" in data["page"]
                and data["page"]["totalElements"] > 0
            ):
                for event in data.get("_embedded", {}).get("events", []):
                    event_info = {
                        "id": event.get("id"),
                        "name": event.get("name"),
                        "type": event.get("type"),
                        "url": event.get("url"),
                        "start_time": event.get("dates", {})
                        .get("start", {})
                        .get("dateTime"),
                        "venue": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("name"),
                        "venue_capacity": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("capacity"),
                        "venue_address": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("address", {})
                        .get("line1"),
                        "venue_city": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("city", {})
                        .get("name"),
                        "venue_lat": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("location", {})
                        .get("latitude"),
                        "venue_lon": event.get("_embedded", {})
                        .get("venues", [{}])[0]
                        .get("location", {})
                        .get("longitude"),
                        "classifications": event.get("classifications", [{}])[0]
                        if event.get("classifications")
                        else {},
                        "price_ranges": event.get("priceRanges", []),
                        "attendance_expectation": self._estimate_attendance(event),
                    }
                    events.append(event_info)

            return events

        except requests.exceptions.RequestException as e:
            print(f"Error fetching events: {e}")
            return []

    def _estimate_attendance(self, event):
        """Estimate attendance based on event data."""
        # This is a simplified estimation logic that could be improved
        venue_capacity = (
            event.get("_embedded", {}).get("venues", [{}])[0].get("capacity")
        )

        if venue_capacity:
            try:
                venue_capacity = int(venue_capacity)
                # For popular events, assume higher attendance
                if (
                    "popular" in event.get("name", "").lower()
                    or event.get("popularity", 0) > 0.7
                ):
                    return int(venue_capacity * 0.9)
                return int(venue_capacity * 0.7)  # Default assumption
            except (ValueError, TypeError):
                pass

        # If no capacity data, estimate based on venue type
        venue_name = (
            event.get("_embedded", {}).get("venues", [{}])[0].get("name", "").lower()
        )
        if "stadium" in venue_name:
            return 30000
        elif "arena" in venue_name:
            return 15000
        elif "theater" or "theatre" in venue_name:
            return 2000
        elif "hall" in venue_name:
            return 1000
        else:
            return 500  # Default modest attendance

    def analyze_events_with_ai(self, events, current_lat, current_lon):
        """
        Analyze events using OpenAI to predict ride demand patterns.

        Args:
            events: List of event dictionaries
            current_lat: Current latitude
            current_lon: Current longitude

        Returns:
            Dictionary containing AI analysis and ride prediction
        """
        if not events:
            return {"message": "No events found to analyze.", "recommendations": []}

        # Simplify event data to reduce token count
        simplified_events = []
        for event in events:
            simplified_event = {
                "name": event.get("name"),
                "venue": event.get("venue"),
                "start_time": event.get("start_time"),
                "venue_capacity": event.get("venue_capacity"),
                "venue_address": event.get("venue_address"),
                "venue_city": event.get("venue_city"),
                "attendance_expectation": event.get("attendance_expectation"),
                # Include basic price info if available
                "price_range": f"{event['price_ranges'][0].get('min', 'N/A')}-{event['price_ranges'][0].get('max', 'N/A')}"
                if event.get("price_ranges")
                else "N/A",
            }
            simplified_events.append(simplified_event)

        # Limit to top 10 events by expected attendance
        simplified_events.sort(
            key=lambda x: x.get("attendance_expectation", 0), reverse=True
        )
        simplified_events = simplified_events[:10]

        # Format events for the AI prompt
        events_text = json.dumps(simplified_events, indent=2)

        # Simplified prompt
        prompt = f"""
        Analyze these top events near ({current_lat}, {current_lon}):
        {events_text}

        Provide concise JSON with:
        1. high_demand_events: List of event names likely to generate significant ride demand
        2. peak_times: Expected peak ride request times
        3. ride_volume_estimates: Estimated total rides needed
        """

        try:
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=self.api_key_openai)

            # Call OpenAI API with reduced max_tokens
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a transportation analyst. Provide brief, focused analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,  # Reduced from 1500
            )

            # Extract and parse the AI's response
            analysis_text = response.choices[0].message.content

            # Find JSON in the response
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}") + 1

            if json_start >= 0 and json_end > 0:
                analysis_json = json.loads(analysis_text[json_start:json_end])
                return analysis_json
            else:
                return {
                    "message": "AI analysis completed but couldn't be parsed as JSON",
                    "raw_analysis": analysis_text,
                }

        except Exception as e:
            print(f"Error analyzing events with AI: {e}")
            return {"error": str(e), "message": "Failed to analyze events with AI"}

    def notify_operations_team(self, analysis, webhook_url=None):
        """
        Send the analysis to operations team via webhook.

        Args:
            analysis: The analysis dictionary
            webhook_url: URL to send the notification

        Returns:
            Boolean indicating success/failure
        """
        if not webhook_url:
            print("Analysis results:", json.dumps(analysis, indent=2))
            return True

        try:
            notification = {
                "timestamp": datetime.datetime.now().isoformat(),
                "title": "Upcoming Event Ride Demand Forecast",
                "analysis": analysis,
            }

            response = requests.post(
                webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(notification),
            )
            response.raise_for_status()
            return True

        except Exception as e:
            print(f"Error sending notification: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys from environment variables
    EVENT_API_KEY = os.getenv("TICKETMASTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not EVENT_API_KEY or not OPENAI_API_KEY:
        raise ValueError(
            "Please set TICKETMASTER_API_KEY and OPENAI_API_KEY in your .env file"
        )

    # Example coordinates (New York City)
    current_lat = 40.7128
    current_lon = -74.0060

    # Initialize service
    service = EventAnalysisService(EVENT_API_KEY, OPENAI_API_KEY)

    # Get nearby events
    events = service.get_nearby_events(current_lat, current_lon)

    # Analyze events and predict ride demand
    analysis = service.analyze_events_with_ai(events, current_lat, current_lon)

    # Send to operations team (or just print for this example)
    service.notify_operations_team(analysis)
