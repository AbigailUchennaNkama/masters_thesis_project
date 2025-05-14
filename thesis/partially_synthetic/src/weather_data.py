import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
import time


# Split events dataframe into two halves
import pandas as pd

# Split events dataframe into three parts
def split_eventsdf_into_three(df):
    """
    Splits the events dataframe into three roughly equal parts.

    Args:
        df (pd.DataFrame): The events dataframe.

    Returns:
        tuple: A tuple containing the three splits of the dataframe.
    """
    # Calculate split points
    first_split_index = len(df) // 3  
    second_split_index = 2 * (len(df) // 3)  
    
    # Split into three parts using the calculated indices
    first_split = df.iloc[:first_split_index]  
    second_split = df.iloc[first_split_index:second_split_index]  
    last_split = df.iloc[second_split_index:] 

    return first_split, second_split, last_split


def concat_all_splits(first_split, second_split, last_split):
    # Concatenate the two halves back together
    df_event_combined = pd.concat([first_split, second_split, last_split], axis=0)
    # Reset the index of the combined dataframe
    df_event_combined = df_event_combined.reset_index(drop=True)
    
    return df_event_combined


# def get_weather_for_events(events_df):
#     """
#     Extract weather data for each event in the dataframe
    
#     Args:
#         events_df: DataFrame containing event data with start_time, lat, and lng columns
        
#     Returns:
#         DataFrame with weather data columns added
#     """
#     # Setup the Open-Meteo API client with cache and retry on error
#     cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
#     retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
#     openmeteo = openmeteo_requests.Client(session=retry_session)
    
#     # Create a copy of the dataframe to avoid modifying the original
#     result_df = events_df.copy()
    
#     # Add empty columns for weather data
#     result_df['weather_code'] = None
#     result_df['temperature_2m_mean'] = None
#     result_df['precipitation_sum'] = None
#     result_df['precipitation_hours'] = None
#     result_df['wind_speed_10m_max'] = None
    
#     # Process events in batches to avoid overwhelming the API
#     batch_size = 50
#     num_events = len(events_df)
    
#     for i in range(0, num_events, batch_size):
#         batch = events_df.iloc[i:min(i+batch_size, num_events)]
#         print(f"Processing batch {i//batch_size + 1}/{(num_events+batch_size-1)//batch_size}")
        
#         # Process each event in the batch
#         for idx, event in batch.iterrows():
#             # Check if lat and lng are valid
#             if pd.isna(event['lat_x']) or pd.isna(event['lng_x']):
#                 print(f"Skipping event {event['event_id']} - missing coordinates")
#                 continue
                
#             # Parse the start_time to get the date
#             try:
#                 event_time = pd.to_datetime(event['start_time'])
#                 event_date = event_time.strftime('%Y-%m-%d')
#             except:
#                 print(f"Skipping event {event['event_id']} - invalid date format")
#                 continue
            
#             # Prepare API parameters
#             params = {
#                 "latitude": event['lat_x'],
#                 "longitude": event['lng_x'],
#                 "start_date": event_date,
#                 "end_date": event_date,
#                 "daily": ["weather_code", "temperature_2m_mean", "precipitation_sum", 
#                           "precipitation_hours", "wind_speed_10m_max"],
#                 "timeformat": "unixtime",
#                 "timezone": "GMT"  # Using GMT as a default
#             }
            
#             try:
#                 # Make API request
#                 responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
#                 response = responses[0]
                
#                 # Process daily data
#                 daily = response.Daily()
                
#                 # Get weather data
#                 weather_code = daily.Variables(0).ValuesAsNumpy()[0]
#                 temperature = daily.Variables(1).ValuesAsNumpy()[0]
#                 precipitation_sum = daily.Variables(2).ValuesAsNumpy()[0]
#                 precipitation_hours = daily.Variables(3).ValuesAsNumpy()[0]
#                 wind_speed = daily.Variables(4).ValuesAsNumpy()[0]
                
#                 # Store data in result dataframe
#                 result_df.loc[idx, 'weather_code'] = weather_code
#                 result_df.loc[idx, 'temperature_2m_mean'] = temperature
#                 result_df.loc[idx, 'precipitation_sum'] = precipitation_sum
#                 result_df.loc[idx, 'precipitation_hours'] = precipitation_hours
#                 result_df.loc[idx, 'wind_speed_10m_max'] = wind_speed
                
#                 # Add a small delay to avoid rate limiting
#                 time.sleep(0.1)
                
#             except Exception as e:
#                 print(f"Error getting weather for event {event['event_id']}: {e}")
        
#         # Add a delay between batches to avoid rate limiting
#         time.sleep(1)
    
#     return result_df




def get_weather_info(events_df, lat_col='lat', lon_col='lng', id_col='event_id'):
    """
    Extract weather data for each event in the dataframe
    
    Args:
        events_df: DataFrame containing event data with start_time, lat, and lng columns
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        id_col: Column name for event ID
        
    Returns:
        DataFrame with weather data columns added
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = events_df.copy()
    
    # Add empty columns for weather data
    result_df['weather_code'] = None
    result_df['temperature_2m_mean'] = None
    result_df['precipitation_sum'] = None
    result_df['precipitation_hours'] = None
    result_df['wind_speed_10m_max'] = None
    
    # Process events in batches to avoid overwhelming the API
    batch_size = 50
    num_events = len(events_df)
    
    for i in range(0, num_events, batch_size):
        batch = events_df.iloc[i:min(i+batch_size, num_events)]
        print(f"Processing batch {i//batch_size + 1}/{(num_events+batch_size-1)//batch_size}")
        
        # Process each event in the batch
        for idx, event in batch.iterrows():
            # Check if lat and lng are valid
            if pd.isna(event[lat_col]) or pd.isna(event[lon_col]):
                print(f"Skipping event {event[id_col]} - missing coordinates")
                continue
                
            # Parse the start_time to get the date
            try:
                event_time = pd.to_datetime(event['start_time'])
                event_date = event_time.strftime('%Y-%m-%d')
            except:
                print(f"Skipping event {event[id_col]} - invalid date format")
                continue
            
            # Prepare API parameters
            params = {
                "latitude": event[lat_col],
                "longitude": event[lon_col],
                "start_date": event_date,
                "end_date": event_date,
                "daily": ["weather_code", "temperature_2m_mean", "precipitation_sum", 
                          "precipitation_hours", "wind_speed_10m_max"],
                "timeformat": "unixtime",
                "timezone": "GMT"  # Using GMT as a default
            }
            
            try:
                # Make API request
                responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
                response = responses[0]
                
                # Process daily data
                daily = response.Daily()
                
                # Get weather data
                weather_code = daily.Variables(0).ValuesAsNumpy()[0]
                temperature = daily.Variables(1).ValuesAsNumpy()[0]
                precipitation_sum = daily.Variables(2).ValuesAsNumpy()[0]
                precipitation_hours = daily.Variables(3).ValuesAsNumpy()[0]
                wind_speed = daily.Variables(4).ValuesAsNumpy()[0]
                
                # Store data in result dataframe
                result_df.loc[idx, 'weather_code'] = weather_code
                result_df.loc[idx, 'temperature_2m_mean'] = temperature
                result_df.loc[idx, 'precipitation_sum'] = precipitation_sum
                result_df.loc[idx, 'precipitation_hours'] = precipitation_hours
                result_df.loc[idx, 'wind_speed_10m_max'] = wind_speed
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting weather for event {event[id_col]}: {e}")
        
        # Add a delay between batches to avoid rate limiting
        time.sleep(1)
    
    return result_df