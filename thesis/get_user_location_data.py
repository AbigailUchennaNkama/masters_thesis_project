# def add_user_coordinates_scalable(users_df, checkpoint_file='geocoding_progress.pkl'):
#     """
#     Add latitude and longitude to users based on their existing location data
#     Optimized for large datasets with checkpointing and batch processing
    
#     Parameters:
#     -----------
#     users_df : pandas.DataFrame
#         DataFrame containing a 'location' column
#     checkpoint_file : str
#         File path to save checkpointing data
        
#     Returns:
#     --------
#     pandas.DataFrame
#         Original DataFrame with added 'lat' and 'lng' columns
#     """
    
#     import pandas as pd
#     import numpy as np
#     import os
#     import time
#     import re
#     from geopy.geocoders import Nominatim
#     from geopy.extra.rate_limiter import RateLimiter
#     from datetime import datetime
    
#     # Create a copy of the DataFrame to avoid SettingWithCopyWarning
#     # This is critical - ensures we're working with a true copy, not a view
#     users_df = users_df.copy()
    
#     start_time = datetime.now()
#     print(f"Starting geocoding process at {start_time.strftime('%H:%M:%S')} for {len(users_df)} locations")
    
#     # Load previous progress if available
#     if os.path.exists(checkpoint_file):
#         try:
#             cache_data = pd.read_pickle(checkpoint_file)
#             location_cache = cache_data.get('location_cache', {})
#             completed_indices = cache_data.get('completed_indices', [])
#             print(f"Loaded cache with {len(location_cache)} locations and {len(completed_indices)} completed indices")
#         except Exception as e:
#             print(f"Error loading checkpoint file: {e}")
#             location_cache = {}
#             completed_indices = []
#     else:
#         location_cache = {}
#         completed_indices = []
    
#     # Known problematic locations with manual coordinates
#     known_locations = {
#         "Djokja Yogyakarta Indonesia": (-7.797, 110.370),  # Yogyakarta coordinates
#         "Santo Domingo  Dominican Republic": (18.486, -69.932),  # Santo Domingo coordinates
#         # Add more problematic locations as you discover them
#     }
    
#     # Initialize geocoder with rate limiting
#     geolocator = Nominatim(user_agent="geopy_user_locations_batch")
#     geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)
    
#     def clean_location(location):
#         """Thoroughly clean and standardize location strings"""
#         if pd.isna(location) or not location or location.isspace():
#             return None
            
#         # Convert to string if not already and strip whitespace
#         location = str(location).strip()
        
#         # Skip empty strings or just whitespace
#         if not location or location.isspace():
#             return None
            
#         # Replace multiple spaces with a single space
#         location = re.sub(r'\s+', ' ', location)
        
#         # Add commas between city/state/country if missing
#         parts = re.split(r'\s{2,}', location)
#         if len(parts) > 1:
#             location = ", ".join(parts)
            
#         return location
    
#     def get_coordinates(location, attempt_count=0):
#         """Get coordinates with fallback mechanisms and retry logic"""
#         # First check for None, empty or whitespace-only strings
#         if pd.isna(location) or not location or str(location).isspace():
#             return None, None
            
#         # Clean the location string
#         original_loc = str(location).strip()
#         cleaned_loc = clean_location(location)
#         if not cleaned_loc:
#             return None, None
            
#         # Check if this is a known problematic location with manual coordinates
#         if original_loc in known_locations:
#             return known_locations[original_loc]
            
#         # Check cache for exact match
#         if cleaned_loc in location_cache:
#             return location_cache[cleaned_loc]
        
#         # First attempt with cleaned location
#         if attempt_count == 0:
#             try:
#                 loc = geocode(cleaned_loc)
#                 if loc:
#                     coords = (round(loc.latitude, 3), round(loc.longitude, 3))
#                     location_cache[cleaned_loc] = coords
#                     return coords
#             except Exception as e:
#                 print(f"Error geocoding '{cleaned_loc}': {e}")
#                 time.sleep(1)
        
#         # Second attempt: try alternative formats (city, country)
#         if attempt_count <= 1:
#             try:
#                 parts = cleaned_loc.split(',')
#                 if len(parts) > 1:
#                     # Keep first and last parts (typically city and country)
#                     simplified = f"{parts[0].strip()}, {parts[-1].strip()}"
#                     loc = geocode(simplified)
#                     if loc:
#                         coords = (round(loc.latitude, 3), round(loc.longitude, 3))
#                         location_cache[cleaned_loc] = coords
#                         return coords
#             except Exception as e:
#                 print(f"Second attempt error geocoding '{cleaned_loc}': {e}")
#                 time.sleep(1)
        
#         # Both attempts failed
#         return None, None
    
#     # Prepare result DataFrame - properly initialize columns
#     # This uses proper DataFrame assignment instead of direct assignment
#     if 'lat' not in users_df.columns:
#         users_df.loc[:, 'lat'] = np.nan
#     if 'lng' not in users_df.columns:
#         users_df.loc[:, 'lng'] = np.nan
    
#     # Process in smaller batches to save progress regularly
#     BATCH_SIZE = 100
#     total_batches = (len(users_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
#     # Skip already processed indices
#     pending_indices = [i for i in range(len(users_df)) if i not in completed_indices]
    
#     for batch_idx, batch_start in enumerate(range(0, len(pending_indices), BATCH_SIZE)):
#         batch_end = min(batch_start + BATCH_SIZE, len(pending_indices))
#         batch_indices = pending_indices[batch_start:batch_end]
        
#         print(f"\nProcessing batch {batch_idx+1}/{total_batches} ({len(batch_indices)} locations)")
#         batch_start_time = time.time()
#         success_count = 0
        
#         for idx in batch_indices:
#             location = users_df.iloc[idx]['location']
            
#             try:
#                 lat, lng = get_coordinates(location)
#                 if lat is not None and lng is not None:
#                     # Proper way to set values using loc to avoid SettingWithCopyWarning
#                     users_df.loc[idx, 'lat'] = lat
#                     users_df.loc[idx, 'lng'] = lng
#                     success_count += 1
#                 completed_indices.append(idx)
#             except Exception as e:
#                 print(f"Unexpected error processing row {idx} '{location}': {e}")
            
#             # Save checkpoint every 20 locations
#             if len(completed_indices) % 20 == 0:
#                 checkpoint_data = {
#                     'location_cache': location_cache,
#                     'completed_indices': completed_indices
#                 }
#                 pd.to_pickle(checkpoint_data, checkpoint_file)
        
#         # Save batch checkpoint
#         checkpoint_data = {
#             'location_cache': location_cache,
#             'completed_indices': completed_indices
#         }
#         pd.to_pickle(checkpoint_data, checkpoint_file)
        
#         batch_time = time.time() - batch_start_time
#         print(f"Batch {batch_idx+1} completed: {success_count}/{len(batch_indices)} successful ({batch_time:.1f}s)")
#         print(f"Overall progress: {len(completed_indices)}/{len(users_df)} rows processed ({len(completed_indices)/len(users_df)*100:.1f}%)")
        
#         # Estimate remaining time
#         if batch_idx > 0 and success_count > 0:
#             rows_left = len(users_df) - len(completed_indices)
#             time_per_row = batch_time / len(batch_indices)
#             est_time_left = rows_left * time_per_row
#             est_hours = int(est_time_left // 3600)
#             est_minutes = int((est_time_left % 3600) // 60)
#             print(f"Estimated time remaining: {est_hours}h {est_minutes}m")
    
#     # Final report
#     success_count = users_df['lat'].notna().sum()
#     total_time = (datetime.now() - start_time).total_seconds()
#     hours = int(total_time // 3600)
#     minutes = int((total_time % 3600) // 60)
#     seconds = int(total_time % 60)
    
#     print(f"\nGeocoding completed in {hours}h {minutes}m {seconds}s")
#     print(f"Successfully geocoded {success_count} of {len(users_df)} locations ({success_count/len(users_df)*100:.1f}%)")
    
#     # Create a report of failed locations
#     failed_df = users_df[users_df['lat'].isna()].copy()
#     if not failed_df.empty:
#         print(f"\nFailed to geocode {len(failed_df)} locations")
#         failed_df.to_csv('failed_geocodes.csv', index=False)
#         print("Failed locations saved to 'failed_geocodes.csv'")
        
#         # Show sample of failed locations
#         sample_size = min(10, len(failed_df))
#         print(f"\nSample of failed locations:")
#         for loc in failed_df['location'].head(sample_size).tolist():
#             print(f"  - '{loc}'")
    
#     return users_df
# if__name__==__main__()
# users_df = add_user_coordinates_scalable(users[["location"]])


import pandas as pd
import numpy as np
import os
import time
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime


def add_user_coordinates(users_df, checkpoint_file='geocoding_progress.pkl'):
    """Adds lat/lng to users based on location, with caching and retries."""
    users_df = users_df.copy()
    start_time = datetime.now()
    print(f"Starting geocoding at {start_time:%H:%M:%S} for {len(users_df)} locations")

    # Load cache
    location_cache, completed_indices = load_checkpoint(checkpoint_file)

    known_locations = {
        "Djokja Yogyakarta Indonesia": (-7.797, 110.370),
        "Santo Domingo  Dominican Republic": (18.486, -69.932),
    }

    geolocator = Nominatim(user_agent="geopy_user_locations_batch")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)

    def clean_location(location):
        if not isinstance(location, str) or not location.strip():
            return None
        location = re.sub(r'\s+', ' ', location).strip()
        parts = re.split(r'\s{2,}', location)
        return ", ".join(parts) if len(parts) > 1 else location

    def get_coordinates(location, attempt_count=0):
        if not isinstance(location, str) or not location.strip():
            return None, None

        cleaned_loc = clean_location(location)
        if not cleaned_loc:
            return None, None

        if location in known_locations:  # Original location check
            return known_locations[location]

        if cleaned_loc in location_cache:
            return location_cache[cleaned_loc]

        try:
            loc = geocode(cleaned_loc)
            if loc:
                coords = (round(loc.latitude, 3), round(loc.longitude, 3))
                location_cache[cleaned_loc] = coords
                return coords
        except Exception as e:
            print(f"Error geocoding '{cleaned_loc}': {e}")
            time.sleep(1)

        if attempt_count <= 1:
            try:
                parts = cleaned_loc.split(',')
                if len(parts) > 1:
                    simplified = f"{parts[0].strip()}, {parts[-1].strip()}"
                    loc = geocode(simplified)
                    if loc:
                        coords = (round(loc.latitude, 3), round(loc.longitude, 3))
                        location_cache[cleaned_loc] = coords
                        return coords
            except Exception as e:
                print(f"Second attempt error geocoding '{cleaned_loc}': {e}")
                time.sleep(1)
        return None, None

    users_df[['lat', 'lng']] = np.nan  # Efficiently initialize columns

    BATCH_SIZE = 100
    total_batches = (len(users_df) + BATCH_SIZE - 1) // BATCH_SIZE
    pending_indices = [i for i in range(len(users_df)) if i not in completed_indices]

    for batch_idx, batch_start in enumerate(range(0, len(pending_indices), BATCH_SIZE)):
        batch_end = min(batch_start + BATCH_SIZE, len(pending_indices))
        batch_indices = pending_indices[batch_start:batch_end]

        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_indices)} locations)")
        batch_start_time = time.time()
        success_count = 0

        for idx in batch_indices:
            location = users_df.loc[idx, 'location']  # Access location using .loc
            try:
                lat, lng = get_coordinates(location)
                if lat is not None and lng is not None:
                    users_df.loc[idx, 'lat'] = lat
                    users_df.loc[idx, 'lng'] = lng
                    success_count += 1
                completed_indices.append(idx)  # Track processed indices
            except Exception as e:
                print(f"Unexpected error processing row {idx} '{location}': {e}")

        save_checkpoint(location_cache, completed_indices, checkpoint_file)  # Save at end of each batch

        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx + 1} completed: {success_count}/{len(batch_indices)} successful ({batch_time:.1f}s)")
        print(
            f"Overall progress: {len(completed_indices)}/{len(users_df)} rows processed ({len(completed_indices) / len(users_df) * 100:.1f}%)"
        )
        estimate_time_remaining(len(users_df), len(completed_indices), batch_time, len(batch_indices))

    # Final reporting and failed location handling
    finalize_geocoding(users_df, start_time)
    return users_df

def load_checkpoint(checkpoint_file):
    """Loads location cache and completed indices from a checkpoint file."""
    try:
        if os.path.exists(checkpoint_file):
            cache_data = pd.read_pickle(checkpoint_file)
            location_cache = cache_data.get('location_cache', {})
            completed_indices = cache_data.get('completed_indices', [])
            print(f"Loaded cache: {len(location_cache)} locations, {len(completed_indices)} completed indices")
        else:
            location_cache = {}
            completed_indices = []
        return location_cache, completed_indices
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        return {}, []

def save_checkpoint(location_cache, completed_indices, checkpoint_file):
    """Saves location cache and completed indices to a file."""
    checkpoint_data = {'location_cache': location_cache, 'completed_indices': completed_indices}
    try:
        pd.to_pickle(checkpoint_data, checkpoint_file)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def estimate_time_remaining(total_rows, completed_rows, batch_time, batch_size):
    """Estimates and prints the remaining time for geocoding."""
    if completed_rows > 0 and batch_size > 0:
        rows_left = total_rows - completed_rows
        time_per_row = batch_time / batch_size
        est_time_left = rows_left * time_per_row
        est_hours = int(est_time_left // 3600)
        est_minutes = int((est_time_left % 3600) // 60)
        print(f"Estimated time remaining: {est_hours}h {est_minutes}m")

def finalize_geocoding(users_df, start_time):
    """Final reporting, failed location handling, and saves failed geocodes."""
    success_count = users_df['lat'].notna().sum()
    total_time = (datetime.now() - start_time).total_seconds()
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nGeocoding completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Successfully geocoded {success_count} of {len(users_df)} locations ({success_count / len(users_df) * 100:.1f}%)")

    failed_df = users_df[users_df['lat'].isna()].copy()
    if not failed_df.empty:
        print(f"\nFailed to geocode {len(failed_df)} locations")
        failed_df.to_csv('failed_geocodes.csv', index=False)
        print("Failed locations saved to 'failed_geocodes.csv'")
        sample_size = min(10, len(failed_df))
        print(f"\nSample of failed locations:")
        for loc in failed_df['location'].head(sample_size).tolist():
            print(f"  - '{loc}'")

#if __name__ == "__main__":
    # # Example Usage (replace with your actual DataFrame loading)
    # data = {'location': ['Medan Indonesia', 'Djokja Yogyakarta Indonesia', 'Santo Domingo  Dominican Republic', 'Some Unknown Place']}
    # users = pd.DataFrame(data)
    # users_with_coords = add_user_coordinates(users)
    # print(users_with_coords)
