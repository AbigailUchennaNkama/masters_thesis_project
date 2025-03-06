import pandas as pd
import numpy as np
import os
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime

def add_user_coordinates(users_df, checkpoint_file='geocoding_progress.pkl'):
    """
    Add latitude and longitude to users based on their existing location data
    Optimized for large datasets with checkpointing and batch processing
    """
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    users_df = users_df.copy()
    
    # Reset the checkpoint file if it exists
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    start_time = datetime.now()
    print(f"Starting geocoding process at {start_time.strftime('%H:%M:%S')} for {len(users_df)} locations")
    
    # Initialize location cache and completed indices
    location_cache = {}
    completed_indices = []
    
    # Initialize geocoder with rate limiting
    geolocator = Nominatim(user_agent=f"heartpath_geocoding_{int(time.time())}")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
    
    def get_coordinates(location, attempt_count=0):
        """Get coordinates with fallback mechanisms and retry logic"""
        # First check for None, empty or whitespace-only strings
        if pd.isna(location) or not location or str(location).isspace():
            return None, None
        
        # First attempt with original location
        try:
            loc = geocode(location)
            if loc:
                coords = (round(loc.latitude, 3), round(loc.longitude, 3))
                return coords
        except Exception as e:
            print(f"Error geocoding '{location}': {e}")
            time.sleep(1)
        
        # Second attempt: try alternative formats (city, country)
        try:
            parts = location.split(',')
            if len(parts) > 1:
                # Keep first and last parts (typically city and country)
                simplified = f"{parts[0].strip()}, {parts[-1].strip()}"
                loc = geocode(simplified)
                if loc:
                    coords = (round(loc.latitude, 3), round(loc.longitude, 3))
                    return coords
        except Exception as e:
            print(f"Second attempt error geocoding '{location}': {e}")
            time.sleep(1)
        
        # Both attempts failed
        return None, None
    
    # Prepare result DataFrame
    users_df['lat'] = np.nan
    users_df['lng'] = np.nan
    
    # Process in smaller batches to save progress regularly
    BATCH_SIZE = 100
    total_batches = (len(users_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(users_df))
        batch = users_df.iloc[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx+1}/{total_batches} ({len(batch)} locations)")
        batch_start_time = time.time()
        success_count = 0
        
        for idx, row in batch.iterrows():
            location = row['location']
            
            try:
                # Attempt to get coordinates
                coords = get_coordinates(location)
                
                if coords[0] is not None and coords[1] is not None:
                    # Set coordinates
                    users_df.loc[idx, 'lat'] = coords[0]
                    users_df.loc[idx, 'lng'] = coords[1]
                    success_count += 1
                
            except Exception as e:
                print(f"Unexpected error processing row {idx} '{location}': {e}")
        
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx+1} completed: {success_count}/{len(batch)} successful ({batch_time:.1f}s)")
    
    # Final report
    success_count = users_df['lat'].notna().sum()
    total_time = (datetime.now() - start_time).total_seconds()
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\nGeocoding completed in {hours}h {minutes}m {seconds}s")
    print(f"Successfully geocoded {success_count} of {len(users_df)} locations ({success_count/len(users_df)*100:.1f}%)")
    
    # Create a report of failed locations
    failed_df = users_df[users_df['lat'].isna()].copy()
    if not failed_df.empty:
        print(f"\nFailed to geocode {len(failed_df)} locations")
        failed_df.to_csv('failed_geocodes.csv', index=False)
        print("Failed locations saved to 'failed_geocodes.csv'")
        
        # Show sample of failed locations
        sample_size = min(10, len(failed_df))
        print(f"\nSample of failed locations:")
        for loc in failed_df['location'].head(sample_size).tolist():
            print(f"  - '{loc}'")
    
    return users_df