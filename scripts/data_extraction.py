"""
data_extraction.py
==================
Fetches movie data from TMDb API including basic details and credits.

Usage:
    python data_extraction.py
    
Output:
    - data/raw/movies_raw.json (raw API responses)
    - data/raw/credits_raw.json (cast and crew data)
"""

import pandas as pd
import requests
import json
import os
import importlib.util
import time
from datetime import datetime
# from new_config import TMDB_API_KEY   <--- Facing issues in the import
# Loading new_config dynamically
config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'new_config.py')
)

spec = importlib.util.spec_from_file_location("new_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

TMDB_API_KEY = config.TMDB_API_KEY

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "https://api.themoviedb.org/3"
RATE_LIMIT_DELAY = 0.25  # Seconds between requests

# Movie IDs to fetch
MOVIE_IDS = [
    0, 299534, 19995, 140607, 299536, 597, 135397, 420818,
    24428, 168259, 99861, 284054, 12445, 181808, 330457,
    351286, 109445, 321612, 260513
]

# Output directories
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_movie_details(movie_id):
    """
    Fetch basic movie details from TMDb API.
    
    Args:
        movie_id (int): TMDb movie ID
        
    Returns:
        dict: Movie data or None if request fails
    """
    url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ✗ Error {response.status_code} for movie ID {movie_id}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed for movie ID {movie_id}: {e}")
        return None


def fetch_movie_credits(movie_id):
    """
    Fetch cast and crew information from TMDb API.
    
    Args:
        movie_id (int): TMDb movie ID
        
    Returns:
        dict: Credits data or None if request fails
    """
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ✗ Error {response.status_code} for credits of movie ID {movie_id}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed for credits of movie ID {movie_id}: {e}")
        return None


def save_to_json(data, filename):
    """
    Save data to JSON file.
    
    Args:
        data (list or dict): Data to save
        filename (str): Output filename
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved to {filepath}")


# ============================================================================
# MAIN EXTRACTION LOGIC
# ============================================================================

def extract_all_movies(movie_ids):
    """
    Fetch movie details and credits for all movie IDs.
    
    Args:
        movie_ids (list): List of TMDb movie IDs
        
    Returns:
        tuple: (movies_data, credits_data)
    """
    print("="*70)
    print("TMDb DATA EXTRACTION")
    print("="*70)
    print(f"Starting extraction for {len(movie_ids)} movies...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    movies_data = []
    credits_data = []
    failed_ids = []
    
    # Fetch movie details
    print("PHASE 1: Fetching Movie Details")
    print("-" * 70)
    
    for idx, movie_id in enumerate(movie_ids, 1):
        print(f"[{idx}/{len(movie_ids)}] Fetching movie ID: {movie_id}...")
        
        # Fetch basic details
        movie_info = fetch_movie_details(movie_id)
        
        if movie_info:
            movies_data.append(movie_info)
            print(f"  ✓ Got: {movie_info.get('title', 'Unknown Title')}")
        else:
            failed_ids.append(movie_id)
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Fetch credits
    print("\n" + "="*70)
    print("PHASE 2: Fetching Cast & Crew Data")
    print("-" * 70)
    
    for idx, movie_info in enumerate(movies_data, 1):
        movie_id = movie_info['id']
        print(f"[{idx}/{len(movies_data)}] Fetching credits for: {movie_info['title']}...")
        
        credits = fetch_movie_credits(movie_id)
        
        if credits:
            # Add movie_id to credits for easy merging later
            credits['movie_id'] = movie_id
            credits_data.append(credits)
            print(f"  ✓ Got {len(credits.get('cast', []))} cast, {len(credits.get('crew', []))} crew")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"✓ Successfully fetched: {len(movies_data)} movies")
    print(f"✓ Successfully fetched: {len(credits_data)} credits")
    print(f"✗ Failed fetches: {len(failed_ids)} movies")
    
    if failed_ids:
        print(f"  Failed IDs: {failed_ids}")
    
    return movies_data, credits_data


def main():
    """Main execution function."""
    
    # Test API connection first
    print("Testing API connection...")
    test_movie = fetch_movie_details(299534)  # Avengers: Endgame
    
    if test_movie:
        print(f"✓ API connection successful!")
        print(f"  Test movie: {test_movie.get('title')}\n")
    else:
        print("✗ API connection failed. Check your API key in new_config.py")
        return
    
    # Extract all movies
    movies_data, credits_data = extract_all_movies(MOVIE_IDS)
    
    # Save raw data
    if movies_data:
        print("\nSaving raw data...")
        save_to_json(movies_data, 'movies_raw.json')
        save_to_json(credits_data, 'credits_raw.json')
        
        # Also save as DataFrame for quick inspection
        movies_df = pd.DataFrame(movies_data)
        movies_df.to_csv(os.path.join(RAW_DATA_DIR, 'movies_raw.csv'), index=False)
        print(f"✓ Saved CSV to {os.path.join(RAW_DATA_DIR, 'movies_raw.csv')}")
        
        print("\n" + "="*70)
        print("DATA EXTRACTION COMPLETE!")
        print("="*70)
        print(f"Total movies extracted: {len(movies_data)}")
        print(f"Files saved in: {RAW_DATA_DIR}/")
    else:
        print("\n✗ No data was extracted. Please check errors above.")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()