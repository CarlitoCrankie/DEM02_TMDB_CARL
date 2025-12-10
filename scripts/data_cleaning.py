"""
data_cleaning.py
================
Cleans and transforms raw TMDb movie data into analysis-ready format.

This script performs:
- JSON field extraction (genres, companies, languages, etc.)
- Data type conversions
- Handling unrealistic values (zeros, placeholders)
- Fetching cast and director information
- Data quality filtering
- Final dataset preparation

Usage:
    python data_cleaning.py
    
Input:
    - data/raw/movies_raw.json
    - data/raw/credits_raw.json (optional, will fetch if missing)
    
Output:
    - data/processed/movies_cleaned.csv
    - data/processed/movies_cleaned.json
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime
import importlib.util
# from new_config import TMDB_API_KEY

# Load new_config dynamically
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
RATE_LIMIT_DELAY = 0.1  # Seconds between API requests

# Input/Output paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Columns to drop (not needed for analysis)
COLUMNS_TO_DROP = ['original_title', 'adult', 'imdb_id', 'video', 'homepage']

# JSON columns that need extraction
JSON_COLUMNS = [
    'belongs_to_collection',
    'genres',
    'production_companies',
    'production_countries',
    'spoken_languages'
]

# Placeholder values to replace with NaN
PLACEHOLDER_VALUES = ['No Data', 'N/A', 'Unknown', 'None', 'n/a', '', ' ']


# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================

def load_raw_data():
    """
    Load raw movie data from JSON file.
    
    Returns:
        list: List of movie dictionaries from TMDb API
    """
    print("="*70)
    print("STEP 1: LOADING RAW DATA")
    print("="*70)
    
    raw_file = os.path.join(RAW_DATA_DIR, 'movies_raw.json')
    
    if not os.path.exists(raw_file):
        print(f"✗ Error: {raw_file} not found!")
        print("  Please run data_extraction.py first.")
        return None
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✓ Loaded {len(raw_data)} movies from {raw_file}")
    return raw_data


# ============================================================================
# STEP 2: EXAMINE JSON COLUMNS
# ============================================================================

def examine_json_columns(raw_data):
    """
    Inspect the structure of nested JSON columns.
    
    Args:
        raw_data (list): Raw movie data
    """
    print("\n" + "="*70)
    print("STEP 2: EXAMINING JSON COLUMN STRUCTURE")
    print("="*70)
    
    if not raw_data:
        print("No data to examine.")
        return
    
    # Use first valid movie (skip if ID 0 failed)
    sample_movie = raw_data[0] if raw_data[0].get('id') != 0 else raw_data[1]
    
    print(f"\nSample movie: {sample_movie.get('title', 'Unknown')}")
    print("-" * 70)
    
    for col in JSON_COLUMNS:
        print(f"\n{col.upper()}:")
        sample_data = sample_movie.get(col)
        
        if sample_data:
            print(f"  Type: {type(sample_data).__name__}")
            print(f"  Sample: {sample_data}")
        else:
            print("  No data available")


# ============================================================================
# STEP 3: EXTRACTION FUNCTIONS
# ============================================================================

def extract_collection_name(collection_data):
    """
    Extract collection name from belongs_to_collection field.
    
    Args:
        collection_data (dict or None): Collection dictionary from API
        
    Returns:
        str or None: Collection name
        
    Example:
        Input:  {'id': 87096, 'name': 'Avatar Collection', ...}
        Output: 'Avatar Collection'
    """
    if collection_data and isinstance(collection_data, dict):
        return collection_data.get('name')
    return None


def extract_names_pipe_separated(data_list):
    """
    Extract 'name' field from list of dicts and join with pipe separator.
    Also filters out empty strings and whitespace.
    
    Args:
        data_list (list or None): List of dictionaries with 'name' key
        
    Returns:
        str or None: Pipe-separated names (e.g., "Action|Adventure")
        
    Example:
        Input:  [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]
        Output: 'Action|Adventure'
    """
    if not data_list or not isinstance(data_list, list):
        return None
    
    # Filter out empty/blank names and strip whitespace
    names = [
        item.get('name').strip()
        for item in data_list 
        if 'name' in item and item.get('name') and item.get('name').strip()
    ]
    
    return '|'.join(names) if names else None


def test_extraction_functions(raw_data):
    """
    Test extraction functions on sample data.
    
    Args:
        raw_data (list): Raw movie data
    """
    print("\n" + "="*70)
    print("STEP 3: TESTING EXTRACTION FUNCTIONS")
    print("="*70)
    
    if not raw_data:
        print("No data to test.")
        return
    
    # Get test movie (skip ID 0 if it failed)
    test_movie = raw_data[0] if raw_data[0].get('id') != 0 else raw_data[1]
    
    print(f"\nTest movie: {test_movie.get('title', 'Unknown')}")
    print("-" * 70)
    
    # Test collection extraction
    print("\n1. COLLECTION:")
    original = test_movie.get('belongs_to_collection')
    extracted = extract_collection_name(original)
    print(f"  Original:  {original}")
    print(f"  Extracted: {extracted}")
    
    # Test genres extraction
    print("\n2. GENRES:")
    original = test_movie.get('genres')
    extracted = extract_names_pipe_separated(original)
    print(f"  Original:  {original}")
    print(f"  Extracted: {extracted}")
    
    # Test production companies extraction
    print("\n3. PRODUCTION COMPANIES:")
    original = test_movie.get('production_companies')
    extracted = extract_names_pipe_separated(original)
    print(f"  Original:  {original}")
    print(f"  Extracted: {extracted}")


# ============================================================================
# STEP 4: CREATE CLEANED DATAFRAME
# ============================================================================

def create_cleaned_dataframe(raw_data):
    """
    Transform raw JSON data into structured DataFrame with extracted fields.
    
    Args:
        raw_data (list): Raw movie data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with extracted fields
    """
    print("\n" + "="*70)
    print("STEP 4: CREATING CLEANED DATAFRAME")
    print("="*70)
    
    movies_df_clean = pd.DataFrame([
        {
            'id': movie['id'],
            'title': movie['title'],
            'tagline': movie.get('tagline'),
            'overview': movie.get('overview'),
            'status': movie.get('status'),
            'release_date': movie['release_date'],
            'runtime': movie['runtime'],
            'budget': movie['budget'],
            'revenue': movie['revenue'],
            'vote_average': movie['vote_average'],
            'vote_count': movie['vote_count'],
            'popularity': movie['popularity'],
            'original_language': movie['original_language'],
            'poster_path': movie.get('poster_path'),
            
            # EXTRACTED FIELDS (pipe-separated)
            'collection': extract_collection_name(movie.get('belongs_to_collection')),
            'genres': extract_names_pipe_separated(movie.get('genres')),
            'spoken_languages': extract_names_pipe_separated(movie.get('spoken_languages')),
            'production_countries': extract_names_pipe_separated(movie.get('production_countries')),
            'production_companies': extract_names_pipe_separated(movie.get('production_companies')),
        }
        for movie in raw_data
    ])
    
    print(f"✓ Created DataFrame with {len(movies_df_clean)} rows and {len(movies_df_clean.columns)} columns")
    
    # Display sample
    print("\nSample extracted fields:")
    print(movies_df_clean[['title', 'collection', 'genres', 'spoken_languages']].head())
    
    return movies_df_clean


# ============================================================================
# STEP 5: INSPECT WITH VALUE_COUNTS
# ============================================================================

def inspect_data_quality(df):
    """
    Inspect extracted columns using value_counts to identify anomalies.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("STEP 5: DATA QUALITY INSPECTION (VALUE_COUNTS)")
    print("="*70)
    
    # 1. Collection inspection
    print("\n1. COLLECTION (Franchise):")
    print(df['collection'].value_counts(dropna=False))
    
    # 2. Genres (split and count)
    print("\n2. GENRES (individual):")
    print(df['genres'].str.split('|').explode().value_counts())
    
    # 3. Spoken languages
    print("\n3. SPOKEN LANGUAGES (individual):")
    print(df['spoken_languages'].str.split('|').explode().value_counts())
    
    # 4. Production countries
    print("\n4. PRODUCTION COUNTRIES (individual):")
    print(df['production_countries'].str.split('|').explode().value_counts())
    
    # 5. Production companies (top 15)
    print("\n5. PRODUCTION COMPANIES (top 15):")
    print(df['production_companies'].str.split('|').explode().value_counts().head(15))
    
    # Missing data summary
    print("\n" + "="*70)
    print("MISSING DATA SUMMARY:")
    print("="*70)
    missing = df[['collection', 'genres', 'spoken_languages', 
                  'production_countries', 'production_companies']].isnull().sum()
    print(missing)


# ============================================================================
# STEP 6: CONVERT DATA TYPES
# ============================================================================

def convert_data_types(df):
    """
    Convert columns to appropriate data types.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        
    Returns:
        pd.DataFrame: DataFrame with converted types
    """
    print("\n" + "="*70)
    print("STEP 6: CONVERTING DATA TYPES")
    print("="*70)
    
    df = df.copy()
    
    # Numeric columns
    numeric_columns = ['id', 'runtime', 'budget', 'revenue', 
                      'vote_average', 'vote_count', 'popularity']
    
    print("\nConverting numeric columns...")
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  ✓ Converted '{col}' to numeric")
    
    # Datetime conversion
    print("\nConverting release_date to datetime...")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    print("  ✓ Converted 'release_date' to datetime")
    
    # Verify conversions
    print("\nVerified data types:")
    print(df[['title', 'budget', 'release_date', 'popularity']].dtypes)
    
    print("\nSample of converted data:")
    print(df[['title', 'budget', 'release_date', 'popularity']].head())
    
    return df


# ============================================================================
# STEP 7: REPLACE UNREALISTIC VALUES
# ============================================================================

def replace_unrealistic_values(df):
    """
    Replace zero values, convert to million USD, handle placeholder text.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("STEP 7: REPLACING UNREALISTIC VALUES")
    print("="*70)
    
    df = df.copy()
    
    # A) Replace zeros with NaN
    print("\n7A. Replacing zero values with NaN...")
    zero_replacement_cols = ['budget', 'revenue', 'runtime']
    
    for col in zero_replacement_cols:
        zero_count = (df[col] == 0).sum()
        df[col] = df[col].replace(0, pd.NA)
        print(f"  ✓ Replaced {zero_count} zeros in '{col}' with NaN")
    
    # B) Convert to million USD
    print("\n7B. Converting budget and revenue to million USD...")
    df['budget_musd'] = df['budget'] / 1_000_000
    df['revenue_musd'] = round(df['revenue'] / 1_000_000, 2)
    print("  ✓ Created 'budget_musd' and 'revenue_musd' columns")
    
    # Drop original budget and revenue
    df = df.drop(columns=['budget', 'revenue'])
    print("  ✓ Dropped original 'budget' and 'revenue' columns")
    
    # C) Handle vote_count == 0
    print("\n7C. Analyzing movies with vote_count == 0...")
    zero_votes = df[df['vote_count'] == 0]
    print(f"  Found {len(zero_votes)} movies with 0 votes")
    
    if len(zero_votes) > 0:
        print("  These movies:")
        print(zero_votes[['title', 'vote_count', 'vote_average']])
        
        # Set vote_average to NaN where vote_count is 0
        df.loc[df['vote_count'] == 0, 'vote_average'] = pd.NA
        print("  ✓ Set vote_average to NaN for movies with 0 votes")
    
    # D) Replace placeholder text
    print("\n7D. Replacing placeholder text with NaN...")
    placeholder_cols = ['overview', 'tagline']
    
    for col in placeholder_cols:
        if col in df.columns:
            # Replace empty strings and whitespace
            df[col] = df[col].replace('', pd.NA)
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('', pd.NA)
            
            # Replace known placeholders (case-insensitive)
            for placeholder in PLACEHOLDER_VALUES:
                mask = df[col].str.lower() == placeholder.lower()
                df.loc[mask, col] = pd.NA
            
            print(f"  ✓ Cleaned '{col}'")
    
    # Show results
    print("\nData after unrealistic value replacement:")
    print(df[['title', 'budget_musd', 'revenue_musd', 'runtime', 
              'vote_count', 'vote_average']].head(10))
    
    return df


# ============================================================================
# STEP 8: REMOVE DUPLICATES AND INVALID ROWS
# ============================================================================

def remove_duplicates_and_invalid(df):
    """
    Remove duplicate movies and rows with missing critical fields.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("STEP 8: REMOVING DUPLICATES AND INVALID ROWS")
    print("="*70)
    
    df = df.copy()
    
    # Check for duplicates
    print(f"\nRows before duplicate removal: {len(df)}")
    duplicates = df.duplicated(subset=['id'], keep='first').sum()
    print(f"Duplicate IDs found: {duplicates}")
    
    # Remove duplicates based on ID
    df = df.drop_duplicates(subset=['id'], keep='first')
    print(f"Rows after duplicate removal: {len(df)}")
    
    # Drop rows with missing ID or title
    print("\nDropping rows with missing ID or title...")
    before_drop = len(df)
    df = df.dropna(subset=['id', 'title'])
    after_drop = len(df)
    print(f"  Rows before: {before_drop}")
    print(f"  Rows after:  {after_drop}")
    print(f"  Dropped:     {before_drop - after_drop} rows")
    
    return df


# ============================================================================
# STEP 9: FILTER BY DATA COMPLETENESS
# ============================================================================

def filter_by_completeness(df, min_non_nan=10):
    """
    Keep only rows with sufficient non-NaN values.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        min_non_nan (int): Minimum number of non-NaN columns required
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print("\n" + "="*70)
    print("STEP 9: FILTERING BY DATA COMPLETENESS")
    print("="*70)
    
    df = df.copy()
    
    # Count non-NaN values per row
    non_nan_counts = df.notna().sum(axis=1)
    
    print(f"\nDistribution of non-NaN values per row:")
    print(non_nan_counts.value_counts().sort_index())
    
    # Filter
    print(f"\nKeeping rows with at least {min_non_nan} non-NaN values...")
    before_filter = len(df)
    df = df[non_nan_counts >= min_non_nan]
    after_filter = len(df)
    
    print(f"  Rows before: {before_filter}")
    print(f"  Rows after:  {after_filter}")
    print(f"  Removed:     {before_filter - after_filter} rows")
    
    return df


# ============================================================================
# STEP 10: FILTER FOR RELEASED MOVIES
# ============================================================================

def filter_released_movies(df):
    """
    Keep only movies with status='Released' and drop the status column.
    
    Args:
        df (pd.DataFrame): DataFrame to filter
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print("\n" + "="*70)
    print("STEP 10: FILTERING FOR RELEASED MOVIES")
    print("="*70)
    
    df = df.copy()
    
    if 'status' not in df.columns:
        print("  'status' column not found - skipping this step")
        return df
    
    # Show status distribution
    print("\nStatus value counts:")
    print(df['status'].value_counts())
    
    # Filter for Released movies only
    before_filter = len(df)
    df = df[df['status'] == 'Released']
    after_filter = len(df)
    
    print(f"\n  Movies before filter: {before_filter}")
    print(f"  Released movies:      {after_filter}")
    print(f"  Removed:              {before_filter - after_filter}")
    
    # Drop status column
    df = df.drop(columns=['status'])
    print("  ✓ Dropped 'status' column")
    
    return df


# ============================================================================
# STEP 11: FETCH CAST AND DIRECTOR DATA
# ============================================================================

def fetch_credits(movie_id):
    """
    Fetch cast and crew data from TMDb API.
    
    Args:
        movie_id (int): TMDb movie ID
        
    Returns:
        dict or None: Credits data
    """
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def extract_director(crew_data):
    """
    Extract director's name from crew list.
    
    Args:
        crew_data (list): Crew list from credits
        
    Returns:
        str or None: Director's name
    """
    if not crew_data:
        return None
    
    directors = [person['name'] for person in crew_data if person.get('job') == 'Director']
    return directors[0] if directors else None


def extract_cast_names(cast_list, limit=5):
    """
    Extract top N cast member names.
    
    Args:
        cast_list (list): Cast list from credits
        limit (int): Number of cast members to extract
        
    Returns:
        str or None: Pipe-separated cast names
    """
    if not cast_list:
        return None
    
    cast_names = [person['name'] for person in cast_list[:limit]]
    return '|'.join(cast_names) if cast_names else None


def add_credits_data(df):
    """
    Fetch and add cast and director information to DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with movie IDs
        
    Returns:
        pd.DataFrame: DataFrame with added credits columns
    """
    print("\n" + "="*70)
    print("STEP 11: FETCHING CAST AND DIRECTOR DATA")
    print("="*70)
    
    print(f"\nFetching credits for {len(df)} movies...")
    
    credits_data = []
    
    for idx, row in df.iterrows():
        movie_id = row['id']
        title = row['title']
        
        print(f"  [{idx+1}/{len(df)}] Fetching credits for: {title}...")
        
        credits = fetch_credits(int(movie_id))
        
        if credits:
            director = extract_director(credits.get('crew', []))
            cast = extract_cast_names(credits.get('cast', []), limit=5)
            cast_size = len(credits.get('cast', []))
            crew_size = len(credits.get('crew', []))
            
            credits_data.append({
                'id': movie_id,
                'director': director,
                'cast': cast,
                'cast_size': cast_size,
                'crew_size': crew_size
            })
            
            print(f"    ✓ Director: {director}, Cast: {cast_size}, Crew: {crew_size}")
        else:
            print(f"    ✗ Failed to fetch credits")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Create credits DataFrame and merge
    credits_df = pd.DataFrame(credits_data)
    df = df.merge(credits_df, on='id', how='left')
    
    print("\n✓ Credits data added!")
    print("\nSample with credits:")
    print(df[['title', 'director', 'cast', 'cast_size']].head())
    
    return df


# ============================================================================
# STEP 12: REORDER COLUMNS AND FINALIZE
# ============================================================================

def reorder_and_finalize(df):
    """
    Reorder columns to desired structure and reset index.
    
    Args:
        df (pd.DataFrame): DataFrame to finalize
        
    Returns:
        pd.DataFrame: Final cleaned DataFrame
    """
    print("\n" + "="*70)
    print("STEP 12: REORDERING COLUMNS AND FINALIZING")
    print("="*70)
    
    df = df.copy()
    
    # Rename 'collection' back to 'belongs_to_collection'
    if 'collection' in df.columns:
        df = df.rename(columns={'collection': 'belongs_to_collection'})
        print("✓ Renamed 'collection' to 'belongs_to_collection'")
    
    # Desired column order
    desired_columns = [
        'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
        'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
        'production_countries', 'vote_count', 'vote_average', 'popularity',
        'runtime', 'overview', 'spoken_languages', 'poster_path',
        'cast', 'cast_size', 'director', 'crew_size'
    ]
    
    # Keep only existing columns
    existing_columns = [col for col in desired_columns if col in df.columns]
    missing_columns = [col for col in desired_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n⚠️  Missing columns: {missing_columns}")
    
    # Reorder and reset index
    df_final = df[existing_columns].copy()
    df_final = df_final.reset_index(drop=True)
    
    print("\n" + "="*70)
    print("FINAL CLEANED DATAFRAME")
    print("="*70)
    print(f"Shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    
    print("\nFirst few rows:")
    print(df_final.head())
    
    print("\nData types:")
    print(df_final.dtypes)
    
    print("\nMissing values:")
    print(df_final.isnull().sum())
    
    return df_final


# ============================================================================
# STEP 13: SAVE CLEANED DATA
# ============================================================================

def save_cleaned_data(df):
    """
    Save cleaned DataFrame to CSV and JSON formats.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
    """
    print("\n" + "="*70)
    print("STEP 13: SAVING CLEANED DATA")
    print("="*70)
    
    # Save as CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'movies_cleaned.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(PROCESSED_DATA_DIR, 'movies_cleaned.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    print(f"\nCleaned data saved in: {PROCESSED_DATA_DIR}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    print("="*70)
    print("TMDb DATA CLEANING AND TRANSFORMATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Load raw data
    raw_data = load_raw_data()
    if not raw_data:
        print("✗ Failed to load raw data. Exiting.")
        return
    
    # Step 2: Examine JSON columns (optional inspection)
    examine_json_columns(raw_data)
    
    # Step 3: Test extraction functions
    test_extraction_functions(raw_data)
    
    # Step 4: Create cleaned DataFrame
    df = create_cleaned_dataframe(raw_data)
    
    # Step 5: Inspect data quality
    inspect_data_quality(df)
    
    # Step 6: Convert data types
    df = convert_data_types(df)
    
    # Step 7: Replace unrealistic values
    df = replace_unrealistic_values(df)
    
    # Step 8: Remove duplicates and invalid rows
    df = remove_duplicates_and_invalid(df)
    
    # Step 9: Filter by data completeness
    df = filter_by_completeness(df, min_non_nan=10)
    
    # Step 10: Filter for released movies
    df = filter_released_movies(df)
    
    # Step 11: Add cast and director data
    df = add_credits_data(df)
    
    # Step 12: Reorder and finalize
    df_final = reorder_and_finalize(df)
    
    # Step 13: Save cleaned data
    save_cleaned_data(df_final)
    
    print("\n" + "="*70)
    print("DATA CLEANING COMPLETE!")
    print("="*70)
    print(f"Final dataset: {df_final.shape[0]} movies, {df_final.shape[1]} columns")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()