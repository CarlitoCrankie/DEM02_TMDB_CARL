"""
analysis.py
===========
Performs KPI calculations, rankings, and comparative analysis on cleaned movie data.

This script includes:
- KPI calculations (Profit, ROI)
- Best/Worst movie rankings across multiple metrics
- Advanced search queries
- Franchise vs. Standalone comparison
- Most successful franchises analysis
- Director performance analysis

Usage:
    python analysis.py
    
Input:
    - data/processed/movies_cleaned.csv
    
Output:
    - data/analysis/kpi_results.csv
    - data/analysis/rankings_summary.txt
    - Console output with all analysis results
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_DATA_DIR = "data/processed"
ANALYSIS_OUTPUT_DIR = "data/analysis"
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# KPI thresholds
MIN_BUDGET_FOR_ROI = 10  # Million USD
MIN_VOTES_FOR_RATING = 10  # Minimum votes for rating analysis


# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================

def load_cleaned_data():
    """
    Load the cleaned movie dataset.
    
    Returns:
        pd.DataFrame: Cleaned movie data
    """
    print("="*70)
    print("STEP 1: LOADING CLEANED DATA")
    print("="*70)
    
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'movies_cleaned.csv')
    
    if not os.path.exists(csv_path):
        print(f"✗ Error: {csv_path} not found!")
        print("  Please run data_cleaning.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Convert release_date back to datetime (CSV stores as string)
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    print(f"✓ Loaded {len(df)} movies from {csv_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


# ============================================================================
# STEP 2: CALCULATE KPIs (KEY PERFORMANCE INDICATORS)
# ============================================================================

def calculate_kpis(df):
    """
    Calculate Profit and ROI for all movies.
    
    Args:
        df (pd.DataFrame): Cleaned movie data
        
    Returns:
        pd.DataFrame: DataFrame with added KPI columns
    """
    print("\n" + "="*70)
    print("STEP 2: CALCULATING KPIs (KEY PERFORMANCE INDICATORS)")
    print("="*70)
    
    # Create a copy to avoid modifying original
    movies_kpi = df.copy()
    
    # 1. Calculate Profit (Revenue - Budget)
    print("\n1. Calculating Profit...")
    movies_kpi['profit_musd'] = movies_kpi['revenue_musd'] - movies_kpi['budget_musd']
    profit_count = movies_kpi['profit_musd'].notna().sum()
    print(f"   ✓ Profit calculated for {profit_count} movies")
    
    # 2. Calculate ROI (Return on Investment)
    print("\n2. Calculating ROI (Return on Investment)...")
    print(f"   Only for movies with budget >= ${MIN_BUDGET_FOR_ROI}M")
    
    # np.where: Vectorized if-else
    # Syntax: np.where(condition, value_if_true, value_if_false)
    movies_kpi['roi'] = np.where(
        movies_kpi['budget_musd'] >= MIN_BUDGET_FOR_ROI,  # Condition
        movies_kpi['profit_musd'] / movies_kpi['budget_musd'],  # If True: calculate ROI
        np.nan  # If False: set to NaN
    )
    
    roi_count = movies_kpi['roi'].notna().sum()
    print(f"   ✓ ROI calculated for {roi_count} movies with budget >= ${MIN_BUDGET_FOR_ROI}M")
    
    # 3. Display sample calculations
    print("\n3. Sample KPI Calculations:")
    print("-" * 70)
    sample_cols = ['title', 'budget_musd', 'revenue_musd', 'profit_musd', 'roi']
    print(movies_kpi[sample_cols].head(10))
    
    # Summary statistics
    print("\n4. KPI Summary Statistics:")
    print("-" * 70)
    print(f"Mean Profit:   ${movies_kpi['profit_musd'].mean():.2f}M")
    print(f"Median Profit: ${movies_kpi['profit_musd'].median():.2f}M")
    print(f"Mean ROI:      {movies_kpi['roi'].mean():.2f}x")
    print(f"Median ROI:    {movies_kpi['roi'].median():.2f}x")
    
    return movies_kpi


# ============================================================================
# STEP 3: RANKING FUNCTION (USER-DEFINED FUNCTION)
# ============================================================================

def rank_movies(df, metric, ascending=False, top_n=10, min_votes=None, min_budget=None):
    """
    Rank movies by a specific metric with optional filters.
    
    This is a reusable function that handles all ranking operations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The movies DataFrame
    metric : str
        Column name to rank by (e.g., 'revenue_musd', 'roi')
    ascending : bool
        True for lowest first (worst), False for highest first (best)
    top_n : int
        Number of top results to return
    min_votes : int, optional
        Minimum vote_count required (for rating-based rankings)
    min_budget : float, optional
        Minimum budget_musd required (for ROI rankings)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with ranked movies and relevant columns
        
    Example:
    --------
    >>> # Get top 5 highest revenue movies
    >>> top_revenue = rank_movies(df, 'revenue_musd', ascending=False, top_n=5)
    
    >>> # Get worst ROI movies with budget >= $10M
    >>> worst_roi = rank_movies(df, 'roi', ascending=True, min_budget=10)
    """
    
    # Create a copy to avoid modifying original
    filtered_df = df.copy()
    
    # Apply filters based on parameters
    if min_votes is not None:
        # Filter: Keep only movies with sufficient votes
        filtered_df = filtered_df[filtered_df['vote_count'] >= min_votes]
        print(f"   Filtered to {len(filtered_df)} movies with >= {min_votes} votes")
    
    if min_budget is not None:
        # Filter: Keep only movies with sufficient budget
        filtered_df = filtered_df[filtered_df['budget_musd'] >= min_budget]
        print(f"   Filtered to {len(filtered_df)} movies with budget >= ${min_budget}M")
    
    # Remove rows where the metric is NaN (can't rank missing values)
    filtered_df = filtered_df.dropna(subset=[metric])
    
    # Sort by the specified metric
    ranked = filtered_df.sort_values(by=metric, ascending=ascending)
    
    # Select relevant columns for display
    display_cols = [
        'title', 'release_date', 'budget_musd', 'revenue_musd',
        'profit_musd', 'roi', 'vote_average', 'vote_count',
        'popularity', 'director'
    ]
    
    # Keep only columns that exist in the DataFrame
    display_cols = [col for col in display_cols if col in ranked.columns]
    
    # Return top N rows
    return ranked[display_cols].head(top_n)


# ============================================================================
# STEP 4: TEST RANKING FUNCTION
# ============================================================================

def test_ranking_function(movies_kpi):
    """
    Test the ranking function with a simple query.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
    """
    print("\n" + "="*70)
    print("STEP 3: TESTING RANKING FUNCTION")
    print("="*70)
    
    print("\nTop 5 Highest Revenue Movies:")
    print("-" * 70)
    top_5_revenue = rank_movies(movies_kpi, 'revenue_musd', ascending=False, top_n=5)
    print(top_5_revenue[['title', 'revenue_musd', 'director']])


# ============================================================================
# STEP 5: COMPREHENSIVE MOVIE RANKINGS
# ============================================================================

def perform_all_rankings(movies_kpi):
    """
    Perform all 10 ranking analyses and display results.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        
    Returns:
        dict: Dictionary containing all ranking results
    """
    print("\n" + "="*70)
    print("STEP 4: COMPREHENSIVE MOVIE RANKINGS")
    print("="*70)
    
    rankings = {}
    
    # 1. HIGHEST REVENUE
    print("\n" + "="*70)
    print("1. HIGHEST REVENUE MOVIES")
    print("="*70)
    highest_revenue = rank_movies(movies_kpi, 'revenue_musd', ascending=False, top_n=10)
    print(highest_revenue[['title', 'revenue_musd', 'director']])
    rankings['highest_revenue'] = highest_revenue
    
    # 2. HIGHEST BUDGET
    print("\n" + "="*70)
    print("2. HIGHEST BUDGET MOVIES")
    print("="*70)
    highest_budget = rank_movies(movies_kpi, 'budget_musd', ascending=False, top_n=10)
    print(highest_budget[['title', 'budget_musd', 'director']])
    rankings['highest_budget'] = highest_budget
    
    # 3. HIGHEST PROFIT
    print("\n" + "="*70)
    print("3. HIGHEST PROFIT MOVIES (Revenue - Budget)")
    print("="*70)
    highest_profit = rank_movies(movies_kpi, 'profit_musd', ascending=False, top_n=10)
    print(highest_profit[['title', 'profit_musd', 'budget_musd', 'revenue_musd']])
    rankings['highest_profit'] = highest_profit
    
    # 4. LOWEST PROFIT (Biggest Flops)
    print("\n" + "="*70)
    print("4. LOWEST PROFIT MOVIES (Biggest Flops)")
    print("="*70)
    lowest_profit = rank_movies(movies_kpi, 'profit_musd', ascending=True, top_n=10)
    print(lowest_profit[['title', 'profit_musd', 'budget_musd', 'revenue_musd']])
    rankings['lowest_profit'] = lowest_profit
    
    # 5. HIGHEST ROI (Best Investment)
    print("\n" + "="*70)
    print("5. HIGHEST ROI MOVIES (Budget >= $10M)")
    print("="*70)
    print("ROI = Profit / Budget (e.g., ROI of 2.0 = $2 earned per $1 spent)")
    highest_roi = rank_movies(movies_kpi, 'roi', ascending=False, top_n=10, min_budget=10)
    print(highest_roi[['title', 'roi', 'budget_musd', 'revenue_musd']])
    rankings['highest_roi'] = highest_roi
    
    # 6. LOWEST ROI (Worst Investment)
    print("\n" + "="*70)
    print("6. LOWEST ROI MOVIES (Budget >= $10M)")
    print("="*70)
    lowest_roi = rank_movies(movies_kpi, 'roi', ascending=True, top_n=10, min_budget=10)
    print(lowest_roi[['title', 'roi', 'budget_musd', 'revenue_musd']])
    rankings['lowest_roi'] = lowest_roi
    
    # 7. MOST VOTED MOVIES
    print("\n" + "="*70)
    print("7. MOST VOTED MOVIES")
    print("="*70)
    most_voted = rank_movies(movies_kpi, 'vote_count', ascending=False, top_n=10)
    print(most_voted[['title', 'vote_count', 'vote_average']])
    rankings['most_voted'] = most_voted
    
    # 8. HIGHEST RATED MOVIES (min 10 votes)
    print("\n" + "="*70)
    print("8. HIGHEST RATED MOVIES (Minimum 10 votes)")
    print("="*70)
    highest_rated = rank_movies(movies_kpi, 'vote_average', ascending=False, 
                                 top_n=10, min_votes=MIN_VOTES_FOR_RATING)
    print(highest_rated[['title', 'vote_average', 'vote_count']])
    rankings['highest_rated'] = highest_rated
    
    # 9. LOWEST RATED MOVIES (min 10 votes)
    print("\n" + "="*70)
    print("9. LOWEST RATED MOVIES (Minimum 10 votes)")
    print("="*70)
    lowest_rated = rank_movies(movies_kpi, 'vote_average', ascending=True, 
                                top_n=10, min_votes=MIN_VOTES_FOR_RATING)
    print(lowest_rated[['title', 'vote_average', 'vote_count']])
    rankings['lowest_rated'] = lowest_rated
    
    # 10. MOST POPULAR MOVIES
    print("\n" + "="*70)
    print("10. MOST POPULAR MOVIES")
    print("="*70)
    most_popular = rank_movies(movies_kpi, 'popularity', ascending=False, top_n=10)
    print(most_popular[['title', 'popularity', 'vote_average']])
    rankings['most_popular'] = most_popular
    
    return rankings


# ============================================================================
# STEP 6: ADVANCED SEARCH QUERIES
# ============================================================================

def perform_advanced_searches(movies_kpi):
    """
    Execute complex multi-condition search queries.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        
    Returns:
        dict: Dictionary containing search results
    """
    print("\n" + "="*70)
    print("STEP 5: ADVANCED SEARCH QUERIES")
    print("="*70)
    
    searches = {}
    
    # SEARCH 1: Best-rated Science Fiction Action movies with Bruce Willis
    print("\n" + "="*70)
    print("SEARCH 1: Sci-Fi Action Movies Starring Bruce Willis")
    print("="*70)
    
    search1 = movies_kpi[
        # Condition 1: Genre contains "Science Fiction"
        (movies_kpi['genres'].str.contains('Science Fiction', case=False, na=False)) &
        # Condition 2: Genre contains "Action"
        (movies_kpi['genres'].str.contains('Action', case=False, na=False)) &
        # Condition 3: Cast contains "Bruce Willis"
        (movies_kpi['cast'].str.contains('Bruce Willis', case=False, na=False))
    ].sort_values('vote_average', ascending=False)
    
    if len(search1) > 0:
        print(search1[['title', 'vote_average', 'genres', 'cast', 'director']])
        searches['bruce_willis_scifi'] = search1
    else:
        print("⚠️  No movies found matching criteria")
        print("   (Note: Bruce Willis may not be in your dataset)")
    
    # SEARCH 2: Uma Thurman movies directed by Quentin Tarantino
    print("\n" + "="*70)
    print("SEARCH 2: Uma Thurman + Quentin Tarantino Collaborations")
    print("="*70)
    
    search2 = movies_kpi[
        # Condition 1: Cast contains "Uma Thurman"
        (movies_kpi['cast'].str.contains('Uma Thurman', case=False, na=False)) &
        # Condition 2: Director is "Quentin Tarantino"
        (movies_kpi['director'].str.contains('Quentin Tarantino', case=False, na=False))
    ].sort_values('runtime', ascending=True)
    
    if len(search2) > 0:
        print(search2[['title', 'runtime', 'vote_average', 'director', 'cast']])
        searches['tarantino_thurman'] = search2
    else:
        print("⚠️  No movies found matching criteria")
        print("   (Note: These collaborations may not be in your dataset)")
    
    return searches


# ============================================================================
# STEP 7: FRANCHISE VS. STANDALONE ANALYSIS
# ============================================================================

def analyze_franchise_vs_standalone(movies_kpi):
    """
    Compare performance metrics between franchise and standalone movies.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        
    Returns:
        pd.DataFrame: Comparison results
    """
    print("\n" + "="*70)
    print("STEP 6: FRANCHISE VS. STANDALONE MOVIE PERFORMANCE")
    print("="*70)
    
    # Create franchise indicator (boolean: True if part of collection)
    movies_kpi['is_franchise'] = movies_kpi['belongs_to_collection'].notna()
    
    # Create readable label for grouping
    movies_kpi['movie_type'] = movies_kpi['is_franchise'].map({
        True: 'Franchise',
        False: 'Standalone'
    })
    
    # Group by movie type and calculate aggregate metrics
    franchise_comparison = movies_kpi.groupby('movie_type').agg({
        'revenue_musd': ['mean', 'median', 'count'],
        'budget_musd': 'mean',
        'roi': 'median',
        'popularity': 'mean',
        'vote_average': 'mean'
    }).round(2)
    
    # Flatten multi-level column names
    # From: ('revenue_musd', 'mean') -> To: 'revenue_musd_mean'
    franchise_comparison.columns = ['_'.join(col).strip() for col in franchise_comparison.columns]
    
    # Rename for better readability
    franchise_comparison = franchise_comparison.rename(columns={
        'revenue_musd_mean': 'Mean Revenue (M USD)',
        'revenue_musd_median': 'Median Revenue (M USD)',
        'revenue_musd_count': 'Number of Movies',
        'budget_musd_mean': 'Mean Budget (M USD)',
        'roi_median': 'Median ROI',
        'popularity_mean': 'Mean Popularity',
        'vote_average_mean': 'Mean Rating'
    })
    
    print("\n📊 FRANCHISE VS. STANDALONE COMPARISON")
    print(franchise_comparison)
    
    # Calculate and display insights (percentage differences)
    if len(franchise_comparison) == 2:
        franchise_row = franchise_comparison.loc['Franchise']
        standalone_row = franchise_comparison.loc['Standalone']
        
        print("\n" + "="*70)
        print("INSIGHTS:")
        print("="*70)
        
        # Revenue difference
        revenue_diff = ((franchise_row['Mean Revenue (M USD)'] - standalone_row['Mean Revenue (M USD)']) 
                        / standalone_row['Mean Revenue (M USD)'] * 100)
        print(f"• Franchise movies earn {revenue_diff:.1f}% more revenue on average")
        
        # Budget difference
        budget_diff = ((franchise_row['Mean Budget (M USD)'] - standalone_row['Mean Budget (M USD)']) 
                       / standalone_row['Mean Budget (M USD)'] * 100)
        print(f"• Franchise movies have {budget_diff:.1f}% higher budgets on average")
        
        # ROI difference
        roi_diff = ((franchise_row['Median ROI'] - standalone_row['Median ROI']) 
                    / standalone_row['Median ROI'] * 100)
        direction = 'better' if roi_diff > 0 else 'worse'
        print(f"• Franchise movies have {abs(roi_diff):.1f}% {direction} median ROI")
    
    # Show detailed breakdown
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN:")
    print("="*70)
    
    print("\nFranchise Movies:")
    franchise_movies = movies_kpi[movies_kpi['is_franchise']][['title', 'belongs_to_collection', 'revenue_musd', 'roi']]
    print(franchise_movies)
    
    print("\nStandalone Movies:")
    standalone_movies = movies_kpi[~movies_kpi['is_franchise']][['title', 'revenue_musd', 'roi']]
    print(standalone_movies)
    
    return franchise_comparison


# ============================================================================
# STEP 8: MOST SUCCESSFUL FRANCHISES
# ============================================================================

def analyze_franchises(movies_kpi):
    """
    Analyze performance of movie franchises (collections).
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        
    Returns:
        pd.DataFrame: Franchise analysis results
    """
    print("\n" + "="*70)
    print("STEP 7: MOST SUCCESSFUL MOVIE FRANCHISES")
    print("="*70)
    
    # Filter only franchise movies (those belonging to a collection)
    franchise_movies = movies_kpi[movies_kpi['belongs_to_collection'].notna()].copy()
    
    if len(franchise_movies) == 0:
        print("No franchise movies found in dataset.")
        return None
    
    # Group by collection name
    franchise_analysis = franchise_movies.groupby('belongs_to_collection').agg({
        'title': 'count',  # Number of movies in franchise
        'budget_musd': ['sum', 'mean'],
        'revenue_musd': ['sum', 'mean'],
        'vote_average': 'mean',
        'roi': 'mean'
    }).round(2)
    
    # Flatten column names
    franchise_analysis.columns = ['_'.join(col).strip() for col in franchise_analysis.columns]
    
    # Rename for clarity
    franchise_analysis = franchise_analysis.rename(columns={
        'title_count': 'Number of Movies',
        'budget_musd_sum': 'Total Budget (M USD)',
        'budget_musd_mean': 'Mean Budget (M USD)',
        'revenue_musd_sum': 'Total Revenue (M USD)',
        'revenue_musd_mean': 'Mean Revenue (M USD)',
        'vote_average_mean': 'Mean Rating',
        'roi_mean': 'Mean ROI'
    })
    
    # Sort by total revenue (most successful first)
    franchise_analysis = franchise_analysis.sort_values('Total Revenue (M USD)', ascending=False)
    
    print("\n📊 FRANCHISE PERFORMANCE METRICS")
    print(franchise_analysis)
    
    # Highlight top franchise
    if len(franchise_analysis) > 0:
        top_franchise = franchise_analysis.index[0]
        top_revenue = franchise_analysis.iloc[0]['Total Revenue (M USD)']
        top_count = franchise_analysis.iloc[0]['Number of Movies']
        
        print("\n" + "="*70)
        print("🏆 TOP FRANCHISE:")
        print("="*70)
        print(f"Franchise: {top_franchise}")
        print(f"Total Revenue: ${top_revenue:,.2f} Million")
        print(f"Number of Movies: {int(top_count)}")
        print(f"Average Revenue per Movie: ${top_revenue/top_count:,.2f} Million")
    
    return franchise_analysis


# ============================================================================
# STEP 9: MOST SUCCESSFUL DIRECTORS
# ============================================================================

def analyze_directors(movies_kpi):
    """
    Analyze performance of directors across their filmography.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        
    Returns:
        pd.DataFrame: Director analysis results
    """
    print("\n" + "="*70)
    print("STEP 8: MOST SUCCESSFUL DIRECTORS")
    print("="*70)
    
    # Filter movies with director data
    director_movies = movies_kpi[movies_kpi['director'].notna()].copy()
    
    if len(director_movies) == 0:
        print("No director data found in dataset.")
        return None
    
    # Group by director
    director_analysis = director_movies.groupby('director').agg({
        'title': 'count',  # Number of movies directed
        'revenue_musd': ['sum', 'mean'],
        'budget_musd': 'mean',
        'vote_average': 'mean',
        'roi': 'mean'
    }).round(2)
    
    # Flatten column names
    director_analysis.columns = ['_'.join(col).strip() for col in director_analysis.columns]
    
    # Rename for clarity
    director_analysis = director_analysis.rename(columns={
        'title_count': 'Number of Movies',
        'revenue_musd_sum': 'Total Revenue (M USD)',
        'revenue_musd_mean': 'Mean Revenue per Movie (M USD)',
        'budget_musd_mean': 'Mean Budget (M USD)',
        'vote_average_mean': 'Mean Rating',
        'roi_mean': 'Mean ROI'
    })
    
    # Sort by total revenue
    director_analysis = director_analysis.sort_values('Total Revenue (M USD)', ascending=False)
    
    print("\n📊 DIRECTOR PERFORMANCE METRICS")
    print(director_analysis)
    
    # Highlight top director
    if len(director_analysis) > 0:
        top_director = director_analysis.index[0]
        top_director_revenue = director_analysis.iloc[0]['Total Revenue (M USD)']
        top_director_count = director_analysis.iloc[0]['Number of Movies']
        top_director_rating = director_analysis.iloc[0]['Mean Rating']
        
        print("\n" + "="*70)
        print("🎬 TOP DIRECTOR:")
        print("="*70)
        print(f"Director: {top_director}")
        print(f"Total Revenue: ${top_director_revenue:,.2f} Million")
        print(f"Number of Movies: {int(top_director_count)}")
        print(f"Average Revenue per Movie: ${top_director_revenue/top_director_count:,.2f} Million")
        print(f"Average Rating: {top_director_rating:.2f}/10")
    
    return director_analysis


# ============================================================================
# STEP 10: SAVE ANALYSIS RESULTS
# ============================================================================

def save_analysis_results(movies_kpi, rankings, franchise_analysis, director_analysis):
    """
    Save all analysis results to files.
    
    Args:
        movies_kpi (pd.DataFrame): DataFrame with KPIs
        rankings (dict): Dictionary of ranking results
        franchise_analysis (pd.DataFrame): Franchise analysis results
        director_analysis (pd.DataFrame): Director analysis results
    """
    print("\n" + "="*70)
    print("STEP 9: SAVING ANALYSIS RESULTS")
    print("="*70)
    
    # Save main KPI dataset
    kpi_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'movies_with_kpis.csv')
    movies_kpi.to_csv(kpi_path, index=False)
    print(f"✓ Saved KPI dataset: {kpi_path}")
    
    # Save franchise analysis
    if franchise_analysis is not None:
        franchise_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'franchise_analysis.csv')
        franchise_analysis.to_csv(franchise_path)
        print(f"✓ Saved franchise analysis: {franchise_path}")
    
    # Save director analysis
    if director_analysis is not None:
        director_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'director_analysis.csv')
        director_analysis.to_csv(director_path)
        print(f"✓ Saved director analysis: {director_path}")
            
    # Save text summary
    summary_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TMDb MOVIE ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Movies Analyzed: {len(movies_kpi)}\n\n")
        
        f.write("TOP 5 HIGHEST REVENUE MOVIES:\n")
        f.write("-"*70 + "\n")
        if 'highest_revenue' in rankings:
            f.write(rankings['highest_revenue'][['title', 'revenue_musd']].to_string() + "\n\n")
        
        f.write("TOP 5 HIGHEST ROI MOVIES:\n")
        f.write("-"*70 + "\n")
        if 'highest_roi' in rankings:
            f.write(rankings['highest_roi'][['title', 'roi']].head(5).to_string() + "\n\n")
    
    print(f"✓ Saved text summary: {summary_path}")
    print(f"\nAll analysis results saved in: {ANALYSIS_OUTPUT_DIR}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for analysis."""
    
    print("="*70)
    print("TMDb MOVIE ANALYSIS PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Load cleaned data
    df = load_cleaned_data()
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return
    
    # Step 2: Calculate KPIs
    movies_kpi = calculate_kpis(df)
    
    # Step 3: Test ranking function
    test_ranking_function(movies_kpi)
    
    # Step 4: Perform all rankings
    rankings = perform_all_rankings(movies_kpi)
    
    # Step 5: Advanced searches
    searches = perform_advanced_searches(movies_kpi)
    
    # Step 6: Franchise vs. Standalone
    franchise_comparison = analyze_franchise_vs_standalone(movies_kpi)
    
    # Step 7: Franchise analysis
    franchise_analysis = analyze_franchises(movies_kpi)
    
    # Step 8: Director analysis
    director_analysis = analyze_directors(movies_kpi)
    
    # Step 9: Save results
    save_analysis_results(movies_kpi, rankings, franchise_analysis, director_analysis)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()