"""
visualization.py
================
Creates comprehensive visualizations for TMDb movie analysis.

This script generates:
- Revenue vs. Budget scatter plot with break-even line
- ROI distribution by genre (box plots)
- Popularity vs. Rating analysis
- Yearly box office performance trends
- Franchise vs. Standalone comparison charts
- Correlation heatmap

Usage:
    python visualization.py
    
Input:
    - data/analysis/movies_with_kpis.csv
    
Output:
    - visualizations/*.png (6 visualization files)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

ANALYSIS_DATA_DIR = "data/analysis"
VIZ_OUTPUT_DIR = "visualizations"
os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Color schemes
FRANCHISE_COLORS = {'Franchise': 'red', 'Standalone': 'blue'}


# ============================================================================
# STEP 1: LOAD DATA WITH KPIs
# ============================================================================

def load_kpi_data():
    """
    Load the dataset with calculated KPIs.
    
    Returns:
        pd.DataFrame: Movie data with KPIs
    """
    print("="*70)
    print("STEP 1: LOADING DATA FOR VISUALIZATION")
    print("="*70)
    
    csv_path = os.path.join(ANALYSIS_DATA_DIR, 'movies_with_kpis.csv')
    
    if not os.path.exists(csv_path):
        print(f"✗ Error: {csv_path} not found!")
        print("  Please run analysis.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Convert date column back to datetime
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    print(f"✓ Loaded {len(df)} movies from {csv_path}")
    print(f"  Columns available: {list(df.columns)}")
    
    return df


# ============================================================================
# VISUALIZATION 1: REVENUE VS. BUDGET TRENDS
# ============================================================================

def viz_revenue_vs_budget(df):
    """
    Create scatter plot showing Revenue vs. Budget with break-even line.
    
    Purpose: Identify profitable vs. unprofitable movies
    Key Feature: Diagonal break-even line where revenue = budget
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 1: REVENUE VS. BUDGET TRENDS")
    print("="*70)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    # Color by rating, size by popularity
    scatter = ax.scatter(
        df['budget_musd'],
        df['revenue_musd'],
        c=df['vote_average'],  # Color coding by rating
        s=df['popularity'] * 3,  # Size by popularity
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add break-even line (y = x)
    max_val = max(df['budget_musd'].max(), df['revenue_musd'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, 
            label='Break-even line (Revenue = Budget)', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Budget (Million USD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Revenue (Million USD)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue vs. Budget: Investment Returns Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar for ratings
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Vote Average (Rating)', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Annotate top 3 revenue movies
    top_3_revenue = df.nlargest(3, 'revenue_musd')
    for idx, row in top_3_revenue.iterrows():
        ax.annotate(
            row['title'],
            xy=(row['budget_musd'], row['revenue_musd']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black')
        )
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'revenue_vs_budget.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print insights
    above_line = (df['revenue_musd'] > df['budget_musd']).sum()
    total = len(df.dropna(subset=['revenue_musd', 'budget_musd']))
    print(f"\n📊 INSIGHTS:")
    print(f"  • {above_line}/{total} movies ({above_line/total*100:.1f}%) are profitable")
    print(f"  • Movies above the line made more than they cost")
    print(f"  • Bubble size = popularity, color = rating")


# ============================================================================
# VISUALIZATION 2: ROI DISTRIBUTION BY GENRE
# ============================================================================

def viz_roi_by_genre(df):
    """
    Create box plots showing ROI distribution across genres.
    
    Purpose: Compare investment efficiency by genre
    Key Feature: Shows distribution, median, and outliers
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 2: ROI DISTRIBUTION BY GENRE")
    print("="*70)
    
    # Prepare data: explode genres (split pipe-separated values)
    genre_roi_data = df[df['roi'].notna()].copy()
    genre_roi_data['genres_list'] = genre_roi_data['genres'].str.split('|')
    
    # Explode genres into separate rows
    genre_roi_exploded = genre_roi_data.explode('genres_list')
    
    # Remove whitespace
    genre_roi_exploded['genres_list'] = genre_roi_exploded['genres_list'].str.strip()
    
    # Get top 8 genres by count
    top_genres = genre_roi_exploded['genres_list'].value_counts().head(8).index
    
    # Filter to top genres only
    genre_roi_top = genre_roi_exploded[genre_roi_exploded['genres_list'].isin(top_genres)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create box plot
    sns.boxplot(
        data=genre_roi_top,
        x='genres_list',
        y='roi',
        palette='Set2',
        ax=ax
    )
    
    # Overlay individual points (strip plot)
    sns.stripplot(
        data=genre_roi_top,
        x='genres_list',
        y='roi',
        color='black',
        alpha=0.3,
        size=4,
        ax=ax
    )
    
    # Labels and title
    ax.set_xlabel('Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI (Return on Investment)', fontsize=12, fontweight='bold')
    ax.set_title('ROI Distribution by Genre: Which Genres Deliver Best Returns?', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add reference lines
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
               label='Break-even (ROI = 1.0)', alpha=0.7)
    ax.axhline(y=0, color='darkred', linestyle='--', linewidth=2, 
               label='Total Loss (ROI = 0)', alpha=0.7)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'roi_by_genre.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print insights
    genre_roi_stats = genre_roi_top.groupby('genres_list')['roi'].agg(['mean', 'median']).round(2)
    genre_roi_stats = genre_roi_stats.sort_values('mean', ascending=False)
    print(f"\n📊 INSIGHTS:")
    print(f"  Genre ROI Statistics (sorted by mean):")
    print(genre_roi_stats)
    best_genre = genre_roi_stats.index[0]
    best_roi = genre_roi_stats.iloc[0]['mean']
    print(f"\n  🏆 Best genre: {best_genre} (Mean ROI: {best_roi:.2f})")


# ============================================================================
# VISUALIZATION 3: POPULARITY VS. RATING
# ============================================================================

def viz_popularity_vs_rating(df):
    """
    Create scatter plot comparing popularity and rating.
    
    Purpose: Explore relationship between commercial success and quality
    Key Feature: Franchise vs. Standalone color coding
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 3: POPULARITY VS. RATING")
    print("="*70)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot franchise and standalone separately for color coding
    for franchise_status in [True, False]:
        subset = df[df['is_franchise'] == franchise_status]
        label = 'Franchise' if franchise_status else 'Standalone'
        color = FRANCHISE_COLORS[label]
        
        ax.scatter(
            subset['popularity'],
            subset['vote_average'],
            c=color,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=label
        )
    
    # Labels and title
    ax.set_xlabel('Popularity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Vote Average (Rating)', fontsize=12, fontweight='bold')
    ax.set_title('Popularity vs. Rating: Are Popular Movies Also Highly Rated?', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add reference lines (medians)
    median_rating = df['vote_average'].median()
    median_popularity = df['popularity'].median()
    
    ax.axhline(y=median_rating, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=median_popularity, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add quadrant labels
    ax.text(median_popularity * 1.5, median_rating * 1.05, 
            'Popular & Highly Rated', fontsize=10, style='italic', alpha=0.7)
    ax.text(median_popularity * 0.3, median_rating * 1.05, 
            'Critically Acclaimed\n(Hidden Gems)', fontsize=10, style='italic', alpha=0.7)
    ax.text(median_popularity * 1.5, median_rating * 0.95, 
            'Popular but Lower Rated', fontsize=10, style='italic', alpha=0.7)
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'popularity_vs_rating.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Calculate correlation
    correlation = df[['popularity', 'vote_average']].corr().iloc[0, 1]
    print(f"\n📊 INSIGHTS:")
    print(f"  • Correlation: {correlation:.3f}")
    if correlation > 0.5:
        print("    → Strong positive: Popular movies tend to be highly rated")
    elif correlation > 0.3:
        print("    → Moderate positive: Some relationship exists")
    else:
        print("    → Weak: Popularity and quality are largely independent")
    
    # Quadrant analysis
    high_pop_high_rate = ((df['popularity'] > median_popularity) & 
                          (df['vote_average'] > median_rating)).sum()
    print(f"  • {high_pop_high_rate} movies are both popular AND highly rated")


# ============================================================================
# VISUALIZATION 4: YEARLY TRENDS IN BOX OFFICE
# ============================================================================

def viz_yearly_trends(df):
    """
    Create multi-panel plot showing trends over time.
    
    Purpose: Track industry evolution (revenue, budget, rating, volume)
    Key Feature: 4 subplots showing different metrics
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 4: YEARLY TRENDS IN BOX OFFICE PERFORMANCE")
    print("="*70)
    
    # Extract year from release_date
    df['release_year'] = df['release_date'].dt.year
    
    # Group by year
    yearly_trends = df.groupby('release_year').agg({
        'revenue_musd': 'mean',
        'budget_musd': 'mean',
        'vote_average': 'mean',
        'title': 'count'  # Movie count per year
    }).round(2)
    
    yearly_trends = yearly_trends.rename(columns={'title': 'movie_count'})
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average Revenue over time
    axes[0, 0].plot(yearly_trends.index, yearly_trends['revenue_musd'], 
                    marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 0].set_title('Average Revenue Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Year', fontsize=10)
    axes[0, 0].set_ylabel('Average Revenue (M USD)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Budget over time
    axes[0, 1].plot(yearly_trends.index, yearly_trends['budget_musd'], 
                    marker='s', linewidth=2, markersize=8, color='blue')
    axes[0, 1].set_title('Average Budget Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Year', fontsize=10)
    axes[0, 1].set_ylabel('Average Budget (M USD)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Rating over time
    axes[1, 0].plot(yearly_trends.index, yearly_trends['vote_average'], 
                    marker='^', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_title('Average Rating Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Year', fontsize=10)
    axes[1, 0].set_ylabel('Average Vote Rating', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of movies per year
    axes[1, 1].bar(yearly_trends.index, yearly_trends['movie_count'], 
                   color='purple', alpha=0.7)
    axes[1, 1].set_title('Number of Movies Per Year (in dataset)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Year', fontsize=10)
    axes[1, 1].set_ylabel('Movie Count', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Box Office Performance Trends Over Time', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'yearly_trends.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print insights
    print(f"\n📊 INSIGHTS:")
    print(f"  Yearly Statistics:")
    print(yearly_trends)
    
    if len(yearly_trends) > 1:
        first_year = yearly_trends.index[0]
        last_year = yearly_trends.index[-1]
        revenue_growth = ((yearly_trends.loc[last_year, 'revenue_musd'] - 
                          yearly_trends.loc[first_year, 'revenue_musd']) / 
                         yearly_trends.loc[first_year, 'revenue_musd'] * 100)
        print(f"\n  • Revenue change from {first_year} to {last_year}: {revenue_growth:+.1f}%")


# ============================================================================
# VISUALIZATION 5: FRANCHISE VS. STANDALONE COMPARISON
# ============================================================================

def viz_franchise_vs_standalone(df):
    """
    Create bar charts comparing franchise and standalone movies.
    
    Purpose: Visual business model comparison
    Key Feature: Multiple metrics side-by-side
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 5: FRANCHISE VS. STANDALONE SUCCESS COMPARISON")
    print("="*70)
    
    # Prepare comparison data
    comparison_metrics = df.groupby('movie_type').agg({
        'revenue_musd': 'mean',
        'budget_musd': 'mean',
        'roi': 'median',
        'vote_average': 'mean',
        'popularity': 'mean'
    }).round(2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = [
        ('revenue_musd', 'Mean Revenue (M USD)', 'green'),
        ('budget_musd', 'Mean Budget (M USD)', 'blue'),
        ('roi', 'Median ROI', 'orange'),
        ('vote_average', 'Mean Rating', 'red'),
        ('popularity', 'Mean Popularity', 'purple')
    ]
    
    # Create bar plots for each metric
    for idx, (metric, title, color) in enumerate(metrics):
        ax = axes[idx]
        
        comparison_metrics[metric].plot(kind='bar', ax=ax, color=color, 
                                        alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    # Hide the last subplot (we only have 5 metrics)
    axes[5].axis('off')
    
    plt.suptitle('Franchise vs. Standalone Movies: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'franchise_vs_standalone.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print comparison
    print(f"\n📊 DETAILED COMPARISON:")
    print(comparison_metrics)
    
    # Winners by metric
    print(f"\n🏆 WINNERS BY METRIC:")
    for metric, title, _ in metrics:
        winner = comparison_metrics[metric].idxmax()
        value = comparison_metrics[metric].max()
        print(f"  • {title}: {winner} ({value:.2f})")


# ============================================================================
# VISUALIZATION 6: CORRELATION HEATMAP
# ============================================================================

def viz_correlation_heatmap(df):
    """
    Create correlation heatmap showing relationships between metrics.
    
    Purpose: Identify which factors correlate with success
    Key Feature: Color-coded correlation matrix
    
    Args:
        df (pd.DataFrame): Movie data
    """
    print("\n" + "="*70)
    print("VISUALIZATION 6: CORRELATION HEATMAP")
    print("="*70)
    
    # Select numeric columns for correlation
    numeric_cols = ['budget_musd', 'revenue_musd', 'profit_musd', 'roi',
                    'vote_average', 'vote_count', 'popularity', 'runtime']
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Show correlation values
        fmt='.2f',  # 2 decimal places
        cmap='coolwarm',  # Color scheme
        center=0,  # Center colormap at 0
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix: Relationships Between Movie Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(VIZ_OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Print strong correlations
    print(f"\n📊 STRONG CORRELATIONS (|correlation| > 0.7):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                print(f"  • {var1} ↔ {var2}: {corr_value:.3f}")
                if corr_value > 0:
                    print(f"    → Positive: When {var1} ↑, {var2} ↑")
                else:
                    print(f"    → Negative: When {var1} ↑, {var2} ↓")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for visualizations."""
    
    print("="*70)
    print("TMDb MOVIE DATA VISUALIZATION PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    df = load_kpi_data()
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    print("-" * 70)
    
    viz_revenue_vs_budget(df)
    viz_roi_by_genre(df)
    viz_popularity_vs_rating(df)
    viz_yearly_trends(df)
    viz_franchise_vs_standalone(df)
    viz_correlation_heatmap(df)
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY")
    print("="*70)
    print("\n✅ Created the following visualizations:")
    print("  1. revenue_vs_budget.png - Investment returns analysis")
    print("  2. roi_by_genre.png - Genre efficiency comparison")
    print("  3. popularity_vs_rating.png - Quality vs. commercial success")
    print("  4. yearly_trends.png - Industry evolution over time")
    print("  5. franchise_vs_standalone.png - Business model comparison")
    print("  6. correlation_heatmap.png - Metric relationships")
    print(f"\nAll visualizations saved in: {VIZ_OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()