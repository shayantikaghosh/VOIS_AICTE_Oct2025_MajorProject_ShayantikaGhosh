import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
FILE_PATH = 'Netflix Dataset.csv'
# Define colors for visualization consistency
NETFLIX_RED = '#E50914'
LIGHT_GREY = '#6e7072'

def load_and_preprocess_data(file_path):
    """Loads the dataset and performs cleaning, date conversion, and preparation."""
    print("--- 1. Data Loading and Preprocessing ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the file name and path.")
        return None

    # Rename column for easier access
    df.rename(columns={'Category': 'Content_Type', 'Type': 'Genre'}, inplace=True)

    # Convert Release_Date to datetime, then extract Release_Year
    df['Release_Date'] = pd.to_datetime(df['Release_Date'])
    df['Release_Year'] = df['Release_Date'].dt.year

    # Handling missing 'Country' data: impute with 'Unknown' or mode, but for country analysis, we drop NA for accuracy.
    # For simplicity, we fill NA in Country and Genre with 'Missing' to keep all rows for general stats,
    # but we'll drop them for specific Objective 3 analysis.
    df['Country'].fillna('Missing', inplace=True)
    df['Genre'].fillna('Missing', inplace=True)
    
    print(f"Dataset loaded with {len(df)} records.")
    print(f"Earliest Release Year: {df['Release_Year'].min()}, Latest Release Year: {df['Release_Year'].max()}")
    return df

def explode_data_for_counting(df, column):
    """Splits and 'explodes' a column with comma-separated values for accurate counting."""
    # Create a copy to avoid SettingWithCopyWarning
    df_temp = df.copy()
    
    # Drop rows where the column is 'Missing' or blank, as they won't contribute to trends
    if column in ['Country', 'Genre']:
        df_temp = df_temp[df_temp[column] != 'Missing']
        
    # Split and clean entries
    exploded_df = df_temp.assign(
        Split_Value=df_temp[column].str.split(', ')
    ).explode('Split_Value')
    
    # Clean up whitespace and ensure titles are capitalized for consistency
    exploded_df['Split_Value'] = exploded_df['Split_Value'].str.strip()
    
    return exploded_df

def objective_1_content_type_evolution(df):
    """Analyzes the distribution of Movies vs. TV Shows over the years (Objective 1)."""
    print("\n--- 2. Objective 1: Movies vs. TV Shows Content Evolution ---")
    
    # Count content type additions per year
    content_by_year = df.groupby(['Release_Year', 'Content_Type']).size().unstack(fill_value=0)
    
    # Filter for the main trend period (e.g., last 10 years, or from 2014 onwards)
    start_year = content_by_year.index.max() - 10 if content_by_year.index.max() > 2014 else content_by_year.index.min()
    content_by_year_filtered = content_by_year[content_by_year.index >= start_year]
    
    # Visualization: Dual Line Chart
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(
        x=content_by_year_filtered.index, 
        y=content_by_year_filtered['Movie'], 
        label='Movies Added', 
        color='blue', 
        linewidth=2, 
        marker='o'
    )
    
    sns.lineplot(
        x=content_by_year_filtered.index, 
        y=content_by_year_filtered['TV Show'], 
        label='TV Shows Added', 
        color=NETFLIX_RED, 
        linewidth=2, 
        marker='s'
    )

    plt.title(f'Annual Content Additions (Movies vs. TV Shows): {start_year} - {content_by_year.index.max()}', fontsize=16)
    plt.xlabel('Release Year', fontsize=12)
    plt.ylabel('Number of Titles Added', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Content Type')
    plt.xticks(content_by_year_filtered.index, rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Provide a key finding
    peak_movie = content_by_year['Movie'].max()
    peak_tv = content_by_year['TV Show'].max()
    peak_year_movie = content_by_year['Movie'].idxmax()
    peak_year_tv = content_by_year['TV Show'].idxmax()

    print(f"\n--- Key Findings (Content Type) ---")
    print(f"Overall Catalog Composition: Movie: {len(df[df['Content_Type'] == 'Movie'])}, TV Show: {len(df[df['Content_Type'] == 'TV Show'])}")
    print(f"Peak Movie Additions ({peak_movie} titles) occurred in {peak_year_movie}.")
    print(f"Peak TV Show Additions ({peak_tv} titles) occurred in {peak_year_tv}.")
    print("-" * 35)

def objective_2_genre_popularity(df, top_n=10):
    """Analyzes the most popular genres and their recent changes (Objective 2)."""
    print("\n--- 3. Objective 2: Genre Popularity and Shift ---")
    
    # Explode the Genre column
    genres_exploded_df = explode_data_for_counting(df, 'Genre')
    genres_exploded_df.rename(columns={'Split_Value': 'Individual_Genre'}, inplace=True)
    
    # 1. Overall Top Genres
    overall_genre_counts = genres_exploded_df['Individual_Genre'].value_counts().nlargest(top_n)
    
    print(f"\n--- Overall Top {top_n} Genres (All Time) ---")
    print(overall_genre_counts)
    
    # 2. Genre shift analysis (Focusing on a period, e.g., 2018 onwards as suggested by modern content strategy)
    recent_years = [y for y in range(df['Release_Year'].max() - 3, df['Release_Year'].max() + 1)]
    recent_genres_df = genres_exploded_df[genres_exploded_df['Release_Year'].isin(recent_years)]
    
    recent_genre_trends = recent_genres_df.groupby(['Individual_Genre', 'Release_Year']).size().unstack(fill_value=0)
    
    # Filter to only the top overall genres for a focused plot
    top_genres_names = overall_genre_counts.index.tolist()
    recent_genre_trends_focused = recent_genre_trends[recent_genre_trends.index.isin(top_genres_names)]
    
    # Sort the focused trends by the latest year's count to prioritize current relevance
    recent_genre_trends_focused['Total_Recent'] = recent_genre_trends_focused.sum(axis=1)
    recent_genre_trends_focused = recent_genre_trends_focused.sort_values(by=recent_years[-1], ascending=False).drop(columns=['Total_Recent'])
    
    # Visualization: Stacked Bar Chart for Recent Trend
    ax = recent_genre_trends_focused.plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 7), 
        color=sns.color_palette("Spectral", n_colors=len(recent_years))
    )
    plt.title(f'Top {top_n} Genre Popularity Shift ({recent_years[0]} - {recent_years[-1]})', fontsize=16)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of Titles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Release Year', loc='upper right')
    plt.tight_layout()
    plt.show()

def objective_3_country_contribution(df, top_n=10):
    """Compares country-wise contributions to the catalog (Objective 3)."""
    print("\n--- 4. Objective 3: Global Country Contribution ---")
    
    # Explode the Country column
    country_exploded_df = explode_data_for_counting(df, 'Country')
    country_exploded_df.rename(columns={'Split_Value': 'Individual_Country'}, inplace=True)
    
    # 1. Overall Top Countries
    overall_country_counts = country_exploded_df['Individual_Country'].value_counts().nlargest(top_n)
    
    print(f"\n--- Overall Top {top_n} Content-Contributing Countries ---")
    print(overall_country_counts)
    
    # Visualization: Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=overall_country_counts.index, 
        y=overall_country_counts.values, 
        palette=sns.color_palette("viridis", n_colors=top_n)
    )
    
    plt.title(f'Top {top_n} Countries by Content Contribution', fontsize=16)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Total Content Count (Movies & TV Shows)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 2. Country contribution ratio (US vs. International)
    total_content = len(country_exploded_df)
    us_count = overall_country_counts.get('United States', 0)
    
    # The 'International' category is implicitly everything else
    non_us_count = total_content - us_count
    
    print(f"\n--- US vs. International Content Split ---")
    print(f"Total entries counted: {total_content}")
    print(f"United States content entries: {us_count} ({us_count / total_content * 100:.1f}%)")
    print(f"Non-US content entries: {non_us_count} ({non_us_count / total_content * 100:.1f}%)")
    print("-" * 35)

def run_full_analysis():
    """Executes the entire project analysis pipeline."""
    df_raw = load_and_preprocess_data(FILE_PATH)
    
    if df_raw is not None:
        objective_1_content_type_evolution(df_raw)
        objective_2_genre_popularity(df_raw)
        objective_3_country_contribution(df_raw)
        
        print("\n--- Project Analysis Complete ---")
        print("The visualizations and key findings address all three project objectives.")

if __name__ == "__main__":
    run_full_analysis()