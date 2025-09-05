import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

input_path = './cleaned_gemini_reddit_stories.csv'
df = pd.read_csv(input_path)

# Print basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Get categorisation columns
categorisation_cols = [col for col in df.columns if col.startswith('categorisation_')]
print(f"\nCategorisation columns ({len(categorisation_cols)}):", categorisation_cols)

# Get unique values for grouping variables
age_groups = df['age_group'].unique()
income_groups = df['income_group'].unique()
family_types = df['family_type'].unique()

print(f"\nAge groups ({len(age_groups)}):", sorted(age_groups))
print(f"Income groups ({len(income_groups)}):", sorted(income_groups))
print(f"Family types ({len(family_types)}):", sorted(family_types))

def create_comprehensive_analysis():
    """
    Create a grid visualization with:
    - Rows: Each categorisation column + number_child
    - Columns: ALL unique values from age_group, income_group, family_type
    - Each cell: Histogram of score distribution for that category in that specific demographic value
    """
    
    # Columns to analyze (categorisation + number_child)
    analysis_cols = categorisation_cols + ['number_child']
    
    # Clean and convert numeric columns
    df_clean = df.copy()
    
    # Convert categorisation columns to numeric, handling errors
    for col in categorisation_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Convert number_child to numeric, handling string values
    df_clean['number_child'] = pd.to_numeric(df_clean['number_child'], errors='coerce')
    
    # Remove rows where all categorisation columns are NaN
    df_clean = df_clean.dropna(subset=analysis_cols, how='all')
    
    print(f"After cleaning: {len(df_clean)} rows remaining")
    
    # Create list of all demographic values with their types (excluding unknown)
    all_demographics = []
    
    # Add age groups (excluding unknown)
    for age_val in sorted(df_clean['age_group'].unique()):
        if age_val != 'unknown':
            all_demographics.append(('age_group', age_val, f"Age: {age_val}"))
    
    # Add income groups (excluding unknown)
    for income_val in sorted(df_clean['income_group'].unique()):
        if income_val != 'unknown':
            all_demographics.append(('income_group', income_val, f"Income: {income_val}"))
    
    # Add family types (excluding unknown)
    for family_val in sorted(df_clean['family_type'].unique()):
        if family_val != 'unknown':
            all_demographics.append(('family_type', family_val, f"Family: {family_val}"))
    
    print(f"Total demographic columns: {len(all_demographics)}")
    print("Demographics:", [demo[2] for demo in all_demographics])
    
    # Create figure: rows = analysis_cols, columns = all demographic values
    n_rows = len(analysis_cols)
    n_cols = len(all_demographics)
    
    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    # Iterate through each analysis column (rows)
    for row_idx, analysis_col in enumerate(analysis_cols):
        print(f"\nCreating row {row_idx + 1}/{n_rows}: {analysis_col}")
        
        # Add row label on the left
        if row_idx < n_rows:
            row_label = analysis_col.replace("categorisation_", "").replace("_", " ").title()
            # Calculate the exact center position of each row based on GridSpec layout
            # Account for the adjusted margins in subplots_adjust
            plot_top = 0.93
            plot_bottom = 0.07  # Standard matplotlib bottom margin
            plot_height = plot_top - plot_bottom
            row_height = plot_height / n_rows
            # Position at exact center of each row
            y_position = plot_top - (row_idx + 0.5) * row_height
            
            fig.text(0.005, y_position, row_label,  # Moved further left to prevent overlap
                    fontsize=8, weight='bold', rotation=0,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
        
        # Iterate through each demographic value (columns)
        for col_idx, (demo_col, demo_value, demo_label) in enumerate(all_demographics):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Get data for this specific demographic value
            subset = df_clean[df_clean[demo_col] == demo_value]
            
            if len(subset) > 0 and analysis_col in subset.columns:
                data = subset[analysis_col].dropna()
                
                if len(data) > 0:
                    # Create histogram
                    ax.hist(data.values, bins=min(10, len(data.unique())), alpha=0.7, 
                           color=colors[col_idx % len(colors)], 
                           edgecolor='black', linewidth=0.5)
                    
                    # Add statistics
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    
                    # Add mean line
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    
                    # Add statistics text
                    stats_text = f'n={len(data)}\nμ={mean_val:.2f}\nσ={std_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    
                    # Customize the plot
                    # Only show title on top row
                    if row_idx == 0:
                        ax.set_title(f'{demo_label}', fontsize=9, weight='bold')
                    
                    # Only show x-axis label on bottom row
                    if row_idx == n_rows - 1:
                        ax.set_xlabel('Score', fontsize=8)
                    else:
                        ax.set_xlabel('')
                    
                    # Only show y-axis label on leftmost column
                    if col_idx == 0:
                        ax.set_ylabel('Frequency', fontsize=8)
                    else:
                        ax.set_ylabel('')
                    
                    # Set reasonable axis limits
                    if analysis_col == 'number_child':
                        ax.set_xlim(-0.5, 4.5)
                    else:
                        ax.set_xlim(0, 10)
                
                else:
                    ax.text(0.5, 0.5, 'No Data\n(all NaN)', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    # Only show title on top row
                    if row_idx == 0:
                        ax.set_title(f'{demo_label}', fontsize=9)
            else:
                ax.text(0.5, 0.5, f'No Data\n(n=0)', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                # Only show title on top row
                if row_idx == 0:
                    ax.set_title(f'{demo_label}', fontsize=9)
            
            # Improve readability
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
    
    # Add overall title
    fig.suptitle(f'Complete Distribution Analysis: {n_rows} Categories × {n_cols} Demographics\n' + 
                 'Each cell shows histogram for one category in one demographic group', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.2)  # Increased left margin to prevent overlap
    
    # Save the plot
    plt.savefig('comprehensive_demographics_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Run the analysis
fig = create_comprehensive_analysis()

# Add a summary analysis
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Clean data for summary
df_summary = df.copy()
for col in categorisation_cols:
    df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
df_summary['number_child'] = pd.to_numeric(df_summary['number_child'], errors='coerce')

# Overall statistics by demographic groups
print("\n1. AGE GROUP ANALYSIS:")
age_summary = df_summary.groupby('age_group')[categorisation_cols + ['number_child']].mean().round(3)
print(age_summary)

print("\n2. INCOME GROUP ANALYSIS:")
income_summary = df_summary.groupby('income_group')[categorisation_cols + ['number_child']].mean().round(3)
print(income_summary)

print("\n3. FAMILY TYPE ANALYSIS:")
family_summary = df_summary.groupby('family_type')[categorisation_cols + ['number_child']].mean().round(3)
print(family_summary)

# Find highest scoring categories
print("\n4. HIGHEST SCORING CATEGORIES:")
for col in categorisation_cols + ['number_child']:
    max_score = df_summary[col].max()
    if pd.notna(max_score):
        print(f"\n{col.replace('categorisation_', '').replace('_', ' ').title()}:")
        print(f"  Max score: {max_score}")
        
        # Find which demographics have highest scores
        for demo_col in ['age_group', 'income_group', 'family_type']:
            demo_means = df_summary.groupby(demo_col)[col].mean().sort_values(ascending=False)
            print(f"  Highest in {demo_col}: {demo_means.index[0]} ({demo_means.iloc[0]:.3f})")

print(f"\nVisualization saved as: comprehensive_demographics_histograms.png")