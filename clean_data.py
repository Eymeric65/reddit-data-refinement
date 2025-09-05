import pandas as pd
import json
import ast
import re

def replace_values_with_mapping(df, column_mappings):
    """
    Replace values in DataFrame columns using mapping dictionaries.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify
    column_mappings (dict): Dictionary where keys are column names and values are mapping dictionaries
    
    Returns:
    pd.DataFrame: DataFrame with replaced values
    
    Example:
    mappings = {
        'age_group': {
            '28': '20-30',
            '28F': '20-30', 
            'early 20s': '20-30'
        }
    }
    """
    df_copy = df.copy()
    
    for column, mapping in column_mappings.items():
        if column in df_copy.columns:
            # Convert to string for consistent matching
            df_copy[column] = df_copy[column].astype(str)
            
            # Apply direct mappings first
            for old_value, new_value in mapping.items():
                df_copy[column] = df_copy[column].replace(old_value, new_value)
    
    return df_copy

def replace_values_with_patterns(df, column_patterns):
    """
    Replace values in DataFrame columns using regex patterns.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to modify
    column_patterns (dict): Dictionary where keys are column names and values are lists of (pattern, replacement) tuples
    
    Returns:
    pd.DataFrame: DataFrame with replaced values
    
    Example:
    patterns = {
        'age_group': [
            (r'.*20.*', '20-30'),  # anything containing "20"
            (r'.*30.*', '30-40'),  # anything containing "30"
        ]
    }
    """
    df_copy = df.copy()
    
    for column, patterns in column_patterns.items():
        if column in df_copy.columns:
            # Convert to string for consistent matching
            df_copy[column] = df_copy[column].astype(str)
            
            # Apply pattern replacements
            for pattern, replacement in patterns:
                df_copy[column] = df_copy[column].str.replace(pattern, replacement, regex=True)
    
    return df_copy

input_path = './processed_gemini_reddit_stories.csv'
df = pd.read_csv(input_path)

# 1) Keep only non-empty values
col = 'gemini_result'
s = df[col].dropna().astype(str).str.strip()
s = s[s.ne('') & s.ne('nan') & s.ne('None')]

# 2) Parse each row into a Python object (dict/list)
def parse_obj(x):
    # Already a dict/list? (in case CSV preserved python objects)
    if isinstance(x, (dict, list)):
        return x
    # Try JSON first
    try:
        return json.loads(x)
    except Exception:
        pass
    # Try Python literal (handles single quotes, True/False/None)
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

parsed = s.apply(parse_obj)

# 3) Keep only dicts/lists that normalized can handle
records = [p for p in parsed.tolist() if isinstance(p, (dict, list))]

# Quick diagnostics
print(f"Total rows: {len(df)}")
print(f"Non-empty gemini_result: {len(s)}")
print(f"Parsable objects: {sum(isinstance(p, (dict, list)) for p in parsed)}")

# 4) Normalize
if records:
    df_gemini = pd.json_normalize(records, sep="_")
    # Optional: bring along original row index to rejoin later
    df_gemini.insert(0, 'source_index', s.index.to_list())
else:
    df_gemini = pd.DataFrame()

print(df_gemini.columns)
print(df_gemini.shape)


print("unique columns:", df_gemini["age_group"].unique())
print("unique columns:", df_gemini["income_group"].unique())
print("unique columns:", df_gemini["family_type"].unique())
print("unique number_child values:", df_gemini["number_child"].unique())

# Example usage of the replacement functions
print("\n" + "="*50)
print("APPLYING VALUE REPLACEMENTS")
print("="*50)

# Define mappings for age groups
age_mappings = {
    'age_group': {
        '28': '20-30',
        '28F, 36M': '20-30',
        'mid-twenties': '20-30',
        'young couple': '20-30',
        'late 20s to early 30s': '20-30',
        '30-35': '30-40',
        '30-40': '30-40',
        'early adulthood (20s)': '20-30',
        '30s-40s': '30-40',
        '30s': '30-40',
        'couple in their mid-30s': '30-40',
        'young adult': '20-30',
        '20-30': '20-30',
        'adult': '20-30',
        '35-45': '30-40',
        'young adults': '20-30',
        'early 30s': '30-40',
        '30-31': '30-40',
        '28-35': '20-30',
        'early to mid-20s': '20-30',
        'Couple likely late 30s to early 40s, son is a teenager (15 years old)': '30-40',
        'late 20s-early 30s': '20-30',
        '29-30': '20-30',
        'late 20s - early 30s': '20-30',
        'late 20s/early 30s': '20-30',
        'couple in their late 20s to early 30s': '20-30',
        '20s-early 30s': '20-30',
        '35-36': '30-40',
        '20s-30s': '20-30',
        '25-35': '20-30'
    }
}

# Define mappings for family type (fixing typo)
family_mappings = {
    'family_type': {
        'extented': 'extended'  # Fix the typo
    }
}

# Define mappings for number_child (cleaning text values)
number_child_mappings = {
    'number_child': {
        'one': '1',
        'expecting first child': '0',  # Not yet born
        'first child': '1',
        'presumed 1 or more, likely young children': '1',
        'unknown': '0'  # Assume 0 for unknown cases
    }
}

def clean_number_child_column(df):
    """
    Clean the number_child column by converting text to numbers and handling edge cases
    """
    df_copy = df.copy()
    
    # First apply text mappings
    df_copy = replace_values_with_mapping(df_copy, number_child_mappings)
    
    # Convert to string first, then to numeric
    df_copy['number_child'] = df_copy['number_child'].astype(str)
    
    # Handle any remaining edge cases with regex
    df_copy['number_child'] = df_copy['number_child'].str.extract(r'(\d+)').fillna('0')
    
    # Convert to numeric, coercing errors to NaN, then fill NaN with 0
    df_copy['number_child'] = pd.to_numeric(df_copy['number_child'], errors='coerce').fillna(0).astype(int)
    
    return df_copy

# Apply the mappings
df_cleaned = replace_values_with_mapping(df_gemini, age_mappings)
df_cleaned = replace_values_with_mapping(df_cleaned, family_mappings)

# Clean the number_child column
df_cleaned = clean_number_child_column(df_cleaned)

print("\nAfter cleaning:")
print("Age groups:", sorted(df_cleaned["age_group"].unique()))
print("Income groups:", sorted(df_cleaned["income_group"].unique()))
print("Family types:", sorted(df_cleaned["family_type"].unique()))
print("Number of children (unique values):", sorted(df_cleaned["number_child"].unique()))
print("Number of children (value counts):")
print(df_cleaned["number_child"].value_counts().sort_index())

# Validation: Check data types
print(f"\nData type validation:")
print(f"number_child dtype: {df_cleaned['number_child'].dtype}")
print(f"age_group dtype: {df_cleaned['age_group'].dtype}")
print(f"income_group dtype: {df_cleaned['income_group'].dtype}")
print(f"family_type dtype: {df_cleaned['family_type'].dtype}")

# Check for any remaining NaN values
print(f"\nMissing values check:")
print(f"number_child NaN count: {df_cleaned['number_child'].isna().sum()}")
print(f"age_group NaN count: {df_cleaned['age_group'].isna().sum()}")
print(f"income_group NaN count: {df_cleaned['income_group'].isna().sum()}")
print(f"family_type NaN count: {df_cleaned['family_type'].isna().sum()}")

# Summary statistics for number_child
print(f"\nNumber of children summary statistics:")
print(f"Mean: {df_cleaned['number_child'].mean():.2f}")
print(f"Median: {df_cleaned['number_child'].median():.2f}")
print(f"Min: {df_cleaned['number_child'].min()}")
print(f"Max: {df_cleaned['number_child'].max()}")
print(f"Standard deviation: {df_cleaned['number_child'].std():.2f}")

# Save the cleaned data
output_path = './cleaned_gemini_reddit_stories.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")

