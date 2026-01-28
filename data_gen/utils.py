import re
import json


def extract_json_from_output(output: str) -> str:
    """
    Safely extract the first JSON object from the LLM's output.
    Handles optional ```json ... ``` markdown wrapping.
    """
    # Remove ```json ... ``` or ``` ... ``` if present
    match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", output)
    if match:
        return match.group(1)

    # Fallback: find first JSON block anywhere
    match = re.search(r"({[\s\S]*})", output)
    if match:
        return match.group(1)

    # Final fallback: assume full output is raw JSON string
    return output


def compute_data_stats(df_or_list):
    """
    Compute data statistics for category and level distributions.
    
    Used by the question generation prompt to guide balanced data generation.
    The LLM uses these stats to prioritize underrepresented categories/levels.
    
    Args:
        df_or_list: Either a pandas DataFrame or a list of dicts with 'category' and 'level' keys
        
    Returns:
        Tuple of (data_stats_dict, last_vals_str)
        - data_stats_dict: Dictionary with category/level names as keys and percentage strings as values
        - last_vals_str: String describing the last few category/level values for deprioritization
    """
    import pandas as pd
    
    # Convert list to DataFrame if needed
    if isinstance(df_or_list, list):
        df = pd.DataFrame(df_or_list)
    else:
        df = df_or_list.copy()
    
    total = len(df)
    if total == 0:
        # Return empty stats with 0% for all categories
        categories = ['relation', 'reach', 'size', 'orientation', 'instance_location', 
                      'depth', 'distance', 'count', 'existence']
        levels = ['easy', 'medium', 'hard']
        data_stats = {cat: "0.0%" for cat in categories}
        data_stats.update({level: "0.0%" for level in levels})
        return data_stats, "No data yet"
    
    data_stats = {}
    
    # Category statistics (9 categories for spatial reasoning)
    categories = ['relation', 'reach', 'size', 'orientation', 'instance_location', 
                  'depth', 'distance', 'count', 'existence']
    
    # Target: equal distribution (~11.1% each for 9 categories)
    for cat in categories:
        count = len(df[df['category'] == cat]) if 'category' in df.columns else 0
        percentage = (count / total) * 100
        data_stats[cat] = f"{percentage:.1f}%"
    
    # Level statistics (target: 40% easy, 40% medium, 20% hard)
    levels = ['easy', 'medium', 'hard']
    for level in levels:
        count = len(df[df['level'] == level]) if 'level' in df.columns else 0
        percentage = (count / total) * 100
        data_stats[level] = f"{percentage:.1f}%"
    
    # Add total count for context
    data_stats['total_samples'] = str(total)
    
    # Get last 3 category/level values for deprioritization
    # This helps avoid generating the same category/level consecutively
    last_n = min(3, len(df))
    if last_n > 0:
        if isinstance(df_or_list, list):
            last_entries = df_or_list[-last_n:]
        else:
            last_entries = df.tail(last_n).to_dict('records')
        
        last_cats = [entry.get('category', 'unknown') for entry in last_entries]
        last_levels = [entry.get('level', 'unknown') for entry in last_entries]
        last_vals = f"Categories: {last_cats}, Levels: {last_levels}"
    else:
        last_vals = "No previous entries"
    
    return data_stats, last_vals

