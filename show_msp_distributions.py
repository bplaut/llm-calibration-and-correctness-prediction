import os
import glob
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import utils

def parse_filename(filename):
    """Extract model and conf_type from filename."""
    basename = os.path.basename(filename)
    
    # Extract model (m=...)
    model_match = re.search(r'm=([^_]+)', basename)
    model = model_match.group(1) if model_match else None
    
    # Extract conf_type (c=...)
    conf_type_match = re.search(r'c=([^_]+)', basename)
    conf_type = conf_type_match.group(1) if conf_type_match else None
    
    return model, conf_type

def process_file(filepath):
    """Process a single file and return list of max confidence values."""
    max_values = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header if present
    start_idx = 1 if lines[0].strip().startswith('grade') else 0
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(' ', 1)  # Split on first space only
        if len(parts) != 2:
            continue
            
        grade, confidence_str = parts
        
        # Parse comma-separated confidence values
        try:
            confidence_values = [float(x.strip()) for x in confidence_str.split(',')]
            max_val = max(confidence_values)
            max_values.append(max_val)
        except ValueError:
            print(f"Warning: Could not parse confidence values in line: {line}")
            continue
    
    return max_values

def compute_statistics(values):
    """Compute distribution statistics for a list of values."""
    if not values:
        return None
    
    arr = np.array(values)
    stats = {
        'count': len(arr),
        'mean': np.mean(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'p25': np.percentile(arr, 25),
        'p50': np.percentile(arr, 50),
        'p75': np.percentile(arr, 75),
        'p90': np.percentile(arr, 90),
        'p95': np.percentile(arr, 95),
    }
    return stats

def main(data_directory='.'):
    """Main analysis function."""
    
    # Find all relevant files
    pattern = os.path.join(data_directory, 'd=*_m=*_q=*_c=*_p=*.txt')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern in {data_directory}")
        return
    
    # Group files by model and conf_type
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    for filepath in files:
        model, conf_type = parse_filename(filepath)
        
        if model is None or conf_type is None:
            print(f"Warning: Could not parse model/conf_type from {filepath}")
            continue
            
        if conf_type not in ['prob', 'logit']:
            continue
            
        max_values = process_file(filepath)
        grouped_data[model][conf_type].extend(max_values)
    
    # Compute and print statistics
    print("Confidence Distribution Analysis")
    print("=" * 50)
    
    for model in utils.sort_models(grouped_data.keys()):
        print(f"\nModel: {model}")
        print("-" * 30)
        
        for conf_type in ['prob', 'logit']:
            if conf_type not in grouped_data[model]:
                continue
                
            values = grouped_data[model][conf_type]
            stats = compute_statistics(values)
            
            if stats is None:
                print(f"  {conf_type}: No data")
                continue
            
            print(f"  {conf_type.upper()}:")
            print(f"    Count:  {stats['count']:6d}")
            print(f"    Mean:   {stats['mean']:8.4f}")
            print(f"    Min:    {stats['min']:8.4f}")
            print(f"    P25:    {stats['p25']:8.4f}")
            print(f"    P50:    {stats['p50']:8.4f}")
            print(f"    P75:    {stats['p75']:8.4f}")
            print(f"    P90:    {stats['p90']:8.4f}")
            print(f"    P95:    {stats['p95']:8.4f}")
            print(f"    Max:    {stats['max']:8.4f}")

if __name__ == "__main__":
    import sys
    
    # Allow specifying directory as command line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(directory)
