"""
Main script for Handwritten Math Solutions.
"""

import argparse
from pathlib import Path

from src.config import config
from src.train import train

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Handwritten Math Solutions model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with annotations')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to output directory')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with paths
    config['data']['data_dir'] = args.data_dir
    config['data']['csv_file'] = args.csv_file
    config['callbacks']['model_checkpoint']['dirpath'] = str(output_dir / 'checkpoints')
    
    # Train model
    train()

if __name__ == '__main__':
    main() 