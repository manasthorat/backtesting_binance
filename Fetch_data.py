import requests
import json
import pandas as pd
from datetime import datetime
import os
import time

class BinanceHistoricalFetcher:
    def __init__(self, config_file="config.json"):
        """Initialize with configuration file"""
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Config file {config_file} not found!")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_file}")
            return None
    
    def fetch_historical_data(self):
        """Fetch historical kline data based on config"""
        if not self.config:
            return None
            
        if self.config.get("fetch_all", False):
            return self.fetch_all_historical_data()
        else:
            return self.fetch_single_request()
    
    def fetch_single_request(self):
        """Fetch data with single API request"""
        # Build API URL
        base_url = self.config["api"]["base_url"]
        endpoint = self.config["api"]["endpoint"]
        url = f"{base_url}{endpoint}"
        
        # Build parameters
        params = {
            "symbol": self.config["symbol"],
            "interval": self.config["timeframe"],
            "limit": min(self.config["limit"], 1000)  # Max 1000 per request
        }
        
        # Add optional time parameters
        if self.config.get("start_time"):
            params["startTime"] = self.config["start_time"]
        if self.config.get("end_time"):
            params["endTime"] = self.config["end_time"]
        
        try:
            print(f"Fetching {self.config['symbol']} data...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = self.process_data(data)
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    
    def fetch_all_historical_data(self):
        """Fetch all available historical data using pagination"""
        base_url = self.config["api"]["base_url"]
        endpoint = self.config["api"]["endpoint"]
        url = f"{base_url}{endpoint}"
        
        all_data = []
        max_requests = self.config.get("max_requests", 100)
        requests_made = 0
        
        # Start from end_time if provided, otherwise get latest data first
        current_end_time = self.config.get("end_time")
        start_time = self.config.get("start_time")
        
        print(f"Fetching all available {self.config['symbol']} data...")
        print(f"Maximum requests: {max_requests}")
        
        while requests_made < max_requests:
            params = {
                "symbol": self.config["symbol"],
                "interval": self.config["timeframe"],
                "limit": 1000  # Always use max limit for efficiency
            }
            
            if current_end_time:
                params["endTime"] = current_end_time
            if start_time:
                params["startTime"] = start_time
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    print("No more data available")
                    break
                
                all_data.extend(data)
                requests_made += 1
                
                print(f"Request {requests_made}: Fetched {len(data)} records")
                
                # Update end_time to the first timestamp of current batch for next request
                # This ensures we get older data in the next request
                first_timestamp = data[0][0]  # First timestamp in the batch
                
                # If we've reached the start_time, break
                if start_time and first_timestamp <= start_time:
                    print("Reached specified start time")
                    break
                
                current_end_time = first_timestamp - 1  # Go back 1ms to avoid overlap
                
                # Rate limiting: respect API limits (1200/min, 20/sec)
                time.sleep(0.1)  # 100ms delay between requests
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break
        
        if all_data:
            # Remove duplicates and sort by timestamp
            df = self.process_data(all_data)
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Filter by start_time if specified
            if start_time:
                df = df[df['timestamp'] >= pd.to_datetime(start_time, unit='ms')]
            
            print(f"Total records fetched: {len(df)}")
            return df
        
        return None
    
    def process_data(self, raw_data):
        """Process raw API data into DataFrame"""
        df = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_asset_volume', 'taker_buy_base_asset_volume', 
                         'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Select only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def save_data(self, df):
        """Save data based on config output settings"""
        if df is None:
            return False
            
        output_config = self.config["output"]
        filename = output_config["filename"]
        file_format = output_config["format"].lower()
        
        try:
            if file_format == "csv":
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
            elif file_format == "json":
                df.to_json(filename, orient='records', date_format='iso')
                print(f"Data saved to {filename}")
            else:
                print(f"Unsupported format: {file_format}")
                return False
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def run(self):
        """Main execution function"""
        print(f"Starting data fetch with config:")
        print(f"Symbol: {self.config['symbol']}")
        print(f"Timeframe: {self.config['timeframe']}")
        print(f"Limit: {self.config['limit']}")
        print("-" * 40)
        
        # Fetch data
        df = self.fetch_historical_data()
        
        if df is not None:
            print(f"Successfully fetched {len(df)} records")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Display first few rows
            print("\nFirst 5 records:")
            print(df.head().to_string(index=False))
            
            # Save data
            self.save_data(df)
            return df
        else:
            print("Failed to fetch data")
            return None

def main():
    """Main function"""
    fetcher = BinanceHistoricalFetcher()
    data = fetcher.run()
    
if __name__ == "__main__":
    main()