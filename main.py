import subprocess

def run_script(script_name: str):
    """Run a Python script as a subprocess and handle errors."""
    try:
        print(f"🔹 Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"✅ Finished {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script_name}: {e}")
        exit(1)

def main():
    # Step 1: Fetch Data
    run_script("fetch_data.py")
    
    # Step 2: Run Backtest
    run_script("backtest.py")

if __name__ == "__main__":
    main()
