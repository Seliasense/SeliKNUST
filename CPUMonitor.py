import psutil
import datetime
import time
import threading
import csv
import config

def monitor_cpu_performance_realtime(filename="cpu_realtime.csv", duration=60, interval=1):
    """
    Monitor CPU performance when triggered and write to CSV file
    """
    try:
        with open(filename, 'a', newline='') as f:  # Use 'a' for append mode, newline='' for proper CSV formatting
            writer = csv.writer(f)
            
            # Write header if file is new
            if f.tell() == 0:
                writer.writerow(["timestamp", "cpu_percent", "memory_percent", "cpu_freq", "load_avg"])
            
            start_time = time.time()
            end_time = start_time + duration
            
            print(f"CPU monitoring started for {duration} seconds...")
            
            while time.time() < end_time and config.CPU_MONITOR_TRIGGER:
                timestamp = datetime.datetime.now().isoformat()
                cpu_percent = psutil.cpu_percent(interval=interval)
                # config.Mcpu_percent = cpu_percent
                memory_percent = psutil.virtual_memory().percent
                # config.Mmemory_percent = memory_percent
                
                # Additional metrics
                try:
                    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
                    # config.Mcpu_freq = cpu_freq
                except:
                    cpu_freq = "N/A"
                    # config.Mcpu_freq = cpu_freq
                
                try:
                    load_avg = psutil.getloadavg()[0]
                    # config.Mload_avg =  load_avg
                except:
                    load_avg = "N/A"
                    # config.Mload_avg =  load_avg
                
                # Write data as CSV row
                writer.writerow([timestamp, cpu_percent, memory_percent, cpu_freq, load_avg])
                f.flush()
                
                print(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
                
        print("CPU monitoring completed")
        
    except Exception as e:
        print(f" Error during CPU monitoring: {e}")

def trigger_cpu_monitoring(enable=True, duration=60, filename="cpu_monitor.csv"):
    """
    Trigger CPU monitoring from anywhere in your script
    """
    
    config.CPU_MONITOR_TRIGGER = enable
    
    if enable:
        # Start monitoring in a separate thread to avoid blocking
        thread = threading.Thread(
            target=monitor_cpu_performance_realtime,
            args=(filename, duration),
            daemon=True
        )
        thread.start()
        return thread
    return None

