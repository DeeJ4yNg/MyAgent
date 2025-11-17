"""
Python script to check and return device ID from dsregcmd /status command
"""

#!/usr/bin/env python3
"""
Script to check and return the device ID from dsregcmd /status command.
This script is designed for Windows systems where dsregcmd is available.
"""

import subprocess
import re
import sys


def get_device_id():
    """
    Execute dsregcmd /status and extract the Device ID from the output.
    
    Returns:
        str: The device ID if found, None otherwise
    """
    try:
        # Run the dsregcmd /status command
        result = subprocess.run(
            ['dsregcmd', '/status'],
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )
        
        # Parse the output to find the Device ID
        output = result.stdout
        
        # Look for Device ID pattern (usually in format: "Device ID: XXXX")
        device_id_match = re.search(r'Device ID:\s*(.+)', output, re.IGNORECASE)
        
        if device_id_match:
            device_id = device_id_match.group(1).strip()
            return device_id
        else:
            print("Device ID not found in dsregcmd output.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running dsregcmd: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except FileNotFoundError:
        print("dsregcmd command not found. This script requires Windows with dsregcmd available.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    """Main function to run the script."""
    print("Checking device ID...")
    device_id = get_device_id()
    
    if device_id:
        print(f"Device ID: {device_id}")
        return device_id
    else:
        print("Failed to retrieve device ID.")
        sys.exit(1)


if __name__ == "__main__":
    main()