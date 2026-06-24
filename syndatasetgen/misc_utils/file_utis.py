# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:41:58 2026

@author: mhf
"""
import os
import json

def list_files_by_extension(directory, extension):
    """
    List all files in the given directory that match the specified extension.
    
    :param directory: Path to the directory to search.
    :param extension: File extension to filter by (e.g., '.txt').
    :return: List of matching file names.
    """
    # Validate inputs
    if not isinstance(directory, str) or not isinstance(extension, str):
        raise TypeError("Both directory and extension must be strings.")
    
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Ensure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension

    try:
        # List and filter files
        return [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extension.lower())
        ]
    except PermissionError:
        print(f"Permission denied: Cannot access {directory}")
        return []
    except OSError as e:
        print(f"Error accessing directory: {e}")
        return []
    

def save_dict_to_json(data, filename):
    """
    Save a Python dictionary to a JSON file with validation and error handling.
    """
    # Validate the dictionary
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    # Validate filename
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")

    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    try:
        # Write dictionary to JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Dictionary successfully written to {filename}")

    except (OSError, TypeError, ValueError) as e:
        print(f"Error writing JSON file: {e}")


# Example usage
if __name__ == "__main__":
    sample_dict = {"name": "Alice", "age": 30, "languages": ["Python", "JavaScript"]}

    save_dict_to_json(sample_dict, "output.json")
