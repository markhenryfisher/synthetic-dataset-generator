# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:41:58 2026

@author: mhf
"""
import os

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
    