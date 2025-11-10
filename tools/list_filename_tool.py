#!/usr/bin/env python3
"""
list_filename_tool.py - Tool for listing all files and subdirectories

This tool can recursively list all files and subdirectories in a specified directory, supporting various filtering options.
"""

import os
from typing import List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ListFilesToolInput(BaseModel):
    directory: str = Field(".", description="Directory path to scan, defaults to current directory")
    recursive: bool = Field(True, description="Whether to recursively scan subdirectories")
    include_hidden: bool = Field(False, description="Whether to include hidden files and directories")
    file_types: Optional[List[str]] = Field(None, description="Optional, only list files with specified extensions, e.g. ['txt', 'py']")
    max_depth: Optional[int] = Field(None, description="Optional, limit recursion depth")


class ListFilesTool(BaseTool):
    name: str = "list_files"
    description: str = (
        "Recursively list all files and subdirectories in a specified directory. "
        "Supports various filtering options, including file type filtering, hidden file inclusion, recursion depth limitation, etc. "
        "Can display file sizes and directory structure, and provide statistics for files and directories."
    )
    args_schema: type = ListFilesToolInput

    def _run(
        self, 
        directory: str = ".", 
        recursive: bool = True, 
        include_hidden: bool = False,
        file_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None
    ) -> str:
        """
        List all files and subdirectories in a directory
        
        Args:
            directory: Directory path to scan
            recursive: Whether to recursively scan subdirectories
            include_hidden: Whether to include hidden files and directories
            file_types: Optional, only list files with specified extensions, e.g. ['.txt', '.py']
            max_depth: Optional, limit recursion depth
            
        Returns:
            Formatted list of files and directories
        """
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                return f"Error: Directory '{directory}' does not exist"
            
            # Check if it's a directory
            if not os.path.isdir(directory):
                return f"Error: '{directory}' is not a directory"
            
            # Normalize file type list
            if file_types:
                file_types = [ext.lower() if ext.startswith('.') else f".{ext.lower()}" for ext in file_types]
            
            result = []
            result.append(f"Directory: {os.path.abspath(directory)}")
            result.append("=" * 80)
            
            def scan_dir(current_dir: str, depth: int = 0):
                """Recursively scan directory"""
                if max_depth is not None and depth > max_depth:
                    return
                
                try:
                    items = []
                    # Get all items in the directory
                    for item in os.listdir(current_dir):
                        item_path = os.path.join(current_dir, item)
                        
                        # Skip hidden files (if not specified to include hidden files)
                        if not include_hidden and item.startswith('.'):
                            continue
                        
                        items.append((item, item_path, os.path.isdir(item_path)))
                    
                    # Sort by type and name: directories first, then files
                    items.sort(key=lambda x: (not x[2], x[0].lower()))
                    
                    # Display items
                    for name, path, is_dir in items:
                        indent = "  " * depth
                        icon = "📁" if is_dir else "📄"
                        
                        # If it's a file and file type filtering is specified, check extension
                        if not is_dir and file_types:
                            file_ext = os.path.splitext(name)[1].lower()
                            if file_ext not in file_types:
                                continue
                        
                        # Get file size (files only)
                        size_info = ""
                        if not is_dir:
                            try:
                                size = os.path.getsize(path)
                                if size < 1024:
                                    size_info = f" ({size} bytes)"
                                elif size < 1024 * 1024:
                                    size_info = f" ({size/1024:.1f} KB)"
                                else:
                                    size_info = f" ({size/(1024*1024):.1f} MB)"
                            except:
                                pass
                        
                        result.append(f"{indent}{icon} {name}{size_info}")
                        
                        # Recursively process subdirectories
                        if is_dir and recursive:
                            scan_dir(path, depth + 1)
                
                except PermissionError:
                    result.append(f"{'  ' * depth}⚠️ Insufficient permissions, unable to access this directory")
                except Exception as e:
                    result.append(f"{'  ' * depth}⚠️ Error: {str(e)}")
            
            # Start scanning
            scan_dir(directory)
            
            # Statistics
            total_files = len([line for line in result if "📄" in line])
            total_dirs = len([line for line in result if "📁" in line])
            result.append("")
            result.append(f"Total: {total_files} files, {total_dirs} directories")
            
            return "\n".join(result)
        
        except Exception as e:
            return f"Error scanning directory: {str(e)}"


# For backward compatibility, keep the original function
def list_files(
    directory: str, 
    recursive: bool = True, 
    include_hidden: bool = False,
    file_types: Optional[List[str]] = None,
    max_depth: Optional[int] = None
) -> str:
    """Backward compatible function interface"""
    tool = ListFilesTool()
    return tool._run(directory, recursive, include_hidden, file_types, max_depth)


if __name__ == "__main__":
    import argparse
    
    # Command line entry point
    parser = argparse.ArgumentParser(description="List all files and subdirectories in a directory")
    parser.add_argument("directory", nargs="?", default=".", help="Directory path to scan (defaults to current directory)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recursively scan subdirectories")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files and directories")
    parser.add_argument("--file-types", nargs="+", help="Only list files with specified extensions, e.g.: txt py pdf")
    parser.add_argument("--max-depth", type=int, help="Limit recursion depth")
    
    args = parser.parse_args()
    
    content = list_files(
        args.directory,
        recursive=not args.no_recursive,
        include_hidden=args.include_hidden,
        file_types=args.file_types,
        max_depth=args.max_depth
    )
    print(content)