#!/usr/bin/env python3
"""
write_txt_tool.py - Tool for writing content to text files as short-term memory storage

This tool can write content to text files as short-term memory storage, supporting both append and overwrite modes.
"""

import os
import argparse
from datetime import datetime
from typing import Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class WriteMemoryToolInput(BaseModel):
    content: str = Field(..., description="Content to write")
    file_path: Optional[str] = Field(None, description="Optional, target file path (if not specified, will use default path)")
    mode: str = Field("append", description="Write mode, 'append' or 'overwrite'")
    add_timestamp: bool = Field(True, description="Whether to add timestamp before content")


class ReadMemoryToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read")


class ListMemoriesToolInput(BaseModel):
    memory_dir: Optional[str] = Field(None, description="Memory directory path (if not specified, uses default path)")


class WriteMemoryTool(BaseTool):
    name: str = "write_memory"
    description: str = (
        "Write content to text files as short-term memory storage. "
        "Supports both append and overwrite modes, can automatically create directory structure. "
        "If file path is not specified, will create a new file in the memory folder under the current directory. "
        "Can choose whether to add timestamp."
    )
    args_schema: type = WriteMemoryToolInput

    def _run(self, content: str, file_path: Optional[str] = None, mode: str = "append", add_timestamp: bool = True) -> str:
        """
        Write content to a text file
        
        Args:
            content: Content to write
            file_path: Optional, target file path (if not specified, will use default path)
            mode: Write mode, "append" or "overwrite"
            add_timestamp: Whether to add timestamp before content
            
        Returns:
            Operation result message
        """
        try:
            # If file path is not specified, use default path
            if not file_path:
                # Create a "memory" folder in the current directory
                memory_dir = os.path.join(os.getcwd(), "memory")
                os.makedirs(memory_dir, exist_ok=True)
                
                # Use date time as file name
                now = datetime.now()
                file_name = f"memory_{now.strftime('%Y%m%d_%H%M%S')}.txt"
                file_path = os.path.join(memory_dir, file_name)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Prepare content to write
            write_content = content
            if add_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_content = f"[{timestamp}] {content}"
            
            # Write to file based on mode
            write_mode = "a" if mode == "append" else "w"
            with open(file_path, write_mode, encoding="utf-8") as f:
                f.write(write_content)
                
                # If append mode, add newline (unless content already ends with newline)
                if mode == "append" and not content.endswith("\n"):
                    f.write("\n")
            
            # Get file information
            file_size = os.path.getsize(file_path)
            file_size_str = f"{file_size} bytes"
            if file_size > 1024:
                file_size_str = f"{file_size/1024:.1f} KB"
            if file_size > 1024 * 1024:
                file_size_str = f"{file_size/(1024*1024):.1f} MB"
            
            return f"Successfully wrote content to file: {os.path.abspath(file_path)} (size: {file_size_str})"
        
        except Exception as e:
            return f"Error writing file: {str(e)}"

class ReadMemoryTool(BaseTool):
    name: str = "read_memory"
    description: str = (
        "Read content from text files. "
        "Can read the entire file or only the first few lines. "
        "Supports reading large files, will automatically truncate if content is too long."
    )
    args_schema: type = ReadMemoryToolInput

    def _run(self, file_path: str) -> str:
        """
        Read content from a text file
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            str: File content or error message
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File does not exist: {file_path}"
            
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # If content is too long, truncate it
            max_length = 10000  # Maximum 10,000 characters
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated...]"
            
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"


class ListMemoriesTool(BaseTool):
    name: str = "list_memories"
    description: str = (
        "List all memory files in the memory directory. "
        "Can specify a custom memory directory path. "
        "Returns a list of file names and their sizes."
    )
    args_schema: type = ListMemoriesToolInput

    def _run(self, memory_dir: Optional[str] = None) -> str:
        """
        List all memory files in the memory directory
        
        Args:
            memory_dir: Optional, memory directory path
            
        Returns:
            str: List of memory files and their sizes
        """
        try:
            # If memory directory is not specified, use default path
            if not memory_dir:
                memory_dir = os.path.join(os.getcwd(), "memory")
            
            # Check if directory exists
            if not os.path.exists(memory_dir):
                return f"Memory directory does not exist: {memory_dir}"
            
            # Get all files in the directory
            files = []
            for filename in os.listdir(memory_dir):
                file_path = os.path.join(memory_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    files.append((filename, file_size))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(memory_dir, x[0])), reverse=True)
            
            if not files:
                return "No memory files found."
            
            # Format the result
            result = f"Found {len(files)} memory files:\n\n"
            for filename, file_size in files:
                file_size_str = f"{file_size} bytes"
                if file_size > 1024:
                    file_size_str = f"{file_size/1024:.1f} KB"
                elif file_size > 1024 * 1024:
                    file_size_str = f"{file_size/(1024*1024):.1f} MB"
                
                file_path = os.path.join(memory_dir, filename)
                mtime = os.path.getmtime(file_path)
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                result += f"- {filename} ({file_size_str}, modified: {mtime_str})\n"
            
            return result
        except Exception as e:
            return f"Error listing memory files: {str(e)}"


# For backward compatibility, keep the original function
def write_to_file(
    content: str, 
    file_path: Optional[str] = None, 
    mode: str = "append",
    add_timestamp: bool = True
) -> str:
    """Backward compatible function interface"""
    tool = WriteMemoryTool()
    return tool._run(content, file_path, mode, add_timestamp)


def read_memory_file(file_path: str) -> str:
    """Backward compatible function interface"""
    tool = ReadMemoryTool()
    return tool._run(file_path)


def list_memory_files(memory_dir: Optional[str] = None) -> str:
    """Backward compatible function interface"""
    tool = ListMemoriesTool()
    return tool._run(memory_dir)


if __name__ == "__main__":
    """Command line entry point"""
    parser = argparse.ArgumentParser(description="Write content to text files as short-term memory storage")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Write command
    write_parser = subparsers.add_parser("write", help="Write content to a text file")
    write_parser.add_argument("content", help="Content to write")
    write_parser.add_argument("--file", help="Target file path (optional)")
    write_parser.add_argument("--mode", choices=["append", "overwrite"], default="append", help="Write mode (default: append)")
    write_parser.add_argument("--no-timestamp", action="store_true", help="Do not add timestamp")
    
    # Read command
    read_parser = subparsers.add_parser("read", help="Read memory file content")
    read_parser.add_argument("file", help="File path to read")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all memory files in the memory directory")
    list_parser.add_argument("--dir", help="Memory directory path (optional)")
    
    args = parser.parse_args()
    
    if args.command == "write":
        result = write_to_file(
            args.content,
            args.file,
            args.mode,
            not args.no_timestamp
        )
        print(result)
    elif args.command == "read":
        result = read_memory_file(args.file)
        print(result)
    elif args.command == "list":
        result = list_memory_files(args.dir)
        print(result)
    else:
        parser.print_help()