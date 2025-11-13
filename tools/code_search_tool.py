"""
Code Search Tool

This tool searches for code patterns, functions, classes, or text within code files.
"""

import os
import re
from typing import Dict, List, Optional, Any, Tuple
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def search_code(search_pattern: str, search_directory: str = ".", 
                file_pattern: str = "*", case_sensitive: bool = False,
                search_type: str = "text") -> str:
    """
    Search for code patterns, functions, classes, or text within code files.
    
    Args:
        search_pattern: Pattern to search for (regex supported)
        search_directory: Directory to search in (default: current directory)
        file_pattern: File pattern to match (e.g., "*.py", "*.js")
        case_sensitive: Whether the search should be case sensitive
        search_type: Type of search ("text", "function", "class", "import")
    
    Returns:
        Search results with file paths and line numbers
    """
    console = Console()
    
    # Validate search directory
    if not os.path.exists(search_directory):
        return f"Error: Directory '{search_directory}' does not exist."
    
    # Compile regex pattern
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex_pattern = re.compile(search_pattern, flags)
    except re.error as e:
        return f"Error in regex pattern: {str(e)}"
    
    # Find matching files
    matching_files = []
    
    # Walk through directory tree
    for root, dirs, files in os.walk(search_directory):
        # Skip hidden directories and common build/cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  d not in ['node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build']]
        
        for file in files:
            # Check if file matches the pattern
            if file_pattern == "*" or _matches_file_pattern(file, file_pattern):
                file_path = os.path.join(root, file)
                matching_files.append(file_path)
    
    # Search within files
    results = []
    
    for file_path in matching_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            file_matches = []
            
            # Search based on type
            if search_type == "function":
                file_matches = _search_functions(lines, regex_pattern, file_path)
            elif search_type == "class":
                file_matches = _search_classes(lines, regex_pattern, file_path)
            elif search_type == "import":
                file_matches = _search_imports(lines, regex_pattern, file_path)
            else:  # text search
                file_matches = _search_text(lines, regex_pattern, file_path)
            
            if file_matches:
                results.extend(file_matches)
                
        except Exception as e:
            # Skip files that can't be read
            continue
    
    # Format results
    if not results:
        return f"No matches found for pattern '{search_pattern}' in directory '{search_directory}'."
    
    # Group results by file
    results_by_file = {}
    for match in results:
        file_path = match['file']
        if file_path not in results_by_file:
            results_by_file[file_path] = []
        results_by_file[file_path].append(match)
    
    # Format output
    output = f"## Search Results for '{search_pattern}' ({search_type})\n\n"
    
    for file_path, matches in results_by_file.items():
        output += f"### {file_path}\n\n"
        
        for match in matches:
            line_num = match['line']
            line_content = match['content'].strip()
            
            if search_type == "function":
                output += f"- **Function** (Line {line_num}): {line_content}\n"
            elif search_type == "class":
                output += f"- **Class** (Line {line_num}): {line_content}\n"
            elif search_type == "import":
                output += f"- **Import** (Line {line_num}): {line_content}\n"
            else:  # text search
                output += f"- Line {line_num}: {line_content}\n"
        
        output += "\n"
    
    return output


@tool
def find_function_usage(function_name: str, search_directory: str = ".",
                       file_pattern: str = "*") -> str:
    """
    Find all usages of a specific function within code files.
    
    Args:
        function_name: Name of the function to find usages for
        search_directory: Directory to search in (default: current directory)
        file_pattern: File pattern to match (e.g., "*.py", "*.js")
    
    Returns:
        List of function usages with file paths and line numbers
    """
    console = Console()
    
    # Create a regex pattern to match function calls
    # This is a simplified pattern and might not catch all cases
    pattern = rf"{function_name}\s*\("
    
    return search_code(pattern, search_directory, file_pattern, False, "text")


@tool
def find_class_usage(class_name: str, search_directory: str = ".",
                    file_pattern: str = "*") -> str:
    """
    Find all usages of a specific class within code files.
    
    Args:
        class_name: Name of the class to find usages for
        search_directory: Directory to search in (default: current directory)
        file_pattern: File pattern to match (e.g., "*.py", "*.js")
    
    Returns:
        List of class usages with file paths and line numbers
    """
    console = Console()
    
    # Create a regex pattern to match class usage
    # This is a simplified pattern and might not catch all cases
    patterns = [
        rf"{class_name}\s*\(",      # Class instantiation
        rf"class\s+\w+\s*\(\s*{class_name}\s*\)",  # Class inheritance
        rf":\s*{class_name}\s*$",   # Type annotation
        rf"as\s+{class_name}",      # Import alias
        rf"\.{class_name}\s*\(",    # Method call on instance
    ]
    
    all_results = []
    
    for pattern in patterns:
        result = search_code(pattern, search_directory, file_pattern, False, "text")
        if "No matches found" not in result:
            all_results.append(result)
    
    if not all_results:
        return f"No usages of class '{class_name}' found in directory '{search_directory}'."
    
    # Combine results
    output = f"## Class Usage Results for '{class_name}'\n\n"
    output += "\n".join(all_results)
    
    return output


def _matches_file_pattern(filename: str, pattern: str) -> bool:
    """Check if a filename matches a pattern (supports wildcards)."""
    # Convert pattern to regex
    regex_pattern = pattern.replace(".", r"\.").replace("*", r".*").replace("?", r".")
    return re.match(regex_pattern, filename) is not None


def _search_functions(lines: List[str], pattern: re.Pattern, file_path: str) -> List[Dict[str, Any]]:
    """Search for function definitions in code."""
    results = []
    
    # Function definition patterns for different languages
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.py':
        func_pattern = re.compile(r"def\s+(\w+)\s*\(", re.IGNORECASE)
    elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
        func_pattern = re.compile(r"(?:function\s+(\w+)\s*\(|(\w+)\s*:\s*function|const\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))", re.IGNORECASE)
    elif file_ext == '.java':
        func_pattern = re.compile(r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(", re.IGNORECASE)
    elif file_ext in ['.cpp', '.c', '.cc', '.cxx']:
        func_pattern = re.compile(r"(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{?", re.IGNORECASE)
    else:
        # Generic pattern
        func_pattern = re.compile(r"(?:function|def)\s+(\w+)\s*\(", re.IGNORECASE)
    
    for i, line in enumerate(lines, 1):
        # Check for function definition
        func_match = func_pattern.search(line)
        if func_match:
            func_name = func_match.group(1)
            if pattern.search(func_name):
                results.append({
                    'file': file_path,
                    'line': i,
                    'content': line.strip(),
                    'type': 'function'
                })
    
    return results


def _search_classes(lines: List[str], pattern: re.Pattern, file_path: str) -> List[Dict[str, Any]]:
    """Search for class definitions in code."""
    results = []
    
    # Class definition patterns for different languages
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.py':
        class_pattern = re.compile(r"class\s+(\w+)", re.IGNORECASE)
    elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
        class_pattern = re.compile(r"class\s+(\w+)", re.IGNORECASE)
    elif file_ext == '.java':
        class_pattern = re.compile(r"(?:public\s+)?class\s+(\w+)", re.IGNORECASE)
    elif file_ext in ['.cpp', '.c', '.cc', '.cxx']:
        class_pattern = re.compile(r"class\s+(\w+)", re.IGNORECASE)
    else:
        # Generic pattern
        class_pattern = re.compile(r"class\s+(\w+)", re.IGNORECASE)
    
    for i, line in enumerate(lines, 1):
        # Check for class definition
        class_match = class_pattern.search(line)
        if class_match:
            class_name = class_match.group(1)
            if pattern.search(class_name):
                results.append({
                    'file': file_path,
                    'line': i,
                    'content': line.strip(),
                    'type': 'class'
                })
    
    return results


def _search_imports(lines: List[str], pattern: re.Pattern, file_path: str) -> List[Dict[str, Any]]:
    """Search for import statements in code."""
    results = []
    
    # Import patterns for different languages
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.py':
        import_patterns = [
            re.compile(r"import\s+(.+)", re.IGNORECASE),
            re.compile(r"from\s+(.+)\s+import", re.IGNORECASE)
        ]
    elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
        import_patterns = [
            re.compile(r"import\s+.+\s+from\s+(.+)", re.IGNORECASE),
            re.compile(r"const\s+.+\s*=\s*require\((.+)\)", re.IGNORECASE)
        ]
    elif file_ext == '.java':
        import_patterns = [
            re.compile(r"import\s+(.+);", re.IGNORECASE)
        ]
    elif file_ext in ['.cpp', '.c', '.cc', '.cxx']:
        import_patterns = [
            re.compile(r"#include\s+[<\"](.+)[>\"]", re.IGNORECASE)
        ]
    else:
        # Generic pattern
        import_patterns = [
            re.compile(r"(?:import|include|require)\s+(.+)", re.IGNORECASE)
        ]
    
    for i, line in enumerate(lines, 1):
        for import_pattern in import_patterns:
            # Check for import statement
            import_match = import_pattern.search(line)
            if import_match:
                import_statement = import_match.group(1)
                if pattern.search(import_statement):
                    results.append({
                        'file': file_path,
                        'line': i,
                        'content': line.strip(),
                        'type': 'import'
                    })
    
    return results


def _search_text(lines: List[str], pattern: re.Pattern, file_path: str) -> List[Dict[str, Any]]:
    """Search for text pattern in code."""
    results = []
    
    for i, line in enumerate(lines, 1):
        # Check for pattern match
        match = pattern.search(line)
        if match:
            results.append({
                'file': file_path,
                'line': i,
                'content': line.strip(),
                'type': 'text'
            })
    
    return results