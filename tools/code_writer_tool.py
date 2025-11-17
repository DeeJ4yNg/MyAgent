"""
Code Writer Tool

This tool helps write new code files or modify existing ones based on requirements.
"""

import os
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def write_code(file_path: str, code_content: str, description: str = "", overwrite: bool = False) -> str:
    """
    Write code to a file, creating it if it doesn't exist or modifying it if it does.
    
    Args:
        file_path: Path to the code file to write or modify
        code_content: The code content to write to the file
        description: Description of what the code does (for documentation)
        overwrite: Whether to overwrite the file if it already exists
    
    Returns:
        Success message with details about the operation
    """
    console = Console()
    
    # Check if file exists and overwrite is False
    if os.path.exists(file_path) and not overwrite:
        return f"Error: File '{file_path}' already exists. Use overwrite=True to replace it."
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            return f"Error creating directory '{dir_path}': {str(e)}"
    
    # Write the code to the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Create a header comment if description is provided
    if description and file_ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.cc', '.cxx', '.ps1', '.ps']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # Add header comment based on file type
            if file_ext == '.py':
                header = f'"""\n{description}\n"""\n\n'
            elif file_ext in ['.ps1', '.ps']:
                header = f'<#\n{description}\n#>\n\n'
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                header = f'/*\n{description}\n*/\n\n'
            elif file_ext == '.java':
                header = f'/*\n * {description}\n */\n\n'
            else:  # C/C++
                header = f'/*\n * {description}\n */\n\n'
            
            # Write the file with header
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header + existing_content)
                
        except Exception as e:
            return f"Error adding header to file '{file_path}': {str(e)}"
    
    # Return success message
    action = "created" if not os.path.exists(file_path + ".bak") else "overwritten"
    return f"Success: File '{file_path}' {action} with {len(code_content)} characters of code."


@tool
def append_to_file(file_path: str, content: str, position: str = "end") -> str:
    """
    Append content to an existing file at the specified position.
    
    Args:
        file_path: Path to the file to append content to
        content: Content to append to the file
        position: Where to append the content ("beginning", "end", or a line number)
    
    Returns:
        Success message with details about the operation
    """
    console = Console()
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    try:
        # Read existing content
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # Determine where to insert the new content
        if position == "beginning":
            new_content = content + "\n" + existing_content
        elif position == "end":
            new_content = existing_content + "\n" + content
        else:
            # Assume position is a line number
            try:
                line_num = int(position)
                lines = existing_content.split('\n')
                if line_num <= 0 or line_num > len(lines) + 1:
                    return f"Error: Invalid line number {line_num}. File has {len(lines)} lines."
                
                # Insert content at the specified line
                lines.insert(line_num - 1, content)
                new_content = '\n'.join(lines)
            except ValueError:
                return f"Error: Invalid position '{position}'. Use 'beginning', 'end', or a line number."
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    except Exception as e:
        return f"Error modifying file '{file_path}': {str(e)}"
    
    return f"Success: Content appended to '{file_path}' at position '{position}'."


@tool
def create_function(file_path: str, function_name: str, function_code: str, 
                   language: str = "python", description: str = "") -> str:
    """
    Create a new function in an existing file or create a new file with the function.
    
    Args:
        file_path: Path to the file where the function should be added
        function_name: Name of the function to create
        function_code: Body of the function (without signature)
        language: Programming language (python, javascript, java, cpp)
        description: Description of what the function does
    
    Returns:
        Success message with details about the operation
    """
    console = Console()
    
    # Determine function signature based on language
    if language.lower() == "python":
        signature = f"def {function_name}():\n"
        # Indent the function code
        indented_code = '\n'.join(['    ' + line for line in function_code.split('\n')])
        full_function = signature + indented_code
    elif language.lower() in ["powershell", "ps1", "ps"]:
        signature = f"function {function_name} {{\n"
        indented_code = '\n'.join(['    ' + line for line in function_code.split('\n')])
        full_function = signature + indented_code + "\n}"
    elif language.lower() in ["javascript", "js"]:
        signature = f"function {function_name}() {{\n"
        indented_code = '\n'.join(['    ' + line for line in function_code.split('\n')])
        full_function = signature + indented_code + "\n}"
    elif language.lower() == "java":
        signature = f"public void {function_name}() {{\n"
        indented_code = '\n'.join(['    ' + line for line in function_code.split('\n')])
        full_function = signature + indented_code + "\n}"
    elif language.lower() in ["cpp", "c++"]:
        signature = f"void {function_name}() {{\n"
        indented_code = '\n'.join(['    ' + line for line in function_code.split('\n')])
        full_function = signature + indented_code + "\n}"
    else:
        return f"Error: Unsupported language '{language}'. Supported: python, powershell, javascript, java, cpp"
    
    # Add doc comment if description is provided
    if description:
        if language.lower() == "python":
            docstring = f'    """\n    {description}\n    """\n'
            full_function = signature + docstring + indented_code
        elif language.lower() in ["powershell", "ps1", "ps"]:
            docstring = f"    <#\n    {description}\n    #>\n"
            full_function = docstring + signature + indented_code + "\n}"
        elif language.lower() in ["javascript", "js"]:
            docstring = f"    /**\n     * {description}\n     */\n"
            full_function = docstring + signature + indented_code + "\n}"
        elif language.lower() == "java":
            docstring = f"    /**\n     * {description}\n     */\n"
            full_function = docstring + signature + indented_code + "\n}"
        elif language.lower() in ["cpp", "c++"]:
            docstring = f"    /**\n     * {description}\n     */\n"
            full_function = docstring + signature + indented_code + "\n}"
    
    # Check if file exists
    if os.path.exists(file_path):
        # Append to existing file
        return append_to_file(file_path, full_function, "end")
    else:
        # Create new file with the function
        return write_code(file_path, full_function, description, overwrite=False)


@tool
def create_class(file_path: str, class_name: str, class_code: str, 
                language: str = "python", description: str = "") -> str:
    """
    Create a new class in an existing file or create a new file with the class.
    
    Args:
        file_path: Path to the file where the class should be added
        class_name: Name of the class to create
        class_code: Body of the class (without class declaration)
        language: Programming language (python, javascript, java, cpp)
        description: Description of what the class does
    
    Returns:
        Success message with details about the operation
    """
    console = Console()
    
    # Determine class declaration based on language
    if language.lower() == "python":
        declaration = f"class {class_name}:\n"
        # Indent the class code
        indented_code = '\n'.join(['    ' + line for line in class_code.split('\n')])
        full_class = declaration + indented_code
    elif language.lower() in ["javascript", "js"]:
        declaration = f"class {class_name} {{\n"
        indented_code = '\n'.join(['    ' + line for line in class_code.split('\n')])
        full_class = declaration + indented_code + "\n}"
    elif language.lower() == "java":
        declaration = f"public class {class_name} {{\n"
        indented_code = '\n'.join(['    ' + line for line in class_code.split('\n')])
        full_class = declaration + indented_code + "\n}"
    elif language.lower() in ["cpp", "c++"]:
        declaration = f"class {class_name} {{\npublic:\n"
        indented_code = '\n'.join(['    ' + line for line in class_code.split('\n')])
        full_class = declaration + indented_code + "\n};"
    else:
        return f"Error: Unsupported language '{language}'. Supported: python, javascript, java, cpp"
    
    # Add doc comment if description is provided
    if description:
        if language.lower() == "python":
            docstring = f'    """\n    {description}\n    """\n'
            full_class = declaration + docstring + indented_code
        elif language.lower() in ["javascript", "js"]:
            docstring = f"/**\n * {description}\n */\n"
            full_class = docstring + declaration + indented_code + "\n}"
        elif language.lower() == "java":
            docstring = f"/**\n * {description}\n */\n"
            full_class = docstring + declaration + indented_code + "\n}"
        elif language.lower() in ["cpp", "c++"]:
            docstring = f"/**\n * {description}\n */\n"
            full_class = docstring + declaration + indented_code + "\n};"
    
    # Check if file exists
    if os.path.exists(file_path):
        # Append to existing file
        return append_to_file(file_path, full_class, "end")
    else:
        # Create new file with the class
        return write_code(file_path, full_class, description, overwrite=False)