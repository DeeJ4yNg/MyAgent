"""
Code Execution Tool

This tool helps execute code snippets and show the results.
"""

import os
import subprocess
import sys
import tempfile
import json
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def execute_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """
    Execute a code snippet and return the output.
    
    Args:
        code: Code snippet to execute
        language: Programming language (python, javascript, java, cpp, bash)
        timeout: Timeout in seconds (default: 30)
    
    Returns:
        Execution output or error message
    """
    console = Console()
    
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix=_get_file_extension(language), delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    
    try:
        # Execute the code based on language
        if language.lower() == "python":
            return _execute_python_code(temp_file_path, timeout)
        elif language.lower() in ["javascript", "js"]:
            return _execute_javascript_code(temp_file_path, timeout)
        elif language.lower() == "java":
            return _execute_java_code(temp_file_path, timeout)
        elif language.lower() in ["cpp", "c++", "c"]:
            return _execute_cpp_code(temp_file_path, timeout, language.lower())
        elif language.lower() == "bash":
            return _execute_bash_code(temp_file_path, timeout)
        elif language.lower() in ["powershell", "ps1", "ps"]:
            return _execute_powershell_code(temp_file_path, timeout)
        else:
            return f"Error: Unsupported language '{language}'. Supported: python, powershell, javascript, java, cpp, c, bash."
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
            
            # Clean up additional files created during execution
            if language.lower() == "java":
                # Remove .class files
                class_file = temp_file_path.replace('.java', '.class')
                if os.path.exists(class_file):
                    os.unlink(class_file)
            elif language.lower() in ["cpp", "c++", "c"]:
                # Remove executable files
                exe_file = temp_file_path.replace(_get_file_extension(language), '')
                if os.path.exists(exe_file):
                    os.unlink(exe_file)
        except:
            pass


@tool
def run_script(file_path: str, arguments: str = "", timeout: int = 30) -> str:
    """
    Run a script file with optional arguments.
    
    Args:
        file_path: Path to the script file to run
        arguments: Command-line arguments to pass to the script
        timeout: Timeout in seconds (default: 30)
    
    Returns:
        Script output or error message
    """
    console = Console()
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    # Determine language from file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Build command based on file extension
    if file_ext == ".py":
        command = [sys.executable, file_path]
    elif file_ext in [".js", ".jsx"]:
        command = ["node", file_path]
    elif file_ext in [".ts", ".tsx"]:
        command = ["npx", "ts-node", file_path]
    elif file_ext == ".java":
        # Compile first
        compile_result = subprocess.run(
            ["javac", file_path],
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            return f"Compilation error:\n{compile_result.stderr}"
        
        # Get class name from file
        class_name = os.path.basename(file_path).replace('.java', '')
        command = ["java", "-cp", os.path.dirname(file_path), class_name]
    elif file_ext in [".cpp", ".c++", ".cxx"]:
        # Compile first
        exe_path = file_path.replace(file_ext, '')
        compile_result = subprocess.run(
            ["g++", "-o", exe_path, file_path],
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            return f"Compilation error:\n{compile_result.stderr}"
        
        command = [exe_path]
    elif file_ext == ".c":
        # Compile first
        exe_path = file_path.replace(file_ext, '')
        compile_result = subprocess.run(
            ["gcc", "-o", exe_path, file_path],
            capture_output=True,
            text=True
        )
        if compile_result.returncode != 0:
            return f"Compilation error:\n{compile_result.stderr}"
        
        command = [exe_path]
    elif file_ext in [".sh", ".bash"]:
        command = ["bash", file_path]
    elif file_ext in [".ps1", ".ps"]:
        command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", file_path]
    else:
        return f"Error: Unsupported file extension '{file_ext}'."
    
    # Add arguments if provided
    if arguments:
        command.extend(arguments.split())
    
    # Execute the command
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(file_path)
        )
        
        output = f"## Execution Results for {os.path.basename(file_path)}\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing script: {str(e)}"


@tool
def run_command(command: str, timeout: int = 30, working_directory: str = ".") -> str:
    """
    Run a shell command and return the output.
    
    Args:
        command: Shell command to execute
        timeout: Timeout in seconds (default: 30)
        working_directory: Directory to run the command in (default: current directory)
    
    Returns:
        Command output or error message
    """
    console = Console()
    
    # Validate working directory
    if not os.path.exists(working_directory):
        return f"Error: Working directory '{working_directory}' does not exist."
    
    # Execute the command
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_directory
        )
        
        output = f"## Command Execution Results\n\n"
        output += f"Command: `{command}`\n"
        output += f"Working Directory: `{working_directory}`\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"


def _get_file_extension(language: str) -> str:
    """Get file extension for a programming language."""
    extensions = {
        "python": ".py",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "java": ".java",
        "cpp": ".cpp",
        "c++": ".cpp",
        "c": ".c",
        "bash": ".sh",
        "shell": ".sh",
        "powershell": ".ps1",
        "ps1": ".ps1",
        "ps": ".ps1"
    }
    return extensions.get(language.lower(), ".txt")


def _execute_python_code(file_path: str, timeout: int) -> str:
    """Execute Python code from a file."""
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"## Python Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


def _execute_javascript_code(file_path: str, timeout: int) -> str:
    """Execute JavaScript code from a file."""
    try:
        result = subprocess.run(
            ["node", file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"## JavaScript Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: JavaScript execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing JavaScript code: {str(e)}"


def _execute_java_code(file_path: str, timeout: int) -> str:
    """Execute Java code from a file."""
    try:
        # Compile the Java code
        compile_result = subprocess.run(
            ["javac", file_path],
            capture_output=True,
            text=True
        )
        
        if compile_result.returncode != 0:
            return f"## Java Compilation Error\n\n```\n{compile_result.stderr}\n```\n\n"
        
        # Get class name from file
        class_name = os.path.basename(file_path).replace('.java', '')
        class_path = os.path.dirname(file_path)
        
        # Run the compiled Java code
        result = subprocess.run(
            ["java", "-cp", class_path, class_name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"## Java Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Java execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Java code: {str(e)}"


def _execute_cpp_code(file_path: str, timeout: int, language: str) -> str:
    """Execute C/C++ code from a file."""
    try:
        # Compile the C/C++ code
        exe_path = file_path.replace(_get_file_extension(language), '')
        
        if language in ["cpp", "c++"]:
            compile_result = subprocess.run(
                ["g++", "-o", exe_path, file_path],
                capture_output=True,
                text=True
            )
        else:  # C
            compile_result = subprocess.run(
                ["gcc", "-o", exe_path, file_path],
                capture_output=True,
                text=True
            )
        
        if compile_result.returncode != 0:
            return f"## {language.upper()} Compilation Error\n\n```\n{compile_result.stderr}\n```\n\n"
        
        # Run the compiled C/C++ code
        result = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"## {language.upper()} Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: {language.upper()} execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing {language.upper()} code: {str(e)}"


def _execute_bash_code(file_path: str, timeout: int) -> str:
    """Execute Bash code from a file."""
    try:
        result = subprocess.run(
            ["bash", file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = f"## Bash Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Bash execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing Bash code: {str(e)}"


def _execute_powershell_code(file_path: str, timeout: int) -> str:
    """Execute PowerShell code from a file."""
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True  # Use shell=True on Windows for better PowerShell support
        )
        
        output = f"## PowerShell Execution Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: PowerShell execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing PowerShell code: {str(e)}"