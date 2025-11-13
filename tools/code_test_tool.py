"""
Code Test Tool

This tool helps run tests on code and analyze test results.
"""

import os
import subprocess
import sys
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def run_code_tests(test_directory: str = ".", test_framework: str = "auto") -> str:
    """
    Run tests on code using various testing frameworks.
    
    Args:
        test_directory: Directory containing tests (default: current directory)
        test_framework: Testing framework to use (auto, pytest, unittest, jest, mocha, junit)
    
    Returns:
        Test results with details about passed/failed tests
    """
    console = Console()
    
    # Validate test directory
    if not os.path.exists(test_directory):
        return f"Error: Directory '{test_directory}' does not exist."
    
    # Auto-detect testing framework if needed
    if test_framework == "auto":
        test_framework = _detect_test_framework(test_directory)
        if not test_framework:
            return "Error: Could not auto-detect testing framework. Please specify one explicitly."
    
    # Run tests based on framework
    if test_framework == "pytest":
        return _run_pytest(test_directory)
    elif test_framework == "unittest":
        return _run_unittest(test_directory)
    elif test_framework == "jest":
        return _run_jest(test_directory)
    elif test_framework == "mocha":
        return _run_mocha(test_directory)
    elif test_framework == "junit":
        return _run_junit(test_directory)
    else:
        return f"Error: Unsupported testing framework '{test_framework}'. Supported: auto, pytest, unittest, jest, mocha, junit."


@tool
def create_test_file(file_path: str, test_content: str, test_framework: str = "auto") -> str:
    """
    Create a test file with the specified content.
    
    Args:
        file_path: Path to the test file to create
        test_content: Content of the test file
        test_framework: Testing framework to use (auto, pytest, unittest, jest, mocha, junit)
    
    Returns:
        Success message with details about the created test file
    """
    console = Console()
    
    # Auto-detect testing framework if needed
    if test_framework == "auto":
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".py":
            # Check if pytest is available
            try:
                import pytest
                test_framework = "pytest"
            except ImportError:
                test_framework = "unittest"
        elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
            # Default to jest for JavaScript/TypeScript
            test_framework = "jest"
        elif file_ext == ".java":
            test_framework = "junit"
        else:
            return f"Error: Could not auto-detect testing framework for file extension '{file_ext}'. Please specify one explicitly."
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            return f"Error creating directory '{dir_path}': {str(e)}"
    
    # Add framework-specific imports/setup if needed
    if test_framework == "pytest" and not test_content.strip().startswith("import pytest"):
        test_content = "import pytest\n\n" + test_content
    elif test_framework == "unittest" and not "import unittest" in test_content:
        test_content = "import unittest\n\n" + test_content
    elif test_framework == "jest" and not test_content.strip().startswith(("import", "const", "let", "var")):
        test_content = "// Jest test file\n\n" + test_content
    elif test_framework == "mocha" and not test_content.strip().startswith(("import", "const", "let", "var")):
        test_content = "// Mocha test file\n\nconst assert = require('assert');\n\n" + test_content
    elif test_framework == "junit" and not "import" in test_content and not "package" in test_content:
        test_content = "// JUnit test file\n\nimport org.junit.Test;\nimport static org.junit.Assert.*;\n\n" + test_content
    
    # Write the test file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
    except Exception as e:
        return f"Error writing test file '{file_path}': {str(e)}"
    
    return f"Success: Test file '{file_path}' created for {test_framework} framework."


@tool
def generate_unit_test(file_path: str, function_name: str, test_framework: str = "auto") -> str:
    """
    Generate a basic unit test for a specific function.
    
    Args:
        file_path: Path to the file containing the function to test
        function_name: Name of the function to generate a test for
        test_framework: Testing framework to use (auto, pytest, unittest, jest, mocha, junit)
    
    Returns:
        Generated unit test code
    """
    console = Console()
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    # Read the file to extract function information
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Auto-detect testing framework if needed
    if test_framework == "auto":
        if file_ext == ".py":
            # Check if pytest is available
            try:
                import pytest
                test_framework = "pytest"
            except ImportError:
                test_framework = "unittest"
        elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
            # Default to jest for JavaScript/TypeScript
            test_framework = "jest"
        elif file_ext == ".java":
            test_framework = "junit"
        else:
            return f"Error: Could not auto-detect testing framework for file extension '{file_ext}'. Please specify one explicitly."
    
    # Extract function signature (simplified)
    function_signature = _extract_function_signature(content, function_name, file_ext)
    if not function_signature:
        return f"Error: Could not find function '{function_name}' in file '{file_path}'."
    
    # Generate test based on framework
    if test_framework == "pytest":
        return _generate_pytest_test(function_name, function_signature)
    elif test_framework == "unittest":
        return _generate_unittest_test(function_name, function_signature)
    elif test_framework == "jest":
        return _generate_jest_test(function_name, function_signature)
    elif test_framework == "mocha":
        return _generate_mocha_test(function_name, function_signature)
    elif test_framework == "junit":
        return _generate_junit_test(function_name, function_signature)
    else:
        return f"Error: Unsupported testing framework '{test_framework}'."


def _detect_test_framework(directory: str) -> Optional[str]:
    """Detect the testing framework used in a directory."""
    # Check for pytest
    if _has_files_with_pattern(directory, ["test_*.py", "*_test.py", "conftest.py"]):
        return "pytest"
    
    # Check for unittest
    if _has_files_with_pattern(directory, ["test_*.py", "*_test.py"]):
        return "unittest"
    
    # Check for Jest
    if _has_files_with_pattern(directory, ["*.test.js", "*.test.jsx", "*.test.ts", "*.test.tsx"]):
        return "jest"
    
    # Check for Mocha
    if _has_files_with_pattern(directory, ["*.spec.js", "*.spec.jsx", "*.spec.ts", "*.spec.tsx"]):
        return "mocha"
    
    # Check for JUnit
    if _has_files_with_pattern(directory, ["*Test.java"]):
        return "junit"
    
    return None


def _has_files_with_pattern(directory: str, patterns: List[str]) -> bool:
    """Check if directory has files matching any of the patterns."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            for pattern in patterns:
                if _matches_file_pattern(file, pattern):
                    return True
    return False


def _matches_file_pattern(filename: str, pattern: str) -> bool:
    """Check if a filename matches a pattern (supports wildcards)."""
    import re
    # Convert pattern to regex
    regex_pattern = pattern.replace(".", r"\.").replace("*", r".*").replace("?", r".")
    return re.match(regex_pattern, filename) is not None


def _run_pytest(test_directory: str) -> str:
    """Run pytest tests."""
    try:
        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", test_directory],
            capture_output=True,
            text=True
        )
        
        output = f"## Pytest Test Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error running pytest: {str(e)}"


def _run_unittest(test_directory: str) -> str:
    """Run unittest tests."""
    try:
        # Run unittest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-v", test_directory],
            capture_output=True,
            text=True
        )
        
        output = f"## Unittest Test Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error running unittest: {str(e)}"


def _run_jest(test_directory: str) -> str:
    """Run Jest tests."""
    try:
        # Run Jest with verbose output
        result = subprocess.run(
            ["npx", "jest", "--verbose", test_directory],
            capture_output=True,
            text=True
        )
        
        output = f"## Jest Test Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error running Jest: {str(e)}"


def _run_mocha(test_directory: str) -> str:
    """Run Mocha tests."""
    try:
        # Run Mocha with verbose output
        result = subprocess.run(
            ["npx", "mocha", "--recursive", test_directory],
            capture_output=True,
            text=True
        )
        
        output = f"## Mocha Test Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error running Mocha: {str(e)}"


def _run_junit(test_directory: str) -> str:
    """Run JUnit tests."""
    try:
        # Run JUnit tests
        result = subprocess.run(
            ["./gradlew", "test", "--info"],
            cwd=test_directory,
            capture_output=True,
            text=True
        )
        
        output = f"## JUnit Test Results\n\n"
        output += f"Exit Code: {result.returncode}\n\n"
        
        if result.stdout:
            output += "### Standard Output:\n\n"
            output += f"```\n{result.stdout}\n```\n\n"
        
        if result.stderr:
            output += "### Standard Error:\n\n"
            output += f"```\n{result.stderr}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error running JUnit: {str(e)}"


def _extract_function_signature(content: str, function_name: str, file_ext: str) -> Optional[str]:
    """Extract function signature from code content."""
    import re
    
    if file_ext == ".py":
        pattern = rf"def\s+{function_name}\s*\([^)]*\):"
    elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
        pattern = rf"(?:function\s+{function_name}\s*\([^)]*\)|{function_name}\s*:\s*function\s*\([^)]*\)|const\s+{function_name}\s*=\s*(?:function|\([^)]*\)\s*=>))"
    elif file_ext == ".java":
        pattern = rf"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+{function_name}\s*\([^)]*\)"
    elif file_ext in [".cpp", ".c", ".cc", ".cxx"]:
        pattern = rf"(?:\w+\s+)?{function_name}\s*\([^)]*\)\s*\{{?"
    else:
        return None
    
    match = re.search(pattern, content)
    if match:
        return match.group(0)
    
    return None


def _generate_pytest_test(function_name: str, function_signature: str) -> str:
    """Generate a pytest test for a function."""
    return f"""

def test_{function_name}():
    # TODO: Implement test for {{function_name}}
    # Example:
    # result = {{function_name}}(input_value)
    # assert result == expected_value
    pass
"""


def _generate_unittest_test(function_name: str, function_signature: str) -> str:
    """Generate a unittest test for a function."""
    return f"""

class Test{function_name.title()}(unittest.TestCase):
    def test_{function_name}(self):
        # TODO: Implement test for {{function_name}}
        # Example:
        # result = {{function_name}}(input_value)
        # self.assertEqual(result, expected_value)
        pass
"""



def _generate_jest_test(function_name: str, function_signature: str) -> str:
    """Generate a Jest test for a function."""
    return f"""
test('{function_name}', () => {{
    // TODO: Implement test for {function_name}
    // Example:
    // const result = {function_name}(inputValue);
    // expect(result).toBe(expectedValue);
}});
"""


def _generate_mocha_test(function_name: str, function_signature: str) -> str:
    """Generate a Mocha test for a function."""
    return f"""
describe('{function_name}', () => {{
    it('should work correctly', () => {{
        // TODO: Implement test for {function_name}
        // Example:
        // const result = {function_name}(inputValue);
        // assert.equal(result, expectedValue);
    }});
}});
"""


def _generate_junit_test(function_name: str, function_signature: str) -> str:
    """Generate a JUnit test for a function."""
    class_name = function_name.title()
    return f"""

@Test
public void test{class_name}() {{
    // TODO: Implement test for {{function_name}}
    // Example:
    // ResultType result = {{function_name}}(inputValue);
    // assertEquals(expectedValue, result);
}}
"""