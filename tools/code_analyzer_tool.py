"""
Code Analyzer Tool

This tool analyzes code files for quality issues, bugs, performance problems,
and provides suggestions for improvement.
"""

import os
import ast
from typing import Dict, List, Optional, Any
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def analyze_code(file_path: str, analysis_type: str = "general") -> str:
    """
    Analyze a code file for quality issues, bugs, performance problems, and best practices.
    
    Args:
        file_path: Path to the code file to analyze
        analysis_type: Type of analysis to perform (general, security, performance, style)
    
    Returns:
        Analysis results with suggestions for improvement
    """
    console = Console()
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."
    
    # Check if file is readable
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Perform analysis based on file type
    if file_ext == '.py':
        return _analyze_python_code(code_content, file_path, analysis_type)
    elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
        return _analyze_javascript_code(code_content, file_path, analysis_type)
    elif file_ext in ['.java']:
        return _analyze_java_code(code_content, file_path, analysis_type)
    elif file_ext in ['.cpp', '.c', '.cc', '.cxx']:
        return _analyze_cpp_code(code_content, file_path, analysis_type)
    else:
        return f"Unsupported file type: {file_ext}. Supported types: .py, .js, .jsx, .ts, .tsx, .java, .cpp, .c, .cc, .cxx"


def _analyze_python_code(code: str, file_path: str, analysis_type: str) -> str:
    """Analyze Python code for various issues."""
    console = Console()
    issues = []
    suggestions = []
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Check for various issues
        issues.extend(_check_python_syntax(tree))
        if analysis_type in ["general", "style"]:
            issues.extend(_check_python_style(tree, code))
        if analysis_type in ["general", "security"]:
            issues.extend(_check_python_security(tree))
        if analysis_type in ["general", "performance"]:
            issues.extend(_check_python_performance(tree))
        
        # Generate suggestions based on issues
        suggestions = _generate_python_suggestions(issues)
        
    except SyntaxError as e:
        return f"Syntax Error in {file_path}: {str(e)}"
    
    # Format results
    result = f"## Code Analysis Results for {file_path}\n\n"
    
    if issues:
        result += "### Issues Found:\n\n"
        for i, issue in enumerate(issues, 1):
            result += f"{i}. **{issue['type']}** (Line {issue.get('line', 'N/A')}): {issue['message']}\n"
    else:
        result += "### No issues found! The code looks good.\n"
    
    if suggestions:
        result += "\n### Suggestions for Improvement:\n\n"
        for i, suggestion in enumerate(suggestions, 1):
            result += f"{i}. {suggestion}\n"
    
    return result


def _check_python_syntax(tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for syntax-related issues in Python code."""
    issues = []
    
    # This is a placeholder for more sophisticated syntax checks
    # The AST parsing itself would catch syntax errors
    
    return issues


def _check_python_style(tree: ast.AST, code: str) -> List[Dict[str, Any]]:
    """Check for style issues in Python code."""
    issues = []
    lines = code.split('\n')
    
    # Check for line length
    for i, line in enumerate(lines, 1):
        if len(line) > 88:  # Black default line length
            issues.append({
                'type': 'Style',
                'line': i,
                'message': f'Line too long ({len(line)} > 88 characters)'
            })
    
    # Check for TODO/FIXME comments
    for i, line in enumerate(lines, 1):
        if 'TODO' in line or 'FIXME' in line:
            issues.append({
                'type': 'Style',
                'line': i,
                'message': 'Contains TODO/FIXME comment'
            })
    
    return issues


def _check_python_security(tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for security issues in Python code."""
    issues = []
    
    class SecurityVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Check for dangerous functions
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile']:
                    issues.append({
                        'type': 'Security',
                        'line': node.lineno,
                        'message': f'Use of dangerous function: {node.func.id}'
                    })
            
            # Check for shell commands
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ['system', 'popen', 'call', 'check_output']:
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                        issues.append({
                            'type': 'Security',
                            'line': node.lineno,
                            'message': f'Use of potentially dangerous os function: {node.func.attr}'
                        })
            
            self.generic_visit(node)
    
    SecurityVisitor().visit(tree)
    return issues


def _check_python_performance(tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for performance issues in Python code."""
    issues = []
    
    class PerformanceVisitor(ast.NodeVisitor):
        def visit_For(self, node):
            # Check for inefficient loops
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range' and len(node.iter.args) > 0:
                    # Check if range is used with len()
                    if (len(node.iter.args) == 1 and 
                        isinstance(node.iter.args[0], ast.Call) and
                        isinstance(node.iter.args[0].func, ast.Name) and
                        node.iter.args[0].func.id == 'len'):
                        issues.append({
                            'type': 'Performance',
                            'line': node.lineno,
                            'message': 'Consider using enumerate() instead of range(len())'
                        })
            
            self.generic_visit(node)
    
    PerformanceVisitor().visit(tree)
    return issues


def _generate_python_suggestions(issues: List[Dict[str, Any]]) -> List[str]:
    """Generate suggestions based on found issues."""
    suggestions = []
    
    # Group issues by type
    issue_types = {}
    for issue in issues:
        issue_type = issue['type']
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    # Generate suggestions for each issue type
    if 'Style' in issue_types:
        suggestions.append("Consider using a code formatter like Black to maintain consistent style.")
        suggestions.append("Set up a linter like Flake8 or Pylint to catch style issues automatically.")
    
    if 'Security' in issue_types:
        suggestions.append("Review the use of dangerous functions and consider safer alternatives.")
        suggestions.append("Implement proper input validation and sanitization.")
    
    if 'Performance' in issue_types:
        suggestions.append("Consider using more Pythonic constructs for better performance.")
        suggestions.append("Profile your code to identify and optimize bottlenecks.")
    
    return suggestions


def _analyze_javascript_code(code: str, file_path: str, analysis_type: str) -> str:
    """Analyze JavaScript/TypeScript code for various issues."""
    # This is a placeholder implementation
    # In a real implementation, you might use ESLint or other tools
    
    result = f"## Code Analysis Results for {file_path}\n\n"
    result += "### JavaScript/TypeScript Analysis\n\n"
    result += "Note: JavaScript/TypeScript analysis is not fully implemented in this demo.\n"
    result += "Consider using tools like ESLint, JSHint, or TypeScript compiler for comprehensive analysis.\n\n"
    
    # Basic checks
    lines = code.split('\n')
    issues = []
    
    # Check for console.log statements
    for i, line in enumerate(lines, 1):
        if 'console.log' in line:
            issues.append({
                'type': 'Style',
                'line': i,
                'message': 'Contains console.log statement (should be removed in production)'
            })
    
    if issues:
        result += "### Issues Found:\n\n"
        for i, issue in enumerate(issues, 1):
            result += f"{i}. **{issue['type']}** (Line {issue.get('line', 'N/A')}): {issue['message']}\n"
    else:
        result += "### No obvious issues found in basic checks.\n"
    
    return result


def _analyze_java_code(code: str, file_path: str, analysis_type: str) -> str:
    """Analyze Java code for various issues."""
    # This is a placeholder implementation
    # In a real implementation, you might use tools like Checkstyle, PMD, or SpotBugs
    
    result = f"## Code Analysis Results for {file_path}\n\n"
    result += "### Java Analysis\n\n"
    result += "Note: Java analysis is not fully implemented in this demo.\n"
    result += "Consider using tools like Checkstyle, PMD, or SpotBugs for comprehensive analysis.\n\n"
    
    return result


def _analyze_cpp_code(code: str, file_path: str, analysis_type: str) -> str:
    """Analyze C/C++ code for various issues."""
    # This is a placeholder implementation
    # In a real implementation, you might use tools like Clang-Tidy, Cppcheck, or CppLint
    
    result = f"## Code Analysis Results for {file_path}\n\n"
    result += "### C/C++ Analysis\n\n"
    result += "Note: C/C++ analysis is not fully implemented in this demo.\n"
    result += "Consider using tools like Clang-Tidy, Cppcheck, or CppLint for comprehensive analysis.\n\n"
    
    return result