"""
Log Analysis Tools

Tools for analyzing log files, searching for patterns, and identifying root causes.
"""

import os
import re
import glob
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown


@tool
def search_log_files(
    search_pattern: str,
    log_directory: str = ".",
    file_pattern: str = "*.log",
    case_sensitive: bool = False,
    max_results: int = 100,
    time_range: Optional[str] = None
) -> str:
    """
    Search for patterns in log files within a directory.
    
    Args:
        search_pattern: Pattern to search for (regex supported)
        log_directory: Directory to search in (default: current directory)
        file_pattern: File pattern to match (e.g., "*.log", "*.txt", "*.evt")
        case_sensitive: Whether the search should be case sensitive
        max_results: Maximum number of results to return
        time_range: Optional time range filter (e.g., "last 24 hours", "2024-01-01 to 2024-01-02")
    
    Returns:
        Search results with file paths, line numbers, and matched content
    """
    console = Console()
    
    # Expand environment variables in path
    log_directory = os.path.expandvars(log_directory)
    
    # Validate search directory
    if not os.path.exists(log_directory):
        return f"Error: Directory '{log_directory}' does not exist."
    
    # Compile regex pattern
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex_pattern = re.compile(search_pattern, flags)
    except re.error as e:
        return f"Error in regex pattern: {str(e)}"
    
    # Find matching log files
    log_files = []
    if os.path.isdir(log_directory):
        # Search recursively
        for root, dirs, files in os.walk(log_directory):
            for file in files:
                if _matches_file_pattern(file, file_pattern):
                    log_files.append(os.path.join(root, file))
    else:
        # Single file
        if _matches_file_pattern(log_directory, file_pattern):
            log_files.append(log_directory)
    
    if not log_files:
        return f"No log files found matching pattern '{file_pattern}' in '{log_directory}'."
    
    # Search within files
    results = []
    total_matches = 0
    
    for file_path in log_files:
        try:
            # Check file size (skip very large files)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                continue
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            file_matches = []
            for line_num, line in enumerate(lines, 1):
                if total_matches >= max_results:
                    break
                
                match = regex_pattern.search(line)
                if match:
                    file_matches.append({
                        'line': line_num,
                        'content': line.strip(),
                        'match': match.group()
                    })
                    total_matches += 1
            
            if file_matches:
                results.append({
                    'file': file_path,
                    'matches': file_matches
                })
        
        except Exception as e:
            # Skip files that can't be read
            continue
    
    # Format results
    if not results:
        return f"No matches found for pattern '{search_pattern}' in log files."
    
    output = f"## Log Search Results\n\n"
    output += f"**Pattern:** `{search_pattern}`\n"
    output += f"**Directory:** `{log_directory}`\n"
    output += f"**Total Matches:** {total_matches}\n\n"
    
    for result in results:
        output += f"### {result['file']}\n\n"
        for match in result['matches'][:20]:  # Limit to 20 matches per file
            output += f"**Line {match['line']}:** `{match['content'][:200]}`\n\n"
        
        if len(result['matches']) > 20:
            output += f"*... and {len(result['matches']) - 20} more matches*\n\n"
    
    return output


@tool
def analyze_log_errors(
    log_file_path: str,
    error_keywords: Optional[List[str]] = None,
    max_lines: int = 10000
) -> str:
    """
    Analyze a log file for errors and extract error patterns.
    
    Args:
        log_file_path: Path to the log file to analyze
        error_keywords: Optional list of error keywords to search for (default: common error patterns)
        max_lines: Maximum number of lines to analyze
    
    Returns:
        Analysis of errors found in the log file
    """
    console = Console()
    
    # Expand environment variables
    log_file_path = os.path.expandvars(log_file_path)
    
    # Check if file exists
    if not os.path.exists(log_file_path):
        return f"Error: Log file '{log_file_path}' does not exist."
    
    # Default error keywords
    if error_keywords is None:
        error_keywords = [
            'error', 'exception', 'failed', 'failure', 'critical',
            'fatal', 'warning', 'timeout', 'denied', 'access denied',
            'permission', 'not found', 'cannot', 'unable', 'invalid'
        ]
    
    # Read log file
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
    except Exception as e:
        return f"Error reading log file: {str(e)}"
    
    # Analyze errors
    error_patterns = {}
    error_lines = []
    
    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()
        for keyword in error_keywords:
            if keyword.lower() in line_lower:
                if keyword not in error_patterns:
                    error_patterns[keyword] = []
                error_patterns[keyword].append({
                    'line': line_num,
                    'content': line.strip()[:200]
                })
                error_lines.append({
                    'line': line_num,
                    'keyword': keyword,
                    'content': line.strip()[:200]
                })
                break
    
    # Format output
    output = f"## Log Error Analysis\n\n"
    output += f"**File:** `{log_file_path}`\n"
    output += f"**Total Lines Analyzed:** {len(lines)}\n"
    output += f"**Total Errors Found:** {len(error_lines)}\n\n"
    
    if error_lines:
        output += "### Error Summary by Keyword\n\n"
        for keyword, matches in error_patterns.items():
            output += f"- **{keyword}**: {len(matches)} occurrences\n"
        
        output += "\n### Sample Error Lines\n\n"
        for error in error_lines[:50]:  # Show first 50 errors
            output += f"**Line {error['line']}** ({error['keyword']}):\n"
            output += f"```\n{error['content']}\n```\n\n"
        
        if len(error_lines) > 50:
            output += f"*... and {len(error_lines) - 50} more errors*\n\n"
    else:
        output += "No errors found matching the specified keywords.\n"
    
    return output


@tool
def summarize_log_root_cause(
    log_file_path: str,
    issue_description: str,
    max_lines: int = 10000
) -> str:
    """
    Analyze a log file to identify the root cause of an issue.
    
    Args:
        log_file_path: Path to the log file to analyze
        issue_description: Description of the issue to investigate
        max_lines: Maximum number of lines to analyze
    
    Returns:
        Summary of root cause analysis
    """
    console = Console()
    
    # Expand environment variables
    log_file_path = os.path.expandvars(log_file_path)
    
    # Check if file exists
    if not os.path.exists(log_file_path):
        return f"Error: Log file '{log_file_path}' does not exist."
    
    # Read log file
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
    except Exception as e:
        return f"Error reading log file: {str(e)}"
    
    # Extract relevant information
    error_lines = []
    warning_lines = []
    timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}:\d{2}:\d{2}')
    
    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ['error', 'exception', 'failed', 'fatal', 'critical']):
            error_lines.append({
                'line': line_num,
                'content': line.strip()
            })
        elif 'warning' in line_lower:
            warning_lines.append({
                'line': line_num,
                'content': line.strip()
            })
    
    # Analyze patterns
    output = f"## Root Cause Analysis\n\n"
    output += f"**Issue Description:** {issue_description}\n"
    output += f"**Log File:** `{log_file_path}`\n"
    output += f"**Total Lines Analyzed:** {len(lines)}\n"
    output += f"**Errors Found:** {len(error_lines)}\n"
    output += f"**Warnings Found:** {len(warning_lines)}\n\n"
    
    if error_lines:
        output += "### Error Timeline\n\n"
        # Show first and last errors
        if len(error_lines) > 0:
            output += f"**First Error (Line {error_lines[0]['line']}):**\n"
            output += f"```\n{error_lines[0]['content']}\n```\n\n"
        
        if len(error_lines) > 1:
            output += f"**Last Error (Line {error_lines[-1]['line']}):**\n"
            output += f"```\n{error_lines[-1]['content']}\n```\n\n"
        
        # Find most common error patterns
        error_patterns = {}
        for error in error_lines:
            # Extract error type (first word after "error" or "exception")
            pattern_match = re.search(r'(error|exception|failed|fatal):?\s*([A-Za-z]+)', error['content'], re.IGNORECASE)
            if pattern_match:
                pattern = pattern_match.group(2).lower()
                error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        if error_patterns:
            output += "### Most Common Error Patterns\n\n"
            sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:10]:
                output += f"- **{pattern}**: {count} occurrences\n"
            output += "\n"
    
    # Generate root cause hypothesis
    output += "### Root Cause Hypothesis\n\n"
    
    if error_lines:
        # Analyze the first error
        first_error = error_lines[0]['content']
        output += f"Based on the analysis, the root cause appears to be related to:\n\n"
        output += f"1. **Initial Error:** The first error occurred at line {error_lines[0]['line']}\n"
        output += f"2. **Error Pattern:** {first_error[:200]}\n"
        
        if len(error_lines) > 10:
            output += f"3. **Error Frequency:** Multiple errors ({len(error_lines)} total) suggest a systemic issue\n"
        
        # Look for common patterns
        if 'permission' in first_error.lower() or 'access denied' in first_error.lower():
            output += "4. **Likely Cause:** Permission or access control issue\n"
        elif 'timeout' in first_error.lower():
            output += "4. **Likely Cause:** Timeout or resource availability issue\n"
        elif 'not found' in first_error.lower() or 'missing' in first_error.lower():
            output += "4. **Likely Cause:** Missing resource or configuration issue\n"
        else:
            output += "4. **Likely Cause:** Application or system error (further investigation needed)\n"
    else:
        output += "No clear errors found in the log file. The issue may be:\n"
        output += "1. Related to warnings or informational messages\n"
        output += "2. Not logged in this file\n"
        output += "3. Requires analysis of related log files\n"
    
    output += "\n### Recommendations\n\n"
    output += "1. Review the error timeline above\n"
    output += "2. Check system configuration and permissions\n"
    output += "3. Verify related services and dependencies\n"
    output += "4. Review recent system changes\n"
    
    return output


@tool
def find_windows_event_logs(
    log_name: str = "Application",
    event_id: Optional[int] = None,
    level: Optional[str] = None,
    max_events: int = 100
) -> str:
    """
    Search Windows Event Logs for specific events.
    
    Args:
        log_name: Name of the event log (Application, System, Security, etc.)
        event_id: Optional event ID to filter by
        level: Optional level filter (Error, Warning, Information, Critical)
        max_events: Maximum number of events to return
    
    Returns:
        Event log entries matching the criteria
    """
    console = Console()
    
    try:
        import win32evtlog
        import win32evtlogutil
        import win32con
    except ImportError:
        return "Error: pywin32 is required for Windows Event Log access. Install it with: pip install pywin32"
    
    # Check if running on Windows
    if os.name != 'nt':
        return "Error: Windows Event Log access is only available on Windows systems."
    
    try:
        # Open event log
        hand = win32evtlog.OpenEventLog(None, log_name)
        
        if not hand:
            return f"Error: Could not open event log '{log_name}'"
        
        # Read events
        events = []
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        
        while True:
            events_batch = win32evtlog.ReadEventLog(hand, flags, 0)
            if not events_batch:
                break
            
            for event in events_batch:
                # Filter by event ID if specified
                if event_id is not None and event.EventID != event_id:
                    continue
                
                # Filter by level if specified
                if level is not None:
                    level_map = {
                        'error': win32con.EVENTLOG_ERROR_TYPE,
                        'warning': win32con.EVENTLOG_WARNING_TYPE,
                        'information': win32con.EVENTLOG_INFORMATION_TYPE,
                        'critical': win32con.EVENTLOG_ERROR_TYPE
                    }
                    if event.EventType != level_map.get(level.lower()):
                        continue
                
                events.append({
                    'time': event.TimeGenerated.Format(),
                    'event_id': event.EventID,
                    'level': win32evtlogutil.SafeFormatMessage(event, log_name),
                    'source': event.SourceName,
                    'message': win32evtlogutil.SafeFormatMessage(event, log_name)
                })
                
                if len(events) >= max_events:
                    break
            
            if len(events) >= max_events:
                break
        
        win32evtlog.CloseEventLog(hand)
        
        # Format output
        if not events:
            return f"No events found matching the criteria in '{log_name}' log."
        
        output = f"## Windows Event Log Results\n\n"
        output += f"**Log Name:** {log_name}\n"
        output += f"**Total Events Found:** {len(events)}\n\n"
        
        for event in events:
            output += f"### Event {event['event_id']}\n\n"
            output += f"**Time:** {event['time']}\n"
            output += f"**Source:** {event['source']}\n"
            output += f"**Level:** {event['level']}\n"
            output += f"**Message:**\n```\n{event['message'][:500]}\n```\n\n"
        
        return output
        
    except Exception as e:
        return f"Error accessing Windows Event Log: {str(e)}"


def _matches_file_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches the pattern."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)

