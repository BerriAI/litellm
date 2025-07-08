#!/usr/bin/env python3
"""
Run LLM Translation Tests and Generate Beautiful Markdown Report

This script runs the LLM translation tests and generates a comprehensive
markdown report with provider-specific breakdowns and test statistics.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message: str, color: str = Colors.RESET):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.RESET}")

def get_provider_from_test_file(test_file: str) -> str:
    """Map test file names to provider names"""
    provider_mapping = {
        'test_anthropic': 'Anthropic',
        'test_azure': 'Azure',
        'test_bedrock': 'AWS Bedrock',
        'test_openai': 'OpenAI',
        'test_vertex': 'Google Vertex AI',
        'test_gemini': 'Google Vertex AI',
        'test_cohere': 'Cohere',
        'test_databricks': 'Databricks',
        'test_groq': 'Groq',
        'test_together': 'Together AI',
        'test_mistral': 'Mistral',
        'test_deepseek': 'DeepSeek',
        'test_replicate': 'Replicate',
        'test_huggingface': 'HuggingFace',
        'test_fireworks': 'Fireworks AI',
        'test_perplexity': 'Perplexity',
        'test_cloudflare': 'Cloudflare',
        'test_voyage': 'Voyage AI',
        'test_xai': 'xAI',
        'test_nvidia': 'NVIDIA',
        'test_watsonx': 'IBM watsonx',
        'test_azure_ai': 'Azure AI',
        'test_snowflake': 'Snowflake',
        'test_infinity': 'Infinity',
        'test_jina': 'Jina AI',
        'test_deepgram': 'Deepgram',
        'test_clarifai': 'Clarifai',
        'test_triton': 'Triton',
    }
    
    for key, provider in provider_mapping.items():
        if key in test_file:
            return provider
    
    # For cross-provider test files
    if any(name in test_file for name in ['test_optional_params', 'test_prompt_factory', 
                                           'test_router', 'test_text_completion']):
        return f'Cross-Provider Tests ({test_file})'
    
    return 'Other Tests'

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def generate_markdown_report(junit_xml_path: str, output_path: str, tag: str = None, commit: str = None):
    """Generate a beautiful markdown report from JUnit XML"""
    try:
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()
        
        # Handle both testsuite and testsuites root
        if root.tag == 'testsuites':
            suites = root.findall('testsuite')
        else:
            suites = [root]
        
        # Overall statistics
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0.0
        
        # Provider breakdown
        provider_stats = defaultdict(lambda: {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0, 'time': 0.0})
        provider_tests = defaultdict(list)
        
        for suite in suites:
            total_tests += int(suite.get('tests', 0))
            total_failures += int(suite.get('failures', 0))
            total_errors += int(suite.get('errors', 0))
            total_skipped += int(suite.get('skipped', 0))
            total_time += float(suite.get('time', 0))
            
            for testcase in suite.findall('testcase'):
                classname = testcase.get('classname', '')
                test_name = testcase.get('name', '')
                test_time = float(testcase.get('time', 0))
                
                # Extract test file name from classname
                if '.' in classname:
                    parts = classname.split('.')
                    test_file = parts[-2] if len(parts) > 1 else 'unknown'
                else:
                    test_file = 'unknown'
                
                provider = get_provider_from_test_file(test_file)
                provider_stats[provider]['time'] += test_time
                
                # Check test status
                if testcase.find('failure') is not None:
                    provider_stats[provider]['failed'] += 1
                    failure = testcase.find('failure')
                    failure_msg = failure.get('message', '') if failure is not None else ''
                    provider_tests[provider].append({
                        'name': test_name,
                        'status': 'FAILED',
                        'time': test_time,
                        'message': failure_msg
                    })
                elif testcase.find('error') is not None:
                    provider_stats[provider]['errors'] += 1
                    error = testcase.find('error')
                    error_msg = error.get('message', '') if error is not None else ''
                    provider_tests[provider].append({
                        'name': test_name,
                        'status': 'ERROR',
                        'time': test_time,
                        'message': error_msg
                    })
                elif testcase.find('skipped') is not None:
                    provider_stats[provider]['skipped'] += 1
                    skip = testcase.find('skipped')
                    skip_msg = skip.get('message', '') if skip is not None else ''
                    provider_tests[provider].append({
                        'name': test_name,
                        'status': 'SKIPPED',
                        'time': test_time,
                        'message': skip_msg
                    })
                else:
                    provider_stats[provider]['passed'] += 1
                    provider_tests[provider].append({
                        'name': test_name,
                        'status': 'PASSED',
                        'time': test_time,
                        'message': ''
                    })
        
        passed = total_tests - total_failures - total_errors - total_skipped
        
        # Generate the markdown report
        with open(output_path, 'w') as f:
            # Header
            f.write("# LLM Translation Test Results\n\n")
            
            # Metadata table
            f.write("## Test Run Information\n\n")
            f.write("| Field | Value |\n")
            f.write("|-------|-------|\n")
            f.write(f"| **Tag** | `{tag or 'N/A'}` |\n")
            f.write(f"| **Date** | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} |\n")
            f.write(f"| **Commit** | `{commit or 'N/A'}` |\n")
            f.write(f"| **Duration** | {format_duration(total_time)} |\n")
            f.write("\n")
            
            # Overall statistics with visual elements
            f.write("## Overall Statistics\n\n")
            
            # Summary box
            f.write("```\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"├── Passed:  {passed:>4} ({(passed/total_tests)*100 if total_tests > 0 else 0:.1f}%)\n")
            f.write(f"├── Failed:  {total_failures:>4} ({(total_failures/total_tests)*100 if total_tests > 0 else 0:.1f}%)\n")
            f.write(f"├── Errors:  {total_errors:>4} ({(total_errors/total_tests)*100 if total_tests > 0 else 0:.1f}%)\n")
            f.write(f"└── Skipped: {total_skipped:>4} ({(total_skipped/total_tests)*100 if total_tests > 0 else 0:.1f}%)\n")
            f.write("```\n\n")
            
            
            # Provider summary table
            f.write("## Results by Provider\n\n")
            f.write("| Provider | Total | Pass | Fail | Error | Skip | Pass Rate | Duration |\n")
            f.write("|----------|-------|------|------|-------|------|-----------|----------|")
            
            # Sort providers: specific providers first, then cross-provider tests
            sorted_providers = []
            cross_provider = []
            for p in sorted(provider_stats.keys()):
                if 'Cross-Provider' in p or p == 'Other Tests':
                    cross_provider.append(p)
                else:
                    sorted_providers.append(p)
            
            all_providers = sorted_providers + cross_provider
            
            for provider in all_providers:
                stats = provider_stats[provider]
                total = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
                pass_rate = (stats['passed'] / total * 100) if total > 0 else 0
                
                f.write(f"\n| {provider} | {total} | {stats['passed']} | {stats['failed']} | ")
                f.write(f"{stats['errors']} | {stats['skipped']} | {pass_rate:.1f}% | ")
                f.write(f"{format_duration(stats['time'])} |")
            
            # Detailed test results by provider
            f.write("\n\n## Detailed Test Results\n\n")
            
            for provider in sorted_providers:
                if provider_tests[provider]:
                    stats = provider_stats[provider]
                    total = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
                    
                    f.write(f"### {provider}\n\n")
                    f.write(f"**Summary:** {stats['passed']}/{total} passed ")
                    f.write(f"({(stats['passed']/total)*100 if total > 0 else 0:.1f}%) ")
                    f.write(f"in {format_duration(stats['time'])}\n\n")
                    
                    # Group tests by status
                    tests_by_status = defaultdict(list)
                    for test in provider_tests[provider]:
                        tests_by_status[test['status']].append(test)
                    
                    # Show failed tests first (if any)
                    if tests_by_status['FAILED']:
                        f.write("<details>\n<summary>Failed Tests</summary>\n\n")
                        for test in tests_by_status['FAILED']:
                            f.write(f"- `{test['name']}` ({test['time']:.2f}s)\n")
                            if test['message']:
                                # Truncate long error messages
                                msg = test['message'][:200] + '...' if len(test['message']) > 200 else test['message']
                                f.write(f"  > {msg}\n")
                        f.write("\n</details>\n\n")
                    
                    # Show errors (if any)
                    if tests_by_status['ERROR']:
                        f.write("<details>\n<summary>Error Tests</summary>\n\n")
                        for test in tests_by_status['ERROR']:
                            f.write(f"- `{test['name']}` ({test['time']:.2f}s)\n")
                        f.write("\n</details>\n\n")
                    
                    # Show passed tests in collapsible section
                    if tests_by_status['PASSED']:
                        f.write("<details>\n<summary>Passed Tests</summary>\n\n")
                        for test in tests_by_status['PASSED']:
                            f.write(f"- `{test['name']}` ({test['time']:.2f}s)\n")
                        f.write("\n</details>\n\n")
                    
                    # Show skipped tests (if any)
                    if tests_by_status['SKIPPED']:
                        f.write("<details>\n<summary>Skipped Tests</summary>\n\n")
                        for test in tests_by_status['SKIPPED']:
                            f.write(f"- `{test['name']}`\n")
                        f.write("\n</details>\n\n")
            
            # Cross-provider tests in a separate section
            if cross_provider:
                f.write("### Cross-Provider Tests\n\n")
                for provider in cross_provider:
                    if provider_tests[provider]:
                        stats = provider_stats[provider]
                        total = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
                        
                        f.write(f"#### {provider}\n\n")
                        f.write(f"**Summary:** {stats['passed']}/{total} passed ")
                        f.write(f"({(stats['passed']/total)*100 if total > 0 else 0:.1f}%)\n\n")
                        
                        # For cross-provider tests, just show counts
                        f.write(f"- Passed: {stats['passed']}\n")
                        if stats['failed'] > 0:
                            f.write(f"- Failed: {stats['failed']}\n")
                        if stats['errors'] > 0:
                            f.write(f"- Errors: {stats['errors']}\n")
                        if stats['skipped'] > 0:
                            f.write(f"- Skipped: {stats['skipped']}\n")
                        f.write("\n")
            
        
        print_colored(f"Report generated: {output_path}", Colors.GREEN)
        
    except Exception as e:
        print_colored(f"Error generating report: {e}", Colors.RED)
        raise

def run_tests(test_path: str = "tests/llm_translation/", 
              junit_xml: str = "test-results/junit.xml",
              report_path: str = "test-results/llm_translation_report.md",
              tag: str = None,
              commit: str = None) -> int:
    """Run the LLM translation tests and generate report"""
    
    # Create test results directory
    os.makedirs(os.path.dirname(junit_xml), exist_ok=True)
    
    print_colored("Starting LLM Translation Tests", Colors.BOLD + Colors.BLUE)
    print_colored(f"Test directory: {test_path}", Colors.CYAN)
    print_colored(f"Output: {junit_xml}", Colors.CYAN)
    print()
    
    # Run pytest
    cmd = [
        "poetry", "run", "pytest", test_path,
        f"--junitxml={junit_xml}",
        "-v",
        "--tb=short",
        "--maxfail=500",
        "-n", "auto"
    ]
    
    # Add timeout if pytest-timeout is installed
    try:
        subprocess.run(["poetry", "run", "python", "-c", "import pytest_timeout"], 
                      capture_output=True, check=True)
        cmd.extend(["--timeout=300"])
    except:
        print_colored("Warning: pytest-timeout not installed, skipping timeout option", Colors.YELLOW)
    
    print_colored("Running pytest with command:", Colors.YELLOW)
    print(f"   {' '.join(cmd)}")
    print()
    
    # Run the tests
    result = subprocess.run(cmd, capture_output=False)
    
    # Generate the report regardless of test outcome
    if os.path.exists(junit_xml):
        print()
        print_colored("Generating test report...", Colors.BLUE)
        generate_markdown_report(junit_xml, report_path, tag, commit)
        
        # Print summary to console
        print()
        print_colored("Test Summary:", Colors.BOLD + Colors.PURPLE)
        
        # Parse XML for quick summary
        tree = ET.parse(junit_xml)
        root = tree.getroot()
        
        if root.tag == 'testsuites':
            suites = root.findall('testsuite')
        else:
            suites = [root]
        
        total = sum(int(s.get('tests', 0)) for s in suites)
        failures = sum(int(s.get('failures', 0)) for s in suites)
        errors = sum(int(s.get('errors', 0)) for s in suites)
        skipped = sum(int(s.get('skipped', 0)) for s in suites)
        passed = total - failures - errors - skipped
        
        print(f"   Total:   {total}")
        print_colored(f"   Passed:  {passed}", Colors.GREEN)
        if failures > 0:
            print_colored(f"   Failed:  {failures}", Colors.RED)
        if errors > 0:
            print_colored(f"   Errors:  {errors}", Colors.RED)
        if skipped > 0:
            print_colored(f"   Skipped: {skipped}", Colors.YELLOW)
        
        if total > 0:
            pass_rate = (passed / total) * 100
            color = Colors.GREEN if pass_rate >= 80 else Colors.YELLOW if pass_rate >= 60 else Colors.RED
            print_colored(f"   Pass Rate: {pass_rate:.1f}%", color)
    else:
        print_colored("No test results found!", Colors.RED)
    
    print()
    print_colored("Test run complete!", Colors.BOLD + Colors.GREEN)
    
    return result.returncode

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM Translation Tests")
    parser.add_argument("--test-path", default="tests/llm_translation/", 
                        help="Path to test directory")
    parser.add_argument("--junit-xml", default="test-results/junit.xml",
                        help="Path for JUnit XML output")
    parser.add_argument("--report", default="test-results/llm_translation_report.md",
                        help="Path for markdown report")
    parser.add_argument("--tag", help="Git tag or version")
    parser.add_argument("--commit", help="Git commit SHA")
    
    args = parser.parse_args()
    
    # Get git info if not provided
    if not args.commit:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                args.commit = result.stdout.strip()
        except:
            pass
    
    if not args.tag:
        try:
            result = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                args.tag = result.stdout.strip()
        except:
            pass
    
    exit_code = run_tests(
        test_path=args.test_path,
        junit_xml=args.junit_xml,
        report_path=args.report,
        tag=args.tag,
        commit=args.commit
    )
    
    sys.exit(exit_code)