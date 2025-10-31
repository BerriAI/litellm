"""
Command-line entry point for memory leak analysis.

This allows running the analyzer as:
    python -m tests.memory_leak_tests.profiler <profile_file.json>

Instead of:
    python -m tests.memory_leak_tests.profiler.analyze_profiles <profile_file.json>
"""

import sys
from .analyze_profiles import analyze_profile_file


def main():
    """Main entry point for command-line analysis."""
    if len(sys.argv) < 2:
        print("Usage: python -m tests.memory_leak_tests.profiler <profile_file.json> [--growth N]")
        print("\nOptions:")
        print("  --growth N    Show top N locations by memory growth (default: 20)")
        print("\nExamples:")
        print("  # Show default growth analysis (top 20)")
        print("  python -m tests.memory_leak_tests.profiler endpoint_profiles/chat_completions.json")
        print("\n  # Show top 30 growing locations")
        print("  python -m tests.memory_leak_tests.profiler endpoint_profiles/chat_completions.json --growth 30")
        print("\nNote: Run from the project root directory")
        sys.exit(1)
    
    profile_file = sys.argv[1]
    growth_n = 20  # Default
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--growth' and i + 1 < len(sys.argv):
            growth_n = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    analyze_profile_file(profile_file, show_growth=growth_n)


if __name__ == '__main__':
    main()

