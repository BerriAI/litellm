"""
Memory leak detection algorithms.

Provides functions for:
- Detecting error-induced memory leaks
- Detecting general memory leaks based on growth patterns
"""

from typing import Dict, List, Optional, Tuple

from ..constants import (
    DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    DEFAULT_ERROR_SPIKE_STABILIZATION_TOLERANCE_PERCENT,
    DEFAULT_ERROR_SPIKE_MIN_STABLE_BATCHES,
    DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT,
    DEFAULT_LEAK_DETECTION_STABILIZATION_TOLERANCE_MB,
    DEFAULT_LEAK_DETECTION_TAIL_SAMPLES,
    NEAR_ZERO_MEMORY_THRESHOLD_MB,
)


def detect_error_induced_memory_leak(
    memory_samples: List[float],
    error_counts: List[int],
    error_spike_threshold_percent: float = DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT,
    stabilization_tolerance_percent: float = DEFAULT_ERROR_SPIKE_STABILIZATION_TOLERANCE_PERCENT,
    min_stable_batches: int = DEFAULT_ERROR_SPIKE_MIN_STABLE_BATCHES
) -> Tuple[bool, str, List[int]]:
    """
    Detect error-induced memory leaks where errors cause memory spikes that don't get released.
    
    This function identifies the pattern where:
    - A batch has 50%+ memory increase from the previous batch
    - That batch has errors
    - Memory stabilizes at the higher level after the error (doesn't continue growing)
    
    Args:
        memory_samples: List of memory measurements in MB for each batch
        error_counts: List of error counts for each batch
        error_spike_threshold_percent: Minimum percent increase to consider a spike (default: 50%)
        stabilization_tolerance_percent: Max percent variation to consider memory stable (default: 10%)
        min_stable_batches: Minimum batches after spike to check for stabilization (default: 2)
        
    Returns:
        Tuple of:
            - bool: Whether error-induced leak was detected
            - str: Detailed message about the findings
            - List[int]: Batch indices where error-induced spikes occurred (1-indexed)
    """
    if len(memory_samples) < 2 or len(error_counts) < 2:
        return False, "", []
    
    error_spike_batches = []
    non_stabilized_spikes = []
    
    for i in range(1, len(memory_samples)):
        prev_memory = memory_samples[i - 1]
        curr_memory = memory_samples[i]
        curr_errors = error_counts[i]
        
        # Skip if previous memory is too small to calculate meaningful percentage
        if prev_memory < NEAR_ZERO_MEMORY_THRESHOLD_MB:
            continue
        
        # Calculate percent increase
        percent_increase = ((curr_memory - prev_memory) / prev_memory) * 100
        
        # Check if this batch has a significant spike AND errors
        if percent_increase >= error_spike_threshold_percent and curr_errors > 0:
            # Now verify that memory stabilized after the spike
            # Check batches after this spike to see if they stayed at similar level
            batches_after_spike = len(memory_samples) - (i + 1)
            
            if batches_after_spike >= min_stable_batches:
                # Check if subsequent batches are stable (within tolerance of spike level)
                subsequent_batches = memory_samples[i + 1:i + 1 + min_stable_batches]
                
                # Check if all subsequent batches are within tolerance of the spike level
                is_stable = True
                max_variation = 0
                for next_memory in subsequent_batches:
                    variation_percent = abs((next_memory - curr_memory) / curr_memory * 100)
                    max_variation = max(max_variation, variation_percent)
                    if variation_percent > stabilization_tolerance_percent:
                        is_stable = False
                
                # Categorize based on stabilization
                if is_stable:
                    error_spike_batches.append(i + 1)  # Convert to 1-indexed for display
                else:
                    # Track spikes that didn't stabilize
                    non_stabilized_spikes.append({
                        'batch': i + 1,
                        'prev_memory': prev_memory,
                        'curr_memory': curr_memory,
                        'percent_increase': percent_increase,
                        'errors': curr_errors,
                        'max_variation': max_variation,
                        'next_batches': subsequent_batches
                    })
            else:
                # Not enough batches after spike to verify stabilization
                # Still flag it but note this in the message
                error_spike_batches.append(i + 1)
    
    # Build message based on what we found
    message_parts = []
    found_stabilized = len(error_spike_batches) > 0
    found_non_stabilized = len(non_stabilized_spikes) > 0
    
    if found_stabilized:
        # Build detailed message for stabilized spikes
        spike_details = []
        for batch_num in error_spike_batches:
            batch_idx = batch_num - 1  # Convert back to 0-indexed
            prev_memory = memory_samples[batch_idx - 1]
            curr_memory = memory_samples[batch_idx]
            percent_increase = ((curr_memory - prev_memory) / prev_memory) * 100
            errors = error_counts[batch_idx]
            
            # Check stabilization info
            batches_after = len(memory_samples) - (batch_idx + 1)
            if batches_after >= min_stable_batches:
                # Calculate average of stable batches
                stable_batches = memory_samples[batch_idx + 1:batch_idx + 1 + min_stable_batches]
                avg_after = sum(stable_batches) / len(stable_batches)
                stabilization_info = f" → stabilized at {avg_after:.2f} MB"
            else:
                stabilization_info = f" (insufficient batches after spike to confirm stabilization)"
            
            spike_details.append(
                f"  • Batch {batch_num}: {prev_memory:.2f} MB → {curr_memory:.2f} MB "
                f"(+{percent_increase:.1f}%) with {errors} error(s){stabilization_info}"
            )
        
        message_parts.append(
            f"ERROR-INDUCED MEMORY LEAK DETECTED\n"
            f"\n"
            f"Memory spikes occurred in batch(es) with errors and did not fully recover:\n"
            f"{chr(10).join(spike_details)}\n"
            f"\n"
            f"This indicates that error handling is not properly releasing resources.\n"
            f"Check error paths for unreleased connections, buffers, or cached data."
        )
    
    if found_non_stabilized:
        # Add information about non-stabilized spikes
        non_stable_details = []
        for spike in non_stabilized_spikes:
            next_mems = ", ".join([f"{m:.2f}" for m in spike['next_batches'][:3]])
            non_stable_details.append(
                f"  • Batch {spike['batch']}: {spike['prev_memory']:.2f} MB → {spike['curr_memory']:.2f} MB "
                f"(+{spike['percent_increase']:.1f}%) with {spike['errors']} error(s)\n"
                f"    Next batches: {next_mems} MB (variation {spike['max_variation']:.1f}%, did NOT stabilize)"
            )
        
        if found_stabilized:
            message_parts.append(
                f"\n"
                f"NOTE: Additional error spikes detected but memory did NOT stabilize:\n"
                f"{chr(10).join(non_stable_details)}\n"
                f"\n"
                f"These spikes show continued growth after the error, suggesting a continuous\n"
                f"memory leak pattern rather than error-induced stabilization."
            )
        else:
            message_parts.append(
                f"Error spikes detected with continued growth:\n"
                f"{chr(10).join(non_stable_details)}\n"
                f"\n"
                f"Memory continues to grow after error rather than stabilizing.\n"
                f"This suggests a continuous memory leak pattern that may have been\n"
                f"triggered by the error. Check both error paths AND ongoing operations."
            )
    
    if not found_stabilized and not found_non_stabilized:
        return False, "", []
    
    message = "\n".join(message_parts)
    
    # Only return True if we found stabilized spikes (the true error-induced pattern)
    return found_stabilized, message, error_spike_batches


def detect_memory_leak(
    growth_metrics: Dict[str, float],
    memory_samples: List[float],
    error_counts: Optional[List[int]] = None,
    max_growth_percent: float = DEFAULT_LEAK_DETECTION_MAX_GROWTH_PERCENT,
    stabilization_tolerance_mb: float = DEFAULT_LEAK_DETECTION_STABILIZATION_TOLERANCE_MB,
    tail_samples: int = DEFAULT_LEAK_DETECTION_TAIL_SAMPLES,
    error_spike_threshold_percent: float = DEFAULT_SIGNIFICANT_MEMORY_GROWTH_PERCENT
) -> Tuple[bool, str]:
    """
    Detect memory leaks based on growth metrics and continuous growth patterns.
    
    Args:
        growth_metrics: Dictionary from analyze_memory_growth()
        memory_samples: Original memory samples (not smoothed)
        error_counts: Optional list of error counts per batch for error-induced leak detection
        max_growth_percent: Maximum allowed growth percentage
        stabilization_tolerance_mb: Minimum growth to consider significant (MB)
        tail_samples: Number of final samples to check for continuous growth
        error_spike_threshold_percent: Minimum percent increase to consider an error spike
        
    Returns:
        Tuple of (leak_detected: bool, message: str)
    """
    initial_avg = growth_metrics['initial_avg']
    final_avg = growth_metrics['final_avg']
    growth = growth_metrics['growth']
    growth_percent = growth_metrics['growth_percent']
    
    # Check for error-induced memory leaks first (most specific pattern)
    if error_counts is not None:
        error_leak_detected, error_leak_message, spike_batches = detect_error_induced_memory_leak(
            memory_samples, error_counts, error_spike_threshold_percent
        )
        if error_leak_detected:
            return (True, error_leak_message)
    
    # Handle near-zero memory scenarios (tracemalloc may not track very small allocations)
    # If both initial and final averages are very close to zero, consider it as no growth
    if initial_avg < NEAR_ZERO_MEMORY_THRESHOLD_MB and final_avg < NEAR_ZERO_MEMORY_THRESHOLD_MB:
        # No meaningful memory to track - skip leak detection
        return (False, "Memory measurements near zero — tracemalloc tracking not sufficient for this workload")
    
    # Recalculate growth_percent properly to avoid division by zero issues
    if initial_avg > NEAR_ZERO_MEMORY_THRESHOLD_MB:
        growth_percent = (growth / initial_avg * 100)
    elif growth > NEAR_ZERO_MEMORY_THRESHOLD_MB:
        # If we started near zero but grew substantially, that's a potential leak
        growth_percent = (growth / NEAR_ZERO_MEMORY_THRESHOLD_MB * 100)
    else:
        growth_percent = 0
    
    # Check if overall growth exceeds threshold
    if growth_percent > max_growth_percent:
        return (
            True,
            f"Memory grew by {growth_percent:.1f}% "
            f"(>{max_growth_percent}% threshold) — possible leak"
        )
    
    # Check for continuous growth in final samples
    tail = memory_samples[-tail_samples:]
    if len(tail) >= 2:
        continuous_growth = all(
            (tail[i + 1] - tail[i]) > stabilization_tolerance_mb
            for i in range(len(tail) - 1)
        )
        if continuous_growth:
            return (
                True,
                "Continuous memory growth in final batches — possible leak"
            )
    
    return (False, "Memory stabilized — no leak detected")

