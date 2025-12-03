"""
Demo: What does tiktoken produce for repeated characters?
"""
import tiktoken

# Get the encoding used by GPT models
encoding = tiktoken.get_encoding("cl100k_base")

# Example 1: Normal text
normal_text = "Hello world, how are you today?"
normal_tokens = encoding.encode(normal_text)
print("=" * 60)
print("NORMAL TEXT:")
print(f"Input: {normal_text}")
print(f"Length: {len(normal_text)} chars")
print(f"Tokens: {normal_tokens}")
print(f"Token count: {len(normal_tokens)} tokens")
print(f"Ratio: {len(normal_text) / len(normal_tokens):.2f} chars/token")
print()

# Example 2: Repeated box-drawing characters (like in bad_request.json)
box_char = "─"
repeated_box = box_char * 100  # 100 box-drawing chars
box_tokens = encoding.encode(repeated_box)
print("=" * 60)
print("REPEATED BOX-DRAWING CHARS (─):")
print(f"Input: {repeated_box[:50]}... (100 total)")
print(f"Length: {len(repeated_box)} chars")
print(f"Tokens: {box_tokens[:20]}... (first 20 shown)")
print(f"Token count: {len(box_tokens)} tokens")
print(f"Ratio: {len(repeated_box) / len(box_tokens):.2f} chars/token")
print()

# Example 3: Repeated spaces
spaces = " " * 100
space_tokens = encoding.encode(spaces)
print("=" * 60)
print("REPEATED SPACES:")
print(f"Input: '{spaces[:20]}...' (100 total)")
print(f"Length: {len(spaces)} chars")
print(f"Tokens: {space_tokens[:20]}... (first 20 shown)")
print(f"Token count: {len(space_tokens)} tokens")
print(f"Ratio: {len(spaces) / len(space_tokens):.2f} chars/token")
print()

# Example 4: Mixed repetitive content (like your payload)
mixed = "─" * 50 + " " * 50 + "text" + "\n" * 10
mixed_tokens = encoding.encode(mixed)
print("=" * 60)
print("MIXED REPETITIVE (50 ─, 50 spaces, text, 10 newlines):")
print(f"Length: {len(mixed)} chars")
print(f"Token count: {len(mixed_tokens)} tokens")
print(f"Ratio: {len(mixed) / len(mixed_tokens):.2f} chars/token")
print()

# Example 5: What would your 2.7M chars look like?
your_payload_chars = 2_687_926
estimated_box_chars = int(your_payload_chars * 0.49)  # 49% box chars
estimated_spaces = int(your_payload_chars * 0.34)     # 34% spaces
estimated_other = your_payload_chars - estimated_box_chars - estimated_spaces

print("=" * 60)
print("YOUR PAYLOAD ESTIMATE (2,687,926 chars):")
print(f"  Box chars (49%): {estimated_box_chars:,}")
print(f"  Spaces (34%):    {estimated_spaces:,}")
print(f"  Other (17%):     {estimated_other:,}")
print()

# Estimate tokens for each type
box_sample = "─" * 1000
box_sample_tokens = len(encoding.encode(box_sample))
box_ratio = len(box_sample) / box_sample_tokens

space_sample = " " * 1000
space_sample_tokens = len(encoding.encode(space_sample))
space_ratio = len(space_sample) / space_sample_tokens

estimated_box_tokens = estimated_box_chars / box_ratio
estimated_space_tokens = estimated_spaces / space_ratio
estimated_other_tokens = estimated_other / 4  # Assume 4 chars/token for text

total_estimated_tokens = estimated_box_tokens + estimated_space_tokens + estimated_other_tokens

print(f"Estimated tokens:")
print(f"  From box chars:  {estimated_box_tokens:,.0f}")
print(f"  From spaces:     {estimated_space_tokens:,.0f}")
print(f"  From other:      {estimated_other_tokens:,.0f}")
print(f"  TOTAL:           {total_estimated_tokens:,.0f} tokens")
print()
print(f"Why it takes 138 seconds:")
print(f"  - tiktoken must process EVERY character")
print(f"  - No caching/optimization for repetition")
print(f"  - BPE algorithm runs on full 2.7M string")
print(f"  - Single-threaded operation")
print(f"  - ~19,400 chars/second processing rate")

print()
print("=" * 60)
print("KEY INSIGHT:")
print("Even though the content is 83% repetitive (box + spaces),")
print("tiktoken CANNOT optimize this and must process every single")
print("character through its Byte Pair Encoding algorithm.")
print("=" * 60)

