#!/usr/bin/env python3
"""
Recraft AI Image Generation with LiteLLM - Usage Examples

This example demonstrates how to use Recraft AI for image generation through LiteLLM.
Recraft AI provides state-of-the-art image generation with support for various styles.

Requirements:
- LiteLLM installed (`pip install litellm`)
- Recraft API key from https://www.recraft.ai

Usage:
    python recraft_image_generation_example.py
"""

import os
import asyncio
import litellm

# Set your Recraft API key
# os.environ["RECRAFT_API_KEY"] = "your-api-key-here"

async def basic_image_generation():
    """Example 1: Basic image generation with Recraft V3."""
    print("Example 1: Basic image generation")
    
    response = await litellm.aimage_generation(
        model="recraft/recraftv3",
        prompt="a beautiful sunset over mountains",
        api_key="your-api-key"
    )
    
    print(f"Generated image URL: {response.data[0].url}")
    return response

async def advanced_image_generation():
    """Example 2: Advanced image generation with Recraft-specific parameters."""
    print("\nExample 2: Advanced image generation")
    
    response = await litellm.aimage_generation(
        model="recraft/recraftv3",
        prompt="a futuristic city skyline",
        style="digital_illustration",
        substyle="hand_drawn",
        size="1024x1024",
        n=2,
        negative_prompt="blurry, low quality",
        controls={
            "colors": [{"rgb": [0, 255, 0]}],  # Green color preference
            "artistic_level": 3  # Medium artistic level
        },
        api_key="your-api-key"
    )
    
    for i, image in enumerate(response.data):
        print(f"Generated image {i+1} URL: {image.url}")
    return response

async def vector_illustration():
    """Example 3: Vector illustration generation."""
    print("\nExample 3: Vector illustration")
    
    response = await litellm.aimage_generation(
        model="recraft/recraftv3",
        prompt="a minimalist logo of a tree",
        style="vector_illustration",
        substyle="line_art",
        size="1024x1024",
        api_key="your-api-key"
    )
    
    print(f"Generated vector image URL: {response.data[0].url}")
    return response

async def custom_style_example():
    """Example 4: Using custom style reference."""
    print("\nExample 4: Custom style reference")
    
    # Note: You would need to create a custom style first via Recraft API
    # and get the style_id
    response = await litellm.aimage_generation(
        model="recraft/recraftv3",
        prompt="a cartoon character",
        style_id="your-custom-style-id",  # Replace with actual style ID
        api_key="your-api-key"
    )
    
    print(f"Generated custom style image URL: {response.data[0].url}")
    return response

def sync_image_generation():
    """Example 5: Synchronous image generation."""
    print("\nExample 5: Synchronous generation")
    
    response = litellm.image_generation(
        model="recraft/recraftv3",
        prompt="a magical forest scene",
        style="digital_illustration",
        api_key="your-api-key"
    )
    
    print(f"Generated image URL: {response.data[0].url}")
    return response

async def text_in_image():
    """Example 6: Recraft V3 text layout feature."""
    print("\nExample 6: Text layout (Recraft V3)")
    
    response = await litellm.aimage_generation(
        model="recraft/recraftv3",
        prompt="cute red panda with a sign",
        style="digital_illustration",
        text_layout=[
            {
                "text": "Recraft",
                "bbox": [[0.3, 0.45], [0.6, 0.45], [0.6, 0.55], [0.3, 0.55]],
            },
            {
                "text": "AI",
                "bbox": [[0.62, 0.45], [0.70, 0.45], [0.70, 0.55], [0.62, 0.55]],
            },
        ],
        api_key="your-api-key"
    )
    
    print(f"Generated text image URL: {response.data[0].url}")
    return response

async def main():
    """Run all examples."""
    print("🎨 Recraft AI Image Generation Examples\n")
    
    # Check if API key is set
    api_key = os.getenv("RECRAFT_API_KEY")
    if not api_key:
        print("⚠️  Warning: RECRAFT_API_KEY environment variable not set.")
        print("   Set your API key to run actual generation examples.")
        print("   For now, showing code examples only.\n")
        show_code_examples()
        return
    
    try:
        # Run all examples
        await basic_image_generation()
        await advanced_image_generation()
        await vector_illustration()
        await custom_style_example()
        sync_image_generation()
        await text_in_image()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure your API key is valid and you have sufficient credits.")

def show_code_examples():
    """Show code examples without making actual API calls."""
    
    examples = [
        {
            "title": "Basic Usage",
            "code": '''response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a beautiful sunset over mountains",
    api_key="your-api-key"
)'''
        },
        {
            "title": "Advanced with Recraft Features",
            "code": '''response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a futuristic city skyline",
    style="digital_illustration",
    substyle="hand_drawn",
    size="1024x1024",
    n=2,
    negative_prompt="blurry, low quality",
    controls={"colors": [{"rgb": [0, 255, 0]}]},
    api_key="your-api-key"
)'''
        },
        {
            "title": "Vector Illustration",
            "code": '''response = await litellm.aimage_generation(
    model="recraft/recraftv3", 
    prompt="minimalist tree logo",
    style="vector_illustration",
    substyle="line_art",
    api_key="your-api-key"
)'''
        },
        {
            "title": "Custom Style",
            "code": '''response = await litellm.aimage_generation(
    model="recraft/recraftv3",
    prompt="a cartoon character", 
    style_id="custom-style-uuid",
    api_key="your-api-key"
)'''
        },
        {
            "title": "Synchronous Call",
            "code": '''response = litellm.image_generation(
    model="recraft/recraftv3",
    prompt="a magical forest",
    style="digital_illustration", 
    api_key="your-api-key"
)'''
        }
    ]
    
    for example in examples:
        print(f"📝 {example['title']}:")
        print(example['code'])
        print()
    
    print("\n📚 Supported Recraft Models:")
    print("- recraft/recraftv3 (default, latest)")
    print("- recraft/recraftv2")
    
    print("\n🎨 Supported Styles:")
    print("- realistic_image (photographic)")
    print("- digital_illustration (artwork)")
    print("- vector_illustration (scalable graphics)")
    print("- icon (simple symbols)")
    
    print("\n⚙️  Recraft-Specific Parameters:")
    print("- style: Base style for generation")
    print("- substyle: Style refinement")
    print("- negative_prompt: What to avoid")
    print("- controls: Advanced settings")
    print("- text_layout: Text positioning (V3)")
    print("- style_id: Custom style reference")
    
    print("\n🔧 Setup:")
    print("1. Get API key: https://www.recraft.ai")
    print("2. Set environment: export RECRAFT_API_KEY='your-key'")
    print("3. Install LiteLLM: pip install litellm")
    print("4. Use model: 'recraft/recraftv3'")

if __name__ == "__main__":
    asyncio.run(main())