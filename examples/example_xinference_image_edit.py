#!/usr/bin/env python3
"""
Example script demonstrating Xinference image edit capabilities in LiteLLM.

This script shows how to use the newly implemented Xinference image edit
functionality through LiteLLM.

Requirements:
- LiteLLM installed
- Xinference server running (default: http://127.0.0.1:9997/v1)
- A compatible image generation model deployed in Xinference

Usage:
    python example_xinference_image_edit.py
"""

import os
from litellm import image_edit


def example_basic_image_edit():
    """
    Basic example: Edit an image using a text prompt.
    """
    print("Example 1: Basic Image Edit")
    print("-" * 60)
    
    # Configure Xinference API base
    os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
    
    try:
        # Note: You'll need an actual image file for this to work
        # with open("input_image.png", "rb") as image_file:
        #     response = image_edit(
        #         model="xinference/stabilityai/stable-diffusion-3.5-large",
        #         image=image_file,
        #         prompt="Make the colors more vibrant and add a sunset in the background",
        #         n=1,
        #         size="1024x1024",
        #         response_format="url"
        #     )
        #     
        #     print("Image edit successful!")
        #     print(f"Generated image URL: {response.data[0].url}")
        
        print("To run this example, uncomment the code and provide an input image.")
        print("Make sure Xinference is running with a compatible model.")
        
    except Exception as e:
        print(f"Error: {e}")


def example_image_edit_with_mask():
    """
    Advanced example: Edit specific parts of an image using a mask.
    """
    print("\nExample 2: Image Edit with Mask")
    print("-" * 60)
    
    try:
        # Note: You'll need actual image and mask files
        # with open("input_image.png", "rb") as image_file, \
        #      open("mask_image.png", "rb") as mask_file:
        #     
        #     response = image_edit(
        #         model="xinference/stabilityai/stable-diffusion-3.5-large",
        #         image=image_file,
        #         mask=mask_file,  # Transparent areas will be edited
        #         prompt="Add a beautiful garden in the masked area",
        #         n=1,
        #         size="1024x1024",
        #         response_format="b64_json",
        #         api_base="http://127.0.0.1:9997/v1"
        #     )
        #     
        #     print("Image edit with mask successful!")
        #     # The response will contain base64-encoded image data
        #     print(f"Generated {len(response.data)} image(s)")
        
        print("To run this example, uncomment the code and provide:")
        print("  1. An input image (input_image.png)")
        print("  2. A mask image with transparent areas (mask_image.png)")
        
    except Exception as e:
        print(f"Error: {e}")


def example_multiple_variations():
    """
    Example: Generate multiple variations of an edited image.
    """
    print("\nExample 3: Generate Multiple Variations")
    print("-" * 60)
    
    try:
        # Note: You'll need an actual image file
        # with open("input_image.png", "rb") as image_file:
        #     response = image_edit(
        #         model="xinference/stabilityai/stable-diffusion-3.5-large",
        #         image=image_file,
        #         prompt="Transform into a watercolor painting style",
        #         n=3,  # Generate 3 variations
        #         size="1024x1024",
        #         response_format="url"
        #     )
        #     
        #     print(f"Generated {len(response.data)} variations!")
        #     for i, img in enumerate(response.data, 1):
        #         print(f"  Variation {i}: {img.url}")
        
        print("To run this example, uncomment the code and provide an input image.")
        print("This will generate multiple variations of the edited image.")
        
    except Exception as e:
        print(f"Error: {e}")


def show_supported_parameters():
    """
    Display supported parameters for Xinference image edit.
    """
    print("\nSupported Parameters")
    print("=" * 60)
    
    parameters = {
        "model": {
            "type": "string",
            "required": True,
            "description": "The Xinference model to use (e.g., xinference/stabilityai/stable-diffusion-3.5-large)"
        },
        "image": {
            "type": "file",
            "required": True,
            "description": "The image to edit (file object opened in binary mode)"
        },
        "prompt": {
            "type": "string",
            "required": True,
            "description": "Text description of the desired edit"
        },
        "mask": {
            "type": "file",
            "required": False,
            "description": "Image with transparent areas indicating where to edit"
        },
        "n": {
            "type": "integer",
            "required": False,
            "description": "Number of images to generate (1-10)"
        },
        "size": {
            "type": "string",
            "required": False,
            "description": "Size of generated images (e.g., '1024x1024')"
        },
        "response_format": {
            "type": "string",
            "required": False,
            "description": "Response format: 'url' or 'b64_json'"
        },
        "api_base": {
            "type": "string",
            "required": False,
            "description": "Xinference API base URL (default: http://127.0.0.1:9997/v1)"
        }
    }
    
    for param, details in parameters.items():
        req = "Required" if details["required"] else "Optional"
        print(f"\n{param} ({details['type']}) - {req}")
        print(f"  {details['description']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Xinference Image Edit Examples")
    print("=" * 60)
    
    show_supported_parameters()
    
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    example_basic_image_edit()
    example_image_edit_with_mask()
    example_multiple_variations()
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Ensure Xinference is running:")
    print("   xinference-local --host 127.0.0.1 --port 9997")
    print("2. Deploy an image generation model in Xinference")
    print("3. Uncomment and run the examples above")
    print("=" * 60)
