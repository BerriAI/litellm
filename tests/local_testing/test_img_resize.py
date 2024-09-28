from typing import Literal


def calculage_img_tokens(
    width,
    height,
    mode: Literal["low", "high", "auto"] = "auto",
    base_tokens: int = 85,  # openai default - https://openai.com/pricing
):
    if mode == "low":
        return base_tokens
    elif mode == "high" or mode == "auto":
        resized_width, resized_height = resize_image_high_res(
            width=width, height=height
        )
        tiles_needed_high_res = calculate_tiles_needed(resized_width, resized_height)
        tile_tokens = (base_tokens * 2) * tiles_needed_high_res
        total_tokens = base_tokens + tile_tokens
        return total_tokens


def resize_image_high_res(width, height):
    # Maximum dimensions for high res mode
    max_short_side = 768
    max_long_side = 2000

    # Determine the longer and shorter sides
    longer_side = max(width, height)
    shorter_side = min(width, height)

    # Calculate the aspect ratio
    aspect_ratio = longer_side / shorter_side

    # Resize based on the short side being 768px
    if width <= height:  # Portrait or square
        resized_width = max_short_side
        resized_height = int(resized_width * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_height > max_long_side:
            resized_height = max_long_side
            resized_width = int(resized_height / aspect_ratio)
    else:  # Landscape
        resized_height = max_short_side
        resized_width = int(resized_height * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_width > max_long_side:
            resized_width = max_long_side
            resized_height = int(resized_width / aspect_ratio)

    return resized_width, resized_height


# Test the function with the given example
def calculate_tiles_needed(
    resized_width, resized_height, tile_width=512, tile_height=512
):
    tiles_across = (resized_width + tile_width - 1) // tile_width
    tiles_down = (resized_height + tile_height - 1) // tile_height
    total_tiles = tiles_across * tiles_down
    return total_tiles


# Test high res mode with 1875 x 768 image
resized_width_high_res = 1875
resized_height_high_res = 768
tiles_needed_high_res = calculate_tiles_needed(
    resized_width_high_res, resized_height_high_res
)
print(
    f"Tiles needed for high res image ({resized_width_high_res}x{resized_height_high_res}): {tiles_needed_high_res}"
)

# If you had the original size and needed to resize and then calculate tiles:
original_size = (10000, 4096)
resized_size_high_res = resize_image_high_res(*original_size)
print(f"Resized dimensions in high res mode: {resized_size_high_res}")
tiles_needed = calculate_tiles_needed(*resized_size_high_res)
print(f"Tiles needed for high res image {resized_size_high_res}: {tiles_needed}")
