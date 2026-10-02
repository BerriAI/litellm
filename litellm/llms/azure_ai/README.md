`/chat/completion` calls route through `openai.py`

FLUX.2 image pricing uses `output_cost_per_image_first_megapixel` for the complete first output megapixel,
`output_cost_per_pixel` for additional output megapixels, and `input_cost_per_megapixel` for reference images.
A megapixel is 1,048,576 pixels. Without a first-megapixel price, the output pixel rate applies to every
rounded output megapixel. Returned image dimensions take precedence over requested dimensions

Existing `output_cost_per_image`, `input_cost_per_image`, and `input_cost_per_pixel` overrides keep their
legacy meaning unless a new megapixel pricing field is supplied. This applies to Router deployments and
`register_model()` overrides. The legacy catalog fields also retain their values for older LiteLLM clients

For custom megapixel pricing, set `output_cost_per_pixel` and `input_cost_per_megapixel` independently.
Set `output_cost_per_image_first_megapixel` when the first output megapixel has a different price.
An explicit zero is a free rate. A flat `output_cost_per_image` remains the complete output price unless
an explicit first-megapixel price replaces it
