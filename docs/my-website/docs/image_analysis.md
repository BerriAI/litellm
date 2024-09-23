
# Image Analysis

## Description

This route enables the analysis and interpretation of images using pre-trained models. You can use it to describe images, extract structured information, or perform other vision-related tasks based on the instructions and models provided.

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "azure_ai/phi35-vision-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the given image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://www.imperial.ac.uk/media/migration/administration-and-support-services/03--tojpeg_1487001148545_x2.jpg"
            }
          }
        ]
      }
    ]
  }'
```

### Input Parameters

- **model**: _string_  
    The model to use for image analysis. Example: `azure_ai/phi35-vision-instruct`.
    
- **messages**: _array_  
    An array of message objects that form the conversation between the user and the assistant.
    - **role**: _string_  
        The role of the message sender. Can be `"system"`, `"user"`, or `"assistant"`.
    - **content**: _array_  
        An array of content items. Each content item can be either a text instruction or an image URL.
        - **type**: _string_  
            Specifies the type of content; either `"text"` or `"image_url"`.
            - If **type** is `"text"`, the content item must include:
                - **text**: _string_  
                    The text instruction or message content (e.g., "Describe the image", "Extract information from this receipt").
            - If **type** is `"image_url"`, the content item must include:
                - **image_url**: _object_  
                    An object containing:
                    - **url**: _string_  
                        The URL of the image to be analyzed.

### Example: Structured Data Extraction

This example demonstrates how to use the API to extract structured data, such as fields from a receipt.


```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "azure_ai/phi35-vision-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant. Your task is to extract data from payment receipts in JSON format following the schema: {\"date\": \"receipt date\", \"transaction\": \"transaction number\", \"location\": \"transaction location\", \"amount\": \"transaction amount\"}."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Extract and provide the information from this receipt in JSON format."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://www.imperial.ac.uk/media/migration/administration-and-support-services/03--tojpeg_1487001148545_x2.jpg"
            }
          }
        ]
      }
    ]
  }'
```

### Required Fields

When making a request to the API, the following fields are required:

- **model**: _string_  
    The identifier of the model used for image analysis (e.g., `azure_ai/phi35-vision-instruct`).
    
- **messages**: _array_  
    The conversation messages, including the user's instructions and the image URL.
    
### Example Model for Image Analysis

In this example, the **`azure_ai/phi35-vision-instruct`** model was used to demonstrate image analysis tasks. This model supports structured JSON output by default for tasks such as image description and data extraction.

However, other vision models can also be used for image analysis tasks, with the possibility of requiring additional configuration for structured outputs.

### Supported Output Formats

- **JSON**:  
    Extracted structured data in JSON format (default support in `phi35-vision-instruct`).
    
- **Text Description**:  
    A descriptive text of the image content.
