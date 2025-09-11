import { MessageType } from "./types";
import { EndpointType } from "./mode_endpoint_mapping";


interface CodeGenMetadata {
	tags?: string[];
	vector_stores?: string[];
	guardrails?: string[];
}

interface GenerateCodeParams {
	apiKeySource: 'session' | 'custom';
	accessToken: string | null;
	apiKey: string;
	inputMessage: string;
	chatHistory: MessageType[];
	selectedTags: string[];
	selectedVectorStores: string[];
	selectedGuardrails: string[];
	selectedMCPTools: string[];
	endpointType: string;
	selectedModel: string | undefined;
	selectedSdk: 'openai' | 'azure';
}

export const generateCodeSnippet = (params: GenerateCodeParams): string => {
	const {
		apiKeySource,
		accessToken,
		apiKey,
		inputMessage,
		chatHistory,
		selectedTags,
		selectedVectorStores,
		selectedGuardrails,
		selectedMCPTools,
		endpointType,
		selectedModel,
		selectedSdk,
	} = params;
	const effectiveApiKey = apiKeySource === 'session' ? accessToken : apiKey;
	const apiBase = window.location.origin;

	// Always get the input message early on, regardless of what happens later
	const userPrompt = inputMessage || "Your prompt here"; // Fallback if inputMessage is empty

	// Safely escape the prompt to prevent issues with quotes
	const safePrompt = userPrompt.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n');

	const messages = chatHistory
			.filter(msg => !msg.isImage)
			.map(({ role, content }) => ({ role, content }));

	const metadata: CodeGenMetadata = {};
	if (selectedTags.length > 0) metadata.tags = selectedTags;
	if (selectedVectorStores.length > 0) metadata.vector_stores = selectedVectorStores;
	if (selectedGuardrails.length > 0) metadata.guardrails = selectedGuardrails;

	const modelNameForCode = selectedModel || 'your-model-name';

	const clientInitialization = selectedSdk === 'azure'
			? `import openai

client = openai.AzureOpenAI(
	api_key="${effectiveApiKey || 'YOUR_LITELLM_API_KEY'}",
	azure_endpoint="${apiBase}",
	api_version="2024-02-01"
)`
			: `import openai

client = openai.OpenAI(
	api_key="${effectiveApiKey || 'YOUR_LITELLM_API_KEY'}",
	base_url="${apiBase}"
)`;

	let endpointSpecificCode;
	switch (endpointType) {
			case EndpointType.CHAT: {
					const metadataIsNotEmpty = Object.keys(metadata).length > 0;
					let extraBodyCode = '';
					if (metadataIsNotEmpty) {
						const extraBodyObject = { metadata };
						const extraBodyString = JSON.stringify(extraBodyObject, null, 2);
						const indentedExtraBodyString = extraBodyString.split('\n').map(line => ' '.repeat(4) + line).join('\n').trim();

						extraBodyCode = `,\n    extra_body=${indentedExtraBodyString}`;
					}

					// Create example for chat completions with optional image support
					const messagesExample = messages.length > 0 ? messages : [{ role: "user", content: userPrompt }];

					endpointSpecificCode = `
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${modelNameForCode}",
    messages=${JSON.stringify(messagesExample, null, 4)}${extraBodyCode}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${modelNameForCode}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${safePrompt}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${extraBodyCode}
# )
# print(response_with_file)
`;
					break;
			}
			case EndpointType.RESPONSES: {
				const metadataIsNotEmpty = Object.keys(metadata).length > 0;
				let extraBodyCode = '';
				if (metadataIsNotEmpty) {
					const extraBodyObject = { metadata };
					const extraBodyString = JSON.stringify(extraBodyObject, null, 2);
					const indentedExtraBodyString = extraBodyString.split('\n').map(line => ' '.repeat(4) + line).join('\n').trim();

					extraBodyCode = `,\n    extra_body=${indentedExtraBodyString}`;
				}

				// Create example for responses API with optional image support
				const inputExample = messages.length > 0 ? messages : [{ role: "user", content: userPrompt }];
				
				endpointSpecificCode = `
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${modelNameForCode}",
    input=${JSON.stringify(inputExample, null, 4)}${extraBodyCode}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${modelNameForCode}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${safePrompt}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${extraBodyCode}
# )
# print(response_with_file.output_text)
`;
				break;
			}
			case EndpointType.IMAGE:
				if (selectedSdk === 'azure') {
						endpointSpecificCode = `
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${modelNameForCode}",
	prompt="${inputMessage}",
	n=1
)

json_response = json.loads(result.model_dump_json())

# Set the directory for the stored image
image_dir = os.path.join(os.curdir, 'images')

# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
	os.mkdir(image_dir)

# Initialize the image path
image_filename = f"generated_image_{int(time.time())}.png"
image_path = os.path.join(image_dir, image_filename)

try:
	# Retrieve the generated image
	if json_response.get("data") && len(json_response["data"]) > 0 && json_response["data"][0].get("url"):
			image_url = json_response["data"][0]["url"]
			generated_image = requests.get(image_url).content
			with open(image_path, "wb") as image_file:
					image_file.write(generated_image)

			print(f"Image saved to {image_path}")
			# Display the image
			image = Image.open(image_path)
			image.show()
	else:
			print("Could not find image URL in response.")
			print("Full response:", json_response)
except Exception as e:
	print(f"An error occurred: {e}")
	print("Full response:", json_response)
`;
					} else {
						endpointSpecificCode = `
import base64
import os
import time
import json
from PIL import Image
import requests

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to create a file (simplified for this example)
def create_file(image_path):
	# In a real implementation, this would upload the file to OpenAI
	# For this example, we'll just return a placeholder ID
	return f"file_{os.path.basename(image_path).replace('.', '_')}"

# The prompt entered by the user
prompt = "${safePrompt}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${modelNameForCode}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`;
				}
				break;

			case EndpointType.IMAGE_EDITS:
				if (selectedSdk === 'azure') {
						endpointSpecificCode = `
import base64
import os
import time
import json
from PIL import Image
import requests

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# The prompt entered by the user
prompt = "${safePrompt}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${modelNameForCode}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`;
					} else {
							endpointSpecificCode = `
import base64
import os
import time

# Helper function to encode images to base64
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to create a file (simplified for this example)
def create_file(image_path):
	# In a real implementation, this would upload the file to OpenAI
	# For this example, we'll just return a placeholder ID
	return f"file_{os.path.basename(image_path).replace('.', '_')}"

# The prompt entered by the user
prompt = "${safePrompt}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${modelNameForCode}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
							},
							{
									"type": "input_image",
									"file_id": file_id2,
							}
					],
			}
	],
	tools=[{"type": "image_generation"}],
)

# Process the response
image_generation_calls = [
	output
	for output in response.output
	if output.type == "image_generation_call"
]

image_data = [output.result for output in image_generation_calls]

if image_data:
	image_base64 = image_data[0]
	image_filename = f"edited_image_{int(time.time())}.png"
	with open(image_filename, "wb") as f:
			f.write(base64.b64decode(image_base64))
	print(f"Image saved to {image_filename}")
else:
	# If no image is generated, there might be a text response with an explanation
	text_response = [output.text for output in response.output if hasattr(output, 'text')]
	if text_response:
			print("No image generated. Model response:")
			print("\\n".join(text_response))
	else:
			print("No image data found in response.")
	print("Full response for debugging:")
	print(response)
`;
				}
				break;
		default:
				endpointSpecificCode = "\n# Code generation for this endpoint is not implemented yet.";
	}

	const finalCode = `${clientInitialization}\n${endpointSpecificCode}`;

	return finalCode;
}; 