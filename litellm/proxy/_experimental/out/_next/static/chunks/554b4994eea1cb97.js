(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,771674,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var s=e.i(9583),i=a.forwardRef(function(e,i){return a.createElement(s.default,(0,t.default)({},e,{ref:i,icon:r}))});e.s(["UserOutlined",0,i],771674)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var s=e.i(9583),i=a.forwardRef(function(e,i){return a.createElement(s.default,(0,t.default)({},e,{ref:i,icon:r}))});e.s(["SafetyOutlined",0,i],602073)},818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return s}});let r=e.r(271645);function s(e,t){let a=(0,r.useRef)(null),s=(0,r.useRef)(null);return(0,r.useCallback)(r=>{if(null===r){let e=a.current;e&&(a.current=null,e());let t=s.current;t&&(s.current=null,t())}else e&&(a.current=i(e,r)),t&&(s.current=i(t,r))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},190272,785913,e=>{"use strict";var t,a,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),s=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>s,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:r,apiKey:i,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:m,selectedMCPServers:p,mcpServers:u,mcpServerToolRestrictions:g,selectedVoice:x,endpointType:h,selectedModel:f,selectedSdk:_,proxySettings:b}=e,v="session"===a?r:i,y=window.location.origin,j=b?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?y=j:b?.PROXY_BASE_URL&&(y=b.PROXY_BASE_URL);let N=l||"Your prompt here",A=N.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),w={};o.length>0&&(w.tags=o),c.length>0&&(w.vector_stores=c),d.length>0&&(w.guardrails=d),m.length>0&&(w.policies=m);let T=f||"your-model-name",C="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case s.CHAT:{let e=Object.keys(w).length>0,a="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let r=S.length>0?S:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${T}",
    messages=${JSON.stringify(r,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${T}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${A}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${a}
# )
# print(response_with_file)
`;break}case s.RESPONSES:{let e=Object.keys(w).length>0,a="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let r=S.length>0?S:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${T}",
    input=${JSON.stringify(r,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${T}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${A}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case s.IMAGE:t="azure"===_?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${T}",
	prompt="${l}",
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
`:`
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
prompt = "${A}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
`;break;case s.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${A}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
`:`
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
prompt = "${A}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
`;break;case s.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${l||"Your string here"}",
	model="${T}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case s.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${T}",
	file=audio_file${l?`,
	prompt="${l.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case s.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${T}",
	input="${l||"Your text to convert to speech here"}",
	voice="${x}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${T}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${t}`}],190272)},94629,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,a],94629)},916925,e=>{"use strict";var t,a=((t={}).A2A_Agent="A2A Agent",t.AIML="AI/ML API",t.Bedrock="Amazon Bedrock",t.Anthropic="Anthropic",t.AssemblyAI="AssemblyAI",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.Cerebras="Cerebras",t.Cohere="Cohere",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.ElevenLabs="ElevenLabs",t.FalAI="Fal AI",t.FireworksAI="Fireworks AI",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.Hosted_Vllm="vllm",t.Infinity="Infinity",t.JinaAI="Jina AI",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.Ollama="Ollama",t.OpenAI="OpenAI",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.Perplexity="Perplexity",t.RunwayML="RunwayML",t.Sambanova="Sambanova",t.Snowflake="Snowflake",t.TogetherAI="TogetherAI",t.Triton="Triton",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.xAI="xAI",t.SAP="SAP Generative AI Hub",t.Watsonx="Watsonx",t);let r={A2A_Agent:"a2a_agent",AIML:"aiml",OpenAI:"openai",OpenAI_Text:"text-completion-openai",Azure:"azure",Azure_AI_Studio:"azure_ai",Anthropic:"anthropic",Google_AI_Studio:"gemini",Bedrock:"bedrock",Groq:"groq",MiniMax:"minimax",MistralAI:"mistral",Cohere:"cohere",OpenAI_Compatible:"openai",OpenAI_Text_Compatible:"text-completion-openai",Vertex_AI:"vertex_ai",Databricks:"databricks",Dashscope:"dashscope",xAI:"xai",Deepseek:"deepseek",Ollama:"ollama",AssemblyAI:"assemblyai",Cerebras:"cerebras",Sambanova:"sambanova",Perplexity:"perplexity",RunwayML:"runwayml",TogetherAI:"together_ai",Openrouter:"openrouter",Oracle:"oci",Snowflake:"snowflake",FireworksAI:"fireworks_ai",GradientAI:"gradient_ai",Triton:"triton",Deepgram:"deepgram",ElevenLabs:"elevenlabs",FalAI:"fal_ai",SageMaker:"sagemaker_chat",Voyage:"voyage",JinaAI:"jina_ai",VolcEngine:"volcengine",DeepInfra:"deepinfra",Hosted_Vllm:"hosted_vllm",Infinity:"infinity",SAP:"sap",Watsonx:"watsonx"},s="../ui/assets/logos/",i={"A2A Agent":`${s}a2a_agent.png`,"AI/ML API":`${s}aiml_api.svg`,Anthropic:`${s}anthropic.svg`,AssemblyAI:`${s}assemblyai_small.png`,Azure:`${s}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${s}microsoft_azure.svg`,"Amazon Bedrock":`${s}bedrock.svg`,"AWS SageMaker":`${s}bedrock.svg`,Cerebras:`${s}cerebras.svg`,Cohere:`${s}cohere.svg`,"Databricks (Qwen API)":`${s}databricks.svg`,Dashscope:`${s}dashscope.svg`,Deepseek:`${s}deepseek.svg`,"Fireworks AI":`${s}fireworks.svg`,Groq:`${s}groq.svg`,"Google AI Studio":`${s}google.svg`,vllm:`${s}vllm.png`,Infinity:`${s}infinity.png`,MiniMax:`${s}minimax.svg`,"Mistral AI":`${s}mistral.svg`,Ollama:`${s}ollama.svg`,OpenAI:`${s}openai_small.svg`,"OpenAI Text Completion":`${s}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${s}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${s}openai_small.svg`,Openrouter:`${s}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${s}oracle.svg`,Perplexity:`${s}perplexity-ai.svg`,RunwayML:`${s}runwayml.png`,Sambanova:`${s}sambanova.svg`,Snowflake:`${s}snowflake.svg`,TogetherAI:`${s}togetherai.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${s}google.svg`,xAI:`${s}xai.svg`,GradientAI:`${s}gradientai.svg`,Triton:`${s}nvidia_triton.png`,Deepgram:`${s}deepgram.png`,ElevenLabs:`${s}elevenlabs.png`,"Fal AI":`${s}fal_ai.jpg`,"Voyage AI":`${s}voyage.webp`,"Jina AI":`${s}jina.png`,VolcEngine:`${s}volcengine.png`,DeepInfra:`${s}deepinfra.png`,"SAP Generative AI Hub":`${s}sap.png`};e.s(["Providers",()=>a,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:i[e],displayName:e}}let t=Object.keys(r).find(t=>r[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let s=a[t];return{logo:i[s],displayName:s}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let a=r[e];console.log(`Provider mapped to: ${a}`);let s=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let r=t.litellm_provider;(r===a||"string"==typeof r&&r.includes(a))&&s.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&s.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&s.push(e)}))),s},"providerLogoMap",0,i,"provider_map",0,r])},262218,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),r=e.i(529681),s=e.i(702779),i=e.i(563113),l=e.i(763731),n=e.i(121872),o=e.i(242064);e.i(296059);var c=e.i(915654);e.i(262370);var d=e.i(135551),m=e.i(183293),p=e.i(246422),u=e.i(838378);let g=e=>{let{lineWidth:t,fontSizeIcon:a,calc:r}=e,s=e.fontSizeSM;return(0,u.mergeToken)(e,{tagFontSize:s,tagLineHeight:(0,c.unit)(r(e.lineHeightSM).mul(s).equal()),tagIconSize:r(a).sub(r(t).mul(2)).equal(),tagPaddingHorizontal:8,tagBorderlessBg:e.defaultBg})},x=e=>({defaultBg:new d.FastColor(e.colorFillQuaternary).onBackground(e.colorBgContainer).toHexString(),defaultColor:e.colorText}),h=(0,p.genStyleHooks)("Tag",e=>(e=>{let{paddingXXS:t,lineWidth:a,tagPaddingHorizontal:r,componentCls:s,calc:i}=e,l=i(r).sub(a).equal(),n=i(t).sub(a).equal();return{[s]:Object.assign(Object.assign({},(0,m.resetComponent)(e)),{display:"inline-block",height:"auto",marginInlineEnd:e.marginXS,paddingInline:l,fontSize:e.tagFontSize,lineHeight:e.tagLineHeight,whiteSpace:"nowrap",background:e.defaultBg,border:`${(0,c.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,opacity:1,transition:`all ${e.motionDurationMid}`,textAlign:"start",position:"relative",[`&${s}-rtl`]:{direction:"rtl"},"&, a, a:hover":{color:e.defaultColor},[`${s}-close-icon`]:{marginInlineStart:n,fontSize:e.tagIconSize,color:e.colorIcon,cursor:"pointer",transition:`all ${e.motionDurationMid}`,"&:hover":{color:e.colorTextHeading}},[`&${s}-has-color`]:{borderColor:"transparent",[`&, a, a:hover, ${e.iconCls}-close, ${e.iconCls}-close:hover`]:{color:e.colorTextLightSolid}},"&-checkable":{backgroundColor:"transparent",borderColor:"transparent",cursor:"pointer",[`&:not(${s}-checkable-checked):hover`]:{color:e.colorPrimary,backgroundColor:e.colorFillSecondary},"&:active, &-checked":{color:e.colorTextLightSolid},"&-checked":{backgroundColor:e.colorPrimary,"&:hover":{backgroundColor:e.colorPrimaryHover}},"&:active":{backgroundColor:e.colorPrimaryActive}},"&-hidden":{display:"none"},[`> ${e.iconCls} + span, > span + ${e.iconCls}`]:{marginInlineStart:l}}),[`${s}-borderless`]:{borderColor:"transparent",background:e.tagBorderlessBg}}})(g(e)),x);var f=function(e,t){var a={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(a[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var s=0,r=Object.getOwnPropertySymbols(e);s<r.length;s++)0>t.indexOf(r[s])&&Object.prototype.propertyIsEnumerable.call(e,r[s])&&(a[r[s]]=e[r[s]]);return a};let _=t.forwardRef((e,r)=>{let{prefixCls:s,style:i,className:l,checked:n,children:c,icon:d,onChange:m,onClick:p}=e,u=f(e,["prefixCls","style","className","checked","children","icon","onChange","onClick"]),{getPrefixCls:g,tag:x}=t.useContext(o.ConfigContext),_=g("tag",s),[b,v,y]=h(_),j=(0,a.default)(_,`${_}-checkable`,{[`${_}-checkable-checked`]:n},null==x?void 0:x.className,l,v,y);return b(t.createElement("span",Object.assign({},u,{ref:r,style:Object.assign(Object.assign({},i),null==x?void 0:x.style),className:j,onClick:e=>{null==m||m(!n),null==p||p(e)}}),d,t.createElement("span",null,c)))});var b=e.i(403541);let v=(0,p.genSubStyleComponent)(["Tag","preset"],e=>{let t;return t=g(e),(0,b.genPresetColor)(t,(e,{textColor:a,lightBorderColor:r,lightColor:s,darkColor:i})=>({[`${t.componentCls}${t.componentCls}-${e}`]:{color:a,background:s,borderColor:r,"&-inverse":{color:t.colorTextLightSolid,background:i,borderColor:i},[`&${t.componentCls}-borderless`]:{borderColor:"transparent"}}}))},x),y=(e,t,a)=>{let r="string"!=typeof a?a:a.charAt(0).toUpperCase()+a.slice(1);return{[`${e.componentCls}${e.componentCls}-${t}`]:{color:e[`color${a}`],background:e[`color${r}Bg`],borderColor:e[`color${r}Border`],[`&${e.componentCls}-borderless`]:{borderColor:"transparent"}}}},j=(0,p.genSubStyleComponent)(["Tag","status"],e=>{let t=g(e);return[y(t,"success","Success"),y(t,"processing","Info"),y(t,"error","Error"),y(t,"warning","Warning")]},x);var N=function(e,t){var a={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(a[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var s=0,r=Object.getOwnPropertySymbols(e);s<r.length;s++)0>t.indexOf(r[s])&&Object.prototype.propertyIsEnumerable.call(e,r[s])&&(a[r[s]]=e[r[s]]);return a};let A=t.forwardRef((e,c)=>{let{prefixCls:d,className:m,rootClassName:p,style:u,children:g,icon:x,color:f,onClose:_,bordered:b=!0,visible:y}=e,A=N(e,["prefixCls","className","rootClassName","style","children","icon","color","onClose","bordered","visible"]),{getPrefixCls:S,direction:w,tag:T}=t.useContext(o.ConfigContext),[C,k]=t.useState(!0),I=(0,r.default)(A,["closeIcon","closable"]);t.useEffect(()=>{void 0!==y&&k(y)},[y]);let M=(0,s.isPresetColor)(f),E=(0,s.isPresetStatusColor)(f),O=M||E,$=Object.assign(Object.assign({backgroundColor:f&&!O?f:void 0},null==T?void 0:T.style),u),P=S("tag",d),[L,z,D]=h(P),R=(0,a.default)(P,null==T?void 0:T.className,{[`${P}-${f}`]:O,[`${P}-has-color`]:f&&!O,[`${P}-hidden`]:!C,[`${P}-rtl`]:"rtl"===w,[`${P}-borderless`]:!b},m,p,z,D),H=e=>{e.stopPropagation(),null==_||_(e),e.defaultPrevented||k(!1)},[,F]=(0,i.useClosable)((0,i.pickClosable)(e),(0,i.pickClosable)(T),{closable:!1,closeIconRender:e=>{let r=t.createElement("span",{className:`${P}-close-icon`,onClick:H},e);return(0,l.replaceElement)(e,r,e=>({onClick:t=>{var a;null==(a=null==e?void 0:e.onClick)||a.call(e,t),H(t)},className:(0,a.default)(null==e?void 0:e.className,`${P}-close-icon`)}))}}),G="function"==typeof A.onClick||g&&"a"===g.type,B=x||null,U=B?t.createElement(t.Fragment,null,B,g&&t.createElement("span",null,g)):g,K=t.createElement("span",Object.assign({},I,{ref:c,className:R,style:$}),U,F,M&&t.createElement(v,{key:"preset",prefixCls:P}),E&&t.createElement(j,{key:"status",prefixCls:P}));return L(G?t.createElement(n.default,{component:"Tag"},K):K)});A.CheckableTag=_,e.s(["Tag",0,A],262218)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},115571,e=>{"use strict";let t="local-storage-change";function a(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function r(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function s(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function i(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>a,"getLocalStorageItem",()=>r,"removeLocalStorageItem",()=>i,"setLocalStorageItem",()=>s])},976883,174886,e=>{"use strict";var t=e.i(843476),a=e.i(275144),r=e.i(434626),s=e.i(271645);let i=s.forwardRef(function(e,t){return s.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),s.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"}))});var l=e.i(994388),n=e.i(304967),o=e.i(599724),c=e.i(629569),d=e.i(212931),m=e.i(199133),p=e.i(653496),u=e.i(262218),g=e.i(592968),x=e.i(991124);e.s(["Copy",()=>x.default],174886);var x=x;let h=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);var f=e.i(798496),_=e.i(727749),b=e.i(402874),v=e.i(764205),y=e.i(190272),j=e.i(785913),N=e.i(916925);let{TabPane:A}=p.Tabs;e.s(["default",0,({accessToken:e,isEmbedded:S=!1})=>{let w,T,C,k,I,M,E,[O,$]=(0,s.useState)(null),[P,L]=(0,s.useState)(null),[z,D]=(0,s.useState)(null),[R,H]=(0,s.useState)("LiteLLM Gateway"),[F,G]=(0,s.useState)(null),[B,U]=(0,s.useState)(""),[K,V]=(0,s.useState)({}),[W,q]=(0,s.useState)(!0),[Y,J]=(0,s.useState)(!0),[X,Q]=(0,s.useState)(!0),[Z,ee]=(0,s.useState)(""),[et,ea]=(0,s.useState)(""),[er,es]=(0,s.useState)(""),[ei,el]=(0,s.useState)([]),[en,eo]=(0,s.useState)([]),[ec,ed]=(0,s.useState)([]),[em,ep]=(0,s.useState)([]),[eu,eg]=(0,s.useState)([]),[ex,eh]=(0,s.useState)("I'm alive! ✓"),[ef,e_]=(0,s.useState)(!1),[eb,ev]=(0,s.useState)(!1),[ey,ej]=(0,s.useState)(!1),[eN,eA]=(0,s.useState)(null),[eS,ew]=(0,s.useState)(null),[eT,eC]=(0,s.useState)(null),[ek,eI]=(0,s.useState)({}),[eM,eE]=(0,s.useState)("models");(0,s.useEffect)(()=>{(async()=>{try{await (0,v.getUiConfig)()}catch(e){console.error("Failed to get UI config:",e)}let e=async()=>{try{q(!0);let e=await (0,v.modelHubPublicModelsCall)();console.log("ModelHubData:",e),$(e)}catch(e){console.error("There was an error fetching the public model data",e),eh("Service unavailable")}finally{q(!1)}},t=async()=>{try{J(!0);let e=await (0,v.agentHubPublicModelsCall)();console.log("AgentHubData:",e),L(e)}catch(e){console.error("There was an error fetching the public agent data",e)}finally{J(!1)}},a=async()=>{try{Q(!0);let e=await (0,v.mcpHubPublicServersCall)();console.log("MCPHubData:",e),D(e)}catch(e){console.error("There was an error fetching the public MCP server data",e)}finally{Q(!1)}};(async()=>{let e=await (0,v.getPublicModelHubInfo)();console.log("Public Model Hub Info:",e),H(e.docs_title),G(e.custom_docs_description),U(e.litellm_version),V(e.useful_links||{})})(),e(),t(),a()})()},[]),(0,s.useEffect)(()=>{},[Z,ei,en,ec]);let eO=(0,s.useMemo)(()=>{if(!O||!Array.isArray(O))return[];let e=O;if(Z.trim()){let t=Z.toLowerCase(),a=t.split(/\s+/),r=O.filter(e=>{let r=e.model_group.toLowerCase();return!!r.includes(t)||a.every(e=>r.includes(e))});r.length>0&&(e=r.sort((e,a)=>{let r=e.model_group.toLowerCase(),s=a.model_group.toLowerCase(),i=1e3*(r===t),l=1e3*(s===t),n=100*!!r.startsWith(t),o=100*!!s.startsWith(t),c=50*!!t.split(/\s+/).every(e=>r.includes(e)),d=50*!!t.split(/\s+/).every(e=>s.includes(e)),m=r.length;return l+o+d+(1e3-s.length)-(i+n+c+(1e3-m))}))}return e.filter(e=>{let t=0===ei.length||ei.some(t=>e.providers.includes(t)),a=0===en.length||en.includes(e.mode||""),r=0===ec.length||Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).some(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");return ec.includes(t)});return t&&a&&r})},[O,Z,ei,en,ec]),e$=(0,s.useMemo)(()=>{if(!P||!Array.isArray(P))return[];let e=P;if(et.trim()){let t=et.toLowerCase(),a=t.split(/\s+/);e=(e=P.filter(e=>{let r=e.name.toLowerCase(),s=e.description.toLowerCase();return!!(r.includes(t)||s.includes(t))||a.every(e=>r.includes(e)||s.includes(e))})).sort((e,a)=>{let r=e.name.toLowerCase(),s=a.name.toLowerCase(),i=1e3*(r===t),l=1e3*(s===t),n=100*!!r.startsWith(t),o=100*!!s.startsWith(t),c=i+n+(1e3-r.length);return l+o+(1e3-s.length)-c})}return e.filter(e=>0===em.length||e.skills?.some(e=>e.tags?.some(e=>em.includes(e))))},[P,et,em]),eP=(0,s.useMemo)(()=>{if(!z||!Array.isArray(z))return[];let e=z;if(er.trim()){let t=er.toLowerCase(),a=t.split(/\s+/);e=(e=z.filter(e=>{let r=e.server_name.toLowerCase(),s=(e.mcp_info?.description||"").toLowerCase();return!!(r.includes(t)||s.includes(t))||a.every(e=>r.includes(e)||s.includes(e))})).sort((e,a)=>{let r=e.server_name.toLowerCase(),s=a.server_name.toLowerCase(),i=1e3*(r===t),l=1e3*(s===t),n=100*!!r.startsWith(t),o=100*!!s.startsWith(t),c=i+n+(1e3-r.length);return l+o+(1e3-s.length)-c})}return e.filter(e=>0===eu.length||eu.includes(e.transport))},[z,er,eu]),eL=e=>{navigator.clipboard.writeText(e),_.default.success("Copied to clipboard!")},ez=e=>e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),eD=e=>`$${(1e6*e).toFixed(4)}`,eR=e=>e?e>=1e3?`${(e/1e3).toFixed(0)}K`:e.toString():"N/A";return(0,t.jsx)(a.ThemeProvider,{accessToken:e,children:(0,t.jsxs)("div",{className:S?"w-full":"min-h-screen bg-white",children:[!S&&(0,t.jsx)(b.default,{userID:null,userEmail:null,userRole:null,premiumUser:!1,setProxySettings:eI,proxySettings:ek,accessToken:e||null,isPublicPage:!0,isDarkMode:!1,toggleDarkMode:()=>{}}),(0,t.jsxs)("div",{className:S?"w-full p-6":"w-full px-8 py-12",children:[S&&(0,t.jsx)("div",{className:"mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg",children:(0,t.jsx)("p",{className:"text-sm text-gray-700",children:"These are models, agents, and MCP servers your proxy admin has indicated are available in your company."})}),!S&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"About"}),(0,t.jsx)("p",{className:"text-gray-700 mb-6 text-base leading-relaxed",children:F||"Proxy Server to call 100+ LLMs in the OpenAI format."}),(0,t.jsx)("div",{className:"flex items-center space-x-3 text-sm text-gray-600",children:(0,t.jsxs)("span",{className:"flex items-center",children:[(0,t.jsx)("span",{className:"w-4 h-4 mr-2",children:"🔧"}),"Built with litellm: v",B]})})]}),K&&Object.keys(K).length>0&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Useful Links"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",children:Object.entries(K||{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:a})=>(0,t.jsxs)("button",{onClick:()=>window.open(a,"_blank"),className:"flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200",children:[(0,t.jsx)(r.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)(o.Text,{className:"text-sm font-medium",children:e})]},e))})]}),!S&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Health and Endpoint Status"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:(0,t.jsxs)(o.Text,{className:"text-green-600 font-medium text-sm",children:["Service status: ",ex]})})]}),(0,t.jsx)(n.Card,{className:"p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:(0,t.jsxs)(p.Tabs,{activeKey:eM,onChange:eE,size:"large",className:"public-hub-tabs",children:[(0,t.jsxs)(A,{tab:"Model Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Models"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search Models:"}),(0,t.jsx)(g.Tooltip,{title:"Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search model names... (smart search enabled)",value:Z,onChange:e=>ee(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Provider:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:ei,onChange:e=>el(e),placeholder:"Select providers",className:"w-full",size:"large",allowClear:!0,optionRender:e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e.value);return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[a&&(0,t.jsx)("img",{src:a,alt:e.label,className:"w-5 h-5 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e.label})]})},children:O&&Array.isArray(O)&&(w=new Set,O.forEach(e=>{e.providers.forEach(e=>w.add(e))}),Array.from(w)).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Mode:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:en,onChange:e=>eo(e),placeholder:"Select modes",className:"w-full",size:"large",allowClear:!0,children:O&&Array.isArray(O)&&(T=new Set,O.forEach(e=>{e.mode&&T.add(e.mode)}),Array.from(T)).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Features:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:ec,onChange:e=>ed(e),placeholder:"Select features",className:"w-full",size:"large",allowClear:!0,children:O&&Array.isArray(O)&&(C=new Set,O.forEach(e=>{Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).forEach(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");C.add(t)})}),Array.from(C).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Model Name",accessorKey:"model_group",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(g.Tooltip,{title:e.original.model_group,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eA(e.original),e_(!0)},children:e.original.model_group})})}),size:150},{header:"Providers",accessorKey:"providers",enableSorting:!0,cell:({row:e})=>{let a=e.original.providers;return(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:a.map(e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e);return(0,t.jsxs)("div",{className:"flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs",children:[a&&(0,t.jsx)("img",{src:a,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]},e)})})},size:120},{header:"Mode",accessorKey:"mode",enableSorting:!0,cell:({row:e})=>{let a=e.original.mode;return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:(e=>{switch(e?.toLowerCase()){case"chat":return"💬";case"rerank":return"🔄";case"embedding":return"📄";default:return"🤖"}})(a||"")}),(0,t.jsx)(o.Text,{children:a||"Chat"})]})},size:100},{header:"Max Input",accessorKey:"max_input_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-center",children:eR(e.original.max_input_tokens)}),size:100,meta:{className:"text-center"}},{header:"Max Output",accessorKey:"max_output_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-center",children:eR(e.original.max_output_tokens)}),size:100,meta:{className:"text-center"}},{header:"Input $/1M",accessorKey:"input_cost_per_token",enableSorting:!0,cell:({row:e})=>{let a=e.original.input_cost_per_token;return(0,t.jsx)(o.Text,{className:"text-center",children:a?eD(a):"Free"})},size:100,meta:{className:"text-center"}},{header:"Output $/1M",accessorKey:"output_cost_per_token",enableSorting:!0,cell:({row:e})=>{let a=e.original.output_cost_per_token;return(0,t.jsx)(o.Text,{className:"text-center",children:a?eD(a):"Free"})},size:100,meta:{className:"text-center"}},{header:"Features",accessorKey:"supports_vision",enableSorting:!1,cell:({row:e})=>{let a=Object.entries(e.original).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>ez(e));return 0===a.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):1===a.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:a[0]})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:a[0]}),(0,t.jsx)(g.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Features:"}),a.map((e,a)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e]},a))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",a.length-1]})})]})},size:120},{header:"Health Status",accessorKey:"health_status",enableSorting:!0,cell:({row:e})=>{let a=e.original,r="healthy"===a.health_status?"green":"unhealthy"===a.health_status?"red":"default",s=a.health_response_time?`Response Time: ${Number(a.health_response_time).toFixed(2)}ms`:"N/A",i=a.health_checked_at?`Last Checked: ${new Date(a.health_checked_at).toLocaleString()}`:"N/A";return(0,t.jsx)(g.Tooltip,{title:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{children:s}),(0,t.jsx)("div",{children:i})]}),children:(0,t.jsx)(u.Tag,{color:r,children:(0,t.jsx)("span",{className:"capitalize",children:a.health_status??"Unknown"})},a.model_group)})},size:100},{header:"Limits",accessorKey:"rpm",enableSorting:!0,cell:({row:e})=>{var a,r;let s,i=e.original;return(0,t.jsx)(o.Text,{className:"text-xs text-gray-600",children:(a=i.rpm,r=i.tpm,s=[],a&&s.push(`RPM: ${a.toLocaleString()}`),r&&s.push(`TPM: ${r.toLocaleString()}`),s.length>0?s.join(", "):"N/A")})},size:150}],data:eO,isLoading:W,defaultSorting:[{id:"model_group",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",eO.length," of ",O?.length||0," models"]})})]},"models"),P&&Array.isArray(P)&&P.length>0&&(0,t.jsxs)(A,{tab:"Agent Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Agents"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search Agents:"}),(0,t.jsx)(g.Tooltip,{title:"Search agents by name or description",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search agent names or descriptions...",value:et,onChange:e=>ea(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Skills:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:em,onChange:e=>ep(e),placeholder:"Select skills",className:"w-full",size:"large",allowClear:!0,children:P&&Array.isArray(P)&&(k=new Set,P.forEach(e=>{e.skills?.forEach(e=>{e.tags?.forEach(e=>k.add(e))})}),Array.from(k).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Agent Name",accessorKey:"name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(g.Tooltip,{title:e.original.name,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{ew(e.original),ev(!0)},children:e.original.name})})}),size:150},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>{let a=e.original.description,r=a.length>80?a.substring(0,80)+"...":a;return(0,t.jsx)(g.Tooltip,{title:a,children:(0,t.jsx)(o.Text,{className:"text-sm text-gray-700",children:r})})},size:250},{header:"Version",accessorKey:"version",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-sm",children:e.original.version}),size:80},{header:"Provider",accessorKey:"provider",enableSorting:!1,cell:({row:e})=>{let a=e.original.provider;return a?(0,t.jsx)("div",{className:"text-sm",children:(0,t.jsx)(o.Text,{className:"font-medium",children:a.organization})}):(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"})},size:120},{header:"Skills",accessorKey:"skills",enableSorting:!1,cell:({row:e})=>{let a=e.original.skills||[];return 0===a.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):1===a.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:a[0].name})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:a[0].name}),(0,t.jsx)(g.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Skills:"}),a.map((e,a)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e.name]},a))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-purple-600 cursor-pointer hover:text-purple-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",a.length-1]})})]})},size:150},{header:"Capabilities",accessorKey:"capabilities",enableSorting:!1,cell:({row:e})=>{let a=Object.entries(e.original.capabilities||{}).filter(([e,t])=>!0===t).map(([e])=>e);return 0===a.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:a.map(e=>(0,t.jsx)(u.Tag,{color:"green",className:"text-xs capitalize",children:e},e))})},size:150}],data:e$,isLoading:Y,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",e$.length," of ",P?.length||0," agents"]})})]},"agents"),z&&Array.isArray(z)&&z.length>0&&(0,t.jsxs)(A,{tab:"MCP Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available MCP Servers"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search MCP Servers:"}),(0,t.jsx)(g.Tooltip,{title:"Search MCP servers by name or description",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search MCP server names or descriptions...",value:er,onChange:e=>es(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Transport:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:eu,onChange:e=>eg(e),placeholder:"Select transport types",className:"w-full",size:"large",allowClear:!0,children:z&&Array.isArray(z)&&(I=new Set,z.forEach(e=>{e.transport&&I.add(e.transport)}),Array.from(I).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Server Name",accessorKey:"server_name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(g.Tooltip,{title:e.original.server_name,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eC(e.original),ej(!0)},children:e.original.server_name})})}),size:150},{header:"Description",accessorKey:"mcp_info.description",enableSorting:!1,cell:({row:e})=>{let a=e.original.mcp_info?.description||"-",r=a.length>80?a.substring(0,80)+"...":a;return(0,t.jsx)(g.Tooltip,{title:a,children:(0,t.jsx)(o.Text,{className:"text-sm text-gray-700",children:r})})},size:250},{header:"URL",accessorKey:"url",enableSorting:!1,cell:({row:e})=>{let a=e.original.url,r=a.length>40?a.substring(0,40)+"...":a;return(0,t.jsx)(g.Tooltip,{title:a,children:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)(o.Text,{className:"text-xs font-mono",children:r}),(0,t.jsx)(x.default,{onClick:()=>eL(a),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-3 h-3"})]})})},size:200},{header:"Transport",accessorKey:"transport",enableSorting:!0,cell:({row:e})=>{let a=e.original.transport;return(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs uppercase",children:a})},size:100},{header:"Auth Type",accessorKey:"auth_type",enableSorting:!0,cell:({row:e})=>{let a=e.original.auth_type;return(0,t.jsx)(u.Tag,{color:"none"===a?"gray":"green",className:"text-xs capitalize",children:a})},size:100}],data:eP,isLoading:X,defaultSorting:[{id:"server_name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",eP.length," of ",z?.length||0," MCP servers"]})})]},"mcp")]})})]}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eN?.model_group||"Model Details"}),eN&&(0,t.jsx)(g.Tooltip,{title:"Copy model name",children:(0,t.jsx)(x.default,{onClick:()=>eL(eN.model_group),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ef,footer:null,onOk:()=>{e_(!1),eA(null)},onCancel:()=>{e_(!1),eA(null)},children:eN&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Model Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Model Name:"}),(0,t.jsx)(o.Text,{children:eN.model_group})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Mode:"}),(0,t.jsx)(o.Text,{children:eN.mode||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Providers:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:eN.providers.map(e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e);return(0,t.jsx)(u.Tag,{color:"blue",children:(0,t.jsxs)("div",{className:"flex items-center space-x-1",children:[a&&(0,t.jsx)("img",{src:a,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]})},e)})})]})]}),eN.model_group.includes("*")&&(0,t.jsx)("div",{className:"bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4",children:(0,t.jsxs)("div",{className:"flex items-start space-x-2",children:[(0,t.jsx)(h,{className:"w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium text-blue-900 mb-2",children:"Wildcard Routing"}),(0,t.jsxs)(o.Text,{className:"text-sm text-blue-800 mb-2",children:["This model uses wildcard routing. You can pass any value where you see the"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:"*"})," symbol."]}),(0,t.jsxs)(o.Text,{className:"text-sm text-blue-800",children:["For example, with"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group}),", you can use any string (",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group.replace("*","my-custom-value")}),") that matches this pattern."]})]})]})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Token & Cost Information"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Max Input Tokens:"}),(0,t.jsx)(o.Text,{children:eN.max_input_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Max Output Tokens:"}),(0,t.jsx)(o.Text,{children:eN.max_output_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Input Cost per 1M Tokens:"}),(0,t.jsx)(o.Text,{children:eN.input_cost_per_token?eD(eN.input_cost_per_token):"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Output Cost per 1M Tokens:"}),(0,t.jsx)(o.Text,{children:eN.output_cost_per_token?eD(eN.output_cost_per_token):"Not specified"})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:(M=Object.entries(eN).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e),E=["green","blue","purple","orange","red","yellow"],0===M.length?(0,t.jsx)(o.Text,{className:"text-gray-500",children:"No special capabilities listed"}):M.map((e,a)=>(0,t.jsx)(u.Tag,{color:E[a%E.length],children:ez(e)},e)))})]}),(eN.tpm||eN.rpm)&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Rate Limits"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[eN.tpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Tokens per Minute:"}),(0,t.jsx)(o.Text,{children:eN.tpm.toLocaleString()})]}),eN.rpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Requests per Minute:"}),(0,t.jsx)(o.Text,{children:eN.rpm.toLocaleString()})]})]})]}),eN.supported_openai_params&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Supported OpenAI Parameters"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:eN.supported_openai_params.map(e=>(0,t.jsx)(u.Tag,{color:"green",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:(0,y.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,j.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"})})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eL((0,y.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,j.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"}))},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eS?.name||"Agent Details"}),eS&&(0,t.jsx)(g.Tooltip,{title:"Copy agent name",children:(0,t.jsx)(x.default,{onClick:()=>eL(eS.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:eb,footer:null,onOk:()=>{ev(!1),ew(null)},onCancel:()=>{ev(!1),ew(null)},children:eS&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Agent Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Name:"}),(0,t.jsx)(o.Text,{children:eS.name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Version:"}),(0,t.jsx)(o.Text,{children:eS.version})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(o.Text,{children:eS.description})]}),eS.url&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"URL:"}),(0,t.jsx)("a",{href:eS.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all",children:eS.url})]})]})]}),eS.capabilities&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:Object.entries(eS.capabilities).filter(([e,t])=>!0===t).map(([e])=>(0,t.jsx)(u.Tag,{color:"green",className:"capitalize",children:e},e))})]}),eS.skills&&eS.skills.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Skills"}),(0,t.jsx)("div",{className:"space-y-4",children:eS.skills.map((e,a)=>(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"flex items-start justify-between mb-2",children:(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium text-base",children:e.name}),(0,t.jsx)(o.Text,{className:"text-sm text-gray-600",children:e.description})]})}),e.tags&&e.tags.length>0&&(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-2",children:e.tags.map(e=>(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:e},e))})]},a))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Input/Output Modes"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Input Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:eS.defaultInputModes?.map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Output Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:eS.defaultOutputModes?.map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]})]})]}),eS.documentationUrl&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Documentation"}),(0,t.jsxs)("a",{href:eS.documentationUrl,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 flex items-center space-x-2",children:[(0,t.jsx)(r.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)("span",{children:"View Documentation"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example (A2A Protocol)"}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 1: Retrieve Agent Card"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`base_url = '${eS.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eL(`from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

base_url = '${eS.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 2: Call the Agent"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eL(`client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})]})}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eT?.server_name||"MCP Server Details"}),eT&&(0,t.jsx)(g.Tooltip,{title:"Copy server name",children:(0,t.jsx)(x.default,{onClick:()=>eL(eT.server_name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ey,footer:null,onOk:()=>{ej(!1),eC(null)},onCancel:()=>{ej(!1),eC(null)},children:eT&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Server Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Server Name:"}),(0,t.jsx)(o.Text,{children:eT.server_name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Transport:"}),(0,t.jsx)(u.Tag,{color:"blue",children:eT.transport})]}),eT.alias&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Alias:"}),(0,t.jsx)(o.Text,{children:eT.alias})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Auth Type:"}),(0,t.jsx)(u.Tag,{color:"none"===eT.auth_type?"gray":"green",children:eT.auth_type})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(o.Text,{children:eT.mcp_info?.description||"-"})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"URL:"}),(0,t.jsxs)("a",{href:eT.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eT.url}),(0,t.jsx)(r.ExternalLinkIcon,{className:"w-4 h-4"})]})]})]})]}),eT.mcp_info&&Object.keys(eT.mcp_info).length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Additional Information"}),(0,t.jsx)("div",{className:"bg-gray-50 p-4 rounded-lg",children:(0,t.jsx)("pre",{className:"text-xs overflow-x-auto",children:JSON.stringify(eT.mcp_info,null,2)})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eT.server_name}": {
            "url": "http://localhost:4000/${eT.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call a tool
        response = await client.call_tool(
            name="tool_name", 
            arguments={"arg": "value"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eL(`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eT.server_name}": {
            "url": "http://localhost:4000/${eT.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call a tool
        response = await client.call_tool(
            name="tool_name", 
            arguments={"arg": "value"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})})]})})}],976883)}]);