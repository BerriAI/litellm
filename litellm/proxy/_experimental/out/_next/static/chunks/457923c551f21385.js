(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,771674,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var r=e.i(9583),i=s.forwardRef(function(e,i){return s.createElement(r.default,(0,t.default)({},e,{ref:i,icon:a}))});e.s(["UserOutlined",0,i],771674)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),i=s.forwardRef(function(e,i){return s.createElement(r.default,(0,t.default)({},e,{ref:i,icon:a}))});e.s(["SafetyOutlined",0,i],602073)},818581,(e,t,s)=>{"use strict";Object.defineProperty(s,"__esModule",{value:!0}),Object.defineProperty(s,"useMergedRef",{enumerable:!0,get:function(){return r}});let a=e.r(271645);function r(e,t){let s=(0,a.useRef)(null),r=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=s.current;e&&(s.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(s.current=i(e,a)),t&&(r.current=i(t,a))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let s=e(t);return"function"==typeof s?s:()=>e(null)}}("function"==typeof s.default||"object"==typeof s.default&&null!==s.default)&&void 0===s.default.__esModule&&(Object.defineProperty(s.default,"__esModule",{value:!0}),Object.assign(s.default,s),t.exports=s.default)},62478,e=>{"use strict";var t=e.i(764205);let s=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,s])},190272,785913,e=>{"use strict";var t,s,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((s={}).IMAGE="image",s.VIDEO="video",s.CHAT="chat",s.RESPONSES="responses",s.IMAGE_EDITS="image_edits",s.ANTHROPIC_MESSAGES="anthropic_messages",s.EMBEDDINGS="embeddings",s.SPEECH="speech",s.TRANSCRIPTION="transcription",s.A2A_AGENTS="a2a_agents",s.MCP="mcp",s);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:s,accessToken:a,apiKey:i,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:m,selectedMCPServers:p,mcpServers:u,mcpServerToolRestrictions:x,selectedVoice:g,endpointType:h,selectedModel:_,selectedSdk:f,proxySettings:b}=e,j="session"===s?a:i,y=window.location.origin,v=b?.LITELLM_UI_API_DOC_BASE_URL;v&&v.trim()?y=v:b?.PROXY_BASE_URL&&(y=b.PROXY_BASE_URL);let N=l||"Your prompt here",T=N.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};o.length>0&&(S.tags=o),c.length>0&&(S.vector_stores=c),d.length>0&&(S.guardrails=d),m.length>0&&(S.policies=m);let C=_||"your-model-name",A="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${j||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${j||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case r.CHAT:{let e=Object.keys(S).length>0,s="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=w.length>0?w:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(a,null,4)}${s}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${C}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${T}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${s}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(S).length>0,s="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=w.length>0?w:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(a,null,4)}${s}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${C}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${T}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${s}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===f?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${C}",
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
prompt = "${T}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case r.IMAGE_EDITS:t="azure"===f?`
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
prompt = "${T}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
prompt = "${T}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case r.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${l||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${l?`,
	prompt="${l.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${l||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${A}
${t}`}],190272)},976883,174886,879664,e=>{"use strict";var t=e.i(843476),s=e.i(275144),a=e.i(434626),r=e.i(271645);let i=r.forwardRef(function(e,t){return r.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),r.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"}))});var l=e.i(994388),n=e.i(304967),o=e.i(599724),c=e.i(629569),d=e.i(212931),m=e.i(199133),p=e.i(653496),u=e.i(262218),x=e.i(592968),g=e.i(991124);e.s(["Copy",()=>g.default],174886);var g=g;let h=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>h],879664);var _=e.i(798496),f=e.i(727749),b=e.i(402874),j=e.i(764205),y=e.i(190272),v=e.i(785913),N=e.i(916925);let{TabPane:T}=p.Tabs;e.s(["default",0,({accessToken:e,isEmbedded:w=!1})=>{let S,C,A,k,M,E,I,[P,O]=(0,r.useState)(null),[L,R]=(0,r.useState)(null),[z,D]=(0,r.useState)(null),[$,H]=(0,r.useState)("LiteLLM Gateway"),[U,K]=(0,r.useState)(null),[F,G]=(0,r.useState)(""),[B,q]=(0,r.useState)({}),[W,Y]=(0,r.useState)(!0),[V,J]=(0,r.useState)(!0),[X,Q]=(0,r.useState)(!0),[Z,ee]=(0,r.useState)(""),[et,es]=(0,r.useState)(""),[ea,er]=(0,r.useState)(""),[ei,el]=(0,r.useState)([]),[en,eo]=(0,r.useState)([]),[ec,ed]=(0,r.useState)([]),[em,ep]=(0,r.useState)([]),[eu,ex]=(0,r.useState)([]),[eg,eh]=(0,r.useState)("I'm alive! ✓"),[e_,ef]=(0,r.useState)(!1),[eb,ej]=(0,r.useState)(!1),[ey,ev]=(0,r.useState)(!1),[eN,eT]=(0,r.useState)(null),[ew,eS]=(0,r.useState)(null),[eC,eA]=(0,r.useState)(null),[ek,eM]=(0,r.useState)({}),[eE,eI]=(0,r.useState)("models");(0,r.useEffect)(()=>{(async()=>{try{await (0,j.getUiConfig)()}catch(e){console.error("Failed to get UI config:",e)}let e=async()=>{try{Y(!0);let e=await (0,j.modelHubPublicModelsCall)();console.log("ModelHubData:",e),O(e)}catch(e){console.error("There was an error fetching the public model data",e),eh("Service unavailable")}finally{Y(!1)}},t=async()=>{try{J(!0);let e=await (0,j.agentHubPublicModelsCall)();console.log("AgentHubData:",e),R(e)}catch(e){console.error("There was an error fetching the public agent data",e)}finally{J(!1)}},s=async()=>{try{Q(!0);let e=await (0,j.mcpHubPublicServersCall)();console.log("MCPHubData:",e),D(e)}catch(e){console.error("There was an error fetching the public MCP server data",e)}finally{Q(!1)}};(async()=>{let e=await (0,j.getPublicModelHubInfo)();console.log("Public Model Hub Info:",e),H(e.docs_title),K(e.custom_docs_description),G(e.litellm_version),q(e.useful_links||{})})(),e(),t(),s()})()},[]),(0,r.useEffect)(()=>{},[Z,ei,en,ec]);let eP=(0,r.useMemo)(()=>{if(!P||!Array.isArray(P))return[];let e=P;if(Z.trim()){let t=Z.toLowerCase(),s=t.split(/\s+/),a=P.filter(e=>{let a=e.model_group.toLowerCase();return!!a.includes(t)||s.every(e=>a.includes(e))});a.length>0&&(e=a.sort((e,s)=>{let a=e.model_group.toLowerCase(),r=s.model_group.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=50*!!t.split(/\s+/).every(e=>a.includes(e)),d=50*!!t.split(/\s+/).every(e=>r.includes(e)),m=a.length;return l+o+d+(1e3-r.length)-(i+n+c+(1e3-m))}))}return e.filter(e=>{let t=0===ei.length||ei.some(t=>e.providers.includes(t)),s=0===en.length||en.includes(e.mode||""),a=0===ec.length||Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).some(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");return ec.includes(t)});return t&&s&&a})},[P,Z,ei,en,ec]),eO=(0,r.useMemo)(()=>{if(!L||!Array.isArray(L))return[];let e=L;if(et.trim()){let t=et.toLowerCase(),s=t.split(/\s+/);e=(e=L.filter(e=>{let a=e.name.toLowerCase(),r=e.description.toLowerCase();return!!(a.includes(t)||r.includes(t))||s.every(e=>a.includes(e)||r.includes(e))})).sort((e,s)=>{let a=e.name.toLowerCase(),r=s.name.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===em.length||e.skills?.some(e=>e.tags?.some(e=>em.includes(e))))},[L,et,em]),eL=(0,r.useMemo)(()=>{if(!z||!Array.isArray(z))return[];let e=z;if(ea.trim()){let t=ea.toLowerCase(),s=t.split(/\s+/);e=(e=z.filter(e=>{let a=e.server_name.toLowerCase(),r=(e.mcp_info?.description||"").toLowerCase();return!!(a.includes(t)||r.includes(t))||s.every(e=>a.includes(e)||r.includes(e))})).sort((e,s)=>{let a=e.server_name.toLowerCase(),r=s.server_name.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===eu.length||eu.includes(e.transport))},[z,ea,eu]),eR=e=>{navigator.clipboard.writeText(e),f.default.success("Copied to clipboard!")},ez=e=>e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),eD=e=>`$${(1e6*e).toFixed(4)}`,e$=e=>e?e>=1e3?`${(e/1e3).toFixed(0)}K`:e.toString():"N/A";return(0,t.jsx)(s.ThemeProvider,{accessToken:e,children:(0,t.jsxs)("div",{className:w?"w-full":"min-h-screen bg-white",children:[!w&&(0,t.jsx)(b.default,{userID:null,userEmail:null,userRole:null,premiumUser:!1,setProxySettings:eM,proxySettings:ek,accessToken:e||null,isPublicPage:!0,isDarkMode:!1,toggleDarkMode:()=>{}}),(0,t.jsxs)("div",{className:w?"w-full p-6":"w-full px-8 py-12",children:[w&&(0,t.jsx)("div",{className:"mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg",children:(0,t.jsx)("p",{className:"text-sm text-gray-700",children:"These are models, agents, and MCP servers your proxy admin has indicated are available in your company."})}),!w&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"About"}),(0,t.jsx)("p",{className:"text-gray-700 mb-6 text-base leading-relaxed",children:U||"Proxy Server to call 100+ LLMs in the OpenAI format."}),(0,t.jsx)("div",{className:"flex items-center space-x-3 text-sm text-gray-600",children:(0,t.jsxs)("span",{className:"flex items-center",children:[(0,t.jsx)("span",{className:"w-4 h-4 mr-2",children:"🔧"}),"Built with litellm: v",F]})})]}),B&&Object.keys(B).length>0&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Useful Links"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",children:Object.entries(B||{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:s})=>(0,t.jsxs)("button",{onClick:()=>window.open(s,"_blank"),className:"flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200",children:[(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)(o.Text,{className:"text-sm font-medium",children:e})]},e))})]}),!w&&(0,t.jsxs)(n.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(c.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Health and Endpoint Status"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:(0,t.jsxs)(o.Text,{className:"text-green-600 font-medium text-sm",children:["Service status: ",eg]})})]}),(0,t.jsx)(n.Card,{className:"p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:(0,t.jsxs)(p.Tabs,{activeKey:eE,onChange:eI,size:"large",className:"public-hub-tabs",children:[(0,t.jsxs)(T,{tab:"Model Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Models"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search Models:"}),(0,t.jsx)(x.Tooltip,{title:"Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search model names... (smart search enabled)",value:Z,onChange:e=>ee(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Provider:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:ei,onChange:e=>el(e),placeholder:"Select providers",className:"w-full",size:"large",allowClear:!0,optionRender:e=>{let{logo:s}=(0,N.getProviderLogoAndName)(e.value);return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[s&&(0,t.jsx)("img",{src:s,alt:e.label,className:"w-5 h-5 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e.label})]})},children:P&&Array.isArray(P)&&(S=new Set,P.forEach(e=>{e.providers.forEach(e=>S.add(e))}),Array.from(S)).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Mode:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:en,onChange:e=>eo(e),placeholder:"Select modes",className:"w-full",size:"large",allowClear:!0,children:P&&Array.isArray(P)&&(C=new Set,P.forEach(e=>{e.mode&&C.add(e.mode)}),Array.from(C)).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Features:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:ec,onChange:e=>ed(e),placeholder:"Select features",className:"w-full",size:"large",allowClear:!0,children:P&&Array.isArray(P)&&(A=new Set,P.forEach(e=>{Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).forEach(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");A.add(t)})}),Array.from(A).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(_.ModelDataTable,{columns:[{header:"Model Name",accessorKey:"model_group",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(x.Tooltip,{title:e.original.model_group,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eT(e.original),ef(!0)},children:e.original.model_group})})}),size:150},{header:"Providers",accessorKey:"providers",enableSorting:!0,cell:({row:e})=>{let s=e.original.providers;return(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:s.map(e=>{let{logo:s}=(0,N.getProviderLogoAndName)(e);return(0,t.jsxs)("div",{className:"flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs",children:[s&&(0,t.jsx)("img",{src:s,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]},e)})})},size:120},{header:"Mode",accessorKey:"mode",enableSorting:!0,cell:({row:e})=>{let s=e.original.mode;return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:(e=>{switch(e?.toLowerCase()){case"chat":return"💬";case"rerank":return"🔄";case"embedding":return"📄";default:return"🤖"}})(s||"")}),(0,t.jsx)(o.Text,{children:s||"Chat"})]})},size:100},{header:"Max Input",accessorKey:"max_input_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-center",children:e$(e.original.max_input_tokens)}),size:100,meta:{className:"text-center"}},{header:"Max Output",accessorKey:"max_output_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-center",children:e$(e.original.max_output_tokens)}),size:100,meta:{className:"text-center"}},{header:"Input $/1M",accessorKey:"input_cost_per_token",enableSorting:!0,cell:({row:e})=>{let s=e.original.input_cost_per_token;return(0,t.jsx)(o.Text,{className:"text-center",children:s?eD(s):"Free"})},size:100,meta:{className:"text-center"}},{header:"Output $/1M",accessorKey:"output_cost_per_token",enableSorting:!0,cell:({row:e})=>{let s=e.original.output_cost_per_token;return(0,t.jsx)(o.Text,{className:"text-center",children:s?eD(s):"Free"})},size:100,meta:{className:"text-center"}},{header:"Features",accessorKey:"supports_vision",enableSorting:!1,cell:({row:e})=>{let s=Object.entries(e.original).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>ez(e));return 0===s.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):1===s.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:s[0]})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:s[0]}),(0,t.jsx)(x.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Features:"}),s.map((e,s)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e]},s))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",s.length-1]})})]})},size:120},{header:"Health Status",accessorKey:"health_status",enableSorting:!0,cell:({row:e})=>{let s=e.original,a="healthy"===s.health_status?"green":"unhealthy"===s.health_status?"red":"default",r=s.health_response_time?`Response Time: ${Number(s.health_response_time).toFixed(2)}ms`:"N/A",i=s.health_checked_at?`Last Checked: ${new Date(s.health_checked_at).toLocaleString()}`:"N/A";return(0,t.jsx)(x.Tooltip,{title:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{children:r}),(0,t.jsx)("div",{children:i})]}),children:(0,t.jsx)(u.Tag,{color:a,children:(0,t.jsx)("span",{className:"capitalize",children:s.health_status??"Unknown"})},s.model_group)})},size:100},{header:"Limits",accessorKey:"rpm",enableSorting:!0,cell:({row:e})=>{var s,a;let r,i=e.original;return(0,t.jsx)(o.Text,{className:"text-xs text-gray-600",children:(s=i.rpm,a=i.tpm,r=[],s&&r.push(`RPM: ${s.toLocaleString()}`),a&&r.push(`TPM: ${a.toLocaleString()}`),r.length>0?r.join(", "):"N/A")})},size:150}],data:eP,isLoading:W,defaultSorting:[{id:"model_group",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",eP.length," of ",P?.length||0," models"]})})]},"models"),L&&Array.isArray(L)&&L.length>0&&(0,t.jsxs)(T,{tab:"Agent Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Agents"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search Agents:"}),(0,t.jsx)(x.Tooltip,{title:"Search agents by name or description",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search agent names or descriptions...",value:et,onChange:e=>es(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Skills:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:em,onChange:e=>ep(e),placeholder:"Select skills",className:"w-full",size:"large",allowClear:!0,children:L&&Array.isArray(L)&&(k=new Set,L.forEach(e=>{e.skills?.forEach(e=>{e.tags?.forEach(e=>k.add(e))})}),Array.from(k).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(_.ModelDataTable,{columns:[{header:"Agent Name",accessorKey:"name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(x.Tooltip,{title:e.original.name,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eS(e.original),ej(!0)},children:e.original.name})})}),size:150},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>{let s=e.original.description,a=s.length>80?s.substring(0,80)+"...":s;return(0,t.jsx)(x.Tooltip,{title:s,children:(0,t.jsx)(o.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"Version",accessorKey:"version",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Text,{className:"text-sm",children:e.original.version}),size:80},{header:"Provider",accessorKey:"provider",enableSorting:!1,cell:({row:e})=>{let s=e.original.provider;return s?(0,t.jsx)("div",{className:"text-sm",children:(0,t.jsx)(o.Text,{className:"font-medium",children:s.organization})}):(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"})},size:120},{header:"Skills",accessorKey:"skills",enableSorting:!1,cell:({row:e})=>{let s=e.original.skills||[];return 0===s.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):1===s.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:s[0].name})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:s[0].name}),(0,t.jsx)(x.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Skills:"}),s.map((e,s)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e.name]},s))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-purple-600 cursor-pointer hover:text-purple-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",s.length-1]})})]})},size:150},{header:"Capabilities",accessorKey:"capabilities",enableSorting:!1,cell:({row:e})=>{let s=Object.entries(e.original.capabilities||{}).filter(([e,t])=>!0===t).map(([e])=>e);return 0===s.length?(0,t.jsx)(o.Text,{className:"text-gray-400",children:"-"}):(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:s.map(e=>(0,t.jsx)(u.Tag,{color:"green",className:"text-xs capitalize",children:e},e))})},size:150}],data:eO,isLoading:V,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",eO.length," of ",L?.length||0," agents"]})})]},"agents"),z&&Array.isArray(z)&&z.length>0&&(0,t.jsxs)(T,{tab:"MCP Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(c.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available MCP Servers"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium text-gray-700",children:"Search MCP Servers:"}),(0,t.jsx)(x.Tooltip,{title:"Search MCP servers by name or description",placement:"top",children:(0,t.jsx)(h,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search MCP server names or descriptions...",value:ea,onChange:e=>er(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Transport:"}),(0,t.jsx)(m.Select,{mode:"multiple",value:eu,onChange:e=>ex(e),placeholder:"Select transport types",className:"w-full",size:"large",allowClear:!0,children:z&&Array.isArray(z)&&(M=new Set,z.forEach(e=>{e.transport&&M.add(e.transport)}),Array.from(M).sort()).map(e=>(0,t.jsx)(m.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(_.ModelDataTable,{columns:[{header:"Server Name",accessorKey:"server_name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(x.Tooltip,{title:e.original.server_name,children:(0,t.jsx)(l.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eA(e.original),ev(!0)},children:e.original.server_name})})}),size:150},{header:"Description",accessorKey:"mcp_info.description",enableSorting:!1,cell:({row:e})=>{let s=e.original.mcp_info?.description||"-",a=s.length>80?s.substring(0,80)+"...":s;return(0,t.jsx)(x.Tooltip,{title:s,children:(0,t.jsx)(o.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"URL",accessorKey:"url",enableSorting:!1,cell:({row:e})=>{let s=e.original.url,a=s.length>40?s.substring(0,40)+"...":s;return(0,t.jsx)(x.Tooltip,{title:s,children:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)(o.Text,{className:"text-xs font-mono",children:a}),(0,t.jsx)(g.default,{onClick:()=>eR(s),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-3 h-3"})]})})},size:200},{header:"Transport",accessorKey:"transport",enableSorting:!0,cell:({row:e})=>{let s=e.original.transport;return(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs uppercase",children:s})},size:100},{header:"Auth Type",accessorKey:"auth_type",enableSorting:!0,cell:({row:e})=>{let s=e.original.auth_type;return(0,t.jsx)(u.Tag,{color:"none"===s?"gray":"green",className:"text-xs capitalize",children:s})},size:100}],data:eL,isLoading:X,defaultSorting:[{id:"server_name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(o.Text,{className:"text-sm text-gray-600",children:["Showing ",eL.length," of ",z?.length||0," MCP servers"]})})]},"mcp")]})})]}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eN?.model_group||"Model Details"}),eN&&(0,t.jsx)(x.Tooltip,{title:"Copy model name",children:(0,t.jsx)(g.default,{onClick:()=>eR(eN.model_group),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:e_,footer:null,onOk:()=>{ef(!1),eT(null)},onCancel:()=>{ef(!1),eT(null)},children:eN&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Model Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Model Name:"}),(0,t.jsx)(o.Text,{children:eN.model_group})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Mode:"}),(0,t.jsx)(o.Text,{children:eN.mode||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Providers:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:eN.providers.map(e=>{let{logo:s}=(0,N.getProviderLogoAndName)(e);return(0,t.jsx)(u.Tag,{color:"blue",children:(0,t.jsxs)("div",{className:"flex items-center space-x-1",children:[s&&(0,t.jsx)("img",{src:s,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]})},e)})})]})]}),eN.model_group.includes("*")&&(0,t.jsx)("div",{className:"bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4",children:(0,t.jsxs)("div",{className:"flex items-start space-x-2",children:[(0,t.jsx)(h,{className:"w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium text-blue-900 mb-2",children:"Wildcard Routing"}),(0,t.jsxs)(o.Text,{className:"text-sm text-blue-800 mb-2",children:["This model uses wildcard routing. You can pass any value where you see the"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:"*"})," symbol."]}),(0,t.jsxs)(o.Text,{className:"text-sm text-blue-800",children:["For example, with"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group}),", you can use any string (",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group.replace("*","my-custom-value")}),") that matches this pattern."]})]})]})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Token & Cost Information"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Max Input Tokens:"}),(0,t.jsx)(o.Text,{children:eN.max_input_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Max Output Tokens:"}),(0,t.jsx)(o.Text,{children:eN.max_output_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Input Cost per 1M Tokens:"}),(0,t.jsx)(o.Text,{children:eN.input_cost_per_token?eD(eN.input_cost_per_token):"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Output Cost per 1M Tokens:"}),(0,t.jsx)(o.Text,{children:eN.output_cost_per_token?eD(eN.output_cost_per_token):"Not specified"})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:(E=Object.entries(eN).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e),I=["green","blue","purple","orange","red","yellow"],0===E.length?(0,t.jsx)(o.Text,{className:"text-gray-500",children:"No special capabilities listed"}):E.map((e,s)=>(0,t.jsx)(u.Tag,{color:I[s%I.length],children:ez(e)},e)))})]}),(eN.tpm||eN.rpm)&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Rate Limits"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[eN.tpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Tokens per Minute:"}),(0,t.jsx)(o.Text,{children:eN.tpm.toLocaleString()})]}),eN.rpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Requests per Minute:"}),(0,t.jsx)(o.Text,{children:eN.rpm.toLocaleString()})]})]})]}),eN.supported_openai_params&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Supported OpenAI Parameters"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:eN.supported_openai_params.map(e=>(0,t.jsx)(u.Tag,{color:"green",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:(0,y.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,v.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"})})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eR((0,y.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,v.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"}))},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ew?.name||"Agent Details"}),ew&&(0,t.jsx)(x.Tooltip,{title:"Copy agent name",children:(0,t.jsx)(g.default,{onClick:()=>eR(ew.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:eb,footer:null,onOk:()=>{ej(!1),eS(null)},onCancel:()=>{ej(!1),eS(null)},children:ew&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Agent Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Name:"}),(0,t.jsx)(o.Text,{children:ew.name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Version:"}),(0,t.jsx)(o.Text,{children:ew.version})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(o.Text,{children:ew.description})]}),ew.url&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"URL:"}),(0,t.jsx)("a",{href:ew.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all",children:ew.url})]})]})]}),ew.capabilities&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:Object.entries(ew.capabilities).filter(([e,t])=>!0===t).map(([e])=>(0,t.jsx)(u.Tag,{color:"green",className:"capitalize",children:e},e))})]}),ew.skills&&ew.skills.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Skills"}),(0,t.jsx)("div",{className:"space-y-4",children:ew.skills.map((e,s)=>(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"flex items-start justify-between mb-2",children:(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium text-base",children:e.name}),(0,t.jsx)(o.Text,{className:"text-sm text-gray-600",children:e.description})]})}),e.tags&&e.tags.length>0&&(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-2",children:e.tags.map(e=>(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:e},e))})]},s))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Input/Output Modes"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Input Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:ew.defaultInputModes?.map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Output Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:ew.defaultOutputModes?.map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]})]})]}),ew.documentationUrl&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Documentation"}),(0,t.jsxs)("a",{href:ew.documentationUrl,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 flex items-center space-x-2",children:[(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)("span",{children:"View Documentation"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example (A2A Protocol)"}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsx)(o.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 1: Retrieve Agent Card"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`base_url = '${ew.url}'

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
        )`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eR(`from a2a.client import A2ACardResolver, A2AClient
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

base_url = '${ew.url}'

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
print(response.model_dump(mode='json', exclude_none=True))`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eR(`client = A2AClient(
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
print(response.model_dump(mode='json', exclude_none=True))`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})]})}),(0,t.jsx)(d.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eC?.server_name||"MCP Server Details"}),eC&&(0,t.jsx)(x.Tooltip,{title:"Copy server name",children:(0,t.jsx)(g.default,{onClick:()=>eR(eC.server_name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ey,footer:null,onOk:()=>{ev(!1),eA(null)},onCancel:()=>{ev(!1),eA(null)},children:eC&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Server Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Server Name:"}),(0,t.jsx)(o.Text,{children:eC.server_name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Transport:"}),(0,t.jsx)(u.Tag,{color:"blue",children:eC.transport})]}),eC.alias&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Alias:"}),(0,t.jsx)(o.Text,{children:eC.alias})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Auth Type:"}),(0,t.jsx)(u.Tag,{color:"none"===eC.auth_type?"gray":"green",children:eC.auth_type})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(o.Text,{children:eC.mcp_info?.description||"-"})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(o.Text,{className:"font-medium",children:"URL:"}),(0,t.jsxs)("a",{href:eC.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eC.url}),(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"})]})]})]})]}),eC.mcp_info&&Object.keys(eC.mcp_info).length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Additional Information"}),(0,t.jsx)("div",{className:"bg-gray-50 p-4 rounded-lg",children:(0,t.jsx)("pre",{className:"text-xs overflow-x-auto",children:JSON.stringify(eC.mcp_info,null,2)})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(o.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eC.server_name}": {
            "url": "http://localhost:4000/${eC.server_name}/mcp",
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
    asyncio.run(main())`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eR(`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eC.server_name}": {
            "url": "http://localhost:4000/${eC.server_name}/mcp",
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