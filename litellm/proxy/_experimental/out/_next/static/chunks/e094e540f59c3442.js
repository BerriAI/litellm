(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},275144,e=>{"use strict";var t=e.i(843476),a=e.i(271645),i=e.i(764205);let n=(0,a.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:r})=>{let[o,s]=(0,a.useState)(null),[l,c]=(0,a.useState)(null);return(0,a.useEffect)(()=>{(async()=>{try{let e=(0,i.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",a=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(a.ok){let e=await a.json();e.values?.logo_url&&s(e.values.logo_url),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,a.useEffect)(()=>{if(l){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=l});else{let e=document.createElement("link");e.rel="icon",e.href=l,document.head.appendChild(e)}}},[l]),(0,t.jsx)(n.Provider,{value:{logoUrl:o,setLogoUrl:s,faviconUrl:l,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,a.useContext)(n);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},115571,e=>{"use strict";let t="local-storage-change";function a(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function i(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function n(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function r(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>a,"getLocalStorageItem",()=>i,"removeLocalStorageItem",()=>r,"setLocalStorageItem",()=>n])},371401,e=>{"use strict";var t=e.i(115571),a=e.i(271645);function i(e){let a=t=>{"disableUsageIndicator"===t.key&&e()},i=t=>{let{key:a}=t.detail;"disableUsageIndicator"===a&&e()};return window.addEventListener("storage",a),window.addEventListener(t.LOCAL_STORAGE_EVENT,i),()=>{window.removeEventListener("storage",a),window.removeEventListener(t.LOCAL_STORAGE_EVENT,i)}}function n(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}function r(){return(0,a.useSyncExternalStore)(i,n)}e.s(["useDisableUsageIndicator",()=>r])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},264843,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M464 512a48 48 0 1096 0 48 48 0 10-96 0zm200 0a48 48 0 1096 0 48 48 0 10-96 0zm-400 0a48 48 0 1096 0 48 48 0 10-96 0zm661.2-173.6c-22.6-53.7-55-101.9-96.3-143.3a444.35 444.35 0 00-143.3-96.3C630.6 75.7 572.2 64 512 64h-2c-60.6.3-119.3 12.3-174.5 35.9a445.35 445.35 0 00-142 96.5c-40.9 41.3-73 89.3-95.2 142.8-23 55.4-34.6 114.3-34.3 174.9A449.4 449.4 0 00112 714v152a46 46 0 0046 46h152.1A449.4 449.4 0 00510 960h2.1c59.9 0 118-11.6 172.7-34.3a444.48 444.48 0 00142.8-95.2c41.3-40.9 73.8-88.7 96.5-142 23.6-55.2 35.6-113.9 35.9-174.5.3-60.9-11.5-120-34.8-175.6zm-151.1 438C704 845.8 611 884 512 884h-1.7c-60.3-.3-120.2-15.3-173.1-43.5l-8.4-4.5H188V695.2l-4.5-8.4C155.3 633.9 140.3 574 140 513.7c-.4-99.7 37.7-193.3 107.6-263.8 69.8-70.5 163.1-109.5 262.8-109.9h1.7c50 0 98.5 9.7 144.2 28.9 44.6 18.7 84.6 45.6 119 80 34.3 34.3 61.3 74.4 80 119 19.4 46.2 29.1 95.2 28.9 145.8-.6 99.6-39.7 192.9-110.1 262.7z"}}]},name:"message",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["MessageOutlined",0,r],264843)},44121,186515,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM115.4 518.9L271.7 642c5.8 4.6 14.4.5 14.4-6.9V388.9c0-7.4-8.5-11.5-14.4-6.9L115.4 505.1a8.74 8.74 0 000 13.8z"}}]},name:"menu-fold",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["MenuFoldOutlined",0,r],44121);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M408 442h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8zm-8 204c0 4.4 3.6 8 8 8h480c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8H408c-4.4 0-8 3.6-8 8v56zm504-486H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zm0 632H120c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h784c4.4 0 8-3.6 8-8v-56c0-4.4-3.6-8-8-8zM142.4 642.1L298.7 519a8.84 8.84 0 000-13.9L142.4 381.9c-5.8-4.6-14.4-.5-14.4 6.9v246.3a8.9 8.9 0 0014.4 7z"}}]},name:"menu-unfold",theme:"outlined"};var s=a.forwardRef(function(e,i){return a.createElement(n.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["MenuUnfoldOutlined",0,s],186515)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["SafetyOutlined",0,r],602073)},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return n}});let i=e.r(271645);function n(e,t){let a=(0,i.useRef)(null),n=(0,i.useRef)(null);return(0,i.useCallback)(i=>{if(null===i){let e=a.current;e&&(a.current=null,e());let t=n.current;t&&(n.current=null,t())}else e&&(a.current=r(e,i)),t&&(n.current=r(t,i))},[e,t])}function r(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},190272,785913,e=>{"use strict";var t,a,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:i,apiKey:r,inputMessage:o,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:m,mcpServers:g,mcpServerToolRestrictions:p,selectedVoice:f,endpointType:h,selectedModel:_,selectedSdk:b,proxySettings:v}=e,w="session"===a?i:r,x=window.location.origin,y=v?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?x=y:v?.PROXY_BASE_URL&&(x=v.PROXY_BASE_URL);let E=o||"Your prompt here",$=E.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),c.length>0&&(C.vector_stores=c),d.length>0&&(C.guardrails=d),u.length>0&&(C.policies=u);let k=_||"your-model-name",O="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${w||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${w||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"
)`;switch(h){case n.CHAT:{let e=Object.keys(C).length>0,a="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=j.length>0?j:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(i,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${k}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${$}"
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
`;break}case n.RESPONSES:{let e=Object.keys(C).length>0,a="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=j.length>0?j:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(i,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${k}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${$}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${k}",
	prompt="${o}",
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
prompt = "${$}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
`;break;case n.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${$}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
prompt = "${$}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
`;break;case n.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${o||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case n.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${o?`,
	prompt="${o.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${o||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${O}
${t}`}],190272)},735049,e=>{"use strict";var t=e.i(654310),a=function(e){if((0,t.default)()&&window.document.documentElement){var a=Array.isArray(e)?e:[e],i=window.document.documentElement;return a.some(function(e){return e in i.style})}return!1},i=function(e,t){if(!a(e))return!1;var i=document.createElement("div"),n=i.style[e];return i.style[e]=t,i.style[e]!==n};function n(e,t){return Array.isArray(e)||void 0===t?a(e):i(e,t)}e.s(["isStyleSupport",()=>n])},190144,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M832 64H296c-4.4 0-8 3.6-8 8v56c0 4.4 3.6 8 8 8h496v688c0 4.4 3.6 8 8 8h56c4.4 0 8-3.6 8-8V96c0-17.7-14.3-32-32-32zM704 192H192c-17.7 0-32 14.3-32 32v530.7c0 8.5 3.4 16.6 9.4 22.6l173.3 173.3c2.2 2.2 4.7 4 7.4 5.5v1.9h4.2c3.5 1.3 7.2 2 11 2H704c17.7 0 32-14.3 32-32V224c0-17.7-14.3-32-32-32zM350 856.2L263.9 770H350v86.2zM664 888H414V746c0-22.1-17.9-40-40-40H232V264h432v624z"}}]},name:"copy",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["default",0,r],190144)},464571,e=>{"use strict";var t=e.i(920228);e.s(["Button",()=>t.default])},185793,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),i=e.i(242064),n=e.i(529681);let r=e=>{let{prefixCls:i,className:n,style:r,size:o,shape:s}=e,l=(0,a.default)({[`${i}-lg`]:"large"===o,[`${i}-sm`]:"small"===o}),c=(0,a.default)({[`${i}-circle`]:"circle"===s,[`${i}-square`]:"square"===s,[`${i}-round`]:"round"===s}),d=t.useMemo(()=>"number"==typeof o?{width:o,height:o,lineHeight:`${o}px`}:{},[o]);return t.createElement("span",{className:(0,a.default)(i,l,c,n),style:Object.assign(Object.assign({},d),r)})};e.i(296059);var o=e.i(694758),s=e.i(915654),l=e.i(246422),c=e.i(838378);let d=new o.Keyframes("ant-skeleton-loading",{"0%":{backgroundPosition:"100% 50%"},"100%":{backgroundPosition:"0 50%"}}),u=e=>({height:e,lineHeight:(0,s.unit)(e)}),m=e=>Object.assign({width:e},u(e)),g=(e,t)=>Object.assign({width:t(e).mul(5).equal(),minWidth:t(e).mul(5).equal()},u(e)),p=e=>Object.assign({width:e},u(e)),f=(e,t,a)=>{let{skeletonButtonCls:i}=e;return{[`${a}${i}-circle`]:{width:t,minWidth:t,borderRadius:"50%"},[`${a}${i}-round`]:{borderRadius:t}}},h=(e,t)=>Object.assign({width:t(e).mul(2).equal(),minWidth:t(e).mul(2).equal()},u(e)),_=(0,l.genStyleHooks)("Skeleton",e=>{let{componentCls:t,calc:a}=e;return(e=>{let{componentCls:t,skeletonAvatarCls:a,skeletonTitleCls:i,skeletonParagraphCls:n,skeletonButtonCls:r,skeletonInputCls:o,skeletonImageCls:s,controlHeight:l,controlHeightLG:c,controlHeightSM:u,gradientFromColor:_,padding:b,marginSM:v,borderRadius:w,titleHeight:x,blockRadius:y,paragraphLiHeight:E,controlHeightXS:$,paragraphMarginTop:j}=e;return{[t]:{display:"table",width:"100%",[`${t}-header`]:{display:"table-cell",paddingInlineEnd:b,verticalAlign:"top",[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:_},m(l)),[`${a}-circle`]:{borderRadius:"50%"},[`${a}-lg`]:Object.assign({},m(c)),[`${a}-sm`]:Object.assign({},m(u))},[`${t}-content`]:{display:"table-cell",width:"100%",verticalAlign:"top",[i]:{width:"100%",height:x,background:_,borderRadius:y,[`+ ${n}`]:{marginBlockStart:u}},[n]:{padding:0,"> li":{width:"100%",height:E,listStyle:"none",background:_,borderRadius:y,"+ li":{marginBlockStart:$}}},[`${n}> li:last-child:not(:first-child):not(:nth-child(2))`]:{width:"61%"}},[`&-round ${t}-content`]:{[`${i}, ${n} > li`]:{borderRadius:w}}},[`${t}-with-avatar ${t}-content`]:{[i]:{marginBlockStart:v,[`+ ${n}`]:{marginBlockStart:j}}},[`${t}${t}-element`]:Object.assign(Object.assign(Object.assign(Object.assign({display:"inline-block",width:"auto"},(e=>{let{borderRadiusSM:t,skeletonButtonCls:a,controlHeight:i,controlHeightLG:n,controlHeightSM:r,gradientFromColor:o,calc:s}=e;return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:t,width:s(i).mul(2).equal(),minWidth:s(i).mul(2).equal()},h(i,s))},f(e,i,a)),{[`${a}-lg`]:Object.assign({},h(n,s))}),f(e,n,`${a}-lg`)),{[`${a}-sm`]:Object.assign({},h(r,s))}),f(e,r,`${a}-sm`))})(e)),(e=>{let{skeletonAvatarCls:t,gradientFromColor:a,controlHeight:i,controlHeightLG:n,controlHeightSM:r}=e;return{[t]:Object.assign({display:"inline-block",verticalAlign:"top",background:a},m(i)),[`${t}${t}-circle`]:{borderRadius:"50%"},[`${t}${t}-lg`]:Object.assign({},m(n)),[`${t}${t}-sm`]:Object.assign({},m(r))}})(e)),(e=>{let{controlHeight:t,borderRadiusSM:a,skeletonInputCls:i,controlHeightLG:n,controlHeightSM:r,gradientFromColor:o,calc:s}=e;return{[i]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:a},g(t,s)),[`${i}-lg`]:Object.assign({},g(n,s)),[`${i}-sm`]:Object.assign({},g(r,s))}})(e)),(e=>{let{skeletonImageCls:t,imageSizeBase:a,gradientFromColor:i,borderRadiusSM:n,calc:r}=e;return{[t]:Object.assign(Object.assign({display:"inline-flex",alignItems:"center",justifyContent:"center",verticalAlign:"middle",background:i,borderRadius:n},p(r(a).mul(2).equal())),{[`${t}-path`]:{fill:"#bfbfbf"},[`${t}-svg`]:Object.assign(Object.assign({},p(a)),{maxWidth:r(a).mul(4).equal(),maxHeight:r(a).mul(4).equal()}),[`${t}-svg${t}-svg-circle`]:{borderRadius:"50%"}}),[`${t}${t}-circle`]:{borderRadius:"50%"}}})(e)),[`${t}${t}-block`]:{width:"100%",[r]:{width:"100%"},[o]:{width:"100%"}},[`${t}${t}-active`]:{[`
        ${i},
        ${n} > li,
        ${a},
        ${r},
        ${o},
        ${s}
      `]:Object.assign({},{background:e.skeletonLoadingBackground,backgroundSize:"400% 100%",animationName:d,animationDuration:e.skeletonLoadingMotionDuration,animationTimingFunction:"ease",animationIterationCount:"infinite"})}}})((0,c.mergeToken)(e,{skeletonAvatarCls:`${t}-avatar`,skeletonTitleCls:`${t}-title`,skeletonParagraphCls:`${t}-paragraph`,skeletonButtonCls:`${t}-button`,skeletonInputCls:`${t}-input`,skeletonImageCls:`${t}-image`,imageSizeBase:a(e.controlHeight).mul(1.5).equal(),borderRadius:100,skeletonLoadingBackground:`linear-gradient(90deg, ${e.gradientFromColor} 25%, ${e.gradientToColor} 37%, ${e.gradientFromColor} 63%)`,skeletonLoadingMotionDuration:"1.4s"}))},e=>{let{colorFillContent:t,colorFill:a}=e;return{color:t,colorGradientEnd:a,gradientFromColor:t,gradientToColor:a,titleHeight:e.controlHeight/2,blockRadius:e.borderRadiusSM,paragraphMarginTop:e.marginLG+e.marginXXS,paragraphLiHeight:e.controlHeight/2}},{deprecatedTokens:[["color","gradientFromColor"],["colorGradientEnd","gradientToColor"]]}),b=e=>{let{prefixCls:i,className:n,style:r,rows:o=0}=e,s=Array.from({length:o}).map((a,i)=>t.createElement("li",{key:i,style:{width:((e,t)=>{let{width:a,rows:i=2}=t;return Array.isArray(a)?a[e]:i-1===e?a:void 0})(i,e)}}));return t.createElement("ul",{className:(0,a.default)(i,n),style:r},s)},v=({prefixCls:e,className:i,width:n,style:r})=>t.createElement("h3",{className:(0,a.default)(e,i),style:Object.assign({width:n},r)});function w(e){return e&&"object"==typeof e?e:{}}let x=e=>{let{prefixCls:n,loading:o,className:s,rootClassName:l,style:c,children:d,avatar:u=!1,title:m=!0,paragraph:g=!0,active:p,round:f}=e,{getPrefixCls:h,direction:x,className:y,style:E}=(0,i.useComponentConfig)("skeleton"),$=h("skeleton",n),[j,C,k]=_($);if(o||!("loading"in e)){let e,i,n=!!u,o=!!m,d=!!g;if(n){let a=Object.assign(Object.assign({prefixCls:`${$}-avatar`},o&&!d?{size:"large",shape:"square"}:{size:"large",shape:"circle"}),w(u));e=t.createElement("div",{className:`${$}-header`},t.createElement(r,Object.assign({},a)))}if(o||d){let e,a;if(o){let a=Object.assign(Object.assign({prefixCls:`${$}-title`},!n&&d?{width:"38%"}:n&&d?{width:"50%"}:{}),w(m));e=t.createElement(v,Object.assign({},a))}if(d){let e,i=Object.assign(Object.assign({prefixCls:`${$}-paragraph`},(e={},n&&o||(e.width="61%"),!n&&o?e.rows=3:e.rows=2,e)),w(g));a=t.createElement(b,Object.assign({},i))}i=t.createElement("div",{className:`${$}-content`},e,a)}let h=(0,a.default)($,{[`${$}-with-avatar`]:n,[`${$}-active`]:p,[`${$}-rtl`]:"rtl"===x,[`${$}-round`]:f},y,s,l,C,k);return j(t.createElement("div",{className:h,style:Object.assign(Object.assign({},E),c)},e,i))}return null!=d?d:null};x.Button=e=>{let{prefixCls:o,className:s,rootClassName:l,active:c,block:d=!1,size:u="default"}=e,{getPrefixCls:m}=t.useContext(i.ConfigContext),g=m("skeleton",o),[p,f,h]=_(g),b=(0,n.default)(e,["prefixCls"]),v=(0,a.default)(g,`${g}-element`,{[`${g}-active`]:c,[`${g}-block`]:d},s,l,f,h);return p(t.createElement("div",{className:v},t.createElement(r,Object.assign({prefixCls:`${g}-button`,size:u},b))))},x.Avatar=e=>{let{prefixCls:o,className:s,rootClassName:l,active:c,shape:d="circle",size:u="default"}=e,{getPrefixCls:m}=t.useContext(i.ConfigContext),g=m("skeleton",o),[p,f,h]=_(g),b=(0,n.default)(e,["prefixCls","className"]),v=(0,a.default)(g,`${g}-element`,{[`${g}-active`]:c},s,l,f,h);return p(t.createElement("div",{className:v},t.createElement(r,Object.assign({prefixCls:`${g}-avatar`,shape:d,size:u},b))))},x.Input=e=>{let{prefixCls:o,className:s,rootClassName:l,active:c,block:d,size:u="default"}=e,{getPrefixCls:m}=t.useContext(i.ConfigContext),g=m("skeleton",o),[p,f,h]=_(g),b=(0,n.default)(e,["prefixCls"]),v=(0,a.default)(g,`${g}-element`,{[`${g}-active`]:c,[`${g}-block`]:d},s,l,f,h);return p(t.createElement("div",{className:v},t.createElement(r,Object.assign({prefixCls:`${g}-input`,size:u},b))))},x.Image=e=>{let{prefixCls:n,className:r,rootClassName:o,style:s,active:l}=e,{getPrefixCls:c}=t.useContext(i.ConfigContext),d=c("skeleton",n),[u,m,g]=_(d),p=(0,a.default)(d,`${d}-element`,{[`${d}-active`]:l},r,o,m,g);return u(t.createElement("div",{className:p},t.createElement("div",{className:(0,a.default)(`${d}-image`,r),style:s},t.createElement("svg",{viewBox:"0 0 1098 1024",xmlns:"http://www.w3.org/2000/svg",className:`${d}-image-svg`},t.createElement("title",null,"Image placeholder"),t.createElement("path",{d:"M365.714286 329.142857q0 45.714286-32.036571 77.677714t-77.677714 32.036571-77.677714-32.036571-32.036571-77.677714 32.036571-77.677714 77.677714-32.036571 77.677714 32.036571 32.036571 77.677714zM950.857143 548.571429l0 256-804.571429 0 0-109.714286 182.857143-182.857143 91.428571 91.428571 292.571429-292.571429zM1005.714286 146.285714l-914.285714 0q-7.460571 0-12.873143 5.412571t-5.412571 12.873143l0 694.857143q0 7.460571 5.412571 12.873143t12.873143 5.412571l914.285714 0q7.460571 0 12.873143-5.412571t5.412571-12.873143l0-694.857143q0-7.460571-5.412571-12.873143t-12.873143-5.412571zM1097.142857 164.571429l0 694.857143q0 37.741714-26.843429 64.585143t-64.585143 26.843429l-914.285714 0q-37.741714 0-64.585143-26.843429t-26.843429-64.585143l0-694.857143q0-37.741714 26.843429-64.585143t64.585143-26.843429l914.285714 0q37.741714 0 64.585143 26.843429t26.843429 64.585143z",className:`${d}-image-path`})))))},x.Node=e=>{let{prefixCls:n,className:r,rootClassName:o,style:s,active:l,children:c}=e,{getPrefixCls:d}=t.useContext(i.ConfigContext),u=d("skeleton",n),[m,g,p]=_(u),f=(0,a.default)(u,`${u}-element`,{[`${u}-active`]:l},g,r,o,p);return m(t.createElement("div",{className:f},t.createElement("div",{className:(0,a.default)(`${u}-image`,r),style:s},c)))},e.s(["default",0,x],185793)},959013,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M482 152h60q8 0 8 8v704q0 8-8 8h-60q-8 0-8-8V160q0-8 8-8z"}},{tag:"path",attrs:{d:"M192 474h672q8 0 8 8v60q0 8-8 8H160q-8 0-8-8v-60q0-8 8-8z"}}]},name:"plus",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["default",0,r],959013)},269200,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("Table"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement("div",{className:(0,i.tremorTwMerge)(n("root"),"overflow-auto",s)},a.default.createElement("table",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("table"),"w-full text-tremor-default","text-tremor-content","dark:text-dark-tremor-content")},l),o))});r.displayName="Table",e.s(["Table",()=>r],269200)},427612,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("TableHead"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement(a.default.Fragment,null,a.default.createElement("thead",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("root"),"text-left","text-tremor-content","dark:text-dark-tremor-content",s)},l),o))});r.displayName="TableHead",e.s(["TableHead",()=>r],427612)},64848,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("TableHeaderCell"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement(a.default.Fragment,null,a.default.createElement("th",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("root"),"whitespace-nowrap text-left font-semibold top-0 px-4 py-3.5","text-tremor-content-strong","dark:text-dark-tremor-content-strong",s)},l),o))});r.displayName="TableHeaderCell",e.s(["TableHeaderCell",()=>r],64848)},942232,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("TableBody"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement(a.default.Fragment,null,a.default.createElement("tbody",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("root"),"align-top divide-y","divide-tremor-border","dark:divide-dark-tremor-border",s)},l),o))});r.displayName="TableBody",e.s(["TableBody",()=>r],942232)},496020,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("TableRow"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement(a.default.Fragment,null,a.default.createElement("tr",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("row"),s)},l),o))});r.displayName="TableRow",e.s(["TableRow",()=>r],496020)},977572,e=>{"use strict";var t=e.i(290571),a=e.i(271645),i=e.i(444755);let n=(0,e.i(673706).makeClassName)("TableCell"),r=a.default.forwardRef((e,r)=>{let{children:o,className:s}=e,l=(0,t.__rest)(e,["children","className"]);return a.default.createElement(a.default.Fragment,null,a.default.createElement("td",Object.assign({ref:r,className:(0,i.tremorTwMerge)(n("root"),"align-middle whitespace-nowrap text-left p-4",s)},l),o))});r.displayName="TableCell",e.s(["TableCell",()=>r],977572)},360820,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["ChevronUpIcon",0,a],360820)},871943,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["ChevronDownIcon",0,a],871943)},94629,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,a],94629)},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},434626,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,a],434626)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var n=e.i(9583),r=a.forwardRef(function(e,r){return a.createElement(n.default,(0,t.default)({},e,{ref:r,icon:i}))});e.s(["CrownOutlined",0,r],100486)},798496,e=>{"use strict";var t=e.i(843476),a=e.i(152990),i=e.i(682830),n=e.i(271645),r=e.i(269200),o=e.i(427612),s=e.i(64848),l=e.i(942232),c=e.i(496020),d=e.i(977572),u=e.i(94629),m=e.i(360820),g=e.i(871943);function p({data:e=[],columns:p,isLoading:f=!1,defaultSorting:h=[],pagination:_,onPaginationChange:b,enablePagination:v=!1,onRowClick:w}){let[x,y]=n.default.useState(h),[E]=n.default.useState("onChange"),[$,j]=n.default.useState({}),[C,k]=n.default.useState({}),O=(0,a.useReactTable)({data:e,columns:p,state:{sorting:x,columnSizing:$,columnVisibility:C,...v&&_?{pagination:_}:{}},columnResizeMode:E,onSortingChange:y,onColumnSizingChange:j,onColumnVisibilityChange:k,...v&&b?{onPaginationChange:b}:{},getCoreRowModel:(0,i.getCoreRowModel)(),getSortedRowModel:(0,i.getSortedRowModel)(),...v?{getPaginationRowModel:(0,i.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(r.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:O.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(o.TableHead,{children:O.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(s.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,a.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(m.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(g.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(l.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:p.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):O.getRowModel().rows.length>0?O.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>w?.(e.original),className:w?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,a.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:p.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>p])}]);