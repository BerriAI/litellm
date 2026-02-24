(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,62478,e=>{"use strict";var t=e.i(764205);let o=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,o])},818581,(e,t,o)=>{"use strict";Object.defineProperty(o,"__esModule",{value:!0}),Object.defineProperty(o,"useMergedRef",{enumerable:!0,get:function(){return i}});let a=e.r(271645);function i(e,t){let o=(0,a.useRef)(null),i=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=o.current;e&&(o.current=null,e());let t=i.current;t&&(i.current=null,t())}else e&&(o.current=r(e,a)),t&&(i.current=r(t,a))},[e,t])}function r(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let o=e(t);return"function"==typeof o?o:()=>e(null)}}("function"==typeof o.default||"object"==typeof o.default&&null!==o.default)&&void 0===o.default.__esModule&&(Object.defineProperty(o.default,"__esModule",{value:!0}),Object.assign(o.default,o),t.exports=o.default)},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var i=e.i(9583),r=o.forwardRef(function(e,r){return o.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["SafetyOutlined",0,r],602073)},190272,785913,e=>{"use strict";var t,o,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((o={}).IMAGE="image",o.VIDEO="video",o.CHAT="chat",o.RESPONSES="responses",o.IMAGE_EDITS="image_edits",o.ANTHROPIC_MESSAGES="anthropic_messages",o.EMBEDDINGS="embeddings",o.SPEECH="speech",o.TRANSCRIPTION="transcription",o.A2A_AGENTS="a2a_agents",o.MCP="mcp",o);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:o,accessToken:a,apiKey:r,inputMessage:n,chatHistory:l,selectedTags:s,selectedVectorStores:p,selectedGuardrails:c,selectedPolicies:g,selectedMCPServers:u,mcpServers:d,mcpServerToolRestrictions:m,selectedVoice:f,endpointType:_,selectedModel:h,selectedSdk:b,proxySettings:v}=e,y="session"===o?a:r,I=window.location.origin,x=v?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?I=x:v?.PROXY_BASE_URL&&(I=v.PROXY_BASE_URL);let A=n||"Your prompt here",w=A.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};s.length>0&&(C.tags=s),p.length>0&&(C.vector_stores=p),c.length>0&&(C.guardrails=c),g.length>0&&(C.policies=g);let $=h||"your-model-name",k="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${I}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${I}"
)`;switch(_){case i.CHAT:{let e=Object.keys(C).length>0,o="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let a=S.length>0?S:[{role:"user",content:A}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${$}",
    messages=${JSON.stringify(a,null,4)}${o}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${$}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${w}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${o}
# )
# print(response_with_file)
`;break}case i.RESPONSES:{let e=Object.keys(C).length>0,o="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let a=S.length>0?S:[{role:"user",content:A}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${$}",
    input=${JSON.stringify(a,null,4)}${o}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${$}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${w}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${o}
# )
# print(response_with_file.output_text)
`;break}case i.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${$}",
	prompt="${n}",
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
prompt = "${w}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${$}",
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
`;break;case i.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${w}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${$}",
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
prompt = "${w}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${$}",
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
`;break;case i.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${$}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${$}",
	file=audio_file${n?`,
	prompt="${n.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${$}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${$}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${k}
${t}`}],190272)},115571,371401,e=>{"use strict";let t="local-storage-change";function o(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function a(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function i(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function r(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>o,"getLocalStorageItem",()=>a,"removeLocalStorageItem",()=>r,"setLocalStorageItem",()=>i],115571);var n=e.i(271645);function l(e){let o=t=>{"disableUsageIndicator"===t.key&&e()},a=t=>{let{key:o}=t.detail;"disableUsageIndicator"===o&&e()};return window.addEventListener("storage",o),window.addEventListener(t,a),()=>{window.removeEventListener("storage",o),window.removeEventListener(t,a)}}function s(){return"true"===a("disableUsageIndicator")}function p(){return(0,n.useSyncExternalStore)(l,s)}e.s(["useDisableUsageIndicator",()=>p],371401)},275144,e=>{"use strict";var t=e.i(843476),o=e.i(271645),a=e.i(764205);let i=(0,o.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:r})=>{let[n,l]=(0,o.useState)(null),[s,p]=(0,o.useState)(null);return(0,o.useEffect)(()=>{(async()=>{try{let e=(0,a.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",o=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(o.ok){let e=await o.json();e.values?.logo_url&&l(e.values.logo_url),e.values?.favicon_url&&p(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,o.useEffect)(()=>{if(s){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=s});else{let e=document.createElement("link");e.rel="icon",e.href=s,document.head.appendChild(e)}}},[s]),(0,t.jsx)(i.Provider,{value:{logoUrl:n,setLogoUrl:l,faviconUrl:s,setFaviconUrl:p},children:e})},"useTheme",0,()=>{let e=(0,o.useContext)(i);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},798496,e=>{"use strict";var t=e.i(843476),o=e.i(152990),a=e.i(682830),i=e.i(271645),r=e.i(269200),n=e.i(427612),l=e.i(64848),s=e.i(942232),p=e.i(496020),c=e.i(977572),g=e.i(94629),u=e.i(360820),d=e.i(871943);function m({data:e=[],columns:m,isLoading:f=!1,defaultSorting:_=[],pagination:h,onPaginationChange:b,enablePagination:v=!1}){let[y,I]=i.default.useState(_),[x]=i.default.useState("onChange"),[A,w]=i.default.useState({}),[S,C]=i.default.useState({}),$=(0,o.useReactTable)({data:e,columns:m,state:{sorting:y,columnSizing:A,columnVisibility:S,...v&&h?{pagination:h}:{}},columnResizeMode:x,onSortingChange:I,onColumnSizingChange:w,onColumnVisibilityChange:C,...v&&b?{onPaginationChange:b}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),...v?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(r.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:$.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:$.getHeaderGroups().map(e=>(0,t.jsx)(p.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,o.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(u.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(d.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(g.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(p.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):$.getRowModel().rows.length>0?$.getRowModel().rows.map(e=>(0,t.jsx)(p.TableRow,{children:e.getVisibleCells().map(e=>(0,t.jsx)(c.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,o.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(p.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>m])},434626,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,o],434626)},262218,e=>{"use strict";e.i(247167);var t=e.i(271645),o=e.i(343794),a=e.i(529681),i=e.i(702779),r=e.i(563113),n=e.i(763731),l=e.i(121872),s=e.i(242064);e.i(296059);var p=e.i(915654);e.i(262370);var c=e.i(135551),g=e.i(183293),u=e.i(246422),d=e.i(838378);let m=e=>{let{lineWidth:t,fontSizeIcon:o,calc:a}=e,i=e.fontSizeSM;return(0,d.mergeToken)(e,{tagFontSize:i,tagLineHeight:(0,p.unit)(a(e.lineHeightSM).mul(i).equal()),tagIconSize:a(o).sub(a(t).mul(2)).equal(),tagPaddingHorizontal:8,tagBorderlessBg:e.defaultBg})},f=e=>({defaultBg:new c.FastColor(e.colorFillQuaternary).onBackground(e.colorBgContainer).toHexString(),defaultColor:e.colorText}),_=(0,u.genStyleHooks)("Tag",e=>(e=>{let{paddingXXS:t,lineWidth:o,tagPaddingHorizontal:a,componentCls:i,calc:r}=e,n=r(a).sub(o).equal(),l=r(t).sub(o).equal();return{[i]:Object.assign(Object.assign({},(0,g.resetComponent)(e)),{display:"inline-block",height:"auto",marginInlineEnd:e.marginXS,paddingInline:n,fontSize:e.tagFontSize,lineHeight:e.tagLineHeight,whiteSpace:"nowrap",background:e.defaultBg,border:`${(0,p.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,opacity:1,transition:`all ${e.motionDurationMid}`,textAlign:"start",position:"relative",[`&${i}-rtl`]:{direction:"rtl"},"&, a, a:hover":{color:e.defaultColor},[`${i}-close-icon`]:{marginInlineStart:l,fontSize:e.tagIconSize,color:e.colorIcon,cursor:"pointer",transition:`all ${e.motionDurationMid}`,"&:hover":{color:e.colorTextHeading}},[`&${i}-has-color`]:{borderColor:"transparent",[`&, a, a:hover, ${e.iconCls}-close, ${e.iconCls}-close:hover`]:{color:e.colorTextLightSolid}},"&-checkable":{backgroundColor:"transparent",borderColor:"transparent",cursor:"pointer",[`&:not(${i}-checkable-checked):hover`]:{color:e.colorPrimary,backgroundColor:e.colorFillSecondary},"&:active, &-checked":{color:e.colorTextLightSolid},"&-checked":{backgroundColor:e.colorPrimary,"&:hover":{backgroundColor:e.colorPrimaryHover}},"&:active":{backgroundColor:e.colorPrimaryActive}},"&-hidden":{display:"none"},[`> ${e.iconCls} + span, > span + ${e.iconCls}`]:{marginInlineStart:n}}),[`${i}-borderless`]:{borderColor:"transparent",background:e.tagBorderlessBg}}})(m(e)),f);var h=function(e,t){var o={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(o[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(o[a[i]]=e[a[i]]);return o};let b=t.forwardRef((e,a)=>{let{prefixCls:i,style:r,className:n,checked:l,children:p,icon:c,onChange:g,onClick:u}=e,d=h(e,["prefixCls","style","className","checked","children","icon","onChange","onClick"]),{getPrefixCls:m,tag:f}=t.useContext(s.ConfigContext),b=m("tag",i),[v,y,I]=_(b),x=(0,o.default)(b,`${b}-checkable`,{[`${b}-checkable-checked`]:l},null==f?void 0:f.className,n,y,I);return v(t.createElement("span",Object.assign({},d,{ref:a,style:Object.assign(Object.assign({},r),null==f?void 0:f.style),className:x,onClick:e=>{null==g||g(!l),null==u||u(e)}}),c,t.createElement("span",null,p)))});var v=e.i(403541);let y=(0,u.genSubStyleComponent)(["Tag","preset"],e=>{let t;return t=m(e),(0,v.genPresetColor)(t,(e,{textColor:o,lightBorderColor:a,lightColor:i,darkColor:r})=>({[`${t.componentCls}${t.componentCls}-${e}`]:{color:o,background:i,borderColor:a,"&-inverse":{color:t.colorTextLightSolid,background:r,borderColor:r},[`&${t.componentCls}-borderless`]:{borderColor:"transparent"}}}))},f),I=(e,t,o)=>{let a="string"!=typeof o?o:o.charAt(0).toUpperCase()+o.slice(1);return{[`${e.componentCls}${e.componentCls}-${t}`]:{color:e[`color${o}`],background:e[`color${a}Bg`],borderColor:e[`color${a}Border`],[`&${e.componentCls}-borderless`]:{borderColor:"transparent"}}}},x=(0,u.genSubStyleComponent)(["Tag","status"],e=>{let t=m(e);return[I(t,"success","Success"),I(t,"processing","Info"),I(t,"error","Error"),I(t,"warning","Warning")]},f);var A=function(e,t){var o={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(o[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(o[a[i]]=e[a[i]]);return o};let w=t.forwardRef((e,p)=>{let{prefixCls:c,className:g,rootClassName:u,style:d,children:m,icon:f,color:h,onClose:b,bordered:v=!0,visible:I}=e,w=A(e,["prefixCls","className","rootClassName","style","children","icon","color","onClose","bordered","visible"]),{getPrefixCls:S,direction:C,tag:$}=t.useContext(s.ConfigContext),[k,E]=t.useState(!0),O=(0,a.default)(w,["closeIcon","closable"]);t.useEffect(()=>{void 0!==I&&E(I)},[I]);let j=(0,i.isPresetColor)(h),T=(0,i.isPresetStatusColor)(h),M=j||T,P=Object.assign(Object.assign({backgroundColor:h&&!M?h:void 0},null==$?void 0:$.style),d),L=S("tag",c),[N,R,D]=_(L),z=(0,o.default)(L,null==$?void 0:$.className,{[`${L}-${h}`]:M,[`${L}-has-color`]:h&&!M,[`${L}-hidden`]:!k,[`${L}-rtl`]:"rtl"===C,[`${L}-borderless`]:!v},g,u,R,D),G=e=>{e.stopPropagation(),null==b||b(e),e.defaultPrevented||E(!1)},[,H]=(0,r.useClosable)((0,r.pickClosable)(e),(0,r.pickClosable)($),{closable:!1,closeIconRender:e=>{let a=t.createElement("span",{className:`${L}-close-icon`,onClick:G},e);return(0,n.replaceElement)(e,a,e=>({onClick:t=>{var o;null==(o=null==e?void 0:e.onClick)||o.call(e,t),G(t)},className:(0,o.default)(null==e?void 0:e.className,`${L}-close-icon`)}))}}),B="function"==typeof w.onClick||m&&"a"===m.type,F=f||null,V=F?t.createElement(t.Fragment,null,F,m&&t.createElement("span",null,m)):m,U=t.createElement("span",Object.assign({},O,{ref:p,className:z,style:P}),V,H,j&&t.createElement(y,{key:"preset",prefixCls:L}),T&&t.createElement(x,{key:"status",prefixCls:L}));return N(B?t.createElement(l.default,{component:"Tag"},U):U)});w.CheckableTag=b,e.s(["Tag",0,w],262218)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var i=e.i(9583),r=o.forwardRef(function(e,r){return o.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["UserOutlined",0,r],771674)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var i=e.i(9583),r=o.forwardRef(function(e,r){return o.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["CrownOutlined",0,r],100486)},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},94629,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,o],94629)},916925,e=>{"use strict";var t,o=((t={}).A2A_Agent="A2A Agent",t.AIML="AI/ML API",t.Bedrock="Amazon Bedrock",t.Anthropic="Anthropic",t.AssemblyAI="AssemblyAI",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.Cerebras="Cerebras",t.Cohere="Cohere",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.ElevenLabs="ElevenLabs",t.FalAI="Fal AI",t.FireworksAI="Fireworks AI",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.Hosted_Vllm="vllm",t.Infinity="Infinity",t.JinaAI="Jina AI",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.Ollama="Ollama",t.OpenAI="OpenAI",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.Perplexity="Perplexity",t.RunwayML="RunwayML",t.Sambanova="Sambanova",t.Snowflake="Snowflake",t.TogetherAI="TogetherAI",t.Triton="Triton",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.xAI="xAI",t.SAP="SAP Generative AI Hub",t.Watsonx="Watsonx",t);let a={A2A_Agent:"a2a_agent",AIML:"aiml",OpenAI:"openai",OpenAI_Text:"text-completion-openai",Azure:"azure",Azure_AI_Studio:"azure_ai",Anthropic:"anthropic",Google_AI_Studio:"gemini",Bedrock:"bedrock",Groq:"groq",MiniMax:"minimax",MistralAI:"mistral",Cohere:"cohere",OpenAI_Compatible:"openai",OpenAI_Text_Compatible:"text-completion-openai",Vertex_AI:"vertex_ai",Databricks:"databricks",Dashscope:"dashscope",xAI:"xai",Deepseek:"deepseek",Ollama:"ollama",AssemblyAI:"assemblyai",Cerebras:"cerebras",Sambanova:"sambanova",Perplexity:"perplexity",RunwayML:"runwayml",TogetherAI:"together_ai",Openrouter:"openrouter",Oracle:"oci",Snowflake:"snowflake",FireworksAI:"fireworks_ai",GradientAI:"gradient_ai",Triton:"triton",Deepgram:"deepgram",ElevenLabs:"elevenlabs",FalAI:"fal_ai",SageMaker:"sagemaker_chat",Voyage:"voyage",JinaAI:"jina_ai",VolcEngine:"volcengine",DeepInfra:"deepinfra",Hosted_Vllm:"hosted_vllm",Infinity:"infinity",SAP:"sap",Watsonx:"watsonx"},i="../ui/assets/logos/",r={"A2A Agent":`${i}a2a_agent.png`,"AI/ML API":`${i}aiml_api.svg`,Anthropic:`${i}anthropic.svg`,AssemblyAI:`${i}assemblyai_small.png`,Azure:`${i}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${i}microsoft_azure.svg`,"Amazon Bedrock":`${i}bedrock.svg`,"AWS SageMaker":`${i}bedrock.svg`,Cerebras:`${i}cerebras.svg`,Cohere:`${i}cohere.svg`,"Databricks (Qwen API)":`${i}databricks.svg`,Dashscope:`${i}dashscope.svg`,Deepseek:`${i}deepseek.svg`,"Fireworks AI":`${i}fireworks.svg`,Groq:`${i}groq.svg`,"Google AI Studio":`${i}google.svg`,vllm:`${i}vllm.png`,Infinity:`${i}infinity.png`,MiniMax:`${i}minimax.svg`,"Mistral AI":`${i}mistral.svg`,Ollama:`${i}ollama.svg`,OpenAI:`${i}openai_small.svg`,"OpenAI Text Completion":`${i}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${i}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${i}openai_small.svg`,Openrouter:`${i}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${i}oracle.svg`,Perplexity:`${i}perplexity-ai.svg`,RunwayML:`${i}runwayml.png`,Sambanova:`${i}sambanova.svg`,Snowflake:`${i}snowflake.svg`,TogetherAI:`${i}togetherai.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${i}google.svg`,xAI:`${i}xai.svg`,GradientAI:`${i}gradientai.svg`,Triton:`${i}nvidia_triton.png`,Deepgram:`${i}deepgram.png`,ElevenLabs:`${i}elevenlabs.png`,"Fal AI":`${i}fal_ai.jpg`,"Voyage AI":`${i}voyage.webp`,"Jina AI":`${i}jina.png`,VolcEngine:`${i}volcengine.png`,DeepInfra:`${i}deepinfra.png`,"SAP Generative AI Hub":`${i}sap.png`};e.s(["Providers",()=>o,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:r[e],displayName:e}}let t=Object.keys(a).find(t=>a[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let i=o[t];return{logo:r[i],displayName:i}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let o=a[e];console.log(`Provider mapped to: ${o}`);let i=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let a=t.litellm_provider;(a===o||"string"==typeof a&&a.includes(o))&&i.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&i.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&i.push(e)}))),i},"providerLogoMap",0,r,"provider_map",0,a])}]);