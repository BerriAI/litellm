(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},798496,e=>{"use strict";var t=e.i(843476),i=e.i(152990),o=e.i(682830),n=e.i(271645),a=e.i(269200),r=e.i(427612),l=e.i(64848),s=e.i(942232),d=e.i(496020),c=e.i(977572),u=e.i(94629),p=e.i(360820),m=e.i(871943);function g({data:e=[],columns:g,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:_,enablePagination:y=!1,onRowClick:x}){let[v,w]=n.default.useState(h),[S]=n.default.useState("onChange"),[j,C]=n.default.useState({}),[$,k]=n.default.useState({}),O=(0,i.useReactTable)({data:e,columns:g,state:{sorting:v,columnSizing:j,columnVisibility:$,...y&&b?{pagination:b}:{}},columnResizeMode:S,onSortingChange:w,onColumnSizingChange:C,onColumnVisibilityChange:k,...y&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...y?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(a.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:O.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(r.TableHead,{children:O.getHeaderGroups().map(e=>(0,t.jsx)(d.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,i.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(p.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(m.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(d.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):O.getRowModel().rows.length>0?O.getRowModel().rows.map(e=>(0,t.jsx)(d.TableRow,{onClick:()=>x?.(e.original),className:x?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(c.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,i.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(d.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>g])},339019,865361,e=>{"use strict";var t,i,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i.INTERACTIONS="interactions",i);let a={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=a[e];return console.log("endpointType:",t),t}return"chat"}],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:o,apiKey:a,inputMessage:r,chatHistory:l,selectedTags:s,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedMCPServers:p,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:_,proxySettings:y}=e,x="session"===i?o:a,v=window.location.origin,w=y?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:y?.PROXY_BASE_URL&&(v=y.PROXY_BASE_URL);let S=r||"Your prompt here",j=S.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),C=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),$={};s.length>0&&($.tags=s),d.length>0&&($.vector_stores=d),c.length>0&&($.guardrails=c),u.length>0&&($.policies=u);let k=b||"your-model-name",O="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case n.CHAT:{let e=Object.keys($).length>0,i="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=C.length>0?C:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(o,null,4)}${i}
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
#                     "text": "${j}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${i}
# )
# print(response_with_file)
`;break}case n.RESPONSES:{let e=Object.keys($).length>0,i="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=C.length>0?C:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(o,null,4)}${i}
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===_?`
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
	prompt="${r}",
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
prompt = "${j}"

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
`;break;case n.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${j}"

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
prompt = "${j}"

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
	input="${r||"Your string here"}",
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
	file=audio_file${r?`,
	prompt="${r.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${r||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${r||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${O}
${t}`}],339019)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["LinkOutlined",0,a],596239)},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(447566),n=e.i(166406),a=e.i(492030),r=e.i(596239);let l=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,l,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:s})=>{let d,[c,u]=(0,i.useState)("overview"),[p,m]=(0,i.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},f="github"===(d=e.source).source&&d.repo?`https://github.com/${d.repo}`:"git-subdir"===d.source&&d.url?d.path?`${d.url}/tree/main/${d.path}`:d.url:"url"===d.source&&d.url?d.url:null,h=l(e),b=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:s,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(o.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>u(e.key),style:{padding:"12px 20px",fontSize:14,color:c===e.key?"#1a73e8":"#5f6368",borderBottom:c===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:c===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===c&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:b.map((e,i)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),f&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:f,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[f.replace("https://",""),(0,t.jsx)(r.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(h,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===p?(0,t.jsx)(a.CheckOutlined,{}):(0,t.jsx)(n.CopyOutlined,{}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:h})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>u("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to"," ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," ","to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{g(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===p?(0,t.jsx)(a.CheckOutlined,{}):(0,t.jsx)(n.CopyOutlined,{}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["SafetyOutlined",0,a],602073)},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return n}});let o=e.r(271645);function n(e,t){let i=(0,o.useRef)(null),n=(0,o.useRef)(null);return(0,o.useCallback)(o=>{if(null===o){let e=i.current;e&&(i.current=null,e());let t=n.current;t&&(n.current=null,t())}else e&&(i.current=a(e,o)),t&&(n.current=a(t,o))},[e,t])}function a(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},283713,e=>{"use strict";var t=e.i(271645),i=e.i(602869),o=e.i(612256);let n="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,o.useUIConfig)(),a=e?.is_control_plane??!1,r=e?.workers??[],[l,s]=(0,t.useState)(()=>localStorage.getItem(n));(0,t.useEffect)(()=>{if(!l||0===r.length)return;let e=r.find(e=>e.worker_id===l);e&&(0,i.switchToWorkerUrl)(e.url)},[l,r]);let d=r.find(e=>e.worker_id===l)??null,c=(0,t.useCallback)(e=>{let t=r.find(t=>t.worker_id===e);t&&(s(e),localStorage.setItem(n,e),(0,i.switchToWorkerUrl)(t.url))},[r]);return{isControlPlane:a,workers:r,selectedWorkerId:l,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{s(null),localStorage.removeItem(n),(0,i.switchToWorkerUrl)(null)},[])}}])},295320,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M704 446H320c-4.4 0-8 3.6-8 8v402c0 4.4 3.6 8 8 8h384c4.4 0 8-3.6 8-8V454c0-4.4-3.6-8-8-8zm-328 64h272v117H376V510zm272 290H376V683h272v117z"}},{tag:"path",attrs:{d:"M424 748a32 32 0 1064 0 32 32 0 10-64 0zm0-178a32 32 0 1064 0 32 32 0 10-64 0z"}},{tag:"path",attrs:{d:"M811.4 368.9C765.6 248 648.9 162 512.2 162S258.8 247.9 213 368.8C126.9 391.5 63.5 470.2 64 563.6 64.6 668 145.6 752.9 247.6 762c4.7.4 8.7-3.3 8.7-8v-60.4c0-4-3-7.4-7-7.9-27-3.4-52.5-15.2-72.1-34.5-24-23.5-37.2-55.1-37.2-88.6 0-28 9.1-54.4 26.2-76.4 16.7-21.4 40.2-36.9 66.1-43.7l37.9-10 13.9-36.7c8.6-22.8 20.6-44.2 35.7-63.5 14.9-19.2 32.6-36 52.4-50 41.1-28.9 89.5-44.2 140-44.2s98.9 15.3 140 44.3c19.9 14 37.5 30.8 52.4 50 15.1 19.3 27.1 40.7 35.7 63.5l13.8 36.6 37.8 10c54.2 14.4 92.1 63.7 92.1 120 0 33.6-13.2 65.1-37.2 88.6-19.5 19.2-44.9 31.1-71.9 34.5-4 .5-6.9 3.9-6.9 7.9V754c0 4.7 4.1 8.4 8.8 8 101.7-9.2 182.5-94 183.2-198.2.6-93.4-62.7-172.1-148.6-194.9z"}}]},name:"cloud-server",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["CloudServerOutlined",0,a],295320)},906579,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),o=e.i(361275),n=e.i(702779),a=e.i(763731),r=e.i(242064);e.i(296059);var l=e.i(915654),s=e.i(694758),d=e.i(183293),c=e.i(403541),u=e.i(246422),p=e.i(838378);let m=new s.Keyframes("antStatusProcessing",{"0%":{transform:"scale(0.8)",opacity:.5},"100%":{transform:"scale(2.4)",opacity:0}}),g=new s.Keyframes("antZoomBadgeIn",{"0%":{transform:"scale(0) translate(50%, -50%)",opacity:0},"100%":{transform:"scale(1) translate(50%, -50%)"}}),f=new s.Keyframes("antZoomBadgeOut",{"0%":{transform:"scale(1) translate(50%, -50%)"},"100%":{transform:"scale(0) translate(50%, -50%)",opacity:0}}),h=new s.Keyframes("antNoWrapperZoomBadgeIn",{"0%":{transform:"scale(0)",opacity:0},"100%":{transform:"scale(1)"}}),b=new s.Keyframes("antNoWrapperZoomBadgeOut",{"0%":{transform:"scale(1)"},"100%":{transform:"scale(0)",opacity:0}}),_=new s.Keyframes("antBadgeLoadingCircle",{"0%":{transformOrigin:"50%"},"100%":{transform:"translate(50%, -50%) rotate(360deg)",transformOrigin:"50%"}}),y=e=>{let{fontHeight:t,lineWidth:i,marginXS:o,colorBorderBg:n}=e,a=e.colorTextLightSolid,r=e.colorError,l=e.colorErrorHover;return(0,p.mergeToken)(e,{badgeFontHeight:t,badgeShadowSize:i,badgeTextColor:a,badgeColor:r,badgeColorHover:l,badgeShadowColor:n,badgeProcessingDuration:"1.2s",badgeRibbonOffset:o,badgeRibbonCornerTransform:"scaleY(0.75)",badgeRibbonCornerFilter:"brightness(75%)"})},x=e=>{let{fontSize:t,lineHeight:i,fontSizeSM:o,lineWidth:n}=e;return{indicatorZIndex:"auto",indicatorHeight:Math.round(t*i)-2*n,indicatorHeightSM:t,dotSize:o/2,textFontSize:o,textFontSizeSM:o,textFontWeight:"normal",statusSize:o/2}},v=(0,u.genStyleHooks)("Badge",e=>(e=>{let{componentCls:t,iconCls:i,antCls:o,badgeShadowSize:n,textFontSize:a,textFontSizeSM:r,statusSize:s,dotSize:u,textFontWeight:p,indicatorHeight:y,indicatorHeightSM:x,marginXS:v,calc:w}=e,S=`${o}-scroll-number`,j=(0,c.genPresetColor)(e,(e,{darkColor:i})=>({[`&${t} ${t}-color-${e}`]:{background:i,[`&:not(${t}-count)`]:{color:i},"a:hover &":{background:i}}}));return{[t]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,d.resetComponent)(e)),{position:"relative",display:"inline-block",width:"fit-content",lineHeight:1,[`${t}-count`]:{display:"inline-flex",justifyContent:"center",zIndex:e.indicatorZIndex,minWidth:y,height:y,color:e.badgeTextColor,fontWeight:p,fontSize:a,lineHeight:(0,l.unit)(y),whiteSpace:"nowrap",textAlign:"center",background:e.badgeColor,borderRadius:w(y).div(2).equal(),boxShadow:`0 0 0 ${(0,l.unit)(n)} ${e.badgeShadowColor}`,transition:`background ${e.motionDurationMid}`,a:{color:e.badgeTextColor},"a:hover":{color:e.badgeTextColor},"a:hover &":{background:e.badgeColorHover}},[`${t}-count-sm`]:{minWidth:x,height:x,fontSize:r,lineHeight:(0,l.unit)(x),borderRadius:w(x).div(2).equal()},[`${t}-multiple-words`]:{padding:`0 ${(0,l.unit)(e.paddingXS)}`,bdi:{unicodeBidi:"plaintext"}},[`${t}-dot`]:{zIndex:e.indicatorZIndex,width:u,minWidth:u,height:u,background:e.badgeColor,borderRadius:"100%",boxShadow:`0 0 0 ${(0,l.unit)(n)} ${e.badgeShadowColor}`},[`${t}-count, ${t}-dot, ${S}-custom-component`]:{position:"absolute",top:0,insetInlineEnd:0,transform:"translate(50%, -50%)",transformOrigin:"100% 0%",[`&${i}-spin`]:{animationName:_,animationDuration:"1s",animationIterationCount:"infinite",animationTimingFunction:"linear"}},[`&${t}-status`]:{lineHeight:"inherit",verticalAlign:"baseline",[`${t}-status-dot`]:{position:"relative",top:-1,display:"inline-block",width:s,height:s,verticalAlign:"middle",borderRadius:"50%"},[`${t}-status-success`]:{backgroundColor:e.colorSuccess},[`${t}-status-processing`]:{overflow:"visible",color:e.colorInfo,backgroundColor:e.colorInfo,borderColor:"currentcolor","&::after":{position:"absolute",top:0,insetInlineStart:0,width:"100%",height:"100%",borderWidth:n,borderStyle:"solid",borderColor:"inherit",borderRadius:"50%",animationName:m,animationDuration:e.badgeProcessingDuration,animationIterationCount:"infinite",animationTimingFunction:"ease-in-out",content:'""'}},[`${t}-status-default`]:{backgroundColor:e.colorTextPlaceholder},[`${t}-status-error`]:{backgroundColor:e.colorError},[`${t}-status-warning`]:{backgroundColor:e.colorWarning},[`${t}-status-text`]:{marginInlineStart:v,color:e.colorText,fontSize:e.fontSize}}}),j),{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:g,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`${t}-zoom-leave`]:{animationName:f,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`&${t}-not-a-wrapper`]:{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:h,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`${t}-zoom-leave`]:{animationName:b,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`&:not(${t}-status)`]:{verticalAlign:"middle"},[`${S}-custom-component, ${t}-count`]:{transform:"none"},[`${S}-custom-component, ${S}`]:{position:"relative",top:"auto",display:"block",transformOrigin:"50% 50%"}},[S]:{overflow:"hidden",transition:`all ${e.motionDurationMid} ${e.motionEaseOutBack}`,[`${S}-only`]:{position:"relative",display:"inline-block",height:y,transition:`all ${e.motionDurationSlow} ${e.motionEaseOutBack}`,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden",[`> p${S}-only-unit`]:{height:y,margin:0,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden"}},[`${S}-symbol`]:{verticalAlign:"top"}},"&-rtl":{direction:"rtl",[`${t}-count, ${t}-dot, ${S}-custom-component`]:{transform:"translate(-50%, -50%)"}}})}})(y(e)),x),w=(0,u.genStyleHooks)(["Badge","Ribbon"],e=>(e=>{let{antCls:t,badgeFontHeight:i,marginXS:o,badgeRibbonOffset:n,calc:a}=e,r=`${t}-ribbon`,s=`${t}-ribbon-wrapper`,u=(0,c.genPresetColor)(e,(e,{darkColor:t})=>({[`&${r}-color-${e}`]:{background:t,color:t}}));return{[s]:{position:"relative"},[r]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,d.resetComponent)(e)),{position:"absolute",top:o,padding:`0 ${(0,l.unit)(e.paddingXS)}`,color:e.colorPrimary,lineHeight:(0,l.unit)(i),whiteSpace:"nowrap",backgroundColor:e.colorPrimary,borderRadius:e.borderRadiusSM,[`${r}-text`]:{color:e.badgeTextColor},[`${r}-corner`]:{position:"absolute",top:"100%",width:n,height:n,color:"currentcolor",border:`${(0,l.unit)(a(n).div(2).equal())} solid`,transform:e.badgeRibbonCornerTransform,transformOrigin:"top",filter:e.badgeRibbonCornerFilter}}),u),{[`&${r}-placement-end`]:{insetInlineEnd:a(n).mul(-1).equal(),borderEndEndRadius:0,[`${r}-corner`]:{insetInlineEnd:0,borderInlineEndColor:"transparent",borderBlockEndColor:"transparent"}},[`&${r}-placement-start`]:{insetInlineStart:a(n).mul(-1).equal(),borderEndStartRadius:0,[`${r}-corner`]:{insetInlineStart:0,borderBlockEndColor:"transparent",borderInlineStartColor:"transparent"}},"&-rtl":{direction:"rtl"}})}})(y(e)),x),S=e=>{let o,{prefixCls:n,value:a,current:r,offset:l=0}=e;return l&&(o={position:"absolute",top:`${l}00%`,left:0}),t.createElement("span",{style:o,className:(0,i.default)(`${n}-only-unit`,{current:r})},a)},j=e=>{let i,o,{prefixCls:n,count:a,value:r}=e,l=Number(r),s=Math.abs(a),[d,c]=t.useState(l),[u,p]=t.useState(s),m=()=>{c(l),p(s)};if(t.useEffect(()=>{let e=setTimeout(m,1e3);return()=>clearTimeout(e)},[l]),d===l||Number.isNaN(l)||Number.isNaN(d))i=[t.createElement(S,Object.assign({},e,{key:l,current:!0}))],o={transition:"none"};else{i=[];let n=l+10,a=[];for(let e=l;e<=n;e+=1)a.push(e);let r=u<s?1:-1,c=a.findIndex(e=>e%10===d);i=(r<0?a.slice(0,c+1):a.slice(c)).map((i,o)=>t.createElement(S,Object.assign({},e,{key:i,value:i%10,offset:r<0?o-c:o,current:o===c}))),o={transform:`translateY(${-function(e,t,i){let o=e,n=0;for(;(o+10)%10!==t;)o+=i,n+=i;return n}(d,l,r)}00%)`}}return t.createElement("span",{className:`${n}-only`,style:o,onTransitionEnd:m},i)};var C=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,o=Object.getOwnPropertySymbols(e);n<o.length;n++)0>t.indexOf(o[n])&&Object.prototype.propertyIsEnumerable.call(e,o[n])&&(i[o[n]]=e[o[n]]);return i};let $=t.forwardRef((e,o)=>{let{prefixCls:n,count:l,className:s,motionClassName:d,style:c,title:u,show:p,component:m="sup",children:g}=e,f=C(e,["prefixCls","count","className","motionClassName","style","title","show","component","children"]),{getPrefixCls:h}=t.useContext(r.ConfigContext),b=h("scroll-number",n),_=Object.assign(Object.assign({},f),{"data-show":p,style:c,className:(0,i.default)(b,s,d),title:u}),y=l;if(l&&Number(l)%1==0){let e=String(l).split("");y=t.createElement("bdi",null,e.map((i,o)=>t.createElement(j,{prefixCls:b,count:Number(l),value:i,key:e.length-o})))}return((null==c?void 0:c.borderColor)&&(_.style=Object.assign(Object.assign({},c),{boxShadow:`0 0 0 1px ${c.borderColor} inset`})),g)?(0,a.cloneElement)(g,e=>({className:(0,i.default)(`${b}-custom-component`,null==e?void 0:e.className,d)})):t.createElement(m,Object.assign({},_,{ref:o}),y)});var k=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,o=Object.getOwnPropertySymbols(e);n<o.length;n++)0>t.indexOf(o[n])&&Object.prototype.propertyIsEnumerable.call(e,o[n])&&(i[o[n]]=e[o[n]]);return i};let O=t.forwardRef((e,l)=>{var s,d,c,u,p;let{prefixCls:m,scrollNumberPrefixCls:g,children:f,status:h,text:b,color:_,count:y=null,overflowCount:x=99,dot:w=!1,size:S="default",title:j,offset:C,style:O,className:E,rootClassName:I,classNames:N,styles:T,showZero:R=!1}=e,z=k(e,["prefixCls","scrollNumberPrefixCls","children","status","text","color","count","overflowCount","dot","size","title","offset","style","className","rootClassName","classNames","styles","showZero"]),{getPrefixCls:A,direction:P,badge:L}=t.useContext(r.ConfigContext),M=A("badge",m),[B,D,H]=v(M),W=y>x?`${x}+`:y,U="0"===W||0===W||"0"===b||0===b,F=null===y||U&&!R,V=(null!=h||null!=_)&&F,G=null!=h||!U,K=w&&!U,q=K?"":W,Y=(0,t.useMemo)(()=>((null==q||""===q)&&(null==b||""===b)||U&&!R)&&!K,[q,U,R,K,b]),Z=(0,t.useRef)(y);Y||(Z.current=y);let J=Z.current,X=(0,t.useRef)(q);Y||(X.current=q);let Q=X.current,ee=(0,t.useRef)(K);Y||(ee.current=K);let et=(0,t.useMemo)(()=>{if(!C)return Object.assign(Object.assign({},null==L?void 0:L.style),O);let e={marginTop:C[1]};return"rtl"===P?e.left=Number.parseInt(C[0],10):e.right=-Number.parseInt(C[0],10),Object.assign(Object.assign(Object.assign({},e),null==L?void 0:L.style),O)},[P,C,O,null==L?void 0:L.style]),ei=null!=j?j:"string"==typeof J||"number"==typeof J?J:void 0,eo=!Y&&(0===b?R:!!b&&!0!==b),en=eo?t.createElement("span",{className:`${M}-status-text`},b):null,ea=J&&"object"==typeof J?(0,a.cloneElement)(J,e=>({style:Object.assign(Object.assign({},et),e.style)})):void 0,er=(0,n.isPresetColor)(_,!1),el=(0,i.default)(null==N?void 0:N.indicator,null==(s=null==L?void 0:L.classNames)?void 0:s.indicator,{[`${M}-status-dot`]:V,[`${M}-status-${h}`]:!!h,[`${M}-color-${_}`]:er}),es={};_&&!er&&(es.color=_,es.background=_);let ed=(0,i.default)(M,{[`${M}-status`]:V,[`${M}-not-a-wrapper`]:!f,[`${M}-rtl`]:"rtl"===P},E,I,null==L?void 0:L.className,null==(d=null==L?void 0:L.classNames)?void 0:d.root,null==N?void 0:N.root,D,H);if(!f&&V&&(b||G||!F)){let e=et.color;return B(t.createElement("span",Object.assign({},z,{className:ed,style:Object.assign(Object.assign(Object.assign({},null==T?void 0:T.root),null==(c=null==L?void 0:L.styles)?void 0:c.root),et)}),t.createElement("span",{className:el,style:Object.assign(Object.assign(Object.assign({},null==T?void 0:T.indicator),null==(u=null==L?void 0:L.styles)?void 0:u.indicator),es)}),eo&&t.createElement("span",{style:{color:e},className:`${M}-status-text`},b)))}return B(t.createElement("span",Object.assign({ref:l},z,{className:ed,style:Object.assign(Object.assign({},null==(p=null==L?void 0:L.styles)?void 0:p.root),null==T?void 0:T.root)}),f,t.createElement(o.default,{visible:!Y,motionName:`${M}-zoom`,motionAppear:!1,motionDeadline:1e3},({className:e})=>{var o,n;let a=A("scroll-number",g),r=ee.current,l=(0,i.default)(null==N?void 0:N.indicator,null==(o=null==L?void 0:L.classNames)?void 0:o.indicator,{[`${M}-dot`]:r,[`${M}-count`]:!r,[`${M}-count-sm`]:"small"===S,[`${M}-multiple-words`]:!r&&Q&&Q.toString().length>1,[`${M}-status-${h}`]:!!h,[`${M}-color-${_}`]:er}),s=Object.assign(Object.assign(Object.assign({},null==T?void 0:T.indicator),null==(n=null==L?void 0:L.styles)?void 0:n.indicator),et);return _&&!er&&((s=s||{}).background=_),t.createElement($,{prefixCls:a,show:!Y,motionClassName:e,className:l,count:Q,title:ei,style:s,key:"scrollNumber"},ea)}),en))});O.Ribbon=e=>{let{className:o,prefixCls:a,style:l,color:s,children:d,text:c,placement:u="end",rootClassName:p}=e,{getPrefixCls:m,direction:g}=t.useContext(r.ConfigContext),f=m("ribbon",a),h=`${f}-wrapper`,[b,_,y]=w(f,h),x=(0,n.isPresetColor)(s,!1),v=(0,i.default)(f,`${f}-placement-${u}`,{[`${f}-rtl`]:"rtl"===g,[`${f}-color-${s}`]:x},o),S={},j={};return s&&!x&&(S.background=s,j.color=s),b(t.createElement("div",{className:(0,i.default)(h,p,_,y)},d,t.createElement("div",{className:(0,i.default)(v,_),style:Object.assign(Object.assign({},S),l)},t.createElement("span",{className:`${f}-text`},c),t.createElement("div",{className:`${f}-corner`,style:j}))))},e.s(["Badge",0,O],906579)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["CrownOutlined",0,a],100486)},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(602869);let n=(0,i.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:a})=>{let[r,l]=(0,i.useState)(null),[s,d]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,o.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let e=await i.json();e.values?.logo_url&&l(e.values.logo_url),e.values?.favicon_url&&d(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,i.useEffect)(()=>{if(s){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=s});else{let e=document.createElement("link");e.rel="icon",e.href=s,document.head.appendChild(e)}}},[s]),(0,t.jsx)(n.Provider,{value:{logoUrl:r,setLogoUrl:l,faviconUrl:s,setFaviconUrl:d},children:e})},"useTheme",0,()=>{let e=(0,i.useContext)(n);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},371401,e=>{"use strict";var t=e.i(115571),i=e.i(271645);function o(e){let i=t=>{"disableUsageIndicator"===t.key&&e()},o=t=>{let{key:i}=t.detail;"disableUsageIndicator"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t.LOCAL_STORAGE_EVENT,o),()=>{window.removeEventListener("storage",i),window.removeEventListener(t.LOCAL_STORAGE_EVENT,o)}}function n(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}function a(){return(0,i.useSyncExternalStore)(o,n)}e.s(["useDisableUsageIndicator",()=>a])},115571,e=>{"use strict";let t="local-storage-change";function i(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function o(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function n(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function a(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>i,"getLocalStorageItem",()=>o,"removeLocalStorageItem",()=>a,"setLocalStorageItem",()=>n])},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},62478,e=>{"use strict";var t=e.i(602869);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},592392,e=>{"use strict";var t=e.i(62478),i=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("proxySettings"),n={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};function a(e){let{data:a}=(0,i.useQuery)({queryKey:[...o.all,e],queryFn:()=>(0,t.fetchProxySettings)(e),enabled:!!e});return a??n}e.s(["default",()=>a])}]);