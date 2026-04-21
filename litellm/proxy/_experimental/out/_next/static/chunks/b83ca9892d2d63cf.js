(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var o=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(o.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["LinkOutlined",0,r],596239)},190272,785913,e=>{"use strict";var t,i,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),o=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>o,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:a,apiKey:r,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:p,selectedGuardrails:d,selectedPolicies:c,selectedMCPServers:g,mcpServers:m,mcpServerToolRestrictions:u,selectedVoice:f,endpointType:_,selectedModel:h,selectedSdk:A,proxySettings:x}=e,b="session"===i?a:r,y=window.location.origin,I=x?.LITELLM_UI_API_DOC_BASE_URL;I&&I.trim()?y=I:x?.PROXY_BASE_URL&&(y=x.PROXY_BASE_URL);let v=n||"Your prompt here",C=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),E=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),O={};l.length>0&&(O.tags=l),p.length>0&&(O.vector_stores=p),d.length>0&&(O.guardrails=d),c.length>0&&(O.policies=c);let T=h||"your-model-name",S="azure"===A?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(_){case o.CHAT:{let e=Object.keys(O).length>0,i="";if(e){let e=JSON.stringify({metadata:O},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=E.length>0?E:[{role:"user",content:v}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${T}",
    messages=${JSON.stringify(a,null,4)}${i}
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
#                     "text": "${C}"
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
`;break}case o.RESPONSES:{let e=Object.keys(O).length>0,i="";if(e){let e=JSON.stringify({metadata:O},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=E.length>0?E:[{role:"user",content:v}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${T}",
    input=${JSON.stringify(a,null,4)}${i}
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
#                 {"type": "input_text", "text": "${C}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case o.IMAGE:t="azure"===A?`
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
prompt = "${C}"

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
`;break;case o.IMAGE_EDITS:t="azure"===A?`
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
prompt = "${C}"

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
prompt = "${C}"

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
`;break;case o.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${T}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case o.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${T}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case o.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${T}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${T}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${t}`}],190272)},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),a=e.i(447566),o=e.i(166406),r=e.i(492030),n=e.i(596239);let s=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,s,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:l})=>{let p,[d,c]=(0,i.useState)("overview"),[g,m]=(0,i.useState)(null),u=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},f="github"===(p=e.source).source&&p.repo?`https://github.com/${p.repo}`:"git-subdir"===p.source&&p.url?p.path?`${p.url}/tree/main/${p.path}`:p.url:"url"===p.source&&p.url?p.url:null,_=s(e),h=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:l,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(a.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>c(e.key),style:{padding:"12px 20px",fontSize:14,color:d===e.key?"#1a73e8":"#5f6368",borderBottom:d===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:d===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===d&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:h.map((e,i)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),f&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:f,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[f.replace("https://",""),(0,t.jsx)(n.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>u(_,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===g?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===g?(0,t.jsx)(r.CheckOutlined,{}):(0,t.jsx)(o.CopyOutlined,{}),"install"===g?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:_})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>c("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{u(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===g?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===g?(0,t.jsx)(r.CheckOutlined,{}):(0,t.jsx)(o.CopyOutlined,{}),"settings"===g?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},94629,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,i],94629)},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},916925,e=>{"use strict";var t,i=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t);let a={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference"},o="../ui/assets/logos/",r={"A2A Agent":`${o}a2a_agent.png`,Ai21:`${o}ai21.svg`,"Ai21 Chat":`${o}ai21.svg`,"AI/ML API":`${o}aiml_api.svg`,"Aiohttp Openai":`${o}openai_small.svg`,Anthropic:`${o}anthropic.svg`,"Anthropic Text":`${o}anthropic.svg`,AssemblyAI:`${o}assemblyai_small.png`,Azure:`${o}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${o}microsoft_azure.svg`,"Azure Text":`${o}microsoft_azure.svg`,Baseten:`${o}baseten.svg`,"Amazon Bedrock":`${o}bedrock.svg`,"Amazon Bedrock Mantle":`${o}bedrock.svg`,"AWS SageMaker":`${o}bedrock.svg`,Cerebras:`${o}cerebras.svg`,Cloudflare:`${o}cloudflare.svg`,Codestral:`${o}mistral.svg`,Cohere:`${o}cohere.svg`,"Cohere Chat":`${o}cohere.svg`,Cometapi:`${o}cometapi.svg`,Cursor:`${o}cursor.svg`,"Databricks (Qwen API)":`${o}databricks.svg`,Dashscope:`${o}dashscope.svg`,Deepseek:`${o}deepseek.svg`,Deepgram:`${o}deepgram.png`,DeepInfra:`${o}deepinfra.png`,ElevenLabs:`${o}elevenlabs.png`,"Fal AI":`${o}fal_ai.jpg`,"Featherless Ai":`${o}featherless.svg`,"Fireworks AI":`${o}fireworks.svg`,Friendliai:`${o}friendli.svg`,"Github Copilot":`${o}github_copilot.svg`,"Google AI Studio":`${o}google.svg`,GradientAI:`${o}gradientai.svg`,Groq:`${o}groq.svg`,vllm:`${o}vllm.png`,Huggingface:`${o}huggingface.svg`,Hyperbolic:`${o}hyperbolic.svg`,Infinity:`${o}infinity.png`,"Jina AI":`${o}jina.png`,"Lambda Ai":`${o}lambda.svg`,"Lm Studio":`${o}lmstudio.svg`,"Meta Llama":`${o}meta_llama.svg`,MiniMax:`${o}minimax.svg`,"Mistral AI":`${o}mistral.svg`,Moonshot:`${o}moonshot.svg`,Morph:`${o}morph.svg`,Nebius:`${o}nebius.svg`,Novita:`${o}novita.svg`,"Nvidia Nim":`${o}nvidia_nim.svg`,Ollama:`${o}ollama.svg`,"Ollama Chat":`${o}ollama.svg`,Oobabooga:`${o}openai_small.svg`,OpenAI:`${o}openai_small.svg`,"Openai Like":`${o}openai_small.svg`,"OpenAI Text Completion":`${o}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${o}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${o}openai_small.svg`,Openrouter:`${o}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${o}oracle.svg`,Perplexity:`${o}perplexity-ai.svg`,Recraft:`${o}recraft.svg`,Replicate:`${o}replicate.svg`,RunwayML:`${o}runwayml.png`,Sagemaker:`${o}bedrock.svg`,Sambanova:`${o}sambanova.svg`,"SAP Generative AI Hub":`${o}sap.png`,Snowflake:`${o}snowflake.svg`,"Text-Completion-Codestral":`${o}mistral.svg`,TogetherAI:`${o}togetherai.svg`,Topaz:`${o}topaz.svg`,Triton:`${o}nvidia_triton.png`,V0:`${o}v0.svg`,"Vercel Ai Gateway":`${o}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${o}google.svg`,"Vertex Ai Beta":`${o}google.svg`,Vllm:`${o}vllm.png`,VolcEngine:`${o}volcengine.png`,"Voyage AI":`${o}voyage.webp`,Watsonx:`${o}watsonx.svg`,"Watsonx Text":`${o}watsonx.svg`,xAI:`${o}xai.svg`,Xinference:`${o}xinference.svg`};e.s(["Providers",()=>i,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:r[e],displayName:e}}let t=Object.keys(a).find(t=>a[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let o=i[t];return{logo:r[o],displayName:o}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let i=a[e];console.log(`Provider mapped to: ${i}`);let o=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let a=t.litellm_provider;(a===i||"string"==typeof a&&a.includes(i))&&o.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&o.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&o.push(e)}))),o},"providerLogoMap",0,r,"provider_map",0,a])},798496,e=>{"use strict";var t=e.i(843476),i=e.i(152990),a=e.i(682830),o=e.i(271645),r=e.i(269200),n=e.i(427612),s=e.i(64848),l=e.i(942232),p=e.i(496020),d=e.i(977572),c=e.i(94629),g=e.i(360820),m=e.i(871943);function u({data:e=[],columns:u,isLoading:f=!1,defaultSorting:_=[],pagination:h,onPaginationChange:A,enablePagination:x=!1,onRowClick:b}){let[y,I]=o.default.useState(_),[v]=o.default.useState("onChange"),[C,E]=o.default.useState({}),[O,T]=o.default.useState({}),S=(0,i.useReactTable)({data:e,columns:u,state:{sorting:y,columnSizing:C,columnVisibility:O,...x&&h?{pagination:h}:{}},columnResizeMode:v,onSortingChange:I,onColumnSizingChange:E,onColumnVisibilityChange:T,...x&&A?{onPaginationChange:A}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),...x?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(r.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:S.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:S.getHeaderGroups().map(e=>(0,t.jsx)(p.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(s.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,i.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(g.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(m.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(c.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(l.TableBody,{children:f?(0,t.jsx)(p.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:u.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):S.getRowModel().rows.length>0?S.getRowModel().rows.map(e=>(0,t.jsx)(p.TableRow,{onClick:()=>b?.(e.original),className:b?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,i.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(p.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:u.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>u])}]);