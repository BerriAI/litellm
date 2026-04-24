(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,62478,e=>{"use strict";var t=e.i(764205);let r=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,r])},292270,e=>{"use strict";let t=(0,e.i(475254).default)("log-out",[["path",{d:"m16 17 5-5-5-5",key:"1bji2h"}],["path",{d:"M21 12H9",key:"dn1m92"}],["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}]]);e.s(["LogOut",()=>t],292270)},818581,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"useMergedRef",{enumerable:!0,get:function(){return i}});let n=e.r(271645);function i(e,t){let r=(0,n.useRef)(null),i=(0,n.useRef)(null);return(0,n.useCallback)(n=>{if(null===n){let e=r.current;e&&(r.current=null,e());let t=i.current;t&&(i.current=null,t())}else e&&(r.current=a(e,n)),t&&(i.current=a(t,n))},[e,t])}function a(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let r=e(t);return"function"==typeof r?r:()=>e(null)}}("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},190272,785913,e=>{"use strict";var t,r,n=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r);let a={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(n).includes(e)){let t=a[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:n,apiKey:a,inputMessage:o,chatHistory:s,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedMCPServers:p,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:x,selectedSdk:_,proxySettings:y}=e,b="session"===r?n:a,w=window.location.origin,j=y?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?w=j:y?.PROXY_BASE_URL&&(w=y.PROXY_BASE_URL);let v=o||"Your prompt here",k=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),d.length>0&&(E.vector_stores=d),c.length>0&&(E.guardrails=c),u.length>0&&(E.policies=u);let C=x||"your-model-name",I="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${w}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${w}"
)`;switch(h){case i.CHAT:{let e=Object.keys(E).length>0,r="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let n=S.length>0?S:[{role:"user",content:v}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(n,null,4)}${r}
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
#                     "text": "${k}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${r}
# )
# print(response_with_file)
`;break}case i.RESPONSES:{let e=Object.keys(E).length>0,r="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let n=S.length>0?S:[{role:"user",content:v}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(n,null,4)}${r}
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
#                 {"type": "input_text", "text": "${k}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case i.IMAGE:t="azure"===_?`
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
prompt = "${k}"

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
`;break;case i.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${k}"

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
prompt = "${k}"

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
`;break;case i.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${o||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${o?`,
	prompt="${o.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${o||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),n=e.i(871689),i=e.i(174886),a=e.i(643531),o=e.i(221345);let s=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,s,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:l})=>{let d,[c,u]=(0,r.useState)("overview"),[p,m]=(0,r.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},f="github"===(d=e.source).source&&d.repo?`https://github.com/${d.repo}`:"git-subdir"===d.source&&d.url?d.path?`${d.url}/tree/main/${d.path}`:d.url:"url"===d.source&&d.url?d.url:null,h=s(e),x=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:l,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(n.ArrowLeft,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>u(e.key),style:{padding:"12px 20px",fontSize:14,color:c===e.key?"#1a73e8":"#5f6368",borderBottom:c===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:c===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===c&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:x.map((e,r)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),f&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:f,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[f.replace("https://",""),(0,t.jsx)(o.Link,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(h,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===p?(0,t.jsx)(a.Check,{}):(0,t.jsx)(i.Copy,{}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:h})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>u("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{g(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===p?(0,t.jsx)(a.Check,{}):(0,t.jsx)(i.Copy,{}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},221345,e=>{"use strict";let t=(0,e.i(475254).default)("link",[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]]);e.s(["Link",()=>t],221345)},115571,e=>{"use strict";let t="local-storage-change";function r(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function n(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function i(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function a(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>r,"getLocalStorageItem",()=>n,"removeLocalStorageItem",()=>a,"setLocalStorageItem",()=>i])},972518,e=>{"use strict";let t=(0,e.i(475254).default)("panel-left-close",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m16 15-3-3 3-3",key:"14y99z"}]]);e.s(["PanelLeftClose",()=>t],972518)},371401,e=>{"use strict";var t=e.i(115571),r=e.i(271645);function n(e){let r=t=>{"disableUsageIndicator"===t.key&&e()},n=t=>{let{key:r}=t.detail;"disableUsageIndicator"===r&&e()};return window.addEventListener("storage",r),window.addEventListener(t.LOCAL_STORAGE_EVENT,n),()=>{window.removeEventListener("storage",r),window.removeEventListener(t.LOCAL_STORAGE_EVENT,n)}}function i(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}function a(){return(0,r.useSyncExternalStore)(n,i)}e.s(["useDisableUsageIndicator",()=>a])},283713,e=>{"use strict";var t=e.i(271645),r=e.i(764205),n=e.i(612256);let i="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,n.useUIConfig)(),a=e?.is_control_plane??!1,o=e?.workers??[],[s,l]=(0,t.useState)(()=>localStorage.getItem(i));(0,t.useEffect)(()=>{if(!s||0===o.length)return;let e=o.find(e=>e.worker_id===s);e&&(0,r.switchToWorkerUrl)(e.url)},[s,o]);let d=o.find(e=>e.worker_id===s)??null,c=(0,t.useCallback)(e=>{let t=o.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(i,e),(0,r.switchToWorkerUrl)(t.url))},[o]);return{isControlPlane:a,workers:o,selectedWorkerId:s,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(i),(0,r.switchToWorkerUrl)(null)},[])}}])},275144,e=>{"use strict";var t=e.i(843476),r=e.i(271645),n=e.i(764205);let i=(0,r.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:a})=>{let[o,s]=(0,r.useState)(null),[l,d]=(0,r.useState)(null);return(0,r.useEffect)(()=>{(async()=>{try{let e=(0,n.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",r=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(r.ok){let e=await r.json();e.values?.logo_url&&s(e.values.logo_url),e.values?.favicon_url&&d(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,r.useEffect)(()=>{if(l){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=l});else{let e=document.createElement("link");e.rel="icon",e.href=l,document.head.appendChild(e)}}},[l]),(0,t.jsx)(i.Provider,{value:{logoUrl:o,setLogoUrl:s,faviconUrl:l,setFaviconUrl:d},children:e})},"useTheme",0,()=>{let e=(0,r.useContext)(i);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},998183,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var n={assign:function(){return l},searchParamsToUrlQuery:function(){return a},urlQueryToSearchParams:function(){return s}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});function a(e){let t={};for(let[r,n]of e.entries()){let e=t[r];void 0===e?t[r]=n:Array.isArray(e)?e.push(n):t[r]=[e,n]}return t}function o(e){return"string"==typeof e?e:("number"!=typeof e||isNaN(e))&&"boolean"!=typeof e?"":String(e)}function s(e){let t=new URLSearchParams;for(let[r,n]of Object.entries(e))if(Array.isArray(n))for(let e of n)t.append(r,o(e));else t.set(r,o(n));return t}function l(e,...t){for(let r of t){for(let t of r.keys())e.delete(t);for(let[t,n]of r.entries())e.append(t,n)}return e}},195057,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var n={formatUrl:function(){return s},formatWithValidation:function(){return d},urlObjectKeys:function(){return l}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=e.r(151836)._(e.r(998183)),o=/https?|ftp|gopher|file/;function s(e){let{auth:t,hostname:r}=e,n=e.protocol||"",i=e.pathname||"",s=e.hash||"",l=e.query||"",d=!1;t=t?encodeURIComponent(t).replace(/%3A/i,":")+"@":"",e.host?d=t+e.host:r&&(d=t+(~r.indexOf(":")?`[${r}]`:r),e.port&&(d+=":"+e.port)),l&&"object"==typeof l&&(l=String(a.urlQueryToSearchParams(l)));let c=e.search||l&&`?${l}`||"";return n&&!n.endsWith(":")&&(n+=":"),e.slashes||(!n||o.test(n))&&!1!==d?(d="//"+(d||""),i&&"/"!==i[0]&&(i="/"+i)):d||(d=""),s&&"#"!==s[0]&&(s="#"+s),c&&"?"!==c[0]&&(c="?"+c),i=i.replace(/[?#]/g,encodeURIComponent),c=c.replace("#","%23"),`${n}${d}${i}${c}${s}`}let l=["auth","hash","host","hostname","href","path","pathname","port","protocol","query","search","slashes"];function d(e){return s(e)}},718967,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var n={DecodeError:function(){return x},MiddlewareNotFoundError:function(){return w},MissingStaticPage:function(){return b},NormalizeError:function(){return _},PageNotFoundError:function(){return y},SP:function(){return f},ST:function(){return h},WEB_VITALS:function(){return a},execOnce:function(){return o},getDisplayName:function(){return u},getLocationOrigin:function(){return d},getURL:function(){return c},isAbsoluteUrl:function(){return l},isResSent:function(){return p},loadGetInitialProps:function(){return g},normalizeRepeatedSlashes:function(){return m},stringifyError:function(){return j}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=["CLS","FCP","FID","INP","LCP","TTFB"];function o(e){let t,r=!1;return(...n)=>(r||(r=!0,t=e(...n)),t)}let s=/^[a-zA-Z][a-zA-Z\d+\-.]*?:/,l=e=>s.test(e);function d(){let{protocol:e,hostname:t,port:r}=window.location;return`${e}//${t}${r?":"+r:""}`}function c(){let{href:e}=window.location,t=d();return e.substring(t.length)}function u(e){return"string"==typeof e?e:e.displayName||e.name||"Unknown"}function p(e){return e.finished||e.headersSent}function m(e){let t=e.split("?");return t[0].replace(/\\/g,"/").replace(/\/\/+/g,"/")+(t[1]?`?${t.slice(1).join("?")}`:"")}async function g(e,t){let r=t.res||t.ctx&&t.ctx.res;if(!e.getInitialProps)return t.ctx&&t.Component?{pageProps:await g(t.Component,t.ctx)}:{};let n=await e.getInitialProps(t);if(r&&p(r))return n;if(!n)throw Object.defineProperty(Error(`"${u(e)}.getInitialProps()" should resolve to an object. But found "${n}" instead.`),"__NEXT_ERROR_CODE",{value:"E394",enumerable:!1,configurable:!0});return n}let f="u">typeof performance,h=f&&["mark","measure","getEntriesByName"].every(e=>"function"==typeof performance[e]);class x extends Error{}class _ extends Error{}class y extends Error{constructor(e){super(),this.code="ENOENT",this.name="PageNotFoundError",this.message=`Cannot find module for page: ${e}`}}class b extends Error{constructor(e,t){super(),this.message=`Failed to load static file for page: ${e} ${t}`}}class w extends Error{constructor(){super(),this.code="ENOENT",this.message="Cannot find the middleware module"}}function j(e){return JSON.stringify({message:e.message,stack:e.stack})}},573668,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"isLocalURL",{enumerable:!0,get:function(){return a}});let n=e.r(718967),i=e.r(652817);function a(e){if(!(0,n.isAbsoluteUrl)(e))return!0;try{let t=(0,n.getLocationOrigin)(),r=new URL(e,t);return r.origin===t&&(0,i.hasBasePath)(r.pathname)}catch(e){return!1}}},284508,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"errorOnce",{enumerable:!0,get:function(){return n}});let n=e=>{}},522016,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var n={default:function(){return x},useLinkStatus:function(){return y}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=e.r(151836),o=e.r(843476),s=a._(e.r(271645)),l=e.r(195057),d=e.r(8372),c=e.r(818581),u=e.r(718967),p=e.r(405550);e.r(233525);let m=e.r(91949),g=e.r(573668),f=e.r(509396);function h(e){return"string"==typeof e?e:(0,l.formatUrl)(e)}function x(t){var r;let n,i,a,[l,x]=(0,s.useOptimistic)(m.IDLE_LINK_STATUS),y=(0,s.useRef)(null),{href:b,as:w,children:j,prefetch:v=null,passHref:k,replace:S,shallow:E,scroll:C,onClick:I,onMouseEnter:N,onTouchStart:T,legacyBehavior:L=!1,onNavigate:P,ref:O,unstable_dynamicOnHover:A,...M}=t;n=j,L&&("string"==typeof n||"number"==typeof n)&&(n=(0,o.jsx)("a",{children:n}));let $=s.default.useContext(d.AppRouterContext),R=!1!==v,B=!1!==v?null===(r=v)||"auto"===r?f.FetchStrategy.PPR:f.FetchStrategy.Full:f.FetchStrategy.PPR,{href:D,as:U}=s.default.useMemo(()=>{let e=h(b);return{href:e,as:w?h(w):e}},[b,w]);if(L){if(n?.$$typeof===Symbol.for("react.lazy"))throw Object.defineProperty(Error("`<Link legacyBehavior>` received a direct child that is either a Server Component, or JSX that was loaded with React.lazy(). This is not supported. Either remove legacyBehavior, or make the direct child a Client Component that renders the Link's `<a>` tag."),"__NEXT_ERROR_CODE",{value:"E863",enumerable:!1,configurable:!0});i=s.default.Children.only(n)}let z=L?i&&"object"==typeof i&&i.ref:O,F=s.default.useCallback(e=>(null!==$&&(y.current=(0,m.mountLinkInstance)(e,D,$,B,R,x)),()=>{y.current&&((0,m.unmountLinkForCurrentNavigation)(y.current),y.current=null),(0,m.unmountPrefetchableInstance)(e)}),[R,D,$,B,x]),H={ref:(0,c.useMergedRef)(F,z),onClick(t){L||"function"!=typeof I||I(t),L&&i.props&&"function"==typeof i.props.onClick&&i.props.onClick(t),!$||t.defaultPrevented||function(t,r,n,i,a,o,l){if("u">typeof window){let d,{nodeName:c}=t.currentTarget;if("A"===c.toUpperCase()&&((d=t.currentTarget.getAttribute("target"))&&"_self"!==d||t.metaKey||t.ctrlKey||t.shiftKey||t.altKey||t.nativeEvent&&2===t.nativeEvent.which)||t.currentTarget.hasAttribute("download"))return;if(!(0,g.isLocalURL)(r)){a&&(t.preventDefault(),location.replace(r));return}if(t.preventDefault(),l){let e=!1;if(l({preventDefault:()=>{e=!0}}),e)return}let{dispatchNavigateAction:u}=e.r(699781);s.default.startTransition(()=>{u(n||r,a?"replace":"push",o??!0,i.current)})}}(t,D,U,y,S,C,P)},onMouseEnter(e){L||"function"!=typeof N||N(e),L&&i.props&&"function"==typeof i.props.onMouseEnter&&i.props.onMouseEnter(e),$&&R&&(0,m.onNavigationIntent)(e.currentTarget,!0===A)},onTouchStart:function(e){L||"function"!=typeof T||T(e),L&&i.props&&"function"==typeof i.props.onTouchStart&&i.props.onTouchStart(e),$&&R&&(0,m.onNavigationIntent)(e.currentTarget,!0===A)}};return(0,u.isAbsoluteUrl)(U)?H.href=U:L&&!k&&("a"!==i.type||"href"in i.props)||(H.href=(0,p.addBasePath)(U)),a=L?s.default.cloneElement(i,H):(0,o.jsx)("a",{...M,...H,children:n}),(0,o.jsx)(_.Provider,{value:l,children:a})}e.r(284508);let _=(0,s.createContext)(m.IDLE_LINK_STATUS),y=()=>(0,s.useContext)(_);("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},402874,521323,636772,e=>{"use strict";var t=e.i(843476),r=e.i(764205),n=e.i(266027);let i=(0,e.i(243652).createQueryKeys)("healthReadiness"),a=async()=>{let e=(0,r.getProxyBaseUrl)(),t=await fetch(`${e}/health/readiness`);if(!t.ok)throw Error(`Failed to fetch health readiness: ${t.statusText}`);return t.json()},o=()=>(0,n.useQuery)({queryKey:i.detail("readiness"),queryFn:a,staleTime:3e5});e.s(["useHealthReadiness",0,o],521323);var s=e.i(115571),l=e.i(271645);function d(e){let t=t=>{"disableBouncingIcon"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableBouncingIcon"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(s.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(s.LOCAL_STORAGE_EVENT,r)}}function c(){return"true"===(0,s.getLocalStorageItem)("disableBouncingIcon")}function u(){return(0,l.useSyncExternalStore)(d,c)}var p=e.i(275144),m=e.i(268004),g=e.i(321836),f=e.i(62478),h=e.i(487486),x=e.i(519455),_=e.i(699375),y=e.i(475254);let b=(0,y.default)("menu",[["path",{d:"M4 12h16",key:"1lakjw"}],["path",{d:"M4 18h16",key:"19g7jn"}],["path",{d:"M4 6h16",key:"1o0s65"}]]);(0,y.default)("moon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]);var w=e.i(972518);(0,y.default)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);var j=e.i(522016);async function v(){let e=(0,r.getProxyBaseUrl)(),t=await fetch(`${e}/public/litellm_blog_posts`);if(!t.ok)throw Error(`Failed to fetch blog posts: ${t.statusText}`);return t.json()}function k(e){let t=t=>{"disableBlogPosts"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableBlogPosts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(s.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(s.LOCAL_STORAGE_EVENT,r)}}function S(){return"true"===(0,s.getLocalStorageItem)("disableBlogPosts")}function E(){return(0,l.useSyncExternalStore)(k,S)}var C=e.i(755146),I=e.i(531278);let N=()=>{let e=E(),{data:r,isLoading:i,isError:a,refetch:o}=(0,n.useQuery)({queryKey:["blogPosts"],queryFn:v,staleTime:36e5,retry:1,retryDelay:0});return e?null:(0,t.jsxs)(C.DropdownMenu,{children:[(0,t.jsx)(C.DropdownMenuTrigger,{asChild:!0,children:(0,t.jsx)(x.Button,{variant:"ghost",children:"Blog"})}),(0,t.jsx)(C.DropdownMenuContent,{align:"end",className:"w-[420px] p-1",children:i?(0,t.jsx)("div",{className:"flex items-center justify-center py-4",children:(0,t.jsx)(I.Loader2,{className:"h-4 w-4 animate-spin text-muted-foreground"})}):a?(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2 p-3",children:[(0,t.jsx)("span",{className:"text-destructive text-sm",children:"Failed to load posts"}),(0,t.jsx)(x.Button,{size:"sm",variant:"outline",onClick:()=>o(),children:"Retry"})]}):r&&0!==r.posts.length?(0,t.jsxs)(t.Fragment,{children:[r.posts.slice(0,5).map(e=>(0,t.jsx)(C.DropdownMenuItem,{asChild:!0,className:"cursor-pointer",children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",className:"block w-full px-2 py-2",children:[(0,t.jsx)("h5",{className:"font-semibold text-sm mb-0.5",children:e.title}),(0,t.jsx)("div",{className:"text-[11px] text-muted-foreground mb-0.5",children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"text-xs line-clamp-2 text-muted-foreground m-0",children:e.description})]})},e.url)),(0,t.jsx)(C.DropdownMenuSeparator,{}),(0,t.jsx)(C.DropdownMenuItem,{asChild:!0,children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",className:"block w-full px-2 py-2 text-sm",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-3 py-3 text-muted-foreground text-sm",children:"No posts available"})})]})};function T(e){let t=t=>{"disableShowPrompts"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableShowPrompts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(s.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(s.LOCAL_STORAGE_EVENT,r)}}function L(){return"true"===(0,s.getLocalStorageItem)("disableShowPrompts")}function P(){return(0,l.useSyncExternalStore)(T,L)}e.s(["useDisableShowPrompts",()=>P],636772);let O=(0,y.default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]),A=(0,y.default)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]]),M=()=>P()?null:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(x.Button,{asChild:!0,variant:"outline",className:"shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow",children:(0,t.jsxs)("a",{href:"https://www.litellm.ai/support",target:"_blank",rel:"noopener noreferrer",children:[(0,t.jsx)(A,{className:"h-4 w-4"}),"Join Slack"]})}),(0,t.jsx)(x.Button,{asChild:!0,variant:"outline",className:"shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow",children:(0,t.jsxs)("a",{href:"https://github.com/BerriAI/litellm",target:"_blank",rel:"noopener noreferrer",children:[(0,t.jsx)(O,{className:"h-4 w-4"}),"Star us on GitHub"]})})]});var $=e.i(135214),R=e.i(371401),B=e.i(746798),D=e.i(664659),U=e.i(243553),z=e.i(292270),F=e.i(263488),H=e.i(98919),G=e.i(284614);let V=({onLogout:e})=>{let{userId:r,userEmail:n,userRole:i,premiumUser:a}=(0,$.default)(),o=P(),d=(0,R.useDisableUsageIndicator)(),c=E(),p=u(),[m,g]=(0,l.useState)(!1);(0,l.useEffect)(()=>{g("true"===(0,s.getLocalStorageItem)("disableShowNewBadge"))},[]);let f=(e,t)=>{t?(0,s.setLocalStorageItem)(e,"true"):(0,s.removeLocalStorageItem)(e),(0,s.emitLocalStorageChange)(e)},y=({label:e,checked:r,onChange:n,ariaLabel:i})=>(0,t.jsxs)("div",{className:"flex items-center justify-between w-full",children:[(0,t.jsx)("span",{className:"text-muted-foreground text-sm",children:e}),(0,t.jsx)(_.Switch,{checked:r,onCheckedChange:n,"aria-label":i??e})]});return(0,t.jsxs)(C.DropdownMenu,{children:[(0,t.jsx)(C.DropdownMenuTrigger,{asChild:!0,children:(0,t.jsxs)(x.Button,{variant:"ghost",className:"gap-2",children:[(0,t.jsx)(G.User,{className:"h-4 w-4"}),(0,t.jsx)("span",{className:"text-sm",children:"User"}),(0,t.jsx)(D.ChevronDown,{className:"h-3 w-3"})]})}),(0,t.jsxs)(C.DropdownMenuContent,{align:"end",className:"w-72 p-3",children:[(0,t.jsxs)("div",{className:"flex flex-col gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(F.Mail,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground truncate max-w-[160px]",children:n||"-"})]}),a?(0,t.jsxs)(h.Badge,{className:"bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 gap-1",children:[(0,t.jsx)(U.Crown,{className:"h-3 w-3"}),"Premium"]}):(0,t.jsx)(B.TooltipProvider,{children:(0,t.jsxs)(B.Tooltip,{children:[(0,t.jsx)(B.TooltipTrigger,{asChild:!0,children:(0,t.jsxs)(h.Badge,{variant:"secondary",className:"gap-1",children:[(0,t.jsx)(U.Crown,{className:"h-3 w-3"}),"Standard"]})}),(0,t.jsx)(B.TooltipContent,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(C.DropdownMenuSeparator,{}),(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(G.User,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsx)("span",{className:"text-sm truncate max-w-[150px]",title:r||"-",children:r||"-"})]}),(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(H.Shield,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{className:"text-sm",children:i})]}),(0,t.jsx)(C.DropdownMenuSeparator,{}),(0,t.jsx)(y,{label:"Hide New Feature Indicators",checked:m,onChange:e=>{g(e),f("disableShowNewBadge",e)},ariaLabel:"Toggle hide new feature indicators"}),(0,t.jsx)(y,{label:"Hide All Prompts",checked:o,onChange:e=>f("disableShowPrompts",e),ariaLabel:"Toggle hide all prompts"}),(0,t.jsx)(y,{label:"Hide Usage Indicator",checked:d,onChange:e=>f("disableUsageIndicator",e),ariaLabel:"Toggle hide usage indicator"}),(0,t.jsx)(y,{label:"Hide Blog Posts",checked:c,onChange:e=>f("disableBlogPosts",e),ariaLabel:"Toggle hide blog posts"}),(0,t.jsx)(y,{label:"Hide Bouncing Icon",checked:p,onChange:e=>f("disableBouncingIcon",e),ariaLabel:"Toggle hide bouncing icon"})]}),(0,t.jsx)(C.DropdownMenuSeparator,{}),(0,t.jsxs)(C.DropdownMenuItem,{onClick:e,children:[(0,t.jsx)(z.LogOut,{className:"h-4 w-4"}),"Logout"]})]})]})};var W=e.i(967489),q=e.i(283713);let K=({onWorkerSwitch:e})=>{let{isControlPlane:r,selectedWorker:n,workers:i}=(0,q.useWorker)();return r&&n?(0,t.jsxs)(W.Select,{value:n.worker_id,onValueChange:e,children:[(0,t.jsx)(W.SelectTrigger,{className:"min-w-[180px]",children:(0,t.jsx)(W.SelectValue,{})}),(0,t.jsx)(W.SelectContent,{children:i.map(e=>(0,t.jsx)(W.SelectItem,{value:e.worker_id,disabled:e.worker_id===n.worker_id,children:e.name},e.worker_id))})]}):null};e.s(["default",0,({userID:e,userEmail:n,userRole:i,premiumUser:a,proxySettings:s,setProxySettings:d,accessToken:c,isPublicPage:_=!1,sidebarCollapsed:y=!1,onToggleSidebar:v,isDarkMode:k,toggleDarkMode:S})=>{let E=(0,r.getProxyBaseUrl)(),[C,I]=(0,l.useState)(""),{logoUrl:T}=(0,p.useTheme)(),{data:L}=o(),P=L?.litellm_version,O=u(),A=T||`${E}/get_image`;return(0,l.useEffect)(()=>{(async()=>{if(c){let e=await (0,f.fetchProxySettings)(c);console.log("response from fetchProxySettings",e),e&&d(e)}})()},[c]),(0,l.useEffect)(()=>{I(s?.PROXY_LOGOUT_URL||"")},[s]),(0,t.jsx)("nav",{className:"bg-background border-b border-border sticky top-0 z-10",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex items-center h-14 px-4",children:[(0,t.jsxs)("div",{className:"flex items-center flex-shrink-0",children:[v&&(0,t.jsx)("button",{onClick:v,className:"flex items-center justify-center w-10 h-10 mr-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors",title:y?"Expand sidebar":"Collapse sidebar","aria-label":y?"Expand sidebar":"Collapse sidebar",children:y?(0,t.jsx)(b,{className:"h-5 w-5"}):(0,t.jsx)(w.PanelLeftClose,{className:"h-5 w-5"})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(j.default,{href:E||"/",className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsx)("div",{className:"h-10 max-w-48 flex items-center justify-center overflow-hidden",children:(0,t.jsx)("img",{src:A,alt:"LiteLLM Brand",className:"max-w-full max-h-full w-auto h-auto object-contain"})})})}),P&&(0,t.jsxs)("div",{className:"relative",children:[!O&&(0,t.jsx)("span",{className:"absolute -top-1 -left-2 text-lg animate-bounce",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(h.Badge,{variant:"outline",className:"relative text-xs font-medium cursor-pointer z-10",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"flex-shrink-0",children:["v",P]})})]})]})]}),(0,t.jsxs)("div",{className:"flex items-center space-x-5 ml-auto",children:[(0,t.jsx)(K,{onWorkerSwitch:e=>{(0,m.clearTokenCookies)(),(0,g.clearStoredReturnUrl)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=`/ui/login?worker=${encodeURIComponent(e)}`}}),(0,t.jsx)(M,{}),!1,(0,t.jsx)(x.Button,{variant:"ghost",asChild:!0,children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",children:"Docs"})}),(0,t.jsx)(N,{}),!_&&(0,t.jsx)(V,{onLogout:()=>{(0,m.clearTokenCookies)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=C}})]})]})})})}],402874)}]);