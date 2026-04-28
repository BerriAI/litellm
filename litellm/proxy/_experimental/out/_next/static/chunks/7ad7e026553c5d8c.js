(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,818581,(e,t,s)=>{"use strict";Object.defineProperty(s,"__esModule",{value:!0}),Object.defineProperty(s,"useMergedRef",{enumerable:!0,get:function(){return r}});let a=e.r(271645);function r(e,t){let s=(0,a.useRef)(null),r=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=s.current;e&&(s.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(s.current=i(e,a)),t&&(r.current=i(t,a))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let s=e(t);return"function"==typeof s?s:()=>e(null)}}("function"==typeof s.default||"object"==typeof s.default&&null!==s.default)&&void 0===s.default.__esModule&&(Object.defineProperty(s.default,"__esModule",{value:!0}),Object.assign(s.default,s),t.exports=s.default)},62478,e=>{"use strict";var t=e.i(764205);let s=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,s])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),i=s.forwardRef(function(e,i){return s.createElement(r.default,(0,t.default)({},e,{ref:i,icon:a}))});e.s(["SafetyOutlined",0,i],602073)},190272,785913,e=>{"use strict";var t,s,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((s={}).IMAGE="image",s.VIDEO="video",s.CHAT="chat",s.RESPONSES="responses",s.IMAGE_EDITS="image_edits",s.ANTHROPIC_MESSAGES="anthropic_messages",s.EMBEDDINGS="embeddings",s.SPEECH="speech",s.TRANSCRIPTION="transcription",s.A2A_AGENTS="a2a_agents",s.MCP="mcp",s.REALTIME="realtime",s);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:s,accessToken:a,apiKey:i,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:m,selectedMCPServers:p,mcpServers:u,mcpServerToolRestrictions:x,selectedVoice:g,endpointType:h,selectedModel:f,selectedSdk:b,proxySettings:_}=e,j="session"===s?a:i,y=window.location.origin,v=_?.LITELLM_UI_API_DOC_BASE_URL;v&&v.trim()?y=v:_?.PROXY_BASE_URL&&(y=_.PROXY_BASE_URL);let N=l||"Your prompt here",w=N.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),T={};o.length>0&&(T.tags=o),c.length>0&&(T.vector_stores=c),d.length>0&&(T.guardrails=d),m.length>0&&(T.policies=m);let C=f||"your-model-name",k="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${j||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${j||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case r.CHAT:{let e=Object.keys(T).length>0,s="";if(e){let e=JSON.stringify({metadata:T},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=S.length>0?S:[{role:"user",content:N}];t=`
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
#     ]${s}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(T).length>0,s="";if(e){let e=JSON.stringify({metadata:T},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=S.length>0?S:[{role:"user",content:N}];t=`
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
#                 {"type": "input_text", "text": "${w}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${s}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===b?`
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
prompt = "${w}"

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
`;break;case r.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${w}"

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
	prompt="${l.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${k}
${t}`}],190272)},652272,209261,e=>{"use strict";var t=e.i(843476),s=e.i(271645),a=e.i(447566),r=e.i(166406),i=e.i(492030),l=e.i(596239);let n=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,n,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:o})=>{let c,[d,m]=(0,s.useState)("overview"),[p,u]=(0,s.useState)(null),x=(e,t)=>{navigator.clipboard.writeText(e),u(t),setTimeout(()=>u(null),2e3)},g="github"===(c=e.source).source&&c.repo?`https://github.com/${c.repo}`:"git-subdir"===c.source&&c.url?c.path?`${c.url}/tree/main/${c.path}`:c.url:"url"===c.source&&c.url?c.url:null,h=n(e),f=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:o,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(a.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>m(e.key),style:{padding:"12px 20px",fontSize:14,color:d===e.key?"#1a73e8":"#5f6368",borderBottom:d===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:d===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===d&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:f.map((e,s)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},s))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),g&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:g,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[g.replace("https://",""),(0,t.jsx)(l.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>x(h,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===p?(0,t.jsx)(i.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:h})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>m("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{x(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===p?(0,t.jsx)(i.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),i=s.forwardRef(function(e,i){return s.createElement(r.default,(0,t.default)({},e,{ref:i,icon:a}))});e.s(["LinkOutlined",0,i],596239)},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},115571,e=>{"use strict";let t="local-storage-change";function s(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function a(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function r(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function i(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>s,"getLocalStorageItem",()=>a,"removeLocalStorageItem",()=>i,"setLocalStorageItem",()=>r])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},798496,e=>{"use strict";var t=e.i(843476),s=e.i(152990),a=e.i(682830),r=e.i(271645),i=e.i(269200),l=e.i(427612),n=e.i(64848),o=e.i(942232),c=e.i(496020),d=e.i(977572),m=e.i(94629),p=e.i(360820),u=e.i(871943);function x({data:e=[],columns:x,isLoading:g=!1,defaultSorting:h=[],pagination:f,onPaginationChange:b,enablePagination:_=!1,onRowClick:j}){let[y,v]=r.default.useState(h),[N]=r.default.useState("onChange"),[w,S]=r.default.useState({}),[T,C]=r.default.useState({}),k=(0,s.useReactTable)({data:e,columns:x,state:{sorting:y,columnSizing:w,columnVisibility:T,..._&&f?{pagination:f}:{}},columnResizeMode:N,onSortingChange:v,onColumnSizingChange:S,onColumnVisibilityChange:C,..._&&b?{onPaginationChange:b}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),..._?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(i.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:k.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(l.TableHead,{children:k.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(n.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,s.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(p.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(u.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(m.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(o.TableBody,{children:g?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:x.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):k.getRowModel().rows.length>0?k.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>j?.(e.original),className:j?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,s.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:x.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>x])},737033,e=>{"use strict";var t=e.i(843476),s=e.i(271645),a=e.i(599724),r=e.i(928685),i=e.i(311451),l=e.i(199133),n=e.i(798496),o=e.i(389083),c=e.i(592968),d=e.i(166406),m=e.i(596239),p=e.i(652272);e.s(["default",0,({skills:e,isLoading:u,isAdmin:x,accessToken:g,publicPage:h=!1,onPublishSuccess:f})=>{let[b,_]=(0,s.useState)(""),[j,y]=(0,s.useState)(void 0),[v,N]=(0,s.useState)(null),w=e.length,S=(0,s.useMemo)(()=>[...new Set(e.map(e=>e.domain).filter(Boolean))],[e]),T=(0,s.useMemo)(()=>[...new Set(e.map(e=>e.namespace).filter(Boolean))],[e]),C=(0,s.useMemo)(()=>{let t=e;if(j&&(t=t.filter(e=>(e.domain||"General")===j)),b.trim()){let e=b.toLowerCase();t=t.filter(t=>t.name.toLowerCase().includes(e)||t.description?.toLowerCase().includes(e)||t.domain?.toLowerCase().includes(e)||t.namespace?.toLowerCase().includes(e)||t.keywords?.some(t=>t.toLowerCase().includes(e)))}return t},[e,b,j]);return v?(0,t.jsx)(p.default,{skill:v,onBack:()=>N(null),isAdmin:x,accessToken:g,onPublishClick:f}):u?(0,t.jsx)("div",{className:"text-center py-16 text-gray-400",children:"Loading skills..."}):(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{className:"grid grid-cols-3 gap-4",children:[(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Total Skills"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:w})]}),(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Namespaces"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:T.length})]}),(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Domains"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:S.length})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-3",children:[(0,t.jsxs)("h3",{className:"text-sm font-semibold text-gray-700",children:["All ",h?"Public ":"","Skills"]}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(l.Select,{placeholder:"All Domains",allowClear:!0,value:j,onChange:e=>y(e),style:{width:160},options:S.map(e=>({label:e,value:e}))}),(0,t.jsx)(i.Input,{prefix:(0,t.jsx)(r.SearchOutlined,{className:"text-gray-400"}),placeholder:"Search by name, namespace, or tag…",value:b,onChange:e=>_(e.target.value),style:{width:280},allowClear:!0})]})]}),(0,t.jsx)(n.ModelDataTable,{columns:((e,s,r=!1)=>[{header:"Skill Name",accessorKey:"name",enableSorting:!0,sortingFn:"alphanumeric",cell:({row:r})=>{let i=r.original;return(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("button",{type:"button",className:"font-medium text-sm cursor-pointer text-blue-600 hover:underline bg-transparent border-none p-0",onClick:()=>e(i),children:i.name}),(0,t.jsx)(c.Tooltip,{title:"Copy skill name",children:(0,t.jsx)(d.CopyOutlined,{onClick:()=>s(i.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 text-xs"})})]}),i.description&&(0,t.jsx)(a.Text,{className:"text-xs text-gray-500 line-clamp-1 md:hidden",children:i.description})]})}},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>(0,t.jsx)(a.Text,{className:"text-xs line-clamp-2",children:e.original.description||"-"})},{header:"Category",accessorKey:"category",enableSorting:!0,cell:({row:e})=>{let s=e.original.category;return s?(0,t.jsx)(o.Badge,{color:"blue",size:"xs",children:s}):(0,t.jsx)(a.Text,{className:"text-xs text-gray-400",children:"-"})}},{header:"Domain",accessorKey:"domain",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(a.Text,{className:"text-xs",children:e.original.domain||"-"})},{header:"Source",accessorKey:"source",enableSorting:!1,cell:({row:e})=>{let s=e.original.source,r=null,i="-";return(s?.source==="github"&&s.repo?(r=`https://github.com/${s.repo}`,i=s.repo):s?.source==="git-subdir"&&s.url?i=(r=s.path?`${s.url}/tree/main/${s.path}`:s.url).replace("https://github.com/",""):s?.source==="url"&&s.url&&(r=s.url,i=s.url.replace(/^https?:\/\//,"")),r)?(0,t.jsxs)("a",{href:r,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 text-xs text-blue-600 hover:underline truncate max-w-[180px]",title:i,children:[(0,t.jsx)("span",{className:"truncate",children:i}),(0,t.jsx)(m.LinkOutlined,{className:"shrink-0",style:{fontSize:10}})]}):(0,t.jsx)(a.Text,{className:"text-xs text-gray-400",children:"-"})}},{header:"Status",accessorKey:"enabled",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Badge,{color:e.original.enabled?"green":"gray",size:"xs",children:e.original.enabled?"Public":"Draft"})}])(e=>N(e),e=>{navigator.clipboard.writeText(e)},h),data:C,isLoading:!1,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-3 text-center",children:(0,t.jsxs)(a.Text,{className:"text-sm text-gray-500",children:["Showing ",C.length," of ",w," skill",1!==w?"s":""]})})]})]})}],737033)},93826,174886,952571,e=>{"use strict";var t=e.i(271645);let s=t.forwardRef(function(e,s){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:s},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"}))});e.s(["SearchIcon",0,s],93826);var a=e.i(991124);e.s(["Copy",()=>a.default],174886);var r=e.i(879664);e.s(["Info",()=>r.default],952571)},976883,e=>{"use strict";var t=e.i(843476),s=e.i(275144),a=e.i(434626),r=e.i(93826),i=e.i(994388),l=e.i(304967),n=e.i(599724),o=e.i(629569),c=e.i(212931),d=e.i(199133),m=e.i(653496),p=e.i(262218),u=e.i(592968),x=e.i(174886),g=e.i(952571),h=e.i(271645),f=e.i(798496),b=e.i(727749),_=e.i(402874),j=e.i(764205),y=e.i(737033),v=e.i(190272),N=e.i(785913),w=e.i(916925);let{TabPane:S}=m.Tabs;e.s(["default",0,({accessToken:e,isEmbedded:T=!1})=>{let C,k,A,M,I,E,z,[L,P]=(0,h.useState)(null),[O,R]=(0,h.useState)(null),[$,D]=(0,h.useState)(null),[H,B]=(0,h.useState)("LiteLLM Gateway"),[K,U]=(0,h.useState)(null),[F,W]=(0,h.useState)(""),[G,V]=(0,h.useState)({}),[q,Y]=(0,h.useState)(!0),[J,X]=(0,h.useState)(!0),[Z,Q]=(0,h.useState)(!0),[ee,et]=(0,h.useState)(""),[es,ea]=(0,h.useState)(""),[er,ei]=(0,h.useState)(""),[el,en]=(0,h.useState)([]),[eo,ec]=(0,h.useState)([]),[ed,em]=(0,h.useState)([]),[ep,eu]=(0,h.useState)([]),[ex,eg]=(0,h.useState)([]),[eh,ef]=(0,h.useState)("I'm alive! ✓"),[eb,e_]=(0,h.useState)(!1),[ej,ey]=(0,h.useState)(!1),[ev,eN]=(0,h.useState)(!1),[ew,eS]=(0,h.useState)(null),[eT,eC]=(0,h.useState)(null),[ek,eA]=(0,h.useState)(null),[eM,eI]=(0,h.useState)({}),[eE,ez]=(0,h.useState)("models"),[eL,eP]=(0,h.useState)([]),[eO,eR]=(0,h.useState)(!1);(0,h.useEffect)(()=>{(async()=>{try{await (0,j.getUiConfig)()}catch(e){console.error("Failed to get UI config:",e)}let e=async()=>{try{Y(!0);let e=await (0,j.modelHubPublicModelsCall)();console.log("ModelHubData:",e),P(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public model data",e),ef("Service unavailable")}finally{Y(!1)}},t=async()=>{try{X(!0);let e=await (0,j.agentHubPublicModelsCall)();console.log("AgentHubData:",e),R(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public agent data",e)}finally{X(!1)}},s=async()=>{try{Q(!0);let e=await (0,j.mcpHubPublicServersCall)();console.log("MCPHubData:",e),D(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public MCP server data",e)}finally{Q(!1)}},a=async()=>{try{eR(!0);let e=await (0,j.skillHubPublicCall)();eP(e.plugins??[])}catch(e){console.error("There was an error fetching the public skill data",e)}finally{eR(!1)}};(async()=>{let e=await (0,j.getPublicModelHubInfo)();console.log("Public Model Hub Info:",e),B(e.docs_title),U(e.custom_docs_description),W(e.litellm_version),V(e.useful_links||{})})(),e(),t(),s(),a()})()},[]),(0,h.useEffect)(()=>{},[ee,el,eo,ed]);let e$=(0,h.useMemo)(()=>{if(!L||!Array.isArray(L))return[];let e=L;if(ee.trim()){let t=ee.toLowerCase(),s=t.split(/\s+/),a=L.filter(e=>{let a=e.model_group.toLowerCase();return!!a.includes(t)||s.every(e=>a.includes(e))});a.length>0&&(e=a.sort((e,s)=>{let a=e.model_group.toLowerCase(),r=s.model_group.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=50*!!t.split(/\s+/).every(e=>a.includes(e)),d=50*!!t.split(/\s+/).every(e=>r.includes(e)),m=a.length;return l+o+d+(1e3-r.length)-(i+n+c+(1e3-m))}))}return e.filter(e=>{let t=0===el.length||el.some(t=>e.providers.includes(t)),s=0===eo.length||eo.includes(e.mode||""),a=0===ed.length||Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).some(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");return ed.includes(t)});return t&&s&&a})},[L,ee,el,eo,ed]),eD=(0,h.useMemo)(()=>{if(!O||!Array.isArray(O))return[];let e=O;if(es.trim()){let t=es.toLowerCase(),s=t.split(/\s+/);e=(e=O.filter(e=>{let a=e.name.toLowerCase(),r=e.description.toLowerCase();return!!(a.includes(t)||r.includes(t))||s.every(e=>a.includes(e)||r.includes(e))})).sort((e,s)=>{let a=e.name.toLowerCase(),r=s.name.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===ep.length||e.skills?.some(e=>e.tags?.some(e=>ep.includes(e))))},[O,es,ep]),eH=(0,h.useMemo)(()=>{if(!$||!Array.isArray($))return[];let e=$;if(er.trim()){let t=er.toLowerCase(),s=t.split(/\s+/);e=(e=$.filter(e=>{let a=e.server_name.toLowerCase(),r=(e.mcp_info?.description||"").toLowerCase();return!!(a.includes(t)||r.includes(t))||s.every(e=>a.includes(e)||r.includes(e))})).sort((e,s)=>{let a=e.server_name.toLowerCase(),r=s.server_name.toLowerCase(),i=1e3*(a===t),l=1e3*(r===t),n=100*!!a.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===ex.length||ex.includes(e.transport))},[$,er,ex]),eB=e=>{navigator.clipboard.writeText(e),b.default.success("Copied to clipboard!")},eK=e=>e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),eU=e=>`$${(1e6*e).toFixed(4)}`,eF=e=>e?e>=1e3?`${(e/1e3).toFixed(0)}K`:e.toString():"N/A";return(0,t.jsx)(s.ThemeProvider,{accessToken:e,children:(0,t.jsxs)("div",{className:T?"w-full":"min-h-screen bg-white",children:[!T&&(0,t.jsx)(_.default,{userID:null,userEmail:null,userRole:null,premiumUser:!1,setProxySettings:eI,proxySettings:eM,accessToken:e||null,isPublicPage:!0,isDarkMode:!1,toggleDarkMode:()=>{}}),(0,t.jsxs)("div",{className:T?"w-full p-6":"w-full px-8 py-12",children:[T&&(0,t.jsx)("div",{className:"mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg",children:(0,t.jsx)("p",{className:"text-sm text-gray-700",children:"These are models, agents, and MCP servers your proxy admin has indicated are available in your company."})}),!T&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"About"}),(0,t.jsx)("p",{className:"text-gray-700 mb-6 text-base leading-relaxed",children:K||"Proxy Server to call 100+ LLMs in the OpenAI format."}),(0,t.jsx)("div",{className:"flex items-center space-x-3 text-sm text-gray-600",children:(0,t.jsxs)("span",{className:"flex items-center",children:[(0,t.jsx)("span",{className:"w-4 h-4 mr-2",children:"🔧"}),"Built with litellm: v",F]})})]}),G&&Object.keys(G).length>0&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Useful Links"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",children:Object.entries(G||{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:s})=>(0,t.jsxs)("button",{onClick:()=>window.open(s,"_blank"),className:"flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200",children:[(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)(n.Text,{className:"text-sm font-medium",children:e})]},e))})]}),!T&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Health and Endpoint Status"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:(0,t.jsxs)(n.Text,{className:"text-green-600 font-medium text-sm",children:["Service status: ",eh]})})]}),(0,t.jsx)(l.Card,{className:"p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:(0,t.jsxs)(m.Tabs,{activeKey:eE,onChange:ez,size:"large",className:"public-hub-tabs",children:[(0,t.jsxs)(S,{tab:"Model Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Models"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Models:"}),(0,t.jsx)(u.Tooltip,{title:"Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'",placement:"top",children:(0,t.jsx)(g.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search model names... (smart search enabled)",value:ee,onChange:e=>et(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Provider:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:el,onChange:e=>en(e),placeholder:"Select providers",className:"w-full",size:"large",allowClear:!0,optionRender:e=>{let{logo:s}=(0,w.getProviderLogoAndName)(e.value);return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[s&&(0,t.jsx)("img",{src:s,alt:e.label,className:"w-5 h-5 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e.label})]})},children:L&&Array.isArray(L)&&(C=new Set,L.forEach(e=>{(e.providers??[]).forEach(e=>C.add(e))}),Array.from(C)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Mode:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eo,onChange:e=>ec(e),placeholder:"Select modes",className:"w-full",size:"large",allowClear:!0,children:L&&Array.isArray(L)&&(k=new Set,L.forEach(e=>{e.mode&&k.add(e.mode)}),Array.from(k)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Features:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ed,onChange:e=>em(e),placeholder:"Select features",className:"w-full",size:"large",allowClear:!0,children:L&&Array.isArray(L)&&(A=new Set,L.forEach(e=>{Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).forEach(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");A.add(t)})}),Array.from(A).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Model Name",accessorKey:"model_group",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.model_group,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eS(e.original),e_(!0)},children:e.original.model_group})})}),size:150},{header:"Providers",accessorKey:"providers",enableSorting:!0,cell:({row:e})=>{let s=e.original.providers??[];return(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:s.map(e=>{let{logo:s}=(0,w.getProviderLogoAndName)(e);return(0,t.jsxs)("div",{className:"flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs",children:[s&&(0,t.jsx)("img",{src:s,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]},e)})})},size:120},{header:"Mode",accessorKey:"mode",enableSorting:!0,cell:({row:e})=>{let s=e.original.mode;return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:(e=>{switch(e?.toLowerCase()){case"chat":return"💬";case"rerank":return"🔄";case"embedding":return"📄";default:return"🤖"}})(s||"")}),(0,t.jsx)(n.Text,{children:s||"Chat"})]})},size:100},{header:"Max Input",accessorKey:"max_input_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eF(e.original.max_input_tokens)}),size:100,meta:{className:"text-center"}},{header:"Max Output",accessorKey:"max_output_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eF(e.original.max_output_tokens)}),size:100,meta:{className:"text-center"}},{header:"Input $/1M",accessorKey:"input_cost_per_token",enableSorting:!0,cell:({row:e})=>{let s=e.original.input_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:s?eU(s):"Free"})},size:100,meta:{className:"text-center"}},{header:"Output $/1M",accessorKey:"output_cost_per_token",enableSorting:!0,cell:({row:e})=>{let s=e.original.output_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:s?eU(s):"Free"})},size:100,meta:{className:"text-center"}},{header:"Features",accessorKey:"supports_vision",enableSorting:!1,cell:({row:e})=>{let s=Object.entries(e.original).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>eK(e));return 0===s.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===s.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs",children:s[0]})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs",children:s[0]}),(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Features:"}),s.map((e,s)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e]},s))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",s.length-1]})})]})},size:120},{header:"Health Status",accessorKey:"health_status",enableSorting:!0,cell:({row:e})=>{let s=e.original,a="healthy"===s.health_status?"green":"unhealthy"===s.health_status?"red":"default",r=s.health_response_time?`Response Time: ${Number(s.health_response_time).toFixed(2)}ms`:"N/A",i=s.health_checked_at?`Last Checked: ${new Date(s.health_checked_at).toLocaleString()}`:"N/A";return(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{children:r}),(0,t.jsx)("div",{children:i})]}),children:(0,t.jsx)(p.Tag,{color:a,children:(0,t.jsx)("span",{className:"capitalize",children:s.health_status??"Unknown"})},s.model_group)})},size:100},{header:"Limits",accessorKey:"rpm",enableSorting:!0,cell:({row:e})=>{var s,a;let r,i=e.original;return(0,t.jsx)(n.Text,{className:"text-xs text-gray-600",children:(s=i.rpm,a=i.tpm,r=[],s&&r.push(`RPM: ${s.toLocaleString()}`),a&&r.push(`TPM: ${a.toLocaleString()}`),r.length>0?r.join(", "):"N/A")})},size:150}],data:e$,isLoading:q,defaultSorting:[{id:"model_group",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",e$.length," of ",L?.length||0," models"]})})]},"models"),O&&Array.isArray(O)&&O.length>0&&(0,t.jsxs)(S,{tab:"Agent Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Agents"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Agents:"}),(0,t.jsx)(u.Tooltip,{title:"Search agents by name or description",placement:"top",children:(0,t.jsx)(g.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search agent names or descriptions...",value:es,onChange:e=>ea(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Skills:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ep,onChange:e=>eu(e),placeholder:"Select skills",className:"w-full",size:"large",allowClear:!0,children:O&&Array.isArray(O)&&(M=new Set,O.forEach(e=>{e.skills?.forEach(e=>{e.tags?.forEach(e=>M.add(e))})}),Array.from(M).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Agent Name",accessorKey:"name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eC(e.original),ey(!0)},children:e.original.name})})}),size:150},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>{let s=e.original.description??"",a=s.length>80?s.substring(0,80)+"...":s;return(0,t.jsx)(u.Tooltip,{title:s,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"Version",accessorKey:"version",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-sm",children:e.original.version}),size:80},{header:"Provider",accessorKey:"provider",enableSorting:!1,cell:({row:e})=>{let s=e.original.provider;return s?(0,t.jsx)("div",{className:"text-sm",children:(0,t.jsx)(n.Text,{className:"font-medium",children:s.organization})}):(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"})},size:120},{header:"Skills",accessorKey:"skills",enableSorting:!1,cell:({row:e})=>{let s=e.original.skills||[];return 0===s.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===s.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:s[0].name})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:s[0].name}),(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Skills:"}),s.map((e,s)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e.name]},s))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-purple-600 cursor-pointer hover:text-purple-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",s.length-1]})})]})},size:150},{header:"Capabilities",accessorKey:"capabilities",enableSorting:!1,cell:({row:e})=>{let s=Object.entries(e.original.capabilities||{}).filter(([e,t])=>!0===t).map(([e])=>e);return 0===s.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:s.map(e=>(0,t.jsx)(p.Tag,{color:"green",className:"text-xs capitalize",children:e},e))})},size:150}],data:eD,isLoading:J,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eD.length," of ",O?.length||0," agents"]})})]},"agents"),$&&Array.isArray($)&&$.length>0&&(0,t.jsxs)(S,{tab:"MCP Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available MCP Servers"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search MCP Servers:"}),(0,t.jsx)(u.Tooltip,{title:"Search MCP servers by name or description",placement:"top",children:(0,t.jsx)(g.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search MCP server names or descriptions...",value:er,onChange:e=>ei(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Transport:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ex,onChange:e=>eg(e),placeholder:"Select transport types",className:"w-full",size:"large",allowClear:!0,children:$&&Array.isArray($)&&(I=new Set,$.forEach(e=>{e.transport&&I.add(e.transport)}),Array.from(I).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Server Name",accessorKey:"server_name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.server_name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eA(e.original),eN(!0)},children:e.original.server_name})})}),size:150},{header:"Description",accessorKey:"mcp_info.description",enableSorting:!1,cell:({row:e})=>{let s=String(e.original.mcp_info?.description??"-"),a=s.length>80?s.substring(0,80)+"...":s;return(0,t.jsx)(u.Tooltip,{title:s,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"URL",accessorKey:"url",enableSorting:!1,cell:({row:e})=>{let s=e.original.url??"",a=s.length>40?s.substring(0,40)+"...":s;return(0,t.jsx)(u.Tooltip,{title:s,children:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)(n.Text,{className:"text-xs font-mono",children:a}),(0,t.jsx)(x.Copy,{onClick:()=>eB(s),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-3 h-3"})]})})},size:200},{header:"Transport",accessorKey:"transport",enableSorting:!0,cell:({row:e})=>{let s=e.original.transport;return(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs uppercase",children:s})},size:100},{header:"Auth Type",accessorKey:"auth_type",enableSorting:!0,cell:({row:e})=>{let s=e.original.auth_type;return(0,t.jsx)(p.Tag,{color:"none"===s?"gray":"green",className:"text-xs capitalize",children:s})},size:100}],data:eH,isLoading:Z,defaultSorting:[{id:"server_name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eH.length," of ",$?.length||0," MCP servers"]})})]},"mcp"),(0,t.jsx)(S,{tab:"Skill Hub",children:(0,t.jsx)(y.default,{skills:eL,isLoading:eO,publicPage:!0})},"skills")]})})]}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ew?.model_group||"Model Details"}),ew&&(0,t.jsx)(u.Tooltip,{title:"Copy model name",children:(0,t.jsx)(x.Copy,{onClick:()=>eB(ew.model_group),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:eb,footer:null,onOk:()=>{e_(!1),eS(null)},onCancel:()=>{e_(!1),eS(null)},children:ew&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Model Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Model Name:"}),(0,t.jsx)(n.Text,{children:ew.model_group})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Mode:"}),(0,t.jsx)(n.Text,{children:ew.mode||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Providers:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(ew.providers??[]).map(e=>{let{logo:s}=(0,w.getProviderLogoAndName)(e);return(0,t.jsx)(p.Tag,{color:"blue",children:(0,t.jsxs)("div",{className:"flex items-center space-x-1",children:[s&&(0,t.jsx)("img",{src:s,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]})},e)})})]})]}),ew.model_group.includes("*")&&(0,t.jsx)("div",{className:"bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4",children:(0,t.jsxs)("div",{className:"flex items-start space-x-2",children:[(0,t.jsx)(g.Info,{className:"w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-blue-900 mb-2",children:"Wildcard Routing"}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800 mb-2",children:["This model uses wildcard routing. You can pass any value where you see the"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:"*"})," symbol."]}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800",children:["For example, with"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:ew.model_group}),", you can use any string (",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:ew.model_group.replaceAll("*","my-custom-value")}),") that matches this pattern."]})]})]})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Token & Cost Information"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Input Tokens:"}),(0,t.jsx)(n.Text,{children:ew.max_input_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Output Tokens:"}),(0,t.jsx)(n.Text,{children:ew.max_output_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:ew.input_cost_per_token?eU(ew.input_cost_per_token):"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:ew.output_cost_per_token?eU(ew.output_cost_per_token):"Not specified"})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:(E=Object.entries(ew).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e),z=["green","blue","purple","orange","red","yellow"],0===E.length?(0,t.jsx)(n.Text,{className:"text-gray-500",children:"No special capabilities listed"}):E.map((e,s)=>(0,t.jsx)(p.Tag,{color:z[s%z.length],children:eK(e)},e)))})]}),(ew.tpm||ew.rpm)&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Rate Limits"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[ew.tpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Tokens per Minute:"}),(0,t.jsx)(n.Text,{children:ew.tpm.toLocaleString()})]}),ew.rpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Requests per Minute:"}),(0,t.jsx)(n.Text,{children:ew.rpm.toLocaleString()})]})]})]}),ew.supported_openai_params&&ew.supported_openai_params.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Supported OpenAI Parameters"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:ew.supported_openai_params.map(e=>(0,t.jsx)(p.Tag,{color:"green",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:(0,v.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,N.getEndpointType)(ew.mode||"chat"),selectedModel:ew.model_group,selectedSdk:"openai"})})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eB((0,v.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,N.getEndpointType)(ew.mode||"chat"),selectedModel:ew.model_group,selectedSdk:"openai"}))},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eT?.name||"Agent Details"}),eT&&(0,t.jsx)(u.Tooltip,{title:"Copy agent name",children:(0,t.jsx)(x.Copy,{onClick:()=>eB(eT.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ej,footer:null,onOk:()=>{ey(!1),eC(null)},onCancel:()=>{ey(!1),eC(null)},children:eT&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Agent Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Name:"}),(0,t.jsx)(n.Text,{children:eT.name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Version:"}),(0,t.jsx)(n.Text,{children:eT.version})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:eT.description})]}),eT.url&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsx)("a",{href:eT.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all",children:eT.url})]})]})]}),eT.capabilities&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:Object.entries(eT.capabilities).filter(([e,t])=>!0===t).map(([e])=>(0,t.jsx)(p.Tag,{color:"green",className:"capitalize",children:e},e))})]}),eT.skills&&eT.skills.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Skills"}),(0,t.jsx)("div",{className:"space-y-4",children:eT.skills.map((e,s)=>(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"flex items-start justify-between mb-2",children:(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-base",children:e.name}),(0,t.jsx)(n.Text,{className:"text-sm text-gray-600",children:e.description})]})}),e.tags&&e.tags.length>0&&(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-2",children:e.tags.map(e=>(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:e},e))})]},s))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Input/Output Modes"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eT.defaultInputModes??[]).map(e=>(0,t.jsx)(p.Tag,{color:"blue",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eT.defaultOutputModes??[]).map(e=>(0,t.jsx)(p.Tag,{color:"blue",children:e},e))})]})]})]}),eT.documentationUrl&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Documentation"}),(0,t.jsxs)("a",{href:eT.documentationUrl,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 flex items-center space-x-2",children:[(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)("span",{children:"View Documentation"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example (A2A Protocol)"}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 1: Retrieve Agent Card"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`base_url = '${eT.url}'

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
        )`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eB(`from a2a.client import A2ACardResolver, A2AClient
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

base_url = '${eT.url}'

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
        )`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 2: Call the Agent"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`client = A2AClient(
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
print(response.model_dump(mode='json', exclude_none=True))`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eB(`client = A2AClient(
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
print(response.model_dump(mode='json', exclude_none=True))`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ek?.server_name||"MCP Server Details"}),ek&&(0,t.jsx)(u.Tooltip,{title:"Copy server name",children:(0,t.jsx)(x.Copy,{onClick:()=>eB(ek.server_name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ev,footer:null,onOk:()=>{eN(!1),eA(null)},onCancel:()=>{eN(!1),eA(null)},children:ek&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Server Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Server Name:"}),(0,t.jsx)(n.Text,{children:ek.server_name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Transport:"}),(0,t.jsx)(p.Tag,{color:"blue",children:ek.transport})]}),ek.alias&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Alias:"}),(0,t.jsx)(n.Text,{children:ek.alias})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Auth Type:"}),(0,t.jsx)(p.Tag,{color:"none"===ek.auth_type?"gray":"green",children:ek.auth_type})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:ek.mcp_info?.description||"-"})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsxs)("a",{href:ek.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ek.url}),(0,t.jsx)(a.ExternalLinkIcon,{className:"w-4 h-4"})]})]})]})]}),ek.mcp_info&&Object.keys(ek.mcp_info).length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Additional Information"}),(0,t.jsx)("div",{className:"bg-gray-50 p-4 rounded-lg",children:(0,t.jsx)("pre",{className:"text-xs overflow-x-auto",children:JSON.stringify(ek.mcp_info,null,2)})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${ek.server_name}": {
            "url": "http://localhost:4000/${ek.server_name}/mcp",
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
    asyncio.run(main())`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eB(`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${ek.server_name}": {
            "url": "http://localhost:4000/${ek.server_name}/mcp",
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
    asyncio.run(main())`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})})]})})}])}]);