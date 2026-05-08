(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,818581,(e,t,a)=>{"use strict";Object.defineProperty(a,"__esModule",{value:!0}),Object.defineProperty(a,"useMergedRef",{enumerable:!0,get:function(){return r}});let s=e.r(271645);function r(e,t){let a=(0,s.useRef)(null),r=(0,s.useRef)(null);return(0,s.useCallback)(s=>{if(null===s){let e=a.current;e&&(a.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(a.current=i(e,s)),t&&(r.current=i(t,s))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let a=e(t);return"function"==typeof a?a:()=>e(null)}}("function"==typeof a.default||"object"==typeof a.default&&null!==a.default)&&void 0===a.default.__esModule&&(Object.defineProperty(a.default,"__esModule",{value:!0}),Object.assign(a.default,a),t.exports=a.default)},62478,e=>{"use strict";var t=e.i(764205);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let s={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),i=a.forwardRef(function(e,i){return a.createElement(r.default,(0,t.default)({},e,{ref:i,icon:s}))});e.s(["SafetyOutlined",0,i],602073)},190272,785913,e=>{"use strict";var t,a,s=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(s).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:s,apiKey:i,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:m,selectedMCPServers:p,mcpServers:u,mcpServerToolRestrictions:g,selectedVoice:x,endpointType:h,selectedModel:f,selectedSdk:_,proxySettings:b}=e,y="session"===a?s:i,v=window.location.origin,j=b?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?v=j:b?.PROXY_BASE_URL&&(v=b.PROXY_BASE_URL);let A=l||"Your prompt here",N=A.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),T=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};o.length>0&&(S.tags=o),c.length>0&&(S.vector_stores=c),d.length>0&&(S.guardrails=d),m.length>0&&(S.policies=m);let C=f||"your-model-name",w="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case r.CHAT:{let e=Object.keys(S).length>0,a="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let s=T.length>0?T:[{role:"user",content:A}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(s,null,4)}${a}
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
#                     "text": "${N}"
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
`;break}case r.RESPONSES:{let e=Object.keys(S).length>0,a="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let s=T.length>0?T:[{role:"user",content:A}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(s,null,4)}${a}
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
#                 {"type": "input_text", "text": "${N}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===_?`
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
prompt = "${N}"

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
`;break;case r.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${N}"

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
prompt = "${N}"

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
	voice="${x}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${w}
${t}`}],190272)},652272,209261,e=>{"use strict";var t=e.i(843476),a=e.i(271645),s=e.i(447566),r=e.i(166406),i=e.i(492030),l=e.i(596239);let n=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,n,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:o})=>{let c,[d,m]=(0,a.useState)("overview"),[p,u]=(0,a.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),u(t),setTimeout(()=>u(null),2e3)},x="github"===(c=e.source).source&&c.repo?`https://github.com/${c.repo}`:"git-subdir"===c.source&&c.url?c.path?`${c.url}/tree/main/${c.path}`:c.url:"url"===c.source&&c.url?c.url:null,h=n(e),f=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:o,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(s.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>m(e.key),style:{padding:"12px 20px",fontSize:14,color:d===e.key?"#1a73e8":"#5f6368",borderBottom:d===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:d===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===d&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:f.map((e,a)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},a))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),x&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:x,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[x.replace("https://",""),(0,t.jsx)(l.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(h,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===p?(0,t.jsx)(i.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:h})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>m("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{g(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===p?(0,t.jsx)(i.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},115571,e=>{"use strict";let t="local-storage-change";function a(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function s(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function r(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function i(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>a,"getLocalStorageItem",()=>s,"removeLocalStorageItem",()=>i,"setLocalStorageItem",()=>r])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let s={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),i=a.forwardRef(function(e,i){return a.createElement(r.default,(0,t.default)({},e,{ref:i,icon:s}))});e.s(["LinkOutlined",0,i],596239)},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},916925,e=>{"use strict";var t,a=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t.ZAI="Z.AI (Zhipu AI)",t);let s={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference",ZAI:"zai"},r="../ui/assets/logos/",i={"A2A Agent":`${r}a2a_agent.png`,Ai21:`${r}ai21.svg`,"Ai21 Chat":`${r}ai21.svg`,"AI/ML API":`${r}aiml_api.svg`,"Aiohttp Openai":`${r}openai_small.svg`,Anthropic:`${r}anthropic.svg`,"Anthropic Text":`${r}anthropic.svg`,AssemblyAI:`${r}assemblyai_small.png`,Azure:`${r}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${r}microsoft_azure.svg`,"Azure Text":`${r}microsoft_azure.svg`,Baseten:`${r}baseten.svg`,"Amazon Bedrock":`${r}bedrock.svg`,"Amazon Bedrock Mantle":`${r}bedrock.svg`,"AWS SageMaker":`${r}bedrock.svg`,Cerebras:`${r}cerebras.svg`,Cloudflare:`${r}cloudflare.svg`,Codestral:`${r}mistral.svg`,Cohere:`${r}cohere.svg`,"Cohere Chat":`${r}cohere.svg`,Cometapi:`${r}cometapi.svg`,Cursor:`${r}cursor.svg`,"Databricks (Qwen API)":`${r}databricks.svg`,Dashscope:`${r}dashscope.svg`,Deepseek:`${r}deepseek.svg`,Deepgram:`${r}deepgram.png`,DeepInfra:`${r}deepinfra.png`,ElevenLabs:`${r}elevenlabs.png`,"Fal AI":`${r}fal_ai.jpg`,"Featherless Ai":`${r}featherless.svg`,"Fireworks AI":`${r}fireworks.svg`,Friendliai:`${r}friendli.svg`,"Github Copilot":`${r}github_copilot.svg`,"Google AI Studio":`${r}google.svg`,GradientAI:`${r}gradientai.svg`,Groq:`${r}groq.svg`,vllm:`${r}vllm.png`,Huggingface:`${r}huggingface.svg`,Hyperbolic:`${r}hyperbolic.svg`,Infinity:`${r}infinity.png`,"Jina AI":`${r}jina.png`,"Lambda Ai":`${r}lambda.svg`,"Lm Studio":`${r}lmstudio.svg`,"Meta Llama":`${r}meta_llama.svg`,MiniMax:`${r}minimax.svg`,"Mistral AI":`${r}mistral.svg`,Moonshot:`${r}moonshot.svg`,Morph:`${r}morph.svg`,Nebius:`${r}nebius.svg`,Novita:`${r}novita.svg`,"Nvidia Nim":`${r}nvidia_nim.svg`,Ollama:`${r}ollama.svg`,"Ollama Chat":`${r}ollama.svg`,Oobabooga:`${r}openai_small.svg`,OpenAI:`${r}openai_small.svg`,"Openai Like":`${r}openai_small.svg`,"OpenAI Text Completion":`${r}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${r}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${r}openai_small.svg`,Openrouter:`${r}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${r}oracle.svg`,Perplexity:`${r}perplexity-ai.svg`,Recraft:`${r}recraft.svg`,Replicate:`${r}replicate.svg`,RunwayML:`${r}runwayml.png`,Sagemaker:`${r}bedrock.svg`,Sambanova:`${r}sambanova.svg`,"SAP Generative AI Hub":`${r}sap.png`,Snowflake:`${r}snowflake.svg`,"Text-Completion-Codestral":`${r}mistral.svg`,TogetherAI:`${r}togetherai.svg`,Topaz:`${r}topaz.svg`,Triton:`${r}nvidia_triton.png`,V0:`${r}v0.svg`,"Vercel Ai Gateway":`${r}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${r}google.svg`,"Vertex Ai Beta":`${r}google.svg`,Vllm:`${r}vllm.png`,VolcEngine:`${r}volcengine.png`,"Voyage AI":`${r}voyage.webp`,Watsonx:`${r}watsonx.svg`,"Watsonx Text":`${r}watsonx.svg`,xAI:`${r}xai.svg`,Xinference:`${r}xinference.svg`};e.s(["Providers",()=>a,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else if("Z.AI (Zhipu AI)"===e)return"zai/glm-4.5";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:i[e],displayName:e}}let t=Object.keys(s).find(t=>s[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let r=a[t];return{logo:i[r],displayName:r}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let a=s[e];console.log(`Provider mapped to: ${a}`);let r=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let s=t.litellm_provider;(s===a||"string"==typeof s&&s.includes(a))&&r.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&r.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&r.push(e)}))),r},"providerLogoMap",0,i,"provider_map",0,s])},798496,e=>{"use strict";var t=e.i(843476),a=e.i(152990),s=e.i(682830),r=e.i(271645),i=e.i(269200),l=e.i(427612),n=e.i(64848),o=e.i(942232),c=e.i(496020),d=e.i(977572),m=e.i(94629),p=e.i(360820),u=e.i(871943);function g({data:e=[],columns:g,isLoading:x=!1,defaultSorting:h=[],pagination:f,onPaginationChange:_,enablePagination:b=!1,onRowClick:y}){let[v,j]=r.default.useState(h),[A]=r.default.useState("onChange"),[N,T]=r.default.useState({}),[S,C]=r.default.useState({}),w=(0,a.useReactTable)({data:e,columns:g,state:{sorting:v,columnSizing:N,columnVisibility:S,...b&&f?{pagination:f}:{}},columnResizeMode:A,onSortingChange:j,onColumnSizingChange:T,onColumnVisibilityChange:C,...b&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,s.getCoreRowModel)(),getSortedRowModel:(0,s.getSortedRowModel)(),...b?{getPaginationRowModel:(0,s.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(i.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:w.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(l.TableHead,{children:w.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(n.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,a.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(p.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(u.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(m.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(o.TableBody,{children:x?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):w.getRowModel().rows.length>0?w.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>y?.(e.original),className:y?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,a.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>g])},737033,e=>{"use strict";var t=e.i(843476),a=e.i(271645),s=e.i(599724),r=e.i(928685),i=e.i(311451),l=e.i(199133),n=e.i(798496),o=e.i(389083),c=e.i(592968),d=e.i(166406),m=e.i(596239),p=e.i(652272);e.s(["default",0,({skills:e,isLoading:u,isAdmin:g,accessToken:x,publicPage:h=!1,onPublishSuccess:f})=>{let[_,b]=(0,a.useState)(""),[y,v]=(0,a.useState)(void 0),[j,A]=(0,a.useState)(null),N=e.length,T=(0,a.useMemo)(()=>[...new Set(e.map(e=>e.domain).filter(Boolean))],[e]),S=(0,a.useMemo)(()=>[...new Set(e.map(e=>e.namespace).filter(Boolean))],[e]),C=(0,a.useMemo)(()=>{let t=e;if(y&&(t=t.filter(e=>(e.domain||"General")===y)),_.trim()){let e=_.toLowerCase();t=t.filter(t=>t.name.toLowerCase().includes(e)||t.description?.toLowerCase().includes(e)||t.domain?.toLowerCase().includes(e)||t.namespace?.toLowerCase().includes(e)||t.keywords?.some(t=>t.toLowerCase().includes(e)))}return t},[e,_,y]);return j?(0,t.jsx)(p.default,{skill:j,onBack:()=>A(null),isAdmin:g,accessToken:x,onPublishClick:f}):u?(0,t.jsx)("div",{className:"text-center py-16 text-gray-400",children:"Loading skills..."}):(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{className:"grid grid-cols-3 gap-4",children:[(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Total Skills"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:N})]}),(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Namespaces"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:S.length})]}),(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-gray-500 mb-1",children:"Domains"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-gray-900",children:T.length})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-3",children:[(0,t.jsxs)("h3",{className:"text-sm font-semibold text-gray-700",children:["All ",h?"Public ":"","Skills"]}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(l.Select,{placeholder:"All Domains",allowClear:!0,value:y,onChange:e=>v(e),style:{width:160},options:T.map(e=>({label:e,value:e}))}),(0,t.jsx)(i.Input,{prefix:(0,t.jsx)(r.SearchOutlined,{className:"text-gray-400"}),placeholder:"Search by name, namespace, or tag…",value:_,onChange:e=>b(e.target.value),style:{width:280},allowClear:!0})]})]}),(0,t.jsx)(n.ModelDataTable,{columns:((e,a,r=!1)=>[{header:"Skill Name",accessorKey:"name",enableSorting:!0,sortingFn:"alphanumeric",cell:({row:r})=>{let i=r.original;return(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("button",{type:"button",className:"font-medium text-sm cursor-pointer text-blue-600 hover:underline bg-transparent border-none p-0",onClick:()=>e(i),children:i.name}),(0,t.jsx)(c.Tooltip,{title:"Copy skill name",children:(0,t.jsx)(d.CopyOutlined,{onClick:()=>a(i.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 text-xs"})})]}),i.description&&(0,t.jsx)(s.Text,{className:"text-xs text-gray-500 line-clamp-1 md:hidden",children:i.description})]})}},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>(0,t.jsx)(s.Text,{className:"text-xs line-clamp-2",children:e.original.description||"-"})},{header:"Category",accessorKey:"category",enableSorting:!0,cell:({row:e})=>{let a=e.original.category;return a?(0,t.jsx)(o.Badge,{color:"blue",size:"xs",children:a}):(0,t.jsx)(s.Text,{className:"text-xs text-gray-400",children:"-"})}},{header:"Domain",accessorKey:"domain",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(s.Text,{className:"text-xs",children:e.original.domain||"-"})},{header:"Source",accessorKey:"source",enableSorting:!1,cell:({row:e})=>{let a=e.original.source,r=null,i="-";return(a?.source==="github"&&a.repo?(r=`https://github.com/${a.repo}`,i=a.repo):a?.source==="git-subdir"&&a.url?i=(r=a.path?`${a.url}/tree/main/${a.path}`:a.url).replace("https://github.com/",""):a?.source==="url"&&a.url&&(r=a.url,i=a.url.replace(/^https?:\/\//,"")),r)?(0,t.jsxs)("a",{href:r,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 text-xs text-blue-600 hover:underline truncate max-w-[180px]",title:i,children:[(0,t.jsx)("span",{className:"truncate",children:i}),(0,t.jsx)(m.LinkOutlined,{className:"shrink-0",style:{fontSize:10}})]}):(0,t.jsx)(s.Text,{className:"text-xs text-gray-400",children:"-"})}},{header:"Status",accessorKey:"enabled",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(o.Badge,{color:e.original.enabled?"green":"gray",size:"xs",children:e.original.enabled?"Public":"Draft"})}])(e=>A(e),e=>{navigator.clipboard.writeText(e)},h),data:C,isLoading:!1,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-3 text-center",children:(0,t.jsxs)(s.Text,{className:"text-sm text-gray-500",children:["Showing ",C.length," of ",N," skill",1!==N?"s":""]})})]})]})}],737033)},93826,174886,952571,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"}))});e.s(["SearchIcon",0,a],93826);var s=e.i(991124);e.s(["Copy",()=>s.default],174886);var r=e.i(879664);e.s(["Info",()=>r.default],952571)},976883,e=>{"use strict";var t=e.i(843476),a=e.i(275144),s=e.i(434626),r=e.i(93826),i=e.i(994388),l=e.i(304967),n=e.i(599724),o=e.i(629569),c=e.i(212931),d=e.i(199133),m=e.i(653496),p=e.i(262218),u=e.i(592968),g=e.i(174886),x=e.i(952571),h=e.i(271645),f=e.i(798496),_=e.i(727749),b=e.i(402874),y=e.i(764205),v=e.i(737033),j=e.i(190272),A=e.i(785913),N=e.i(916925);let{TabPane:T}=m.Tabs;e.s(["default",0,({accessToken:e,isEmbedded:S=!1})=>{let C,w,I,k,E,O,M,[L,$]=(0,h.useState)(null),[R,P]=(0,h.useState)(null),[z,D]=(0,h.useState)(null),[H,B]=(0,h.useState)("LiteLLM Gateway"),[G,F]=(0,h.useState)(null),[U,V]=(0,h.useState)(""),[K,W]=(0,h.useState)({}),[X,q]=(0,h.useState)(!0),[Y,Z]=(0,h.useState)(!0),[J,Q]=(0,h.useState)(!0),[ee,et]=(0,h.useState)(""),[ea,es]=(0,h.useState)(""),[er,ei]=(0,h.useState)(""),[el,en]=(0,h.useState)([]),[eo,ec]=(0,h.useState)([]),[ed,em]=(0,h.useState)([]),[ep,eu]=(0,h.useState)([]),[eg,ex]=(0,h.useState)([]),[eh,ef]=(0,h.useState)("I'm alive! ✓"),[e_,eb]=(0,h.useState)(!1),[ey,ev]=(0,h.useState)(!1),[ej,eA]=(0,h.useState)(!1),[eN,eT]=(0,h.useState)(null),[eS,eC]=(0,h.useState)(null),[ew,eI]=(0,h.useState)(null),[ek,eE]=(0,h.useState)({}),[eO,eM]=(0,h.useState)("models"),[eL,e$]=(0,h.useState)([]),[eR,eP]=(0,h.useState)(!1);(0,h.useEffect)(()=>{(async()=>{try{await (0,y.getUiConfig)()}catch(e){console.error("Failed to get UI config:",e)}let e=async()=>{try{q(!0);let e=await (0,y.modelHubPublicModelsCall)();console.log("ModelHubData:",e),$(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public model data",e),ef("Service unavailable")}finally{q(!1)}},t=async()=>{try{Z(!0);let e=await (0,y.agentHubPublicModelsCall)();console.log("AgentHubData:",e),P(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public agent data",e)}finally{Z(!1)}},a=async()=>{try{Q(!0);let e=await (0,y.mcpHubPublicServersCall)();console.log("MCPHubData:",e),D(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public MCP server data",e)}finally{Q(!1)}},s=async()=>{try{eP(!0);let e=await (0,y.skillHubPublicCall)();e$(e.plugins??[])}catch(e){console.error("There was an error fetching the public skill data",e)}finally{eP(!1)}};(async()=>{let e=await (0,y.getPublicModelHubInfo)();console.log("Public Model Hub Info:",e),B(e.docs_title),F(e.custom_docs_description),V(e.litellm_version),W(e.useful_links||{})})(),e(),t(),a(),s()})()},[]),(0,h.useEffect)(()=>{},[ee,el,eo,ed]);let ez=(0,h.useMemo)(()=>{if(!L||!Array.isArray(L))return[];let e=L;if(ee.trim()){let t=ee.toLowerCase(),a=t.split(/\s+/),s=L.filter(e=>{let s=e.model_group.toLowerCase();return!!s.includes(t)||a.every(e=>s.includes(e))});s.length>0&&(e=s.sort((e,a)=>{let s=e.model_group.toLowerCase(),r=a.model_group.toLowerCase(),i=1e3*(s===t),l=1e3*(r===t),n=100*!!s.startsWith(t),o=100*!!r.startsWith(t),c=50*!!t.split(/\s+/).every(e=>s.includes(e)),d=50*!!t.split(/\s+/).every(e=>r.includes(e)),m=s.length;return l+o+d+(1e3-r.length)-(i+n+c+(1e3-m))}))}return e.filter(e=>{let t=0===el.length||el.some(t=>e.providers.includes(t)),a=0===eo.length||eo.includes(e.mode||""),s=0===ed.length||Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).some(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");return ed.includes(t)});return t&&a&&s})},[L,ee,el,eo,ed]),eD=(0,h.useMemo)(()=>{if(!R||!Array.isArray(R))return[];let e=R;if(ea.trim()){let t=ea.toLowerCase(),a=t.split(/\s+/);e=(e=R.filter(e=>{let s=e.name.toLowerCase(),r=e.description.toLowerCase();return!!(s.includes(t)||r.includes(t))||a.every(e=>s.includes(e)||r.includes(e))})).sort((e,a)=>{let s=e.name.toLowerCase(),r=a.name.toLowerCase(),i=1e3*(s===t),l=1e3*(r===t),n=100*!!s.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-s.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===ep.length||e.skills?.some(e=>e.tags?.some(e=>ep.includes(e))))},[R,ea,ep]),eH=(0,h.useMemo)(()=>{if(!z||!Array.isArray(z))return[];let e=z;if(er.trim()){let t=er.toLowerCase(),a=t.split(/\s+/);e=(e=z.filter(e=>{let s=e.server_name.toLowerCase(),r=(e.mcp_info?.description||"").toLowerCase();return!!(s.includes(t)||r.includes(t))||a.every(e=>s.includes(e)||r.includes(e))})).sort((e,a)=>{let s=e.server_name.toLowerCase(),r=a.server_name.toLowerCase(),i=1e3*(s===t),l=1e3*(r===t),n=100*!!s.startsWith(t),o=100*!!r.startsWith(t),c=i+n+(1e3-s.length);return l+o+(1e3-r.length)-c})}return e.filter(e=>0===eg.length||eg.includes(e.transport))},[z,er,eg]),eB=e=>{navigator.clipboard.writeText(e),_.default.success("Copied to clipboard!")},eG=e=>e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),eF=e=>`$${(1e6*e).toFixed(4)}`,eU=e=>e?e>=1e3?`${(e/1e3).toFixed(0)}K`:e.toString():"N/A";return(0,t.jsx)(a.ThemeProvider,{accessToken:e,children:(0,t.jsxs)("div",{className:S?"w-full":"min-h-screen bg-white",children:[!S&&(0,t.jsx)(b.default,{userID:null,userEmail:null,userRole:null,premiumUser:!1,setProxySettings:eE,proxySettings:ek,accessToken:e||null,isPublicPage:!0,isDarkMode:!1,toggleDarkMode:()=>{}}),(0,t.jsxs)("div",{className:S?"w-full p-6":"w-full px-8 py-12",children:[S&&(0,t.jsx)("div",{className:"mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg",children:(0,t.jsx)("p",{className:"text-sm text-gray-700",children:"These are models, agents, and MCP servers your proxy admin has indicated are available in your company."})}),!S&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"About"}),(0,t.jsx)("p",{className:"text-gray-700 mb-6 text-base leading-relaxed",children:G||"Proxy Server to call 100+ LLMs in the OpenAI format."}),(0,t.jsx)("div",{className:"flex items-center space-x-3 text-sm text-gray-600",children:(0,t.jsxs)("span",{className:"flex items-center",children:[(0,t.jsx)("span",{className:"w-4 h-4 mr-2",children:"🔧"}),"Built with litellm: v",U]})})]}),K&&Object.keys(K).length>0&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Useful Links"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",children:Object.entries(K||{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:a})=>(0,t.jsxs)("button",{onClick:()=>window.open(a,"_blank"),className:"flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200",children:[(0,t.jsx)(s.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)(n.Text,{className:"text-sm font-medium",children:e})]},e))})]}),!S&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Health and Endpoint Status"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:(0,t.jsxs)(n.Text,{className:"text-green-600 font-medium text-sm",children:["Service status: ",eh]})})]}),(0,t.jsx)(l.Card,{className:"p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:(0,t.jsxs)(m.Tabs,{activeKey:eO,onChange:eM,size:"large",className:"public-hub-tabs",children:[(0,t.jsxs)(T,{tab:"Model Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Models"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Models:"}),(0,t.jsx)(u.Tooltip,{title:"Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'",placement:"top",children:(0,t.jsx)(x.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search model names... (smart search enabled)",value:ee,onChange:e=>et(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Provider:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:el,onChange:e=>en(e),placeholder:"Select providers",className:"w-full",size:"large",allowClear:!0,optionRender:e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e.value);return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[a&&(0,t.jsx)("img",{src:a,alt:e.label,className:"w-5 h-5 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e.label})]})},children:L&&Array.isArray(L)&&(C=new Set,L.forEach(e=>{(e.providers??[]).forEach(e=>C.add(e))}),Array.from(C)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Mode:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eo,onChange:e=>ec(e),placeholder:"Select modes",className:"w-full",size:"large",allowClear:!0,children:L&&Array.isArray(L)&&(w=new Set,L.forEach(e=>{e.mode&&w.add(e.mode)}),Array.from(w)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Features:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ed,onChange:e=>em(e),placeholder:"Select features",className:"w-full",size:"large",allowClear:!0,children:L&&Array.isArray(L)&&(I=new Set,L.forEach(e=>{Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).forEach(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");I.add(t)})}),Array.from(I).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Model Name",accessorKey:"model_group",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.model_group,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eT(e.original),eb(!0)},children:e.original.model_group})})}),size:150},{header:"Providers",accessorKey:"providers",enableSorting:!0,cell:({row:e})=>{let a=e.original.providers??[];return(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:a.map(e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e);return(0,t.jsxs)("div",{className:"flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs",children:[a&&(0,t.jsx)("img",{src:a,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]},e)})})},size:120},{header:"Mode",accessorKey:"mode",enableSorting:!0,cell:({row:e})=>{let a=e.original.mode;return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:(e=>{switch(e?.toLowerCase()){case"chat":return"💬";case"rerank":return"🔄";case"embedding":return"📄";default:return"🤖"}})(a||"")}),(0,t.jsx)(n.Text,{children:a||"Chat"})]})},size:100},{header:"Max Input",accessorKey:"max_input_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eU(e.original.max_input_tokens)}),size:100,meta:{className:"text-center"}},{header:"Max Output",accessorKey:"max_output_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eU(e.original.max_output_tokens)}),size:100,meta:{className:"text-center"}},{header:"Input $/1M",accessorKey:"input_cost_per_token",enableSorting:!0,cell:({row:e})=>{let a=e.original.input_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:a?eF(a):"Free"})},size:100,meta:{className:"text-center"}},{header:"Output $/1M",accessorKey:"output_cost_per_token",enableSorting:!0,cell:({row:e})=>{let a=e.original.output_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:a?eF(a):"Free"})},size:100,meta:{className:"text-center"}},{header:"Features",accessorKey:"supports_vision",enableSorting:!1,cell:({row:e})=>{let a=Object.entries(e.original).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>eG(e));return 0===a.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===a.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs",children:a[0]})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs",children:a[0]}),(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Features:"}),a.map((e,a)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e]},a))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",a.length-1]})})]})},size:120},{header:"Health Status",accessorKey:"health_status",enableSorting:!0,cell:({row:e})=>{let a=e.original,s="healthy"===a.health_status?"green":"unhealthy"===a.health_status?"red":"default",r=a.health_response_time?`Response Time: ${Number(a.health_response_time).toFixed(2)}ms`:"N/A",i=a.health_checked_at?`Last Checked: ${new Date(a.health_checked_at).toLocaleString()}`:"N/A";return(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{children:r}),(0,t.jsx)("div",{children:i})]}),children:(0,t.jsx)(p.Tag,{color:s,children:(0,t.jsx)("span",{className:"capitalize",children:a.health_status??"Unknown"})},a.model_group)})},size:100},{header:"Limits",accessorKey:"rpm",enableSorting:!0,cell:({row:e})=>{var a,s;let r,i=e.original;return(0,t.jsx)(n.Text,{className:"text-xs text-gray-600",children:(a=i.rpm,s=i.tpm,r=[],a&&r.push(`RPM: ${a.toLocaleString()}`),s&&r.push(`TPM: ${s.toLocaleString()}`),r.length>0?r.join(", "):"N/A")})},size:150}],data:ez,isLoading:X,defaultSorting:[{id:"model_group",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",ez.length," of ",L?.length||0," models"]})})]},"models"),R&&Array.isArray(R)&&R.length>0&&(0,t.jsxs)(T,{tab:"Agent Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Agents"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Agents:"}),(0,t.jsx)(u.Tooltip,{title:"Search agents by name or description",placement:"top",children:(0,t.jsx)(x.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search agent names or descriptions...",value:ea,onChange:e=>es(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Skills:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ep,onChange:e=>eu(e),placeholder:"Select skills",className:"w-full",size:"large",allowClear:!0,children:R&&Array.isArray(R)&&(k=new Set,R.forEach(e=>{e.skills?.forEach(e=>{e.tags?.forEach(e=>k.add(e))})}),Array.from(k).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Agent Name",accessorKey:"name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eC(e.original),ev(!0)},children:e.original.name})})}),size:150},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>{let a=e.original.description??"",s=a.length>80?a.substring(0,80)+"...":a;return(0,t.jsx)(u.Tooltip,{title:a,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:s})})},size:250},{header:"Version",accessorKey:"version",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-sm",children:e.original.version}),size:80},{header:"Provider",accessorKey:"provider",enableSorting:!1,cell:({row:e})=>{let a=e.original.provider;return a?(0,t.jsx)("div",{className:"text-sm",children:(0,t.jsx)(n.Text,{className:"font-medium",children:a.organization})}):(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"})},size:120},{header:"Skills",accessorKey:"skills",enableSorting:!1,cell:({row:e})=>{let a=e.original.skills||[];return 0===a.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===a.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:a[0].name})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:a[0].name}),(0,t.jsx)(u.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Skills:"}),a.map((e,a)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e.name]},a))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-purple-600 cursor-pointer hover:text-purple-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",a.length-1]})})]})},size:150},{header:"Capabilities",accessorKey:"capabilities",enableSorting:!1,cell:({row:e})=>{let a=Object.entries(e.original.capabilities||{}).filter(([e,t])=>!0===t).map(([e])=>e);return 0===a.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:a.map(e=>(0,t.jsx)(p.Tag,{color:"green",className:"text-xs capitalize",children:e},e))})},size:150}],data:eD,isLoading:Y,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eD.length," of ",R?.length||0," agents"]})})]},"agents"),z&&Array.isArray(z)&&z.length>0&&(0,t.jsxs)(T,{tab:"MCP Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available MCP Servers"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search MCP Servers:"}),(0,t.jsx)(u.Tooltip,{title:"Search MCP servers by name or description",placement:"top",children:(0,t.jsx)(x.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.SearchIcon,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search MCP server names or descriptions...",value:er,onChange:e=>ei(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Transport:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eg,onChange:e=>ex(e),placeholder:"Select transport types",className:"w-full",size:"large",allowClear:!0,children:z&&Array.isArray(z)&&(E=new Set,z.forEach(e=>{e.transport&&E.add(e.transport)}),Array.from(E).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Server Name",accessorKey:"server_name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(u.Tooltip,{title:e.original.server_name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eI(e.original),eA(!0)},children:e.original.server_name})})}),size:150},{header:"Description",accessorKey:"mcp_info.description",enableSorting:!1,cell:({row:e})=>{let a=String(e.original.mcp_info?.description??"-"),s=a.length>80?a.substring(0,80)+"...":a;return(0,t.jsx)(u.Tooltip,{title:a,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:s})})},size:250},{header:"URL",accessorKey:"url",enableSorting:!1,cell:({row:e})=>{let a=e.original.url??"",s=a.length>40?a.substring(0,40)+"...":a;return(0,t.jsx)(u.Tooltip,{title:a,children:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)(n.Text,{className:"text-xs font-mono",children:s}),(0,t.jsx)(g.Copy,{onClick:()=>eB(a),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-3 h-3"})]})})},size:200},{header:"Transport",accessorKey:"transport",enableSorting:!0,cell:({row:e})=>{let a=e.original.transport;return(0,t.jsx)(p.Tag,{color:"blue",className:"text-xs uppercase",children:a})},size:100},{header:"Auth Type",accessorKey:"auth_type",enableSorting:!0,cell:({row:e})=>{let a=e.original.auth_type;return(0,t.jsx)(p.Tag,{color:"none"===a?"gray":"green",className:"text-xs capitalize",children:a})},size:100}],data:eH,isLoading:J,defaultSorting:[{id:"server_name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eH.length," of ",z?.length||0," MCP servers"]})})]},"mcp"),(0,t.jsx)(T,{tab:"Skill Hub",children:(0,t.jsx)(v.default,{skills:eL,isLoading:eR,publicPage:!0})},"skills")]})})]}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eN?.model_group||"Model Details"}),eN&&(0,t.jsx)(u.Tooltip,{title:"Copy model name",children:(0,t.jsx)(g.Copy,{onClick:()=>eB(eN.model_group),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:e_,footer:null,onOk:()=>{eb(!1),eT(null)},onCancel:()=>{eb(!1),eT(null)},children:eN&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Model Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Model Name:"}),(0,t.jsx)(n.Text,{children:eN.model_group})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Mode:"}),(0,t.jsx)(n.Text,{children:eN.mode||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Providers:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eN.providers??[]).map(e=>{let{logo:a}=(0,N.getProviderLogoAndName)(e);return(0,t.jsx)(p.Tag,{color:"blue",children:(0,t.jsxs)("div",{className:"flex items-center space-x-1",children:[a&&(0,t.jsx)("img",{src:a,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]})},e)})})]})]}),eN.model_group.includes("*")&&(0,t.jsx)("div",{className:"bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4",children:(0,t.jsxs)("div",{className:"flex items-start space-x-2",children:[(0,t.jsx)(x.Info,{className:"w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-blue-900 mb-2",children:"Wildcard Routing"}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800 mb-2",children:["This model uses wildcard routing. You can pass any value where you see the"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:"*"})," symbol."]}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800",children:["For example, with"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group}),", you can use any string (",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:eN.model_group.replaceAll("*","my-custom-value")}),") that matches this pattern."]})]})]})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Token & Cost Information"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Input Tokens:"}),(0,t.jsx)(n.Text,{children:eN.max_input_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Output Tokens:"}),(0,t.jsx)(n.Text,{children:eN.max_output_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:eN.input_cost_per_token?eF(eN.input_cost_per_token):"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:eN.output_cost_per_token?eF(eN.output_cost_per_token):"Not specified"})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:(O=Object.entries(eN).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e),M=["green","blue","purple","orange","red","yellow"],0===O.length?(0,t.jsx)(n.Text,{className:"text-gray-500",children:"No special capabilities listed"}):O.map((e,a)=>(0,t.jsx)(p.Tag,{color:M[a%M.length],children:eG(e)},e)))})]}),(eN.tpm||eN.rpm)&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Rate Limits"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[eN.tpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Tokens per Minute:"}),(0,t.jsx)(n.Text,{children:eN.tpm.toLocaleString()})]}),eN.rpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Requests per Minute:"}),(0,t.jsx)(n.Text,{children:eN.rpm.toLocaleString()})]})]})]}),eN.supported_openai_params&&eN.supported_openai_params.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Supported OpenAI Parameters"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:eN.supported_openai_params.map(e=>(0,t.jsx)(p.Tag,{color:"green",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:(0,j.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,A.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"})})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eB((0,j.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,A.getEndpointType)(eN.mode||"chat"),selectedModel:eN.model_group,selectedSdk:"openai"}))},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eS?.name||"Agent Details"}),eS&&(0,t.jsx)(u.Tooltip,{title:"Copy agent name",children:(0,t.jsx)(g.Copy,{onClick:()=>eB(eS.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ey,footer:null,onOk:()=>{ev(!1),eC(null)},onCancel:()=>{ev(!1),eC(null)},children:eS&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Agent Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Name:"}),(0,t.jsx)(n.Text,{children:eS.name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Version:"}),(0,t.jsx)(n.Text,{children:eS.version})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:eS.description})]}),eS.url&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsx)("a",{href:eS.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all",children:eS.url})]})]})]}),eS.capabilities&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:Object.entries(eS.capabilities).filter(([e,t])=>!0===t).map(([e])=>(0,t.jsx)(p.Tag,{color:"green",className:"capitalize",children:e},e))})]}),eS.skills&&eS.skills.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Skills"}),(0,t.jsx)("div",{className:"space-y-4",children:eS.skills.map((e,a)=>(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"flex items-start justify-between mb-2",children:(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-base",children:e.name}),(0,t.jsx)(n.Text,{className:"text-sm text-gray-600",children:e.description})]})}),e.tags&&e.tags.length>0&&(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-2",children:e.tags.map(e=>(0,t.jsx)(p.Tag,{color:"purple",className:"text-xs",children:e},e))})]},a))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Input/Output Modes"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eS.defaultInputModes??[]).map(e=>(0,t.jsx)(p.Tag,{color:"blue",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eS.defaultOutputModes??[]).map(e=>(0,t.jsx)(p.Tag,{color:"blue",children:e},e))})]})]})]}),eS.documentationUrl&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Documentation"}),(0,t.jsxs)("a",{href:eS.documentationUrl,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 flex items-center space-x-2",children:[(0,t.jsx)(s.ExternalLinkIcon,{className:"w-4 h-4"}),(0,t.jsx)("span",{children:"View Documentation"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example (A2A Protocol)"}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 1: Retrieve Agent Card"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`base_url = '${eS.url}'

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
print(response.model_dump(mode='json', exclude_none=True))`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ew?.server_name||"MCP Server Details"}),ew&&(0,t.jsx)(u.Tooltip,{title:"Copy server name",children:(0,t.jsx)(g.Copy,{onClick:()=>eB(ew.server_name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ej,footer:null,onOk:()=>{eA(!1),eI(null)},onCancel:()=>{eA(!1),eI(null)},children:ew&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Server Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Server Name:"}),(0,t.jsx)(n.Text,{children:ew.server_name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Transport:"}),(0,t.jsx)(p.Tag,{color:"blue",children:ew.transport})]}),ew.alias&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Alias:"}),(0,t.jsx)(n.Text,{children:ew.alias})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Auth Type:"}),(0,t.jsx)(p.Tag,{color:"none"===ew.auth_type?"gray":"green",children:ew.auth_type})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:ew.mcp_info?.description||"-"})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsxs)("a",{href:ew.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ew.url}),(0,t.jsx)(s.ExternalLinkIcon,{className:"w-4 h-4"})]})]})]})]}),ew.mcp_info&&Object.keys(ew.mcp_info).length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Additional Information"}),(0,t.jsx)("div",{className:"bg-gray-50 p-4 rounded-lg",children:(0,t.jsx)("pre",{className:"text-xs overflow-x-auto",children:JSON.stringify(ew.mcp_info,null,2)})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${ew.server_name}": {
            "url": "http://localhost:4000/${ew.server_name}/mcp",
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
        "${ew.server_name}": {
            "url": "http://localhost:4000/${ew.server_name}/mcp",
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