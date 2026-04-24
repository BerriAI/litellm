(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,221345,e=>{"use strict";let t=(0,e.i(475254).default)("link",[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]]);e.s(["Link",()=>t],221345)},190272,785913,e=>{"use strict";var t,o,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((o={}).IMAGE="image",o.VIDEO="video",o.CHAT="chat",o.RESPONSES="responses",o.IMAGE_EDITS="image_edits",o.ANTHROPIC_MESSAGES="anthropic_messages",o.EMBEDDINGS="embeddings",o.SPEECH="speech",o.TRANSCRIPTION="transcription",o.A2A_AGENTS="a2a_agents",o.MCP="mcp",o.REALTIME="realtime",o);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:o,accessToken:i,apiKey:r,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:d,selectedGuardrails:p,selectedPolicies:c,selectedMCPServers:m,mcpServers:u,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:x,proxySettings:_}=e,y="session"===o?i:r,v=window.location.origin,w=_?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:_?.PROXY_BASE_URL&&(v=_.PROXY_BASE_URL);let k=n||"Your prompt here",j=k.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),N={};l.length>0&&(N.tags=l),d.length>0&&(N.vector_stores=d),p.length>0&&(N.guardrails=p),c.length>0&&(N.policies=c);let E=b||"your-model-name",I="azure"===x?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case a.CHAT:{let e=Object.keys(N).length>0,o="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let i=S.length>0?S:[{role:"user",content:k}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${E}",
    messages=${JSON.stringify(i,null,4)}${o}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${E}",
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
#     ]${o}
# )
# print(response_with_file)
`;break}case a.RESPONSES:{let e=Object.keys(N).length>0,o="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let i=S.length>0?S:[{role:"user",content:k}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${E}",
    input=${JSON.stringify(i,null,4)}${o}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${E}",
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
#     ]${o}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===x?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${E}",
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
prompt = "${j}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${E}",
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
`;break;case a.IMAGE_EDITS:t="azure"===x?`
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
	model="${E}",
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
	model="${E}",
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
`;break;case a.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${E}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${E}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${E}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${E}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},514764,e=>{"use strict";let t=(0,e.i(475254).default)("send",[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]]);e.s(["Send",()=>t],514764)},382373,387951,e=>{"use strict";var t=e.i(475254);let o=(0,t.default)("volume-2",[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]]);e.s(["Volume2",()=>o],382373);let i=(0,t.default)("mic",[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]]);e.s(["Mic",()=>i],387951)},975558,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-up",[["path",{d:"m5 12 7-7 7 7",key:"hav0vg"}],["path",{d:"M12 19V5",key:"x0mq9r"}]]);e.s(["ArrowUp",()=>t],975558)},989359,989022,e=>{"use strict";var t=e.i(475254);let o=(0,t.default)("eraser",[["path",{d:"M21 21H8a2 2 0 0 1-1.42-.587l-3.994-3.999a2 2 0 0 1 0-2.828l10-10a2 2 0 0 1 2.829 0l5.999 6a2 2 0 0 1 0 2.828L12.834 21",key:"g5wo59"}],["path",{d:"m5.082 11.09 8.828 8.828",key:"1wx5vj"}]]);e.s(["Eraser",()=>o],989359);var i=e.i(843476),a=e.i(746798),r=e.i(503116),n=e.i(212426);let s=(0,t.default)("hash",[["line",{x1:"4",x2:"20",y1:"9",y2:"9",key:"4lhtct"}],["line",{x1:"4",x2:"20",y1:"15",y2:"15",key:"vyu0kd"}],["line",{x1:"10",x2:"8",y1:"3",y2:"21",key:"1ggp8o"}],["line",{x1:"16",x2:"14",y1:"3",y2:"21",key:"weycgp"}]]);var l=e.i(204997),d=e.i(292270),p=e.i(341240),c=e.i(195116);let m=({tooltip:e,icon:t,children:o})=>(0,i.jsx)(a.TooltipProvider,{children:(0,i.jsxs)(a.Tooltip,{children:[(0,i.jsx)(a.TooltipTrigger,{asChild:!0,children:(0,i.jsxs)("div",{className:"flex items-center",children:[t,(0,i.jsx)("span",{children:o})]})}),(0,i.jsx)(a.TooltipContent,{children:e})]})});e.s(["default",0,({timeToFirstToken:e,totalLatency:t,usage:o,toolName:a})=>e||t||o?(0,i.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-border text-xs text-muted-foreground flex flex-wrap gap-3",children:[void 0!==e&&(0,i.jsxs)(m,{tooltip:"Time to first token",icon:(0,i.jsx)(r.Clock,{className:"h-3 w-3 mr-1"}),children:["TTFT: ",(e/1e3).toFixed(2),"s"]}),void 0!==t&&(0,i.jsxs)(m,{tooltip:"Total latency",icon:(0,i.jsx)(r.Clock,{className:"h-3 w-3 mr-1"}),children:["Total Latency: ",(t/1e3).toFixed(2),"s"]}),o?.promptTokens!==void 0&&(0,i.jsxs)(m,{tooltip:"Prompt tokens",icon:(0,i.jsx)(l.LogIn,{className:"h-3 w-3 mr-1"}),children:["In: ",o.promptTokens]}),o?.completionTokens!==void 0&&(0,i.jsxs)(m,{tooltip:"Completion tokens",icon:(0,i.jsx)(d.LogOut,{className:"h-3 w-3 mr-1"}),children:["Out: ",o.completionTokens]}),o?.reasoningTokens!==void 0&&(0,i.jsxs)(m,{tooltip:"Reasoning tokens",icon:(0,i.jsx)(p.Lightbulb,{className:"h-3 w-3 mr-1"}),children:["Reasoning: ",o.reasoningTokens]}),o?.totalTokens!==void 0&&(0,i.jsxs)(m,{tooltip:"Total tokens",icon:(0,i.jsx)(s,{className:"h-3 w-3 mr-1"}),children:["Total: ",o.totalTokens]}),o?.cost!==void 0&&(0,i.jsxs)(m,{tooltip:"Cost",icon:(0,i.jsx)(n.DollarSign,{className:"h-3 w-3 mr-1"}),children:["$",o.cost.toFixed(6)]}),a&&(0,i.jsxs)(m,{tooltip:"Tool used",icon:(0,i.jsx)(c.Wrench,{className:"h-3 w-3 mr-1"}),children:["Tool: ",a]})]}):null],989022)},212426,e=>{"use strict";let t=(0,e.i(475254).default)("dollar-sign",[["line",{x1:"12",x2:"12",y1:"2",y2:"22",key:"7eqyqh"}],["path",{d:"M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6",key:"1b0p4s"}]]);e.s(["DollarSign",()=>t],212426)},204997,e=>{"use strict";let t=(0,e.i(475254).default)("log-in",[["path",{d:"m10 17 5-5-5-5",key:"1bsop3"}],["path",{d:"M15 12H3",key:"6jk70r"}],["path",{d:"M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4",key:"u53s6r"}]]);e.s(["LogIn",()=>t],204997)},219470,341240,e=>{"use strict";e.s(["coy",0,{'code[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",maxHeight:"inherit",height:"inherit",padding:"0 1em",display:"block",overflow:"auto"},'pre[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",position:"relative",margin:".5em 0",overflow:"visible",padding:"1px",backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em"},'pre[class*="language-"] > code':{position:"relative",zIndex:"1",borderLeft:"10px solid #358ccb",boxShadow:"-1px 0px 0px 0px #358ccb, 0px 0px 0px 1px #dfdfdf",backgroundColor:"#fdfdfd",backgroundImage:"linear-gradient(transparent 50%, rgba(69, 142, 209, 0.04) 50%)",backgroundSize:"3em 3em",backgroundOrigin:"content-box",backgroundAttachment:"local"},':not(pre) > code[class*="language-"]':{backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em",position:"relative",padding:".2em",borderRadius:"0.3em",color:"#c92c2c",border:"1px solid rgba(0, 0, 0, 0.1)",display:"inline",whiteSpace:"normal"},'pre[class*="language-"]:before':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"0.18em",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(-2deg)",MozTransform:"rotate(-2deg)",msTransform:"rotate(-2deg)",OTransform:"rotate(-2deg)",transform:"rotate(-2deg)"},'pre[class*="language-"]:after':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"auto",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(2deg)",MozTransform:"rotate(2deg)",msTransform:"rotate(2deg)",OTransform:"rotate(2deg)",transform:"rotate(2deg)",right:"0.75em"},comment:{color:"#7D8B99"},"block-comment":{color:"#7D8B99"},prolog:{color:"#7D8B99"},doctype:{color:"#7D8B99"},cdata:{color:"#7D8B99"},punctuation:{color:"#5F6364"},property:{color:"#c92c2c"},tag:{color:"#c92c2c"},boolean:{color:"#c92c2c"},number:{color:"#c92c2c"},"function-name":{color:"#c92c2c"},constant:{color:"#c92c2c"},symbol:{color:"#c92c2c"},deleted:{color:"#c92c2c"},selector:{color:"#2f9c0a"},"attr-name":{color:"#2f9c0a"},string:{color:"#2f9c0a"},char:{color:"#2f9c0a"},function:{color:"#2f9c0a"},builtin:{color:"#2f9c0a"},inserted:{color:"#2f9c0a"},operator:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},entity:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)",cursor:"help"},url:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},variable:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},atrule:{color:"#1990b8"},"attr-value":{color:"#1990b8"},keyword:{color:"#1990b8"},"class-name":{color:"#1990b8"},regex:{color:"#e90"},important:{color:"#e90",fontWeight:"normal"},".language-css .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},".style .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:".7"},'pre[class*="language-"].line-numbers.line-numbers':{paddingLeft:"0"},'pre[class*="language-"].line-numbers.line-numbers code':{paddingLeft:"3.8em"},'pre[class*="language-"].line-numbers.line-numbers .line-numbers-rows':{left:"0"},'pre[class*="language-"][data-line]':{paddingTop:"0",paddingBottom:"0",paddingLeft:"0"},"pre[data-line] code":{position:"relative",paddingLeft:"4em"},"pre .line-highlight":{marginTop:"0"}}],219470);let t=(0,e.i(475254).default)("lightbulb",[["path",{d:"M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5",key:"1gvzjb"}],["path",{d:"M9 18h6",key:"x1upvd"}],["path",{d:"M10 22h4",key:"ceow96"}]]);e.s(["Lightbulb",()=>t],341240)},286536,e=>{"use strict";let t=(0,e.i(475254).default)("eye",[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["Eye",()=>t],286536)},77705,e=>{"use strict";let t=(0,e.i(475254).default)("eye-off",[["path",{d:"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49",key:"ct8e1f"}],["path",{d:"M14.084 14.158a3 3 0 0 1-4.242-4.242",key:"151rxh"}],["path",{d:"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143",key:"13bj9a"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]]);e.s(["EyeOff",()=>t],77705)},611052,270756,e=>{"use strict";var t=e.i(843476),o=e.i(271645),i=e.i(776639),a=e.i(519455),r=e.i(793479),n=e.i(110204),s=e.i(699375),l=e.i(888259),d=e.i(871689),p=e.i(972520),c=e.i(643531),m=e.i(286536),u=e.i(77705),g=e.i(834161),f=e.i(221345);let h=(0,e.i(475254).default)("lock",[["rect",{width:"18",height:"11",x:"3",y:"11",rx:"2",ry:"2",key:"1w4ew1"}],["path",{d:"M7 11V7a5 5 0 0 1 10 0v4",key:"fwvmzm"}]]);e.s(["Lock",()=>h],270756);var b=e.i(37727),x=e.i(975157);e.s(["ByokCredentialModal",0,({server:e,open:_,onClose:y,onSuccess:v,accessToken:w})=>{let[k,j]=(0,o.useState)(1),[S,N]=(0,o.useState)(""),[E,I]=(0,o.useState)(!1),[M,T]=(0,o.useState)(!0),[C,A]=(0,o.useState)(!1),P=e.alias||e.server_name||"Service",R=P.charAt(0).toUpperCase(),D=()=>{j(1),N(""),I(!1),T(!0),A(!1),y()},O=async()=>{if(!S.trim())return void l.default.error("Please enter your API key");A(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${w}`},body:JSON.stringify({credential:S.trim(),save:M})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}l.default.success(`Connected to ${P}`),v(e.server_id),D()}catch(e){l.default.error(e.message||"Failed to connect")}finally{A(!1)}};return(0,t.jsx)(i.Dialog,{open:_,onOpenChange:e=>e?void 0:D(),children:(0,t.jsx)(i.DialogContent,{className:"max-w-md p-0 overflow-hidden",children:(0,t.jsxs)("div",{className:"relative p-6",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===k?(0,t.jsxs)("button",{onClick:()=>j(1),className:"flex items-center gap-1 text-muted-foreground hover:text-foreground text-sm",children:[(0,t.jsx)(d.ArrowLeft,{className:"h-3.5 w-3.5"})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:(0,x.cn)("w-2 h-2 rounded-full",1===k?"bg-primary":"bg-muted")}),(0,t.jsx)("div",{className:(0,x.cn)("w-2 h-2 rounded-full",2===k?"bg-primary":"bg-muted")})]}),(0,t.jsx)("button",{onClick:D,className:"text-muted-foreground hover:text-foreground","aria-label":"Close",children:(0,t.jsx)(b.X,{className:"h-4 w-4"})})]}),1===k?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(p.ArrowRight,{className:"text-muted-foreground h-4 w-4"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:R})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:["Connect ",P]}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["LiteLLM needs access to ",P," to complete your request."]}),(0,t.jsx)("div",{className:"bg-muted rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-foreground mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-muted-foreground text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to ",P,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-emerald-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,o)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-foreground",children:[(0,t.jsx)(c.Check,{className:"text-emerald-500 flex-shrink-0 h-4 w-4"}),e]},o))})]}),(0,t.jsxs)(a.Button,{onClick:()=>j(2),className:"w-full",children:["Continue to Authentication ",(0,t.jsx)(p.ArrowRight,{className:"h-4 w-4"})]}),(0,t.jsx)(a.Button,{onClick:D,variant:"ghost",className:"mt-3 w-full text-muted-foreground",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4",children:(0,t.jsx)(g.Key,{className:"text-primary h-5 w-5"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["Enter your ",P," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4 space-y-2",children:[(0,t.jsxs)(n.Label,{htmlFor:"byok-api-key",children:[P," API Key"]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(r.Input,{id:"byok-api-key",type:E?"text":"password",placeholder:"Enter your API key",value:S,onChange:e=>N(e.target.value)}),(0,t.jsx)("button",{type:"button",onClick:()=>I(!E),className:"absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground","aria-label":E?"Hide api key":"Show api key",children:E?(0,t.jsx)(u.EyeOff,{size:14}):(0,t.jsx)(m.Eye,{size:14})})]}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-primary hover:underline text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key?"," ",(0,t.jsx)(f.Link,{className:"h-3.5 w-3.5"})]})]}),(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-foreground",children:"Save key for future use"})]}),(0,t.jsx)(s.Switch,{checked:M,onCheckedChange:T})]}),(0,t.jsxs)("div",{className:"bg-primary/5 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(h,{className:"text-primary mt-0.5 flex-shrink-0 h-4 w-4"}),(0,t.jsx)("p",{className:"text-sm text-primary",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)(a.Button,{onClick:O,disabled:C,className:"w-full",children:[(0,t.jsx)(h,{className:"h-4 w-4"})," Connect & Authorize"]})]})]})})})}],611052)},367692,e=>{"use strict";var t=e.i(843476),o=e.i(271645),i=e.i(470152),a=e.i(981140),r=e.i(820783),n=e.i(30030),s=e.i(369340),l=e.i(586318),d=e.i(999682),p=e.i(635804),c=e.i(248425),m=e.i(75830),u=["PageUp","PageDown"],g=["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"],f={"from-left":["Home","PageDown","ArrowDown","ArrowLeft"],"from-right":["Home","PageDown","ArrowDown","ArrowRight"],"from-bottom":["Home","PageDown","ArrowDown","ArrowLeft"],"from-top":["Home","PageDown","ArrowUp","ArrowLeft"]},h="Slider",[b,x,_]=(0,m.createCollection)(h),[y,v]=(0,n.createContextScope)(h,[_]),[w,k]=y(h),j=o.forwardRef((e,r)=>{let{name:n,min:l=0,max:d=100,step:p=1,orientation:c="horizontal",disabled:m=!1,minStepsBetweenThumbs:f=0,defaultValue:h=[l],value:x,onValueChange:_=()=>{},onValueCommit:y=()=>{},inverted:v=!1,form:k,...j}=e,S=o.useRef(new Set),N=o.useRef(0),M="horizontal"===c,[T=[],C]=(0,s.useControllableState)({prop:x,defaultProp:h,onChange:e=>{let t=[...S.current];t[N.current]?.focus(),_(e)}}),A=o.useRef(T);function P(e,t,{commit:o}={commit:!1}){let a,r=(String(p).split(".")[1]||"").length,n=Math.round((Math.round((e-l)/p)*p+l)*(a=Math.pow(10,r)))/a,s=(0,i.clamp)(n,[l,d]);C((e=[])=>{let i=function(e=[],t,o){let i=[...e];return i[o]=t,i.sort((e,t)=>e-t)}(e,s,t);if(!function(e,t){if(t>0)return Math.min(...e.slice(0,-1).map((t,o)=>e[o+1]-t))>=t;return!0}(i,f*p))return e;{N.current=i.indexOf(s);let t=String(i)!==String(e);return t&&o&&y(i),t?i:e}})}return(0,t.jsx)(w,{scope:e.__scopeSlider,name:n,disabled:m,min:l,max:d,valueIndexToChangeRef:N,thumbs:S.current,values:T,orientation:c,form:k,children:(0,t.jsx)(b.Provider,{scope:e.__scopeSlider,children:(0,t.jsx)(b.Slot,{scope:e.__scopeSlider,children:(0,t.jsx)(M?E:I,{"aria-disabled":m,"data-disabled":m?"":void 0,...j,ref:r,onPointerDown:(0,a.composeEventHandlers)(j.onPointerDown,()=>{m||(A.current=T)}),min:l,max:d,inverted:v,onSlideStart:m?void 0:function(e){let t=function(e,t){if(1===e.length)return 0;let o=e.map(e=>Math.abs(e-t)),i=Math.min(...o);return o.indexOf(i)}(T,e);P(e,t)},onSlideMove:m?void 0:function(e){P(e,N.current)},onSlideEnd:m?void 0:function(){let e=A.current[N.current];T[N.current]!==e&&y(T)},onHomeKeyDown:()=>!m&&P(l,0,{commit:!0}),onEndKeyDown:()=>!m&&P(d,T.length-1,{commit:!0}),onStepKeyDown:({event:e,direction:t})=>{if(!m){let o=u.includes(e.key)||e.shiftKey&&g.includes(e.key),i=N.current;P(T[i]+p*(o?10:1)*t,i,{commit:!0})}}})})})})});j.displayName=h;var[S,N]=y(h,{startEdge:"left",endEdge:"right",size:"width",direction:1}),E=o.forwardRef((e,i)=>{let{min:a,max:n,dir:s,inverted:d,onSlideStart:p,onSlideMove:c,onSlideEnd:m,onStepKeyDown:u,...g}=e,[h,b]=o.useState(null),x=(0,r.useComposedRefs)(i,e=>b(e)),_=o.useRef(void 0),y=(0,l.useDirection)(s),v="ltr"===y,w=v&&!d||!v&&d;function k(e){let t=_.current||h.getBoundingClientRect(),o=H([0,t.width],w?[a,n]:[n,a]);return _.current=t,o(e-t.left)}return(0,t.jsx)(S,{scope:e.__scopeSlider,startEdge:w?"left":"right",endEdge:w?"right":"left",direction:w?1:-1,size:"width",children:(0,t.jsx)(M,{dir:y,"data-orientation":"horizontal",...g,ref:x,style:{...g.style,"--radix-slider-thumb-transform":"translateX(-50%)"},onSlideStart:e=>{let t=k(e.clientX);p?.(t)},onSlideMove:e=>{let t=k(e.clientX);c?.(t)},onSlideEnd:()=>{_.current=void 0,m?.()},onStepKeyDown:e=>{let t=f[w?"from-left":"from-right"].includes(e.key);u?.({event:e,direction:t?-1:1})}})})}),I=o.forwardRef((e,i)=>{let{min:a,max:n,inverted:s,onSlideStart:l,onSlideMove:d,onSlideEnd:p,onStepKeyDown:c,...m}=e,u=o.useRef(null),g=(0,r.useComposedRefs)(i,u),h=o.useRef(void 0),b=!s;function x(e){let t=h.current||u.current.getBoundingClientRect(),o=H([0,t.height],b?[n,a]:[a,n]);return h.current=t,o(e-t.top)}return(0,t.jsx)(S,{scope:e.__scopeSlider,startEdge:b?"bottom":"top",endEdge:b?"top":"bottom",size:"height",direction:b?1:-1,children:(0,t.jsx)(M,{"data-orientation":"vertical",...m,ref:g,style:{...m.style,"--radix-slider-thumb-transform":"translateY(50%)"},onSlideStart:e=>{let t=x(e.clientY);l?.(t)},onSlideMove:e=>{let t=x(e.clientY);d?.(t)},onSlideEnd:()=>{h.current=void 0,p?.()},onStepKeyDown:e=>{let t=f[b?"from-bottom":"from-top"].includes(e.key);c?.({event:e,direction:t?-1:1})}})})}),M=o.forwardRef((e,o)=>{let{__scopeSlider:i,onSlideStart:r,onSlideMove:n,onSlideEnd:s,onHomeKeyDown:l,onEndKeyDown:d,onStepKeyDown:p,...m}=e,f=k(h,i);return(0,t.jsx)(c.Primitive.span,{...m,ref:o,onKeyDown:(0,a.composeEventHandlers)(e.onKeyDown,e=>{"Home"===e.key?(l(e),e.preventDefault()):"End"===e.key?(d(e),e.preventDefault()):u.concat(g).includes(e.key)&&(p(e),e.preventDefault())}),onPointerDown:(0,a.composeEventHandlers)(e.onPointerDown,e=>{let t=e.target;t.setPointerCapture(e.pointerId),e.preventDefault(),f.thumbs.has(t)?t.focus():r(e)}),onPointerMove:(0,a.composeEventHandlers)(e.onPointerMove,e=>{e.target.hasPointerCapture(e.pointerId)&&n(e)}),onPointerUp:(0,a.composeEventHandlers)(e.onPointerUp,e=>{let t=e.target;t.hasPointerCapture(e.pointerId)&&(t.releasePointerCapture(e.pointerId),s(e))})})}),T="SliderTrack",C=o.forwardRef((e,o)=>{let{__scopeSlider:i,...a}=e,r=k(T,i);return(0,t.jsx)(c.Primitive.span,{"data-disabled":r.disabled?"":void 0,"data-orientation":r.orientation,...a,ref:o})});C.displayName=T;var A="SliderRange",P=o.forwardRef((e,i)=>{let{__scopeSlider:a,...n}=e,s=k(A,a),l=N(A,a),d=o.useRef(null),p=(0,r.useComposedRefs)(i,d),m=s.values.length,u=s.values.map(e=>L(e,s.min,s.max)),g=m>1?Math.min(...u):0,f=100-Math.max(...u);return(0,t.jsx)(c.Primitive.span,{"data-orientation":s.orientation,"data-disabled":s.disabled?"":void 0,...n,ref:p,style:{...e.style,[l.startEdge]:g+"%",[l.endEdge]:f+"%"}})});P.displayName=A;var R="SliderThumb",D=o.forwardRef((e,i)=>{let a=x(e.__scopeSlider),[n,s]=o.useState(null),l=(0,r.useComposedRefs)(i,e=>s(e)),d=o.useMemo(()=>n?a().findIndex(e=>e.ref.current===n):-1,[a,n]);return(0,t.jsx)(O,{...e,ref:l,index:d})}),O=o.forwardRef((e,i)=>{var n,s,l,d,m;let u,g,{__scopeSlider:f,index:h,name:x,..._}=e,y=k(R,f),v=N(R,f),[w,j]=o.useState(null),S=(0,r.useComposedRefs)(i,e=>j(e)),E=!w||y.form||!!w.closest("form"),I=(0,p.useSize)(w),M=y.values[h],T=void 0===M?0:L(M,y.min,y.max),C=(n=h,(s=y.values.length)>2?`Value ${n+1} of ${s}`:2===s?["Minimum","Maximum"][n]:void 0),A=I?.[v.size],P=A?(l=A,d=T,m=v.direction,g=H([0,50],[0,u=l/2]),(u-g(d)*m)*m):0;return o.useEffect(()=>{if(w)return y.thumbs.add(w),()=>{y.thumbs.delete(w)}},[w,y.thumbs]),(0,t.jsxs)("span",{style:{transform:"var(--radix-slider-thumb-transform)",position:"absolute",[v.startEdge]:`calc(${T}% + ${P}px)`},children:[(0,t.jsx)(b.ItemSlot,{scope:e.__scopeSlider,children:(0,t.jsx)(c.Primitive.span,{role:"slider","aria-label":e["aria-label"]||C,"aria-valuemin":y.min,"aria-valuenow":M,"aria-valuemax":y.max,"aria-orientation":y.orientation,"data-orientation":y.orientation,"data-disabled":y.disabled?"":void 0,tabIndex:y.disabled?void 0:0,..._,ref:S,style:void 0===M?{display:"none"}:e.style,onFocus:(0,a.composeEventHandlers)(e.onFocus,()=>{y.valueIndexToChangeRef.current=h})})}),E&&(0,t.jsx)(z,{name:x??(y.name?y.name+(y.values.length>1?"[]":""):void 0),form:y.form,value:M},h)]})});D.displayName=R;var z=o.forwardRef(({__scopeSlider:e,value:i,...a},n)=>{let s=o.useRef(null),l=(0,r.useComposedRefs)(s,n),p=(0,d.usePrevious)(i);return o.useEffect(()=>{let e=s.current;if(!e)return;let t=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set;if(p!==i&&t){let o=new Event("input",{bubbles:!0});t.call(e,i),e.dispatchEvent(o)}},[p,i]),(0,t.jsx)(c.Primitive.input,{style:{display:"none"},...a,ref:l,defaultValue:i})});function L(e,t,o){return(0,i.clamp)(100/(o-t)*(e-t),[0,100])}function H(e,t){return o=>{if(e[0]===e[1]||t[0]===t[1])return t[0];let i=(t[1]-t[0])/(e[1]-e[0]);return t[0]+i*(o-e[0])}}z.displayName="RadioBubbleInput";var $=e.i(975157);let B=o.forwardRef(({className:e,...o},i)=>(0,t.jsxs)(j,{ref:i,className:(0,$.cn)("relative flex w-full touch-none select-none items-center",e),...o,children:[(0,t.jsx)(C,{className:"relative h-2 w-full grow overflow-hidden rounded-full bg-secondary",children:(0,t.jsx)(P,{className:"absolute h-full bg-primary"})}),(0,t.jsx)(D,{className:"block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"})]}));B.displayName=j.displayName,e.s(["Slider",()=>B],367692)}]);