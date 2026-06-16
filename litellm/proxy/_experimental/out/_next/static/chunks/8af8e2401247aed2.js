(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,190272,785913,e=>{"use strict";var t,a,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a.INTERACTIONS="interactions",a);let o={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=o[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:r,apiKey:o,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:d,selectedGuardrails:m,selectedPolicies:g,selectedMCPServers:p,mcpServers:c,mcpServerToolRestrictions:u,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:_,proxySettings:x}=e,C="session"===a?r:o,v=window.location.origin,w=x?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:x?.PROXY_BASE_URL&&(v=x.PROXY_BASE_URL);let $=n||"Your prompt here",k=$.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),y=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),d.length>0&&(E.vector_stores=d),m.length>0&&(E.guardrails=m),g.length>0&&(E.policies=g);let O=b||"your-model-name",j="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${C||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${C||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case i.CHAT:{let e=Object.keys(E).length>0,a="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let r=y.length>0?y:[{role:"user",content:$}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${O}",
    messages=${JSON.stringify(r,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${O}",
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
#     ]${a}
# )
# print(response_with_file)
`;break}case i.RESPONSES:{let e=Object.keys(E).length>0,a="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let r=y.length>0?y:[{role:"user",content:$}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${O}",
    input=${JSON.stringify(r,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${O}",
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
#     ]${a}
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
	model="${O}",
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
prompt = "${k}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${O}",
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
	model="${O}",
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
	model="${O}",
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
	model="${O}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${O}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${O}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${O}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${j}
${t}`}],190272)},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var i=e.i(9583),o=a.forwardRef(function(e,o){return a.createElement(i.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["ArrowLeftOutlined",0,o],447566)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},304967,e=>{"use strict";var t=e.i(290571),a=e.i(271645),r=e.i(480731),i=e.i(95779),o=e.i(444755),n=e.i(673706);let s=(0,n.makeClassName)("Card"),l=a.default.forwardRef((e,l)=>{let{decoration:d="",decorationColor:m,children:g,className:p}=e,c=(0,t.__rest)(e,["decoration","decorationColor","children","className"]);return a.default.createElement("div",Object.assign({ref:l,className:(0,o.tremorTwMerge)(s("root"),"relative w-full text-left ring-1 rounded-tremor-default p-6","bg-tremor-background ring-tremor-ring shadow-tremor-card","dark:bg-dark-tremor-background dark:ring-dark-tremor-ring dark:shadow-dark-tremor-card",m?(0,n.getColorClassNames)(m,i.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",(e=>{if(!e)return"";switch(e){case r.HorizontalPositions.Left:return"border-l-4";case r.VerticalPositions.Top:return"border-t-4";case r.HorizontalPositions.Right:return"border-r-4";case r.VerticalPositions.Bottom:return"border-b-4";default:return""}})(d),p)},c),g)});l.displayName="Card",e.s(["Card",()=>l],304967)},629569,e=>{"use strict";var t=e.i(290571),a=e.i(95779),r=e.i(444755),i=e.i(673706),o=e.i(271645);let n=o.default.forwardRef((e,n)=>{let{color:s,children:l,className:d}=e,m=(0,t.__rest)(e,["color","children","className"]);return o.default.createElement("p",Object.assign({ref:n,className:(0,r.tremorTwMerge)("font-medium text-tremor-title",s?(0,i.getColorClassNames)(s,a.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",d)},m),l)});n.displayName="Title",e.s(["Title",()=>n],629569)},994388,e=>{"use strict";var t=e.i(290571),a=e.i(829087),r=e.i(271645);let i=["preEnter","entering","entered","preExit","exiting","exited","unmounted"],o=e=>({_s:e,status:i[e],isEnter:e<3,isMounted:6!==e,isResolved:2===e||e>4}),n=e=>e?6:5,s=(e,t,a,r,i)=>{clearTimeout(r.current);let n=o(e);t(n),a.current=n,i&&i({current:n})};var l=e.i(480731),d=e.i(444755),m=e.i(673706);let g=e=>{var a=(0,t.__rest)(e,[]);return r.default.createElement("svg",Object.assign({},a,{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 24 24",fill:"currentColor"}),r.default.createElement("path",{fill:"none",d:"M0 0h24v24H0z"}),r.default.createElement("path",{d:"M18.364 5.636L16.95 7.05A7 7 0 1 0 19 12h2a9 9 0 1 1-2.636-6.364z"}))};var p=e.i(95779);let c={xs:{height:"h-4",width:"w-4"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-6",width:"w-6"},xl:{height:"h-6",width:"w-6"}},u=(e,t)=>{switch(e){case"primary":return{textColor:t?(0,m.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",hoverTextColor:t?(0,m.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,m.getColorClassNames)(t,p.colorPalette.background).bgColor:"bg-tremor-brand dark:bg-dark-tremor-brand",hoverBgColor:t?(0,m.getColorClassNames)(t,p.colorPalette.darkBackground).hoverBgColor:"hover:bg-tremor-brand-emphasis dark:hover:bg-dark-tremor-brand-emphasis",borderColor:t?(0,m.getColorClassNames)(t,p.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",hoverBorderColor:t?(0,m.getColorClassNames)(t,p.colorPalette.darkBorder).hoverBorderColor:"hover:border-tremor-brand-emphasis dark:hover:border-dark-tremor-brand-emphasis"};case"secondary":return{textColor:t?(0,m.getColorClassNames)(t,p.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,m.getColorClassNames)(t,p.colorPalette.text).textColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,m.getColorClassNames)("transparent").bgColor,hoverBgColor:t?(0,d.tremorTwMerge)((0,m.getColorClassNames)(t,p.colorPalette.background).hoverBgColor,"hover:bg-opacity-20 dark:hover:bg-opacity-20"):"hover:bg-tremor-brand-faint dark:hover:bg-dark-tremor-brand-faint",borderColor:t?(0,m.getColorClassNames)(t,p.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand"};case"light":return{textColor:t?(0,m.getColorClassNames)(t,p.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,m.getColorClassNames)(t,p.colorPalette.darkText).hoverTextColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,m.getColorClassNames)("transparent").bgColor,borderColor:"",hoverBorderColor:""}}},f=(0,m.makeClassName)("Button"),h=({loading:e,iconSize:t,iconPosition:a,Icon:i,needMargin:o,transitionStatus:n})=>{let s=o?a===l.HorizontalPositions.Left?(0,d.tremorTwMerge)("-ml-1","mr-1.5"):(0,d.tremorTwMerge)("-mr-1","ml-1.5"):"",m=(0,d.tremorTwMerge)("w-0 h-0"),p={default:m,entering:m,entered:t,exiting:t,exited:m};return e?r.default.createElement(g,{className:(0,d.tremorTwMerge)(f("icon"),"animate-spin shrink-0",s,p.default,p[n]),style:{transition:"width 150ms"}}):r.default.createElement(i,{className:(0,d.tremorTwMerge)(f("icon"),"shrink-0",t,s)})},b=r.default.forwardRef((e,i)=>{let{icon:g,iconPosition:p=l.HorizontalPositions.Left,size:b=l.Sizes.SM,color:_,variant:x="primary",disabled:C,loading:v=!1,loadingText:w,children:$,tooltip:k,className:y}=e,E=(0,t.__rest)(e,["icon","iconPosition","size","color","variant","disabled","loading","loadingText","children","tooltip","className"]),O=v||C,j=void 0!==g||v,N=v&&w,T=!(!$&&!N),I=(0,d.tremorTwMerge)(c[b].height,c[b].width),S="light"!==x?(0,d.tremorTwMerge)("rounded-tremor-default border","shadow-tremor-input","dark:shadow-dark-tremor-input"):"",P=u(x,_),M=("light"!==x?{xs:{paddingX:"px-2.5",paddingY:"py-1.5",fontSize:"text-xs"},sm:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-sm"},md:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-md"},lg:{paddingX:"px-4",paddingY:"py-2.5",fontSize:"text-lg"},xl:{paddingX:"px-4",paddingY:"py-3",fontSize:"text-xl"}}:{xs:{paddingX:"",paddingY:"",fontSize:"text-xs"},sm:{paddingX:"",paddingY:"",fontSize:"text-sm"},md:{paddingX:"",paddingY:"",fontSize:"text-md"},lg:{paddingX:"",paddingY:"",fontSize:"text-lg"},xl:{paddingX:"",paddingY:"",fontSize:"text-xl"}})[b],{tooltipProps:R,getReferenceProps:A}=(0,a.useTooltip)(300),[z,B]=(({enter:e=!0,exit:t=!0,preEnter:a,preExit:i,timeout:l,initialEntered:d,mountOnEnter:m,unmountOnExit:g,onStateChange:p}={})=>{let[c,u]=(0,r.useState)(()=>o(d?2:n(m))),f=(0,r.useRef)(c),h=(0,r.useRef)(0),[b,_]="object"==typeof l?[l.enter,l.exit]:[l,l],x=(0,r.useCallback)(()=>{let e=((e,t)=>{switch(e){case 1:case 0:return 2;case 4:case 3:return n(t)}})(f.current._s,g);e&&s(e,u,f,h,p)},[p,g]);return[c,(0,r.useCallback)(r=>{let o=e=>{switch(s(e,u,f,h,p),e){case 1:b>=0&&(h.current=((...e)=>setTimeout(...e))(x,b));break;case 4:_>=0&&(h.current=((...e)=>setTimeout(...e))(x,_));break;case 0:case 3:h.current=((...e)=>setTimeout(...e))(()=>{isNaN(document.body.offsetTop)||o(e+1)},0)}},l=f.current.isEnter;"boolean"!=typeof r&&(r=!l),r?l||o(e?+!a:2):l&&o(t?i?3:4:n(g))},[x,p,e,t,a,i,b,_,g]),x]})({timeout:50});return(0,r.useEffect)(()=>{B(v)},[v]),r.default.createElement("button",Object.assign({ref:(0,m.mergeRefs)([i,R.refs.setReference]),className:(0,d.tremorTwMerge)(f("root"),"shrink-0 inline-flex justify-center items-center group font-medium outline-none",S,M.paddingX,M.paddingY,M.fontSize,P.textColor,P.bgColor,P.borderColor,P.hoverBorderColor,O?"opacity-50 cursor-not-allowed":(0,d.tremorTwMerge)(u(x,_).hoverTextColor,u(x,_).hoverBgColor,u(x,_).hoverBorderColor),y),disabled:O},A,E),r.default.createElement(a.default,Object.assign({text:k},R)),j&&p!==l.HorizontalPositions.Right?r.default.createElement(h,{loading:v,iconSize:I,iconPosition:p,Icon:g,transitionStatus:z.status,needMargin:T}):null,N||$?r.default.createElement("span",{className:(0,d.tremorTwMerge)(f("text"),"text-tremor-default whitespace-nowrap")},N?w:$):null,j&&p===l.HorizontalPositions.Right?r.default.createElement(h,{loading:v,iconSize:I,iconPosition:p,Icon:g,transitionStatus:z.status,needMargin:T}):null)});b.displayName="Button",e.s(["Button",()=>b],994388)},959013,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M482 152h60q8 0 8 8v704q0 8-8 8h-60q-8 0-8-8V160q0-8 8-8z"}},{tag:"path",attrs:{d:"M192 474h672q8 0 8 8v60q0 8-8 8H160q-8 0-8-8v-60q0-8 8-8z"}}]},name:"plus",theme:"outlined"};var i=e.i(9583),o=a.forwardRef(function(e,o){return a.createElement(i.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["default",0,o],959013)},185793,e=>{"use strict";e.i(247167);var t=e.i(271645),a=e.i(343794),r=e.i(242064),i=e.i(529681);let o=e=>{let{prefixCls:r,className:i,style:o,size:n,shape:s}=e,l=(0,a.default)({[`${r}-lg`]:"large"===n,[`${r}-sm`]:"small"===n}),d=(0,a.default)({[`${r}-circle`]:"circle"===s,[`${r}-square`]:"square"===s,[`${r}-round`]:"round"===s}),m=t.useMemo(()=>"number"==typeof n?{width:n,height:n,lineHeight:`${n}px`}:{},[n]);return t.createElement("span",{className:(0,a.default)(r,l,d,i),style:Object.assign(Object.assign({},m),o)})};e.i(296059);var n=e.i(694758),s=e.i(915654),l=e.i(246422),d=e.i(838378);let m=new n.Keyframes("ant-skeleton-loading",{"0%":{backgroundPosition:"100% 50%"},"100%":{backgroundPosition:"0 50%"}}),g=e=>({height:e,lineHeight:(0,s.unit)(e)}),p=e=>Object.assign({width:e},g(e)),c=(e,t)=>Object.assign({width:t(e).mul(5).equal(),minWidth:t(e).mul(5).equal()},g(e)),u=e=>Object.assign({width:e},g(e)),f=(e,t,a)=>{let{skeletonButtonCls:r}=e;return{[`${a}${r}-circle`]:{width:t,minWidth:t,borderRadius:"50%"},[`${a}${r}-round`]:{borderRadius:t}}},h=(e,t)=>Object.assign({width:t(e).mul(2).equal(),minWidth:t(e).mul(2).equal()},g(e)),b=(0,l.genStyleHooks)("Skeleton",e=>{let{componentCls:t,calc:a}=e;return(e=>{let{componentCls:t,skeletonAvatarCls:a,skeletonTitleCls:r,skeletonParagraphCls:i,skeletonButtonCls:o,skeletonInputCls:n,skeletonImageCls:s,controlHeight:l,controlHeightLG:d,controlHeightSM:g,gradientFromColor:b,padding:_,marginSM:x,borderRadius:C,titleHeight:v,blockRadius:w,paragraphLiHeight:$,controlHeightXS:k,paragraphMarginTop:y}=e;return{[t]:{display:"table",width:"100%",[`${t}-header`]:{display:"table-cell",paddingInlineEnd:_,verticalAlign:"top",[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:b},p(l)),[`${a}-circle`]:{borderRadius:"50%"},[`${a}-lg`]:Object.assign({},p(d)),[`${a}-sm`]:Object.assign({},p(g))},[`${t}-content`]:{display:"table-cell",width:"100%",verticalAlign:"top",[r]:{width:"100%",height:v,background:b,borderRadius:w,[`+ ${i}`]:{marginBlockStart:g}},[i]:{padding:0,"> li":{width:"100%",height:$,listStyle:"none",background:b,borderRadius:w,"+ li":{marginBlockStart:k}}},[`${i}> li:last-child:not(:first-child):not(:nth-child(2))`]:{width:"61%"}},[`&-round ${t}-content`]:{[`${r}, ${i} > li`]:{borderRadius:C}}},[`${t}-with-avatar ${t}-content`]:{[r]:{marginBlockStart:x,[`+ ${i}`]:{marginBlockStart:y}}},[`${t}${t}-element`]:Object.assign(Object.assign(Object.assign(Object.assign({display:"inline-block",width:"auto"},(e=>{let{borderRadiusSM:t,skeletonButtonCls:a,controlHeight:r,controlHeightLG:i,controlHeightSM:o,gradientFromColor:n,calc:s}=e;return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:n,borderRadius:t,width:s(r).mul(2).equal(),minWidth:s(r).mul(2).equal()},h(r,s))},f(e,r,a)),{[`${a}-lg`]:Object.assign({},h(i,s))}),f(e,i,`${a}-lg`)),{[`${a}-sm`]:Object.assign({},h(o,s))}),f(e,o,`${a}-sm`))})(e)),(e=>{let{skeletonAvatarCls:t,gradientFromColor:a,controlHeight:r,controlHeightLG:i,controlHeightSM:o}=e;return{[t]:Object.assign({display:"inline-block",verticalAlign:"top",background:a},p(r)),[`${t}${t}-circle`]:{borderRadius:"50%"},[`${t}${t}-lg`]:Object.assign({},p(i)),[`${t}${t}-sm`]:Object.assign({},p(o))}})(e)),(e=>{let{controlHeight:t,borderRadiusSM:a,skeletonInputCls:r,controlHeightLG:i,controlHeightSM:o,gradientFromColor:n,calc:s}=e;return{[r]:Object.assign({display:"inline-block",verticalAlign:"top",background:n,borderRadius:a},c(t,s)),[`${r}-lg`]:Object.assign({},c(i,s)),[`${r}-sm`]:Object.assign({},c(o,s))}})(e)),(e=>{let{skeletonImageCls:t,imageSizeBase:a,gradientFromColor:r,borderRadiusSM:i,calc:o}=e;return{[t]:Object.assign(Object.assign({display:"inline-flex",alignItems:"center",justifyContent:"center",verticalAlign:"middle",background:r,borderRadius:i},u(o(a).mul(2).equal())),{[`${t}-path`]:{fill:"#bfbfbf"},[`${t}-svg`]:Object.assign(Object.assign({},u(a)),{maxWidth:o(a).mul(4).equal(),maxHeight:o(a).mul(4).equal()}),[`${t}-svg${t}-svg-circle`]:{borderRadius:"50%"}}),[`${t}${t}-circle`]:{borderRadius:"50%"}}})(e)),[`${t}${t}-block`]:{width:"100%",[o]:{width:"100%"},[n]:{width:"100%"}},[`${t}${t}-active`]:{[`
        ${r},
        ${i} > li,
        ${a},
        ${o},
        ${n},
        ${s}
      `]:Object.assign({},{background:e.skeletonLoadingBackground,backgroundSize:"400% 100%",animationName:m,animationDuration:e.skeletonLoadingMotionDuration,animationTimingFunction:"ease",animationIterationCount:"infinite"})}}})((0,d.mergeToken)(e,{skeletonAvatarCls:`${t}-avatar`,skeletonTitleCls:`${t}-title`,skeletonParagraphCls:`${t}-paragraph`,skeletonButtonCls:`${t}-button`,skeletonInputCls:`${t}-input`,skeletonImageCls:`${t}-image`,imageSizeBase:a(e.controlHeight).mul(1.5).equal(),borderRadius:100,skeletonLoadingBackground:`linear-gradient(90deg, ${e.gradientFromColor} 25%, ${e.gradientToColor} 37%, ${e.gradientFromColor} 63%)`,skeletonLoadingMotionDuration:"1.4s"}))},e=>{let{colorFillContent:t,colorFill:a}=e;return{color:t,colorGradientEnd:a,gradientFromColor:t,gradientToColor:a,titleHeight:e.controlHeight/2,blockRadius:e.borderRadiusSM,paragraphMarginTop:e.marginLG+e.marginXXS,paragraphLiHeight:e.controlHeight/2}},{deprecatedTokens:[["color","gradientFromColor"],["colorGradientEnd","gradientToColor"]]}),_=e=>{let{prefixCls:r,className:i,style:o,rows:n=0}=e,s=Array.from({length:n}).map((a,r)=>t.createElement("li",{key:r,style:{width:((e,t)=>{let{width:a,rows:r=2}=t;return Array.isArray(a)?a[e]:r-1===e?a:void 0})(r,e)}}));return t.createElement("ul",{className:(0,a.default)(r,i),style:o},s)},x=({prefixCls:e,className:r,width:i,style:o})=>t.createElement("h3",{className:(0,a.default)(e,r),style:Object.assign({width:i},o)});function C(e){return e&&"object"==typeof e?e:{}}let v=e=>{let{prefixCls:i,loading:n,className:s,rootClassName:l,style:d,children:m,avatar:g=!1,title:p=!0,paragraph:c=!0,active:u,round:f}=e,{getPrefixCls:h,direction:v,className:w,style:$}=(0,r.useComponentConfig)("skeleton"),k=h("skeleton",i),[y,E,O]=b(k);if(n||!("loading"in e)){let e,r,i=!!g,n=!!p,m=!!c;if(i){let a=Object.assign(Object.assign({prefixCls:`${k}-avatar`},n&&!m?{size:"large",shape:"square"}:{size:"large",shape:"circle"}),C(g));e=t.createElement("div",{className:`${k}-header`},t.createElement(o,Object.assign({},a)))}if(n||m){let e,a;if(n){let a=Object.assign(Object.assign({prefixCls:`${k}-title`},!i&&m?{width:"38%"}:i&&m?{width:"50%"}:{}),C(p));e=t.createElement(x,Object.assign({},a))}if(m){let e,r=Object.assign(Object.assign({prefixCls:`${k}-paragraph`},(e={},i&&n||(e.width="61%"),!i&&n?e.rows=3:e.rows=2,e)),C(c));a=t.createElement(_,Object.assign({},r))}r=t.createElement("div",{className:`${k}-content`},e,a)}let h=(0,a.default)(k,{[`${k}-with-avatar`]:i,[`${k}-active`]:u,[`${k}-rtl`]:"rtl"===v,[`${k}-round`]:f},w,s,l,E,O);return y(t.createElement("div",{className:h,style:Object.assign(Object.assign({},$),d)},e,r))}return null!=m?m:null};v.Button=e=>{let{prefixCls:n,className:s,rootClassName:l,active:d,block:m=!1,size:g="default"}=e,{getPrefixCls:p}=t.useContext(r.ConfigContext),c=p("skeleton",n),[u,f,h]=b(c),_=(0,i.default)(e,["prefixCls"]),x=(0,a.default)(c,`${c}-element`,{[`${c}-active`]:d,[`${c}-block`]:m},s,l,f,h);return u(t.createElement("div",{className:x},t.createElement(o,Object.assign({prefixCls:`${c}-button`,size:g},_))))},v.Avatar=e=>{let{prefixCls:n,className:s,rootClassName:l,active:d,shape:m="circle",size:g="default"}=e,{getPrefixCls:p}=t.useContext(r.ConfigContext),c=p("skeleton",n),[u,f,h]=b(c),_=(0,i.default)(e,["prefixCls","className"]),x=(0,a.default)(c,`${c}-element`,{[`${c}-active`]:d},s,l,f,h);return u(t.createElement("div",{className:x},t.createElement(o,Object.assign({prefixCls:`${c}-avatar`,shape:m,size:g},_))))},v.Input=e=>{let{prefixCls:n,className:s,rootClassName:l,active:d,block:m,size:g="default"}=e,{getPrefixCls:p}=t.useContext(r.ConfigContext),c=p("skeleton",n),[u,f,h]=b(c),_=(0,i.default)(e,["prefixCls"]),x=(0,a.default)(c,`${c}-element`,{[`${c}-active`]:d,[`${c}-block`]:m},s,l,f,h);return u(t.createElement("div",{className:x},t.createElement(o,Object.assign({prefixCls:`${c}-input`,size:g},_))))},v.Image=e=>{let{prefixCls:i,className:o,rootClassName:n,style:s,active:l}=e,{getPrefixCls:d}=t.useContext(r.ConfigContext),m=d("skeleton",i),[g,p,c]=b(m),u=(0,a.default)(m,`${m}-element`,{[`${m}-active`]:l},o,n,p,c);return g(t.createElement("div",{className:u},t.createElement("div",{className:(0,a.default)(`${m}-image`,o),style:s},t.createElement("svg",{viewBox:"0 0 1098 1024",xmlns:"http://www.w3.org/2000/svg",className:`${m}-image-svg`},t.createElement("title",null,"Image placeholder"),t.createElement("path",{d:"M365.714286 329.142857q0 45.714286-32.036571 77.677714t-77.677714 32.036571-77.677714-32.036571-32.036571-77.677714 32.036571-77.677714 77.677714-32.036571 77.677714 32.036571 32.036571 77.677714zM950.857143 548.571429l0 256-804.571429 0 0-109.714286 182.857143-182.857143 91.428571 91.428571 292.571429-292.571429zM1005.714286 146.285714l-914.285714 0q-7.460571 0-12.873143 5.412571t-5.412571 12.873143l0 694.857143q0 7.460571 5.412571 12.873143t12.873143 5.412571l914.285714 0q7.460571 0 12.873143-5.412571t5.412571-12.873143l0-694.857143q0-7.460571-5.412571-12.873143t-12.873143-5.412571zM1097.142857 164.571429l0 694.857143q0 37.741714-26.843429 64.585143t-64.585143 26.843429l-914.285714 0q-37.741714 0-64.585143-26.843429t-26.843429-64.585143l0-694.857143q0-37.741714 26.843429-64.585143t64.585143-26.843429l914.285714 0q37.741714 0 64.585143 26.843429t26.843429 64.585143z",className:`${m}-image-path`})))))},v.Node=e=>{let{prefixCls:i,className:o,rootClassName:n,style:s,active:l,children:d}=e,{getPrefixCls:m}=t.useContext(r.ConfigContext),g=m("skeleton",i),[p,c,u]=b(g),f=(0,a.default)(g,`${g}-element`,{[`${g}-active`]:l},c,o,n,u);return p(t.createElement("div",{className:f},t.createElement("div",{className:(0,a.default)(`${g}-image`,o),style:s},d)))},e.s(["default",0,v],185793)},599724,936325,e=>{"use strict";var t=e.i(95779),a=e.i(444755),r=e.i(673706),i=e.i(271645);let o=i.default.forwardRef((e,o)=>{let{color:n,className:s,children:l}=e;return i.default.createElement("p",{ref:o,className:(0,a.tremorTwMerge)("text-tremor-default",n?(0,r.getColorClassNames)(n,t.colorPalette.text).textColor:(0,a.tremorTwMerge)("text-tremor-content","dark:text-dark-tremor-content"),s)},l)});o.displayName="Text",e.s(["default",()=>o],936325),e.s(["Text",()=>o],599724)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var i=e.i(9583),o=a.forwardRef(function(e,o){return a.createElement(i.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["LinkOutlined",0,o],596239)}]);