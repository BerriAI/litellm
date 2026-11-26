(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,731565,e=>{"use strict";var t=e.i(115571),i=e.i(271645);function r(e){let i=t=>{"disableBlogPosts"===t.key&&e()},r=t=>{let{key:i}=t.detail;"disableBlogPosts"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t.M,r),()=>{window.removeEventListener("storage",i),window.removeEventListener(t.M,r)}}function a(){return"true"===(0,t.t)("disableBlogPosts")}e.s(["f",0,function(){return(0,i.useSyncExternalStore)(r,a,()=>!1)}])},636772,e=>{"use strict";var t=e.i(271645),i=e.i(115571);function r(e){let t=t=>{"disableShowPrompts"===t.key&&e()},r=t=>{let{key:i}=t.detail;"disableShowPrompts"===i&&e()};return window.addEventListener("storage",t),window.addEventListener(i.M,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(i.M,r)}}function a(){return"true"===(0,i.t)("disableShowPrompts")}e.s(["f",0,function(){return(0,t.useSyncExternalStore)(r,a,()=>!1)}])},66330,e=>{"use strict";var t=e.i(735450);e.s(["f",0,(e,i,r,a)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${i}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(a)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(r)},
        "state": "Invoice #4471 from Acme Tools. Total $174.00, due in 30 days.",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
            "total": {"type": "number", "unit": "USD"}
        }
    }
)

print(response.json()["answers"])`;return(0,t._)(e)?`import requests

response = requests.post(
    ${JSON.stringify(`${i}/v1/systemone`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(a)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(r)},
        "state": "I was charged twice for my subscription this month.",
        "questions": {
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?"
            }
        }
    }
)

print(response.json()["answers"])`:`import openai

client = openai.OpenAI(
    api_key="your_api_key",
    base_url=${JSON.stringify(i)}  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model=${JSON.stringify(r)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},423680,276701,e=>{"use strict";var t=e.i(843476);let i="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";e.s(["f",0,i],276701);var r=e.i(664659),a=e.i(322831);let l="https://docs.litellm.ai/docs/",n=()=>(0,t.jsx)(r.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0});e.s(["k",0,l,"E",0,()=>{let{t:e}=(0,a.f)();return(0,t.jsxs)("a",{href:l,target:"_blank",rel:"noopener noreferrer",className:i,children:[e("shell.docsLink.label"),(0,t.jsx)(n,{})]})}],423680)},455880,e=>{"use strict";var t=e.i(843476),i=e.i(475254);let r=(0,i.f)("moon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]),a=(0,i.f)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);var l=e.i(363178),n=e.i(271645),o=e.i(322831),s=e.i(519455);let u=()=>()=>{};function d(){let{t:e}=(0,o.f)(),{setTheme:t,resolvedTheme:i}=(0,l.useTheme)(),r=n.default.useSyncExternalStore(u,()=>!0,()=>!1)&&"dark"===i,a=e(r?"themeToggle.switchToLight":"themeToggle.switchToDark");return{isDark:r,label:a,toggleTheme:()=>t(r?"light":"dark")}}e.s(["U",0,()=>{let{isDark:e,label:i,toggleTheme:l}=d();return(0,t.jsx)(s.K,{variant:"ghost",size:"icon-sm","aria-label":i,title:i,className:"text-muted-foreground",onClick:l,children:e?(0,t.jsx)(r,{}):(0,t.jsx)(a,{})})},"X",0,d],455880)},909947,e=>{"use strict";var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:r,accessToken:a,apiKey:l,inputMessage:n,chatHistory:o,selectedTags:s,selectedVectorStores:u,selectedGuardrails:d,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:f,proxySettings:h,customHeaders:_}=e,b="session"===r?a:l,y=window.location.origin,x=h?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?y=x:h?.PROXY_BASE_URL&&(y=h.PROXY_BASE_URL);let v=n||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),$=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};s.length>0&&(k.tags=s),u.length>0&&(k.vector_stores=u),d.length>0&&(k.guardrails=d),c.length>0&&(k.policies=c);let j=g||"your-model-name",P=_&&Object.keys(_).length>0?`,
	default_headers=${JSON.stringify(_,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${P}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${P}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=$.length>0?$:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${j}",
    messages=${JSON.stringify(r,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${j}",
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.Q.RESPONSES:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=$.length>0?$:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${j}",
    input=${JSON.stringify(r,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${j}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${w}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                     "detail": "auto",
#                 },
#             ],
#         }
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.Q.IMAGE:i="azure"===f?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${j}",
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
	model="${j}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
`;break;case t.Q.IMAGE_EDITS:i="azure"===f?`
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
	model="${j}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
	model="${j}",
	input=[
			{
					"role": "user",
					"content": [
							{"type": "input_text", "text": prompt},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image1}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
`;break;case t.Q.EMBEDDINGS:i=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${j}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${j}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${j}",
	input="${n||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${j}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${i}`}])},583408,e=>{"use strict";var t=e.i(843476),i=e.i(555436),r=e.i(322831),a=e.i(519455),l=e.i(950594),n=e.i(967489);e.s(["f",0,function({search:e,tag:o,scan:s,tags:u,active:d,setSearch:c,setTag:p,setScan:m,clear:g}){let{t:f}=(0,r.f)(),h=[{value:"",label:f("skills.filters.allTags")},...[...new Set([...u,...o?[o]:[]])].map(e=>({value:e,label:e}))],_=[{value:"all",label:f("skills.filters.allScans")},{value:"yes",label:f("skills.filters.scanned")},{value:"no",label:f("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(l.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(l.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(l.M,{"aria-label":f("skills.filters.search"),placeholder:f("skills.filters.search"),value:e,onChange:e=>c(e.target.value)})]}),(0,t.jsxs)(n.d,{items:h,value:o,onValueChange:e=>p(e??""),children:[(0,t.jsx)(n.P,{className:"w-44","aria-label":f("skills.filters.tag"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:h.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(n.d,{items:_,value:s,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(n.P,{className:"w-48","aria-label":f("skills.table.columns.securityScan"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:_.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),d&&(0,t.jsx)(a.K,{variant:"ghost",size:"sm",onClick:g,children:f("skills.filters.clear")})]})}])},594107,e=>{"use strict";var t=e.i(843476),i=e.i(494862);e.i(622826);var r=e.i(200208),a=e.i(112179),l=e.i(487486),n=e.i(201787),o=e.i(617885);let s={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(e={}){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:n.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(n.f.language)??"—"})},{id:"keywords",header:n.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(l.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(r=>({id:r,accessorKey:r,size:190,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.identitySortHelp"),children:n.f.t(s[r])})}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[r]??void 0,children:(0,o.b)(i.original[r],e)||n.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:r})=>(0,t.jsx)(i.v,{column:r,title:n.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(r.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:n.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:(0,t.jsx)(a.f,{tone:"neutral",label:n.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{"use strict";var t=e.i(201787);let i=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,r=e=>e.trim().replace(/\/+$/,""),a="github.com",l=/\.zip$/i,n=/^[0-9a-fA-F]{64}$/,o=/^\d{1,3}(\.\d{1,3}){3}$/,s=/^[A-Za-z0-9-]+$/,u=/^[A-Za-z0-9._-]+$/,d=/^https?:\/\//i,c="ssh://",p=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,m=e=>`${e.protocol}//${e.host}${e.pathname.replace(/\/+$/,"")}`,g=e=>e.pathname.split("/").filter(e=>""!==e),f=e=>{try{return new URL(e)}catch{return null}},h=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!o.test(e.hostname),_=e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=f(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&h(i)?i:null},b=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},y=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),x=(e,t)=>{let i=t?.trim();return e&&i?{...e,parsed:{...e.parsed,.../^[a-fA-F0-9]{40}$/.test(i)?{sha:i.toLowerCase()}:{ref:i}},label:`${e.label} (${i})`}:e},v=(e,t,i,a)=>{if(!e.length)return null;try{let l=i?.trim()||decodeURIComponent(e[0]),n=i?.trim()?e.findIndex((t,i)=>decodeURIComponent(e.slice(0,i+1).join("/"))===l)+1:1;if(!n||"blob"===t&&e.length===n)return null;let o=e.slice(n,"blob"===t?-1:void 0).map(decodeURIComponent).join("/"),s="blob"===t?decodeURIComponent(e[e.length-1]):void 0,u=!!(o||s);return{ref:l,path:u?o:r(a??""),pathFromUrl:u,fileFromUrl:s}}catch{return null}},w=(e,a,l,n="reference")=>{let o=g(e);if(o.length<2)return null;let d=o[0],c=o[1].replace(/\.git$/,"");if(!s.test(d)||!u.test(c))return null;let p=`${d}/${c}`,m=`https://github.com/${p}`,f={parsed:{source:"github",repo:p},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:p}),suggestedName:y(c)},h=o[2],_="tree"===h||"blob"===h?v(o.slice(3),h,l,a):void 0;if(null===_||void 0===_&&2!==o.length)return null;let w=_?.path??r(a??"");return"reference"===n&&w&&!i.test(w)?null:x({...w?{parsed:{source:"git-subdir",url:m,path:w},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:p,path:w}),suggestedName:y(b(w))}:f,..._?{pathFromUrl:_.pathFromUrl,fileFromUrl:_.fileFromUrl}:{}},_?.ref??l)},$=(e,a,l,n)=>{let o=r(n??"");return""!==o?i.test(o)?{parsed:{source:"git-subdir",url:a,path:o},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:a,path:o}),suggestedName:y(b(o))}:null:{parsed:{source:"url",url:a},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:a}),suggestedName:y(l)}},k=e=>{let t=_(e);if(!t)return null;let i=t.host.replace(/^www\./,"");return[a,"skills.sh"].includes(i)?f(`${t.protocol}//${i}${t.pathname}${t.search}${t.hash}`):null},j="<your-litellm-key>",P=e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e);e.s(["i",0,j,"x",0,e=>{let t=k(e);return t?.host==="skills.sh"?["import"]:t?["register","import"]:["register"]},"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${j}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,i,r)=>{if("archive"!==t.source||!P(e)||t.sha256&&!n.test(t.sha256))return null;let a=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${i.replace(/\/+$/,"")}${a}`,r);return/^https?:$/.test(e.protocol)&&t.url===e.href?a:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>{if(!["github","url","git-subdir","archive"].includes(e.source))return null;let t="github"===e.source?e.repo&&`https://github.com/${e.repo}`:e.url;if(!t||!d.test(t))return null;if("archive"===e.source)return t;let i=f(t),r=i&&i.hostname.replace(/^www\./,"");if(!i||r!==a&&"gitlab.com"!==r)return t;let l=g(i);if(!(r===a?2===l.length:l.length>=2&&!l.includes("-")))return t;let n=m(i).replace(/\.git$/,""),o=e.sha||e.ref;if(!o)return n;let s="git-subdir"===e.source&&e.path?`/${e.path.split("/").map(encodeURIComponent).join("/")}`:"";return`${n}/${r===a?"tree":"-/tree"}/${encodeURIComponent(o)}${s}`},"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||n.test(e.trim()),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"C",0,(e,t,i,r)=>{let a=k(e);if(!a||a.search||a.hash)return null;if("skills.sh"===a.host)try{let e=a.pathname.replace(/^\/+|\/+$/g,"").split("/").map(decodeURIComponent);if(3!==e.length||e.slice(0,2).some(e=>!u.test(e)))return null;let r=e[2];if(!r)return null;return{selection:{url:a.href,...i?.trim()?{ref:i.trim()}:{},...t?.trim()?{path:t.trim()}:{}},preview:null,skillFromUrl:r}}catch{return null}let l=w(a,t,i,"portable");if(!l||l.fileFromUrl&&"SKILL.md"!==l.fileFromUrl)return null;let n=l.parsed,o=n.sha??n.ref;return{selection:{url:"github"===n.source?`https://github.com/${n.repo}`:n.url??a.href,...r?.trim()?{skill:r.trim()}:{},...o?{ref:o}:{},...n.path||l.fileFromUrl?{path:n.path??""}:{}},preview:l}},"Z",0,(e,i,r)=>{let n=((e,t)=>{let i=e.trim(),r=p.exec(i),a=r?`${c}${r[1]}@${r[2]}/${r[3]}`:i;if(!a.toLowerCase().startsWith(c))return null;let l=f(a);if(!l||""===l.username||""!==l.password||!h(l))return null;let n=a.indexOf("/",c.length);return -1===n||l.pathname!==a.slice(n)||g(l).length<2?null:$("SSH",i,b(l.pathname).replace(/\.git$/i,""),t)})(e,i);if(n)return x(n,r);let o=_(e);if(!o)return null;let s=o.hostname.replace(/^www\./,"");if(s===a&&["tree","blob"].includes(g(o)[2]))return w(o,i,r);if("gitlab.com"===s&&/\/-\/(?:tree|blob)(?:\/|$)/.test(o.pathname))return((e,t,i)=>{let r=g(e),a=r.findIndex((e,t)=>"-"===e&&["tree","blob"].includes(r[t+1])),l=r[a+1];if(a<2||"tree"!==l&&"blob"!==l)return null;let n=v(r.slice(a+2),l,i,t);if(!n)return null;let o=$("Git",`${e.protocol}//${e.host.replace(/^www\./,"")}/${r.slice(0,a).join("/")}`,r[a-1].replace(/\.git$/,""),n.path);return x(o?{...o,pathFromUrl:n.pathFromUrl,fileFromUrl:n.fileFromUrl}:null,n.ref)})(o,i,r);if(l.test(o.pathname))return{parsed:{source:"archive",url:o.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${o.host}${o.pathname}`}),suggestedName:y(b(o.pathname).replace(l,""))};return s===a?w(o,i,r):x(((e,t)=>{if(g(e).length<2)return null;let i=b(e.pathname).replace(/\.git$/,"");return $("Git",m(e),i,t)})(o,i),r)},"c",0,(e,t)=>!!(e?.pathFromUrl&&e.parsed.path&&!(t.trim()||e.parsed.sha||e.parsed.ref?.includes("/"))),"e",0,P])},962600,e=>{"use strict";var t=e.i(438847),i=e.i(45570),r=e.i(899426);let a=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],l={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:a,defaultSort:{id:"name",desc:!1},filterColumns:[]},n={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},o={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,a,"W",0,function(e,t){let i=(0,r.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),a=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),l="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&a&&l},"n",0,function(e){return{sort_by:a.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=a){let r=(0,i.f)({...l,sortFields:e}),[s,u]=(0,t.i)(n,o),d=e=>{u(e),r.onPaginationChange({...r.pagination,pageIndex:0})};return{...r,...s,setTag:e=>d({tag:e}),setScan:e=>d({scan:e}),clear:()=>{d(null),r.setSearch("")},active:!!(r.search.trim()||s.tag||"all"!==s.scan)}}])},33355,e=>{"use strict";var t=e.i(843476),i=e.i(322831),r=e.i(418371),a=e.i(916925),l=e.i(196631),n=e.i(784774),o=e.i(201787);let s=["standard","flex","balanced","priority","ultrafast","batches"],u={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},d=["token","second","image","pixel","character","query","request"],c=["input","cache_read","explicit_cache_read","cache_write","output"],p=["","text","image","audio","video","reasoning"],m=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,g=/^(.+)_above_(\d+)k_tokens$/,f=/^(.+)_above_(\d+hr)$/,h=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,_=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,b=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},y={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},x={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},v={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},w=({charge:e,modality:t},i)=>t?o.f.t("deploymentPricing.withModality",{modality:o.f.t(x[i][t]),charge:o.f.t(y.inline[e])}):o.f.t(y[i][e]),$=(e,t)=>e.cacheDuration?o.f.t("deploymentPricing.withCacheDuration",{charge:w(e,t),duration:e.cacheDuration}):w(e,t),k=e=>o.f.t(v[e]),j=e=>o.f.t("deploymentPricing.rowLabel",{charge:$(e,"start"),unit:k(e.unit)}),P=e=>[c.indexOf(e.charge),d.indexOf(e.unit),p.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],S=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),M=e=>0===e?o.f.t("deploymentPricing.free"):S.format(e),I=(e,t)=>M("token"===t?1e6*e:e),N=/^(.*?)_?cost_per_(.+)$/,C=/^(.+?)_cost(?:_(.+))?$/,z=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},O=e=>{let t=N.exec(e);if(t)return o.f.t("deploymentPricing.rowLabel",{charge:t[1]?z(t[1]):o.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=C.exec(e);return i?z([i[1],i[2]].filter(Boolean).join(" ")):z(e)},L=e=>"price"===e.cost.kind,A=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=b(e,m),[r,a]=b(t,g),[l,n]=b(r,f),o=(e=>{let t=h.exec(e);if(t){let[,e,i="",r]=t;return d.includes(r)?{charge:e,modality:i,unit:r}:null}let i=_.exec(e);if(i){let[,e,t,r=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:r,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:r,unit:"token"}}return null})(l);return o?{kind:"price",...o,cacheDuration:n??"",threshold:null===a?null:1e3*Number(a),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),H=(e,t)=>e.filter((i,r)=>e.findIndex(e=>t(e)===t(i))===r),E=(e,t)=>{let i,r;return i=P(e.cost),r=P(t.cost),i.reduce((e,t,i)=>e||t-r[i],0)},U=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),T=({cost:e,value:t})=>{let i=$(e,"inline"),r=I(t,e.unit);return"token"===e.unit?o.f.t("deploymentPricing.tierPrice",{price:r,charge:i}):o.f.t("deploymentPricing.tierPricePerUnit",{price:r,charge:i,unit:k(e.unit)})};function R({group:e,tiers:r}){let{t:a}=(0,i.f)(),l=null===e.threshold?a("deploymentPricing.basePrices"):a("deploymentPricing.aboveThreshold",{threshold:U(e.threshold)});return(0,t.jsxs)(n.z,{"aria-label":l,children:[null!==e.threshold&&(0,t.jsx)(n.N,{className:"hover:bg-transparent",children:(0,t.jsx)(n.V,{colSpan:r.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:l})}),e.rows.map(e=>(0,t.jsxs)(n.N,{children:[(0,t.jsx)(n.V,{scope:"row",className:"font-normal",children:e.label}),r.map(i=>(0,t.jsx)(n.d,{className:n.J,children:e.prices[i]},i))]},e.label))]})}function F({deployment:e}){let l,d,c,p,m,{t:g}=(0,i.f)(),f=(c=[...H(d=(l=A(e.costs)).filter(L),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),p=s.filter(e=>d.some(({cost:t})=>t.tier===e)),{tiers:p,groups:m=c.map(e=>{let t;return{threshold:e,rows:[...H(t=d.filter(({cost:t})=>t.threshold===e),({cost:e})=>j(e))].sort(E).map(({cost:e})=>({label:j(e),prices:Object.fromEntries(t.filter(t=>j(t.cost)===j(e)).map(e=>[e.cost.tier,I(e.value,e.cost.unit)]))}))}}),hasBlankCells:m.some(({rows:e})=>e.some(({prices:e})=>p.some(t=>!(t in e)))),unknown:l.filter(e=>!L(e)).map(({key:e,value:t})=>({key:e,label:O(e),price:M(t)}))}),h=e.tiered_pricing,_=(0,a.h)(e.provider).displayName;return(0,t.jsxs)("section",{"aria-label":g("deploymentPricing.providerPricing",{provider:_}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,t.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,t.jsx)(r.f,{provider:e.provider}),_]}),f.groups.length>0&&(0,t.jsxs)(n.s,{children:[(0,t.jsx)(n.E,{children:(0,t.jsxs)(n.N,{className:"hover:bg-transparent",children:[(0,t.jsx)(n.V,{children:g("deploymentPricing.chargeColumn")}),f.tiers.map(e=>(0,t.jsx)(n.V,{className:n.J,children:g(u[e])},e))]})}),f.groups.map(e=>(0,t.jsx)(R,{group:e,tiers:f.tiers},e.threshold??"base"))]}),f.hasBlankCells&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:g("deploymentPricing.blankCells")}),h.length>0&&(0,t.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,t.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:g("deploymentPricing.tieredPricing")}),(0,t.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:h.map(e=>(0,t.jsx)("li",{children:(({range:[e,t],costs:i})=>{let r=A(i),a=[...[...r.filter(L)].sort(E).map(T),...r.filter(e=>!L(e)).map(({key:e,value:t})=>o.f.t("deploymentPricing.tierPrice",{price:M(t),charge:O(e).toLowerCase()}))];return o.f.t("deploymentPricing.tierRange",{low:U(e),high:U(t),prices:a.join(" / ")})})(e)},e.range.join("-")))})]}),f.unknown.length>0&&(0,t.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:f.unknown.map(({key:e,label:i,price:r})=>(0,t.jsxs)("div",{className:"contents",children:[(0,t.jsx)("dt",{className:"truncate",children:i}),(0,t.jsx)("dd",{className:"text-right tabular-nums",children:r})]},e))})]})}e.s(["f",0,function({deployments:e,heading:r,className:a}){let{t:n}=(0,i.f)();return e&&0!==e.length?(0,t.jsxs)("div",{className:(0,l.c)("flex flex-col gap-3",a),children:[r,e.length>1&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,i)=>(0,t.jsx)(F,{deployment:e},`${e.provider}-${i}`)),(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.requestAdjustments")})]}):null}],33355)},913761,e=>{"use strict";var t=e.i(602869),i=e.i(916925),r=e.i(735450),a=e.i(201787);let l={chat:"publicModelHub.format.modes.chat",completion:"publicModelHub.format.modes.completion",embedding:"publicModelHub.format.modes.embedding",rerank:"publicModelHub.format.modes.rerank",image_generation:"publicModelHub.format.modes.imageGeneration",image_edit:"publicModelHub.format.modes.imageEdit",audio_speech:"publicModelHub.format.modes.audioSpeech",audio_transcription:"publicModelHub.format.modes.audioTranscription",video_generation:"publicModelHub.format.modes.videoGeneration",realtime:"publicModelHub.format.modes.realtime",responses:"publicModelHub.format.modes.responses",batch:"publicModelHub.format.modes.batch",ocr:"publicModelHub.format.modes.ocr",decisions:"publicModelHub.format.modes.decisions"},n=e=>e.replaceAll("_"," ").replace(/^./,e=>e.toUpperCase()),o=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:4,useGrouping:!1}),s={supports_vision:"publicModelHub.format.capabilities.vision",supports_function_calling:"publicModelHub.format.capabilities.functionCalling",supports_reasoning:"publicModelHub.format.capabilities.reasoning",supports_web_search:"publicModelHub.format.capabilities.webSearch",supports_parallel_function_calling:"publicModelHub.format.capabilities.parallelFunctionCalling",supports_url_context:"publicModelHub.format.capabilities.urlContext",supports_fast_mode:"publicModelHub.format.capabilities.fastMode"},u=Object.keys(s),d=e=>{let t=u.indexOf(e);return -1===t?u.length:t},c=e=>Object.hasOwn(s,e)?a.f.t(s[e]):n(e.replace(/^supports_/,"")),p={healthy:{tone:"success",label:"publicModelHub.format.health.healthy"},unhealthy:{tone:"error",label:"publicModelHub.format.health.unhealthy"}};e.s(["m",0,"text-[15px] leading-[22px] font-semibold","G",0,e=>c(`supports_${e}`),"a",0,e=>0===e?a.f.t("publicModelHub.format.free"):e?o.format(1e6*e):"-","k",0,e=>e?e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${Math.round(e/1e3)}K`:String(e):"-","M",0,()=>(0,t._h)(),"H",0,e=>{let t=e.health_status??"";return Object.hasOwn(p,t)?{tone:p[t].tone,label:a.f.t(p[t].label)}:{tone:"neutral",label:a.f.t("publicModelHub.format.notChecked")}},"S",0,e=>{let t=(0,r.i)(e);return t?Object.hasOwn(l,t)&&a.f.t(l[t])||n(t):"-"},"E",0,e=>Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e).sort((e,t)=>d(e)-d(t)).map(c),"q",0,e=>(0,i.h)(e).logo,"P",0,e=>(0,i.h)(e).displayName.replace(/\s*\(.*\)$/,""),"W",0,(e,t=Date.now())=>{let i=e?new Date(e).getTime():0/0;if(Number.isNaN(i))return"-";let r=Math.max(0,Math.round((t-i)/6e4));if(r<1)return a.f.t("publicModelHub.format.relativeTime.justNow");if(r<60)return a.f.t("publicModelHub.format.relativeTime.minutes",{minutes:r});let l=Math.round(r/60);if(l<24)return a.f.t("publicModelHub.format.relativeTime.hours",{hours:l});let n=Math.round(l/24);return a.f.t("publicModelHub.format.relativeTime.days",{count:n})},"Q",0,e=>Object.entries(e??{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:t})=>({title:e,url:t})),"X",0,e=>e.replaceAll("*","<model-name>")])},489432,e=>{"use strict";var t=e.i(66330),i=e.i(209261),r=e.i(909947),a=e.i(865361),l=e.i(602869),n=e.i(735450),o=e.i(913761);let s="Hello, how are you?",u=e=>{let t;if(!e.pass_through_path)return null;let i=(t=e.pass_through_methods,!t?.length||t.includes("POST")?"POST":t[0]),r=i.replace(/[\\"$`]/g,"\\$&"),a=`${(0,o.M)()}${e.pass_through_path}`.replace(/[\\"$`]/g,"\\$&");return"POST"===i?`curl -X "${r}" "${a}" \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, how are you?"}'`:`curl -X "${r}" "${a}" \\
  -H "Authorization: Bearer your_api_key"`};e.s(["c",0,e=>e.pass_through_path?u(e):e.mode&&"chat"!==e.mode?null:`curl ${(0,o.M)()}/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${(0,o.X)(e.model_group)}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`,"j",0,()=>`/plugin marketplace add ${(0,o.M)()}/claude-code/marketplace.json`,"W",0,e=>{if("passthrough"===e.mode||e.pass_through_path)return u(e)??"";let i=(0,o.X)(e.model_group);if((0,n._)(e.mode))return(0,t.f)(e.mode,(0,o.M)(),i,(0,l.kR)());let d={apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:s,chatHistory:[{role:"user",content:s,isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,a.t)(e.mode||"chat"),selectedModel:i,selectedSdk:"openai",proxySettings:{PROXY_BASE_URL:(0,o.M)()}};return(0,r.f)(d)},"g",0,u,"z",0,e=>(0,i.a)(e)])},466098,348594,831538,e=>{"use strict";var t=e.i(586448),i=e.i(602869),r=e.i(271645),a=e.i(198458),l=e.i(735450);let n="mode",o="providers",s="features",u=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],d=(e,t)=>{let i,r;return i=`filter[${e}][in]`,""===(r=u(t).join(","))?[]:[[i,r]]},c=e=>{switch(e.id){case n:return d(e.id,(0,l.q)(u(e.value)));case o:case s:return d(e.id,e.value);default:return[]}},p=e=>Object.fromEntries(e.flatMap(c)),m=(e,t)=>u(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let r=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?r:[...r,{id:t,value:i}]};e.s(["g",0,s,"e",0,n,"T",0,o,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,p,"R",0,g],348594);let f="/public/v1/model_hub",h=["publicModelHub","list"],_=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await i.Mw.get(f,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,f,"t",0,e=>{let t=(0,a.f)({queryKey:h,fetchPage:b,serializeFilters:p,defaultSorting:_,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,l=(0,r.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),u=(0,r.useCallback)(e=>l(o,e),[l]),d=(0,r.useCallback)(e=>l(n,e),[l]),c=(0,r.useCallback)(e=>l(s,e),[l]);return{...t,providerValues:m(t.columnFilters,o),onProvidersChange:u,modeValues:m(t.columnFilters,n),onModesChange:d,featureValues:m(t.columnFilters,s),onFeaturesChange:c,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let y=["providers","modes","features"];e.s(["f",0,e=>{let[r,a,l]=(0,t.f)({queries:y.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${f}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:r,modes:a,features:l}}],466098)},799676,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(552245),a=e.i(733332);let l=i.createContext(void 0);function n(){let e=i.useContext(l);if(void 0===e)throw Error((0,a.f)(13));return e}let o={imageLoadingStatus:()=>null},s=i.forwardRef(function(e,a){let{className:n,render:s,style:u,...d}=e,[c,p]=i.useState("idle"),m=i.useMemo(()=>({imageLoadingStatus:c,setImageLoadingStatus:p}),[c,p]),g=(0,r.f)("span",e,{state:{imageLoadingStatus:c},ref:a,props:d,stateAttributesMapping:o});return(0,t.jsx)(l.Provider,{value:m,children:g})});var u=e.i(667865),d=e.i(146376),c=e.i(137584),p=e.i(209407),m=e.i(223910),g=e.i(956789);let f={...o,...p.T},h=i.forwardRef(function(e,t){let{className:a,render:l,onLoadingStatusChange:o,style:s,...p}=e,{setImageLoadingStatus:h}=n(),_=function(e,{referrerPolicy:t,crossOrigin:r,sizes:a,srcSet:l}){let[n,o]=i.useState("idle");return(0,d.f)(()=>{if(!e&&!l)return o("error"),g.x;let i=!0,n=new window.Image,s=e=>()=>{i&&o(e)};return o("loading"),n.onload=s("loaded"),n.onerror=s("error"),t&&(n.referrerPolicy=t),n.crossOrigin=r??null,a&&(n.sizes=a),l&&(n.srcset=l),e&&(n.src=e),n.complete&&o(n.naturalWidth>0?"loaded":"error"),()=>{i=!1}},[e,l,a,r,t]),n}(p.src,p),b="loaded"===_,{mounted:y,transitionStatus:x,setMounted:v}=(0,m.f)(b),w=i.useRef(null),$=(0,u.f)(e=>{o?.(e),h(e)});(0,d.f)(()=>{"idle"!==_&&$(_)},[_,$]),(0,d.f)(()=>()=>h("idle"),[h]),(0,c.f)({open:b,ref:w,onComplete(){b||v(!1)}});let k=(0,r.f)("img",e,{state:{imageLoadingStatus:_,transitionStatus:x},ref:[t,w],props:p,stateAttributesMapping:f,enabled:y});return y?k:null});var _=e.i(439957);let b=i.forwardRef(function(e,t){let{className:a,render:l,delay:s,style:u,...d}=e,{imageLoadingStatus:c}=n(),[p,m]=i.useState(void 0===s),g=(0,_.e)();return i.useEffect(()=>(void 0!==s?g.start(s,()=>m(!0)):m(!0),g.clear),[g,s]),(0,r.f)("span",e,{state:{imageLoadingStatus:c},ref:t,props:d,stateAttributesMapping:o,enabled:"loaded"!==c&&(void 0===s||p)})});e.s(["Fallback",0,b,"Image",0,h,"Root",0,s],514751);var y=e.i(514751),y=y,x=y,v=y,w=e.i(196631);let $=i.forwardRef(({className:e,...i},r)=>(0,t.jsx)(y.Root,{ref:r,"data-slot":"avatar",className:(0,w.c)("relative flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full",e),...i}));$.displayName="Avatar";let k=i.forwardRef(({className:e,...i},r)=>(0,t.jsx)(x.Image,{ref:r,"data-slot":"avatar-image",className:(0,w.c)("size-full object-cover",e),...i}));k.displayName="AvatarImage";let j=i.forwardRef(({className:e,...i},r)=>(0,t.jsx)(v.Fallback,{ref:r,"data-slot":"avatar-fallback",className:(0,w.c)("flex size-full items-center justify-center rounded-full text-xs font-medium",e),...i}));j.displayName="AvatarFallback",e.s(["E",0,$,"b",0,j,"Z",0,k],799676)},747862,e=>{"use strict";var t=e.i(843476);e.i(176782),e.i(911825);var i=e.i(225913),r=e.i(196631);e.i(772436);let a=(0,i.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:i,...l}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":i,className:(0,r.c)(a({orientation:i}),e),...l})}])},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(602869),a=e.i(411382),l=e.i(54105),n=e.i(213056);let o=a.z.object({values:a.z.object({logo_url:l.z.string().nullish(),logo_url_dark:l.z.string().nullish(),favicon_url:l.z.string().nullish()}),logo_url_via_proxy:n.z.boolean().optional(),logo_url_dark_via_proxy:n.z.boolean().optional()});function s(e,t){let i=o.parse(e),r=`${t??""}/get_image`;return{logoUrl:i.logo_url_via_proxy?r:i.values.logo_url||null,logoUrlDark:i.logo_url_dark_via_proxy?`${r}?theme=dark`:i.values.logo_url_dark||null,faviconUrl:i.values.favicon_url||null}}let u=(0,i.createContext)(void 0);e.s(["J",0,({children:e,accessToken:a})=>{let[l,n]=(0,i.useState)(null),[o,d]=(0,i.useState)(null),[c,p]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,r._h)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let t=s(await i.json(),e);n(t.logoUrl),d(t.logoUrlDark),p(t.faviconUrl)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,i.useEffect)(()=>{if(c){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=c});else{let e=document.createElement("link");e.rel="icon",e.href=c,document.head.appendChild(e)}}},[c]),(0,t.jsx)(u.Provider,{value:{logoUrl:l,setLogoUrl:n,logoUrlDark:o,setLogoUrlDark:d,faviconUrl:c,setFaviconUrl:p},children:e})},"r",0,s,"i",0,()=>{let e=(0,i.useContext)(u);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},735450,e=>{"use strict";var t=e.i(411382),i=e.i(772325),r=e.i(54105),a=e.i(240192),l=e.i(138620);let n="https://docs.litellm.ai/docs/decisions",o=`${n}#supported-providers`,s=`${n}#self-hosted-laya-and-nimble`,u=new Set(["/v1/decisions","/v1/systemone"]),d=t.z.object({litellm_provider:r.z.string(),mode:r.z.string().nullish(),supported_endpoints:i.z.array(r.z.string()).nullish()}),c="decisions",p=["evaluation",c],m=e=>null!=e&&p.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>u.has(e)),f=e=>{let t=a.z.record(r.z.string(),l.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=d.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},h={decisions:0,"self-hosted":1,"needs-upgrade":2},_=t.z.object({decision_model_providers:i.z.array(r.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),y=(e,t,i)=>({row:e,routeOrder:h[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,n,"d",0,c,"Z",0,o,"L",0,s,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),i=t.filter(([,e])=>g(e)),r=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:r}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(y).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?p:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=_.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function i(e,i){let r=t(e);if(""===r)return!0;let a=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!a.some(e=>e.includes(r))||r.split(/\s+/).every(e=>a.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,r){return e.filter(e=>i(t,r(e)))},"h",0,i,"e",0,function(e,i,r){let a=t(i);if(""===a)return[...e];let l=e=>{let t=r(e).toLowerCase();return 1e3*(t===a)+100*!!t.startsWith(a)+(1e3-t.length)};return[...e].sort((e,t)=>l(t)-l(e))}])}]);