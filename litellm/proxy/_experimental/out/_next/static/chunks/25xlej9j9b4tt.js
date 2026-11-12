(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,66330,e=>{var t=e.i(735450);e.s(["f",0,(e,i,a,s)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${i}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(s)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(a)},
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
    headers={${JSON.stringify(s)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(a)},
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
    model=${JSON.stringify(a)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:a,accessToken:s,apiKey:r,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:p,selectedVoice:u,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:x}=e,b="session"===a?s:r,_=window.location.origin,y=f?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?_=y:f?.PROXY_BASE_URL&&(_=f.PROXY_BASE_URL);let j=l||"Your prompt here",v=j.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};o.length>0&&(k.tags=o),d.length>0&&(k.vector_stores=d),c.length>0&&(k.guardrails=c),p.length>0&&(k.policies=p);let N=g||"your-model-name",$=x&&Object.keys(x).length>0?`,
	default_headers=${JSON.stringify(x,null,2).replace(/\n/g,"\n	")}`:"",P="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${$}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let a=w.length>0?w:[{role:"user",content:j}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${N}",
    messages=${JSON.stringify(a,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${N}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${v}"
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
    extra_body=${e}`}let a=w.length>0?w:[{role:"user",content:j}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${N}",
    input=${JSON.stringify(a,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${N}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${v}"},
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
`;break}case t.Q.IMAGE:i="azure"===h?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${N}",
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
`;break;case t.Q.IMAGE_EDITS:i="azure"===h?`
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
	input="${l||"Your string here"}",
	model="${N}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${N}",
	file=audio_file${l?`,
	prompt="${l.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${N}",
	input="${l||"Your text to convert to speech here"}",
	voice="${u}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${N}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${P}
${i}`}])},583408,e=>{var t=e.i(843476),i=e.i(555436),a=e.i(322831),s=e.i(519455),r=e.i(950594),l=e.i(967489);e.s(["f",0,function({search:e,tag:n,scan:o,tags:d,active:c,setSearch:p,setTag:u,setScan:m,clear:g}){let{t:h}=(0,a.f)(),f=[{value:"",label:h("skills.filters.allTags")},...[...new Set([...d,...n?[n]:[]])].map(e=>({value:e,label:e}))],x=[{value:"all",label:h("skills.filters.allScans")},{value:"yes",label:h("skills.filters.scanned")},{value:"no",label:h("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(r.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(r.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(r.M,{"aria-label":h("skills.filters.search"),placeholder:h("skills.filters.search"),value:e,onChange:e=>p(e.target.value)})]}),(0,t.jsxs)(l.d,{items:f,value:n,onValueChange:e=>u(e??""),children:[(0,t.jsx)(l.P,{className:"w-44","aria-label":h("skills.filters.tag"),children:(0,t.jsx)(l.U,{})}),(0,t.jsx)(l.r,{children:f.map(e=>(0,t.jsx)(l.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(l.d,{items:x,value:o,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(l.P,{className:"w-48","aria-label":h("skills.table.columns.securityScan"),children:(0,t.jsx)(l.U,{})}),(0,t.jsx)(l.r,{children:x.map(e=>(0,t.jsx)(l.T,{value:e.value,children:e.label},e.value))})]}),c&&(0,t.jsx)(s.K,{variant:"ghost",size:"sm",onClick:g,children:h("skills.filters.clear")})]})}])},594107,e=>{var t=e.i(843476),i=e.i(494862);e.i(622826);var a=e.i(200208),s=e.i(112179),r=e.i(487486),l=e.i(201787);let n={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:l.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(l.f.language)??"—"})},{id:"keywords",header:l.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(r.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(e=>({id:e,accessorKey:e,size:190,enableSorting:!0,header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:l.f.t(n[e])}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[e]??void 0,children:i.original[e]||l.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:l.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(a.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:l.f.t("skills.table.scanHelp"),children:l.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:l.f.t("skills.table.scanHelp"),children:(0,t.jsx)(s.f,{tone:"neutral",label:l.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{var t=e.i(201787);let i=/^[0-9a-fA-F]{64}$/,a=/^https?:\/\//i,s="<your-litellm-key>";e.s(["i",0,s,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${s}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,a,s)=>{if("archive"!==t.source||!(e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e))||t.sha256&&!i.test(t.sha256))return null;let r=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${a.replace(/\/+$/,"")}${r}`,s);return/^https?:$/.test(e.protocol)&&t.url===e.href?r:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&a.test(e.url)?e.url:null,"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[]])},962600,e=>{var t=e.i(438847),i=e.i(45570),a=e.i(899426);let s=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],r={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:s,defaultSort:{id:"name",desc:!1},filterColumns:[]},l={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},n={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,s,"W",0,function(e,t){let i=(0,a.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),s=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),r="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&s&&r},"n",0,function(e){return{sort_by:s.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=s){let a=(0,i.f)({...r,sortFields:e}),[o,d]=(0,t.i)(l,n),c=e=>{d(e),a.onPaginationChange({...a.pagination,pageIndex:0})};return{...a,...o,setTag:e=>c({tag:e}),setScan:e=>c({scan:e}),clear:()=>{c(null),a.setSearch("")},active:!!(a.search.trim()||o.tag||"all"!==o.scan)}}])},33355,e=>{var t=e.i(843476),i=e.i(322831),a=e.i(418371),s=e.i(916925),r=e.i(196631),l=e.i(784774),n=e.i(201787);let o=["standard","flex","balanced","priority","ultrafast","batches"],d={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},c=["token","second","image","pixel","character","query","request"],p=["input","cache_read","explicit_cache_read","cache_write","output"],u=["","text","image","audio","video","reasoning"],m=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,g=/^(.+)_above_(\d+)k_tokens$/,h=/^(.+)_above_(\d+hr)$/,f=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,x=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,b=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},_={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},y={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},j={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},v=({charge:e,modality:t},i)=>t?n.f.t("deploymentPricing.withModality",{modality:n.f.t(y[i][t]),charge:n.f.t(_.inline[e])}):n.f.t(_[i][e]),w=(e,t)=>e.cacheDuration?n.f.t("deploymentPricing.withCacheDuration",{charge:v(e,t),duration:e.cacheDuration}):v(e,t),k=e=>n.f.t(j[e]),N=e=>n.f.t("deploymentPricing.rowLabel",{charge:w(e,"start"),unit:k(e.unit)}),$=e=>[p.indexOf(e.charge),c.indexOf(e.unit),u.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],P=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),M=e=>0===e?n.f.t("deploymentPricing.free"):P.format(e),S=(e,t)=>M("token"===t?1e6*e:e),I=/^(.*?)_?cost_per_(.+)$/,C=/^(.+?)_cost(?:_(.+))?$/,z=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},O=e=>{let t=I.exec(e);if(t)return n.f.t("deploymentPricing.rowLabel",{charge:t[1]?z(t[1]):n.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=C.exec(e);return i?z([i[1],i[2]].filter(Boolean).join(" ")):z(e)},H=e=>"price"===e.cost.kind,A=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=b(e,m),[a,s]=b(t,g),[r,l]=b(a,h),n=(e=>{let t=f.exec(e);if(t){let[,e,i="",a]=t;return c.includes(a)?{charge:e,modality:i,unit:a}:null}let i=x.exec(e);if(i){let[,e,t,a=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:a,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:a,unit:"token"}}return null})(r);return n?{kind:"price",...n,cacheDuration:l??"",threshold:null===s?null:1e3*Number(s),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),L=(e,t)=>e.filter((i,a)=>e.findIndex(e=>t(e)===t(i))===a),T=(e,t)=>{let i,a;return i=$(e.cost),a=$(t.cost),i.reduce((e,t,i)=>e||t-a[i],0)},B=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),D=({cost:e,value:t})=>{let i=w(e,"inline"),a=S(t,e.unit);return"token"===e.unit?n.f.t("deploymentPricing.tierPrice",{price:a,charge:i}):n.f.t("deploymentPricing.tierPricePerUnit",{price:a,charge:i,unit:k(e.unit)})};function E({group:e,tiers:a}){let{t:s}=(0,i.f)(),r=null===e.threshold?s("deploymentPricing.basePrices"):s("deploymentPricing.aboveThreshold",{threshold:B(e.threshold)});return(0,t.jsxs)(l.z,{"aria-label":r,children:[null!==e.threshold&&(0,t.jsx)(l.N,{className:"hover:bg-transparent",children:(0,t.jsx)(l.V,{colSpan:a.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:r})}),e.rows.map(e=>(0,t.jsxs)(l.N,{children:[(0,t.jsx)(l.V,{scope:"row",className:"font-normal",children:e.label}),a.map(i=>(0,t.jsx)(l.d,{className:l.J,children:e.prices[i]},i))]},e.label))]})}function U({deployment:e}){let r,c,p,u,m,{t:g}=(0,i.f)(),h=(p=[...L(c=(r=A(e.costs)).filter(H),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),u=o.filter(e=>c.some(({cost:t})=>t.tier===e)),{tiers:u,groups:m=p.map(e=>{let t;return{threshold:e,rows:[...L(t=c.filter(({cost:t})=>t.threshold===e),({cost:e})=>N(e))].sort(T).map(({cost:e})=>({label:N(e),prices:Object.fromEntries(t.filter(t=>N(t.cost)===N(e)).map(e=>[e.cost.tier,S(e.value,e.cost.unit)]))}))}}),hasBlankCells:m.some(({rows:e})=>e.some(({prices:e})=>u.some(t=>!(t in e)))),unknown:r.filter(e=>!H(e)).map(({key:e,value:t})=>({key:e,label:O(e),price:M(t)}))}),f=e.tiered_pricing,x=(0,s.h)(e.provider).displayName;return(0,t.jsxs)("section",{"aria-label":g("deploymentPricing.providerPricing",{provider:x}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,t.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,t.jsx)(a.f,{provider:e.provider}),x]}),h.groups.length>0&&(0,t.jsxs)(l.s,{children:[(0,t.jsx)(l.E,{children:(0,t.jsxs)(l.N,{className:"hover:bg-transparent",children:[(0,t.jsx)(l.V,{children:g("deploymentPricing.chargeColumn")}),h.tiers.map(e=>(0,t.jsx)(l.V,{className:l.J,children:g(d[e])},e))]})}),h.groups.map(e=>(0,t.jsx)(E,{group:e,tiers:h.tiers},e.threshold??"base"))]}),h.hasBlankCells&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:g("deploymentPricing.blankCells")}),f.length>0&&(0,t.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,t.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:g("deploymentPricing.tieredPricing")}),(0,t.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:f.map(e=>(0,t.jsx)("li",{children:(({range:[e,t],costs:i})=>{let a=A(i),s=[...[...a.filter(H)].sort(T).map(D),...a.filter(e=>!H(e)).map(({key:e,value:t})=>n.f.t("deploymentPricing.tierPrice",{price:M(t),charge:O(e).toLowerCase()}))];return n.f.t("deploymentPricing.tierRange",{low:B(e),high:B(t),prices:s.join(" / ")})})(e)},e.range.join("-")))})]}),h.unknown.length>0&&(0,t.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:h.unknown.map(({key:e,label:i,price:a})=>(0,t.jsxs)("div",{className:"contents",children:[(0,t.jsx)("dt",{className:"truncate",children:i}),(0,t.jsx)("dd",{className:"text-right tabular-nums",children:a})]},e))})]})}e.s(["f",0,function({deployments:e,heading:a,className:s}){let{t:l}=(0,i.f)();return e&&0!==e.length?(0,t.jsxs)("div",{className:(0,r.c)("flex flex-col gap-3",s),children:[a,e.length>1&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:l("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,i)=>(0,t.jsx)(U,{deployment:e},`${e.provider}-${i}`)),(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:l("deploymentPricing.requestAdjustments")})]}):null}],33355)},402874,e=>{var t=e.i(843476),i=e.i(143488),a=e.i(912089),s=e.i(636772),r=e.i(283713),l=e.i(602869),n=e.i(782066),o=e.i(275144),d=e.i(886400),c=e.i(321836),p=e.i(487486),u=e.i(972518),m=e.i(799647),g=e.i(522016),h=e.i(322831),f=e.i(731565),x=e.i(492941);let b="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var _=e.i(519455),y=e.i(755146),j=e.i(664659),v=e.i(164668);let w=()=>{let{t:e}=(0,h.f)(),i=(0,f.f)(),{data:a,isLoading:s,isError:r,refetch:l}=(0,x.f)();return i?null:(0,t.jsxs)(y.G,{modal:!1,children:[(0,t.jsxs)(y.X,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(_.K,{variant:"ghost",className:`${b} border-0!`}),children:[e("shell.blogDropdown.trigger"),(0,t.jsx)(j.f,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(y.L,{align:"end",side:"bottom",className:"w-auto",children:s?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(v.f,{role:"img","aria-label":e("shell.blogDropdown.loading"),className:"size-4 animate-spin"})}):r?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:e("shell.blogDropdown.failed")}),(0,t.jsx)(_.K,{variant:"outline",size:"sm",onClick:()=>l(),children:e("shell.blogDropdown.retry")})]}):a&&0!==a.posts.length?(0,t.jsxs)(t.Fragment,{children:[a.posts.slice(0,5).map(e=>(0,t.jsx)(y.r,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(y.U,{}),(0,t.jsx)(y.r,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:e("shell.blogDropdown.viewAll")})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:e("shell.blogDropdown.empty")})})]})},k=()=>(0,t.jsx)(j.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0}),N=()=>{let{t:e}=(0,h.f)();return(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:b,children:[e("shell.docsLink.label"),(0,t.jsx)(k,{})]})};var $=e.i(747862),P=e.i(746798),M=e.i(196631),S=e.i(350682);let I=[{href:"https://www.litellm.ai/support",label:"shell.communityButtons.slackLabel",tooltip:"shell.communityButtons.slackTooltip",Icon:e.i(965519).f},{href:"https://github.com/BerriAI/litellm",label:"shell.communityButtons.githubLabel",tooltip:"shell.communityButtons.githubTooltip",Icon:S.f}],C=()=>{let{t:e}=(0,h.f)();return(0,s.f)()?null:(0,t.jsx)(P.g,{children:(0,t.jsx)($.f,{"aria-label":e("shell.communityButtons.label"),children:I.map(({href:i,label:a,tooltip:s,Icon:r})=>(0,t.jsxs)(P.W,{children:[(0,t.jsx)(P.y,{render:(0,t.jsx)("a",{href:i,target:"_blank",rel:"noopener noreferrer","aria-label":e(a),className:(0,M.c)((0,_.i)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(r,{})}),(0,t.jsx)(P.Y,{children:e(s)})]},i))})})};var z=e.i(895335),O=e.i(135214),H=e.i(556260),A=e.i(319684),L=e.i(115571),T=e.i(222038),B=e.i(708347),D=e.i(344523),E=e.i(243553),U=e.i(465261),R=e.i(292270),F=e.i(263488),q=e.i(581418),K=e.i(284614),J=e.i(618566),Y=e.i(799676),V=e.i(337822),W=e.i(772436),G=e.i(699375),Q=e.i(922407),X=e.i(271645);let Z=({onLogout:e,variant:i="navbar",collapsed:r=!1})=>{let{t:l}=(0,h.f)(),{userId:o,userEmail:d,userRole:c,userRoleLabel:u,isViewOnly:m,premiumUser:g,loginMethod:x}=(0,O.f)(),b=(0,J.useRouter)(),[_,y]=(0,X.useState)(!1),v=(0,s.f)(),w=(0,f.f)(),k=(0,a.f)(),[N,$]=(0,H.f)(o),S=o&&!m&&(0,B.u)(c),[I,C]=(0,X.useState)(!1);(0,X.useEffect)(()=>{C("true"===(0,L.t)("disableShowNewBadge"))},[]);let z=d||o||"user",Z=function(e,t){let i=e?.split("@")[0]?.trim();if(i){let e=i.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(d,o),ee=function(e){let t=0;for(let i=0;i<e.length;i+=1)t=e.charCodeAt(i)+((t<<5)-t);return Math.abs(t)%360}(z),et=(0,T.f)(d,o);return(0,t.jsxs)(V.V,{open:_,onOpenChange:y,children:["sidebar"===i?(0,t.jsxs)(V.J,{render:(0,t.jsx)("button",{type:"button",className:(0,M.c)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",r?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":l("shell.userMenu.triggerAria",{role:u??l("shell.userMenu.unknownRole"),identity:d||o||l("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog",title:r?et:void 0}),children:[(0,t.jsx)(Y.E,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(Y.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:Z})}),!r&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:et}),u&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:u})]}),(0,t.jsx)(D.f,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(V.J,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":l("shell.userMenu.triggerAria",{role:u??l("shell.userMenu.unknownRole"),identity:d||o||l("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog"}),children:[(0,t.jsx)(Y.E,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(Y.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:Z})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:et}),(0,t.jsx)(j.f,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(V.T,{align:"sidebar"===i?"start":"end",side:"sidebar"===i?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(F.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:d||"-"})]}),g?(0,t.jsxs)(p.f,{children:[(0,t.jsx)(E.f,{className:"size-3"}),l("shell.userMenu.premium")]}):(0,t.jsx)(P.g,{children:(0,t.jsxs)(P.W,{children:[(0,t.jsxs)(P.y,{render:(0,t.jsx)(p.f,{variant:"outline"}),children:[(0,t.jsx)(E.f,{className:"size-3"}),l("shell.userMenu.standard")]}),(0,t.jsx)(P.Y,{side:"left",children:l("shell.userMenu.upgradeHint")})]})})]}),(0,t.jsx)(W.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(K.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.userId")})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:o||"-",children:o||"-"}),(0,t.jsx)(Q.f,{value:o,label:l("shell.userMenu.copyUserId")})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(q.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.role")})]}),(0,t.jsx)("span",{children:u})]}),(0,t.jsx)(W.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.hideNewBadges")}),(0,t.jsx)(G.f,{size:"sm",checked:I,onCheckedChange:e=>{C(e),e?(0,L.A)("disableShowNewBadge","true"):(0,L.p)("disableShowNewBadge"),(0,L.s)("disableShowNewBadge")},"aria-label":l("shell.userMenu.toggleHideNewBadges")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.hidePrompts")}),(0,t.jsx)(G.f,{size:"sm",checked:v,onCheckedChange:e=>{e?(0,L.A)("disableShowPrompts","true"):(0,L.p)("disableShowPrompts"),(0,L.s)("disableShowPrompts")},"aria-label":l("shell.userMenu.toggleHidePrompts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.hideBlogPosts")}),(0,t.jsx)(G.f,{size:"sm",checked:w,onCheckedChange:e=>{e?(0,L.A)("disableBlogPosts","true"):(0,L.p)("disableBlogPosts"),(0,L.s)("disableBlogPosts")},"aria-label":l("shell.userMenu.toggleHideBlogPosts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.hideBouncingIcon")}),(0,t.jsx)(G.f,{size:"sm",checked:k,onCheckedChange:e=>{e?(0,L.A)("disableBouncingIcon","true"):(0,L.p)("disableBouncingIcon"),(0,L.s)("disableBouncingIcon")},"aria-label":l("shell.userMenu.toggleHideBouncingIcon")})]}),!0===g&&S&&(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:l("shell.userMenu.hideAgent",{agent:A.f})}),(0,t.jsx)(G.f,{size:"sm",checked:N,onCheckedChange:$,"aria-label":l("shell.userMenu.toggleHideAgent",{agent:A.f})})]})]}),(0,t.jsx)(W.f,{}),"username_password"===x&&(0,t.jsxs)("button",{type:"button",onClick:()=>{y(!1),b.push((0,n.B)("change-password"))},className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(U.f,{className:"size-4"}),l("shell.userMenu.changePassword")]}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(R.f,{className:"size-4"}),l("shell.userMenu.logout")]})]})]})};var ee=e.i(455880),et=e.i(853295),ei=e.i(383862);let ea="h-auto max-h-full w-auto max-w-full object-contain";e.s(["f",0,({accessToken:e,isPublicPage:f=!1,sidebarCollapsed:x=!1,onToggleSidebar:b,deploymentHealth:_,deploymentHealthLink:y,publicNavigation:j})=>{let{t:v}=(0,h.f)(),k=(0,l._h)(),{logoUrl:$,logoUrlDark:P}=(0,o.i)(),{data:S}=(0,i.f)(e),I=S?.litellm_version,O=(0,a.f)(),H=(0,s.f)(),{isControlPlane:A,selectedWorker:L}=(0,r.f)(),T=A&&null!==L,B=$||`${k}/get_image`,D=P||$||`${k}/get_image?theme=dark`,E=(0,d.r)(e);return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:(0,M.c)("flex h-14 items-center px-4",f&&"gap-4 sm:px-6"),children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[b&&(0,t.jsx)("button",{onClick:b,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:v(x?"shell.navbar.expandSidebar":"shell.navbar.collapseSidebar"),children:(0,t.jsx)("span",{className:"text-lg",children:x?(0,t.jsx)(m.f,{className:"size-[18px]"}):(0,t.jsx)(u.f,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.default,{href:(0,n.B)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:(0,M.c)("flex h-10 max-w-48 items-center justify-center overflow-hidden",f&&"h-8 max-w-36 sm:h-10 sm:max-w-48"),children:[(0,t.jsx)("img",{src:B,alt:v("shell.navbar.logoAlt"),className:(0,M.c)(ea,"dark:hidden")}),(0,t.jsx)("img",{src:D,alt:"","aria-hidden":!0,className:(0,M.c)(ea,"hidden dark:block")})]})})}),I&&(0,t.jsxs)("div",{className:"relative",children:[!O&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:v("shell.navbar.thanks"),children:"🌑"}),(0,t.jsx)(p.f,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",I]})})]})]})]}),j,!f&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(et.f,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[T&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(ei.f,{onWorkerSwitch:t=>{(0,d.v)(e).finally(()=>{window.location.href=`${(0,c.D)()}?worker=${encodeURIComponent(t)}`})}})}),(0,t.jsxs)("nav",{"aria-label":v("shell.navbar.productDocs"),className:`flex min-w-0 items-center gap-2 ${T?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(N,{}),(0,t.jsx)(w,{})]}),!H&&(0,t.jsx)("div",{className:"hidden shrink-0 items-center border-l border-border pl-4 sm:flex",children:(0,t.jsx)(C,{})}),!f&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(ee.f,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),_,(0,t.jsx)(z.f,{accessToken:e,deploymentHealthLink:y}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(Z,{onLogout:E})]})})]})]})})})}],402874)},913761,e=>{var t=e.i(602869),i=e.i(916925),a=e.i(735450),s=e.i(201787);let r={chat:"publicModelHub.format.modes.chat",completion:"publicModelHub.format.modes.completion",embedding:"publicModelHub.format.modes.embedding",rerank:"publicModelHub.format.modes.rerank",image_generation:"publicModelHub.format.modes.imageGeneration",image_edit:"publicModelHub.format.modes.imageEdit",audio_speech:"publicModelHub.format.modes.audioSpeech",audio_transcription:"publicModelHub.format.modes.audioTranscription",video_generation:"publicModelHub.format.modes.videoGeneration",realtime:"publicModelHub.format.modes.realtime",responses:"publicModelHub.format.modes.responses",batch:"publicModelHub.format.modes.batch",ocr:"publicModelHub.format.modes.ocr",decisions:"publicModelHub.format.modes.decisions"},l=e=>e.replaceAll("_"," ").replace(/^./,e=>e.toUpperCase()),n=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:4,useGrouping:!1}),o={supports_vision:"publicModelHub.format.capabilities.vision",supports_function_calling:"publicModelHub.format.capabilities.functionCalling",supports_reasoning:"publicModelHub.format.capabilities.reasoning",supports_web_search:"publicModelHub.format.capabilities.webSearch",supports_parallel_function_calling:"publicModelHub.format.capabilities.parallelFunctionCalling",supports_url_context:"publicModelHub.format.capabilities.urlContext",supports_fast_mode:"publicModelHub.format.capabilities.fastMode"},d=Object.keys(o),c=e=>{let t=d.indexOf(e);return -1===t?d.length:t},p=e=>Object.hasOwn(o,e)?s.f.t(o[e]):l(e.replace(/^supports_/,"")),u={healthy:{tone:"success",label:"publicModelHub.format.health.healthy"},unhealthy:{tone:"error",label:"publicModelHub.format.health.unhealthy"}};e.s(["m",0,"text-[15px] leading-[22px] font-semibold","G",0,e=>p(`supports_${e}`),"a",0,e=>0===e?s.f.t("publicModelHub.format.free"):e?n.format(1e6*e):"-","k",0,e=>e?e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${Math.round(e/1e3)}K`:String(e):"-","M",0,()=>(0,t._h)(),"H",0,e=>{let t=e.health_status??"";return Object.hasOwn(u,t)?{tone:u[t].tone,label:s.f.t(u[t].label)}:{tone:"neutral",label:s.f.t("publicModelHub.format.notChecked")}},"S",0,e=>{let t=(0,a.i)(e);return t?Object.hasOwn(r,t)&&s.f.t(r[t])||l(t):"-"},"E",0,e=>Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e).sort((e,t)=>c(e)-c(t)).map(p),"q",0,e=>(0,i.h)(e).logo,"P",0,e=>(0,i.h)(e).displayName.replace(/\s*\(.*\)$/,""),"W",0,(e,t=Date.now())=>{let i=e?new Date(e).getTime():0/0;if(Number.isNaN(i))return"-";let a=Math.max(0,Math.round((t-i)/6e4));if(a<1)return s.f.t("publicModelHub.format.relativeTime.justNow");if(a<60)return s.f.t("publicModelHub.format.relativeTime.minutes",{minutes:a});let r=Math.round(a/60);if(r<24)return s.f.t("publicModelHub.format.relativeTime.hours",{hours:r});let l=Math.round(r/24);return s.f.t("publicModelHub.format.relativeTime.days",{count:l})},"Q",0,e=>Object.entries(e??{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:t})=>({title:e,url:t})),"X",0,e=>e.replaceAll("*","<model-name>")])},489432,e=>{var t=e.i(66330),i=e.i(209261),a=e.i(909947),s=e.i(865361),r=e.i(602869),l=e.i(735450);e.i(201787);var n=e.i(913761);let o="Hello, how are you?",d=e=>{let t;if(!e.pass_through_path)return null;let i=(t=e.pass_through_methods,!t?.length||t.includes("POST")?"POST":t[0]),a=i.replace(/[\\"$`]/g,"\\$&"),s=`${(0,n.M)()}${e.pass_through_path}`.replace(/[\\"$`]/g,"\\$&");return"POST"===i?`curl -X "${a}" "${s}" \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, how are you?"}'`:`curl -X "${a}" "${s}" \\
  -H "Authorization: Bearer your_api_key"`};e.s(["c",0,e=>e.pass_through_path?d(e):e.mode&&"chat"!==e.mode?null:`curl ${(0,n.M)()}/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${(0,n.X)(e.model_group)}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`,"j",0,()=>`/plugin marketplace add ${(0,n.M)()}/claude-code/marketplace.json`,"W",0,e=>{if("passthrough"===e.mode||e.pass_through_path)return d(e)??"";let i=(0,n.X)(e.model_group);if((0,l._)(e.mode))return(0,t.f)(e.mode,(0,n.M)(),i,(0,r.kR)());let c={apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:o,chatHistory:[{role:"user",content:o,isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,s.t)(e.mode||"chat"),selectedModel:i,selectedSdk:"openai",proxySettings:{PROXY_BASE_URL:(0,n.M)()}};return(0,a.f)(c)},"g",0,d,"z",0,e=>(0,i.a)(e)])},466098,348594,831538,e=>{var t=e.i(586448),i=e.i(602869),a=e.i(271645),s=e.i(198458),r=e.i(735450);let l="mode",n="providers",o="features",d=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],c=(e,t)=>{let i,a;return i=`filter[${e}][in]`,""===(a=d(t).join(","))?[]:[[i,a]]},p=e=>{switch(e.id){case l:return c(e.id,(0,r.q)(d(e.value)));case n:case o:return c(e.id,e.value);default:return[]}},u=e=>Object.fromEntries(e.flatMap(p)),m=(e,t)=>d(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let a=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?a:[...a,{id:t,value:i}]};e.s(["g",0,o,"e",0,l,"T",0,n,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,u,"R",0,g],348594);let h="/public/v1/model_hub",f=["publicModelHub","list"],x=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await i.Mw.get(h,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,h,"t",0,e=>{let t=(0,s.f)({queryKey:f,fetchPage:b,serializeFilters:u,defaultSorting:x,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,r=(0,a.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),d=(0,a.useCallback)(e=>r(n,e),[r]),c=(0,a.useCallback)(e=>r(l,e),[r]),p=(0,a.useCallback)(e=>r(o,e),[r]);return{...t,providerValues:m(t.columnFilters,n),onProvidersChange:d,modeValues:m(t.columnFilters,l),onModesChange:c,featureValues:m(t.columnFilters,o),onFeaturesChange:p,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let _=["providers","modes","features"];e.s(["f",0,e=>{let[a,s,r]=(0,t.f)({queries:_.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${h}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:a,modes:s,features:r}}],466098)},747862,e=>{var t=e.i(843476);e.i(176782),e.i(911825);var i=e.i(225913),a=e.i(196631);e.i(772436);let s=(0,i.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:i,...r}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":i,className:(0,a.c)(s({orientation:i}),e),...r})}])},735450,e=>{var t=e.i(411382),i=e.i(772325),a=e.i(54105),s=e.i(240192),r=e.i(138620);let l="https://docs.litellm.ai/docs/decisions",n=`${l}#supported-providers`,o=`${l}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:a.z.string(),mode:a.z.string().nullish(),supported_endpoints:i.z.array(a.z.string()).nullish()}),p="decisions",u=["evaluation",p],m=e=>null!=e&&u.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),h=e=>{let t=s.z.record(a.z.string(),r.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=c.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},x=t.z.object({decision_model_providers:i.z.array(a.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),_=(e,t,i)=>({row:e,routeOrder:f[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,l,"d",0,p,"Z",0,n,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=h(e),i=t.filter(([,e])=>g(e)),a=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:a}},"W",0,(e,t)=>h(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(_).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?p:e??void 0,"q",0,e=>e.flatMap(e=>e===p?u:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=x.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let a=t(e);if(""===a)return!0;let s=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!s.some(e=>e.includes(a))||a.split(/\s+/).every(e=>s.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,a){return e.filter(e=>i(t,a(e)))},"h",0,i,"e",0,function(e,i,a){let s=t(i);if(""===s)return[...e];let r=e=>{let t=a(e).toLowerCase();return 1e3*(t===s)+100*!!t.startsWith(s)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}])})();