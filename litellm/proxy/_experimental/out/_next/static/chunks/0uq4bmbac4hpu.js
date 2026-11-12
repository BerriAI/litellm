(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,66330,e=>{var t=e.i(735450);e.s(["f",0,(e,i,a,r)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${i}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(r)}: "Bearer your_api_key"},
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
    headers={${JSON.stringify(r)}: "Bearer your_api_key"},
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

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:a,accessToken:r,apiKey:n,inputMessage:o,chatHistory:s,selectedTags:l,selectedVectorStores:p,selectedGuardrails:c,selectedPolicies:d,selectedVoice:u,endpointType:m,selectedModel:g,selectedSdk:f,proxySettings:_,customHeaders:h}=e,b="session"===a?r:n,y=window.location.origin,x=_?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?y=x:_?.PROXY_BASE_URL&&(y=_.PROXY_BASE_URL);let v=o||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};l.length>0&&(k.tags=l),p.length>0&&(k.vector_stores=p),c.length>0&&(k.guardrails=c),d.length>0&&(k.policies=d);let $=g||"your-model-name",P=h&&Object.keys(h).length>0?`,
	default_headers=${JSON.stringify(h,null,2).replace(/\n/g,"\n	")}`:"",M="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${P}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${P}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let a=j.length>0?j:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${$}",
    messages=${JSON.stringify(a,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.Q.RESPONSES:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let a=j.length>0?j:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${$}",
    input=${JSON.stringify(a,null,4)}${t}
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
	model="${$}",
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
	model="${$}",
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
	model="${$}",
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
	input="${o||"Your string here"}",
	model="${$}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${$}",
	file=audio_file${o?`,
	prompt="${o.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${$}",
	input="${o||"Your text to convert to speech here"}",
	voice="${u}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${$}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${M}
${i}`}])},583408,e=>{var t=e.i(843476),i=e.i(555436),a=e.i(322831),r=e.i(519455),n=e.i(950594),o=e.i(967489);e.s(["f",0,function({search:e,tag:s,scan:l,tags:p,active:c,setSearch:d,setTag:u,setScan:m,clear:g}){let{t:f}=(0,a.f)(),_=[{value:"",label:f("skills.filters.allTags")},...[...new Set([...p,...s?[s]:[]])].map(e=>({value:e,label:e}))],h=[{value:"all",label:f("skills.filters.allScans")},{value:"yes",label:f("skills.filters.scanned")},{value:"no",label:f("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(n.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(n.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(n.M,{"aria-label":f("skills.filters.search"),placeholder:f("skills.filters.search"),value:e,onChange:e=>d(e.target.value)})]}),(0,t.jsxs)(o.d,{items:_,value:s,onValueChange:e=>u(e??""),children:[(0,t.jsx)(o.P,{className:"w-44","aria-label":f("skills.filters.tag"),children:(0,t.jsx)(o.U,{})}),(0,t.jsx)(o.r,{children:_.map(e=>(0,t.jsx)(o.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(o.d,{items:h,value:l,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(o.P,{className:"w-48","aria-label":f("skills.table.columns.securityScan"),children:(0,t.jsx)(o.U,{})}),(0,t.jsx)(o.r,{children:h.map(e=>(0,t.jsx)(o.T,{value:e.value,children:e.label},e.value))})]}),c&&(0,t.jsx)(r.K,{variant:"ghost",size:"sm",onClick:g,children:f("skills.filters.clear")})]})}])},594107,e=>{var t=e.i(843476),i=e.i(494862);e.i(622826);var a=e.i(200208),r=e.i(112179),n=e.i(487486),o=e.i(201787);let s={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:o.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(o.f.language)??"—"})},{id:"keywords",header:o.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(n.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(e=>({id:e,accessorKey:e,size:190,enableSorting:!0,header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:o.f.t(s[e])}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[e]??void 0,children:i.original[e]||o.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:o.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(a.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:o.f.t("skills.table.scanHelp"),children:o.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:o.f.t("skills.table.scanHelp"),children:(0,t.jsx)(r.f,{tone:"neutral",label:o.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{var t=e.i(201787);let i=/^[0-9a-fA-F]{64}$/,a=/^https?:\/\//i,r="<your-litellm-key>";e.s(["i",0,r,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${r}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,a,r)=>{if("archive"!==t.source||!(e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e))||t.sha256&&!i.test(t.sha256))return null;let n=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${a.replace(/\/+$/,"")}${n}`,r);return/^https?:$/.test(e.protocol)&&t.url===e.href?n:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&a.test(e.url)?e.url:null,"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[]])},962600,e=>{var t=e.i(438847),i=e.i(45570),a=e.i(899426);let r=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],n={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:r,defaultSort:{id:"name",desc:!1},filterColumns:[]},o={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},s={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,r,"W",0,function(e,t){let i=(0,a.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),r=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),n="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&r&&n},"n",0,function(e){return{sort_by:r.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=r){let a=(0,i.f)({...n,sortFields:e}),[l,p]=(0,t.i)(o,s),c=e=>{p(e),a.onPaginationChange({...a.pagination,pageIndex:0})};return{...a,...l,setTag:e=>c({tag:e}),setScan:e=>c({scan:e}),clear:()=>{c(null),a.setSearch("")},active:!!(a.search.trim()||l.tag||"all"!==l.scan)}}])},33355,e=>{var t=e.i(843476),i=e.i(322831),a=e.i(418371),r=e.i(916925),n=e.i(196631),o=e.i(784774),s=e.i(201787);let l=["standard","flex","balanced","priority","ultrafast","batches"],p={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},c=["token","second","image","pixel","character","query","request"],d=["input","cache_read","explicit_cache_read","cache_write","output"],u=["","text","image","audio","video","reasoning"],m=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,g=/^(.+)_above_(\d+)k_tokens$/,f=/^(.+)_above_(\d+hr)$/,_=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,h=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,b=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},y={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},x={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},v={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},w=({charge:e,modality:t},i)=>t?s.f.t("deploymentPricing.withModality",{modality:s.f.t(x[i][t]),charge:s.f.t(y.inline[e])}):s.f.t(y[i][e]),j=(e,t)=>e.cacheDuration?s.f.t("deploymentPricing.withCacheDuration",{charge:w(e,t),duration:e.cacheDuration}):w(e,t),k=e=>s.f.t(v[e]),$=e=>s.f.t("deploymentPricing.rowLabel",{charge:j(e,"start"),unit:k(e.unit)}),P=e=>[d.indexOf(e.charge),c.indexOf(e.unit),u.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],M=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),S=e=>0===e?s.f.t("deploymentPricing.free"):M.format(e),I=(e,t)=>S("token"===t?1e6*e:e),N=/^(.*?)_?cost_per_(.+)$/,O=/^(.+?)_cost(?:_(.+))?$/,H=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},C=e=>{let t=N.exec(e);if(t)return s.f.t("deploymentPricing.rowLabel",{charge:t[1]?H(t[1]):s.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=O.exec(e);return i?H([i[1],i[2]].filter(Boolean).join(" ")):H(e)},A=e=>"price"===e.cost.kind,L=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=b(e,m),[a,r]=b(t,g),[n,o]=b(a,f),s=(e=>{let t=_.exec(e);if(t){let[,e,i="",a]=t;return c.includes(a)?{charge:e,modality:i,unit:a}:null}let i=h.exec(e);if(i){let[,e,t,a=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:a,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:a,unit:"token"}}return null})(n);return s?{kind:"price",...s,cacheDuration:o??"",threshold:null===r?null:1e3*Number(r),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),T=(e,t)=>e.filter((i,a)=>e.findIndex(e=>t(e)===t(i))===a),z=(e,t)=>{let i,a;return i=P(e.cost),a=P(t.cost),i.reduce((e,t,i)=>e||t-a[i],0)},E=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),D=({cost:e,value:t})=>{let i=j(e,"inline"),a=I(t,e.unit);return"token"===e.unit?s.f.t("deploymentPricing.tierPrice",{price:a,charge:i}):s.f.t("deploymentPricing.tierPricePerUnit",{price:a,charge:i,unit:k(e.unit)})};function F({group:e,tiers:a}){let{t:r}=(0,i.f)(),n=null===e.threshold?r("deploymentPricing.basePrices"):r("deploymentPricing.aboveThreshold",{threshold:E(e.threshold)});return(0,t.jsxs)(o.z,{"aria-label":n,children:[null!==e.threshold&&(0,t.jsx)(o.N,{className:"hover:bg-transparent",children:(0,t.jsx)(o.V,{colSpan:a.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:n})}),e.rows.map(e=>(0,t.jsxs)(o.N,{children:[(0,t.jsx)(o.V,{scope:"row",className:"font-normal",children:e.label}),a.map(i=>(0,t.jsx)(o.d,{className:o.J,children:e.prices[i]},i))]},e.label))]})}function R({deployment:e}){let n,c,d,u,m,{t:g}=(0,i.f)(),f=(d=[...T(c=(n=L(e.costs)).filter(A),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),u=l.filter(e=>c.some(({cost:t})=>t.tier===e)),{tiers:u,groups:m=d.map(e=>{let t;return{threshold:e,rows:[...T(t=c.filter(({cost:t})=>t.threshold===e),({cost:e})=>$(e))].sort(z).map(({cost:e})=>({label:$(e),prices:Object.fromEntries(t.filter(t=>$(t.cost)===$(e)).map(e=>[e.cost.tier,I(e.value,e.cost.unit)]))}))}}),hasBlankCells:m.some(({rows:e})=>e.some(({prices:e})=>u.some(t=>!(t in e)))),unknown:n.filter(e=>!A(e)).map(({key:e,value:t})=>({key:e,label:C(e),price:S(t)}))}),_=e.tiered_pricing,h=(0,r.h)(e.provider).displayName;return(0,t.jsxs)("section",{"aria-label":g("deploymentPricing.providerPricing",{provider:h}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,t.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,t.jsx)(a.f,{provider:e.provider}),h]}),f.groups.length>0&&(0,t.jsxs)(o.s,{children:[(0,t.jsx)(o.E,{children:(0,t.jsxs)(o.N,{className:"hover:bg-transparent",children:[(0,t.jsx)(o.V,{children:g("deploymentPricing.chargeColumn")}),f.tiers.map(e=>(0,t.jsx)(o.V,{className:o.J,children:g(p[e])},e))]})}),f.groups.map(e=>(0,t.jsx)(F,{group:e,tiers:f.tiers},e.threshold??"base"))]}),f.hasBlankCells&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:g("deploymentPricing.blankCells")}),_.length>0&&(0,t.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,t.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:g("deploymentPricing.tieredPricing")}),(0,t.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:_.map(e=>(0,t.jsx)("li",{children:(({range:[e,t],costs:i})=>{let a=L(i),r=[...[...a.filter(A)].sort(z).map(D),...a.filter(e=>!A(e)).map(({key:e,value:t})=>s.f.t("deploymentPricing.tierPrice",{price:S(t),charge:C(e).toLowerCase()}))];return s.f.t("deploymentPricing.tierRange",{low:E(e),high:E(t),prices:r.join(" / ")})})(e)},e.range.join("-")))})]}),f.unknown.length>0&&(0,t.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:f.unknown.map(({key:e,label:i,price:a})=>(0,t.jsxs)("div",{className:"contents",children:[(0,t.jsx)("dt",{className:"truncate",children:i}),(0,t.jsx)("dd",{className:"text-right tabular-nums",children:a})]},e))})]})}e.s(["f",0,function({deployments:e,heading:a,className:r}){let{t:o}=(0,i.f)();return e&&0!==e.length?(0,t.jsxs)("div",{className:(0,n.c)("flex flex-col gap-3",r),children:[a,e.length>1&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:o("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,i)=>(0,t.jsx)(R,{deployment:e},`${e.provider}-${i}`)),(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:o("deploymentPricing.requestAdjustments")})]}):null}],33355)},913761,e=>{var t=e.i(602869),i=e.i(916925),a=e.i(735450),r=e.i(201787);let n={chat:"publicModelHub.format.modes.chat",completion:"publicModelHub.format.modes.completion",embedding:"publicModelHub.format.modes.embedding",rerank:"publicModelHub.format.modes.rerank",image_generation:"publicModelHub.format.modes.imageGeneration",image_edit:"publicModelHub.format.modes.imageEdit",audio_speech:"publicModelHub.format.modes.audioSpeech",audio_transcription:"publicModelHub.format.modes.audioTranscription",video_generation:"publicModelHub.format.modes.videoGeneration",realtime:"publicModelHub.format.modes.realtime",responses:"publicModelHub.format.modes.responses",batch:"publicModelHub.format.modes.batch",ocr:"publicModelHub.format.modes.ocr",decisions:"publicModelHub.format.modes.decisions"},o=e=>e.replaceAll("_"," ").replace(/^./,e=>e.toUpperCase()),s=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:4,useGrouping:!1}),l={supports_vision:"publicModelHub.format.capabilities.vision",supports_function_calling:"publicModelHub.format.capabilities.functionCalling",supports_reasoning:"publicModelHub.format.capabilities.reasoning",supports_web_search:"publicModelHub.format.capabilities.webSearch",supports_parallel_function_calling:"publicModelHub.format.capabilities.parallelFunctionCalling",supports_url_context:"publicModelHub.format.capabilities.urlContext",supports_fast_mode:"publicModelHub.format.capabilities.fastMode"},p=Object.keys(l),c=e=>{let t=p.indexOf(e);return -1===t?p.length:t},d=e=>Object.hasOwn(l,e)?r.f.t(l[e]):o(e.replace(/^supports_/,"")),u={healthy:{tone:"success",label:"publicModelHub.format.health.healthy"},unhealthy:{tone:"error",label:"publicModelHub.format.health.unhealthy"}};e.s(["m",0,"text-[15px] leading-[22px] font-semibold","G",0,e=>d(`supports_${e}`),"a",0,e=>0===e?r.f.t("publicModelHub.format.free"):e?s.format(1e6*e):"-","k",0,e=>e?e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${Math.round(e/1e3)}K`:String(e):"-","M",0,()=>(0,t._h)(),"H",0,e=>{let t=e.health_status??"";return Object.hasOwn(u,t)?{tone:u[t].tone,label:r.f.t(u[t].label)}:{tone:"neutral",label:r.f.t("publicModelHub.format.notChecked")}},"S",0,e=>{let t=(0,a.i)(e);return t?Object.hasOwn(n,t)&&r.f.t(n[t])||o(t):"-"},"E",0,e=>Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e).sort((e,t)=>c(e)-c(t)).map(d),"q",0,e=>(0,i.h)(e).logo,"P",0,e=>(0,i.h)(e).displayName.replace(/\s*\(.*\)$/,""),"W",0,(e,t=Date.now())=>{let i=e?new Date(e).getTime():0/0;if(Number.isNaN(i))return"-";let a=Math.max(0,Math.round((t-i)/6e4));if(a<1)return r.f.t("publicModelHub.format.relativeTime.justNow");if(a<60)return r.f.t("publicModelHub.format.relativeTime.minutes",{minutes:a});let n=Math.round(a/60);if(n<24)return r.f.t("publicModelHub.format.relativeTime.hours",{hours:n});let o=Math.round(n/24);return r.f.t("publicModelHub.format.relativeTime.days",{count:o})},"Q",0,e=>Object.entries(e??{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:t})=>({title:e,url:t})),"X",0,e=>e.replaceAll("*","<model-name>")])},489432,e=>{var t=e.i(66330),i=e.i(209261),a=e.i(909947),r=e.i(865361),n=e.i(602869),o=e.i(735450);e.i(201787);var s=e.i(913761);let l="Hello, how are you?",p=e=>{let t;if(!e.pass_through_path)return null;let i=(t=e.pass_through_methods,!t?.length||t.includes("POST")?"POST":t[0]),a=i.replace(/[\\"$`]/g,"\\$&"),r=`${(0,s.M)()}${e.pass_through_path}`.replace(/[\\"$`]/g,"\\$&");return"POST"===i?`curl -X "${a}" "${r}" \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, how are you?"}'`:`curl -X "${a}" "${r}" \\
  -H "Authorization: Bearer your_api_key"`};e.s(["c",0,e=>e.pass_through_path?p(e):e.mode&&"chat"!==e.mode?null:`curl ${(0,s.M)()}/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${(0,s.X)(e.model_group)}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`,"j",0,()=>`/plugin marketplace add ${(0,s.M)()}/claude-code/marketplace.json`,"W",0,e=>{if("passthrough"===e.mode||e.pass_through_path)return p(e)??"";let i=(0,s.X)(e.model_group);if((0,o._)(e.mode))return(0,t.f)(e.mode,(0,s.M)(),i,(0,n.kR)());let c={apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:l,chatHistory:[{role:"user",content:l,isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,r.t)(e.mode||"chat"),selectedModel:i,selectedSdk:"openai",proxySettings:{PROXY_BASE_URL:(0,s.M)()}};return(0,a.f)(c)},"g",0,p,"z",0,e=>(0,i.a)(e)])},466098,348594,831538,e=>{var t=e.i(586448),i=e.i(602869),a=e.i(271645),r=e.i(198458),n=e.i(735450);let o="mode",s="providers",l="features",p=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],c=(e,t)=>{let i,a;return i=`filter[${e}][in]`,""===(a=p(t).join(","))?[]:[[i,a]]},d=e=>{switch(e.id){case o:return c(e.id,(0,n.q)(p(e.value)));case s:case l:return c(e.id,e.value);default:return[]}},u=e=>Object.fromEntries(e.flatMap(d)),m=(e,t)=>p(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let a=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?a:[...a,{id:t,value:i}]};e.s(["g",0,l,"e",0,o,"T",0,s,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,u,"R",0,g],348594);let f="/public/v1/model_hub",_=["publicModelHub","list"],h=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await i.Mw.get(f,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,f,"t",0,e=>{let t=(0,r.f)({queryKey:_,fetchPage:b,serializeFilters:u,defaultSorting:h,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,n=(0,a.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),p=(0,a.useCallback)(e=>n(s,e),[n]),c=(0,a.useCallback)(e=>n(o,e),[n]),d=(0,a.useCallback)(e=>n(l,e),[n]);return{...t,providerValues:m(t.columnFilters,s),onProvidersChange:p,modeValues:m(t.columnFilters,o),onModesChange:c,featureValues:m(t.columnFilters,l),onFeaturesChange:d,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let y=["providers","modes","features"];e.s(["f",0,e=>{let[a,r,n]=(0,t.f)({queries:y.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${f}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:a,modes:r,features:n}}],466098)},735450,e=>{var t=e.i(411382),i=e.i(772325),a=e.i(54105),r=e.i(240192),n=e.i(138620);let o="https://docs.litellm.ai/docs/decisions",s=`${o}#supported-providers`,l=`${o}#self-hosted-laya-and-nimble`,p=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:a.z.string(),mode:a.z.string().nullish(),supported_endpoints:i.z.array(a.z.string()).nullish()}),d="decisions",u=["evaluation",d],m=e=>null!=e&&u.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>p.has(e)),f=e=>{let t=r.z.record(a.z.string(),n.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=c.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},_={decisions:0,"self-hosted":1,"needs-upgrade":2},h=t.z.object({decision_model_providers:i.z.array(a.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),y=(e,t,i)=>({row:e,routeOrder:_[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,o,"d",0,d,"Z",0,s,"L",0,l,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),i=t.filter(([,e])=>g(e)),a=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:a}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(y).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?d:e??void 0,"q",0,e=>e.flatMap(e=>e===d?u:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=h.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let a=t(e);if(""===a)return!0;let r=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!r.some(e=>e.includes(a))||a.split(/\s+/).every(e=>r.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,a){return e.filter(e=>i(t,a(e)))},"h",0,i,"e",0,function(e,i,a){let r=t(i);if(""===r)return[...e];let n=e=>{let t=a(e).toLowerCase();return 1e3*(t===r)+100*!!t.startsWith(r)+(1e3-t.length)};return[...e].sort((e,t)=>n(t)-n(e))}])}])})();