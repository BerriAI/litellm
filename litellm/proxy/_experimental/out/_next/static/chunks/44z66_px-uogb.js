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

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:a,accessToken:r,apiKey:n,inputMessage:s,chatHistory:o,selectedTags:l,selectedVectorStores:p,selectedGuardrails:d,selectedPolicies:c,selectedVoice:u,endpointType:m,selectedModel:g,selectedSdk:f,proxySettings:_,customHeaders:h}=e,y="session"===a?r:n,b=window.location.origin,x=_?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?b=x:_?.PROXY_BASE_URL&&(b=_.PROXY_BASE_URL);let v=s||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};l.length>0&&(k.tags=l),p.length>0&&(k.vector_stores=p),d.length>0&&(k.guardrails=d),c.length>0&&(k.policies=c);let P=g||"your-model-name",$=h&&Object.keys(h).length>0?`,
	default_headers=${JSON.stringify(h,null,2).replace(/\n/g,"\n	")}`:"",I="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${b}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${b}"${$}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let a=j.length>0?j:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${P}",
    messages=${JSON.stringify(a,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${P}",
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
    model="${P}",
    input=${JSON.stringify(a,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${P}",
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
	model="${P}",
	prompt="${s}",
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
	model="${P}",
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
	model="${P}",
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
	model="${P}",
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
	input="${s||"Your string here"}",
	model="${P}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${P}",
	file=audio_file${s?`,
	prompt="${s.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${P}",
	input="${s||"Your text to convert to speech here"}",
	voice="${u}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${P}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${i}`}])},583408,e=>{var t=e.i(843476),i=e.i(555436),a=e.i(322831),r=e.i(519455),n=e.i(950594),s=e.i(967489);e.s(["f",0,function({search:e,tag:o,scan:l,tags:p,active:d,setSearch:c,setTag:u,setScan:m,clear:g}){let{t:f}=(0,a.f)(),_=[{value:"",label:f("skills.filters.allTags")},...[...new Set([...p,...o?[o]:[]])].map(e=>({value:e,label:e}))],h=[{value:"all",label:f("skills.filters.allScans")},{value:"yes",label:f("skills.filters.scanned")},{value:"no",label:f("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(n.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(n.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(n.M,{"aria-label":f("skills.filters.search"),placeholder:f("skills.filters.search"),value:e,onChange:e=>c(e.target.value)})]}),(0,t.jsxs)(s.d,{items:_,value:o,onValueChange:e=>u(e??""),children:[(0,t.jsx)(s.P,{className:"w-44","aria-label":f("skills.filters.tag"),children:(0,t.jsx)(s.U,{})}),(0,t.jsx)(s.r,{children:_.map(e=>(0,t.jsx)(s.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(s.d,{items:h,value:l,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(s.P,{className:"w-48","aria-label":f("skills.table.columns.securityScan"),children:(0,t.jsx)(s.U,{})}),(0,t.jsx)(s.r,{children:h.map(e=>(0,t.jsx)(s.T,{value:e.value,children:e.label},e.value))})]}),d&&(0,t.jsx)(r.K,{variant:"ghost",size:"sm",onClick:g,children:f("skills.filters.clear")})]})}])},594107,e=>{var t=e.i(843476),i=e.i(494862);e.i(622826);var a=e.i(200208),r=e.i(112179),n=e.i(487486),s=e.i(201787);let o={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:s.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(s.f.language)??"—"})},{id:"keywords",header:s.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(n.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(e=>({id:e,accessorKey:e,size:190,enableSorting:!0,header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:s.f.t(o[e])}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[e]??void 0,children:i.original[e]||s.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:a})=>(0,t.jsx)(i.v,{column:a,title:s.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(a.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:s.f.t("skills.table.scanHelp"),children:s.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:s.f.t("skills.table.scanHelp"),children:(0,t.jsx)(r.f,{tone:"neutral",label:s.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{var t=e.i(201787);let i=/^[0-9a-fA-F]{64}$/,a=/^https?:\/\//i,r="<your-litellm-key>";e.s(["i",0,r,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${r}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,a,r)=>{if("archive"!==t.source||!(e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e))||t.sha256&&!i.test(t.sha256))return null;let n=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${a.replace(/\/+$/,"")}${n}`,r);return/^https?:$/.test(e.protocol)&&t.url===e.href?n:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&a.test(e.url)?e.url:null,"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[]])},962600,e=>{var t=e.i(438847),i=e.i(45570),a=e.i(899426);let r=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],n={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:r,defaultSort:{id:"name",desc:!1},filterColumns:[]},s={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},o={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,r,"W",0,function(e,t){let i=(0,a.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),r=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),n="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&r&&n},"n",0,function(e){return{sort_by:r.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=r){let a=(0,i.f)({...n,sortFields:e}),[l,p]=(0,t.i)(s,o),d=e=>{p(e),a.onPaginationChange({...a.pagination,pageIndex:0})};return{...a,...l,setTag:e=>d({tag:e}),setScan:e=>d({scan:e}),clear:()=>{d(null),a.setSearch("")},active:!!(a.search.trim()||l.tag||"all"!==l.scan)}}])},466098,348594,831538,33355,e=>{var t=e.i(586448),i=e.i(602869),a=e.i(271645),r=e.i(198458),n=e.i(735450);let s="mode",o="providers",l="features",p=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],d=(e,t)=>{let i,a;return i=`filter[${e}][in]`,""===(a=p(t).join(","))?[]:[[i,a]]},c=e=>{switch(e.id){case s:return d(e.id,(0,n.q)(p(e.value)));case o:case l:return d(e.id,e.value);default:return[]}},u=e=>Object.fromEntries(e.flatMap(c)),m=(e,t)=>p(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let a=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?a:[...a,{id:t,value:i}]};e.s(["g",0,l,"e",0,s,"T",0,o,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,u,"R",0,g],348594);let f="/public/v1/model_hub",_=["publicModelHub","list"],h=[{id:"model_group",desc:!1}],y=async(e,t)=>{try{return await i.Mw.get(f,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,f,"t",0,e=>{let t=(0,r.f)({queryKey:_,fetchPage:y,serializeFilters:u,defaultSorting:h,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,n=(0,a.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),p=(0,a.useCallback)(e=>n(o,e),[n]),d=(0,a.useCallback)(e=>n(s,e),[n]),c=(0,a.useCallback)(e=>n(l,e),[n]);return{...t,providerValues:m(t.columnFilters,o),onProvidersChange:p,modeValues:m(t.columnFilters,s),onModesChange:d,featureValues:m(t.columnFilters,l),onFeaturesChange:c,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let b=["providers","modes","features"];e.s(["f",0,e=>{let[a,r,n]=(0,t.f)({queries:b.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${f}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:a,modes:r,features:n}}],466098);var x=e.i(843476),v=e.i(322831),w=e.i(418371),j=e.i(916925),k=e.i(196631),P=e.i(784774),$=e.i(201787);let I=["standard","flex","balanced","priority","ultrafast","batches"],N={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},S=["token","second","image","pixel","character","query","request"],O=["input","cache_read","explicit_cache_read","cache_write","output"],C=["","text","image","audio","video","reasoning"],A=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,L=/^(.+)_above_(\d+)k_tokens$/,z=/^(.+)_above_(\d+hr)$/,E=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,T=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,D=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},M={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},R={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},q={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},F=({charge:e,modality:t},i)=>t?$.f.t("deploymentPricing.withModality",{modality:$.f.t(R[i][t]),charge:$.f.t(M.inline[e])}):$.f.t(M[i][e]),U=(e,t)=>e.cacheDuration?$.f.t("deploymentPricing.withCacheDuration",{charge:F(e,t),duration:e.cacheDuration}):F(e,t),B=e=>$.f.t(q[e]),H=e=>$.f.t("deploymentPricing.rowLabel",{charge:U(e,"start"),unit:B(e.unit)}),J=e=>[O.indexOf(e.charge),S.indexOf(e.unit),C.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],K=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),Y=e=>0===e?$.f.t("deploymentPricing.free"):K.format(e),V=(e,t)=>Y("token"===t?1e6*e:e),Q=/^(.*?)_?cost_per_(.+)$/,W=/^(.+?)_cost(?:_(.+))?$/,G=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},Z=e=>{let t=Q.exec(e);if(t)return $.f.t("deploymentPricing.rowLabel",{charge:t[1]?G(t[1]):$.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=W.exec(e);return i?G([i[1],i[2]].filter(Boolean).join(" ")):G(e)},X=e=>"price"===e.cost.kind,ee=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=D(e,A),[a,r]=D(t,L),[n,s]=D(a,z),o=(e=>{let t=E.exec(e);if(t){let[,e,i="",a]=t;return S.includes(a)?{charge:e,modality:i,unit:a}:null}let i=T.exec(e);if(i){let[,e,t,a=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:a,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:a,unit:"token"}}return null})(n);return o?{kind:"price",...o,cacheDuration:s??"",threshold:null===r?null:1e3*Number(r),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),et=(e,t)=>e.filter((i,a)=>e.findIndex(e=>t(e)===t(i))===a),ei=(e,t)=>{let i,a;return i=J(e.cost),a=J(t.cost),i.reduce((e,t,i)=>e||t-a[i],0)},ea=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),er=({cost:e,value:t})=>{let i=U(e,"inline"),a=V(t,e.unit);return"token"===e.unit?$.f.t("deploymentPricing.tierPrice",{price:a,charge:i}):$.f.t("deploymentPricing.tierPricePerUnit",{price:a,charge:i,unit:B(e.unit)})};function en({group:e,tiers:t}){let{t:i}=(0,v.f)(),a=null===e.threshold?i("deploymentPricing.basePrices"):i("deploymentPricing.aboveThreshold",{threshold:ea(e.threshold)});return(0,x.jsxs)(P.z,{"aria-label":a,children:[null!==e.threshold&&(0,x.jsx)(P.N,{className:"hover:bg-transparent",children:(0,x.jsx)(P.V,{colSpan:t.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:a})}),e.rows.map(e=>(0,x.jsxs)(P.N,{children:[(0,x.jsx)(P.V,{scope:"row",className:"font-normal",children:e.label}),t.map(t=>(0,x.jsx)(P.d,{className:P.J,children:e.prices[t]},t))]},e.label))]})}function es({deployment:e}){let t,i,a,r,n,{t:s}=(0,v.f)(),o=(a=[...et(i=(t=ee(e.costs)).filter(X),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),r=I.filter(e=>i.some(({cost:t})=>t.tier===e)),{tiers:r,groups:n=a.map(e=>{let t;return{threshold:e,rows:[...et(t=i.filter(({cost:t})=>t.threshold===e),({cost:e})=>H(e))].sort(ei).map(({cost:e})=>({label:H(e),prices:Object.fromEntries(t.filter(t=>H(t.cost)===H(e)).map(e=>[e.cost.tier,V(e.value,e.cost.unit)]))}))}}),hasBlankCells:n.some(({rows:e})=>e.some(({prices:e})=>r.some(t=>!(t in e)))),unknown:t.filter(e=>!X(e)).map(({key:e,value:t})=>({key:e,label:Z(e),price:Y(t)}))}),l=e.tiered_pricing,p=(0,j.h)(e.provider).displayName;return(0,x.jsxs)("section",{"aria-label":s("deploymentPricing.providerPricing",{provider:p}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,x.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,x.jsx)(w.f,{provider:e.provider}),p]}),o.groups.length>0&&(0,x.jsxs)(P.s,{children:[(0,x.jsx)(P.E,{children:(0,x.jsxs)(P.N,{className:"hover:bg-transparent",children:[(0,x.jsx)(P.V,{children:s("deploymentPricing.chargeColumn")}),o.tiers.map(e=>(0,x.jsx)(P.V,{className:P.J,children:s(N[e])},e))]})}),o.groups.map(e=>(0,x.jsx)(en,{group:e,tiers:o.tiers},e.threshold??"base"))]}),o.hasBlankCells&&(0,x.jsx)("p",{className:"text-xs text-muted-foreground",children:s("deploymentPricing.blankCells")}),l.length>0&&(0,x.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,x.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:s("deploymentPricing.tieredPricing")}),(0,x.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:l.map(e=>(0,x.jsx)("li",{children:(({range:[e,t],costs:i})=>{let a=ee(i),r=[...[...a.filter(X)].sort(ei).map(er),...a.filter(e=>!X(e)).map(({key:e,value:t})=>$.f.t("deploymentPricing.tierPrice",{price:Y(t),charge:Z(e).toLowerCase()}))];return $.f.t("deploymentPricing.tierRange",{low:ea(e),high:ea(t),prices:r.join(" / ")})})(e)},e.range.join("-")))})]}),o.unknown.length>0&&(0,x.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:o.unknown.map(({key:e,label:t,price:i})=>(0,x.jsxs)("div",{className:"contents",children:[(0,x.jsx)("dt",{className:"truncate",children:t}),(0,x.jsx)("dd",{className:"text-right tabular-nums",children:i})]},e))})]})}e.s(["f",0,function({deployments:e,heading:t,className:i}){let{t:a}=(0,v.f)();return e&&0!==e.length?(0,x.jsxs)("div",{className:(0,k.c)("flex flex-col gap-3",i),children:[t,e.length>1&&(0,x.jsx)("p",{className:"text-xs text-muted-foreground",children:a("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,t)=>(0,x.jsx)(es,{deployment:e},`${e.provider}-${t}`)),(0,x.jsx)("p",{className:"text-xs text-muted-foreground",children:a("deploymentPricing.requestAdjustments")})]}):null}],33355)},735450,e=>{var t=e.i(411382),i=e.i(772325),a=e.i(54105),r=e.i(240192),n=e.i(138620);let s="https://docs.litellm.ai/docs/decisions",o=`${s}#supported-providers`,l=`${s}#self-hosted-laya-and-nimble`,p=new Set(["/v1/decisions","/v1/systemone"]),d=t.z.object({litellm_provider:a.z.string(),mode:a.z.string().nullish(),supported_endpoints:i.z.array(a.z.string()).nullish()}),c="decisions",u=["evaluation",c],m=e=>null!=e&&u.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>p.has(e)),f=e=>{let t=r.z.record(a.z.string(),n.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=d.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},_={decisions:0,"self-hosted":1,"needs-upgrade":2},h=t.z.object({decision_model_providers:i.z.array(a.z.string())}),y=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),b=(e,t,i)=>({row:e,routeOrder:_[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,s,"d",0,c,"Z",0,o,"L",0,l,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),i=t.filter(([,e])=>g(e)),a=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:a}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(y).map(b).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?u:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=h.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let a=t(e);if(""===a)return!0;let r=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!r.some(e=>e.includes(a))||a.split(/\s+/).every(e=>r.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,a){return e.filter(e=>i(t,a(e)))},"h",0,i,"e",0,function(e,i,a){let r=t(i);if(""===r)return[...e];let n=e=>{let t=a(e).toLowerCase();return 1e3*(t===r)+100*!!t.startsWith(r)+(1e3-t.length)};return[...e].sort((e,t)=>n(t)-n(e))}])}])})();