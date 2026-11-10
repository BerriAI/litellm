(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,66330,e=>{var t=e.i(735450);e.s(["f",0,(e,i,s,a)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${i}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(a)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(s)},
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
        "model": ${JSON.stringify(s)},
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
    model=${JSON.stringify(s)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:s,accessToken:a,apiKey:r,inputMessage:n,chatHistory:l,selectedTags:o,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:p,selectedVoice:u,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:x}=e,_="session"===s?a:r,b=window.location.origin,y=f?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?b=y:f?.PROXY_BASE_URL&&(b=f.PROXY_BASE_URL);let j=n||"Your prompt here",v=j.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),N={};o.length>0&&(N.tags=o),d.length>0&&(N.vector_stores=d),c.length>0&&(N.guardrails=c),p.length>0&&(N.policies=p);let k=g||"your-model-name",P=x&&Object.keys(x).length>0?`,
	default_headers=${JSON.stringify(x,null,2).replace(/\n/g,"\n	")}`:"",$="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${b}",
	api_version="2024-02-01"${P}
)`:`import openai

client = openai.OpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	base_url="${b}"${P}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(N).length>0,t="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:j}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(s,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${k}",
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
`;break}case t.Q.RESPONSES:{let e=Object.keys(N).length>0,t="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:j}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(s,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${k}",
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
	model="${k}",
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
	model="${k}",
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
	model="${k}",
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
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${n||"Your text to convert to speech here"}",
	voice="${u}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${$}
${i}`}])},583408,e=>{var t=e.i(843476),i=e.i(555436),s=e.i(322831),a=e.i(519455),r=e.i(950594),n=e.i(967489);e.s(["f",0,function({search:e,tag:l,scan:o,tags:d,active:c,setSearch:p,setTag:u,setScan:m,clear:g}){let{t:h}=(0,s.f)(),f=[{value:"",label:h("skills.filters.allTags")},...[...new Set([...d,...l?[l]:[]])].map(e=>({value:e,label:e}))],x=[{value:"all",label:h("skills.filters.allScans")},{value:"yes",label:h("skills.filters.scanned")},{value:"no",label:h("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(r.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(r.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(r.M,{"aria-label":h("skills.filters.search"),placeholder:h("skills.filters.search"),value:e,onChange:e=>p(e.target.value)})]}),(0,t.jsxs)(n.d,{items:f,value:l,onValueChange:e=>u(e??""),children:[(0,t.jsx)(n.P,{className:"w-44","aria-label":h("skills.filters.tag"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:f.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(n.d,{items:x,value:o,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(n.P,{className:"w-48","aria-label":h("skills.table.columns.securityScan"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:x.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),c&&(0,t.jsx)(a.K,{variant:"ghost",size:"sm",onClick:g,children:h("skills.filters.clear")})]})}])},594107,e=>{var t=e.i(843476),i=e.i(494862);e.i(622826);var s=e.i(200208),a=e.i(112179),r=e.i(487486),n=e.i(201787);let l={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:n.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(n.f.language)??"—"})},{id:"keywords",header:n.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(r.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(e=>({id:e,accessorKey:e,size:190,enableSorting:!0,header:({column:s})=>(0,t.jsx)(i.v,{column:s,title:n.f.t(l[e])}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[e]??void 0,children:i.original[e]||n.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:s})=>(0,t.jsx)(i.v,{column:s,title:n.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(s.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:n.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:(0,t.jsx)(a.f,{tone:"neutral",label:n.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{var t=e.i(201787);let i=/^[0-9a-fA-F]{64}$/,s=/^https?:\/\//i,a="<your-litellm-key>";e.s(["i",0,a,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${a}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,s,a)=>{if("archive"!==t.source||!(e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e))||t.sha256&&!i.test(t.sha256))return null;let r=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${s.replace(/\/+$/,"")}${r}`,a);return/^https?:$/.test(e.protocol)&&t.url===e.href?r:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&s.test(e.url)?e.url:null,"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[]])},962600,e=>{var t=e.i(438847),i=e.i(45570),s=e.i(899426);let a=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],r={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:a,defaultSort:{id:"name",desc:!1},filterColumns:[]},n={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},l={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,a,"W",0,function(e,t){let i=(0,s.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),a=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),r="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&a&&r},"n",0,function(e){return{sort_by:a.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=a){let s=(0,i.f)({...r,sortFields:e}),[o,d]=(0,t.i)(n,l),c=e=>{d(e),s.onPaginationChange({...s.pagination,pageIndex:0})};return{...s,...o,setTag:e=>c({tag:e}),setScan:e=>c({scan:e}),clear:()=>{c(null),s.setSearch("")},active:!!(s.search.trim()||o.tag||"all"!==o.scan)}}])},402874,e=>{var t=e.i(843476),i=e.i(143488),s=e.i(912089),a=e.i(636772),r=e.i(283713),n=e.i(602869),l=e.i(782066),o=e.i(275144),d=e.i(886400),c=e.i(321836),p=e.i(487486),u=e.i(972518),m=e.i(799647),g=e.i(522016),h=e.i(322831),f=e.i(731565),x=e.i(492941);let _="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var b=e.i(519455),y=e.i(755146),j=e.i(664659),v=e.i(164668);let w=()=>{let{t:e}=(0,h.f)(),i=(0,f.f)(),{data:s,isLoading:a,isError:r,refetch:n}=(0,x.f)();return i?null:(0,t.jsxs)(y.G,{modal:!1,children:[(0,t.jsxs)(y.X,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(b.K,{variant:"ghost",className:`${_} border-0!`}),children:[e("shell.blogDropdown.trigger"),(0,t.jsx)(j.f,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(y.L,{align:"end",side:"bottom",className:"w-auto",children:a?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(v.f,{role:"img","aria-label":e("shell.blogDropdown.loading"),className:"size-4 animate-spin"})}):r?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:e("shell.blogDropdown.failed")}),(0,t.jsx)(b.K,{variant:"outline",size:"sm",onClick:()=>n(),children:e("shell.blogDropdown.retry")})]}):s&&0!==s.posts.length?(0,t.jsxs)(t.Fragment,{children:[s.posts.slice(0,5).map(e=>(0,t.jsx)(y.r,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(y.U,{}),(0,t.jsx)(y.r,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:e("shell.blogDropdown.viewAll")})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:e("shell.blogDropdown.empty")})})]})},N=()=>(0,t.jsx)(j.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0}),k=()=>{let{t:e}=(0,h.f)();return(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:_,children:[e("shell.docsLink.label"),(0,t.jsx)(N,{})]})};var P=e.i(747862),$=e.i(746798),I=e.i(196631),S=e.i(350682);let C=[{href:"https://www.litellm.ai/support",label:"shell.communityButtons.slackLabel",tooltip:"shell.communityButtons.slackTooltip",Icon:e.i(965519).f},{href:"https://github.com/BerriAI/litellm",label:"shell.communityButtons.githubLabel",tooltip:"shell.communityButtons.githubTooltip",Icon:S.f}],z=()=>{let{t:e}=(0,h.f)();return(0,a.f)()?null:(0,t.jsx)($.g,{children:(0,t.jsx)(P.f,{"aria-label":e("shell.communityButtons.label"),children:C.map(({href:i,label:s,tooltip:a,Icon:r})=>(0,t.jsxs)($.W,{children:[(0,t.jsx)($.y,{render:(0,t.jsx)("a",{href:i,target:"_blank",rel:"noopener noreferrer","aria-label":e(s),className:(0,I.c)((0,b.i)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(r,{})}),(0,t.jsx)($.Y,{children:e(a)})]},i))})})};var A=e.i(895335),O=e.i(135214),M=e.i(556260),L=e.i(319684),B=e.i(115571),D=e.i(222038),U=e.i(708347),E=e.i(344523),R=e.i(243553),T=e.i(465261),F=e.i(292270),q=e.i(263488),H=e.i(581418),J=e.i(284614),K=e.i(618566),Y=e.i(799676),V=e.i(337822),W=e.i(772436),Q=e.i(699375),G=e.i(922407),Z=e.i(271645);let X=({onLogout:e,variant:i="navbar",collapsed:r=!1})=>{let{t:n}=(0,h.f)(),{userId:o,userEmail:d,userRole:c,userRoleLabel:u,isViewOnly:m,premiumUser:g,loginMethod:x}=(0,O.f)(),_=(0,K.useRouter)(),[b,y]=(0,Z.useState)(!1),v=(0,a.f)(),w=(0,f.f)(),N=(0,s.f)(),[k,P]=(0,M.f)(o),S=o&&!m&&(0,U.u)(c),[C,z]=(0,Z.useState)(!1);(0,Z.useEffect)(()=>{z("true"===(0,B.t)("disableShowNewBadge"))},[]);let A=d||o||"user",X=function(e,t){let i=e?.split("@")[0]?.trim();if(i){let e=i.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(d,o),ee=function(e){let t=0;for(let i=0;i<e.length;i+=1)t=e.charCodeAt(i)+((t<<5)-t);return Math.abs(t)%360}(A),et=(0,D.f)(d,o);return(0,t.jsxs)(V.V,{open:b,onOpenChange:y,children:["sidebar"===i?(0,t.jsxs)(V.J,{render:(0,t.jsx)("button",{type:"button",className:(0,I.c)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",r?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":n("shell.userMenu.triggerAria",{role:u??n("shell.userMenu.unknownRole"),identity:d||o||n("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog",title:r?et:void 0}),children:[(0,t.jsx)(Y.E,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(Y.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:X})}),!r&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:et}),u&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:u})]}),(0,t.jsx)(E.f,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(V.J,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":n("shell.userMenu.triggerAria",{role:u??n("shell.userMenu.unknownRole"),identity:d||o||n("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog"}),children:[(0,t.jsx)(Y.E,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(Y.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:X})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:et}),(0,t.jsx)(j.f,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(V.T,{align:"sidebar"===i?"start":"end",side:"sidebar"===i?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(q.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:d||"-"})]}),g?(0,t.jsxs)(p.f,{children:[(0,t.jsx)(R.f,{className:"size-3"}),n("shell.userMenu.premium")]}):(0,t.jsx)($.g,{children:(0,t.jsxs)($.W,{children:[(0,t.jsxs)($.y,{render:(0,t.jsx)(p.f,{variant:"outline"}),children:[(0,t.jsx)(R.f,{className:"size-3"}),n("shell.userMenu.standard")]}),(0,t.jsx)($.Y,{side:"left",children:n("shell.userMenu.upgradeHint")})]})})]}),(0,t.jsx)(W.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(J.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.userId")})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:o||"-",children:o||"-"}),(0,t.jsx)(G.f,{value:o,label:n("shell.userMenu.copyUserId")})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(H.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.role")})]}),(0,t.jsx)("span",{children:u})]}),(0,t.jsx)(W.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideNewBadges")}),(0,t.jsx)(Q.f,{size:"sm",checked:C,onCheckedChange:e=>{z(e),e?(0,B.A)("disableShowNewBadge","true"):(0,B.p)("disableShowNewBadge"),(0,B.s)("disableShowNewBadge")},"aria-label":n("shell.userMenu.toggleHideNewBadges")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hidePrompts")}),(0,t.jsx)(Q.f,{size:"sm",checked:v,onCheckedChange:e=>{e?(0,B.A)("disableShowPrompts","true"):(0,B.p)("disableShowPrompts"),(0,B.s)("disableShowPrompts")},"aria-label":n("shell.userMenu.toggleHidePrompts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideBlogPosts")}),(0,t.jsx)(Q.f,{size:"sm",checked:w,onCheckedChange:e=>{e?(0,B.A)("disableBlogPosts","true"):(0,B.p)("disableBlogPosts"),(0,B.s)("disableBlogPosts")},"aria-label":n("shell.userMenu.toggleHideBlogPosts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideBouncingIcon")}),(0,t.jsx)(Q.f,{size:"sm",checked:N,onCheckedChange:e=>{e?(0,B.A)("disableBouncingIcon","true"):(0,B.p)("disableBouncingIcon"),(0,B.s)("disableBouncingIcon")},"aria-label":n("shell.userMenu.toggleHideBouncingIcon")})]}),!0===g&&S&&(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideAgent",{agent:L.f})}),(0,t.jsx)(Q.f,{size:"sm",checked:k,onCheckedChange:P,"aria-label":n("shell.userMenu.toggleHideAgent",{agent:L.f})})]})]}),(0,t.jsx)(W.f,{}),"username_password"===x&&(0,t.jsxs)("button",{type:"button",onClick:()=>{y(!1),_.push((0,l.B)("change-password"))},className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(T.f,{className:"size-4"}),n("shell.userMenu.changePassword")]}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(F.f,{className:"size-4"}),n("shell.userMenu.logout")]})]})]})};var ee=e.i(455880),et=e.i(853295),ei=e.i(383862);let es="h-auto max-h-full w-auto max-w-full object-contain";e.s(["f",0,({accessToken:e,isPublicPage:f=!1,sidebarCollapsed:x=!1,onToggleSidebar:_,deploymentHealth:b,deploymentHealthLink:y,publicNavigation:j})=>{let{t:v}=(0,h.f)(),N=(0,n._h)(),{logoUrl:P,logoUrlDark:$}=(0,o.i)(),{data:S}=(0,i.f)(e),C=S?.litellm_version,O=(0,s.f)(),M=(0,a.f)(),{isControlPlane:L,selectedWorker:B}=(0,r.f)(),D=L&&null!==B,U=P||`${N}/get_image`,E=$||P||`${N}/get_image?theme=dark`,R=(0,d.r)(e);return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:(0,I.c)("flex h-14 items-center px-4",f&&"gap-4 sm:px-6"),children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[_&&(0,t.jsx)("button",{onClick:_,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:v(x?"shell.navbar.expandSidebar":"shell.navbar.collapseSidebar"),children:(0,t.jsx)("span",{className:"text-lg",children:x?(0,t.jsx)(m.f,{className:"size-[18px]"}):(0,t.jsx)(u.f,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.default,{href:(0,l.B)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:(0,I.c)("flex h-10 max-w-48 items-center justify-center overflow-hidden",f&&"h-8 max-w-36 sm:h-10 sm:max-w-48"),children:[(0,t.jsx)("img",{src:U,alt:v("shell.navbar.logoAlt"),className:(0,I.c)(es,"dark:hidden")}),(0,t.jsx)("img",{src:E,alt:"","aria-hidden":!0,className:(0,I.c)(es,"hidden dark:block")})]})})}),C&&(0,t.jsxs)("div",{className:"relative",children:[!O&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:v("shell.navbar.thanks"),children:"🌑"}),(0,t.jsx)(p.f,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",C]})})]})]})]}),j,!f&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(et.f,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[D&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(ei.f,{onWorkerSwitch:t=>{(0,d.v)(e).finally(()=>{window.location.href=`${(0,c.D)()}?worker=${encodeURIComponent(t)}`})}})}),(0,t.jsxs)("nav",{"aria-label":v("shell.navbar.productDocs"),className:`flex min-w-0 items-center gap-2 ${D?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(k,{}),(0,t.jsx)(w,{})]}),!M&&(0,t.jsx)("div",{className:"hidden shrink-0 items-center border-l border-border pl-4 sm:flex",children:(0,t.jsx)(z,{})}),!f&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(ee.f,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),b,(0,t.jsx)(A.f,{accessToken:e,deploymentHealthLink:y}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(X,{onLogout:R})]})})]})]})})})}],402874)},466098,348594,831538,33355,e=>{var t=e.i(586448),i=e.i(602869),s=e.i(271645),a=e.i(198458),r=e.i(735450);let n="mode",l="providers",o="features",d=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],c=(e,t)=>{let i,s;return i=`filter[${e}][in]`,""===(s=d(t).join(","))?[]:[[i,s]]},p=e=>{switch(e.id){case n:return c(e.id,(0,r.q)(d(e.value)));case l:case o:return c(e.id,e.value);default:return[]}},u=e=>Object.fromEntries(e.flatMap(p)),m=(e,t)=>d(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let s=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?s:[...s,{id:t,value:i}]};e.s(["g",0,o,"e",0,n,"T",0,l,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,u,"R",0,g],348594);let h="/public/v1/model_hub",f=["publicModelHub","list"],x=[{id:"model_group",desc:!1}],_=async(e,t)=>{try{return await i.Mw.get(h,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,h,"t",0,e=>{let t=(0,a.f)({queryKey:f,fetchPage:_,serializeFilters:u,defaultSorting:x,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,r=(0,s.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),d=(0,s.useCallback)(e=>r(l,e),[r]),c=(0,s.useCallback)(e=>r(n,e),[r]),p=(0,s.useCallback)(e=>r(o,e),[r]);return{...t,providerValues:m(t.columnFilters,l),onProvidersChange:d,modeValues:m(t.columnFilters,n),onModesChange:c,featureValues:m(t.columnFilters,o),onFeaturesChange:p,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let b=["providers","modes","features"];e.s(["f",0,e=>{let[s,a,r]=(0,t.f)({queries:b.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${h}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:s,modes:a,features:r}}],466098);var y=e.i(843476),j=e.i(322831),v=e.i(418371),w=e.i(916925),N=e.i(196631),k=e.i(784774),P=e.i(201787);let $=["standard","flex","balanced","priority","ultrafast","batches"],I={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},S=["token","second","image","pixel","character","query","request"],C=["input","cache_read","explicit_cache_read","cache_write","output"],z=["","text","image","audio","video","reasoning"],A=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,O=/^(.+)_above_(\d+)k_tokens$/,M=/^(.+)_above_(\d+hr)$/,L=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,B=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,D=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},U={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},E={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},R={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},T=({charge:e,modality:t},i)=>t?P.f.t("deploymentPricing.withModality",{modality:P.f.t(E[i][t]),charge:P.f.t(U.inline[e])}):P.f.t(U[i][e]),F=(e,t)=>e.cacheDuration?P.f.t("deploymentPricing.withCacheDuration",{charge:T(e,t),duration:e.cacheDuration}):T(e,t),q=e=>P.f.t(R[e]),H=e=>P.f.t("deploymentPricing.rowLabel",{charge:F(e,"start"),unit:q(e.unit)}),J=e=>[C.indexOf(e.charge),S.indexOf(e.unit),z.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],K=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),Y=e=>0===e?P.f.t("deploymentPricing.free"):K.format(e),V=(e,t)=>Y("token"===t?1e6*e:e),W=/^(.*?)_?cost_per_(.+)$/,Q=/^(.+?)_cost(?:_(.+))?$/,G=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},Z=e=>{let t=W.exec(e);if(t)return P.f.t("deploymentPricing.rowLabel",{charge:t[1]?G(t[1]):P.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=Q.exec(e);return i?G([i[1],i[2]].filter(Boolean).join(" ")):G(e)},X=e=>"price"===e.cost.kind,ee=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=D(e,A),[s,a]=D(t,O),[r,n]=D(s,M),l=(e=>{let t=L.exec(e);if(t){let[,e,i="",s]=t;return S.includes(s)?{charge:e,modality:i,unit:s}:null}let i=B.exec(e);if(i){let[,e,t,s=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:s,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:s,unit:"token"}}return null})(r);return l?{kind:"price",...l,cacheDuration:n??"",threshold:null===a?null:1e3*Number(a),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),et=(e,t)=>e.filter((i,s)=>e.findIndex(e=>t(e)===t(i))===s),ei=(e,t)=>{let i,s;return i=J(e.cost),s=J(t.cost),i.reduce((e,t,i)=>e||t-s[i],0)},es=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),ea=({cost:e,value:t})=>{let i=F(e,"inline"),s=V(t,e.unit);return"token"===e.unit?P.f.t("deploymentPricing.tierPrice",{price:s,charge:i}):P.f.t("deploymentPricing.tierPricePerUnit",{price:s,charge:i,unit:q(e.unit)})};function er({group:e,tiers:t}){let{t:i}=(0,j.f)(),s=null===e.threshold?i("deploymentPricing.basePrices"):i("deploymentPricing.aboveThreshold",{threshold:es(e.threshold)});return(0,y.jsxs)(k.z,{"aria-label":s,children:[null!==e.threshold&&(0,y.jsx)(k.N,{className:"hover:bg-transparent",children:(0,y.jsx)(k.V,{colSpan:t.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:s})}),e.rows.map(e=>(0,y.jsxs)(k.N,{children:[(0,y.jsx)(k.V,{scope:"row",className:"font-normal",children:e.label}),t.map(t=>(0,y.jsx)(k.d,{className:k.J,children:e.prices[t]},t))]},e.label))]})}function en({deployment:e}){let t,i,s,a,r,{t:n}=(0,j.f)(),l=(s=[...et(i=(t=ee(e.costs)).filter(X),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),a=$.filter(e=>i.some(({cost:t})=>t.tier===e)),{tiers:a,groups:r=s.map(e=>{let t;return{threshold:e,rows:[...et(t=i.filter(({cost:t})=>t.threshold===e),({cost:e})=>H(e))].sort(ei).map(({cost:e})=>({label:H(e),prices:Object.fromEntries(t.filter(t=>H(t.cost)===H(e)).map(e=>[e.cost.tier,V(e.value,e.cost.unit)]))}))}}),hasBlankCells:r.some(({rows:e})=>e.some(({prices:e})=>a.some(t=>!(t in e)))),unknown:t.filter(e=>!X(e)).map(({key:e,value:t})=>({key:e,label:Z(e),price:Y(t)}))}),o=e.tiered_pricing,d=(0,w.h)(e.provider).displayName;return(0,y.jsxs)("section",{"aria-label":n("deploymentPricing.providerPricing",{provider:d}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,y.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,y.jsx)(v.f,{provider:e.provider}),d]}),l.groups.length>0&&(0,y.jsxs)(k.s,{children:[(0,y.jsx)(k.E,{children:(0,y.jsxs)(k.N,{className:"hover:bg-transparent",children:[(0,y.jsx)(k.V,{children:n("deploymentPricing.chargeColumn")}),l.tiers.map(e=>(0,y.jsx)(k.V,{className:k.J,children:n(I[e])},e))]})}),l.groups.map(e=>(0,y.jsx)(er,{group:e,tiers:l.tiers},e.threshold??"base"))]}),l.hasBlankCells&&(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.blankCells")}),o.length>0&&(0,y.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,y.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:n("deploymentPricing.tieredPricing")}),(0,y.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:o.map(e=>(0,y.jsx)("li",{children:(({range:[e,t],costs:i})=>{let s=ee(i),a=[...[...s.filter(X)].sort(ei).map(ea),...s.filter(e=>!X(e)).map(({key:e,value:t})=>P.f.t("deploymentPricing.tierPrice",{price:Y(t),charge:Z(e).toLowerCase()}))];return P.f.t("deploymentPricing.tierRange",{low:es(e),high:es(t),prices:a.join(" / ")})})(e)},e.range.join("-")))})]}),l.unknown.length>0&&(0,y.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:l.unknown.map(({key:e,label:t,price:i})=>(0,y.jsxs)("div",{className:"contents",children:[(0,y.jsx)("dt",{className:"truncate",children:t}),(0,y.jsx)("dd",{className:"text-right tabular-nums",children:i})]},e))})]})}e.s(["f",0,function({deployments:e,heading:t,className:i}){let{t:s}=(0,j.f)();return e&&0!==e.length?(0,y.jsxs)("div",{className:(0,N.c)("flex flex-col gap-3",i),children:[t,e.length>1&&(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:s("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,t)=>(0,y.jsx)(en,{deployment:e},`${e.provider}-${t}`)),(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:s("deploymentPricing.requestAdjustments")})]}):null}],33355)},747862,e=>{var t=e.i(843476);e.i(176782),e.i(911825);var i=e.i(225913),s=e.i(196631);e.i(772436);let a=(0,i.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:i,...r}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":i,className:(0,s.c)(a({orientation:i}),e),...r})}])},735450,e=>{var t=e.i(411382),i=e.i(772325),s=e.i(54105),a=e.i(240192),r=e.i(138620);let n="https://docs.litellm.ai/docs/decisions",l=`${n}#supported-providers`,o=`${n}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:s.z.string(),mode:s.z.string().nullish(),supported_endpoints:i.z.array(s.z.string()).nullish()}),p="decisions",u=["evaluation",p],m=e=>null!=e&&u.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),h=e=>{let t=a.z.record(s.z.string(),r.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=c.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},x=t.z.object({decision_model_providers:i.z.array(s.z.string())}),_=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),b=(e,t,i)=>({row:e,routeOrder:f[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,n,"d",0,p,"Z",0,l,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=h(e),i=t.filter(([,e])=>g(e)),s=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:s}},"W",0,(e,t)=>h(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(_).map(b).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?p:e??void 0,"q",0,e=>e.flatMap(e=>e===p?u:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=x.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let s=t(e);if(""===s)return!0;let a=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!a.some(e=>e.includes(s))||s.split(/\s+/).every(e=>a.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,s){return e.filter(e=>i(t,s(e)))},"h",0,i,"e",0,function(e,i,s){let a=t(i);if(""===a)return[...e];let r=e=>{let t=s(e).toLowerCase();return 1e3*(t===a)+100*!!t.startsWith(a)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}])})();