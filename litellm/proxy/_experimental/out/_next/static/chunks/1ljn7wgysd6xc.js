(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,66330,e=>{"use strict";var t=e.i(735450);e.s(["f",0,(e,i,r,a)=>{if("extraction"===e)return`import requests

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

print(response.choices[0].message.content)`}])},909947,e=>{"use strict";var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:r,accessToken:a,apiKey:l,inputMessage:n,chatHistory:s,selectedTags:o,selectedVectorStores:p,selectedGuardrails:u,selectedPolicies:c,selectedVoice:d,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:_}=e,b="session"===r?a:l,y=window.location.origin,x=f?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?y=x:f?.PROXY_BASE_URL&&(y=f.PROXY_BASE_URL);let $=n||"Your prompt here",v=$.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};o.length>0&&(j.tags=o),p.length>0&&(j.vector_stores=p),u.length>0&&(j.guardrails=u),c.length>0&&(j.policies=c);let k=g||"your-model-name",P=_&&Object.keys(_).length>0?`,
	default_headers=${JSON.stringify(_,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${P}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${P}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=w.length>0?w:[{role:"user",content:$}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(r,null,4)}${t}
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
`;break}case t.Q.RESPONSES:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=w.length>0?w:[{role:"user",content:$}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(r,null,4)}${t}
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
	voice="${d}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${i}`}])},583408,e=>{"use strict";var t=e.i(843476),i=e.i(555436),r=e.i(322831),a=e.i(519455),l=e.i(950594),n=e.i(967489);e.s(["f",0,function({search:e,tag:s,scan:o,tags:p,active:u,setSearch:c,setTag:d,setScan:m,clear:g}){let{t:h}=(0,r.f)(),f=[{value:"",label:h("skills.filters.allTags")},...[...new Set([...p,...s?[s]:[]])].map(e=>({value:e,label:e}))],_=[{value:"all",label:h("skills.filters.allScans")},{value:"yes",label:h("skills.filters.scanned")},{value:"no",label:h("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(l.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(l.B,{children:(0,t.jsx)(i.f,{className:"size-4"})}),(0,t.jsx)(l.M,{"aria-label":h("skills.filters.search"),placeholder:h("skills.filters.search"),value:e,onChange:e=>c(e.target.value)})]}),(0,t.jsxs)(n.d,{items:f,value:s,onValueChange:e=>d(e??""),children:[(0,t.jsx)(n.P,{className:"w-44","aria-label":h("skills.filters.tag"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:f.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(n.d,{items:_,value:o,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(n.P,{className:"w-48","aria-label":h("skills.table.columns.securityScan"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:_.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),u&&(0,t.jsx)(a.K,{variant:"ghost",size:"sm",onClick:g,children:h("skills.filters.clear")})]})}])},594107,e=>{"use strict";var t=e.i(843476),i=e.i(494862);e.i(622826);var r=e.i(200208),a=e.i(112179),l=e.i(487486),n=e.i(201787),s=e.i(617885);let o={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(e={}){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:n.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(n.f.language)??"—"})},{id:"keywords",header:n.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(l.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(r=>({id:r,accessorKey:r,size:190,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.identitySortHelp"),children:n.f.t(o[r])})}),cell:({row:i})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:i.original[r]??void 0,children:(0,s.b)(i.original[r],e)||n.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:r})=>(0,t.jsx)(i.v,{column:r,title:n.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:i})=>(0,t.jsx)(r.S,{value:i.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(i.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:n.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:(0,t.jsx)(a.f,{tone:"neutral",label:n.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{"use strict";var t=e.i(201787);let i=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,r=e=>e.trim().replace(/\/+$/,""),a="github.com",l=/\.zip$/i,n=/^[0-9a-fA-F]{64}$/,s=/^\d{1,3}(\.\d{1,3}){3}$/,o=/^[A-Za-z0-9-]+$/,p=/^[A-Za-z0-9._-]+$/,u=/^https?:\/\//i,c="ssh://",d=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,m=e=>`${e.protocol}//${e.host}${e.pathname.replace(/\/+$/,"")}`,g=e=>e.pathname.split("/").filter(e=>""!==e),h=e=>{try{return new URL(e)}catch{return null}},f=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!s.test(e.hostname),_=e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=h(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&f(i)?i:null},b=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},y=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),x=(e,t)=>{let i=t?.trim();return e&&i?{...e,parsed:{...e.parsed,.../^[a-fA-F0-9]{40}$/.test(i)?{sha:i.toLowerCase()}:{ref:i}},label:`${e.label} (${i})`}:e},$=(e,t,i,a)=>{if(!e.length)return null;try{let l=i?.trim()||decodeURIComponent(e[0]),n=i?.trim()?e.findIndex((t,i)=>decodeURIComponent(e.slice(0,i+1).join("/"))===l)+1:1;if(!n||"blob"===t&&e.length===n)return null;let s=e.slice(n,"blob"===t?-1:void 0).map(decodeURIComponent).join("/"),o="blob"===t?decodeURIComponent(e[e.length-1]):void 0,p=!!(s||o);return{ref:l,path:p?s:r(a??""),pathFromUrl:p,fileFromUrl:o}}catch{return null}},v=(e,a,l,n="reference")=>{let s=g(e);if(s.length<2)return null;let u=s[0],c=s[1].replace(/\.git$/,"");if(!o.test(u)||!p.test(c))return null;let d=`${u}/${c}`,m=`https://github.com/${d}`,h={parsed:{source:"github",repo:d},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:d}),suggestedName:y(c)},f=s[2],_="tree"===f||"blob"===f?$(s.slice(3),f,l,a):void 0;if(null===_||void 0===_&&2!==s.length)return null;let v=_?.path??r(a??"");return"reference"===n&&v&&!i.test(v)?null:x({...v?{parsed:{source:"git-subdir",url:m,path:v},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:d,path:v}),suggestedName:y(b(v))}:h,..._?{pathFromUrl:_.pathFromUrl,fileFromUrl:_.fileFromUrl}:{}},_?.ref??l)},w=(e,a,l,n)=>{let s=r(n??"");return""!==s?i.test(s)?{parsed:{source:"git-subdir",url:a,path:s},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:a,path:s}),suggestedName:y(b(s))}:null:{parsed:{source:"url",url:a},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:a}),suggestedName:y(l)}},j=e=>{let t=_(e);if(!t)return null;let i=t.host.replace(/^www\./,"");return[a,"skills.sh"].includes(i)?h(`${t.protocol}//${i}${t.pathname}${t.search}${t.hash}`):null},k="<your-litellm-key>",P=e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e);e.s(["i",0,k,"x",0,e=>{let t=j(e);return t?.host==="skills.sh"?["import"]:t?["register","import"]:["register"]},"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${k}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,i,r)=>{if("archive"!==t.source||!P(e)||t.sha256&&!n.test(t.sha256))return null;let a=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${i.replace(/\/+$/,"")}${a}`,r);return/^https?:$/.test(e.protocol)&&t.url===e.href?a:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>{if(!["github","url","git-subdir","archive"].includes(e.source))return null;let t="github"===e.source?e.repo&&`https://github.com/${e.repo}`:e.url;if(!t||!u.test(t))return null;if("archive"===e.source)return t;let i=h(t),r=i&&i.hostname.replace(/^www\./,"");if(!i||r!==a&&"gitlab.com"!==r)return t;let l=g(i);if(!(r===a?2===l.length:l.length>=2&&!l.includes("-")))return t;let n=m(i).replace(/\.git$/,""),s=e.sha||e.ref;if(!s)return n;let o="git-subdir"===e.source&&e.path?`/${e.path.split("/").map(encodeURIComponent).join("/")}`:"";return`${n}/${r===a?"tree":"-/tree"}/${encodeURIComponent(s)}${o}`},"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||n.test(e.trim()),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"C",0,(e,t,i,r)=>{let a=j(e);if(!a||a.search||a.hash)return null;if("skills.sh"===a.host)try{let e=a.pathname.replace(/^\/+|\/+$/g,"").split("/").map(decodeURIComponent);if(3!==e.length||e.slice(0,2).some(e=>!p.test(e)))return null;let r=e[2];if(!r)return null;return{selection:{url:a.href,...i?.trim()?{ref:i.trim()}:{},...t?.trim()?{path:t.trim()}:{}},preview:null,skillFromUrl:r}}catch{return null}let l=v(a,t,i,"portable");if(!l||l.fileFromUrl&&"SKILL.md"!==l.fileFromUrl)return null;let n=l.parsed,s=n.sha??n.ref;return{selection:{url:"github"===n.source?`https://github.com/${n.repo}`:n.url??a.href,...r?.trim()?{skill:r.trim()}:{},...s?{ref:s}:{},...n.path||l.fileFromUrl?{path:n.path??""}:{}},preview:l}},"Z",0,(e,i,r)=>{let n=((e,t)=>{let i=e.trim(),r=d.exec(i),a=r?`${c}${r[1]}@${r[2]}/${r[3]}`:i;if(!a.toLowerCase().startsWith(c))return null;let l=h(a);if(!l||""===l.username||""!==l.password||!f(l))return null;let n=a.indexOf("/",c.length);return -1===n||l.pathname!==a.slice(n)||g(l).length<2?null:w("SSH",i,b(l.pathname).replace(/\.git$/i,""),t)})(e,i);if(n)return x(n,r);let s=_(e);if(!s)return null;let o=s.hostname.replace(/^www\./,"");if(o===a&&["tree","blob"].includes(g(s)[2]))return v(s,i,r);if("gitlab.com"===o&&/\/-\/(?:tree|blob)(?:\/|$)/.test(s.pathname))return((e,t,i)=>{let r=g(e),a=r.findIndex((e,t)=>"-"===e&&["tree","blob"].includes(r[t+1])),l=r[a+1];if(a<2||"tree"!==l&&"blob"!==l)return null;let n=$(r.slice(a+2),l,i,t);if(!n)return null;let s=w("Git",`${e.protocol}//${e.host.replace(/^www\./,"")}/${r.slice(0,a).join("/")}`,r[a-1].replace(/\.git$/,""),n.path);return x(s?{...s,pathFromUrl:n.pathFromUrl,fileFromUrl:n.fileFromUrl}:null,n.ref)})(s,i,r);if(l.test(s.pathname))return{parsed:{source:"archive",url:s.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${s.host}${s.pathname}`}),suggestedName:y(b(s.pathname).replace(l,""))};return o===a?v(s,i,r):x(((e,t)=>{if(g(e).length<2)return null;let i=b(e.pathname).replace(/\.git$/,"");return w("Git",m(e),i,t)})(s,i),r)},"c",0,(e,t)=>!!(e?.pathFromUrl&&e.parsed.path&&!(t.trim()||e.parsed.sha||e.parsed.ref?.includes("/"))),"e",0,P])},962600,e=>{"use strict";var t=e.i(438847),i=e.i(45570),r=e.i(899426);let a=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],l={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:a,defaultSort:{id:"name",desc:!1},filterColumns:[]},n={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},s={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,a,"W",0,function(e,t){let i=(0,r.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),a=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),l="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return i&&a&&l},"n",0,function(e){return{sort_by:a.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=a){let r=(0,i.f)({...l,sortFields:e}),[o,p]=(0,t.i)(n,s),u=e=>{p(e),r.onPaginationChange({...r.pagination,pageIndex:0})};return{...r,...o,setTag:e=>u({tag:e}),setScan:e=>u({scan:e}),clear:()=>{u(null),r.setSearch("")},active:!!(r.search.trim()||o.tag||"all"!==o.scan)}}])},33355,e=>{"use strict";var t=e.i(843476),i=e.i(322831),r=e.i(418371),a=e.i(916925),l=e.i(196631),n=e.i(784774),s=e.i(201787);let o=["standard","flex","balanced","priority","ultrafast","batches"],p={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},u=["token","second","image","pixel","character","query","request"],c=["input","cache_read","explicit_cache_read","cache_write","output"],d=["","text","image","audio","video","reasoning"],m=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,g=/^(.+)_above_(\d+)k_tokens$/,h=/^(.+)_above_(\d+hr)$/,f=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,_=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,b=(e,t)=>{let i=t.exec(e);return i?[i[1],i[2]]:[e,null]},y={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},x={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},$={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},v=({charge:e,modality:t},i)=>t?s.f.t("deploymentPricing.withModality",{modality:s.f.t(x[i][t]),charge:s.f.t(y.inline[e])}):s.f.t(y[i][e]),w=(e,t)=>e.cacheDuration?s.f.t("deploymentPricing.withCacheDuration",{charge:v(e,t),duration:e.cacheDuration}):v(e,t),j=e=>s.f.t($[e]),k=e=>s.f.t("deploymentPricing.rowLabel",{charge:w(e,"start"),unit:j(e.unit)}),P=e=>[c.indexOf(e.charge),u.indexOf(e.unit),d.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],S=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),I=e=>0===e?s.f.t("deploymentPricing.free"):S.format(e),M=(e,t)=>I("token"===t?1e6*e:e),N=/^(.*?)_?cost_per_(.+)$/,C=/^(.+?)_cost(?:_(.+))?$/,O=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},H=e=>{let t=N.exec(e);if(t)return s.f.t("deploymentPricing.rowLabel",{charge:t[1]?O(t[1]):s.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let i=C.exec(e);return i?O([i[1],i[2]].filter(Boolean).join(" ")):O(e)},z=e=>"price"===e.cost.kind,A=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,i]=b(e,m),[r,a]=b(t,g),[l,n]=b(r,h),s=(e=>{let t=f.exec(e);if(t){let[,e,i="",r]=t;return u.includes(r)?{charge:e,modality:i,unit:r}:null}let i=_.exec(e);if(i){let[,e,t,r=""]=i;return"creation"===t?e?null:{charge:"cache_write",modality:r,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:r,unit:"token"}}return null})(l);return s?{kind:"price",...s,cacheDuration:n??"",threshold:null===a?null:1e3*Number(a),tier:i??"standard"}:{kind:"unknown",key:e}})(e)})),L=(e,t)=>e.filter((i,r)=>e.findIndex(e=>t(e)===t(i))===r),U=(e,t)=>{let i,r;return i=P(e.cost),r=P(t.cost),i.reduce((e,t,i)=>e||t-r[i],0)},F=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),T=({cost:e,value:t})=>{let i=w(e,"inline"),r=M(t,e.unit);return"token"===e.unit?s.f.t("deploymentPricing.tierPrice",{price:r,charge:i}):s.f.t("deploymentPricing.tierPricePerUnit",{price:r,charge:i,unit:j(e.unit)})};function R({group:e,tiers:r}){let{t:a}=(0,i.f)(),l=null===e.threshold?a("deploymentPricing.basePrices"):a("deploymentPricing.aboveThreshold",{threshold:F(e.threshold)});return(0,t.jsxs)(n.z,{"aria-label":l,children:[null!==e.threshold&&(0,t.jsx)(n.N,{className:"hover:bg-transparent",children:(0,t.jsx)(n.V,{colSpan:r.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:l})}),e.rows.map(e=>(0,t.jsxs)(n.N,{children:[(0,t.jsx)(n.V,{scope:"row",className:"font-normal",children:e.label}),r.map(i=>(0,t.jsx)(n.d,{className:n.J,children:e.prices[i]},i))]},e.label))]})}function E({deployment:e}){let l,u,c,d,m,{t:g}=(0,i.f)(),h=(c=[...L(u=(l=A(e.costs)).filter(z),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),d=o.filter(e=>u.some(({cost:t})=>t.tier===e)),{tiers:d,groups:m=c.map(e=>{let t;return{threshold:e,rows:[...L(t=u.filter(({cost:t})=>t.threshold===e),({cost:e})=>k(e))].sort(U).map(({cost:e})=>({label:k(e),prices:Object.fromEntries(t.filter(t=>k(t.cost)===k(e)).map(e=>[e.cost.tier,M(e.value,e.cost.unit)]))}))}}),hasBlankCells:m.some(({rows:e})=>e.some(({prices:e})=>d.some(t=>!(t in e)))),unknown:l.filter(e=>!z(e)).map(({key:e,value:t})=>({key:e,label:H(e),price:I(t)}))}),f=e.tiered_pricing,_=(0,a.h)(e.provider).displayName;return(0,t.jsxs)("section",{"aria-label":g("deploymentPricing.providerPricing",{provider:_}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,t.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,t.jsx)(r.f,{provider:e.provider}),_]}),h.groups.length>0&&(0,t.jsxs)(n.s,{children:[(0,t.jsx)(n.E,{children:(0,t.jsxs)(n.N,{className:"hover:bg-transparent",children:[(0,t.jsx)(n.V,{children:g("deploymentPricing.chargeColumn")}),h.tiers.map(e=>(0,t.jsx)(n.V,{className:n.J,children:g(p[e])},e))]})}),h.groups.map(e=>(0,t.jsx)(R,{group:e,tiers:h.tiers},e.threshold??"base"))]}),h.hasBlankCells&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:g("deploymentPricing.blankCells")}),f.length>0&&(0,t.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,t.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:g("deploymentPricing.tieredPricing")}),(0,t.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:f.map(e=>(0,t.jsx)("li",{children:(({range:[e,t],costs:i})=>{let r=A(i),a=[...[...r.filter(z)].sort(U).map(T),...r.filter(e=>!z(e)).map(({key:e,value:t})=>s.f.t("deploymentPricing.tierPrice",{price:I(t),charge:H(e).toLowerCase()}))];return s.f.t("deploymentPricing.tierRange",{low:F(e),high:F(t),prices:a.join(" / ")})})(e)},e.range.join("-")))})]}),h.unknown.length>0&&(0,t.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:h.unknown.map(({key:e,label:i,price:r})=>(0,t.jsxs)("div",{className:"contents",children:[(0,t.jsx)("dt",{className:"truncate",children:i}),(0,t.jsx)("dd",{className:"text-right tabular-nums",children:r})]},e))})]})}e.s(["f",0,function({deployments:e,heading:r,className:a}){let{t:n}=(0,i.f)();return e&&0!==e.length?(0,t.jsxs)("div",{className:(0,l.c)("flex flex-col gap-3",a),children:[r,e.length>1&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,i)=>(0,t.jsx)(E,{deployment:e},`${e.provider}-${i}`)),(0,t.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.requestAdjustments")})]}):null}],33355)},913761,e=>{"use strict";var t=e.i(602869),i=e.i(916925),r=e.i(735450),a=e.i(201787);let l={chat:"publicModelHub.format.modes.chat",completion:"publicModelHub.format.modes.completion",embedding:"publicModelHub.format.modes.embedding",rerank:"publicModelHub.format.modes.rerank",image_generation:"publicModelHub.format.modes.imageGeneration",image_edit:"publicModelHub.format.modes.imageEdit",audio_speech:"publicModelHub.format.modes.audioSpeech",audio_transcription:"publicModelHub.format.modes.audioTranscription",video_generation:"publicModelHub.format.modes.videoGeneration",realtime:"publicModelHub.format.modes.realtime",responses:"publicModelHub.format.modes.responses",batch:"publicModelHub.format.modes.batch",ocr:"publicModelHub.format.modes.ocr",decisions:"publicModelHub.format.modes.decisions"},n=e=>e.replaceAll("_"," ").replace(/^./,e=>e.toUpperCase()),s=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:4,useGrouping:!1}),o={supports_vision:"publicModelHub.format.capabilities.vision",supports_function_calling:"publicModelHub.format.capabilities.functionCalling",supports_reasoning:"publicModelHub.format.capabilities.reasoning",supports_web_search:"publicModelHub.format.capabilities.webSearch",supports_parallel_function_calling:"publicModelHub.format.capabilities.parallelFunctionCalling",supports_url_context:"publicModelHub.format.capabilities.urlContext",supports_fast_mode:"publicModelHub.format.capabilities.fastMode"},p=Object.keys(o),u=e=>{let t=p.indexOf(e);return -1===t?p.length:t},c=e=>Object.hasOwn(o,e)?a.f.t(o[e]):n(e.replace(/^supports_/,"")),d={healthy:{tone:"success",label:"publicModelHub.format.health.healthy"},unhealthy:{tone:"error",label:"publicModelHub.format.health.unhealthy"}};e.s(["m",0,"text-[15px] leading-[22px] font-semibold","G",0,e=>c(`supports_${e}`),"a",0,e=>0===e?a.f.t("publicModelHub.format.free"):e?s.format(1e6*e):"-","k",0,e=>e?e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${Math.round(e/1e3)}K`:String(e):"-","M",0,()=>(0,t._h)(),"H",0,e=>{let t=e.health_status??"";return Object.hasOwn(d,t)?{tone:d[t].tone,label:a.f.t(d[t].label)}:{tone:"neutral",label:a.f.t("publicModelHub.format.notChecked")}},"S",0,e=>{let t=(0,r.i)(e);return t?Object.hasOwn(l,t)&&a.f.t(l[t])||n(t):"-"},"E",0,e=>Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e).sort((e,t)=>u(e)-u(t)).map(c),"q",0,e=>(0,i.h)(e).logo,"P",0,e=>(0,i.h)(e).displayName.replace(/\s*\(.*\)$/,""),"W",0,(e,t=Date.now())=>{let i=e?new Date(e).getTime():0/0;if(Number.isNaN(i))return"-";let r=Math.max(0,Math.round((t-i)/6e4));if(r<1)return a.f.t("publicModelHub.format.relativeTime.justNow");if(r<60)return a.f.t("publicModelHub.format.relativeTime.minutes",{minutes:r});let l=Math.round(r/60);if(l<24)return a.f.t("publicModelHub.format.relativeTime.hours",{hours:l});let n=Math.round(l/24);return a.f.t("publicModelHub.format.relativeTime.days",{count:n})},"Q",0,e=>Object.entries(e??{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:t})=>({title:e,url:t})),"X",0,e=>e.replaceAll("*","<model-name>")])},489432,e=>{"use strict";var t=e.i(66330),i=e.i(209261),r=e.i(909947),a=e.i(865361),l=e.i(602869),n=e.i(735450),s=e.i(913761);let o="Hello, how are you?",p=e=>{let t;if(!e.pass_through_path)return null;let i=(t=e.pass_through_methods,!t?.length||t.includes("POST")?"POST":t[0]),r=i.replace(/[\\"$`]/g,"\\$&"),a=`${(0,s.M)()}${e.pass_through_path}`.replace(/[\\"$`]/g,"\\$&");return"POST"===i?`curl -X "${r}" "${a}" \\
  -H "Authorization: Bearer your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "Hello, how are you?"}'`:`curl -X "${r}" "${a}" \\
  -H "Authorization: Bearer your_api_key"`};e.s(["c",0,e=>e.pass_through_path?p(e):e.mode&&"chat"!==e.mode?null:`curl ${(0,s.M)()}/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${(0,s.X)(e.model_group)}",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`,"j",0,()=>`/plugin marketplace add ${(0,s.M)()}/claude-code/marketplace.json`,"W",0,e=>{if("passthrough"===e.mode||e.pass_through_path)return p(e)??"";let i=(0,s.X)(e.model_group);if((0,n._)(e.mode))return(0,t.f)(e.mode,(0,s.M)(),i,(0,l.kR)());let u={apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:o,chatHistory:[{role:"user",content:o,isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,a.t)(e.mode||"chat"),selectedModel:i,selectedSdk:"openai",proxySettings:{PROXY_BASE_URL:(0,s.M)()}};return(0,r.f)(u)},"g",0,p,"z",0,e=>(0,i.a)(e)])},466098,348594,831538,e=>{"use strict";var t=e.i(586448),i=e.i(602869),r=e.i(271645),a=e.i(198458),l=e.i(735450);let n="mode",s="providers",o="features",p=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],u=(e,t)=>{let i,r;return i=`filter[${e}][in]`,""===(r=p(t).join(","))?[]:[[i,r]]},c=e=>{switch(e.id){case n:return u(e.id,(0,l.q)(p(e.value)));case s:case o:return u(e.id,e.value);default:return[]}},d=e=>Object.fromEntries(e.flatMap(c)),m=(e,t)=>p(e.find(e=>e.id===t)?.value),g=(e,t,i)=>{let r=e.filter(e=>e.id!==t);return(Array.isArray(i)?0===i.length:""===i.trim())?r:[...r,{id:t,value:i}]};e.s(["g",0,o,"e",0,n,"T",0,s,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,d,"R",0,g],348594);let h="/public/v1/model_hub",f=["publicModelHub","list"],_=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await i.Mw.get(h,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,h,"t",0,e=>{let t=(0,a.f)({queryKey:f,fetchPage:b,serializeFilters:d,defaultSorting:_,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:i}=t,l=(0,r.useCallback)((e,t)=>i(i=>g(i,e,t)),[i]),p=(0,r.useCallback)(e=>l(s,e),[l]),u=(0,r.useCallback)(e=>l(n,e),[l]),c=(0,r.useCallback)(e=>l(o,e),[l]);return{...t,providerValues:m(t.columnFilters,s),onProvidersChange:p,modeValues:m(t.columnFilters,n),onModesChange:u,featureValues:m(t.columnFilters,o),onFeaturesChange:c,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let y=["providers","modes","features"];e.s(["f",0,e=>{let[r,a,l]=(0,t.f)({queries:y.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>i.Mw.get(`${h}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:r,modes:a,features:l}}],466098)},735450,e=>{"use strict";var t=e.i(411382),i=e.i(772325),r=e.i(54105),a=e.i(240192),l=e.i(138620);let n="https://docs.litellm.ai/docs/decisions",s=`${n}#supported-providers`,o=`${n}#self-hosted-laya-and-nimble`,p=new Set(["/v1/decisions","/v1/systemone"]),u=t.z.object({litellm_provider:r.z.string(),mode:r.z.string().nullish(),supported_endpoints:i.z.array(r.z.string()).nullish()}),c="decisions",d=["evaluation",c],m=e=>null!=e&&d.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>p.has(e)),h=e=>{let t=a.z.record(r.z.string(),l.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=u.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},_=t.z.object({decision_model_providers:i.z.array(r.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),y=(e,t,i)=>({row:e,routeOrder:f[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,n,"d",0,c,"Z",0,s,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=h(e),i=t.filter(([,e])=>g(e)),r=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:r}},"W",0,(e,t)=>h(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(y).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?d:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=_.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function i(e,i){let r=t(e);if(""===r)return!0;let a=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!a.some(e=>e.includes(r))||r.split(/\s+/).every(e=>a.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,r){return e.filter(e=>i(t,r(e)))},"h",0,i,"e",0,function(e,i,r){let a=t(i);if(""===a)return[...e];let l=e=>{let t=r(e).toLowerCase();return 1e3*(t===a)+100*!!t.startsWith(a)+(1e3-t.length)};return[...e].sort((e,t)=>l(t)-l(e))}])}]);