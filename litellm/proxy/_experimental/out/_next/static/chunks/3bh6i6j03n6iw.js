(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,586448,e=>{var t=e.i(271645),s=e.i(540143),i=e.i(869230),r=e.i(915823),a=e.i(619273);function n(e,t){let s=new Set(t);return e.filter(e=>!s.has(e))}var l=class extends r.f{#e;#t;#s;#i;#r;#a;#n;#l;#o;#d=[];constructor(e,t,s){super(),this.#e=e,this.#i=s,this.#s=[],this.#r=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#r.forEach(e=>{e.subscribe(t=>{this.#c(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#r.forEach(e=>{e.destroy()})}setQueries(e,t){this.#s=e,this.#i=t,s.f.batch(()=>{let e=this.#r,t=this.#u(this.#s);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let s=t.map(e=>e.observer),i=s.map(e=>e.getCurrentResult()),r=e.length!==s.length,l=s.some((t,s)=>t!==e[s]),o=r||l,d=!!o||i.some((e,t)=>{let s=this.#t[t];return!s||!(0,a.U)(e,s)});(o||d)&&(o&&(this.#d=t,this.#r=s),this.#t=i,this.hasListeners()&&(o&&(n(e,s).forEach(e=>{e.destroy()}),n(s,e).forEach(e=>{e.subscribe(t=>{this.#c(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#r.map(e=>e.getCurrentQuery())}getObservers(){return this.#r}getOptimisticResult(e,t){let s=this.#u(e),i=s.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),r=s.map(e=>e.defaultedQueryOptions.queryHash);return[i,e=>this.#m(e??i,t,r),()=>this.#h(i,s)]}#h(e,t){return t.map((s,i)=>{let r=e[i];return s.defaultedQueryOptions.notifyOnChangeProps?r:s.observer.trackResult(r,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,s){if(t){let i=this.#o,r=void 0!==s&&void 0!==i&&(i.length!==s.length||s.some((e,t)=>e!==i[t]));return(!this.#a||this.#t!==this.#l||r||t!==this.#n)&&(this.#n=t,this.#l=this.#t,void 0!==s&&(this.#o=s),this.#a=(0,a.B)(this.#a,t(e))),this.#a}return e}#g(){return this.#i?.combine!==void 0&&this.#r.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#u(e){let t=new Map;this.#r.forEach(e=>{let s=e.options.queryHash;if(!s)return;let i=t.get(s);i?i.push(e):t.set(s,[e])});let s=[];return e.forEach(e=>{let r=this.#e.defaultQueryOptions(e),a=t.get(r.queryHash)?.shift()??new i.f(this.#e,r);s.push({defaultedQueryOptions:r,observer:a})}),s}#c(e,t){let s=this.#r.indexOf(e);if(-1!==s){var i;let e;this.#t=(i=this.#t,(e=i.slice(0))[s]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#h(this.#t,this.#d),t=this.#g(),i=this.#a,r=t?i:this.#m(e,this.#i?.combine);(t||i!==r)&&s.f.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},o=e.i(912598),d=e.i(381384),c=e.i(673664),u=e.i(427001),p=e.i(254440);e.s(["f",0,function({queries:e,...r},n){let m=(0,o.d)(n),h=(0,d.f)(),g=(0,c.f)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=h?"isRestoring":"optimistic",t}),[e,m,h]);f.forEach(e=>{(0,p.p)(e);let t=m.getQueryCache().get(e.queryHash);(0,u.l)(e,g,t)}),(0,u.E)(g);let[x]=t.useState(()=>new l(m,f,r)),[b,_,y]=x.getOptimisticResult(f,r.combine),j=!h&&!1!==r.subscribed;t.useSyncExternalStore(t.useCallback(e=>j?x.subscribe(s.f.batchCalls(e)):a.z,[x,j]),()=>x.getCurrentResult(),()=>x.getCurrentResult()),t.useEffect(()=>{x.setQueries(f,r)},[f,r,x]);let v=b.some((e,t)=>(0,p.Q)(f[t],e))?b.flatMap((e,t)=>{let s=f[t];if(s&&(0,p.Q)(s,e)){let e=new i.f(m,s);return(0,p.X)(s,e,g)}return[]}):[];if(v.length>0)throw Promise.all(v);let w=b.find((e,t)=>{let s=f[t];return s&&(0,u.P)({result:e,errorResetBoundary:g,throwOnError:s.throwOnError,query:m.getQueryCache().get(s.queryHash),suspense:s.suspense})});if(w?.error)throw w.error;return _(y())}],586448)},546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},198458,e=>{var t=e.i(655063),s=e.i(266027),i=e.i(271645),r=e.i(741466);e.s(["f",0,function(e){let{queryKey:a,fetchPage:n,serializeFilters:l,defaultSorting:o,defaultPageSize:d,enabled:c}=e,[u,p]=(0,i.useState)(o),[m,h]=(0,i.useState)({pageIndex:0,pageSize:d}),[g,f]=(0,i.useState)([]),[x,b]=(0,i.useState)(""),[_]=(0,t.f)(x,{wait:r.f}),y=(0,i.useMemo)(()=>{let e=u.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=_.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...l(g)}},[u,m.pageIndex,m.pageSize,_,g,l]),j={queryKey:[...a,y],queryFn:({signal:e})=>n(y,e),enabled:c,placeholderData:e=>e},{data:v,isLoading:w,isPlaceholderData:k,isFetching:N,error:P,refetch:$}=(0,s.f)(j),C=(0,i.useCallback)(()=>h(e=>({...e,pageIndex:0})),[]),S=(0,i.useCallback)(e=>{p(e),C()},[C]),I=(0,i.useCallback)(e=>{f(e),C()},[C]),z=(0,i.useCallback)(e=>{b(e),C()},[C]),O=(0,i.useCallback)(()=>{$()},[$]);return{rows:(0,i.useMemo)(()=>v?.data??[],[v]),rowCount:v?.meta.total_count??0,isLoading:w||k,isFetching:N,error:P,refetch:O,sorting:u,onSortingChange:S,pagination:m,onPaginationChange:h,columnFilters:g,onColumnFiltersChange:I,searchValue:x,onSearchChange:z}}])},66330,e=>{var t=e.i(735450);e.s(["f",0,(e,s,i,r)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${s}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(r)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(i)},
        "state": "Invoice #4471 from Acme Tools. Total $174.00, due in 30 days.",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
            "total": {"type": "number", "unit": "USD"}
        }
    }
)

print(response.json()["answers"])`;return(0,t._)(e)?`import requests

response = requests.post(
    ${JSON.stringify(`${s}/v1/systemone`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(r)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(i)},
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
    base_url=${JSON.stringify(s)}  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model=${JSON.stringify(i)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let s,{apiKeySource:i,accessToken:r,apiKey:a,inputMessage:n,chatHistory:l,selectedTags:o,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedVoice:p,endpointType:m,selectedModel:h,selectedSdk:g,proxySettings:f,customHeaders:x}=e,b="session"===i?r:a,_=window.location.origin,y=f?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?_=y:f?.PROXY_BASE_URL&&(_=f.PROXY_BASE_URL);let j=n||"Your prompt here",v=j.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};o.length>0&&(k.tags=o),d.length>0&&(k.vector_stores=d),c.length>0&&(k.guardrails=c),u.length>0&&(k.policies=u);let N=h||"your-model-name",P=x&&Object.keys(x).length>0?`,
	default_headers=${JSON.stringify(x,null,2).replace(/\n/g,"\n	")}`:"",$="azure"===g?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${P}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${P}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:j}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${N}",
    messages=${JSON.stringify(i,null,4)}${t}
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
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:j}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${N}",
    input=${JSON.stringify(i,null,4)}${t}
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
`;break}case t.Q.IMAGE:s="azure"===g?`
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
`;break;case t.Q.IMAGE_EDITS:s="azure"===g?`
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
`;break;case t.Q.EMBEDDINGS:s=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${N}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:s=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${N}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:s=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${N}",
	input="${n||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${N}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:s="\n# Code generation for this endpoint is not implemented yet."}return`${$}
${s}`}])},583408,e=>{var t=e.i(843476),s=e.i(555436),i=e.i(322831),r=e.i(519455),a=e.i(950594),n=e.i(967489);e.s(["f",0,function({search:e,tag:l,scan:o,tags:d,active:c,setSearch:u,setTag:p,setScan:m,clear:h}){let{t:g}=(0,i.f)(),f=[{value:"",label:g("skills.filters.allTags")},...[...new Set([...d,...l?[l]:[]])].map(e=>({value:e,label:e}))],x=[{value:"all",label:g("skills.filters.allScans")},{value:"yes",label:g("skills.filters.scanned")},{value:"no",label:g("skills.filters.notScanned")}];return(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsxs)(a.H,{className:"w-full sm:w-80",children:[(0,t.jsx)(a.B,{children:(0,t.jsx)(s.f,{className:"size-4"})}),(0,t.jsx)(a.M,{"aria-label":g("skills.filters.search"),placeholder:g("skills.filters.search"),value:e,onChange:e=>u(e.target.value)})]}),(0,t.jsxs)(n.d,{items:f,value:l,onValueChange:e=>p(e??""),children:[(0,t.jsx)(n.P,{className:"w-44","aria-label":g("skills.filters.tag"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:f.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)(n.d,{items:x,value:o,onValueChange:e=>m("yes"===e||"no"===e?e:"all"),children:[(0,t.jsx)(n.P,{className:"w-48","aria-label":g("skills.table.columns.securityScan"),children:(0,t.jsx)(n.U,{})}),(0,t.jsx)(n.r,{children:x.map(e=>(0,t.jsx)(n.T,{value:e.value,children:e.label},e.value))})]}),c&&(0,t.jsx)(r.K,{variant:"ghost",size:"sm",onClick:h,children:g("skills.filters.clear")})]})}])},594107,e=>{var t=e.i(843476),s=e.i(494862);e.i(622826);var i=e.i(200208),r=e.i(112179),a=e.i(487486),n=e.i(201787);let l={created_by:"skills.table.columns.createdBy",approved_by:"skills.table.columns.approvedBy"};e.s(["f",0,function(){return[{id:"download_count",accessorKey:"download_count",header:({column:e})=>(0,t.jsx)(s.v,{column:e,title:n.f.t("skills.table.columns.downloads")}),size:120,sortUndefined:"last",accessorFn:e=>e.download_count??void 0,cell:({row:e})=>(0,t.jsx)("span",{className:"tabular-nums",children:e.original.download_count?.toLocaleString(n.f.language)??"—"})},{id:"keywords",header:n.f.t("skills.table.columns.tags"),size:180,enableSorting:!1,cell:({row:e})=>(0,t.jsx)("div",{className:"flex max-w-52 flex-wrap gap-1",children:e.original.keywords?.length?e.original.keywords.map(e=>(0,t.jsx)(a.f,{variant:"secondary",children:e},e)):"—"})},...["created_by","approved_by"].map(e=>({id:e,accessorKey:e,size:190,enableSorting:!0,header:({column:i})=>(0,t.jsx)(s.v,{column:i,title:n.f.t(l[e])}),cell:({row:s})=>(0,t.jsx)("span",{className:"block max-w-48 truncate text-xs",title:s.original[e]??void 0,children:s.original[e]||n.f.t("skills.table.notRecorded")})})),...["created_at","updated_at"].map(e=>({id:e,accessorKey:e,size:140,enableSorting:!0,accessorFn:t=>t[e]?new Date(t[e]).getTime():void 0,sortUndefined:"last",header:({column:i})=>(0,t.jsx)(s.v,{column:i,title:n.f.t("created_at"===e?"skills.table.columns.createdAt":"skills.table.columns.updatedAt")}),cell:({row:s})=>(0,t.jsx)(i.S,{value:s.original[e],precision:"date"})})),{id:"security_scan_passed",accessorKey:"security_scan_passed",size:130,enableSorting:!0,header:({column:e})=>(0,t.jsx)(s.v,{column:e,title:(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:n.f.t("skills.table.columns.securityScan")})}),cell:({row:e})=>(0,t.jsx)("span",{title:n.f.t("skills.table.scanHelp"),children:(0,t.jsx)(r.f,{tone:"neutral",label:n.f.t(e.original.security_scan_passed?"skills.table.publicYes":"skills.table.publicNo")})})}]}])},209261,e=>{var t=e.i(201787);let s=/^[0-9a-fA-F]{64}$/,i=/^https?:\/\//i,r="<your-litellm-key>";e.s(["i",0,r,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${r}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"v",0,(e,t,i,r)=>{if("archive"!==t.source||!(e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e))||t.sha256&&!s.test(t.sha256))return null;let a=t.sha256?`/claude-code/plugins/${e}/archive/${t.sha256}.zip`:`/skill-hub/skills/${e}/archive`;try{let e=new URL(`${i.replace(/\/+$/,"")}${a}`,r);return/^https?:$/.test(e.protocol)&&t.url===e.href?a:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&i.test(e.url)?e.url:null,"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[]])},962600,e=>{var t=e.i(438847),s=e.i(45570),i=e.i(899426);let r=["name","download_count","created_at","updated_at","created_by","approved_by","security_scan_passed"],a={keyPrefix:"skills_",defaultPageSize:50,maxPageSize:100,sortFields:r,defaultSort:{id:"name",desc:!1},filterColumns:[]},n={tag:t.d.withDefault(""),scan:(0,t.b)(["all","yes","no"]).withDefault("all")},l={urlKeys:{tag:"skills_filter_tag",scan:"skills_filter_scan"}};e.s(["O",0,r,"W",0,function(e,t){let s=(0,i.h)(t.search,[e.name,e.description,e.domain,e.namespace,e.created_by,e.approved_by,e.published_by,...e.keywords??[]]),r=!t.tag||(e.keywords??[]).some(e=>e.toLowerCase()===t.tag.toLowerCase()),a="all"===t.scan||!!e.security_scan_passed==("yes"===t.scan);return s&&r&&a},"n",0,function(e){return{sort_by:r.find(t=>t===e[0]?.id)??"name",sort_order:e[0]?.desc?"desc":"asc"}},"f",0,function(e){return[...new Set(e.flatMap(e=>e.keywords??[]))].sort((e,t)=>e.localeCompare(t))},"a",0,function(e=r){let i=(0,s.f)({...a,sortFields:e}),[o,d]=(0,t.i)(n,l),c=e=>{d(e),i.onPaginationChange({...i.pagination,pageIndex:0})};return{...i,...o,setTag:e=>c({tag:e}),setScan:e=>c({scan:e}),clear:()=>{c(null),i.setSearch("")},active:!!(i.search.trim()||o.tag||"all"!==o.scan)}}])},402874,e=>{var t=e.i(843476),s=e.i(143488),i=e.i(912089),r=e.i(636772),a=e.i(283713),n=e.i(602869),l=e.i(782066),o=e.i(275144),d=e.i(886400),c=e.i(321836),u=e.i(487486),p=e.i(972518),m=e.i(799647),h=e.i(522016),g=e.i(322831),f=e.i(731565),x=e.i(492941);let b="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var _=e.i(519455),y=e.i(755146),j=e.i(664659),v=e.i(164668);let w=()=>{let{t:e}=(0,g.f)(),s=(0,f.f)(),{data:i,isLoading:r,isError:a,refetch:n}=(0,x.f)();return s?null:(0,t.jsxs)(y.G,{modal:!1,children:[(0,t.jsxs)(y.X,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(_.K,{variant:"ghost",className:`${b} border-0!`}),children:[e("shell.blogDropdown.trigger"),(0,t.jsx)(j.f,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(y.L,{align:"end",side:"bottom",className:"w-auto",children:r?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(v.f,{role:"img","aria-label":e("shell.blogDropdown.loading"),className:"size-4 animate-spin"})}):a?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:e("shell.blogDropdown.failed")}),(0,t.jsx)(_.K,{variant:"outline",size:"sm",onClick:()=>n(),children:e("shell.blogDropdown.retry")})]}):i&&0!==i.posts.length?(0,t.jsxs)(t.Fragment,{children:[i.posts.slice(0,5).map(e=>(0,t.jsx)(y.r,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(y.U,{}),(0,t.jsx)(y.r,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:e("shell.blogDropdown.viewAll")})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:e("shell.blogDropdown.empty")})})]})},k=()=>(0,t.jsx)(j.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0}),N=()=>{let{t:e}=(0,g.f)();return(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:b,children:[e("shell.docsLink.label"),(0,t.jsx)(k,{})]})};var P=e.i(747862),$=e.i(746798),C=e.i(196631),S=e.i(350682);let I=[{href:"https://www.litellm.ai/support",label:"shell.communityButtons.slackLabel",tooltip:"shell.communityButtons.slackTooltip",Icon:e.i(965519).f},{href:"https://github.com/BerriAI/litellm",label:"shell.communityButtons.githubLabel",tooltip:"shell.communityButtons.githubTooltip",Icon:S.f}],z=()=>{let{t:e}=(0,g.f)();return(0,r.f)()?null:(0,t.jsx)($.g,{children:(0,t.jsx)(P.f,{"aria-label":e("shell.communityButtons.label"),children:I.map(({href:s,label:i,tooltip:r,Icon:a})=>(0,t.jsxs)($.W,{children:[(0,t.jsx)($.y,{render:(0,t.jsx)("a",{href:s,target:"_blank",rel:"noopener noreferrer","aria-label":e(i),className:(0,C.c)((0,_.i)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(a,{})}),(0,t.jsx)($.Y,{children:e(r)})]},s))})})};var O=e.i(895335),M=e.i(135214),R=e.i(556260),A=e.i(319684),L=e.i(115571),E=e.i(222038),B=e.i(708347),q=e.i(344523),D=e.i(243553),U=e.i(465261),H=e.i(292270),T=e.i(263488),F=e.i(581418),Q=e.i(284614),J=e.i(618566),K=e.i(799676),Y=e.i(337822),V=e.i(772436),W=e.i(699375),G=e.i(922407),Z=e.i(271645);let X=({onLogout:e,variant:s="navbar",collapsed:a=!1})=>{let{t:n}=(0,g.f)(),{userId:o,userEmail:d,userRole:c,userRoleLabel:p,isViewOnly:m,premiumUser:h,loginMethod:x}=(0,M.f)(),b=(0,J.useRouter)(),[_,y]=(0,Z.useState)(!1),v=(0,r.f)(),w=(0,f.f)(),k=(0,i.f)(),[N,P]=(0,R.f)(o),S=o&&!m&&(0,B.u)(c),[I,z]=(0,Z.useState)(!1);(0,Z.useEffect)(()=>{z("true"===(0,L.t)("disableShowNewBadge"))},[]);let O=d||o||"user",X=function(e,t){let s=e?.split("@")[0]?.trim();if(s){let e=s.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(d,o),ee=function(e){let t=0;for(let s=0;s<e.length;s+=1)t=e.charCodeAt(s)+((t<<5)-t);return Math.abs(t)%360}(O),et=(0,E.f)(d,o);return(0,t.jsxs)(Y.V,{open:_,onOpenChange:y,children:["sidebar"===s?(0,t.jsxs)(Y.J,{render:(0,t.jsx)("button",{type:"button",className:(0,C.c)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",a?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":n("shell.userMenu.triggerAria",{role:p??n("shell.userMenu.unknownRole"),identity:d||o||n("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog",title:a?et:void 0}),children:[(0,t.jsx)(K.E,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(K.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:X})}),!a&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:et}),p&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:p})]}),(0,t.jsx)(q.f,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(Y.J,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":n("shell.userMenu.triggerAria",{role:p??n("shell.userMenu.unknownRole"),identity:d||o||n("shell.userMenu.unknownIdentity")}),"aria-haspopup":"dialog"}),children:[(0,t.jsx)(K.E,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(K.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${ee} 46% 38%)`},children:X})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:et}),(0,t.jsx)(j.f,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(Y.T,{align:"sidebar"===s?"start":"end",side:"sidebar"===s?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(T.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:d||"-"})]}),h?(0,t.jsxs)(u.f,{children:[(0,t.jsx)(D.f,{className:"size-3"}),n("shell.userMenu.premium")]}):(0,t.jsx)($.g,{children:(0,t.jsxs)($.W,{children:[(0,t.jsxs)($.y,{render:(0,t.jsx)(u.f,{variant:"outline"}),children:[(0,t.jsx)(D.f,{className:"size-3"}),n("shell.userMenu.standard")]}),(0,t.jsx)($.Y,{side:"left",children:n("shell.userMenu.upgradeHint")})]})})]}),(0,t.jsx)(V.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(Q.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.userId")})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:o||"-",children:o||"-"}),(0,t.jsx)(G.f,{value:o,label:n("shell.userMenu.copyUserId")})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(F.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.role")})]}),(0,t.jsx)("span",{children:p})]}),(0,t.jsx)(V.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideNewBadges")}),(0,t.jsx)(W.f,{size:"sm",checked:I,onCheckedChange:e=>{z(e),e?(0,L.A)("disableShowNewBadge","true"):(0,L.p)("disableShowNewBadge"),(0,L.s)("disableShowNewBadge")},"aria-label":n("shell.userMenu.toggleHideNewBadges")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hidePrompts")}),(0,t.jsx)(W.f,{size:"sm",checked:v,onCheckedChange:e=>{e?(0,L.A)("disableShowPrompts","true"):(0,L.p)("disableShowPrompts"),(0,L.s)("disableShowPrompts")},"aria-label":n("shell.userMenu.toggleHidePrompts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideBlogPosts")}),(0,t.jsx)(W.f,{size:"sm",checked:w,onCheckedChange:e=>{e?(0,L.A)("disableBlogPosts","true"):(0,L.p)("disableBlogPosts"),(0,L.s)("disableBlogPosts")},"aria-label":n("shell.userMenu.toggleHideBlogPosts")})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideBouncingIcon")}),(0,t.jsx)(W.f,{size:"sm",checked:k,onCheckedChange:e=>{e?(0,L.A)("disableBouncingIcon","true"):(0,L.p)("disableBouncingIcon"),(0,L.s)("disableBouncingIcon")},"aria-label":n("shell.userMenu.toggleHideBouncingIcon")})]}),!0===h&&S&&(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:n("shell.userMenu.hideAgent",{agent:A.f})}),(0,t.jsx)(W.f,{size:"sm",checked:N,onCheckedChange:P,"aria-label":n("shell.userMenu.toggleHideAgent",{agent:A.f})})]})]}),(0,t.jsx)(V.f,{}),"username_password"===x&&(0,t.jsxs)("button",{type:"button",onClick:()=>{y(!1),b.push((0,l.B)("change-password"))},className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(U.f,{className:"size-4"}),n("shell.userMenu.changePassword")]}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(H.f,{className:"size-4"}),n("shell.userMenu.logout")]})]})]})};var ee=e.i(455880),et=e.i(853295),es=e.i(383862);let ei="h-auto max-h-full w-auto max-w-full object-contain";e.s(["f",0,({accessToken:e,isPublicPage:f=!1,sidebarCollapsed:x=!1,onToggleSidebar:b,deploymentHealth:_,deploymentHealthLink:y,publicNavigation:j})=>{let{t:v}=(0,g.f)(),k=(0,n._h)(),{logoUrl:P,logoUrlDark:$}=(0,o.i)(),{data:S}=(0,s.f)(e),I=S?.litellm_version,M=(0,i.f)(),R=(0,r.f)(),{isControlPlane:A,selectedWorker:L}=(0,a.f)(),E=A&&null!==L,B=P||`${k}/get_image`,q=$||P||`${k}/get_image?theme=dark`,D=(0,d.r)(e);return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:(0,C.c)("flex h-14 items-center px-4",f&&"gap-4 sm:px-6"),children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[b&&(0,t.jsx)("button",{onClick:b,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:v(x?"shell.navbar.expandSidebar":"shell.navbar.collapseSidebar"),children:(0,t.jsx)("span",{className:"text-lg",children:x?(0,t.jsx)(m.f,{className:"size-[18px]"}):(0,t.jsx)(p.f,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(h.default,{href:(0,l.B)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:(0,C.c)("flex h-10 max-w-48 items-center justify-center overflow-hidden",f&&"h-8 max-w-36 sm:h-10 sm:max-w-48"),children:[(0,t.jsx)("img",{src:B,alt:v("shell.navbar.logoAlt"),className:(0,C.c)(ei,"dark:hidden")}),(0,t.jsx)("img",{src:q,alt:"","aria-hidden":!0,className:(0,C.c)(ei,"hidden dark:block")})]})})}),I&&(0,t.jsxs)("div",{className:"relative",children:[!M&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:v("shell.navbar.thanks"),children:"🌑"}),(0,t.jsx)(u.f,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",I]})})]})]})]}),j,!f&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(et.f,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[E&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(es.f,{onWorkerSwitch:t=>{(0,d.v)(e).finally(()=>{window.location.href=`${(0,c.D)()}?worker=${encodeURIComponent(t)}`})}})}),(0,t.jsxs)("nav",{"aria-label":v("shell.navbar.productDocs"),className:`flex min-w-0 items-center gap-2 ${E?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(N,{}),(0,t.jsx)(w,{})]}),!R&&(0,t.jsx)("div",{className:"hidden shrink-0 items-center border-l border-border pl-4 sm:flex",children:(0,t.jsx)(z,{})}),!f&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(ee.f,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),_,(0,t.jsx)(O.f,{accessToken:e,deploymentHealthLink:y}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(X,{onLogout:D})]})})]})]})})})}],402874)},466098,348594,831538,33355,e=>{var t=e.i(586448),s=e.i(602869),i=e.i(271645),r=e.i(198458),a=e.i(735450);let n="mode",l="providers",o="features",d=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],c=(e,t)=>{let s,i;return s=`filter[${e}][in]`,""===(i=d(t).join(","))?[]:[[s,i]]},u=e=>{switch(e.id){case n:return c(e.id,(0,a.q)(d(e.value)));case l:case o:return c(e.id,e.value);default:return[]}},p=e=>Object.fromEntries(e.flatMap(u)),m=(e,t)=>d(e.find(e=>e.id===t)?.value),h=(e,t,s)=>{let i=e.filter(e=>e.id!==t);return(Array.isArray(s)?0===s.length:""===s.trim())?i:[...i,{id:t,value:s}]};e.s(["g",0,o,"e",0,n,"T",0,l,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,p,"R",0,h],348594);let g="/public/v1/model_hub",f=["publicModelHub","list"],x=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await s.Mw.get(g,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,g,"t",0,e=>{let t=(0,r.f)({queryKey:f,fetchPage:b,serializeFilters:p,defaultSorting:x,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:s}=t,a=(0,i.useCallback)((e,t)=>s(s=>h(s,e,t)),[s]),d=(0,i.useCallback)(e=>a(l,e),[a]),c=(0,i.useCallback)(e=>a(n,e),[a]),u=(0,i.useCallback)(e=>a(o,e),[a]);return{...t,providerValues:m(t.columnFilters,l),onProvidersChange:d,modeValues:m(t.columnFilters,n),onModesChange:c,featureValues:m(t.columnFilters,o),onFeaturesChange:u,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let _=["providers","modes","features"];e.s(["f",0,e=>{let[i,r,a]=(0,t.f)({queries:_.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>s.Mw.get(`${g}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:i,modes:r,features:a}}],466098);var y=e.i(843476),j=e.i(322831),v=e.i(418371),w=e.i(916925),k=e.i(196631),N=e.i(784774),P=e.i(201787);let $=["standard","flex","balanced","priority","ultrafast","batches"],C={standard:"deploymentPricing.tiers.standard",flex:"deploymentPricing.tiers.flex",balanced:"deploymentPricing.tiers.balanced",priority:"deploymentPricing.tiers.priority",ultrafast:"deploymentPricing.tiers.ultrafast",batches:"deploymentPricing.tiers.batches"},S=["token","second","image","pixel","character","query","request"],I=["input","cache_read","explicit_cache_read","cache_write","output"],z=["","text","image","audio","video","reasoning"],O=/^(.+)_(flex|balanced|priority|ultrafast|batches)$/,M=/^(.+)_above_(\d+)k_tokens$/,R=/^(.+)_above_(\d+hr)$/,A=/^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/,L=/^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/,E=(e,t)=>{let s=t.exec(e);return s?[s[1],s[2]]:[e,null]},B={start:{input:"deploymentPricing.charges.input",cache_read:"deploymentPricing.charges.cacheRead",explicit_cache_read:"deploymentPricing.charges.explicitCacheRead",cache_write:"deploymentPricing.charges.cacheWrite",output:"deploymentPricing.charges.output"},inline:{input:"deploymentPricing.chargesInline.input",cache_read:"deploymentPricing.chargesInline.cacheRead",explicit_cache_read:"deploymentPricing.chargesInline.explicitCacheRead",cache_write:"deploymentPricing.chargesInline.cacheWrite",output:"deploymentPricing.chargesInline.output"}},q={start:{text:"deploymentPricing.modalities.text",image:"deploymentPricing.modalities.image",audio:"deploymentPricing.modalities.audio",video:"deploymentPricing.modalities.video",reasoning:"deploymentPricing.modalities.reasoning"},inline:{text:"deploymentPricing.modalitiesInline.text",image:"deploymentPricing.modalitiesInline.image",audio:"deploymentPricing.modalitiesInline.audio",video:"deploymentPricing.modalitiesInline.video",reasoning:"deploymentPricing.modalitiesInline.reasoning"}},D={token:"deploymentPricing.units.token",second:"deploymentPricing.units.second",image:"deploymentPricing.units.image",pixel:"deploymentPricing.units.pixel",character:"deploymentPricing.units.character",query:"deploymentPricing.units.query",request:"deploymentPricing.units.request"},U=({charge:e,modality:t},s)=>t?P.f.t("deploymentPricing.withModality",{modality:P.f.t(q[s][t]),charge:P.f.t(B.inline[e])}):P.f.t(B[s][e]),H=(e,t)=>e.cacheDuration?P.f.t("deploymentPricing.withCacheDuration",{charge:U(e,t),duration:e.cacheDuration}):U(e,t),T=e=>P.f.t(D[e]),F=e=>P.f.t("deploymentPricing.rowLabel",{charge:H(e,"start"),unit:T(e.unit)}),Q=e=>[I.indexOf(e.charge),S.indexOf(e.unit),z.indexOf(e.modality),parseInt(e.cacheDuration||"0",10)],J=new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",minimumFractionDigits:2,maximumFractionDigits:10,useGrouping:!1}),K=e=>0===e?P.f.t("deploymentPricing.free"):J.format(e),Y=(e,t)=>K("token"===t?1e6*e:e),V=/^(.*?)_?cost_per_(.+)$/,W=/^(.+?)_cost(?:_(.+))?$/,G=e=>{let t=e.replaceAll("_"," ").trim();return t.charAt(0).toUpperCase()+t.slice(1)},Z=e=>{let t=V.exec(e);if(t)return P.f.t("deploymentPricing.rowLabel",{charge:t[1]?G(t[1]):P.f.t("deploymentPricing.fallbackCost"),unit:t[2].replaceAll("_"," ")});let s=W.exec(e);return s?G([s[1],s[2]].filter(Boolean).join(" ")):G(e)},X=e=>"price"===e.cost.kind,ee=e=>Object.entries(e).map(([e,t])=>({key:e,value:t,cost:(e=>{let[t,s]=E(e,O),[i,r]=E(t,M),[a,n]=E(i,R),l=(e=>{let t=A.exec(e);if(t){let[,e,s="",i]=t;return S.includes(i)?{charge:e,modality:s,unit:i}:null}let s=L.exec(e);if(s){let[,e,t,i=""]=s;return"creation"===t?e?null:{charge:"cache_write",modality:i,unit:"token"}:{charge:e?"explicit_cache_read":"cache_read",modality:i,unit:"token"}}return null})(a);return l?{kind:"price",...l,cacheDuration:n??"",threshold:null===r?null:1e3*Number(r),tier:s??"standard"}:{kind:"unknown",key:e}})(e)})),et=(e,t)=>e.filter((s,i)=>e.findIndex(e=>t(e)===t(s))===i),es=(e,t)=>{let s,i;return s=Q(e.cost),i=Q(t.cost),s.reduce((e,t,s)=>e||t-i[s],0)},ei=e=>e>=1e6?`${+(e/1e6).toFixed(1)}M`:e>=1e3?`${+(e/1e3).toFixed(1)}K`:String(e),er=({cost:e,value:t})=>{let s=H(e,"inline"),i=Y(t,e.unit);return"token"===e.unit?P.f.t("deploymentPricing.tierPrice",{price:i,charge:s}):P.f.t("deploymentPricing.tierPricePerUnit",{price:i,charge:s,unit:T(e.unit)})};function ea({group:e,tiers:t}){let{t:s}=(0,j.f)(),i=null===e.threshold?s("deploymentPricing.basePrices"):s("deploymentPricing.aboveThreshold",{threshold:ei(e.threshold)});return(0,y.jsxs)(N.z,{"aria-label":i,children:[null!==e.threshold&&(0,y.jsx)(N.N,{className:"hover:bg-transparent",children:(0,y.jsx)(N.V,{colSpan:t.length+1,scope:"colgroup",className:"h-8 text-xs text-muted-foreground",children:i})}),e.rows.map(e=>(0,y.jsxs)(N.N,{children:[(0,y.jsx)(N.V,{scope:"row",className:"font-normal",children:e.label}),t.map(t=>(0,y.jsx)(N.d,{className:N.J,children:e.prices[t]},t))]},e.label))]})}function en({deployment:e}){let t,s,i,r,a,{t:n}=(0,j.f)(),l=(i=[...et(s=(t=ee(e.costs)).filter(X),({cost:e})=>e.threshold)].map(({cost:e})=>e.threshold).sort((e,t)=>(e??0)-(t??0)),r=$.filter(e=>s.some(({cost:t})=>t.tier===e)),{tiers:r,groups:a=i.map(e=>{let t;return{threshold:e,rows:[...et(t=s.filter(({cost:t})=>t.threshold===e),({cost:e})=>F(e))].sort(es).map(({cost:e})=>({label:F(e),prices:Object.fromEntries(t.filter(t=>F(t.cost)===F(e)).map(e=>[e.cost.tier,Y(e.value,e.cost.unit)]))}))}}),hasBlankCells:a.some(({rows:e})=>e.some(({prices:e})=>r.some(t=>!(t in e)))),unknown:t.filter(e=>!X(e)).map(({key:e,value:t})=>({key:e,label:Z(e),price:K(t)}))}),o=e.tiered_pricing,d=(0,w.h)(e.provider).displayName;return(0,y.jsxs)("section",{"aria-label":n("deploymentPricing.providerPricing",{provider:d}),className:"flex flex-col gap-2 rounded-lg border border-border p-3",children:[(0,y.jsxs)("h4",{className:"flex items-center gap-2 text-sm font-medium",children:[(0,y.jsx)(v.f,{provider:e.provider}),d]}),l.groups.length>0&&(0,y.jsxs)(N.s,{children:[(0,y.jsx)(N.E,{children:(0,y.jsxs)(N.N,{className:"hover:bg-transparent",children:[(0,y.jsx)(N.V,{children:n("deploymentPricing.chargeColumn")}),l.tiers.map(e=>(0,y.jsx)(N.V,{className:N.J,children:n(C[e])},e))]})}),l.groups.map(e=>(0,y.jsx)(ea,{group:e,tiers:l.tiers},e.threshold??"base"))]}),l.hasBlankCells&&(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:n("deploymentPricing.blankCells")}),o.length>0&&(0,y.jsxs)("div",{className:"flex flex-col gap-1 text-sm",children:[(0,y.jsx)("p",{className:"text-xs font-medium text-muted-foreground",children:n("deploymentPricing.tieredPricing")}),(0,y.jsx)("ul",{className:"flex flex-col gap-0.5 tabular-nums",children:o.map(e=>(0,y.jsx)("li",{children:(({range:[e,t],costs:s})=>{let i=ee(s),r=[...[...i.filter(X)].sort(es).map(er),...i.filter(e=>!X(e)).map(({key:e,value:t})=>P.f.t("deploymentPricing.tierPrice",{price:K(t),charge:Z(e).toLowerCase()}))];return P.f.t("deploymentPricing.tierRange",{low:ei(e),high:ei(t),prices:r.join(" / ")})})(e)},e.range.join("-")))})]}),l.unknown.length>0&&(0,y.jsx)("dl",{className:"grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-sm",children:l.unknown.map(({key:e,label:t,price:s})=>(0,y.jsxs)("div",{className:"contents",children:[(0,y.jsx)("dt",{className:"truncate",children:t}),(0,y.jsx)("dd",{className:"text-right tabular-nums",children:s})]},e))})]})}e.s(["f",0,function({deployments:e,heading:t,className:s}){let{t:i}=(0,j.f)();return e&&0!==e.length?(0,y.jsxs)("div",{className:(0,k.c)("flex flex-col gap-3",s),children:[t,e.length>1&&(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:i("deploymentPricing.deploymentCount",{count:e.length})}),e.map((e,t)=>(0,y.jsx)(en,{deployment:e},`${e.provider}-${t}`)),(0,y.jsx)("p",{className:"text-xs text-muted-foreground",children:i("deploymentPricing.requestAdjustments")})]}):null}],33355)},747862,e=>{var t=e.i(843476);e.i(176782),e.i(911825);var s=e.i(225913),i=e.i(196631);e.i(772436);let r=(0,s.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:s,...a}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":s,className:(0,i.c)(r({orientation:s}),e),...a})}])},735450,e=>{var t=e.i(411382),s=e.i(772325),i=e.i(54105),r=e.i(240192),a=e.i(138620);let n="https://docs.litellm.ai/docs/decisions",l=`${n}#supported-providers`,o=`${n}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:i.z.string(),mode:i.z.string().nullish(),supported_endpoints:s.z.array(i.z.string()).nullish()}),u="decisions",p=["evaluation",u],m=e=>null!=e&&p.includes(e),h=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),g=e=>{let t=r.z.record(i.z.string(),a.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let s=c.safeParse(t);return s.success?[[e,s.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},x=t.z.object({decision_model_providers:s.z.array(i.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),_=(e,t,s)=>({row:e,routeOrder:f[e.route],rankInProvider:s.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,n,"d",0,u,"Z",0,l,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=g(e),s=t.filter(([,e])=>h(e)),i=new Map([...new Set(s.map(([,e])=>e.litellm_provider))].map(e=>{let s=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:s.every(([,e])=>h(e))}]}));return{models:new Set(s.map(([e])=>e)),providers:i}},"W",0,(e,t)=>g(e).filter(([,e])=>h(e)).map(([e,s])=>({model:e,provider:s.litellm_provider,route:t?.has(s.litellm_provider)?"decisions":(s.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(_).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?u:e??void 0,"q",0,e=>e.flatMap(e=>e===u?p:[e]),"m",0,h,"_",0,m,"r",0,(e,t,s)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||s.some(t=>e.models.has(t)),"P",0,e=>{let t=x.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function s(e,s){let i=t(e);if(""===i)return!0;let r=s.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!r.some(e=>e.includes(i))||i.split(/\s+/).every(e=>r.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,i){return e.filter(e=>s(t,i(e)))},"h",0,s,"e",0,function(e,s,i){let r=t(s);if(""===r)return[...e];let a=e=>{let t=i(e).toLowerCase();return 1e3*(t===r)+100*!!t.startsWith(r)+(1e3-t.length)};return[...e].sort((e,t)=>a(t)-a(e))}])}])})();