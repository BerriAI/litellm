(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,586448,e=>{"use strict";var t=e.i(271645),r=e.i(540143),s=e.i(869230),i=e.i(915823),a=e.i(619273);function n(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var o=class extends i.Subscribable{#e;#t;#r;#s;#i;#a;#n;#o;#l;#d=[];constructor(e,t,r){super(),this.#e=e,this.#s=r,this.#r=[],this.#i=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#i.forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#i.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#s=t,r.notifyManager.batch(()=>{let e=this.#i,t=this.#p(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),s=r.map(e=>e.getCurrentResult()),i=e.length!==r.length,o=r.some((t,r)=>t!==e[r]),l=i||o,d=!!l||s.some((e,t)=>{let r=this.#t[t];return!r||!(0,a.shallowEqualObjects)(e,r)});(l||d)&&(l&&(this.#d=t,this.#i=r),this.#t=s,this.hasListeners()&&(l&&(n(e,r).forEach(e=>{e.destroy()}),n(r,e).forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})),this.#c()))})}getCurrentResult(){return this.#t}getQueries(){return this.#i.map(e=>e.getCurrentQuery())}getObservers(){return this.#i}getOptimisticResult(e,t){let r=this.#p(e),s=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),i=r.map(e=>e.defaultedQueryOptions.queryHash);return[s,e=>this.#m(e??s,t,i),()=>this.#g(s,r)]}#g(e,t){return t.map((r,s)=>{let i=e[s];return r.defaultedQueryOptions.notifyOnChangeProps?i:r.observer.trackResult(i,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,r){if(t){let s=this.#l,i=void 0!==r&&void 0!==s&&(s.length!==r.length||r.some((e,t)=>e!==s[t]));return(!this.#a||this.#t!==this.#o||i||t!==this.#n)&&(this.#n=t,this.#o=this.#t,void 0!==r&&(this.#l=r),this.#a=(0,a.replaceEqualDeep)(this.#a,t(e))),this.#a}return e}#h(){return this.#s?.combine!==void 0&&this.#i.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#p(e){let t=new Map;this.#i.forEach(e=>{let r=e.options.queryHash;if(!r)return;let s=t.get(r);s?s.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let i=this.#e.defaultQueryOptions(e),a=t.get(i.queryHash)?.shift()??new s.QueryObserver(this.#e,i);r.push({defaultedQueryOptions:i,observer:a})}),r}#u(e,t){let r=this.#i.indexOf(e);if(-1!==r){var s;let e;this.#t=(s=this.#t,(e=s.slice(0))[r]=t,e),this.#c()}}#c(){if(this.hasListeners()){let e=this.#g(this.#t,this.#d),t=this.#h(),s=this.#a,i=t?s:this.#m(e,this.#s?.combine);(t||s!==i)&&r.notifyManager.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},l=e.i(912598),d=e.i(381384),u=e.i(673664),p=e.i(427001),c=e.i(254440);e.s(["useQueries",0,function({queries:e,...i},n){let m=(0,l.useQueryClient)(n),g=(0,d.useIsRestoring)(),h=(0,u.useQueryErrorResetBoundary)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,m,g]);f.forEach(e=>{(0,c.ensureSuspenseTimers)(e);let t=m.getQueryCache().get(e.queryHash);(0,p.ensurePreventErrorBoundaryRetry)(e,h,t)}),(0,p.useClearResetErrorBoundary)(h);let[b]=t.useState(()=>new o(m,f,i)),[x,_,y]=b.getOptimisticResult(f,i.combine),v=!g&&!1!==i.subscribed;t.useSyncExternalStore(t.useCallback(e=>v?b.subscribe(r.notifyManager.batchCalls(e)):a.noop,[b,v]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(f,i)},[f,i,b]);let j=x.some((e,t)=>(0,c.shouldSuspend)(f[t],e))?x.flatMap((e,t)=>{let r=f[t];if(r&&(0,c.shouldSuspend)(r,e)){let e=new s.QueryObserver(m,r);return(0,c.fetchOptimistic)(r,e,h)}return[]}):[];if(j.length>0)throw Promise.all(j);let w=x.find((e,t)=>{let r=f[t];return r&&(0,p.getHasError)({result:e,errorResetBoundary:h,throwOnError:r.throwOnError,query:m.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(w?.error)throw w.error;return _(y())}],586448)},233565,e=>{"use strict";var t=e.i(246349);e.s(["ChevronRightIcon",()=>t.default])},373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t])},541071,e=>{"use strict";var t=e.i(373488);e.s(["MoreHorizontal",()=>t.default])},546467,e=>{"use strict";let t=(0,e.i(475254).default)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["default",0,t])},778917,e=>{"use strict";var t=e.i(546467);e.s(["ExternalLink",()=>t.default])},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let r,{apiKeySource:s,accessToken:i,apiKey:a,inputMessage:n,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:p,selectedVoice:c,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,x="session"===s?i:a,_=window.location.origin,y=f?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?_=y:f?.PROXY_BASE_URL&&(_=f.PROXY_BASE_URL);let v=n||"Your prompt here",j=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};l.length>0&&(k.tags=l),d.length>0&&(k.vector_stores=d),u.length>0&&(k.guardrails=u),p.length>0&&(k.policies=p);let N=g||"your-model-name",$=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",C="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${$}
)`;switch(m){case t.EndpointType.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${N}",
    messages=${JSON.stringify(s,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.EndpointType.RESPONSES:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${N}",
    input=${JSON.stringify(s,null,4)}${t}
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.EndpointType.IMAGE:r="azure"===h?`
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
prompt = "${j}"

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
`;break;case t.EndpointType.IMAGE_EDITS:r="azure"===h?`
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
	model="${N}",
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
	model="${N}",
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
`;break;case t.EndpointType.EMBEDDINGS:r=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${N}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${N}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:r=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${N}",
	input="${n||"Your text to convert to speech here"}",
	voice="${c}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${r}`}])},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),s=e.i(871689),i=e.i(643531),a=e.i(174886),n=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,d=e=>e.trim().replace(/\/+$/,""),u=/\.(md|markdown|txt|json|ya?ml|toml)$/i,p=/\.zip$/i,c=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,g=/^[A-Za-z0-9-]+$/,h=/^[A-Za-z0-9._-]+$/,f=/^https?:\/\//i,b="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,_=e=>e.pathname.split("/").filter(e=>""!==e),y=e=>{try{return new URL(e)}catch{return null}},v=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),j=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},w=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),k=(e,t,r,s)=>{let i=d(s??"");return""!==i?l.test(i)?{parsed:{source:"git-subdir",url:t,path:i},label:`${e} subdir — ${t} @ ${i}`,suggestedName:w(j(i))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:w(r)}},N=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),$=e=>`/plugin install ${e.name}@litellm`,C=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",I=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&f.test(e.url)?e.url:null;e.s(["buildMarketplaceSettingsSnippet",0,N,"formatInstallCommand",0,$,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,C,"getSourceLink",0,I,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSha256",0,e=>""===e.trim()||c.test(e.trim()),"isValidSubPath",0,e=>{let t=d(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let r=((e,t)=>{let r=e.trim(),s=x.exec(r),i=s?`${b}${s[1]}@${s[2]}/${s[3]}`:r;if(!i.toLowerCase().startsWith(b))return null;let a=y(i);if(!a||""===a.username||""!==a.password||!v(a))return null;let n=i.indexOf("/",b.length);return -1===n||a.pathname!==i.slice(n)||_(a).length<2?null:k("SSH",r,j(a.pathname).replace(/\.git$/i,""),t)})(e,t);if(r)return r;let s=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=y(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&v(r)?r:null})(e);if(!s)return null;if(p.test(s.pathname))return{parsed:{source:"archive",url:s.href},label:`Zip archive — ${s.host}${s.pathname}`,suggestedName:w(j(s.pathname).replace(p,""))};if("github.com"===s.hostname.replace(/^www\./,""))return((e,t)=>{let r=_(e);if(r.length<2)return null;let s=r[0],i=r[1].replace(/\.git$/,"");if(!g.test(s)||!h.test(i))return null;let a=`${s}/${i}`,n=`https://github.com/${a}`,o={parsed:{source:"github",repo:a},label:`GitHub repo — ${a}`,suggestedName:w(i)};if(r.length>=4&&("tree"===r[2]||"blob"===r[2])){let e=r.slice(4),t=j(e.join("/")),s=u.test(t)?e.slice(0,-1):e;if(0===s.length)return o;let i=d(s.join("/"));return l.test(i)?{parsed:{source:"git-subdir",url:n,path:i},label:`GitHub subdir — ${a} @ ${i}`,suggestedName:w(j(i))}:null}if(2!==r.length)return null;let p=d(t??"");return""!==p?l.test(p)?{parsed:{source:"git-subdir",url:n,path:p},label:`GitHub subdir — ${a} @ ${p}`,suggestedName:w(j(p))}:null:o})(s,t);if(_(s).length<2)return null;let i=j(s.pathname).replace(/\.git$/,"");return k("Git",`${s.protocol}//${s.host}${s.pathname.replace(/\/+$/,"")}`,i,t)},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let S=({source:e})=>{let r=I(e),s=r&&"git-subdir"===e.source&&e.path?`${r}/tree/main/${e.path}`:r;return s?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:s,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[s.replace("https://",""),(0,t.jsx)(n.Link2,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:C(e)})]}):null};e.s(["default",0,({skill:e,onBack:n})=>{let[l,d]=(0,r.useState)("overview"),[u,p]=(0,r.useState)(null),c=(e,t)=>{navigator.clipboard.writeText(e),p(t),setTimeout(()=>p(null),2e3)},m=$(e),g=N(window.location.origin),h=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:n,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(s.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:h.map((e,r)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(S,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>c(m,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===u?"text-success":"text-info"),children:["install"===u?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;c(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===u?"text-success":"text-info"),children:["marketplace-cmd"===u?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"marketplace-cmd"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>c(g,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===u?"text-success":"text-info"),children:["settings"===u?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:g})]})]})]})}],652272)},755146,e=>{"use strict";var t=e.i(843476),r=e.i(451512),s=e.i(196631);e.i(233565);var i=e.i(678784);e.s(["DropdownMenu",0,function({...e}){return(0,t.jsx)(r.Menu.Root,{"data-slot":"dropdown-menu",...e})},"DropdownMenuCheckboxItem",0,function({className:e,children:a,checked:n,inset:o,...l}){return(0,t.jsxs)(r.Menu.CheckboxItem,{"data-slot":"dropdown-menu-checkbox-item","data-inset":o,className:(0,s.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),checked:n,...l,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-checkbox-item-indicator",children:(0,t.jsx)(r.Menu.CheckboxItemIndicator,{children:(0,t.jsx)(i.CheckIcon,{})})}),a]})},"DropdownMenuContent",0,function({align:e="start",alignOffset:i=0,side:a="bottom",sideOffset:n=4,className:o,...l}){return(0,t.jsx)(r.Menu.Portal,{children:(0,t.jsx)(r.Menu.Positioner,{className:"isolate z-popup outline-none",align:e,alignOffset:i,side:a,sideOffset:n,children:(0,t.jsx)(r.Menu.Popup,{"data-slot":"dropdown-menu-content",className:(0,s.cn)("z-popup max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-md bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95",o),...l})})})},"DropdownMenuItem",0,function({className:e,inset:i,variant:a="default",...n}){return(0,t.jsx)(r.Menu.Item,{"data-slot":"dropdown-menu-item","data-inset":i,"data-variant":a,className:(0,s.cn)("group/dropdown-menu-item relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground not-data-[variant=destructive]:focus:**:text-accent-foreground data-inset:pl-8 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[variant=destructive]:*:[svg]:text-destructive",e),...n})},"DropdownMenuRadioGroup",0,function({...e}){return(0,t.jsx)(r.Menu.RadioGroup,{"data-slot":"dropdown-menu-radio-group",...e})},"DropdownMenuRadioItem",0,function({className:e,children:a,inset:n,...o}){return(0,t.jsxs)(r.Menu.RadioItem,{"data-slot":"dropdown-menu-radio-item","data-inset":n,className:(0,s.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),...o,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-radio-item-indicator",children:(0,t.jsx)(r.Menu.RadioItemIndicator,{children:(0,t.jsx)(i.CheckIcon,{})})}),a]})},"DropdownMenuSeparator",0,function({className:e,...i}){return(0,t.jsx)(r.Menu.Separator,{"data-slot":"dropdown-menu-separator",className:(0,s.cn)("-mx-1 my-1 h-px bg-border",e),...i})},"DropdownMenuTrigger",0,function({...e}){return(0,t.jsx)(r.Menu.Trigger,{"data-slot":"dropdown-menu-trigger",...e})}])},283713,e=>{"use strict";var t=e.i(271645),r=e.i(602869),s=e.i(612256);let i="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,s.useUIConfig)(),a=e?.is_control_plane??!1,n=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(i));(0,t.useEffect)(()=>{if(!o||0===n.length)return;let e=n.find(e=>e.worker_id===o);e&&(0,r.switchToWorkerUrl)(e.url)},[o,n]);let d=n.find(e=>e.worker_id===o)??null,u=(0,t.useCallback)(e=>{let t=n.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(i,e),(0,r.switchToWorkerUrl)(t.url))},[n]);return{isControlPlane:a,workers:n,selectedWorkerId:o,selectedWorker:d,selectWorker:u,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(i),(0,r.switchToWorkerUrl)(null)},[])}}])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function r(e,r){let s=t(e);if(""===s)return!0;let i=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!i.some(e=>e.includes(s))||s.split(/\s+/).every(e=>i.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,s){return e.filter(e=>r(t,s(e)))},"matchesSearchTerm",0,r,"rankBySearchRelevance",0,function(e,r,s){let i=t(r);if(""===i)return[...e];let a=e=>{let t=s(e).toLowerCase();return 1e3*(t===i)+100*!!t.startsWith(i)+(1e3-t.length)};return[...e].sort((e,t)=>a(t)-a(e))}])}]);