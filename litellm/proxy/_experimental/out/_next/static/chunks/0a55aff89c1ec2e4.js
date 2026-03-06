(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,190272,785913,e=>{"use strict";var t,i,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i);let a={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=a[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:o,apiKey:a,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:p,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:_,selectedSdk:v,proxySettings:b}=e,x="session"===i?o:a,y=window.location.origin,w=b?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?y=w:b?.PROXY_BASE_URL&&(y=b.PROXY_BASE_URL);let S=n||"Your prompt here",j=S.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),k=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),c.length>0&&(C.vector_stores=c),d.length>0&&(C.guardrails=d),u.length>0&&(C.policies=u);let O=_||"your-model-name",N="azure"===v?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case r.CHAT:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${O}",
    messages=${JSON.stringify(o,null,4)}${i}
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
#     ]${i}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${O}",
    input=${JSON.stringify(o,null,4)}${i}
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===v?`
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
prompt = "${j}"

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
`;break;case r.IMAGE_EDITS:t="azure"===v?`
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
prompt = "${j}"

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
`;break;case r.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${O}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${O}",
	file=audio_file${n?`,
	prompt="${n.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${N}
${t}`}],190272)},516015,(e,t,i)=>{},898547,(e,t,i)=>{var o=e.i(247167);e.r(516015);var r=e.r(271645),a=r&&"object"==typeof r&&"default"in r?r:{default:r},n=void 0!==o.default&&o.default.env&&!0,s=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,i=t.name,o=void 0===i?"stylesheet":i,r=t.optimizeForSpeed,a=void 0===r?n:r;c(s(o),"`name` must be a string"),this._name=o,this._deletedRulePlaceholder="#"+o+"-deleted-rule____{}",c("boolean"==typeof a,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=a,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,i=e.prototype;return i.setOptimizeForSpeed=function(e){c("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),c(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},i.isOptimizeForSpeed=function(){return this._optimizeForSpeed},i.inject=function(){var e=this;if(c(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(n||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,i){return"number"==typeof i?e._serverSheet.cssRules[i]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),i},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},i.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},i.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},i.insertRule=function(e,t){if(c(s(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var i=this.getSheet();"number"!=typeof t&&(t=i.cssRules.length);try{i.insertRule(e,t)}catch(t){return n||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var o=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,o))}return this._rulesCount++},i.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var i="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!i.cssRules[e])return e;i.deleteRule(e);try{i.insertRule(t,e)}catch(o){n||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),i.insertRule(this._deletedRulePlaceholder,e)}}else{var o=this._tags[e];c(o,"old rule at index `"+e+"` not found"),o.textContent=t}return e},i.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];c(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},i.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},i.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,i){return i?t=t.concat(Array.prototype.map.call(e.getSheetForTag(i).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},i.makeStyleTag=function(e,t,i){t&&c(s(t),"makeStyleTag accepts only strings as second parameter");var o=document.createElement("style");this._nonce&&o.setAttribute("nonce",this._nonce),o.type="text/css",o.setAttribute("data-"+e,""),t&&o.appendChild(document.createTextNode(t));var r=document.head||document.getElementsByTagName("head")[0];return i?r.insertBefore(o,i):r.appendChild(o),o},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var i=0;i<t.length;i++){var o=t[i];o.enumerable=o.enumerable||!1,o.configurable=!0,"value"in o&&(o.writable=!0),Object.defineProperty(e,o.key,o)}}(e.prototype,t),e}();function c(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var d=function(e){for(var t=5381,i=e.length;i;)t=33*t^e.charCodeAt(--i);return t>>>0},u={};function p(e,t){if(!t)return"jsx-"+e;var i=String(t),o=e+i;return u[o]||(u[o]="jsx-"+d(e+"-"+i)),u[o]}function m(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var i=e+t;return u[i]||(u[i]=t.replace(/__jsx-style-dynamic-selector/g,e)),u[i]}var g=function(){function e(e){var t=void 0===e?{}:e,i=t.styleSheet,o=void 0===i?null:i,r=t.optimizeForSpeed,a=void 0!==r&&r;this._sheet=o||new l({name:"styled-jsx",optimizeForSpeed:a}),this._sheet.inject(),o&&"boolean"==typeof a&&(this._sheet.setOptimizeForSpeed(a),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var i=this.getIdAndRules(e),o=i.styleId,r=i.rules;if(o in this._instancesCounts){this._instancesCounts[o]+=1;return}var a=r.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[o]=a,this._instancesCounts[o]=1},t.remove=function(e){var t=this,i=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(i in this._instancesCounts,"styleId: `"+i+"` not found"),this._instancesCounts[i]-=1,this._instancesCounts[i]<1){var o=this._fromServer&&this._fromServer[i];o?(o.parentNode.removeChild(o),delete this._fromServer[i]):(this._indices[i].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[i]),delete this._instancesCounts[i]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],i=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return i[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,i;return t=this.cssRules(),void 0===(i=e)&&(i={}),t.map(function(e){var t=e[0],o=e[1];return a.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:i.nonce?i.nonce:void 0,dangerouslySetInnerHTML:{__html:o}})})},t.getIdAndRules=function(e){var t=e.children,i=e.dynamic,o=e.id;if(i){var r=p(o,i);return{styleId:r,rules:Array.isArray(t)?t.map(function(e){return m(r,e)}):[m(r,t)]}}return{styleId:p(o),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),f=r.createContext(null);function h(){return new g}function _(){return r.useContext(f)}f.displayName="StyleSheetContext";var v=a.default.useInsertionEffect||a.default.useLayoutEffect,b="u">typeof window?h():void 0;function x(e){var t=b||_();return t&&("u"<typeof window?t.add(e):v(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}x.dynamic=function(e){return e.map(function(e){return p(e[0],e[1])}).join(" ")},i.StyleRegistry=function(e){var t=e.registry,i=e.children,o=r.useContext(f),n=r.useState(function(){return o||t||h()})[0];return a.default.createElement(f.Provider,{value:n},i)},i.createStyleRegistry=h,i.style=x,i.useStyleRegistry=_},437902,(e,t,i)=>{t.exports=e.r(898547).style},132104,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M868 545.5L536.1 163a31.96 31.96 0 00-48.3 0L156 545.5a7.97 7.97 0 006 13.2h81c4.6 0 9-2 12.1-5.5L474 300.9V864c0 4.4 3.6 8 8 8h60c4.4 0 8-3.6 8-8V300.9l218.9 252.3c3 3.5 7.4 5.5 12.1 5.5h81c6.8 0 10.5-8 6-13.2z"}}]},name:"arrow-up",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["ArrowUpOutlined",0,a],132104)},447593,989022,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645),o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6a25.95 25.95 0 0025.6 30.4h723c1.5 0 3-.1 4.4-.4a25.88 25.88 0 0021.2-30zM204 390h272V182h72v208h272v104H204V390zm468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z"}}]},name:"clear",theme:"outlined"},r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["ClearOutlined",0,a],447593);var n=e.i(843476),s=e.i(592968),l=e.i(637235);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 394c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H400V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v236H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h228v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h164c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V394h164zM628 630H400V394h228v236z"}}]},name:"number",theme:"outlined"};var d=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:c}))});let u={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM653.3 424.6l52.2 52.2a8.01 8.01 0 01-4.7 13.6l-179.4 21c-5.1.6-9.5-3.7-8.9-8.9l21-179.4c.8-6.6 8.9-9.4 13.6-4.7l52.4 52.4 256.2-256.2c3.1-3.1 8.2-3.1 11.3 0l42.4 42.4c3.1 3.1 3.1 8.2 0 11.3L653.3 424.6z"}}]},name:"import",theme:"outlined"};var p=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:u}))}),m=e.i(872934),g=e.i(812618),f=e.i(366308),h=e.i(458505);e.s(["default",0,({timeToFirstToken:e,totalLatency:t,usage:i,toolName:o})=>e||t||i?(0,n.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3",children:[void 0!==e&&(0,n.jsx)(s.Tooltip,{title:"Time to first token",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["TTFT: ",(e/1e3).toFixed(2),"s"]})]})}),void 0!==t&&(0,n.jsx)(s.Tooltip,{title:"Total latency",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Total Latency: ",(t/1e3).toFixed(2),"s"]})]})}),i?.promptTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Prompt tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(p,{className:"mr-1"}),(0,n.jsxs)("span",{children:["In: ",i.promptTokens]})]})}),i?.completionTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Completion tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(m.ExportOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Out: ",i.completionTokens]})]})}),i?.reasoningTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Reasoning tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(g.BulbOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Reasoning: ",i.reasoningTokens]})]})}),i?.totalTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Total tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(d,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Total: ",i.totalTokens]})]})}),i?.cost!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Cost",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(h.DollarOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["$",i.cost.toFixed(6)]})]})}),o&&(0,n.jsx)(s.Tooltip,{title:"Tool used",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(f.ToolOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Tool: ",o]})]})})]}):null],989022)},254530,e=>{"use strict";var t=e.i(356449),i=e.i(764205);async function o(e,o,r,a,n,s,l,c,d,u,p,m,g,f,h,_,v,b,x,y,w,S,j,k){console.log=function(){},console.log("isLocal:",!1);let C=y||(0,i.getProxyBaseUrl)(),O={};n&&n.length>0&&(O["x-litellm-tags"]=n.join(","));let N=new t.default.OpenAI({apiKey:a,baseURL:C,dangerouslyAllowBrowser:!0,defaultHeaders:O});try{let t,i=Date.now(),a=!1,n={},y=!1,C=[];for await(let x of(f&&f.length>0&&(f.includes("__all__")?C.push({type:"mcp",server_label:"litellm",server_url:"litellm_proxy/mcp",require_approval:"never"}):f.forEach(e=>{let t=w?.find(t=>t.server_id===e),i=t?.alias||t?.server_name||e,o=S?.[e]||[];C.push({type:"mcp",server_label:"litellm",server_url:`litellm_proxy/mcp/${i}`,require_approval:"never",...o.length>0?{allowed_tools:o}:{}})})),await N.chat.completions.create({model:r,stream:!0,stream_options:{include_usage:!0},litellm_trace_id:u,messages:e,...p?{vector_store_ids:p}:{},...m?{guardrails:m}:{},...g?{policies:g}:{},...C.length>0?{tools:C,tool_choice:"auto"}:{},...void 0!==v?{temperature:v}:{},...void 0!==b?{max_tokens:b}:{},...k?{mock_testing_fallbacks:!0}:{}},{signal:s}))){console.log("Stream chunk:",x);let e=x.choices[0]?.delta;if(console.log("Delta content:",x.choices[0]?.delta?.content),console.log("Delta reasoning content:",e?.reasoning_content),!a&&(x.choices[0]?.delta?.content||e&&e.reasoning_content)&&(a=!0,t=Date.now()-i,console.log("First token received! Time:",t,"ms"),c?(console.log("Calling onTimingData with:",t),c(t)):console.log("onTimingData callback is not defined!")),x.choices[0]?.delta?.content){let e=x.choices[0].delta.content;o(e,x.model)}if(e&&e.image&&h&&(console.log("Image generated:",e.image),h(e.image.url,x.model)),e&&e.reasoning_content){let t=e.reasoning_content;l&&l(t)}if(e&&e.provider_specific_fields?.search_results&&_&&(console.log("Search results found:",e.provider_specific_fields.search_results),_(e.provider_specific_fields.search_results)),e&&e.provider_specific_fields){let t=e.provider_specific_fields;if(t.mcp_list_tools&&!n.mcp_list_tools&&(n.mcp_list_tools=t.mcp_list_tools,j&&!y)){y=!0;let e={type:"response.output_item.done",item_id:"mcp_list_tools",item:{type:"mcp_list_tools",tools:t.mcp_list_tools.map(e=>({name:e.function?.name||e.name||"",description:e.function?.description||e.description||"",input_schema:e.function?.parameters||e.input_schema||{}}))},timestamp:Date.now()};j(e),console.log("MCP list_tools event sent:",e)}t.mcp_tool_calls&&(n.mcp_tool_calls=t.mcp_tool_calls),t.mcp_call_results&&(n.mcp_call_results=t.mcp_call_results),(t.mcp_list_tools||t.mcp_tool_calls||t.mcp_call_results)&&console.log("MCP metadata found in chunk:",{mcp_list_tools:t.mcp_list_tools?"present":"absent",mcp_tool_calls:t.mcp_tool_calls?"present":"absent",mcp_call_results:t.mcp_call_results?"present":"absent"})}if(x.usage&&d){console.log("Usage data found:",x.usage);let e={completionTokens:x.usage.completion_tokens,promptTokens:x.usage.prompt_tokens,totalTokens:x.usage.total_tokens};x.usage.completion_tokens_details?.reasoning_tokens&&(e.reasoningTokens=x.usage.completion_tokens_details.reasoning_tokens),void 0!==x.usage.cost&&null!==x.usage.cost&&(e.cost=parseFloat(x.usage.cost)),d(e)}}j&&(n.mcp_tool_calls||n.mcp_call_results)&&n.mcp_tool_calls&&n.mcp_tool_calls.length>0&&n.mcp_tool_calls.forEach((e,t)=>{let i=e.function?.name||e.name||"",o=e.function?.arguments||e.arguments||"{}",r=n.mcp_call_results?.find(t=>t.tool_call_id===e.id||t.tool_call_id===e.call_id)||n.mcp_call_results?.[t],a={type:"response.output_item.done",item:{type:"mcp_call",name:i,arguments:"string"==typeof o?o:JSON.stringify(o),output:r?.result?"string"==typeof r.result?r.result:JSON.stringify(r.result):void 0},item_id:e.id||e.call_id,timestamp:Date.now()};j(a),console.log("MCP call event sent:",a)});let O=Date.now();x&&x(O-i)}catch(e){throw s?.aborted&&console.log("Chat completion request was cancelled"),e}}e.s(["makeOpenAIChatCompletionRequest",()=>o])},966988,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(464571),r=e.i(918789),a=e.i(650056),n=e.i(219470),s=e.i(755151),l=e.i(240647),c=e.i(812618);e.s(["default",0,({reasoningContent:e})=>{let[d,u]=(0,i.useState)(!0);return e?(0,t.jsxs)("div",{className:"reasoning-content mt-1 mb-2",children:[(0,t.jsxs)(o.Button,{type:"text",className:"flex items-center text-xs text-gray-500 hover:text-gray-700",onClick:()=>u(!d),icon:(0,t.jsx)(c.BulbOutlined,{}),children:[d?"Hide reasoning":"Show reasoning",d?(0,t.jsx)(s.DownOutlined,{className:"ml-1"}):(0,t.jsx)(l.RightOutlined,{className:"ml-1"})]}),d&&(0,t.jsx)("div",{className:"mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md text-sm text-gray-700",children:(0,t.jsx)(r.default,{components:{code({node:e,inline:i,className:o,children:r,...s}){let l=/language-(\w+)/.exec(o||"");return!i&&l?(0,t.jsx)(a.Prism,{style:n.coy,language:l[1],PreTag:"div",className:"rounded-md my-2",...s,children:String(r).replace(/\n$/,"")}):(0,t.jsx)("code",{className:`${o} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`,...s,children:r})}},children:e})})]}):null}])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),o=e.i(914949),r=e.i(404948);let a=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,a],836938);var n=e.i(613541),s=e.i(763731),l=e.i(242064),c=e.i(491816);e.i(793154);var d=e.i(880476),u=e.i(183293),p=e.i(717356),m=e.i(320560),g=e.i(307358),f=e.i(246422),h=e.i(838378),_=e.i(617933);let v=(0,f.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:i}=e,o=(0,h.mergeToken)(e,{popoverBg:t,popoverColor:i});return[(e=>{let{componentCls:t,popoverColor:i,titleMinWidth:o,fontWeightStrong:r,innerPadding:a,boxShadowSecondary:n,colorTextHeading:s,borderRadiusLG:l,zIndexPopup:c,titleMarginBottom:d,colorBgElevated:p,popoverBg:g,titleBorderBottom:f,innerContentPadding:h,titlePadding:_}=e;return[{[t]:Object.assign(Object.assign({},(0,u.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:c,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":p,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:g,backgroundClip:"padding-box",borderRadius:l,boxShadow:n,padding:a},[`${t}-title`]:{minWidth:o,marginBottom:d,color:s,fontWeight:r,borderBottom:f,padding:_},[`${t}-inner-content`]:{color:i,padding:h}})},(0,m.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(o),(e=>{let{componentCls:t}=e;return{[t]:_.PresetColors.map(i=>{let o=e[`${i}6`];return{[`&${t}-${i}`]:{"--antd-arrow-background-color":o,[`${t}-inner`]:{backgroundColor:o},[`${t}-arrow`]:{background:"transparent"}}}})}})(o),(0,p.initZoomMotion)(o,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:i,fontHeight:o,padding:r,wireframe:a,zIndexPopupBase:n,borderRadiusLG:s,marginXS:l,lineType:c,colorSplit:d,paddingSM:u}=e,p=i-o;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:n+30},(0,g.getArrowToken)(e)),(0,m.getArrowOffsetToken)({contentRadius:s,limitVerticalRadius:!0})),{innerPadding:12*!a,titleMarginBottom:a?0:l,titlePadding:a?`${p/2}px ${r}px ${p/2-t}px`:0,titleBorderBottom:a?`${t}px ${c} ${d}`:"none",innerContentPadding:a?`${u}px ${r}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var b=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,o=Object.getOwnPropertySymbols(e);r<o.length;r++)0>t.indexOf(o[r])&&Object.prototype.propertyIsEnumerable.call(e,o[r])&&(i[o[r]]=e[o[r]]);return i};let x=({title:e,content:i,prefixCls:o})=>e||i?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${o}-title`},e),i&&t.createElement("div",{className:`${o}-inner-content`},i)):null,y=e=>{let{hashId:o,prefixCls:r,className:n,style:s,placement:l="top",title:c,content:u,children:p}=e,m=a(c),g=a(u),f=(0,i.default)(o,r,`${r}-pure`,`${r}-placement-${l}`,n);return t.createElement("div",{className:f,style:s},t.createElement("div",{className:`${r}-arrow`}),t.createElement(d.Popup,Object.assign({},e,{className:o,prefixCls:r}),p||t.createElement(x,{prefixCls:r,title:m,content:g})))},w=e=>{let{prefixCls:o,className:r}=e,a=b(e,["prefixCls","className"]),{getPrefixCls:n}=t.useContext(l.ConfigContext),s=n("popover",o),[c,d,u]=v(s);return c(t.createElement(y,Object.assign({},a,{prefixCls:s,hashId:d,className:(0,i.default)(r,u)})))};e.s(["Overlay",0,x,"default",0,w],310730);var S=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,o=Object.getOwnPropertySymbols(e);r<o.length;r++)0>t.indexOf(o[r])&&Object.prototype.propertyIsEnumerable.call(e,o[r])&&(i[o[r]]=e[o[r]]);return i};let j=t.forwardRef((e,d)=>{var u,p;let{prefixCls:m,title:g,content:f,overlayClassName:h,placement:_="top",trigger:b="hover",children:y,mouseEnterDelay:w=.1,mouseLeaveDelay:j=.1,onOpenChange:k,overlayStyle:C={},styles:O,classNames:N}=e,z=S(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:E,className:T,style:R,classNames:I,styles:M}=(0,l.useComponentConfig)("popover"),A=E("popover",m),[$,L,P]=v(A),H=E(),B=(0,i.default)(h,L,P,T,I.root,null==N?void 0:N.root),F=(0,i.default)(I.body,null==N?void 0:N.body),[V,D]=(0,o.default)(!1,{value:null!=(u=e.open)?u:e.visible,defaultValue:null!=(p=e.defaultOpen)?p:e.defaultVisible}),W=(e,t)=>{D(e,!0),null==k||k(e,t)},U=a(g),q=a(f);return $(t.createElement(c.default,Object.assign({placement:_,trigger:b,mouseEnterDelay:w,mouseLeaveDelay:j},z,{prefixCls:A,classNames:{root:B,body:F},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},M.root),R),C),null==O?void 0:O.root),body:Object.assign(Object.assign({},M.body),null==O?void 0:O.body)},ref:d,open:V,onOpenChange:e=>{W(e)},overlay:U||q?t.createElement(x,{prefixCls:A,title:U,content:q}):null,transitionName:(0,n.getTransitionName)(H,"zoom-big",z.transitionName),"data-popover-inject":!0}),(0,s.cloneElement)(y,{onKeyDown:e=>{var i,o;(0,t.isValidElement)(y)&&(null==(o=null==y?void 0:(i=y.props).onKeyDown)||o.call(i,e)),e.keyCode===r.default.ESC&&W(!1,e)}})))});j._InternalPanelDoNotUseOrYouWillBeFired=w,e.s(["default",0,j],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},219470,812618,e=>{"use strict";e.s(["coy",0,{'code[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",maxHeight:"inherit",height:"inherit",padding:"0 1em",display:"block",overflow:"auto"},'pre[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",position:"relative",margin:".5em 0",overflow:"visible",padding:"1px",backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em"},'pre[class*="language-"] > code':{position:"relative",zIndex:"1",borderLeft:"10px solid #358ccb",boxShadow:"-1px 0px 0px 0px #358ccb, 0px 0px 0px 1px #dfdfdf",backgroundColor:"#fdfdfd",backgroundImage:"linear-gradient(transparent 50%, rgba(69, 142, 209, 0.04) 50%)",backgroundSize:"3em 3em",backgroundOrigin:"content-box",backgroundAttachment:"local"},':not(pre) > code[class*="language-"]':{backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em",position:"relative",padding:".2em",borderRadius:"0.3em",color:"#c92c2c",border:"1px solid rgba(0, 0, 0, 0.1)",display:"inline",whiteSpace:"normal"},'pre[class*="language-"]:before':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"0.18em",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(-2deg)",MozTransform:"rotate(-2deg)",msTransform:"rotate(-2deg)",OTransform:"rotate(-2deg)",transform:"rotate(-2deg)"},'pre[class*="language-"]:after':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"auto",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(2deg)",MozTransform:"rotate(2deg)",msTransform:"rotate(2deg)",OTransform:"rotate(2deg)",transform:"rotate(2deg)",right:"0.75em"},comment:{color:"#7D8B99"},"block-comment":{color:"#7D8B99"},prolog:{color:"#7D8B99"},doctype:{color:"#7D8B99"},cdata:{color:"#7D8B99"},punctuation:{color:"#5F6364"},property:{color:"#c92c2c"},tag:{color:"#c92c2c"},boolean:{color:"#c92c2c"},number:{color:"#c92c2c"},"function-name":{color:"#c92c2c"},constant:{color:"#c92c2c"},symbol:{color:"#c92c2c"},deleted:{color:"#c92c2c"},selector:{color:"#2f9c0a"},"attr-name":{color:"#2f9c0a"},string:{color:"#2f9c0a"},char:{color:"#2f9c0a"},function:{color:"#2f9c0a"},builtin:{color:"#2f9c0a"},inserted:{color:"#2f9c0a"},operator:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},entity:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)",cursor:"help"},url:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},variable:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},atrule:{color:"#1990b8"},"attr-value":{color:"#1990b8"},keyword:{color:"#1990b8"},"class-name":{color:"#1990b8"},regex:{color:"#e90"},important:{color:"#e90",fontWeight:"normal"},".language-css .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},".style .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:".7"},'pre[class*="language-"].line-numbers.line-numbers':{paddingLeft:"0"},'pre[class*="language-"].line-numbers.line-numbers code':{paddingLeft:"3.8em"},'pre[class*="language-"].line-numbers.line-numbers .line-numbers-rows':{left:"0"},'pre[class*="language-"][data-line]':{paddingTop:"0",paddingBottom:"0",paddingLeft:"0"},"pre[data-line] code":{position:"relative",paddingLeft:"4em"},"pre .line-highlight":{marginTop:"0"}}],219470),e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M632 888H392c-4.4 0-8 3.6-8 8v32c0 17.7 14.3 32 32 32h192c17.7 0 32-14.3 32-32v-32c0-4.4-3.6-8-8-8zM512 64c-181.1 0-328 146.9-328 328 0 121.4 66 227.4 164 284.1V792c0 17.7 14.3 32 32 32h264c17.7 0 32-14.3 32-32V676.1c98-56.7 164-162.7 164-284.1 0-181.1-146.9-328-328-328zm127.9 549.8L604 634.6V752H420V634.6l-35.9-20.8C305.4 568.3 256 484.5 256 392c0-141.4 114.6-256 256-256s256 114.6 256 256c0 92.5-49.4 176.3-128.1 221.8z"}}]},name:"bulb",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["BulbOutlined",0,a],812618)},84899,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645),o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M931.4 498.9L94.9 79.5c-3.4-1.7-7.3-2.1-11-1.2a15.99 15.99 0 00-11.7 19.3l86.2 352.2c1.3 5.3 5.2 9.6 10.4 11.3l147.7 50.7-147.6 50.7c-5.2 1.8-9.1 6-10.3 11.3L72.2 926.5c-.9 3.7-.5 7.6 1.2 10.9 3.9 7.9 13.5 11.1 21.5 7.2l836.5-417c3.1-1.5 5.6-4.1 7.2-7.1 3.9-8 .7-17.6-7.2-21.6zM170.8 826.3l50.3-205.6 295.2-101.3c2.3-.8 4.2-2.6 5-5 1.4-4.2-.8-8.7-5-10.2L221.1 403 171 198.2l628 314.9-628.2 313.2z"}}]},name:"send",theme:"outlined"},r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["SendOutlined",0,a],84899)},872934,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM770.87 199.13l-52.2-52.2a8.01 8.01 0 014.7-13.6l179.4-21c5.1-.6 9.5 3.7 8.9 8.9l-21 179.4c-.8 6.6-8.9 9.4-13.6 4.7l-52.4-52.4-256.2 256.2a8.03 8.03 0 01-11.3 0l-42.4-42.4a8.03 8.03 0 010-11.3l256.1-256.3z"}}]},name:"export",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["ExportOutlined",0,a],872934)},518617,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm0 76c-205.4 0-372 166.6-372 372s166.6 372 372 372 372-166.6 372-372-166.6-372-372-372zm128.01 198.83c.03 0 .05.01.09.06l45.02 45.01a.2.2 0 01.05.09.12.12 0 010 .07c0 .02-.01.04-.05.08L557.25 512l127.87 127.86a.27.27 0 01.05.06v.02a.12.12 0 010 .07c0 .03-.01.05-.05.09l-45.02 45.02a.2.2 0 01-.09.05.12.12 0 01-.07 0c-.02 0-.04-.01-.08-.05L512 557.25 384.14 685.12c-.04.04-.06.05-.08.05a.12.12 0 01-.07 0c-.03 0-.05-.01-.09-.05l-45.02-45.02a.2.2 0 01-.05-.09.12.12 0 010-.07c0-.02.01-.04.06-.08L466.75 512 338.88 384.14a.27.27 0 01-.05-.06l-.01-.02a.12.12 0 010-.07c0-.03.01-.05.05-.09l45.02-45.02a.2.2 0 01.09-.05.12.12 0 01.07 0c.02 0 .04.01.08.06L512 466.75l127.86-127.86c.04-.05.06-.06.08-.06a.12.12 0 01.07 0z"}}]},name:"close-circle",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["CloseCircleOutlined",0,a],518617)},891547,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(199133),r=e.i(764205);e.s(["default",0,({onChange:e,value:a,className:n,accessToken:s,disabled:l})=>{let[c,d]=(0,i.useState)([]),[u,p]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(s){p(!0);try{let e=await (0,r.getGuardrailsList)(s);console.log("Guardrails response:",e),e.guardrails&&(console.log("Guardrails data:",e.guardrails),d(e.guardrails))}catch(e){console.error("Error fetching guardrails:",e)}finally{p(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onChange:t=>{console.log("Selected guardrails:",t),e(t)},value:a,loading:u,className:n,allowClear:!0,options:c.map(e=>(console.log("Mapping guardrail:",e),{label:`${e.guardrail_name}`,value:e.guardrail_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},921511,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(199133),r=e.i(764205);function a(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let i=e.version_number??1,o=e.version_status??"draft";return{label:`${e.policy_name} — v${i} (${o})${e.description?` — ${e.description}`:""}`,value:"production"===o?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["default",0,({onChange:e,value:n,className:s,accessToken:l,disabled:c,onPoliciesLoaded:d})=>{let[u,p]=(0,i.useState)([]),[m,g]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(l){g(!0);try{let e=await (0,r.getPoliciesList)(l);e.policies&&(p(e.policies),d?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{g(!1)}}})()},[l,d]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",disabled:c,placeholder:c?"Setting policies is a premium feature.":"Select policies (production or published versions)",onChange:t=>{e(t)},value:n,loading:m,className:s,allowClear:!0,options:a(u),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})},"getPolicyOptionEntries",()=>a])},916940,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(199133),r=e.i(764205);e.s(["default",0,({onChange:e,value:a,className:n,accessToken:s,placeholder:l="Select vector stores",disabled:c=!1})=>{let[d,u]=(0,i.useState)([]),[p,m]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(s){m(!0);try{let e=await (0,r.vectorStoreListCall)(s);e.data&&u(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{m(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",placeholder:l,onChange:e,value:a,loading:p,className:n,allowClear:!0,options:d.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,title:e.vector_store_description||e.vector_store_id})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"},disabled:c})})}])},689020,e=>{"use strict";var t=e.i(764205);let i=async e=>{try{let i=await (0,t.modelHubCall)(e);if(console.log("model_info:",i),i?.data.length>0){let e=i.data.map(e=>({model_group:e.model_group,mode:e?.mode}));return e.sort((e,t)=>e.model_group.localeCompare(t.model_group)),e}return[]}catch(e){throw console.error("Error fetching model info:",e),e}};e.s(["fetchAvailableModels",0,i])},983561,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M300 328a60 60 0 10120 0 60 60 0 10-120 0zM852 64H172c-17.7 0-32 14.3-32 32v660c0 17.7 14.3 32 32 32h680c17.7 0 32-14.3 32-32V96c0-17.7-14.3-32-32-32zm-32 660H204V128h616v596zM604 328a60 60 0 10120 0 60 60 0 10-120 0zm250.2 556H169.8c-16.5 0-29.8 14.3-29.8 32v36c0 4.4 3.3 8 7.4 8h729.1c4.1 0 7.4-3.6 7.4-8v-36c.1-17.7-13.2-32-29.7-32zM664 508H360c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h304c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"robot",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["RobotOutlined",0,a],983561)},737434,e=>{"use strict";var t=e.i(184163);e.s(["DownloadOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["ArrowLeftOutlined",0,a],447566)},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["ClockCircleOutlined",0,a],637235)},782273,793916,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M625.9 115c-5.9 0-11.9 1.6-17.4 5.3L254 352H90c-8.8 0-16 7.2-16 16v288c0 8.8 7.2 16 16 16h164l354.5 231.7c5.5 3.6 11.6 5.3 17.4 5.3 16.7 0 32.1-13.3 32.1-32.1V147.1c0-18.8-15.4-32.1-32.1-32.1zM586 803L293.4 611.7l-18-11.7H146V424h129.4l17.9-11.7L586 221v582zm348-327H806c-8.8 0-16 7.2-16 16v40c0 8.8 7.2 16 16 16h128c8.8 0 16-7.2 16-16v-40c0-8.8-7.2-16-16-16zm-41.9 261.8l-110.3-63.7a15.9 15.9 0 00-21.7 5.9l-19.9 34.5c-4.4 7.6-1.8 17.4 5.8 21.8L856.3 800a15.9 15.9 0 0021.7-5.9l19.9-34.5c4.4-7.6 1.7-17.4-5.8-21.8zM760 344a15.9 15.9 0 0021.7 5.9L892 286.2c7.6-4.4 10.2-14.2 5.8-21.8L878 230a15.9 15.9 0 00-21.7-5.9L746 287.8a15.99 15.99 0 00-5.8 21.8L760 344z"}}]},name:"sound",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["SoundOutlined",0,a],782273);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M842 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 140.3-113.7 254-254 254S258 594.3 258 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 168.7 126.6 307.9 290 327.6V884H326.7c-13.7 0-24.7 14.3-24.7 32v36c0 4.4 2.8 8 6.2 8h407.6c3.4 0 6.2-3.6 6.2-8v-36c0-17.7-11-32-24.7-32H548V782.1c165.3-18 294-158 294-328.1zM512 624c93.9 0 170-75.2 170-168V232c0-92.8-76.1-168-170-168s-170 75.2-170 168v224c0 92.8 76.1 168 170 168zm-94-392c0-50.6 41.9-92 94-92s94 41.4 94 92v224c0 50.6-41.9 92-94 92s-94-41.4-94-92V232z"}}]},name:"audio",theme:"outlined"};var s=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["AudioOutlined",0,s],793916)},149192,e=>{"use strict";var t=e.i(864517);e.s(["CloseOutlined",()=>t.default])},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["LinkOutlined",0,a],596239)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["DollarOutlined",0,a],458505)},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["CheckCircleOutlined",0,a],245704)},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var r=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(r.default,(0,t.default)({},e,{ref:a,icon:o}))});e.s(["CodeOutlined",0,a],245094)},611052,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(212931),r=e.i(311451),a=e.i(790848),n=e.i(998573),s=e.i(438957);e.i(247167);var l=e.i(931067);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240zm460 600H232V536h560v304zM484 701v53c0 4.4 3.6 8 8 8h40c4.4 0 8-3.6 8-8v-53a48.01 48.01 0 10-56 0z"}}]},name:"lock",theme:"outlined"};var d=e.i(9583),u=i.forwardRef(function(e,t){return i.createElement(d.default,(0,l.default)({},e,{ref:t,icon:c}))}),p=e.i(492030),m=e.i(266537),g=e.i(447566),f=e.i(149192),h=e.i(596239);e.s(["ByokCredentialModal",0,({server:e,open:l,onClose:c,onSuccess:d,accessToken:_})=>{let[v,b]=(0,i.useState)(1),[x,y]=(0,i.useState)(""),[w,S]=(0,i.useState)(!0),[j,k]=(0,i.useState)(!1),C=e.alias||e.server_name||"Service",O=C.charAt(0).toUpperCase(),N=()=>{b(1),y(""),S(!0),k(!1),c()},z=async()=>{if(!x.trim())return void n.message.error("Please enter your API key");k(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${_}`},body:JSON.stringify({credential:x.trim(),save:w})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}n.message.success(`Connected to ${C}`),d(e.server_id),N()}catch(e){n.message.error(e.message||"Failed to connect")}finally{k(!1)}};return(0,t.jsx)(o.Modal,{open:l,onCancel:N,footer:null,width:480,closeIcon:null,className:"byok-modal",children:(0,t.jsxs)("div",{className:"relative p-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===v?(0,t.jsxs)("button",{onClick:()=>b(1),className:"flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm",children:[(0,t.jsx)(g.ArrowLeftOutlined,{})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${1===v?"bg-blue-500":"bg-gray-300"}`}),(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${2===v?"bg-blue-500":"bg-gray-300"}`})]}),(0,t.jsx)("button",{onClick:N,className:"text-gray-400 hover:text-gray-600",children:(0,t.jsx)(f.CloseOutlined,{})})]}),1===v?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(m.ArrowRightOutlined,{className:"text-gray-400 text-lg"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:O})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:["Connect ",C]}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["LiteLLM needs access to ",C," to complete your request."]}),(0,t.jsx)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-gray-800 mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-gray-500 text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to"," ",C,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-green-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,i)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-gray-700",children:[(0,t.jsx)(p.CheckOutlined,{className:"text-green-500 flex-shrink-0"}),e]},i))})]}),(0,t.jsxs)("button",{onClick:()=>b(2),className:"w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:["Continue to Authentication ",(0,t.jsx)(m.ArrowRightOutlined,{})]}),(0,t.jsx)("button",{onClick:N,className:"mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-4",children:(0,t.jsx)(s.KeyOutlined,{className:"text-blue-400 text-xl"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["Enter your ",C," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsxs)("label",{className:"block text-sm font-semibold text-gray-800 mb-2",children:[C," API Key"]}),(0,t.jsx)(r.Input.Password,{placeholder:"Enter your API key",value:x,onChange:e=>y(e.target.value),size:"large",className:"rounded-lg"}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-500 hover:text-blue-700 text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key? ",(0,t.jsx)(h.LinkOutlined,{})]})]}),(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-gray-800",children:"Save key for future use"})]}),(0,t.jsx)(a.Switch,{checked:w,onChange:S})]}),(0,t.jsxs)("div",{className:"bg-blue-50 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(u,{className:"text-blue-400 mt-0.5 flex-shrink-0"}),(0,t.jsx)("p",{className:"text-sm text-blue-700",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)("button",{onClick:z,disabled:j,className:"w-full bg-blue-500 hover:bg-blue-600 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:[(0,t.jsx)(u,{})," Connect & Authorize"]})]})]})})}],611052)}]);