(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,190272,785913,e=>{"use strict";var t,s,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((s={}).IMAGE="image",s.VIDEO="video",s.CHAT="chat",s.RESPONSES="responses",s.IMAGE_EDITS="image_edits",s.ANTHROPIC_MESSAGES="anthropic_messages",s.EMBEDDINGS="embeddings",s.SPEECH="speech",s.TRANSCRIPTION="transcription",s.A2A_AGENTS="a2a_agents",s.MCP="mcp",s);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:s,accessToken:a,apiKey:r,inputMessage:n,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:p,selectedMCPServers:u,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:h,endpointType:f,selectedModel:x,selectedSdk:v,proxySettings:_}=e,y="session"===s?a:r,b=window.location.origin,j=_?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?b=j:_?.PROXY_BASE_URL&&(b=_.PROXY_BASE_URL);let S=n||"Your prompt here",w=S.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),N=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),d.length>0&&(C.vector_stores=d),c.length>0&&(C.guardrails=c),p.length>0&&(C.policies=p);let k=x||"your-model-name",A="azure"===v?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${b}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${b}"
)`;switch(f){case i.CHAT:{let e=Object.keys(C).length>0,s="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=N.length>0?N:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(a,null,4)}${s}
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
#     ]${s}
# )
# print(response_with_file)
`;break}case i.RESPONSES:{let e=Object.keys(C).length>0,s="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();s=`,
    extra_body=${e}`}let a=N.length>0?N:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(a,null,4)}${s}
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
#                 {"type": "input_text", "text": "${w}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${s}
# )
# print(response_with_file.output_text)
`;break}case i.IMAGE:t="azure"===v?`
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
prompt = "${w}"

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
`;break;case i.IMAGE_EDITS:t="azure"===v?`
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
	model="${k}",
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
prompt = "${w}"

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
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${n?`,
	prompt="${n.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${n||"Your text to convert to speech here"}",
	voice="${h}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${A}
${t}`}],190272)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var i=e.i(9583),r=s.forwardRef(function(e,r){return s.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["LinkOutlined",0,r],596239)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var i=e.i(9583),r=s.forwardRef(function(e,r){return s.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["ClockCircleOutlined",0,r],637235)},516015,(e,t,s)=>{},898547,(e,t,s)=>{var a=e.i(247167);e.r(516015);var i=e.r(271645),r=i&&"object"==typeof i&&"default"in i?i:{default:i},n=void 0!==a.default&&a.default.env&&!0,o=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,s=t.name,a=void 0===s?"stylesheet":s,i=t.optimizeForSpeed,r=void 0===i?n:i;d(o(a),"`name` must be a string"),this._name=a,this._deletedRulePlaceholder="#"+a+"-deleted-rule____{}",d("boolean"==typeof r,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=r,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,s=e.prototype;return s.setOptimizeForSpeed=function(e){d("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),d(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},s.isOptimizeForSpeed=function(){return this._optimizeForSpeed},s.inject=function(){var e=this;if(d(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(n||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,s){return"number"==typeof s?e._serverSheet.cssRules[s]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),s},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},s.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},s.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},s.insertRule=function(e,t){if(d(o(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var s=this.getSheet();"number"!=typeof t&&(t=s.cssRules.length);try{s.insertRule(e,t)}catch(t){return n||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var a=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,a))}return this._rulesCount++},s.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var s="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!s.cssRules[e])return e;s.deleteRule(e);try{s.insertRule(t,e)}catch(a){n||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),s.insertRule(this._deletedRulePlaceholder,e)}}else{var a=this._tags[e];d(a,"old rule at index `"+e+"` not found"),a.textContent=t}return e},s.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];d(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},s.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},s.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,s){return s?t=t.concat(Array.prototype.map.call(e.getSheetForTag(s).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},s.makeStyleTag=function(e,t,s){t&&d(o(t),"makeStyleTag accepts only strings as second parameter");var a=document.createElement("style");this._nonce&&a.setAttribute("nonce",this._nonce),a.type="text/css",a.setAttribute("data-"+e,""),t&&a.appendChild(document.createTextNode(t));var i=document.head||document.getElementsByTagName("head")[0];return s?i.insertBefore(a,s):i.appendChild(a),a},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var s=0;s<t.length;s++){var a=t[s];a.enumerable=a.enumerable||!1,a.configurable=!0,"value"in a&&(a.writable=!0),Object.defineProperty(e,a.key,a)}}(e.prototype,t),e}();function d(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var c=function(e){for(var t=5381,s=e.length;s;)t=33*t^e.charCodeAt(--s);return t>>>0},p={};function u(e,t){if(!t)return"jsx-"+e;var s=String(t),a=e+s;return p[a]||(p[a]="jsx-"+c(e+"-"+s)),p[a]}function m(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var s=e+t;return p[s]||(p[s]=t.replace(/__jsx-style-dynamic-selector/g,e)),p[s]}var g=function(){function e(e){var t=void 0===e?{}:e,s=t.styleSheet,a=void 0===s?null:s,i=t.optimizeForSpeed,r=void 0!==i&&i;this._sheet=a||new l({name:"styled-jsx",optimizeForSpeed:r}),this._sheet.inject(),a&&"boolean"==typeof r&&(this._sheet.setOptimizeForSpeed(r),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var s=this.getIdAndRules(e),a=s.styleId,i=s.rules;if(a in this._instancesCounts){this._instancesCounts[a]+=1;return}var r=i.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[a]=r,this._instancesCounts[a]=1},t.remove=function(e){var t=this,s=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(s in this._instancesCounts,"styleId: `"+s+"` not found"),this._instancesCounts[s]-=1,this._instancesCounts[s]<1){var a=this._fromServer&&this._fromServer[s];a?(a.parentNode.removeChild(a),delete this._fromServer[s]):(this._indices[s].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[s]),delete this._instancesCounts[s]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],s=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return s[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,s;return t=this.cssRules(),void 0===(s=e)&&(s={}),t.map(function(e){var t=e[0],a=e[1];return r.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:s.nonce?s.nonce:void 0,dangerouslySetInnerHTML:{__html:a}})})},t.getIdAndRules=function(e){var t=e.children,s=e.dynamic,a=e.id;if(s){var i=u(a,s);return{styleId:i,rules:Array.isArray(t)?t.map(function(e){return m(i,e)}):[m(i,t)]}}return{styleId:u(a),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),h=i.createContext(null);function f(){return new g}function x(){return i.useContext(h)}h.displayName="StyleSheetContext";var v=r.default.useInsertionEffect||r.default.useLayoutEffect,_="u">typeof window?f():void 0;function y(e){var t=_||x();return t&&("u"<typeof window?t.add(e):v(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}y.dynamic=function(e){return e.map(function(e){return u(e[0],e[1])}).join(" ")},s.StyleRegistry=function(e){var t=e.registry,s=e.children,a=i.useContext(h),n=i.useState(function(){return a||t||f()})[0];return r.default.createElement(h.Provider,{value:n},s)},s.createStyleRegistry=f,s.style=y,s.useStyleRegistry=x},437902,(e,t,s)=>{t.exports=e.r(898547).style},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),s=e.i(343794),a=e.i(914949),i=e.i(404948);let r=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,r],836938);var n=e.i(613541),o=e.i(763731),l=e.i(242064),d=e.i(491816);e.i(793154);var c=e.i(880476),p=e.i(183293),u=e.i(717356),m=e.i(320560),g=e.i(307358),h=e.i(246422),f=e.i(838378),x=e.i(617933);let v=(0,h.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:s}=e,a=(0,f.mergeToken)(e,{popoverBg:t,popoverColor:s});return[(e=>{let{componentCls:t,popoverColor:s,titleMinWidth:a,fontWeightStrong:i,innerPadding:r,boxShadowSecondary:n,colorTextHeading:o,borderRadiusLG:l,zIndexPopup:d,titleMarginBottom:c,colorBgElevated:u,popoverBg:g,titleBorderBottom:h,innerContentPadding:f,titlePadding:x}=e;return[{[t]:Object.assign(Object.assign({},(0,p.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:d,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":u,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:g,backgroundClip:"padding-box",borderRadius:l,boxShadow:n,padding:r},[`${t}-title`]:{minWidth:a,marginBottom:c,color:o,fontWeight:i,borderBottom:h,padding:x},[`${t}-inner-content`]:{color:s,padding:f}})},(0,m.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(a),(e=>{let{componentCls:t}=e;return{[t]:x.PresetColors.map(s=>{let a=e[`${s}6`];return{[`&${t}-${s}`]:{"--antd-arrow-background-color":a,[`${t}-inner`]:{backgroundColor:a},[`${t}-arrow`]:{background:"transparent"}}}})}})(a),(0,u.initZoomMotion)(a,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:s,fontHeight:a,padding:i,wireframe:r,zIndexPopupBase:n,borderRadiusLG:o,marginXS:l,lineType:d,colorSplit:c,paddingSM:p}=e,u=s-a;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:n+30},(0,g.getArrowToken)(e)),(0,m.getArrowOffsetToken)({contentRadius:o,limitVerticalRadius:!0})),{innerPadding:12*!r,titleMarginBottom:r?0:l,titlePadding:r?`${u/2}px ${i}px ${u/2-t}px`:0,titleBorderBottom:r?`${t}px ${d} ${c}`:"none",innerContentPadding:r?`${p}px ${i}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var _=function(e,t){var s={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(s[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(s[a[i]]=e[a[i]]);return s};let y=({title:e,content:s,prefixCls:a})=>e||s?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${a}-title`},e),s&&t.createElement("div",{className:`${a}-inner-content`},s)):null,b=e=>{let{hashId:a,prefixCls:i,className:n,style:o,placement:l="top",title:d,content:p,children:u}=e,m=r(d),g=r(p),h=(0,s.default)(a,i,`${i}-pure`,`${i}-placement-${l}`,n);return t.createElement("div",{className:h,style:o},t.createElement("div",{className:`${i}-arrow`}),t.createElement(c.Popup,Object.assign({},e,{className:a,prefixCls:i}),u||t.createElement(y,{prefixCls:i,title:m,content:g})))},j=e=>{let{prefixCls:a,className:i}=e,r=_(e,["prefixCls","className"]),{getPrefixCls:n}=t.useContext(l.ConfigContext),o=n("popover",a),[d,c,p]=v(o);return d(t.createElement(b,Object.assign({},r,{prefixCls:o,hashId:c,className:(0,s.default)(i,p)})))};e.s(["Overlay",0,y,"default",0,j],310730);var S=function(e,t){var s={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(s[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(s[a[i]]=e[a[i]]);return s};let w=t.forwardRef((e,c)=>{var p,u;let{prefixCls:m,title:g,content:h,overlayClassName:f,placement:x="top",trigger:_="hover",children:b,mouseEnterDelay:j=.1,mouseLeaveDelay:w=.1,onOpenChange:N,overlayStyle:C={},styles:k,classNames:A}=e,O=S(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:E,className:T,style:I,classNames:R,styles:P}=(0,l.useComponentConfig)("popover"),M=E("popover",m),[L,$,z]=v(M),F=E(),D=(0,s.default)(f,$,z,T,R.root,null==A?void 0:A.root),U=(0,s.default)(R.body,null==A?void 0:A.body),[B,H]=(0,a.default)(!1,{value:null!=(p=e.open)?p:e.visible,defaultValue:null!=(u=e.defaultOpen)?u:e.defaultVisible}),V=(e,t)=>{H(e,!0),null==N||N(e,t)},G=r(g),W=r(h);return L(t.createElement(d.default,Object.assign({placement:x,trigger:_,mouseEnterDelay:j,mouseLeaveDelay:w},O,{prefixCls:M,classNames:{root:D,body:U},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},P.root),I),C),null==k?void 0:k.root),body:Object.assign(Object.assign({},P.body),null==k?void 0:k.body)},ref:c,open:B,onOpenChange:e=>{V(e)},overlay:G||W?t.createElement(y,{prefixCls:M,title:G,content:W}):null,transitionName:(0,n.getTransitionName)(F,"zoom-big",O.transitionName),"data-popover-inject":!0}),(0,o.cloneElement)(b,{onKeyDown:e=>{var s,a;(0,t.isValidElement)(b)&&(null==(a=null==b?void 0:(s=b.props).onKeyDown)||a.call(s,e)),e.keyCode===i.default.ESC&&V(!1,e)}})))});w._InternalPanelDoNotUseOrYouWillBeFired=j,e.s(["default",0,w],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},872934,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM770.87 199.13l-52.2-52.2a8.01 8.01 0 014.7-13.6l179.4-21c5.1-.6 9.5 3.7 8.9 8.9l-21 179.4c-.8 6.6-8.9 9.4-13.6 4.7l-52.4-52.4-256.2 256.2a8.03 8.03 0 01-11.3 0l-42.4-42.4a8.03 8.03 0 010-11.3l256.1-256.3z"}}]},name:"export",theme:"outlined"};var i=e.i(9583),r=s.forwardRef(function(e,r){return s.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["ExportOutlined",0,r],872934)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),s=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var i=e.i(9583),r=s.forwardRef(function(e,r){return s.createElement(i.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["DollarOutlined",0,r],458505)},903446,e=>{"use strict";let t=(0,e.i(475254).default)("settings",[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["default",()=>t])},213970,657150,e=>{"use strict";var t=e.i(843476),s=e.i(271645),a=e.i(220486),i=e.i(727749),r=e.i(447593),n=e.i(955135),o=e.i(91500),l=e.i(646563),d=e.i(464571),c=e.i(311451),p=e.i(199133),u=e.i(592968),m=e.i(422233),g=e.i(761793),h=e.i(964421),f=e.i(254530),x=e.i(689020),v=e.i(921687),_=e.i(953860),y=e.i(903446),y=y,b=e.i(37727),j=e.i(475254);let S=(0,j.default)("bot",[["path",{d:"M12 8V4H8",key:"hb8ula"}],["rect",{width:"16",height:"12",x:"4",y:"8",rx:"2",key:"enze0r"}],["path",{d:"M2 14h2",key:"vft8re"}],["path",{d:"M20 14h2",key:"4cs60a"}],["path",{d:"M15 13v2",key:"1xurst"}],["path",{d:"M9 13v2",key:"rq6x2g"}]]);e.s(["default",()=>S],657150);var w=e.i(531278);let N=(0,j.default)("user-round",[["circle",{cx:"12",cy:"8",r:"5",key:"1hypcn"}],["path",{d:"M20 21a8 8 0 0 0-16 0",key:"rfgkzh"}]]);var C=e.i(918789),k=e.i(650056),A=e.i(219470),O=e.i(843153),E=e.i(966988),T=e.i(989022),I=e.i(152401);function R({messages:e,isLoading:s}){if(0===e.length)return(0,t.jsx)("div",{className:"h-full"});let a=[],i=0;for(;i<e.length;){let t=e[i];if("user"===t.role){let s=e[i+1];if(s?.role==="assistant"){a.push({user:t,assistant:s}),i+=2;continue}a.push({user:t})}else"assistant"===t.role&&a.push({assistant:t});i+=1}let r=e=>(0,t.jsxs)("div",{className:"whitespace-pre-wrap break-words",style:{wordWrap:"break-word",overflowWrap:"break-word",wordBreak:"break-word",hyphens:"auto"},children:[(0,t.jsx)(O.default,{message:e}),(0,t.jsx)(C.default,{components:{code({node:e,inline:s,className:a,children:i,...r}){let n=/language-(\w+)/.exec(a||"");return!s&&n?(0,t.jsx)(k.Prism,{style:A.coy,language:n[1],PreTag:"div",className:"rounded-md my-2",wrapLines:!0,wrapLongLines:!0,...r,children:String(i).replace(/\n$/,"")}):(0,t.jsx)("code",{className:`${a} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`,...r,children:i})},pre:({node:e,...s})=>(0,t.jsx)("pre",{style:{overflowX:"auto",maxWidth:"100%"},...s})},children:"string"==typeof e.content?e.content:""})]});return(0,t.jsxs)("div",{className:"flex flex-col gap-6 min-w-0 w-full p-4",children:[a.map((e,i)=>{let n=e.assistant,o=n?.model||"Assistant";return(0,t.jsxs)("div",{className:"space-y-4",children:[e.user&&(0,t.jsxs)("div",{className:"space-y-2 min-w-0",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("div",{className:"flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-100 text-blue-600",children:(0,t.jsx)(N,{size:16})}),(0,t.jsx)("div",{className:"text-sm font-semibold text-gray-700",children:"You"})]}),r(e.user)]}),(0,t.jsx)("div",{className:"border-t border-gray-200"}),n?(0,t.jsxs)("div",{className:"space-y-3 min-w-0",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("div",{className:"flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-100 text-gray-600",children:(0,t.jsx)(S,{size:16})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)("span",{className:"text-sm font-semibold text-gray-700",children:o}),n.toolName&&(0,t.jsx)("span",{className:"rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600",children:n.toolName})]})]}),n.reasoningContent&&(0,t.jsx)(E.default,{reasoningContent:n.reasoningContent}),n.searchResults&&(0,t.jsx)(I.SearchResultsDisplay,{searchResults:n.searchResults}),r(n),(n.timeToFirstToken||n.totalLatency||n.usage)&&(0,t.jsx)(T.default,{timeToFirstToken:n.timeToFirstToken,totalLatency:n.totalLatency,usage:n.usage,toolName:n.toolName})]}):s&&i===a.length-1?(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm text-gray-500",children:[(0,t.jsx)(w.Loader2,{size:18,className:"animate-spin"}),(0,t.jsx)("span",{children:"Generating response..."})]}):(0,t.jsx)("div",{className:"text-sm text-gray-500",children:"Waiting for a response..."})]},i)}),s&&0===a.length&&(0,t.jsxs)("div",{className:"flex items-center gap-2 text-gray-500",children:[(0,t.jsx)(w.Loader2,{size:18,className:"animate-spin"}),(0,t.jsx)("span",{children:"Generating response..."})]})]})}var P=e.i(482725);function M({value:e,options:s,loading:a,config:i,onChange:r}){return(0,t.jsx)(p.Select,{value:e||void 0,placeholder:a?`Loading ${i.selectorLabel.toLowerCase()}s...`:i.selectorPlaceholder,onChange:r,loading:a,showSearch:!0,filterOption:(e,t)=>(t?.label??"").toLowerCase().includes(e.toLowerCase()),options:s,className:"w-48 md:w-64 lg:w-72",notFoundContent:a?(0,t.jsx)("div",{className:"flex items-center justify-center py-2",children:(0,t.jsx)(P.Spin,{size:"small"})}):`No ${i.selectorLabel.toLowerCase()}s available`})}var L=e.i(318059),$=e.i(916940),z=e.i(891547),F=e.i(536916),D=e.i(312361),U=e.i(282786),B=e.i(850627);let H="/v1/chat/completions",V="/a2a",G={[H]:{id:H,label:"/v1/chat/completions",selectorType:"model",selectorLabel:"Model",selectorPlaceholder:"Select a model",inputPlaceholder:"Send a prompt to compare models",loadingMessage:"Gathering responses from all models...",validationMessage:"Select a model before sending a message."},[V]:{id:V,label:"/a2a (Agents)",selectorType:"agent",selectorLabel:"Agent",selectorPlaceholder:"Select an agent",inputPlaceholder:"Send a message to compare agents",loadingMessage:"Gathering responses from all agents...",validationMessage:"Select an agent before sending a message."}},W=e=>"agent"===G[e].selectorType,q=(e,t)=>W(t)?e.agent:e.model;function K({comparison:e,onUpdate:a,onRemove:i,canRemove:r,selectorOptions:n,isLoadingOptions:o,endpointConfig:l,apiKey:d}){let c=W(l.id),p=q(e,l.id),[u,m]=(0,s.useState)(!1),g=(t,s)=>{a({[t]:s},e.applyAcrossModels?{applyToAll:!0,keysToApply:[t]}:void 0)},h=e.useAdvancedParams?1:.4,f=e.useAdvancedParams?"text-gray-700":"text-gray-400",x=(0,t.jsxs)("div",{className:"w-[300px] max-h-[65vh] overflow-y-auto relative",children:[(0,t.jsx)("button",{onClick:()=>{m(!1)},className:"absolute top-0 right-0 p-1 hover:bg-gray-100 rounded transition-colors text-gray-500 hover:text-gray-700 z-10",children:(0,t.jsx)(b.X,{size:14})}),(0,t.jsxs)("div",{className:"space-y-2",children:[(0,t.jsx)("div",{className:"flex items-center gap-2",children:(0,t.jsx)(F.Checkbox,{checked:e.applyAcrossModels,onChange:t=>{t.target.checked?a({applyAcrossModels:!0,temperature:e.temperature,maxTokens:e.maxTokens,tags:[...e.tags],vectorStores:[...e.vectorStores],guardrails:[...e.guardrails],useAdvancedParams:e.useAdvancedParams},{applyToAll:!0,keysToApply:["temperature","maxTokens","tags","vectorStores","guardrails","useAdvancedParams"]}):a({applyAcrossModels:!1})},children:(0,t.jsx)("span",{className:"text-xs font-medium",children:"Sync Settings Across Models"})})}),(0,t.jsx)(D.Divider,{className:"border-gray-200"}),(0,t.jsxs)("div",{children:[(0,t.jsx)("h4",{className:"text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide",children:"General Settings"}),(0,t.jsxs)("div",{className:"space-y-2",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)("label",{className:"text-xs font-medium text-gray-600 block mb-0.5",children:"Tags"}),(0,t.jsx)(L.default,{value:e.tags,onChange:e=>g("tags",e),accessToken:d})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("label",{className:"text-xs font-medium text-gray-600 block mb-0.5",children:"Vector Stores"}),(0,t.jsx)($.default,{value:e.vectorStores,onChange:e=>g("vectorStores",e),accessToken:d})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("label",{className:"text-xs font-medium text-gray-600 block mb-0.5",children:"Guardrails"}),(0,t.jsx)(z.default,{value:e.guardrails,onChange:e=>g("guardrails",e),accessToken:d})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("h4",{className:"text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide",children:"Advanced Settings"}),(0,t.jsxs)("div",{className:"space-y-2",children:[(0,t.jsx)("div",{className:"flex items-center gap-2 pb-1",children:(0,t.jsx)(F.Checkbox,{checked:e.useAdvancedParams,onChange:t=>{a({useAdvancedParams:t.target.checked},e.applyAcrossModels?{applyToAll:!0,keysToApply:["useAdvancedParams"]}:void 0)},children:(0,t.jsx)("span",{className:"text-sm font-medium",children:"Use Advanced Parameters"})})}),(0,t.jsxs)("div",{className:"space-y-2 transition-opacity duration-200",style:{opacity:h},children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-1",children:[(0,t.jsx)("label",{className:`text-xs font-medium ${f}`,children:"Temperature"}),(0,t.jsx)("span",{className:`text-xs ${f}`,children:e.temperature.toFixed(2)})]}),(0,t.jsx)(B.Slider,{min:0,max:2,step:.01,value:e.temperature,onChange:e=>{g("temperature",Math.min(2,Math.max(0,Number((Array.isArray(e)?e[0]:e).toFixed(2)))))},disabled:!e.useAdvancedParams})]}),(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-1",children:[(0,t.jsx)("label",{className:`text-xs font-medium ${f}`,children:"Max Tokens"}),(0,t.jsx)("span",{className:`text-xs ${f}`,children:e.maxTokens})]}),(0,t.jsx)(B.Slider,{min:1,max:32768,step:1,value:e.maxTokens,onChange:e=>{g("maxTokens",Math.min(32768,Math.max(1,Math.round(Array.isArray(e)?e[0]:e))))},disabled:!e.useAdvancedParams})]})]})]})]})]})]});return(0,t.jsxs)("div",{className:"bg-white first:border-l-0 border-l border-gray-200 flex flex-col min-h-0",children:[(0,t.jsxs)("div",{className:"border-b flex items-center justify-between gap-3 px-4 py-3",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3 flex-1",children:[(0,t.jsx)(M,{value:p,options:n,loading:o,config:l,onChange:e=>a(c?{agent:e}:{model:e})}),(0,t.jsx)("div",{className:"flex items-center gap-2",children:(0,t.jsx)(U.Popover,{content:x,trigger:[],open:u,onOpenChange:()=>{},placement:"bottomRight",destroyTooltipOnHide:!1,children:(0,t.jsx)("button",{onClick:e=>{e.stopPropagation(),m(e=>!e)},className:`p-2 rounded-lg transition-colors ${u?"bg-gray-200 text-gray-700":"hover:bg-gray-100 text-gray-600"}`,children:(0,t.jsx)(y.default,{size:18})})})})]}),r&&(0,t.jsx)("button",{onClick:e=>{e.stopPropagation(),i()},className:"p-2 hover:bg-red-50 text-red-600 rounded-lg transition-colors",children:(0,t.jsx)(b.X,{size:18})})]}),(0,t.jsx)("div",{className:"relative flex-1 flex flex-col min-h-0",children:(0,t.jsx)("div",{className:"flex-1 max-h-[calc(100vh-385px)] overflow-auto rounded-b-2xl",children:(0,t.jsx)(R,{messages:e.messages,isLoading:e.isLoading})})})]})}var Y=e.i(132104);let{TextArea:X}=c.Input;function J({value:e,onChange:s,onSend:a,disabled:i,hasAttachment:r,uploadComponent:n}){let o=!i&&(e.trim().length>0||!!r);return(0,t.jsx)("div",{className:"flex items-center gap-2",children:(0,t.jsxs)("div",{className:"flex items-center flex-1 bg-white border border-gray-300 rounded-xl px-3 py-1 min-h-[44px]",children:[n&&(0,t.jsx)("div",{className:"flex-shrink-0 mr-2",children:n}),(0,t.jsx)(X,{value:e,onChange:e=>s(e.target.value),onKeyDown:e=>{"Enter"===e.key&&!e.shiftKey&&(e.preventDefault(),o&&a())},placeholder:"Type your message... (Shift+Enter for new line)",disabled:i,className:"flex-1",autoSize:{minRows:1,maxRows:4},style:{resize:"none",border:"none",boxShadow:"none",background:"transparent",padding:"4px 0",fontSize:"14px",lineHeight:"20px"}}),(0,t.jsx)(d.Button,{onClick:a,disabled:!o,icon:(0,t.jsx)(Y.ArrowUpOutlined,{}),shape:"circle"})]})})}let Z=["Can you summarize the key points?","What assumptions did you make?","What are the next steps?"],Q=["Write me a poem","Explain quantum computing","Draft a polite email requesting a meeting"];function ee({accessToken:e,disabledPersonalKeyCreation:a}){let[y,b]=(0,s.useState)([{id:"1",model:"",agent:"",messages:[],isLoading:!1,tags:[],mcpTools:[],vectorStores:[],guardrails:[],temperature:1,maxTokens:2048,applyAcrossModels:!1,useAdvancedParams:!1},{id:"2",model:"",agent:"",messages:[],isLoading:!1,tags:[],mcpTools:[],vectorStores:[],guardrails:[],temperature:1,maxTokens:2048,applyAcrossModels:!1,useAdvancedParams:!1}]),[j,S]=(0,s.useState)([]),[w,N]=(0,s.useState)([]),[C,k]=(0,s.useState)(!1),[A,O]=(0,s.useState)(!1),[E,T]=(0,s.useState)(H),I=G[E],R=W(E),P=R?w.map(e=>({value:e.agent_name,label:e.agent_name||e.agent_id})):j.map(e=>({value:e,label:e})),M=R?A:C,[L,$]=(0,s.useState)(""),[z,F]=(0,s.useState)(null),[D,U]=(0,s.useState)(null),[B,V]=(0,s.useState)(a?"custom":"session"),[Y,X]=(0,s.useState)(""),[ee,et]=(0,s.useState)(""),[es]=(0,s.useState)(()=>sessionStorage.getItem("customProxyBaseUrl")||"");(0,s.useEffect)(()=>{let e=setTimeout(()=>{et(Y)},300);return()=>clearTimeout(e)},[Y]),(0,s.useEffect)(()=>()=>{D&&URL.revokeObjectURL(D)},[D]);let ea=(0,s.useMemo)(()=>"session"===B?e||"":ee.trim(),[B,e,ee]),ei=(0,s.useMemo)(()=>y.length>0&&y.every(e=>!e.isLoading&&e.messages.some(e=>"assistant"===e.role)),[y]);(0,s.useEffect)(()=>{let e=!0;return(async()=>{if(!ea)return S([]);k(!0);try{let t=await (0,x.fetchAvailableModels)(ea);if(!e)return;let s=Array.from(new Set(t.map(e=>e.model_group)));S(s)}catch(t){console.error("CompareUI: failed to fetch models",t),e&&S([])}finally{e&&k(!1)}})(),()=>{e=!1}},[ea]),(0,s.useEffect)(()=>{let e=!0;return(async()=>{if(!ea||!R)return N([]);O(!0);try{let t=await (0,v.fetchAvailableAgents)(ea,es||void 0);if(!e)return;N(t)}catch(t){console.error("CompareUI: failed to fetch agents",t),e&&N([])}finally{e&&O(!1)}})(),()=>{e=!1}},[ea,R]),(0,s.useEffect)(()=>{0!==j.length&&b(e=>e.map((e,t)=>({...e,temperature:e.temperature??1,maxTokens:e.maxTokens??2048,applyAcrossModels:e.applyAcrossModels??!1,useAdvancedParams:e.useAdvancedParams??!1,...e.model?{}:{model:j[t%j.length]??""}})))},[j]);let er=()=>{D&&URL.revokeObjectURL(D),F(null),U(null)},en=(e,t)=>{b(s=>s.map(s=>{if(s.id!==e)return s;let a=[...s.messages],i=a[a.length-1];return i&&"assistant"===i.role?a[a.length-1]={...i,timeToFirstToken:t}:i&&"user"===i.role&&a.push({role:"assistant",content:"",timeToFirstToken:t}),{...s,messages:a}}))},eo=(e,t)=>{b(s=>s.map(s=>{if(s.id!==e)return s;let a=[...s.messages],i=a[a.length-1];return i&&"assistant"===i.role?a[a.length-1]={...i,totalLatency:t}:i&&"user"===i.role&&a.push({role:"assistant",content:"",totalLatency:t}),{...s,messages:a}}))},el=!!e,ed=async e=>{let t=e.trim(),s=!!z;if(!t&&!s)return;if(!ea)return void i.default.fromBackend("Please provide a Virtual Key or select Current UI Session");if(0===y.length)return;if(y.some(e=>{let t;return!((t=q(e,E))&&t.trim())}))return void i.default.fromBackend(I.validationMessage);let a=s?await (0,h.createChatMultimodalMessage)(t,z):{role:"user",content:t},r=(0,h.createChatDisplayMessage)(t,s,D||void 0,z?.name),n=new Map;y.forEach(e=>{let s=e.traceId??(0,m.v4)(),i=[...e.messages.map(({role:e,content:t})=>({role:e,content:Array.isArray(t)||"string"==typeof t?t:""})),a];n.set(e.id,{id:e.id,model:e.model,agent:e.agent,inputMessage:t,traceId:s,tags:e.tags,vectorStores:e.vectorStores,guardrails:e.guardrails,temperature:e.temperature,maxTokens:e.maxTokens,displayMessages:[...e.messages,r],apiChatHistory:i})}),0!==n.size&&(b(e=>e.map(e=>{let t=n.get(e.id);return t?{...e,traceId:t.traceId,messages:t.displayMessages,isLoading:!0}:e})),$(""),er(),n.forEach(e=>{let t=e.tags.length>0?e.tags:void 0,s=e.vectorStores.length>0?e.vectorStores:void 0,a=e.guardrails.length>0?e.guardrails:void 0,r=y.find(t=>t.id===e.id),n=r?.useAdvancedParams??!1;(R?(0,_.makeA2AStreamMessageRequest)(e.agent,e.inputMessage,(t,s)=>{b(a=>a.map(a=>{if(a.id!==e.id)return a;let i=[...a.messages],r=i[i.length-1];return r&&"assistant"===r.role?i[i.length-1]={...r,content:t,model:r.model??s}:i.push({role:"assistant",content:t,model:s}),{...a,messages:i}}))},ea,void 0,t=>en(e.id,t),t=>eo(e.id,t),void 0,es||void 0):(0,f.makeOpenAIChatCompletionRequest)(e.apiChatHistory,(t,s)=>{var a;return a=e.id,void(t&&b(e=>e.map(e=>{if(e.id!==a)return e;let i=[...e.messages],r=i[i.length-1];if(r&&"assistant"===r.role){let e="string"==typeof r.content?r.content:"";i[i.length-1]={...r,content:e+t,model:r.model??s}}else i.push({role:"assistant",content:t,model:s});return{...e,messages:i}})))},e.model,ea,t,void 0,t=>{var s;return s=e.id,void(t&&b(e=>e.map(e=>{if(e.id!==s)return e;let a=[...e.messages],i=a[a.length-1];return i&&"assistant"===i.role?a[a.length-1]={...i,reasoningContent:(i.reasoningContent||"")+t}:i&&"user"===i.role&&a.push({role:"assistant",content:"",reasoningContent:t}),{...e,messages:a}})))},t=>en(e.id,t),t=>{var s;return s=e.id,void b(e=>e.map(e=>{if(e.id!==s)return e;let a=[...e.messages],i=a[a.length-1];return i&&"assistant"===i.role&&(a[a.length-1]={...i,usage:t,toolName:void 0}),{...e,messages:a}}))},e.traceId,s,a,void 0,void 0,void 0,t=>{var s;return s=e.id,void(t&&b(e=>e.map(e=>{if(e.id!==s)return e;let a=[...e.messages],i=a[a.length-1];return i&&"assistant"===i.role&&(a[a.length-1]={...i,searchResults:t}),{...e,messages:a}})))},n?e.temperature:void 0,n?e.maxTokens:void 0,t=>eo(e.id,t),es||void 0)).catch(t=>{let s=t instanceof Error?t.message:String(t);console.error("CompareUI: failed to fetch response",t),i.default.fromBackend(s),b(t=>t.map(t=>{if(t.id!==e.id)return t;let a=[...t.messages],i=a[a.length-1],r=i&&"assistant"===i.role&&"string"==typeof i.content?i.content:"";return i&&"assistant"===i.role?a[a.length-1]={...i,content:r?`${r}
Error fetching response: ${s}`:`Error fetching response: ${s}`}:a.push({role:"assistant",content:`Error fetching response: ${s}`}),{...t,messages:a}}))}).finally(()=>{b(t=>t.map(t=>t.id===e.id?{...t,isLoading:!1}:t))})}))},ec=e=>{$(e)},ep=y.some(e=>e.messages.length>0),eu=y.some(e=>e.isLoading),em=!!z,eg=!!z?.name.toLowerCase().endsWith(".pdf"),eh=!ep&&!eu&&!em;return(0,t.jsx)("div",{className:"w-full h-full p-4 bg-white",children:(0,t.jsxs)("div",{className:"rounded-2xl border border-gray-200 bg-white shadow-sm min-h-[calc(100vh-160px)] flex flex-col",children:[(0,t.jsx)("div",{className:"border-b px-4 py-2",children:(0,t.jsxs)("div",{className:"flex flex-wrap items-center justify-between gap-3",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)("span",{className:"text-sm font-medium text-gray-600",children:"Virtual Key Source"}),(0,t.jsxs)(p.Select,{value:B,onChange:e=>V(e),disabled:a,className:"w-48",children:[(0,t.jsx)(p.Select.Option,{value:"session",disabled:!el,children:"Current UI Session"}),(0,t.jsx)(p.Select.Option,{value:"custom",children:"Virtual Key"})]}),"custom"===B&&(0,t.jsx)(c.Input.Password,{value:Y,onChange:e=>X(e.target.value),placeholder:"Enter Virtual Key",className:"w-56"})]}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)("span",{className:"text-sm font-medium text-gray-600",children:"Endpoint"}),(0,t.jsx)(p.Select,{value:E,onChange:e=>T(e),className:"w-56",children:Object.values(G).map(e=>({value:e.id,label:e.label})).map(e=>(0,t.jsx)(p.Select.Option,{value:e.value,children:e.label},e.value))})]}),(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)(d.Button,{onClick:()=>{b(e=>e.map(e=>({...e,messages:[],traceId:void 0,isLoading:!1}))),$(""),er()},disabled:!ep,icon:(0,t.jsx)(r.ClearOutlined,{}),children:"Clear All Chats"}),(0,t.jsx)(u.Tooltip,{title:y.length>=3?"Compare up to 3 models at a time":"Add another comparison",children:(0,t.jsx)(d.Button,{onClick:()=>{if(y.length>=3)return;let e=j[y.length%(j.length||1)]??"",t=w[y.length%(w.length||1)]?.agent_name??"",s={id:Date.now().toString(),model:e,agent:t,messages:[],isLoading:!1,tags:[],mcpTools:[],vectorStores:[],guardrails:[],temperature:1,maxTokens:2048,applyAcrossModels:!1,useAdvancedParams:!1};b(e=>[...e,s])},disabled:y.length>=3,icon:(0,t.jsx)(l.PlusOutlined,{}),children:"Add Comparison"})})]})]})}),(0,t.jsx)("div",{className:"grid flex-1 min-h-0 auto-rows-[minmax(0,1fr)]",style:{gridTemplateColumns:`repeat(${y.length}, minmax(0, 1fr))`},children:y.map(e=>(0,t.jsx)(K,{comparison:e,onUpdate:(t,s)=>{var a;return a=e.id,void b(e=>{if(s?.applyToAll&&s.keysToApply?.length){let i={};s.keysToApply.forEach(e=>{let s=t[e];void 0!==s&&(i[e]=Array.isArray(s)?[...s]:s)});let r=Object.keys(i).length>0;return e.map(e=>e.id===a?{...e,...t}:r?{...e,...i}:e)}return e.map(e=>e.id===a?{...e,...t}:e)})},onRemove:()=>{var t;return t=e.id,void(y.length>1&&b(e=>e.filter(e=>e.id!==t)))},canRemove:y.length>1,selectorOptions:P,isLoadingOptions:M,endpointConfig:I,apiKey:ea},e.id))}),(0,t.jsx)("div",{className:"flex justify-center pb-4",children:(0,t.jsx)("div",{className:"w-full max-w-3xl px-4",children:(0,t.jsxs)("div",{className:"border border-gray-200 shadow-lg rounded-xl bg-white p-4",children:[(0,t.jsx)("div",{className:"flex items-center justify-between gap-4 mb-3 min-h-8",children:em?(0,t.jsx)("span",{className:"text-sm text-gray-500",children:"Attachment ready to send"}):eh?(0,t.jsx)("div",{className:"flex items-center gap-2 overflow-x-auto",children:Q.map(e=>(0,t.jsx)("button",{type:"button",onClick:()=>ec(e),className:"shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-100 cursor-pointer",children:e},e))}):ei&&!em?(0,t.jsx)("div",{className:"flex items-center gap-2 overflow-x-auto",children:Z.map(e=>(0,t.jsx)("button",{type:"button",onClick:()=>ec(e),className:"shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-100 cursor-pointer",children:e},e))}):eu?(0,t.jsxs)("span",{className:"flex items-center gap-2 text-sm text-gray-500",children:[(0,t.jsx)("span",{className:"h-2 w-2 rounded-full bg-blue-500 animate-pulse","aria-hidden":!0}),I.loadingMessage]}):(0,t.jsx)("span",{className:"text-sm text-gray-500",children:I.inputPlaceholder})}),z&&(0,t.jsx)("div",{className:"mb-3",children:(0,t.jsxs)("div",{className:"flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsx)("div",{className:"relative inline-block",children:eg?(0,t.jsx)("div",{className:"w-10 h-10 rounded-md bg-red-500 flex items-center justify-center",children:(0,t.jsx)(o.FilePdfOutlined,{style:{fontSize:"16px",color:"white"}})}):(0,t.jsx)("img",{src:D||"",alt:"Upload preview",className:"w-10 h-10 rounded-md border border-gray-200 object-cover"})}),(0,t.jsxs)("div",{className:"flex-1 min-w-0",children:[(0,t.jsx)("div",{className:"text-sm font-medium text-gray-900 truncate",children:z.name}),(0,t.jsx)("div",{className:"text-xs text-gray-500",children:eg?"PDF":"Image"})]}),(0,t.jsx)("button",{className:"flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors",onClick:er,children:(0,t.jsx)(n.DeleteOutlined,{style:{fontSize:"12px"}})})]})}),(0,t.jsx)(J,{value:L,onChange:e=>{$(e)},onSend:()=>{ed(L)},disabled:0===y.length||y.every(e=>e.isLoading),hasAttachment:em,uploadComponent:(0,t.jsx)(g.default,{chatUploadedImage:z,chatImagePreviewUrl:D,onImageUpload:e=>(D&&URL.revokeObjectURL(D),F(e),U(URL.createObjectURL(e)),!1),onRemoveImage:er})})]})})})]})})}var et=e.i(653824),es=e.i(881073),ea=e.i(197647),ei=e.i(723731),er=e.i(404206),en=e.i(135214),eo=e.i(62478);function el(){let{accessToken:e,userRole:i,userId:r,disabledPersonalKeyCreation:n,token:o}=(0,en.default)(),[l,d]=(0,s.useState)(void 0);return(0,s.useEffect)(()=>{(async()=>{if(e){let t=await (0,eo.fetchProxySettings)(e);t&&d({PROXY_BASE_URL:t.PROXY_BASE_URL,LITELLM_UI_API_DOC_BASE_URL:t.LITELLM_UI_API_DOC_BASE_URL})}})()},[e]),(0,t.jsxs)(et.TabGroup,{className:"h-full w-full",children:[(0,t.jsxs)(es.TabList,{className:"mb-0",children:[(0,t.jsx)(ea.Tab,{children:"Chat"}),(0,t.jsx)(ea.Tab,{children:"Compare"})]}),(0,t.jsxs)(ei.TabPanels,{className:"h-full",children:[(0,t.jsx)(er.TabPanel,{className:"h-full",children:(0,t.jsx)(a.default,{accessToken:e,token:o,userRole:i,userID:r,disabledPersonalKeyCreation:n,proxySettings:l})}),(0,t.jsx)(er.TabPanel,{className:"h-full",children:(0,t.jsx)(ee,{accessToken:e,disabledPersonalKeyCreation:n})})]})]})}e.s(["default",()=>el],213970)}]);