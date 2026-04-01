(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,190272,785913,e=>{"use strict";var t,i,n=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i);let o={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(n).includes(e)){let t=o[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:n,apiKey:o,inputMessage:a,chatHistory:s,selectedTags:l,selectedVectorStores:p,selectedGuardrails:u,selectedPolicies:c,selectedMCPServers:d,mcpServers:m,mcpServerToolRestrictions:f,selectedVoice:g,endpointType:h,selectedModel:_,selectedSdk:y,proxySettings:v}=e,b="session"===i?n:o,S=window.location.origin,x=v?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?S=x:v?.PROXY_BASE_URL&&(S=v.PROXY_BASE_URL);let w=a||"Your prompt here",O=w.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),E=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),p.length>0&&(C.vector_stores=p),u.length>0&&(C.guardrails=u),c.length>0&&(C.policies=c);let j=_||"your-model-name",I="azure"===y?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${S}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${S}"
)`;switch(h){case r.CHAT:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let n=E.length>0?E:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${j}",
    messages=${JSON.stringify(n,null,4)}${i}
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
#                     "text": "${O}"
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
    extra_body=${e}`}let n=E.length>0?E:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${j}",
    input=${JSON.stringify(n,null,4)}${i}
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
#                 {"type": "input_text", "text": "${O}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===y?`
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
	prompt="${a}",
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
prompt = "${O}"

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
`;break;case r.IMAGE_EDITS:t="azure"===y?`
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
prompt = "${O}"

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
prompt = "${O}"

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
	input="${a||"Your string here"}",
	model="${j}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${j}",
	file=audio_file${a?`,
	prompt="${a.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${j}",
	input="${a||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${j}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["LinkOutlined",0,o],596239)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["ClockCircleOutlined",0,o],637235)},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["CheckCircleOutlined",0,o],245704)},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["CodeOutlined",0,o],245094)},516015,(e,t,i)=>{},898547,(e,t,i)=>{var n=e.i(247167);e.r(516015);var r=e.r(271645),o=r&&"object"==typeof r&&"default"in r?r:{default:r},a=void 0!==n.default&&n.default.env&&!0,s=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,i=t.name,n=void 0===i?"stylesheet":i,r=t.optimizeForSpeed,o=void 0===r?a:r;p(s(n),"`name` must be a string"),this._name=n,this._deletedRulePlaceholder="#"+n+"-deleted-rule____{}",p("boolean"==typeof o,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=o,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,i=e.prototype;return i.setOptimizeForSpeed=function(e){p("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),p(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},i.isOptimizeForSpeed=function(){return this._optimizeForSpeed},i.inject=function(){var e=this;if(p(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(a||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,i){return"number"==typeof i?e._serverSheet.cssRules[i]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),i},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},i.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},i.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},i.insertRule=function(e,t){if(p(s(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var i=this.getSheet();"number"!=typeof t&&(t=i.cssRules.length);try{i.insertRule(e,t)}catch(t){return a||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var n=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,n))}return this._rulesCount++},i.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var i="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!i.cssRules[e])return e;i.deleteRule(e);try{i.insertRule(t,e)}catch(n){a||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),i.insertRule(this._deletedRulePlaceholder,e)}}else{var n=this._tags[e];p(n,"old rule at index `"+e+"` not found"),n.textContent=t}return e},i.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];p(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},i.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},i.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,i){return i?t=t.concat(Array.prototype.map.call(e.getSheetForTag(i).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},i.makeStyleTag=function(e,t,i){t&&p(s(t),"makeStyleTag accepts only strings as second parameter");var n=document.createElement("style");this._nonce&&n.setAttribute("nonce",this._nonce),n.type="text/css",n.setAttribute("data-"+e,""),t&&n.appendChild(document.createTextNode(t));var r=document.head||document.getElementsByTagName("head")[0];return i?r.insertBefore(n,i):r.appendChild(n),n},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var i=0;i<t.length;i++){var n=t[i];n.enumerable=n.enumerable||!1,n.configurable=!0,"value"in n&&(n.writable=!0),Object.defineProperty(e,n.key,n)}}(e.prototype,t),e}();function p(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var u=function(e){for(var t=5381,i=e.length;i;)t=33*t^e.charCodeAt(--i);return t>>>0},c={};function d(e,t){if(!t)return"jsx-"+e;var i=String(t),n=e+i;return c[n]||(c[n]="jsx-"+u(e+"-"+i)),c[n]}function m(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var i=e+t;return c[i]||(c[i]=t.replace(/__jsx-style-dynamic-selector/g,e)),c[i]}var f=function(){function e(e){var t=void 0===e?{}:e,i=t.styleSheet,n=void 0===i?null:i,r=t.optimizeForSpeed,o=void 0!==r&&r;this._sheet=n||new l({name:"styled-jsx",optimizeForSpeed:o}),this._sheet.inject(),n&&"boolean"==typeof o&&(this._sheet.setOptimizeForSpeed(o),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var i=this.getIdAndRules(e),n=i.styleId,r=i.rules;if(n in this._instancesCounts){this._instancesCounts[n]+=1;return}var o=r.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[n]=o,this._instancesCounts[n]=1},t.remove=function(e){var t=this,i=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(i in this._instancesCounts,"styleId: `"+i+"` not found"),this._instancesCounts[i]-=1,this._instancesCounts[i]<1){var n=this._fromServer&&this._fromServer[i];n?(n.parentNode.removeChild(n),delete this._fromServer[i]):(this._indices[i].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[i]),delete this._instancesCounts[i]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],i=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return i[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,i;return t=this.cssRules(),void 0===(i=e)&&(i={}),t.map(function(e){var t=e[0],n=e[1];return o.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:i.nonce?i.nonce:void 0,dangerouslySetInnerHTML:{__html:n}})})},t.getIdAndRules=function(e){var t=e.children,i=e.dynamic,n=e.id;if(i){var r=d(n,i);return{styleId:r,rules:Array.isArray(t)?t.map(function(e){return m(r,e)}):[m(r,t)]}}return{styleId:d(n),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),g=r.createContext(null);function h(){return new f}function _(){return r.useContext(g)}g.displayName="StyleSheetContext";var y=o.default.useInsertionEffect||o.default.useLayoutEffect,v="u">typeof window?h():void 0;function b(e){var t=v||_();return t&&("u"<typeof window?t.add(e):y(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}b.dynamic=function(e){return e.map(function(e){return d(e[0],e[1])}).join(" ")},i.StyleRegistry=function(e){var t=e.registry,i=e.children,n=r.useContext(g),a=r.useState(function(){return n||t||h()})[0];return o.default.createElement(g.Provider,{value:a},i)},i.createStyleRegistry=h,i.style=b,i.useStyleRegistry=_},437902,(e,t,i)=>{t.exports=e.r(898547).style},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),n=e.i(914949),r=e.i(404948);let o=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,o],836938);var a=e.i(613541),s=e.i(763731),l=e.i(242064),p=e.i(491816);e.i(793154);var u=e.i(880476),c=e.i(183293),d=e.i(717356),m=e.i(320560),f=e.i(307358),g=e.i(246422),h=e.i(838378),_=e.i(617933);let y=(0,g.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:i}=e,n=(0,h.mergeToken)(e,{popoverBg:t,popoverColor:i});return[(e=>{let{componentCls:t,popoverColor:i,titleMinWidth:n,fontWeightStrong:r,innerPadding:o,boxShadowSecondary:a,colorTextHeading:s,borderRadiusLG:l,zIndexPopup:p,titleMarginBottom:u,colorBgElevated:d,popoverBg:f,titleBorderBottom:g,innerContentPadding:h,titlePadding:_}=e;return[{[t]:Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:p,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":d,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:f,backgroundClip:"padding-box",borderRadius:l,boxShadow:a,padding:o},[`${t}-title`]:{minWidth:n,marginBottom:u,color:s,fontWeight:r,borderBottom:g,padding:_},[`${t}-inner-content`]:{color:i,padding:h}})},(0,m.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(n),(e=>{let{componentCls:t}=e;return{[t]:_.PresetColors.map(i=>{let n=e[`${i}6`];return{[`&${t}-${i}`]:{"--antd-arrow-background-color":n,[`${t}-inner`]:{backgroundColor:n},[`${t}-arrow`]:{background:"transparent"}}}})}})(n),(0,d.initZoomMotion)(n,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:i,fontHeight:n,padding:r,wireframe:o,zIndexPopupBase:a,borderRadiusLG:s,marginXS:l,lineType:p,colorSplit:u,paddingSM:c}=e,d=i-n;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:a+30},(0,f.getArrowToken)(e)),(0,m.getArrowOffsetToken)({contentRadius:s,limitVerticalRadius:!0})),{innerPadding:12*!o,titleMarginBottom:o?0:l,titlePadding:o?`${d/2}px ${r}px ${d/2-t}px`:0,titleBorderBottom:o?`${t}px ${p} ${u}`:"none",innerContentPadding:o?`${c}px ${r}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var v=function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&0>t.indexOf(n)&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,n=Object.getOwnPropertySymbols(e);r<n.length;r++)0>t.indexOf(n[r])&&Object.prototype.propertyIsEnumerable.call(e,n[r])&&(i[n[r]]=e[n[r]]);return i};let b=({title:e,content:i,prefixCls:n})=>e||i?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${n}-title`},e),i&&t.createElement("div",{className:`${n}-inner-content`},i)):null,S=e=>{let{hashId:n,prefixCls:r,className:a,style:s,placement:l="top",title:p,content:c,children:d}=e,m=o(p),f=o(c),g=(0,i.default)(n,r,`${r}-pure`,`${r}-placement-${l}`,a);return t.createElement("div",{className:g,style:s},t.createElement("div",{className:`${r}-arrow`}),t.createElement(u.Popup,Object.assign({},e,{className:n,prefixCls:r}),d||t.createElement(b,{prefixCls:r,title:m,content:f})))},x=e=>{let{prefixCls:n,className:r}=e,o=v(e,["prefixCls","className"]),{getPrefixCls:a}=t.useContext(l.ConfigContext),s=a("popover",n),[p,u,c]=y(s);return p(t.createElement(S,Object.assign({},o,{prefixCls:s,hashId:u,className:(0,i.default)(r,c)})))};e.s(["Overlay",0,b,"default",0,x],310730);var w=function(e,t){var i={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&0>t.indexOf(n)&&(i[n]=e[n]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,n=Object.getOwnPropertySymbols(e);r<n.length;r++)0>t.indexOf(n[r])&&Object.prototype.propertyIsEnumerable.call(e,n[r])&&(i[n[r]]=e[n[r]]);return i};let O=t.forwardRef((e,u)=>{var c,d;let{prefixCls:m,title:f,content:g,overlayClassName:h,placement:_="top",trigger:v="hover",children:S,mouseEnterDelay:x=.1,mouseLeaveDelay:O=.1,onOpenChange:E,overlayStyle:C={},styles:j,classNames:I}=e,R=w(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:$,className:z,style:A,classNames:N,styles:k}=(0,l.useComponentConfig)("popover"),T=$("popover",m),[P,F,M]=y(T),D=$(),L=(0,i.default)(h,F,M,z,N.root,null==I?void 0:I.root),H=(0,i.default)(N.body,null==I?void 0:I.body),[B,V]=(0,n.default)(!1,{value:null!=(c=e.open)?c:e.visible,defaultValue:null!=(d=e.defaultOpen)?d:e.defaultVisible}),G=(e,t)=>{V(e,!0),null==E||E(e,t)},U=o(f),q=o(g);return P(t.createElement(p.default,Object.assign({placement:_,trigger:v,mouseEnterDelay:x,mouseLeaveDelay:O},R,{prefixCls:T,classNames:{root:L,body:H},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},k.root),A),C),null==j?void 0:j.root),body:Object.assign(Object.assign({},k.body),null==j?void 0:j.body)},ref:u,open:B,onOpenChange:e=>{G(e)},overlay:U||q?t.createElement(b,{prefixCls:T,title:U,content:q}):null,transitionName:(0,a.getTransitionName)(D,"zoom-big",R.transitionName),"data-popover-inject":!0}),(0,s.cloneElement)(S,{onKeyDown:e=>{var i,n;(0,t.isValidElement)(S)&&(null==(n=null==S?void 0:(i=S.props).onKeyDown)||n.call(i,e)),e.keyCode===r.default.ESC&&G(!1,e)}})))});O._InternalPanelDoNotUseOrYouWillBeFired=x,e.s(["default",0,O],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},872934,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM770.87 199.13l-52.2-52.2a8.01 8.01 0 014.7-13.6l179.4-21c5.1-.6 9.5 3.7 8.9 8.9l-21 179.4c-.8 6.6-8.9 9.4-13.6 4.7l-52.4-52.4-256.2 256.2a8.03 8.03 0 01-11.3 0l-42.4-42.4a8.03 8.03 0 010-11.3l256.1-256.3z"}}]},name:"export",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["ExportOutlined",0,o],872934)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var r=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(r.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["DollarOutlined",0,o],458505)}]);