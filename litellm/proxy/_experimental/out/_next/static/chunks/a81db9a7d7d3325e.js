(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,434166,e=>{"use strict";function t(e,t){window.sessionStorage.setItem(e,btoa(encodeURIComponent(t).replace(/%([0-9A-F]{2})/g,(e,t)=>String.fromCharCode(parseInt(t,16)))))}function o(e){try{let t=window.sessionStorage.getItem(e);if(null===t)return null;return decodeURIComponent(atob(t).split("").map(e=>"%"+e.charCodeAt(0).toString(16).padStart(2,"0")).join(""))}catch{return null}}e.s(["getSecureItem",()=>o,"setSecureItem",()=>t])},516015,(e,t,o)=>{},898547,(e,t,o)=>{var i=e.i(247167);e.r(516015);var n=e.r(271645),s=n&&"object"==typeof n&&"default"in n?n:{default:n},a=void 0!==i.default&&i.default.env&&!0,r=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,o=t.name,i=void 0===o?"stylesheet":o,n=t.optimizeForSpeed,s=void 0===n?a:n;c(r(i),"`name` must be a string"),this._name=i,this._deletedRulePlaceholder="#"+i+"-deleted-rule____{}",c("boolean"==typeof s,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=s,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,o=e.prototype;return o.setOptimizeForSpeed=function(e){c("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),c(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},o.isOptimizeForSpeed=function(){return this._optimizeForSpeed},o.inject=function(){var e=this;if(c(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(a||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,o){return"number"==typeof o?e._serverSheet.cssRules[o]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),o},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},o.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},o.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},o.insertRule=function(e,t){if(c(r(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var o=this.getSheet();"number"!=typeof t&&(t=o.cssRules.length);try{o.insertRule(e,t)}catch(t){return a||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var i=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,i))}return this._rulesCount++},o.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var o="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!o.cssRules[e])return e;o.deleteRule(e);try{o.insertRule(t,e)}catch(i){a||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),o.insertRule(this._deletedRulePlaceholder,e)}}else{var i=this._tags[e];c(i,"old rule at index `"+e+"` not found"),i.textContent=t}return e},o.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];c(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},o.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},o.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,o){return o?t=t.concat(Array.prototype.map.call(e.getSheetForTag(o).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},o.makeStyleTag=function(e,t,o){t&&c(r(t),"makeStyleTag accepts only strings as second parameter");var i=document.createElement("style");this._nonce&&i.setAttribute("nonce",this._nonce),i.type="text/css",i.setAttribute("data-"+e,""),t&&i.appendChild(document.createTextNode(t));var n=document.head||document.getElementsByTagName("head")[0];return o?n.insertBefore(i,o):n.appendChild(i),i},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var o=0;o<t.length;o++){var i=t[o];i.enumerable=i.enumerable||!1,i.configurable=!0,"value"in i&&(i.writable=!0),Object.defineProperty(e,i.key,i)}}(e.prototype,t),e}();function c(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var d=function(e){for(var t=5381,o=e.length;o;)t=33*t^e.charCodeAt(--o);return t>>>0},p={};function m(e,t){if(!t)return"jsx-"+e;var o=String(t),i=e+o;return p[i]||(p[i]="jsx-"+d(e+"-"+o)),p[i]}function u(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var o=e+t;return p[o]||(p[o]=t.replace(/__jsx-style-dynamic-selector/g,e)),p[o]}var f=function(){function e(e){var t=void 0===e?{}:e,o=t.styleSheet,i=void 0===o?null:o,n=t.optimizeForSpeed,s=void 0!==n&&n;this._sheet=i||new l({name:"styled-jsx",optimizeForSpeed:s}),this._sheet.inject(),i&&"boolean"==typeof s&&(this._sheet.setOptimizeForSpeed(s),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var o=this.getIdAndRules(e),i=o.styleId,n=o.rules;if(i in this._instancesCounts){this._instancesCounts[i]+=1;return}var s=n.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[i]=s,this._instancesCounts[i]=1},t.remove=function(e){var t=this,o=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(o in this._instancesCounts,"styleId: `"+o+"` not found"),this._instancesCounts[o]-=1,this._instancesCounts[o]<1){var i=this._fromServer&&this._fromServer[o];i?(i.parentNode.removeChild(i),delete this._fromServer[o]):(this._indices[o].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[o]),delete this._instancesCounts[o]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],o=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return o[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,o;return t=this.cssRules(),void 0===(o=e)&&(o={}),t.map(function(e){var t=e[0],i=e[1];return s.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:o.nonce?o.nonce:void 0,dangerouslySetInnerHTML:{__html:i}})})},t.getIdAndRules=function(e){var t=e.children,o=e.dynamic,i=e.id;if(o){var n=m(i,o);return{styleId:n,rules:Array.isArray(t)?t.map(function(e){return u(n,e)}):[u(n,t)]}}return{styleId:m(i),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),g=n.createContext(null);function h(){return new f}function _(){return n.useContext(g)}g.displayName="StyleSheetContext";var b=s.default.useInsertionEffect||s.default.useLayoutEffect,x="u">typeof window?h():void 0;function v(e){var t=x||_();return t&&("u"<typeof window?t.add(e):b(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}v.dynamic=function(e){return e.map(function(e){return m(e[0],e[1])}).join(" ")},o.StyleRegistry=function(e){var t=e.registry,o=e.children,i=n.useContext(g),a=n.useState(function(){return i||t||h()})[0];return s.default.createElement(g.Provider,{value:a},o)},o.createStyleRegistry=h,o.style=v,o.useStyleRegistry=_},437902,(e,t,o)=>{t.exports=e.r(898547).style},254530,452598,e=>{"use strict";e.i(247167);var t=e.i(356449),o=e.i(764205);async function i(e,i,n,s,a,r,l,c,d,p,m,u,f,g,h,_,b,x,v,y,j,w,S,k,N){console.log=function(){},console.log("isLocal:",!1);let C=y||(0,o.getProxyBaseUrl)(),z={};a&&a.length>0&&(z["x-litellm-tags"]=a.join(","));let I=new t.default.OpenAI({apiKey:s,baseURL:C,dangerouslyAllowBrowser:!0,defaultHeaders:z});try{let t,o=Date.now(),s=!1,a={},y=!1,C=[];for await(let v of(g&&g.length>0&&(g.includes("__all__")?C.push({type:"mcp",server_label:"litellm",server_url:"litellm_proxy/mcp",require_approval:"never"}):g.forEach(e=>{if(e.startsWith("toolset:")){let t=e.slice(8),o=N?.find(e=>e.toolset_id===t),i=o?.toolset_name||t;C.push({type:"mcp",server_label:i,server_url:`litellm_proxy/mcp/${encodeURIComponent(i)}`,require_approval:"never"})}else{let t=j?.find(t=>t.server_id===e),o=t?.alias||t?.server_name||e,i=w?.[e]||[];C.push({type:"mcp",server_label:"litellm",server_url:`litellm_proxy/mcp/${o}`,require_approval:"never",...i.length>0?{allowed_tools:i}:{}})}})),await I.chat.completions.create({model:n,stream:!0,stream_options:{include_usage:!0},litellm_trace_id:p,messages:e,...m?{vector_store_ids:m}:{},...u?{guardrails:u}:{},...f?{policies:f}:{},...C.length>0?{tools:C,tool_choice:"auto"}:{},...void 0!==b?{temperature:b}:{},...void 0!==x?{max_tokens:x}:{},...k?{mock_testing_fallbacks:!0}:{}},{signal:r}))){console.log("Stream chunk:",v);let e=v.choices[0]?.delta;if(console.log("Delta content:",v.choices[0]?.delta?.content),console.log("Delta reasoning content:",e?.reasoning_content),!s&&(v.choices[0]?.delta?.content||e&&e.reasoning_content)&&(s=!0,t=Date.now()-o,console.log("First token received! Time:",t,"ms"),c?(console.log("Calling onTimingData with:",t),c(t)):console.log("onTimingData callback is not defined!")),v.choices[0]?.delta?.content){let e=v.choices[0].delta.content;i(e,v.model)}if(e&&e.image&&h&&(console.log("Image generated:",e.image),h(e.image.url,v.model)),e&&e.reasoning_content){let t=e.reasoning_content;l&&l(t)}if(e&&e.provider_specific_fields?.search_results&&_&&(console.log("Search results found:",e.provider_specific_fields.search_results),_(e.provider_specific_fields.search_results)),e&&e.provider_specific_fields){let t=e.provider_specific_fields;if(t.mcp_list_tools&&!a.mcp_list_tools&&(a.mcp_list_tools=t.mcp_list_tools,S&&!y)){y=!0;let e={type:"response.output_item.done",item_id:"mcp_list_tools",item:{type:"mcp_list_tools",tools:t.mcp_list_tools.map(e=>({name:e.function?.name||e.name||"",description:e.function?.description||e.description||"",input_schema:e.function?.parameters||e.input_schema||{}}))},timestamp:Date.now()};S(e),console.log("MCP list_tools event sent:",e)}t.mcp_tool_calls&&(a.mcp_tool_calls=t.mcp_tool_calls),t.mcp_call_results&&(a.mcp_call_results=t.mcp_call_results),(t.mcp_list_tools||t.mcp_tool_calls||t.mcp_call_results)&&console.log("MCP metadata found in chunk:",{mcp_list_tools:t.mcp_list_tools?"present":"absent",mcp_tool_calls:t.mcp_tool_calls?"present":"absent",mcp_call_results:t.mcp_call_results?"present":"absent"})}if(v.usage&&d){console.log("Usage data found:",v.usage);let e={completionTokens:v.usage.completion_tokens,promptTokens:v.usage.prompt_tokens,totalTokens:v.usage.total_tokens};v.usage.completion_tokens_details?.reasoning_tokens&&(e.reasoningTokens=v.usage.completion_tokens_details.reasoning_tokens),void 0!==v.usage.cost&&null!==v.usage.cost&&(e.cost=parseFloat(v.usage.cost)),d(e)}}S&&(a.mcp_tool_calls||a.mcp_call_results)&&a.mcp_tool_calls&&a.mcp_tool_calls.length>0&&a.mcp_tool_calls.forEach((e,t)=>{let o=e.function?.name||e.name||"",i=e.function?.arguments||e.arguments||"{}",n=a.mcp_call_results?.find(t=>t.tool_call_id===e.id||t.tool_call_id===e.call_id)||a.mcp_call_results?.[t],s={type:"response.output_item.done",item:{type:"mcp_call",name:o,arguments:"string"==typeof i?i:JSON.stringify(i),output:n?.result?"string"==typeof n.result?n.result:JSON.stringify(n.result):void 0},item_id:e.id||e.call_id,timestamp:Date.now()};S(s),console.log("MCP call event sent:",s)});let z=Date.now();v&&v(z-o)}catch(e){throw r?.aborted&&console.log("Chat completion request was cancelled"),e}}e.s(["makeOpenAIChatCompletionRequest",()=>i],254530);var n=e.i(727749);async function s(e,i,a,r,l=[],c,d,p,m,u,f,g,h,_,b,x,v,y,j,w,S,k,N){if(!r)throw Error("Virtual Key is required");if(!a||""===a.trim())throw Error("Model is required. Please select a model before sending a request.");console.log=function(){};let C=w||(0,o.getProxyBaseUrl)(),z={};l&&l.length>0&&(z["x-litellm-tags"]=l.join(","));let I=new t.default.OpenAI({apiKey:r,baseURL:C,dangerouslyAllowBrowser:!0,defaultHeaders:z});try{let t=Date.now(),o=!1,n=e.map(e=>(Array.isArray(e.content),{role:e.role,content:e.content,type:"message"})),s=[];_&&_.length>0&&(_.includes("__all__")?s.push({type:"mcp",server_label:"litellm",server_url:`${C}/mcp`,require_approval:"never"}):_.forEach(e=>{if(e.startsWith("toolset:")){let t=e.slice(8),o=N?.find(e=>e.toolset_id===t),i=o?.toolset_name||t;s.push({type:"mcp",server_label:i,server_url:`${C}/mcp/${encodeURIComponent(i)}`,require_approval:"never"})}else{let t=S?.find(t=>t.server_id===e),o=t?.server_name||e,i=k?.[e]||[];s.push({type:"mcp",server_label:o,server_url:`${C}/mcp/${encodeURIComponent(o)}`,require_approval:"never",...i.length>0?{allowed_tools:i}:{}})}})),y&&s.push({type:"code_interpreter",container:{type:"auto"}});let r=await I.responses.create({model:a,input:n,stream:!0,litellm_trace_id:u,...b?{previous_response_id:b}:{},...f?{vector_store_ids:f}:{},...g?{guardrails:g}:{},...h?{policies:h}:{},...s.length>0?{tools:s,tool_choice:"auto"}:{}},{signal:c}),l="",w={code:"",containerId:""};for await(let e of r)if(console.log("Response event:",e),"object"==typeof e&&null!==e){if((e.type?.startsWith("response.mcp_")||"response.output_item.done"===e.type&&(e.item?.type==="mcp_list_tools"||e.item?.type==="mcp_call"))&&(console.log("MCP event received:",e),v)){let t={type:e.type,sequence_number:e.sequence_number,output_index:e.output_index,item_id:e.item_id||e.item?.id,item:e.item,delta:e.delta,arguments:e.arguments,timestamp:Date.now()};v(t)}"response.output_item.done"===e.type&&e.item?.type==="mcp_call"&&e.item?.name&&(l=e.item.name,console.log("MCP tool used:",l)),R=w;var R,T=w="response.output_item.done"===e.type&&e.item?.type==="code_interpreter_call"?(console.log("Code interpreter call completed:",e.item),{code:e.item.code||"",containerId:e.item.container_id||""}):R;if("response.output_item.done"===e.type&&e.item?.type==="message"&&e.item?.content&&j){for(let t of e.item.content)if("output_text"===t.type&&t.annotations){let e=t.annotations.filter(e=>"container_file_citation"===e.type);(e.length>0||T.code)&&j({code:T.code,containerId:T.containerId,annotations:e})}}if("response.role.delta"===e.type)continue;if("response.output_text.delta"===e.type&&"string"==typeof e.delta){let n=e.delta;if(console.log("Text delta",n),n.length>0&&(i("assistant",n,a),!o)){o=!0;let e=Date.now()-t;console.log("First token received! Time:",e,"ms"),p&&p(e)}}if("response.reasoning.delta"===e.type&&"delta"in e){let t=e.delta;"string"==typeof t&&d&&d(t)}if("response.completed"===e.type&&"response"in e){let t=e.response,o=t.usage;if(console.log("Usage data:",o),console.log("Response completed event:",t),t.id&&x&&(console.log("Response ID for session management:",t.id),x(t.id)),o&&m){console.log("Usage data:",o);let e={completionTokens:o.output_tokens,promptTokens:o.input_tokens,totalTokens:o.total_tokens};o.completion_tokens_details?.reasoning_tokens&&(e.reasoningTokens=o.completion_tokens_details.reasoning_tokens),m(e,l)}}}return r}catch(e){throw c?.aborted?console.log("Responses API request was cancelled"):n.default.fromBackend(`Error occurred while generating model response. Please try again. Error: ${e}`),e}}e.s(["makeOpenAIResponsesRequest",()=>s],452598)},355343,e=>{"use strict";var t=e.i(843476),o=e.i(437902),i=e.i(898586),n=e.i(362024);let{Text:s}=i.Typography,{Panel:a}=n.Collapse;e.s(["default",0,({events:e,className:i})=>{if(console.log("MCPEventsDisplay: Received events:",e),!e||0===e.length)return console.log("MCPEventsDisplay: No events, returning null"),null;let s=e.find(e=>"response.output_item.done"===e.type&&e.item?.type==="mcp_list_tools"&&e.item.tools&&e.item.tools.length>0),r=e.filter(e=>"response.output_item.done"===e.type&&e.item?.type==="mcp_call");return(console.log("MCPEventsDisplay: toolsEvent:",s),console.log("MCPEventsDisplay: mcpCallEvents:",r),s||0!==r.length)?(0,t.jsxs)("div",{className:`jsx-32b14b04f420f3ac mcp-events-display ${i||""}`,children:[(0,t.jsx)(o.default,{id:"32b14b04f420f3ac",children:".openai-mcp-tools.jsx-32b14b04f420f3ac{margin:0;padding:0;position:relative}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse.jsx-32b14b04f420f3ac,.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-item.jsx-32b14b04f420f3ac{background:0 0!important;border:none!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-header.jsx-32b14b04f420f3ac{color:#9ca3af!important;background:0 0!important;border:none!important;min-height:20px!important;padding:0 0 0 20px!important;font-size:14px!important;font-weight:400!important;line-height:20px!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-header.jsx-32b14b04f420f3ac:hover{color:#6b7280!important;background:0 0!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-content.jsx-32b14b04f420f3ac{background:0 0!important;border:none!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-content-box.jsx-32b14b04f420f3ac{padding:4px 0 0 20px!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-expand-icon.jsx-32b14b04f420f3ac{color:#9ca3af!important;justify-content:center!important;align-items:center!important;width:16px!important;height:16px!important;font-size:10px!important;display:flex!important;position:absolute!important;top:2px!important;left:2px!important}.openai-mcp-tools.jsx-32b14b04f420f3ac .ant-collapse-expand-icon.jsx-32b14b04f420f3ac:hover{color:#6b7280!important}.openai-vertical-line.jsx-32b14b04f420f3ac{opacity:.8;background-color:#f3f4f6;width:.5px;position:absolute;top:18px;bottom:0;left:9px}.tool-item.jsx-32b14b04f420f3ac{color:#4b5563;z-index:1;background:#fff;margin:0;padding:0;font-family:ui-monospace,SFMono-Regular,SF Mono,Monaco,Consolas,Liberation Mono,Courier New,monospace;font-size:13px;line-height:18px;position:relative}.mcp-section.jsx-32b14b04f420f3ac{z-index:1;background:#fff;margin-bottom:12px;position:relative}.mcp-section.jsx-32b14b04f420f3ac:last-child{margin-bottom:0}.mcp-section-header.jsx-32b14b04f420f3ac{color:#6b7280;margin-bottom:4px;font-size:13px;font-weight:500}.mcp-code-block.jsx-32b14b04f420f3ac{background:#f9fafb;border:1px solid #f3f4f6;border-radius:6px;padding:8px;font-size:12px}.mcp-json.jsx-32b14b04f420f3ac{color:#374151;white-space:pre-wrap;word-wrap:break-word;margin:0;font-family:ui-monospace,SFMono-Regular,SF Mono,Monaco,Consolas,Liberation Mono,Courier New,monospace}.mcp-approved.jsx-32b14b04f420f3ac{color:#6b7280;align-items:center;font-size:13px;display:flex}.mcp-checkmark.jsx-32b14b04f420f3ac{color:#10b981;margin-right:6px;font-weight:700}.mcp-response-content.jsx-32b14b04f420f3ac{color:#374151;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,SF Mono,Monaco,Consolas,Liberation Mono,Courier New,monospace;font-size:13px;line-height:1.5}"}),(0,t.jsxs)("div",{className:"jsx-32b14b04f420f3ac openai-mcp-tools",children:[(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac openai-vertical-line"}),(0,t.jsxs)(n.Collapse,{ghost:!0,size:"small",expandIconPosition:"start",defaultActiveKey:s?["list-tools"]:r.map((e,t)=>`mcp-call-${t}`),children:[s&&(0,t.jsx)(a,{header:"List tools",children:(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac",children:s.item?.tools?.map((e,o)=>(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac tool-item",children:e.name},o))})},"list-tools"),r.map((e,o)=>(0,t.jsx)(a,{header:e.item?.name||"Tool call",children:(0,t.jsxs)("div",{className:"jsx-32b14b04f420f3ac",children:[(0,t.jsxs)("div",{className:"jsx-32b14b04f420f3ac mcp-section",children:[(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac mcp-section-header",children:"Request"}),(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac mcp-code-block",children:e.item?.arguments&&(0,t.jsx)("pre",{className:"jsx-32b14b04f420f3ac mcp-json",children:(()=>{try{return JSON.stringify(JSON.parse(e.item.arguments),null,2)}catch(t){return e.item.arguments}})()})})]}),(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac mcp-section",children:(0,t.jsxs)("div",{className:"jsx-32b14b04f420f3ac mcp-approved",children:[(0,t.jsx)("span",{className:"jsx-32b14b04f420f3ac mcp-checkmark",children:"✓"})," Approved"]})}),e.item?.output&&(0,t.jsxs)("div",{className:"jsx-32b14b04f420f3ac mcp-section",children:[(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac mcp-section-header",children:"Response"}),(0,t.jsx)("div",{className:"jsx-32b14b04f420f3ac mcp-response-content",children:e.item.output})]})]})},`mcp-call-${o}`))]})]})]}):(console.log("MCPEventsDisplay: No valid events found, returning null"),null)}])},966988,e=>{"use strict";var t=e.i(843476),o=e.i(271645),i=e.i(464571),n=e.i(918789),s=e.i(650056),a=e.i(219470),r=e.i(755151),l=e.i(240647),c=e.i(812618);e.s(["default",0,({reasoningContent:e})=>{let[d,p]=(0,o.useState)(!0);return e?(0,t.jsxs)("div",{className:"reasoning-content mt-1 mb-2",children:[(0,t.jsxs)(i.Button,{type:"text",className:"flex items-center text-xs text-gray-500 hover:text-gray-700",onClick:()=>p(!d),icon:(0,t.jsx)(c.BulbOutlined,{}),children:[d?"Hide reasoning":"Show reasoning",d?(0,t.jsx)(r.DownOutlined,{className:"ml-1"}):(0,t.jsx)(l.RightOutlined,{className:"ml-1"})]}),d&&(0,t.jsx)("div",{className:"mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md text-sm text-gray-700",children:(0,t.jsx)(n.default,{components:{code({node:e,inline:o,className:i,children:n,...r}){let l=/language-(\w+)/.exec(i||"");return!o&&l?(0,t.jsx)(s.Prism,{style:a.coy,language:l[1],PreTag:"div",className:"rounded-md my-2",...r,children:String(n).replace(/\n$/,"")}):(0,t.jsx)("code",{className:`${i} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`,...r,children:n})}},children:e})})]}):null}])},84899,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645),i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M931.4 498.9L94.9 79.5c-3.4-1.7-7.3-2.1-11-1.2a15.99 15.99 0 00-11.7 19.3l86.2 352.2c1.3 5.3 5.2 9.6 10.4 11.3l147.7 50.7-147.6 50.7c-5.2 1.8-9.1 6-10.3 11.3L72.2 926.5c-.9 3.7-.5 7.6 1.2 10.9 3.9 7.9 13.5 11.1 21.5 7.2l836.5-417c3.1-1.5 5.6-4.1 7.2-7.1 3.9-8 .7-17.6-7.2-21.6zM170.8 826.3l50.3-205.6 295.2-101.3c2.3-.8 4.2-2.6 5-5 1.4-4.2-.8-8.7-5-10.2L221.1 403 171 198.2l628 314.9-628.2 313.2z"}}]},name:"send",theme:"outlined"},n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["SendOutlined",0,s],84899)},518617,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm0 76c-205.4 0-372 166.6-372 372s166.6 372 372 372 372-166.6 372-372-166.6-372-372-372zm128.01 198.83c.03 0 .05.01.09.06l45.02 45.01a.2.2 0 01.05.09.12.12 0 010 .07c0 .02-.01.04-.05.08L557.25 512l127.87 127.86a.27.27 0 01.05.06v.02a.12.12 0 010 .07c0 .03-.01.05-.05.09l-45.02 45.02a.2.2 0 01-.09.05.12.12 0 01-.07 0c-.02 0-.04-.01-.08-.05L512 557.25 384.14 685.12c-.04.04-.06.05-.08.05a.12.12 0 01-.07 0c-.03 0-.05-.01-.09-.05l-45.02-45.02a.2.2 0 01-.05-.09.12.12 0 010-.07c0-.02.01-.04.06-.08L466.75 512 338.88 384.14a.27.27 0 01-.05-.06l-.01-.02a.12.12 0 010-.07c0-.03.01-.05.05-.09l45.02-45.02a.2.2 0 01.09-.05.12.12 0 01.07 0c.02 0 .04.01.08.06L512 466.75l127.86-127.86c.04-.05.06-.06.08-.06a.12.12 0 01.07 0z"}}]},name:"close-circle",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["CloseCircleOutlined",0,s],518617)},149192,e=>{"use strict";var t=e.i(864517);e.s(["CloseOutlined",()=>t.default])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["CheckCircleOutlined",0,s],245704)},782273,793916,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M625.9 115c-5.9 0-11.9 1.6-17.4 5.3L254 352H90c-8.8 0-16 7.2-16 16v288c0 8.8 7.2 16 16 16h164l354.5 231.7c5.5 3.6 11.6 5.3 17.4 5.3 16.7 0 32.1-13.3 32.1-32.1V147.1c0-18.8-15.4-32.1-32.1-32.1zM586 803L293.4 611.7l-18-11.7H146V424h129.4l17.9-11.7L586 221v582zm348-327H806c-8.8 0-16 7.2-16 16v40c0 8.8 7.2 16 16 16h128c8.8 0 16-7.2 16-16v-40c0-8.8-7.2-16-16-16zm-41.9 261.8l-110.3-63.7a15.9 15.9 0 00-21.7 5.9l-19.9 34.5c-4.4 7.6-1.8 17.4 5.8 21.8L856.3 800a15.9 15.9 0 0021.7-5.9l19.9-34.5c4.4-7.6 1.7-17.4-5.8-21.8zM760 344a15.9 15.9 0 0021.7 5.9L892 286.2c7.6-4.4 10.2-14.2 5.8-21.8L878 230a15.9 15.9 0 00-21.7-5.9L746 287.8a15.99 15.99 0 00-5.8 21.8L760 344z"}}]},name:"sound",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["SoundOutlined",0,s],782273);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M842 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 140.3-113.7 254-254 254S258 594.3 258 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 168.7 126.6 307.9 290 327.6V884H326.7c-13.7 0-24.7 14.3-24.7 32v36c0 4.4 2.8 8 6.2 8h407.6c3.4 0 6.2-3.6 6.2-8v-36c0-17.7-11-32-24.7-32H548V782.1c165.3-18 294-158 294-328.1zM512 624c93.9 0 170-75.2 170-168V232c0-92.8-76.1-168-170-168s-170 75.2-170 168v224c0 92.8 76.1 168 170 168zm-94-392c0-50.6 41.9-92 94-92s94 41.4 94 92v224c0 50.6-41.9 92-94 92s-94-41.4-94-92V232z"}}]},name:"audio",theme:"outlined"};var r=o.forwardRef(function(e,i){return o.createElement(n.default,(0,t.default)({},e,{ref:i,icon:a}))});e.s(["AudioOutlined",0,r],793916)},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["ArrowLeftOutlined",0,s],447566)},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["LinkOutlined",0,s],596239)},190272,785913,e=>{"use strict";var t,o,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((o={}).IMAGE="image",o.VIDEO="video",o.CHAT="chat",o.RESPONSES="responses",o.IMAGE_EDITS="image_edits",o.ANTHROPIC_MESSAGES="anthropic_messages",o.EMBEDDINGS="embeddings",o.SPEECH="speech",o.TRANSCRIPTION="transcription",o.A2A_AGENTS="a2a_agents",o.MCP="mcp",o.REALTIME="realtime",o);let s={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=s[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:o,accessToken:i,apiKey:s,inputMessage:a,chatHistory:r,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:p,selectedMCPServers:m,mcpServers:u,mcpServerToolRestrictions:f,selectedVoice:g,endpointType:h,selectedModel:_,selectedSdk:b,proxySettings:x}=e,v="session"===o?i:s,y=window.location.origin,j=x?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?y=j:x?.PROXY_BASE_URL&&(y=x.PROXY_BASE_URL);let w=a||"Your prompt here",S=w.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),k=r.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),N={};l.length>0&&(N.tags=l),c.length>0&&(N.vector_stores=c),d.length>0&&(N.guardrails=d),p.length>0&&(N.policies=p);let C=_||"your-model-name",z="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case n.CHAT:{let e=Object.keys(N).length>0,o="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let i=k.length>0?k:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(i,null,4)}${o}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${C}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${S}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${o}
# )
# print(response_with_file)
`;break}case n.RESPONSES:{let e=Object.keys(N).length>0,o="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let i=k.length>0?k:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(i,null,4)}${o}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${C}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${S}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${o}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${C}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case n.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${C}",
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
`;break;case n.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case n.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${a||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${z}
${t}`}],190272)},611052,e=>{"use strict";var t=e.i(843476),o=e.i(271645),i=e.i(212931),n=e.i(311451),s=e.i(790848),a=e.i(888259),r=e.i(438957);e.i(247167);var l=e.i(931067);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240zm460 600H232V536h560v304zM484 701v53c0 4.4 3.6 8 8 8h40c4.4 0 8-3.6 8-8v-53a48.01 48.01 0 10-56 0z"}}]},name:"lock",theme:"outlined"};var d=e.i(9583),p=o.forwardRef(function(e,t){return o.createElement(d.default,(0,l.default)({},e,{ref:t,icon:c}))}),m=e.i(492030),u=e.i(266537),f=e.i(447566),g=e.i(149192),h=e.i(596239);e.s(["ByokCredentialModal",0,({server:e,open:l,onClose:c,onSuccess:d,accessToken:_})=>{let[b,x]=(0,o.useState)(1),[v,y]=(0,o.useState)(""),[j,w]=(0,o.useState)(!0),[S,k]=(0,o.useState)(!1),N=e.alias||e.server_name||"Service",C=N.charAt(0).toUpperCase(),z=()=>{x(1),y(""),w(!0),k(!1),c()},I=async()=>{if(!v.trim())return void a.default.error("Please enter your API key");k(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${_}`},body:JSON.stringify({credential:v.trim(),save:j})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}a.default.success(`Connected to ${N}`),d(e.server_id),z()}catch(e){a.default.error(e.message||"Failed to connect")}finally{k(!1)}};return(0,t.jsx)(i.Modal,{open:l,onCancel:z,footer:null,width:480,closeIcon:null,className:"byok-modal",children:(0,t.jsxs)("div",{className:"relative p-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===b?(0,t.jsxs)("button",{onClick:()=>x(1),className:"flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm",children:[(0,t.jsx)(f.ArrowLeftOutlined,{})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${1===b?"bg-blue-500":"bg-gray-300"}`}),(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${2===b?"bg-blue-500":"bg-gray-300"}`})]}),(0,t.jsx)("button",{onClick:z,className:"text-gray-400 hover:text-gray-600",children:(0,t.jsx)(g.CloseOutlined,{})})]}),1===b?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(u.ArrowRightOutlined,{className:"text-gray-400 text-lg"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:C})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:["Connect ",N]}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["LiteLLM needs access to ",N," to complete your request."]}),(0,t.jsx)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-gray-800 mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-gray-500 text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to"," ",N,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-green-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,o)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-gray-700",children:[(0,t.jsx)(m.CheckOutlined,{className:"text-green-500 flex-shrink-0"}),e]},o))})]}),(0,t.jsxs)("button",{onClick:()=>x(2),className:"w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:["Continue to Authentication ",(0,t.jsx)(u.ArrowRightOutlined,{})]}),(0,t.jsx)("button",{onClick:z,className:"mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-4",children:(0,t.jsx)(r.KeyOutlined,{className:"text-blue-400 text-xl"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["Enter your ",N," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsxs)("label",{className:"block text-sm font-semibold text-gray-800 mb-2",children:[N," API Key"]}),(0,t.jsx)(n.Input.Password,{placeholder:"Enter your API key",value:v,onChange:e=>y(e.target.value),size:"large",className:"rounded-lg"}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-500 hover:text-blue-700 text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key? ",(0,t.jsx)(h.LinkOutlined,{})]})]}),(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-gray-800",children:"Save key for future use"})]}),(0,t.jsx)(s.Switch,{checked:j,onChange:w})]}),(0,t.jsxs)("div",{className:"bg-blue-50 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(p,{className:"text-blue-400 mt-0.5 flex-shrink-0"}),(0,t.jsx)("p",{className:"text-sm text-blue-700",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)("button",{onClick:I,disabled:S,className:"w-full bg-blue-500 hover:bg-blue-600 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:[(0,t.jsx)(p,{})," Connect & Authorize"]})]})]})})}],611052)},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["CodeOutlined",0,s],245094)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["DollarOutlined",0,s],458505)},219470,812618,e=>{"use strict";e.s(["coy",0,{'code[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",maxHeight:"inherit",height:"inherit",padding:"0 1em",display:"block",overflow:"auto"},'pre[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",position:"relative",margin:".5em 0",overflow:"visible",padding:"1px",backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em"},'pre[class*="language-"] > code':{position:"relative",zIndex:"1",borderLeft:"10px solid #358ccb",boxShadow:"-1px 0px 0px 0px #358ccb, 0px 0px 0px 1px #dfdfdf",backgroundColor:"#fdfdfd",backgroundImage:"linear-gradient(transparent 50%, rgba(69, 142, 209, 0.04) 50%)",backgroundSize:"3em 3em",backgroundOrigin:"content-box",backgroundAttachment:"local"},':not(pre) > code[class*="language-"]':{backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em",position:"relative",padding:".2em",borderRadius:"0.3em",color:"#c92c2c",border:"1px solid rgba(0, 0, 0, 0.1)",display:"inline",whiteSpace:"normal"},'pre[class*="language-"]:before':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"0.18em",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(-2deg)",MozTransform:"rotate(-2deg)",msTransform:"rotate(-2deg)",OTransform:"rotate(-2deg)",transform:"rotate(-2deg)"},'pre[class*="language-"]:after':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"auto",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(2deg)",MozTransform:"rotate(2deg)",msTransform:"rotate(2deg)",OTransform:"rotate(2deg)",transform:"rotate(2deg)",right:"0.75em"},comment:{color:"#7D8B99"},"block-comment":{color:"#7D8B99"},prolog:{color:"#7D8B99"},doctype:{color:"#7D8B99"},cdata:{color:"#7D8B99"},punctuation:{color:"#5F6364"},property:{color:"#c92c2c"},tag:{color:"#c92c2c"},boolean:{color:"#c92c2c"},number:{color:"#c92c2c"},"function-name":{color:"#c92c2c"},constant:{color:"#c92c2c"},symbol:{color:"#c92c2c"},deleted:{color:"#c92c2c"},selector:{color:"#2f9c0a"},"attr-name":{color:"#2f9c0a"},string:{color:"#2f9c0a"},char:{color:"#2f9c0a"},function:{color:"#2f9c0a"},builtin:{color:"#2f9c0a"},inserted:{color:"#2f9c0a"},operator:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},entity:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)",cursor:"help"},url:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},variable:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},atrule:{color:"#1990b8"},"attr-value":{color:"#1990b8"},keyword:{color:"#1990b8"},"class-name":{color:"#1990b8"},regex:{color:"#e90"},important:{color:"#e90",fontWeight:"normal"},".language-css .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},".style .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:".7"},'pre[class*="language-"].line-numbers.line-numbers':{paddingLeft:"0"},'pre[class*="language-"].line-numbers.line-numbers code':{paddingLeft:"3.8em"},'pre[class*="language-"].line-numbers.line-numbers .line-numbers-rows':{left:"0"},'pre[class*="language-"][data-line]':{paddingTop:"0",paddingBottom:"0",paddingLeft:"0"},"pre[data-line] code":{position:"relative",paddingLeft:"4em"},"pre .line-highlight":{marginTop:"0"}}],219470),e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M632 888H392c-4.4 0-8 3.6-8 8v32c0 17.7 14.3 32 32 32h192c17.7 0 32-14.3 32-32v-32c0-4.4-3.6-8-8-8zM512 64c-181.1 0-328 146.9-328 328 0 121.4 66 227.4 164 284.1V792c0 17.7 14.3 32 32 32h264c17.7 0 32-14.3 32-32V676.1c98-56.7 164-162.7 164-284.1 0-181.1-146.9-328-328-328zm127.9 549.8L604 634.6V752H420V634.6l-35.9-20.8C305.4 568.3 256 484.5 256 392c0-141.4 114.6-256 256-256s256 114.6 256 256c0 92.5-49.4 176.3-128.1 221.8z"}}]},name:"bulb",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["BulbOutlined",0,s],812618)},132104,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M868 545.5L536.1 163a31.96 31.96 0 00-48.3 0L156 545.5a7.97 7.97 0 006 13.2h81c4.6 0 9-2 12.1-5.5L474 300.9V864c0 4.4 3.6 8 8 8h60c4.4 0 8-3.6 8-8V300.9l218.9 252.3c3 3.5 7.4 5.5 12.1 5.5h81c6.8 0 10.5-8 6-13.2z"}}]},name:"arrow-up",theme:"outlined"};var n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["ArrowUpOutlined",0,s],132104)},447593,989022,e=>{"use strict";e.i(247167);var t=e.i(931067),o=e.i(271645),i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6a25.95 25.95 0 0025.6 30.4h723c1.5 0 3-.1 4.4-.4a25.88 25.88 0 0021.2-30zM204 390h272V182h72v208h272v104H204V390zm468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z"}}]},name:"clear",theme:"outlined"},n=e.i(9583),s=o.forwardRef(function(e,s){return o.createElement(n.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["ClearOutlined",0,s],447593);var a=e.i(843476),r=e.i(592968),l=e.i(637235);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 394c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H400V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v236H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h228v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h164c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V394h164zM628 630H400V394h228v236z"}}]},name:"number",theme:"outlined"};var d=o.forwardRef(function(e,i){return o.createElement(n.default,(0,t.default)({},e,{ref:i,icon:c}))});let p={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM653.3 424.6l52.2 52.2a8.01 8.01 0 01-4.7 13.6l-179.4 21c-5.1.6-9.5-3.7-8.9-8.9l21-179.4c.8-6.6 8.9-9.4 13.6-4.7l52.4 52.4 256.2-256.2c3.1-3.1 8.2-3.1 11.3 0l42.4 42.4c3.1 3.1 3.1 8.2 0 11.3L653.3 424.6z"}}]},name:"import",theme:"outlined"};var m=o.forwardRef(function(e,i){return o.createElement(n.default,(0,t.default)({},e,{ref:i,icon:p}))}),u=e.i(872934),f=e.i(812618),g=e.i(366308),h=e.i(458505);e.s(["default",0,({timeToFirstToken:e,totalLatency:t,usage:o,toolName:i})=>e||t||o?(0,a.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3",children:[void 0!==e&&(0,a.jsx)(r.Tooltip,{title:"Time to first token",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["TTFT: ",(e/1e3).toFixed(2),"s"]})]})}),void 0!==t&&(0,a.jsx)(r.Tooltip,{title:"Total latency",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["Total Latency: ",(t/1e3).toFixed(2),"s"]})]})}),o?.promptTokens!==void 0&&(0,a.jsx)(r.Tooltip,{title:"Prompt tokens",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(m,{className:"mr-1"}),(0,a.jsxs)("span",{children:["In: ",o.promptTokens]})]})}),o?.completionTokens!==void 0&&(0,a.jsx)(r.Tooltip,{title:"Completion tokens",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(u.ExportOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["Out: ",o.completionTokens]})]})}),o?.reasoningTokens!==void 0&&(0,a.jsx)(r.Tooltip,{title:"Reasoning tokens",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(f.BulbOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["Reasoning: ",o.reasoningTokens]})]})}),o?.totalTokens!==void 0&&(0,a.jsx)(r.Tooltip,{title:"Total tokens",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(d,{className:"mr-1"}),(0,a.jsxs)("span",{children:["Total: ",o.totalTokens]})]})}),o?.cost!==void 0&&(0,a.jsx)(r.Tooltip,{title:"Cost",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(h.DollarOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["$",o.cost.toFixed(6)]})]})}),i&&(0,a.jsx)(r.Tooltip,{title:"Tool used",children:(0,a.jsxs)("div",{className:"flex items-center",children:[(0,a.jsx)(g.ToolOutlined,{className:"mr-1"}),(0,a.jsxs)("span",{children:["Tool: ",i]})]})})]}):null],989022)}]);