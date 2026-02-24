(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},107233,37727,e=>{"use strict";var t=e.i(603908);e.s(["Plus",()=>t.default],107233);var r=e.i(841947);e.s(["X",()=>r.default],37727)},983561,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M300 328a60 60 0 10120 0 60 60 0 10-120 0zM852 64H172c-17.7 0-32 14.3-32 32v660c0 17.7 14.3 32 32 32h680c17.7 0 32-14.3 32-32V96c0-17.7-14.3-32-32-32zm-32 660H204V128h616v596zM604 328a60 60 0 10120 0 60 60 0 10-120 0zm250.2 556H169.8c-16.5 0-29.8 14.3-29.8 32v36c0 4.4 3.3 8 7.4 8h729.1c4.1 0 7.4-3.6 7.4-8v-36c.1-17.7-13.2-32-29.7-32zM664 508H360c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h304c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"robot",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["RobotOutlined",0,n],983561)},678784,678745,e=>{"use strict";let t=(0,e.i(475254).default)("check",[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]]);e.s(["default",()=>t],678745),e.s(["CheckIcon",()=>t],678784)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},689020,e=>{"use strict";var t=e.i(764205);let r=async e=>{try{let r=await (0,t.modelHubCall)(e);if(console.log("model_info:",r),r?.data.length>0){let e=r.data.map(e=>({model_group:e.model_group,mode:e?.mode}));return e.sort((e,t)=>e.model_group.localeCompare(t.model_group)),e}return[]}catch(e){throw console.error("Error fetching model info:",e),e}};e.s(["fetchAvailableModels",0,r])},603908,e=>{"use strict";let t=(0,e.i(475254).default)("plus",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]]);e.s(["default",()=>t])},841947,e=>{"use strict";let t=(0,e.i(475254).default)("x",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]]);e.s(["default",()=>t])},916940,e=>{"use strict";var t=e.i(843476),r=e.i(271645),i=e.i(199133),o=e.i(764205);e.s(["default",0,({onChange:e,value:n,className:a,accessToken:s,placeholder:l="Select vector stores",disabled:c=!1})=>{let[u,d]=(0,r.useState)([]),[p,h]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(s){h(!0);try{let e=await (0,o.vectorStoreListCall)(s);e.data&&d(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{h(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(i.Select,{mode:"multiple",placeholder:l,onChange:e,value:n,loading:p,className:a,allowClear:!0,options:u.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,title:e.vector_store_description||e.vector_store_id})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"},disabled:c})})}])},737434,e=>{"use strict";var t=e.i(184163);e.s(["DownloadOutlined",()=>t.default])},59935,(e,t,r)=>{var i;let o;e.e,i=function e(){var t,r="u">typeof self?self:"u">typeof window?window:void 0!==r?r:{},i=!r.document&&!!r.postMessage,o=r.IS_PAPA_WORKER||!1,n={},a=0,s={};function l(e){this._handle=null,this._finished=!1,this._completed=!1,this._halted=!1,this._input=null,this._baseIndex=0,this._partialLine="",this._rowCount=0,this._start=0,this._nextChunk=null,this.isFirstChunk=!0,this._completeResults={data:[],errors:[],meta:{}},(function(e){var t=v(e);t.chunkSize=parseInt(t.chunkSize),e.step||e.chunk||(t.chunkSize=null),this._handle=new h(t),(this._handle.streamer=this)._config=t}).call(this,e),this.parseChunk=function(e,t){var i=parseInt(this._config.skipFirstNLines)||0;if(this.isFirstChunk&&0<i){let t=this._config.newline;t||(n=this._config.quoteChar||'"',t=this._handle.guessLineEndings(e,n)),e=[...e.split(t).slice(i)].join(t)}this.isFirstChunk&&k(this._config.beforeFirstChunk)&&void 0!==(n=this._config.beforeFirstChunk(e))&&(e=n),this.isFirstChunk=!1,this._halted=!1;var i=this._partialLine+e,n=(this._partialLine="",this._handle.parse(i,this._baseIndex,!this._finished));if(!this._handle.paused()&&!this._handle.aborted()){if(e=n.meta.cursor,this._finished||(this._partialLine=i.substring(e-this._baseIndex),this._baseIndex=e),n&&n.data&&(this._rowCount+=n.data.length),i=this._finished||this._config.preview&&this._rowCount>=this._config.preview,o)r.postMessage({results:n,workerId:s.WORKER_ID,finished:i});else if(k(this._config.chunk)&&!t){if(this._config.chunk(n,this._handle),this._handle.paused()||this._handle.aborted())return void(this._halted=!0);this._completeResults=n=void 0}return this._config.step||this._config.chunk||(this._completeResults.data=this._completeResults.data.concat(n.data),this._completeResults.errors=this._completeResults.errors.concat(n.errors),this._completeResults.meta=n.meta),this._completed||!i||!k(this._config.complete)||n&&n.meta.aborted||(this._config.complete(this._completeResults,this._input),this._completed=!0),i||n&&n.meta.paused||this._nextChunk(),n}this._halted=!0},this._sendError=function(e){k(this._config.error)?this._config.error(e):o&&this._config.error&&r.postMessage({workerId:s.WORKER_ID,error:e,finished:!1})}}function c(e){var t;(e=e||{}).chunkSize||(e.chunkSize=s.RemoteChunkSize),l.call(this,e),this._nextChunk=i?function(){this._readChunk(),this._chunkLoaded()}:function(){this._readChunk()},this.stream=function(e){this._input=e,this._nextChunk()},this._readChunk=function(){if(this._finished)this._chunkLoaded();else{if(t=new XMLHttpRequest,this._config.withCredentials&&(t.withCredentials=this._config.withCredentials),i||(t.onload=y(this._chunkLoaded,this),t.onerror=y(this._chunkError,this)),t.open(this._config.downloadRequestBody?"POST":"GET",this._input,!i),this._config.downloadRequestHeaders){var e,r,o=this._config.downloadRequestHeaders;for(r in o)t.setRequestHeader(r,o[r])}this._config.chunkSize&&(e=this._start+this._config.chunkSize-1,t.setRequestHeader("Range","bytes="+this._start+"-"+e));try{t.send(this._config.downloadRequestBody)}catch(e){this._chunkError(e.message)}i&&0===t.status&&this._chunkError()}},this._chunkLoaded=function(){let e;4===t.readyState&&(t.status<200||400<=t.status?this._chunkError():(this._start+=this._config.chunkSize||t.responseText.length,this._finished=!this._config.chunkSize||this._start>=(null!==(e=(e=t).getResponseHeader("Content-Range"))?parseInt(e.substring(e.lastIndexOf("/")+1)):-1),this.parseChunk(t.responseText)))},this._chunkError=function(e){e=t.statusText||e,this._sendError(Error(e))}}function u(e){(e=e||{}).chunkSize||(e.chunkSize=s.LocalChunkSize),l.call(this,e);var t,r,i="u">typeof FileReader;this.stream=function(e){this._input=e,r=e.slice||e.webkitSlice||e.mozSlice,i?((t=new FileReader).onload=y(this._chunkLoaded,this),t.onerror=y(this._chunkError,this)):t=new FileReaderSync,this._nextChunk()},this._nextChunk=function(){this._finished||this._config.preview&&!(this._rowCount<this._config.preview)||this._readChunk()},this._readChunk=function(){var e=this._input,o=(this._config.chunkSize&&(o=Math.min(this._start+this._config.chunkSize,this._input.size),e=r.call(e,this._start,o)),t.readAsText(e,this._config.encoding));i||this._chunkLoaded({target:{result:o}})},this._chunkLoaded=function(e){this._start+=this._config.chunkSize,this._finished=!this._config.chunkSize||this._start>=this._input.size,this.parseChunk(e.target.result)},this._chunkError=function(){this._sendError(t.error)}}function d(e){var t;l.call(this,e=e||{}),this.stream=function(e){return t=e,this._nextChunk()},this._nextChunk=function(){var e,r;if(!this._finished)return t=(e=this._config.chunkSize)?(r=t.substring(0,e),t.substring(e)):(r=t,""),this._finished=!t,this.parseChunk(r)}}function p(e){l.call(this,e=e||{});var t=[],r=!0,i=!1;this.pause=function(){l.prototype.pause.apply(this,arguments),this._input.pause()},this.resume=function(){l.prototype.resume.apply(this,arguments),this._input.resume()},this.stream=function(e){this._input=e,this._input.on("data",this._streamData),this._input.on("end",this._streamEnd),this._input.on("error",this._streamError)},this._checkIsFinished=function(){i&&1===t.length&&(this._finished=!0)},this._nextChunk=function(){this._checkIsFinished(),t.length?this.parseChunk(t.shift()):r=!0},this._streamData=y(function(e){try{t.push("string"==typeof e?e:e.toString(this._config.encoding)),r&&(r=!1,this._checkIsFinished(),this.parseChunk(t.shift()))}catch(e){this._streamError(e)}},this),this._streamError=y(function(e){this._streamCleanUp(),this._sendError(e)},this),this._streamEnd=y(function(){this._streamCleanUp(),i=!0,this._streamData("")},this),this._streamCleanUp=y(function(){this._input.removeListener("data",this._streamData),this._input.removeListener("end",this._streamEnd),this._input.removeListener("error",this._streamError)},this)}function h(e){var t,r,i,o,n=/^\s*-?(\d+\.?|\.\d+|\d+\.\d+)([eE][-+]?\d+)?\s*$/,a=/^((\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z)))$/,l=this,c=0,u=0,d=!1,p=!1,h=[],m={data:[],errors:[],meta:{}};function _(t){return"greedy"===e.skipEmptyLines?""===t.join("").trim():1===t.length&&0===t[0].length}function b(){if(m&&i&&(w("Delimiter","UndetectableDelimiter","Unable to auto-detect delimiting character; defaulted to '"+s.DefaultDelimiter+"'"),i=!1),e.skipEmptyLines&&(m.data=m.data.filter(function(e){return!_(e)})),y()){if(m)if(Array.isArray(m.data[0])){for(var t,r=0;y()&&r<m.data.length;r++)m.data[r].forEach(o);m.data.splice(0,1)}else m.data.forEach(o);function o(t,r){k(e.transformHeader)&&(t=e.transformHeader(t,r)),h.push(t)}}function l(t,r){for(var i=e.header?{}:[],o=0;o<t.length;o++){var s=o,l=t[o],l=((t,r)=>(e.dynamicTypingFunction&&void 0===e.dynamicTyping[t]&&(e.dynamicTyping[t]=e.dynamicTypingFunction(t)),!0===(e.dynamicTyping[t]||e.dynamicTyping))?"true"===r||"TRUE"===r||"false"!==r&&"FALSE"!==r&&((e=>{if(n.test(e)&&-0x20000000000000<(e=parseFloat(e))&&e<0x20000000000000)return 1})(r)?parseFloat(r):a.test(r)?new Date(r):""===r?null:r):r)(s=e.header?o>=h.length?"__parsed_extra":h[o]:s,l=e.transform?e.transform(l,s):l);"__parsed_extra"===s?(i[s]=i[s]||[],i[s].push(l)):i[s]=l}return e.header&&(o>h.length?w("FieldMismatch","TooManyFields","Too many fields: expected "+h.length+" fields but parsed "+o,u+r):o<h.length&&w("FieldMismatch","TooFewFields","Too few fields: expected "+h.length+" fields but parsed "+o,u+r)),i}m&&(e.header||e.dynamicTyping||e.transform)&&(t=1,!m.data.length||Array.isArray(m.data[0])?(m.data=m.data.map(l),t=m.data.length):m.data=l(m.data,0),e.header&&m.meta&&(m.meta.fields=h),u+=t)}function y(){return e.header&&0===h.length}function w(e,t,r,i){e={type:e,code:t,message:r},void 0!==i&&(e.row=i),m.errors.push(e)}k(e.step)&&(o=e.step,e.step=function(t){m=t,y()?b():(b(),0!==m.data.length&&(c+=t.data.length,e.preview&&c>e.preview?r.abort():(m.data=m.data[0],o(m,l))))}),this.parse=function(o,n,a){var l=e.quoteChar||'"',l=(e.newline||(e.newline=this.guessLineEndings(o,l)),i=!1,e.delimiter?k(e.delimiter)&&(e.delimiter=e.delimiter(o),m.meta.delimiter=e.delimiter):((l=((t,r,i,o,n)=>{var a,l,c,u;n=n||[",","	","|",";",s.RECORD_SEP,s.UNIT_SEP];for(var d=0;d<n.length;d++){for(var p,h=n[d],f=0,m=0,b=0,v=(c=void 0,new g({comments:o,delimiter:h,newline:r,preview:10}).parse(t)),y=0;y<v.data.length;y++)i&&_(v.data[y])?b++:(m+=p=v.data[y].length,void 0===c?c=p:0<p&&(f+=Math.abs(p-c),c=p));0<v.data.length&&(m/=v.data.length-b),(void 0===l||f<=l)&&(void 0===u||u<m)&&1.99<m&&(l=f,a=h,u=m)}return{successful:!!(e.delimiter=a),bestDelimiter:a}})(o,e.newline,e.skipEmptyLines,e.comments,e.delimitersToGuess)).successful?e.delimiter=l.bestDelimiter:(i=!0,e.delimiter=s.DefaultDelimiter),m.meta.delimiter=e.delimiter),v(e));return e.preview&&e.header&&l.preview++,t=o,m=(r=new g(l)).parse(t,n,a),b(),d?{meta:{paused:!0}}:m||{meta:{paused:!1}}},this.paused=function(){return d},this.pause=function(){d=!0,r.abort(),t=k(e.chunk)?"":t.substring(r.getCharIndex())},this.resume=function(){l.streamer._halted?(d=!1,l.streamer.parseChunk(t,!0)):setTimeout(l.resume,3)},this.aborted=function(){return p},this.abort=function(){p=!0,r.abort(),m.meta.aborted=!0,k(e.complete)&&e.complete(m),t=""},this.guessLineEndings=function(e,t){e=e.substring(0,1048576);var t=RegExp(f(t)+"([^]*?)"+f(t),"gm"),r=(e=e.replace(t,"")).split("\r"),t=e.split("\n"),e=1<t.length&&t[0].length<r[0].length;if(1===r.length||e)return"\n";for(var i=0,o=0;o<r.length;o++)"\n"===r[o][0]&&i++;return i>=r.length/2?"\r\n":"\r"}}function f(e){return e.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}function g(e){var t=(e=e||{}).delimiter,r=e.newline,i=e.comments,o=e.step,n=e.preview,a=e.fastMode,l=null,c=!1,u=null==e.quoteChar?'"':e.quoteChar,d=u;if(void 0!==e.escapeChar&&(d=e.escapeChar),("string"!=typeof t||-1<s.BAD_DELIMITERS.indexOf(t))&&(t=","),i===t)throw Error("Comment character same as delimiter");!0===i?i="#":("string"!=typeof i||-1<s.BAD_DELIMITERS.indexOf(i))&&(i=!1),"\n"!==r&&"\r"!==r&&"\r\n"!==r&&(r="\n");var p=0,h=!1;this.parse=function(s,g,m){if("string"!=typeof s)throw Error("Input must be a string");var _=s.length,b=t.length,v=r.length,y=i.length,w=k(o),S=[],x=[],C=[],E=p=0;if(!s)return N();if(a||!1!==a&&-1===s.indexOf(u)){for(var O=s.split(r),R=0;R<O.length;R++){if(C=O[R],p+=C.length,R!==O.length-1)p+=r.length;else if(m)break;if(!i||C.substring(0,y)!==i){if(w){if(S=[],M(C.split(t)),$(),h)return N()}else M(C.split(t));if(n&&n<=R)return S=S.slice(0,n),N(!0)}}return N()}for(var I=s.indexOf(t,p),j=s.indexOf(r,p),z=RegExp(f(d)+f(u),"g"),A=s.indexOf(u,p);;)if(s[p]===u)for(A=p,p++;;){if(-1===(A=s.indexOf(u,A+1)))return m||x.push({type:"Quotes",code:"MissingQuotes",message:"Quoted field unterminated",row:S.length,index:p}),P();if(A===_-1)return P(s.substring(p,A).replace(z,u));if(u===d&&s[A+1]===d)A++;else if(u===d||0===A||s[A-1]!==d){-1!==I&&I<A+1&&(I=s.indexOf(t,A+1));var T=F(-1===(j=-1!==j&&j<A+1?s.indexOf(r,A+1):j)?I:Math.min(I,j));if(s.substr(A+1+T,b)===t){C.push(s.substring(p,A).replace(z,u)),s[p=A+1+T+b]!==u&&(A=s.indexOf(u,p)),I=s.indexOf(t,p),j=s.indexOf(r,p);break}if(T=F(j),s.substring(A+1+T,A+1+T+v)===r){if(C.push(s.substring(p,A).replace(z,u)),L(A+1+T+v),I=s.indexOf(t,p),A=s.indexOf(u,p),w&&($(),h))return N();if(n&&S.length>=n)return N(!0);break}x.push({type:"Quotes",code:"InvalidQuotes",message:"Trailing quote on quoted field is malformed",row:S.length,index:p}),A++}}else if(i&&0===C.length&&s.substring(p,p+y)===i){if(-1===j)return N();p=j+v,j=s.indexOf(r,p),I=s.indexOf(t,p)}else if(-1!==I&&(I<j||-1===j))C.push(s.substring(p,I)),p=I+b,I=s.indexOf(t,p);else{if(-1===j)break;if(C.push(s.substring(p,j)),L(j+v),w&&($(),h))return N();if(n&&S.length>=n)return N(!0)}return P();function M(e){S.push(e),E=p}function F(e){return -1!==e&&(e=s.substring(A+1,e))&&""===e.trim()?e.length:0}function P(e){return m||(void 0===e&&(e=s.substring(p)),C.push(e),p=_,M(C),w&&$()),N()}function L(e){p=e,M(C),C=[],j=s.indexOf(r,p)}function N(i){if(e.header&&!g&&S.length&&!c){var o=S[0],n=Object.create(null),a=new Set(o);let t=!1;for(let r=0;r<o.length;r++){let i=o[r];if(n[i=k(e.transformHeader)?e.transformHeader(i,r):i]){let e,s=n[i];for(;e=i+"_"+s,s++,a.has(e););a.add(e),o[r]=e,n[i]++,t=!0,(l=null===l?{}:l)[e]=i}else n[i]=1,o[r]=i;a.add(i)}t&&console.warn("Duplicate headers found and renamed."),c=!0}return{data:S,errors:x,meta:{delimiter:t,linebreak:r,aborted:h,truncated:!!i,cursor:E+(g||0),renamedHeaders:l}}}function $(){o(N()),S=[],x=[]}},this.abort=function(){h=!0},this.getCharIndex=function(){return p}}function m(e){var t=e.data,r=n[t.workerId],i=!1;if(t.error)r.userError(t.error,t.file);else if(t.results&&t.results.data){var o={abort:function(){i=!0,_(t.workerId,{data:[],errors:[],meta:{aborted:!0}})},pause:b,resume:b};if(k(r.userStep)){for(var a=0;a<t.results.data.length&&(r.userStep({data:t.results.data[a],errors:t.results.errors,meta:t.results.meta},o),!i);a++);delete t.results}else k(r.userChunk)&&(r.userChunk(t.results,o,t.file),delete t.results)}t.finished&&!i&&_(t.workerId,t.results)}function _(e,t){var r=n[e];k(r.userComplete)&&r.userComplete(t),r.terminate(),delete n[e]}function b(){throw Error("Not implemented.")}function v(e){if("object"!=typeof e||null===e)return e;var t,r=Array.isArray(e)?[]:{};for(t in e)r[t]=v(e[t]);return r}function y(e,t){return function(){e.apply(t,arguments)}}function k(e){return"function"==typeof e}return s.parse=function(t,i){var o,l,h,f=(i=i||{}).dynamicTyping||!1;if(k(f)&&(i.dynamicTypingFunction=f,f={}),i.dynamicTyping=f,i.transform=!!k(i.transform)&&i.transform,!i.worker||!s.WORKERS_SUPPORTED){let e;return f=null,s.NODE_STREAM_INPUT,"string"==typeof t?(t=65279!==(e=t).charCodeAt(0)?e:e.slice(1),f=new(i.download?c:d)(i)):!0===t.readable&&k(t.read)&&k(t.on)?f=new p(i):(r.File&&t instanceof File||t instanceof Object)&&(f=new u(i)),f.stream(t)}(f=!!s.WORKERS_SUPPORTED&&(l=r.URL||r.webkitURL||null,h=e.toString(),o=s.BLOB_URL||(s.BLOB_URL=l.createObjectURL(new Blob(["var global = (function() { if (typeof self !== 'undefined') { return self; } if (typeof window !== 'undefined') { return window; } if (typeof global !== 'undefined') { return global; } return {}; })(); global.IS_PAPA_WORKER=true; ","(",h,")();"],{type:"text/javascript"}))),(o=new r.Worker(o)).onmessage=m,o.id=a++,n[o.id]=o)).userStep=i.step,f.userChunk=i.chunk,f.userComplete=i.complete,f.userError=i.error,i.step=k(i.step),i.chunk=k(i.chunk),i.complete=k(i.complete),i.error=k(i.error),delete i.worker,f.postMessage({input:t,config:i,workerId:f.id})},s.unparse=function(e,t){var r=!1,i=!0,o=",",n="\r\n",a='"',l=a+a,c=!1,u=null,d=!1,p=((()=>{if("object"==typeof t){if("string"!=typeof t.delimiter||s.BAD_DELIMITERS.filter(function(e){return -1!==t.delimiter.indexOf(e)}).length||(o=t.delimiter),("boolean"==typeof t.quotes||"function"==typeof t.quotes||Array.isArray(t.quotes))&&(r=t.quotes),"boolean"!=typeof t.skipEmptyLines&&"string"!=typeof t.skipEmptyLines||(c=t.skipEmptyLines),"string"==typeof t.newline&&(n=t.newline),"string"==typeof t.quoteChar&&(a=t.quoteChar),"boolean"==typeof t.header&&(i=t.header),Array.isArray(t.columns)){if(0===t.columns.length)throw Error("Option columns is empty");u=t.columns}void 0!==t.escapeChar&&(l=t.escapeChar+a),t.escapeFormulae instanceof RegExp?d=t.escapeFormulae:"boolean"==typeof t.escapeFormulae&&t.escapeFormulae&&(d=/^[=+\-@\t\r].*$/)}})(),RegExp(f(a),"g"));if("string"==typeof e&&(e=JSON.parse(e)),Array.isArray(e)){if(!e.length||Array.isArray(e[0]))return h(null,e,c);if("object"==typeof e[0])return h(u||Object.keys(e[0]),e,c)}else if("object"==typeof e)return"string"==typeof e.data&&(e.data=JSON.parse(e.data)),Array.isArray(e.data)&&(e.fields||(e.fields=e.meta&&e.meta.fields||u),e.fields||(e.fields=Array.isArray(e.data[0])?e.fields:"object"==typeof e.data[0]?Object.keys(e.data[0]):[]),Array.isArray(e.data[0])||"object"==typeof e.data[0]||(e.data=[e.data])),h(e.fields||[],e.data||[],c);throw Error("Unable to serialize unrecognized input");function h(e,t,r){var a="",s=("string"==typeof e&&(e=JSON.parse(e)),"string"==typeof t&&(t=JSON.parse(t)),Array.isArray(e)&&0<e.length),l=!Array.isArray(t[0]);if(s&&i){for(var c=0;c<e.length;c++)0<c&&(a+=o),a+=g(e[c],c);0<t.length&&(a+=n)}for(var u=0;u<t.length;u++){var d=(s?e:t[u]).length,p=!1,h=s?0===Object.keys(t[u]).length:0===t[u].length;if(r&&!s&&(p="greedy"===r?""===t[u].join("").trim():1===t[u].length&&0===t[u][0].length),"greedy"===r&&s){for(var f=[],m=0;m<d;m++){var _=l?e[m]:m;f.push(t[u][_])}p=""===f.join("").trim()}if(!p){for(var b=0;b<d;b++){0<b&&!h&&(a+=o);var v=s&&l?e[b]:b;a+=g(t[u][v],b)}u<t.length-1&&(!r||0<d&&!h)&&(a+=n)}}return a}function g(e,t){var i,n;return null==e?"":e.constructor===Date?JSON.stringify(e).slice(1,25):(n=!1,d&&"string"==typeof e&&d.test(e)&&(e="'"+e,n=!0),i=e.toString().replace(p,l),(n=n||!0===r||"function"==typeof r&&r(e,t)||Array.isArray(r)&&r[t]||((e,t)=>{for(var r=0;r<t.length;r++)if(-1<e.indexOf(t[r]))return!0;return!1})(i,s.BAD_DELIMITERS)||-1<i.indexOf(o)||" "===i.charAt(0)||" "===i.charAt(i.length-1))?a+i+a:i)}},s.RECORD_SEP="\x1e",s.UNIT_SEP="\x1f",s.BYTE_ORDER_MARK="\uFEFF",s.BAD_DELIMITERS=["\r","\n",'"',s.BYTE_ORDER_MARK],s.WORKERS_SUPPORTED=!i&&!!r.Worker,s.NODE_STREAM_INPUT=1,s.LocalChunkSize=0xa00000,s.RemoteChunkSize=5242880,s.DefaultDelimiter=",",s.Parser=g,s.ParserHandle=h,s.NetworkStreamer=c,s.FileStreamer=u,s.StringStreamer=d,s.ReadableStreamStreamer=p,r.jQuery&&((t=r.jQuery).fn.parse=function(e){var i=e.config||{},o=[];return this.each(function(e){if(!("INPUT"===t(this).prop("tagName").toUpperCase()&&"file"===t(this).attr("type").toLowerCase()&&r.FileReader)||!this.files||0===this.files.length)return!0;for(var n=0;n<this.files.length;n++)o.push({file:this.files[n],inputElem:this,instanceConfig:t.extend({},i)})}),n(),this;function n(){if(0===o.length)k(e.complete)&&e.complete();else{var r,i,n,l=o[0];if(k(e.before)){var c=e.before(l.file,l.inputElem);if("object"==typeof c){if("abort"===c.action)return r=l.file,i=l.inputElem,n=c.reason,void(k(e.error)&&e.error({name:"AbortError"},r,i,n));if("skip"===c.action)return void a();"object"==typeof c.config&&(l.instanceConfig=t.extend(l.instanceConfig,c.config))}else if("skip"===c)return void a()}var u=l.instanceConfig.complete;l.instanceConfig.complete=function(e){k(u)&&u(e,l.file,l.inputElem),a()},s.parse(l.file,l.instanceConfig)}}function a(){o.splice(0,1),n()}}),o&&(r.onmessage=function(e){e=e.data,void 0===s.WORKER_ID&&e&&(s.WORKER_ID=e.workerId),"string"==typeof e.input?r.postMessage({workerId:s.WORKER_ID,results:s.parse(e.input,e.config),finished:!0}):(r.File&&e.input instanceof File||e.input instanceof Object)&&(e=s.parse(e.input,e.config))&&r.postMessage({workerId:s.WORKER_ID,results:e,finished:!0})}),(c.prototype=Object.create(l.prototype)).constructor=c,(u.prototype=Object.create(l.prototype)).constructor=u,(d.prototype=Object.create(d.prototype)).constructor=d,(p.prototype=Object.create(l.prototype)).constructor=p,s},"function"==typeof define&&define.amd?void 0!==(o=i())&&e.v(o):t.exports=i()},891547,e=>{"use strict";var t=e.i(843476),r=e.i(271645),i=e.i(199133),o=e.i(764205);e.s(["default",0,({onChange:e,value:n,className:a,accessToken:s,disabled:l})=>{let[c,u]=(0,r.useState)([]),[d,p]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(s){p(!0);try{let e=await (0,o.getGuardrailsList)(s);console.log("Guardrails response:",e),e.guardrails&&(console.log("Guardrails data:",e.guardrails),u(e.guardrails))}catch(e){console.error("Error fetching guardrails:",e)}finally{p(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(i.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onChange:t=>{console.log("Selected guardrails:",t),e(t)},value:n,loading:d,className:a,allowClear:!0,options:c.map(e=>(console.log("Mapping guardrail:",e),{label:`${e.guardrail_name}`,value:e.guardrail_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},921511,e=>{"use strict";var t=e.i(843476),r=e.i(271645),i=e.i(199133),o=e.i(764205);function n(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let r=e.version_number??1,i=e.version_status??"draft";return{label:`${e.policy_name} — v${r} (${i})${e.description?` — ${e.description}`:""}`,value:"production"===i?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["default",0,({onChange:e,value:a,className:s,accessToken:l,disabled:c,onPoliciesLoaded:u})=>{let[d,p]=(0,r.useState)([]),[h,f]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(l){f(!0);try{let e=await (0,o.getPoliciesList)(l);e.policies&&(p(e.policies),u?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{f(!1)}}})()},[l,u]),(0,t.jsx)("div",{children:(0,t.jsx)(i.Select,{mode:"multiple",disabled:c,placeholder:c?"Setting policies is a premium feature.":"Select policies (production or published versions)",onChange:t=>{e(t)},value:a,loading:h,className:s,allowClear:!0,options:n(d),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})},"getPolicyOptionEntries",()=>n])},367240,54943,555436,e=>{"use strict";var t=e.i(475254);let r=(0,t.default)("rotate-ccw",[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]]);e.s(["RotateCcw",()=>r],367240);let i=(0,t.default)("search",[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]]);e.s(["default",()=>i],54943),e.s(["Search",()=>i],555436)},646563,e=>{"use strict";var t=e.i(959013);e.s(["PlusOutlined",()=>t.default])},987432,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M893.3 293.3L730.7 130.7c-7.5-7.5-16.7-13-26.7-16V112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V338.5c0-17-6.7-33.2-18.7-45.2zM384 184h256v104H384V184zm456 656H184V184h136v136c0 17.7 14.3 32 32 32h320c17.7 0 32-14.3 32-32V205.8l136 136V840zM512 442c-79.5 0-144 64.5-144 144s64.5 144 144 144 144-64.5 144-144-64.5-144-144-144zm0 224c-44.2 0-80-35.8-80-80s35.8-80 80-80 80 35.8 80 80-35.8 80-80 80z"}}]},name:"save",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["SaveOutlined",0,n],987432)},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["CodeOutlined",0,n],245094)},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["CheckCircleOutlined",0,n],245704)},431343,569074,e=>{"use strict";var t=e.i(475254);let r=(0,t.default)("play",[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]]);e.s(["Play",()=>r],431343);let i=(0,t.default)("upload",[["path",{d:"M12 3v12",key:"1x0j5s"}],["path",{d:"m17 8-5-5-5 5",key:"7q97r8"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}]]);e.s(["Upload",()=>i],569074)},98919,e=>{"use strict";var t=e.i(918549);e.s(["Shield",()=>t.default])},727612,e=>{"use strict";let t=(0,e.i(475254).default)("trash-2",[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]]);e.s(["Trash2",()=>t],727612)},918549,e=>{"use strict";let t=(0,e.i(475254).default)("shield",[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}]]);e.s(["default",()=>t])},673709,e=>{"use strict";var t=e.i(843476),r=e.i(271645),i=e.i(678784);let o=(0,e.i(475254).default)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var n=e.i(650056);let a={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};e.s(["default",0,({code:e,language:s})=>{let[l,c]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-gray-200 overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),c(!0),setTimeout(()=>c(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 z-10","aria-label":"Copy code",children:l?(0,t.jsx)(i.CheckIcon,{size:16}):(0,t.jsx)(o,{size:16})}),(0,t.jsx)(n.Prism,{language:s,style:a,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",backgroundColor:"#fafafa"},showLineNumbers:!0,children:e})]})}],673709)},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),r=e.i(343794),i=e.i(914949),o=e.i(404948);let n=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,n],836938);var a=e.i(613541),s=e.i(763731),l=e.i(242064),c=e.i(491816);e.i(793154);var u=e.i(880476),d=e.i(183293),p=e.i(717356),h=e.i(320560),f=e.i(307358),g=e.i(246422),m=e.i(838378),_=e.i(617933);let b=(0,g.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:r}=e,i=(0,m.mergeToken)(e,{popoverBg:t,popoverColor:r});return[(e=>{let{componentCls:t,popoverColor:r,titleMinWidth:i,fontWeightStrong:o,innerPadding:n,boxShadowSecondary:a,colorTextHeading:s,borderRadiusLG:l,zIndexPopup:c,titleMarginBottom:u,colorBgElevated:p,popoverBg:f,titleBorderBottom:g,innerContentPadding:m,titlePadding:_}=e;return[{[t]:Object.assign(Object.assign({},(0,d.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:c,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":p,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:f,backgroundClip:"padding-box",borderRadius:l,boxShadow:a,padding:n},[`${t}-title`]:{minWidth:i,marginBottom:u,color:s,fontWeight:o,borderBottom:g,padding:_},[`${t}-inner-content`]:{color:r,padding:m}})},(0,h.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(i),(e=>{let{componentCls:t}=e;return{[t]:_.PresetColors.map(r=>{let i=e[`${r}6`];return{[`&${t}-${r}`]:{"--antd-arrow-background-color":i,[`${t}-inner`]:{backgroundColor:i},[`${t}-arrow`]:{background:"transparent"}}}})}})(i),(0,p.initZoomMotion)(i,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:r,fontHeight:i,padding:o,wireframe:n,zIndexPopupBase:a,borderRadiusLG:s,marginXS:l,lineType:c,colorSplit:u,paddingSM:d}=e,p=r-i;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:a+30},(0,f.getArrowToken)(e)),(0,h.getArrowOffsetToken)({contentRadius:s,limitVerticalRadius:!0})),{innerPadding:12*!n,titleMarginBottom:n?0:l,titlePadding:n?`${p/2}px ${o}px ${p/2-t}px`:0,titleBorderBottom:n?`${t}px ${c} ${u}`:"none",innerContentPadding:n?`${d}px ${o}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var v=function(e,t){var r={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(r[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var o=0,i=Object.getOwnPropertySymbols(e);o<i.length;o++)0>t.indexOf(i[o])&&Object.prototype.propertyIsEnumerable.call(e,i[o])&&(r[i[o]]=e[i[o]]);return r};let y=({title:e,content:r,prefixCls:i})=>e||r?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${i}-title`},e),r&&t.createElement("div",{className:`${i}-inner-content`},r)):null,k=e=>{let{hashId:i,prefixCls:o,className:a,style:s,placement:l="top",title:c,content:d,children:p}=e,h=n(c),f=n(d),g=(0,r.default)(i,o,`${o}-pure`,`${o}-placement-${l}`,a);return t.createElement("div",{className:g,style:s},t.createElement("div",{className:`${o}-arrow`}),t.createElement(u.Popup,Object.assign({},e,{className:i,prefixCls:o}),p||t.createElement(y,{prefixCls:o,title:h,content:f})))},w=e=>{let{prefixCls:i,className:o}=e,n=v(e,["prefixCls","className"]),{getPrefixCls:a}=t.useContext(l.ConfigContext),s=a("popover",i),[c,u,d]=b(s);return c(t.createElement(k,Object.assign({},n,{prefixCls:s,hashId:u,className:(0,r.default)(o,d)})))};e.s(["Overlay",0,y,"default",0,w],310730);var S=function(e,t){var r={};for(var i in e)Object.prototype.hasOwnProperty.call(e,i)&&0>t.indexOf(i)&&(r[i]=e[i]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var o=0,i=Object.getOwnPropertySymbols(e);o<i.length;o++)0>t.indexOf(i[o])&&Object.prototype.propertyIsEnumerable.call(e,i[o])&&(r[i[o]]=e[i[o]]);return r};let x=t.forwardRef((e,u)=>{var d,p;let{prefixCls:h,title:f,content:g,overlayClassName:m,placement:_="top",trigger:v="hover",children:k,mouseEnterDelay:w=.1,mouseLeaveDelay:x=.1,onOpenChange:C,overlayStyle:E={},styles:O,classNames:R}=e,I=S(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:j,className:z,style:A,classNames:T,styles:M}=(0,l.useComponentConfig)("popover"),F=j("popover",h),[P,L,N]=b(F),$=j(),D=(0,r.default)(m,L,N,z,T.root,null==R?void 0:R.root),H=(0,r.default)(T.body,null==R?void 0:R.body),[B,q]=(0,i.default)(!1,{value:null!=(d=e.open)?d:e.visible,defaultValue:null!=(p=e.defaultOpen)?p:e.defaultVisible}),U=(e,t)=>{q(e,!0),null==C||C(e,t)},V=n(f),W=n(g);return P(t.createElement(c.default,Object.assign({placement:_,trigger:v,mouseEnterDelay:w,mouseLeaveDelay:x},I,{prefixCls:F,classNames:{root:D,body:H},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},M.root),A),E),null==O?void 0:O.root),body:Object.assign(Object.assign({},M.body),null==O?void 0:O.body)},ref:u,open:B,onOpenChange:e=>{U(e)},overlay:V||W?t.createElement(y,{prefixCls:F,title:V,content:W}):null,transitionName:(0,a.getTransitionName)($,"zoom-big",I.transitionName),"data-popover-inject":!0}),(0,s.cloneElement)(k,{onKeyDown:e=>{var r,i;(0,t.isValidElement)(k)&&(null==(i=null==k?void 0:(r=k.props).onKeyDown)||i.call(r,e)),e.keyCode===o.default.ESC&&U(!1,e)}})))});x._InternalPanelDoNotUseOrYouWillBeFired=w,e.s(["default",0,x],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},872934,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM770.87 199.13l-52.2-52.2a8.01 8.01 0 014.7-13.6l179.4-21c5.1-.6 9.5 3.7 8.9 8.9l-21 179.4c-.8 6.6-8.9 9.4-13.6 4.7l-52.4-52.4-256.2 256.2a8.03 8.03 0 01-11.3 0l-42.4-42.4a8.03 8.03 0 010-11.3l256.1-256.3z"}}]},name:"export",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["ExportOutlined",0,n],872934)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["DollarOutlined",0,n],458505)},903446,e=>{"use strict";let t=(0,e.i(475254).default)("settings",[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["default",()=>t])},190272,785913,e=>{"use strict";var t,r,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),o=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r);let n={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>o,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(i).includes(e)){let t=n[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:i,apiKey:n,inputMessage:a,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:u,selectedPolicies:d,selectedMCPServers:p,mcpServers:h,mcpServerToolRestrictions:f,selectedVoice:g,endpointType:m,selectedModel:_,selectedSdk:b,proxySettings:v}=e,y="session"===r?i:n,k=window.location.origin,w=v?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?k=w:v?.PROXY_BASE_URL&&(k=v.PROXY_BASE_URL);let S=a||"Your prompt here",x=S.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),C=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),c.length>0&&(E.vector_stores=c),u.length>0&&(E.guardrails=u),d.length>0&&(E.policies=d);let O=_||"your-model-name",R="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${k}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${k}"
)`;switch(m){case o.CHAT:{let e=Object.keys(E).length>0,r="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let i=C.length>0?C:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${O}",
    messages=${JSON.stringify(i,null,4)}${r}
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
#                     "text": "${x}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${r}
# )
# print(response_with_file)
`;break}case o.RESPONSES:{let e=Object.keys(E).length>0,r="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let i=C.length>0?C:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${O}",
    input=${JSON.stringify(i,null,4)}${r}
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
#                 {"type": "input_text", "text": "${x}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case o.IMAGE:t="azure"===b?`
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
prompt = "${x}"

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
`;break;case o.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${x}"

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
prompt = "${x}"

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
`;break;case o.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${O}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case o.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${O}",
	file=audio_file${a?`,
	prompt="${a.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case o.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${O}",
	input="${a||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${O}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${R}
${t}`}],190272)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["LinkOutlined",0,n],596239)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var o=e.i(9583),n=r.forwardRef(function(e,n){return r.createElement(o.default,(0,t.default)({},e,{ref:n,icon:i}))});e.s(["ClockCircleOutlined",0,n],637235)},516015,(e,t,r)=>{},898547,(e,t,r)=>{var i=e.i(247167);e.r(516015);var o=e.r(271645),n=o&&"object"==typeof o&&"default"in o?o:{default:o},a=void 0!==i.default&&i.default.env&&!0,s=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,r=t.name,i=void 0===r?"stylesheet":r,o=t.optimizeForSpeed,n=void 0===o?a:o;c(s(i),"`name` must be a string"),this._name=i,this._deletedRulePlaceholder="#"+i+"-deleted-rule____{}",c("boolean"==typeof n,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=n,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,r=e.prototype;return r.setOptimizeForSpeed=function(e){c("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),c(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},r.isOptimizeForSpeed=function(){return this._optimizeForSpeed},r.inject=function(){var e=this;if(c(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(a||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,r){return"number"==typeof r?e._serverSheet.cssRules[r]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),r},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},r.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},r.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},r.insertRule=function(e,t){if(c(s(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var r=this.getSheet();"number"!=typeof t&&(t=r.cssRules.length);try{r.insertRule(e,t)}catch(t){return a||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var i=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,i))}return this._rulesCount++},r.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var r="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!r.cssRules[e])return e;r.deleteRule(e);try{r.insertRule(t,e)}catch(i){a||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),r.insertRule(this._deletedRulePlaceholder,e)}}else{var i=this._tags[e];c(i,"old rule at index `"+e+"` not found"),i.textContent=t}return e},r.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];c(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},r.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},r.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,r){return r?t=t.concat(Array.prototype.map.call(e.getSheetForTag(r).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},r.makeStyleTag=function(e,t,r){t&&c(s(t),"makeStyleTag accepts only strings as second parameter");var i=document.createElement("style");this._nonce&&i.setAttribute("nonce",this._nonce),i.type="text/css",i.setAttribute("data-"+e,""),t&&i.appendChild(document.createTextNode(t));var o=document.head||document.getElementsByTagName("head")[0];return r?o.insertBefore(i,r):o.appendChild(i),i},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var r=0;r<t.length;r++){var i=t[r];i.enumerable=i.enumerable||!1,i.configurable=!0,"value"in i&&(i.writable=!0),Object.defineProperty(e,i.key,i)}}(e.prototype,t),e}();function c(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var u=function(e){for(var t=5381,r=e.length;r;)t=33*t^e.charCodeAt(--r);return t>>>0},d={};function p(e,t){if(!t)return"jsx-"+e;var r=String(t),i=e+r;return d[i]||(d[i]="jsx-"+u(e+"-"+r)),d[i]}function h(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var r=e+t;return d[r]||(d[r]=t.replace(/__jsx-style-dynamic-selector/g,e)),d[r]}var f=function(){function e(e){var t=void 0===e?{}:e,r=t.styleSheet,i=void 0===r?null:r,o=t.optimizeForSpeed,n=void 0!==o&&o;this._sheet=i||new l({name:"styled-jsx",optimizeForSpeed:n}),this._sheet.inject(),i&&"boolean"==typeof n&&(this._sheet.setOptimizeForSpeed(n),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var r=this.getIdAndRules(e),i=r.styleId,o=r.rules;if(i in this._instancesCounts){this._instancesCounts[i]+=1;return}var n=o.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[i]=n,this._instancesCounts[i]=1},t.remove=function(e){var t=this,r=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(r in this._instancesCounts,"styleId: `"+r+"` not found"),this._instancesCounts[r]-=1,this._instancesCounts[r]<1){var i=this._fromServer&&this._fromServer[r];i?(i.parentNode.removeChild(i),delete this._fromServer[r]):(this._indices[r].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[r]),delete this._instancesCounts[r]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],r=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return r[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,r;return t=this.cssRules(),void 0===(r=e)&&(r={}),t.map(function(e){var t=e[0],i=e[1];return n.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:r.nonce?r.nonce:void 0,dangerouslySetInnerHTML:{__html:i}})})},t.getIdAndRules=function(e){var t=e.children,r=e.dynamic,i=e.id;if(r){var o=p(i,r);return{styleId:o,rules:Array.isArray(t)?t.map(function(e){return h(o,e)}):[h(o,t)]}}return{styleId:p(i),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),g=o.createContext(null);function m(){return new f}function _(){return o.useContext(g)}g.displayName="StyleSheetContext";var b=n.default.useInsertionEffect||n.default.useLayoutEffect,v="u">typeof window?m():void 0;function y(e){var t=v||_();return t&&("u"<typeof window?t.add(e):b(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}y.dynamic=function(e){return e.map(function(e){return p(e[0],e[1])}).join(" ")},r.StyleRegistry=function(e){var t=e.registry,r=e.children,i=o.useContext(g),a=o.useState(function(){return i||t||m()})[0];return n.default.createElement(g.Provider,{value:a},r)},r.createStyleRegistry=m,r.style=y,r.useStyleRegistry=_},437902,(e,t,r)=>{t.exports=e.r(898547).style}]);