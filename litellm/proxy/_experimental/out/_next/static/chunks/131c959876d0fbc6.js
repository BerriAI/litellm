(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},37727,e=>{"use strict";var t=e.i(841947);e.s(["X",()=>t.default])},107233,e=>{"use strict";var t=e.i(603908);e.s(["Plus",()=>t.default])},678745,e=>{"use strict";let t=(0,e.i(475254).default)("check",[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]]);e.s(["default",()=>t])},689020,e=>{"use strict";var t=e.i(764205);let i=async e=>{try{let i=await (0,t.modelHubCall)(e);if(console.log("model_info:",i),i?.data.length>0){let e=i.data.map(e=>({model_group:e.model_group,mode:e?.mode}));return e.sort((e,t)=>e.model_group.localeCompare(t.model_group)),e}return[]}catch(e){throw console.error("Error fetching model info:",e),e}};e.s(["fetchAvailableModels",0,i])},983561,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M300 328a60 60 0 10120 0 60 60 0 10-120 0zM852 64H172c-17.7 0-32 14.3-32 32v660c0 17.7 14.3 32 32 32h680c17.7 0 32-14.3 32-32V96c0-17.7-14.3-32-32-32zm-32 660H204V128h616v596zM604 328a60 60 0 10120 0 60 60 0 10-120 0zm250.2 556H169.8c-16.5 0-29.8 14.3-29.8 32v36c0 4.4 3.3 8 7.4 8h729.1c4.1 0 7.4-3.6 7.4-8v-36c.1-17.7-13.2-32-29.7-32zM664 508H360c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h304c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"robot",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["RobotOutlined",0,a],983561)},603908,e=>{"use strict";let t=(0,e.i(475254).default)("plus",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]]);e.s(["default",()=>t])},841947,e=>{"use strict";let t=(0,e.i(475254).default)("x",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]]);e.s(["default",()=>t])},916940,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),n=e.i(764205);e.s(["default",0,({onChange:e,value:a,className:s,accessToken:o,placeholder:l="Select vector stores",disabled:u=!1})=>{let[c,d]=(0,i.useState)([]),[p,h]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(o){h(!0);try{let e=await (0,n.vectorStoreListCall)(o);e.data&&d(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{h(!1)}}})()},[o]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",placeholder:l,onChange:e,value:a,loading:p,className:s,allowClear:!0,options:c.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,title:e.vector_store_description||e.vector_store_id})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"},disabled:u})})}])},737434,e=>{"use strict";var t=e.i(184163);e.s(["DownloadOutlined",()=>t.default])},59935,(e,t,i)=>{var r;let n;e.e,r=function e(){var t,i="u">typeof self?self:"u">typeof window?window:void 0!==i?i:{},r=!i.document&&!!i.postMessage,n=i.IS_PAPA_WORKER||!1,a={},s=0,o={};function l(e){this._handle=null,this._finished=!1,this._completed=!1,this._halted=!1,this._input=null,this._baseIndex=0,this._partialLine="",this._rowCount=0,this._start=0,this._nextChunk=null,this.isFirstChunk=!0,this._completeResults={data:[],errors:[],meta:{}},(function(e){var t=v(e);t.chunkSize=parseInt(t.chunkSize),e.step||e.chunk||(t.chunkSize=null),this._handle=new h(t),(this._handle.streamer=this)._config=t}).call(this,e),this.parseChunk=function(e,t){var r=parseInt(this._config.skipFirstNLines)||0;if(this.isFirstChunk&&0<r){let t=this._config.newline;t||(a=this._config.quoteChar||'"',t=this._handle.guessLineEndings(e,a)),e=[...e.split(t).slice(r)].join(t)}this.isFirstChunk&&S(this._config.beforeFirstChunk)&&void 0!==(a=this._config.beforeFirstChunk(e))&&(e=a),this.isFirstChunk=!1,this._halted=!1;var r=this._partialLine+e,a=(this._partialLine="",this._handle.parse(r,this._baseIndex,!this._finished));if(!this._handle.paused()&&!this._handle.aborted()){if(e=a.meta.cursor,this._finished||(this._partialLine=r.substring(e-this._baseIndex),this._baseIndex=e),a&&a.data&&(this._rowCount+=a.data.length),r=this._finished||this._config.preview&&this._rowCount>=this._config.preview,n)i.postMessage({results:a,workerId:o.WORKER_ID,finished:r});else if(S(this._config.chunk)&&!t){if(this._config.chunk(a,this._handle),this._handle.paused()||this._handle.aborted())return void(this._halted=!0);this._completeResults=a=void 0}return this._config.step||this._config.chunk||(this._completeResults.data=this._completeResults.data.concat(a.data),this._completeResults.errors=this._completeResults.errors.concat(a.errors),this._completeResults.meta=a.meta),this._completed||!r||!S(this._config.complete)||a&&a.meta.aborted||(this._config.complete(this._completeResults,this._input),this._completed=!0),r||a&&a.meta.paused||this._nextChunk(),a}this._halted=!0},this._sendError=function(e){S(this._config.error)?this._config.error(e):n&&this._config.error&&i.postMessage({workerId:o.WORKER_ID,error:e,finished:!1})}}function u(e){var t;(e=e||{}).chunkSize||(e.chunkSize=o.RemoteChunkSize),l.call(this,e),this._nextChunk=r?function(){this._readChunk(),this._chunkLoaded()}:function(){this._readChunk()},this.stream=function(e){this._input=e,this._nextChunk()},this._readChunk=function(){if(this._finished)this._chunkLoaded();else{if(t=new XMLHttpRequest,this._config.withCredentials&&(t.withCredentials=this._config.withCredentials),r||(t.onload=b(this._chunkLoaded,this),t.onerror=b(this._chunkError,this)),t.open(this._config.downloadRequestBody?"POST":"GET",this._input,!r),this._config.downloadRequestHeaders){var e,i,n=this._config.downloadRequestHeaders;for(i in n)t.setRequestHeader(i,n[i])}this._config.chunkSize&&(e=this._start+this._config.chunkSize-1,t.setRequestHeader("Range","bytes="+this._start+"-"+e));try{t.send(this._config.downloadRequestBody)}catch(e){this._chunkError(e.message)}r&&0===t.status&&this._chunkError()}},this._chunkLoaded=function(){let e;4===t.readyState&&(t.status<200||400<=t.status?this._chunkError():(this._start+=this._config.chunkSize||t.responseText.length,this._finished=!this._config.chunkSize||this._start>=(null!==(e=(e=t).getResponseHeader("Content-Range"))?parseInt(e.substring(e.lastIndexOf("/")+1)):-1),this.parseChunk(t.responseText)))},this._chunkError=function(e){e=t.statusText||e,this._sendError(Error(e))}}function c(e){(e=e||{}).chunkSize||(e.chunkSize=o.LocalChunkSize),l.call(this,e);var t,i,r="u">typeof FileReader;this.stream=function(e){this._input=e,i=e.slice||e.webkitSlice||e.mozSlice,r?((t=new FileReader).onload=b(this._chunkLoaded,this),t.onerror=b(this._chunkError,this)):t=new FileReaderSync,this._nextChunk()},this._nextChunk=function(){this._finished||this._config.preview&&!(this._rowCount<this._config.preview)||this._readChunk()},this._readChunk=function(){var e=this._input,n=(this._config.chunkSize&&(n=Math.min(this._start+this._config.chunkSize,this._input.size),e=i.call(e,this._start,n)),t.readAsText(e,this._config.encoding));r||this._chunkLoaded({target:{result:n}})},this._chunkLoaded=function(e){this._start+=this._config.chunkSize,this._finished=!this._config.chunkSize||this._start>=this._input.size,this.parseChunk(e.target.result)},this._chunkError=function(){this._sendError(t.error)}}function d(e){var t;l.call(this,e=e||{}),this.stream=function(e){return t=e,this._nextChunk()},this._nextChunk=function(){var e,i;if(!this._finished)return t=(e=this._config.chunkSize)?(i=t.substring(0,e),t.substring(e)):(i=t,""),this._finished=!t,this.parseChunk(i)}}function p(e){l.call(this,e=e||{});var t=[],i=!0,r=!1;this.pause=function(){l.prototype.pause.apply(this,arguments),this._input.pause()},this.resume=function(){l.prototype.resume.apply(this,arguments),this._input.resume()},this.stream=function(e){this._input=e,this._input.on("data",this._streamData),this._input.on("end",this._streamEnd),this._input.on("error",this._streamError)},this._checkIsFinished=function(){r&&1===t.length&&(this._finished=!0)},this._nextChunk=function(){this._checkIsFinished(),t.length?this.parseChunk(t.shift()):i=!0},this._streamData=b(function(e){try{t.push("string"==typeof e?e:e.toString(this._config.encoding)),i&&(i=!1,this._checkIsFinished(),this.parseChunk(t.shift()))}catch(e){this._streamError(e)}},this),this._streamError=b(function(e){this._streamCleanUp(),this._sendError(e)},this),this._streamEnd=b(function(){this._streamCleanUp(),r=!0,this._streamData("")},this),this._streamCleanUp=b(function(){this._input.removeListener("data",this._streamData),this._input.removeListener("end",this._streamEnd),this._input.removeListener("error",this._streamError)},this)}function h(e){var t,i,r,n,a=/^\s*-?(\d+\.?|\.\d+|\d+\.\d+)([eE][-+]?\d+)?\s*$/,s=/^((\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z)))$/,l=this,u=0,c=0,d=!1,p=!1,h=[],g={data:[],errors:[],meta:{}};function _(t){return"greedy"===e.skipEmptyLines?""===t.join("").trim():1===t.length&&0===t[0].length}function y(){if(g&&r&&(w("Delimiter","UndetectableDelimiter","Unable to auto-detect delimiting character; defaulted to '"+o.DefaultDelimiter+"'"),r=!1),e.skipEmptyLines&&(g.data=g.data.filter(function(e){return!_(e)})),b()){if(g)if(Array.isArray(g.data[0])){for(var t,i=0;b()&&i<g.data.length;i++)g.data[i].forEach(n);g.data.splice(0,1)}else g.data.forEach(n);function n(t,i){S(e.transformHeader)&&(t=e.transformHeader(t,i)),h.push(t)}}function l(t,i){for(var r=e.header?{}:[],n=0;n<t.length;n++){var o=n,l=t[n],l=((t,i)=>(e.dynamicTypingFunction&&void 0===e.dynamicTyping[t]&&(e.dynamicTyping[t]=e.dynamicTypingFunction(t)),!0===(e.dynamicTyping[t]||e.dynamicTyping))?"true"===i||"TRUE"===i||"false"!==i&&"FALSE"!==i&&((e=>{if(a.test(e)&&-0x20000000000000<(e=parseFloat(e))&&e<0x20000000000000)return 1})(i)?parseFloat(i):s.test(i)?new Date(i):""===i?null:i):i)(o=e.header?n>=h.length?"__parsed_extra":h[n]:o,l=e.transform?e.transform(l,o):l);"__parsed_extra"===o?(r[o]=r[o]||[],r[o].push(l)):r[o]=l}return e.header&&(n>h.length?w("FieldMismatch","TooManyFields","Too many fields: expected "+h.length+" fields but parsed "+n,c+i):n<h.length&&w("FieldMismatch","TooFewFields","Too few fields: expected "+h.length+" fields but parsed "+n,c+i)),r}g&&(e.header||e.dynamicTyping||e.transform)&&(t=1,!g.data.length||Array.isArray(g.data[0])?(g.data=g.data.map(l),t=g.data.length):g.data=l(g.data,0),e.header&&g.meta&&(g.meta.fields=h),c+=t)}function b(){return e.header&&0===h.length}function w(e,t,i,r){e={type:e,code:t,message:i},void 0!==r&&(e.row=r),g.errors.push(e)}S(e.step)&&(n=e.step,e.step=function(t){g=t,b()?y():(y(),0!==g.data.length&&(u+=t.data.length,e.preview&&u>e.preview?i.abort():(g.data=g.data[0],n(g,l))))}),this.parse=function(n,a,s){var l=e.quoteChar||'"',l=(e.newline||(e.newline=this.guessLineEndings(n,l)),r=!1,e.delimiter?S(e.delimiter)&&(e.delimiter=e.delimiter(n),g.meta.delimiter=e.delimiter):((l=((t,i,r,n,a)=>{var s,l,u,c;a=a||[",","	","|",";",o.RECORD_SEP,o.UNIT_SEP];for(var d=0;d<a.length;d++){for(var p,h=a[d],f=0,g=0,y=0,v=(u=void 0,new m({comments:n,delimiter:h,newline:i,preview:10}).parse(t)),b=0;b<v.data.length;b++)r&&_(v.data[b])?y++:(g+=p=v.data[b].length,void 0===u?u=p:0<p&&(f+=Math.abs(p-u),u=p));0<v.data.length&&(g/=v.data.length-y),(void 0===l||f<=l)&&(void 0===c||c<g)&&1.99<g&&(l=f,s=h,c=g)}return{successful:!!(e.delimiter=s),bestDelimiter:s}})(n,e.newline,e.skipEmptyLines,e.comments,e.delimitersToGuess)).successful?e.delimiter=l.bestDelimiter:(r=!0,e.delimiter=o.DefaultDelimiter),g.meta.delimiter=e.delimiter),v(e));return e.preview&&e.header&&l.preview++,t=n,g=(i=new m(l)).parse(t,a,s),y(),d?{meta:{paused:!0}}:g||{meta:{paused:!1}}},this.paused=function(){return d},this.pause=function(){d=!0,i.abort(),t=S(e.chunk)?"":t.substring(i.getCharIndex())},this.resume=function(){l.streamer._halted?(d=!1,l.streamer.parseChunk(t,!0)):setTimeout(l.resume,3)},this.aborted=function(){return p},this.abort=function(){p=!0,i.abort(),g.meta.aborted=!0,S(e.complete)&&e.complete(g),t=""},this.guessLineEndings=function(e,t){e=e.substring(0,1048576);var t=RegExp(f(t)+"([^]*?)"+f(t),"gm"),i=(e=e.replace(t,"")).split("\r"),t=e.split("\n"),e=1<t.length&&t[0].length<i[0].length;if(1===i.length||e)return"\n";for(var r=0,n=0;n<i.length;n++)"\n"===i[n][0]&&r++;return r>=i.length/2?"\r\n":"\r"}}function f(e){return e.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}function m(e){var t=(e=e||{}).delimiter,i=e.newline,r=e.comments,n=e.step,a=e.preview,s=e.fastMode,l=null,u=!1,c=null==e.quoteChar?'"':e.quoteChar,d=c;if(void 0!==e.escapeChar&&(d=e.escapeChar),("string"!=typeof t||-1<o.BAD_DELIMITERS.indexOf(t))&&(t=","),r===t)throw Error("Comment character same as delimiter");!0===r?r="#":("string"!=typeof r||-1<o.BAD_DELIMITERS.indexOf(r))&&(r=!1),"\n"!==i&&"\r"!==i&&"\r\n"!==i&&(i="\n");var p=0,h=!1;this.parse=function(o,m,g){if("string"!=typeof o)throw Error("Input must be a string");var _=o.length,y=t.length,v=i.length,b=r.length,w=S(n),E=[],x=[],k=[],O=p=0;if(!o)return N();if(s||!1!==s&&-1===o.indexOf(c)){for(var C=o.split(i),R=0;R<C.length;R++){if(k=C[R],p+=k.length,R!==C.length-1)p+=i.length;else if(g)break;if(!r||k.substring(0,b)!==r){if(w){if(E=[],F(k.split(t)),D(),h)return N()}else F(k.split(t));if(a&&a<=R)return E=E.slice(0,a),N(!0)}}return N()}for(var I=o.indexOf(t,p),j=o.indexOf(i,p),A=RegExp(f(d)+f(c),"g"),T=o.indexOf(c,p);;)if(o[p]===c)for(T=p,p++;;){if(-1===(T=o.indexOf(c,T+1)))return g||x.push({type:"Quotes",code:"MissingQuotes",message:"Quoted field unterminated",row:E.length,index:p}),P();if(T===_-1)return P(o.substring(p,T).replace(A,c));if(c===d&&o[T+1]===d)T++;else if(c===d||0===T||o[T-1]!==d){-1!==I&&I<T+1&&(I=o.indexOf(t,T+1));var z=M(-1===(j=-1!==j&&j<T+1?o.indexOf(i,T+1):j)?I:Math.min(I,j));if(o.substr(T+1+z,y)===t){k.push(o.substring(p,T).replace(A,c)),o[p=T+1+z+y]!==c&&(T=o.indexOf(c,p)),I=o.indexOf(t,p),j=o.indexOf(i,p);break}if(z=M(j),o.substring(T+1+z,T+1+z+v)===i){if(k.push(o.substring(p,T).replace(A,c)),L(T+1+z+v),I=o.indexOf(t,p),T=o.indexOf(c,p),w&&(D(),h))return N();if(a&&E.length>=a)return N(!0);break}x.push({type:"Quotes",code:"InvalidQuotes",message:"Trailing quote on quoted field is malformed",row:E.length,index:p}),T++}}else if(r&&0===k.length&&o.substring(p,p+b)===r){if(-1===j)return N();p=j+v,j=o.indexOf(i,p),I=o.indexOf(t,p)}else if(-1!==I&&(I<j||-1===j))k.push(o.substring(p,I)),p=I+y,I=o.indexOf(t,p);else{if(-1===j)break;if(k.push(o.substring(p,j)),L(j+v),w&&(D(),h))return N();if(a&&E.length>=a)return N(!0)}return P();function F(e){E.push(e),O=p}function M(e){return -1!==e&&(e=o.substring(T+1,e))&&""===e.trim()?e.length:0}function P(e){return g||(void 0===e&&(e=o.substring(p)),k.push(e),p=_,F(k),w&&D()),N()}function L(e){p=e,F(k),k=[],j=o.indexOf(i,p)}function N(r){if(e.header&&!m&&E.length&&!u){var n=E[0],a=Object.create(null),s=new Set(n);let t=!1;for(let i=0;i<n.length;i++){let r=n[i];if(a[r=S(e.transformHeader)?e.transformHeader(r,i):r]){let e,o=a[r];for(;e=r+"_"+o,o++,s.has(e););s.add(e),n[i]=e,a[r]++,t=!0,(l=null===l?{}:l)[e]=r}else a[r]=1,n[i]=r;s.add(r)}t&&console.warn("Duplicate headers found and renamed."),u=!0}return{data:E,errors:x,meta:{delimiter:t,linebreak:i,aborted:h,truncated:!!r,cursor:O+(m||0),renamedHeaders:l}}}function D(){n(N()),E=[],x=[]}},this.abort=function(){h=!0},this.getCharIndex=function(){return p}}function g(e){var t=e.data,i=a[t.workerId],r=!1;if(t.error)i.userError(t.error,t.file);else if(t.results&&t.results.data){var n={abort:function(){r=!0,_(t.workerId,{data:[],errors:[],meta:{aborted:!0}})},pause:y,resume:y};if(S(i.userStep)){for(var s=0;s<t.results.data.length&&(i.userStep({data:t.results.data[s],errors:t.results.errors,meta:t.results.meta},n),!r);s++);delete t.results}else S(i.userChunk)&&(i.userChunk(t.results,n,t.file),delete t.results)}t.finished&&!r&&_(t.workerId,t.results)}function _(e,t){var i=a[e];S(i.userComplete)&&i.userComplete(t),i.terminate(),delete a[e]}function y(){throw Error("Not implemented.")}function v(e){if("object"!=typeof e||null===e)return e;var t,i=Array.isArray(e)?[]:{};for(t in e)i[t]=v(e[t]);return i}function b(e,t){return function(){e.apply(t,arguments)}}function S(e){return"function"==typeof e}return o.parse=function(t,r){var n,l,h,f=(r=r||{}).dynamicTyping||!1;if(S(f)&&(r.dynamicTypingFunction=f,f={}),r.dynamicTyping=f,r.transform=!!S(r.transform)&&r.transform,!r.worker||!o.WORKERS_SUPPORTED){let e;return f=null,o.NODE_STREAM_INPUT,"string"==typeof t?(t=65279!==(e=t).charCodeAt(0)?e:e.slice(1),f=new(r.download?u:d)(r)):!0===t.readable&&S(t.read)&&S(t.on)?f=new p(r):(i.File&&t instanceof File||t instanceof Object)&&(f=new c(r)),f.stream(t)}(f=!!o.WORKERS_SUPPORTED&&(l=i.URL||i.webkitURL||null,h=e.toString(),n=o.BLOB_URL||(o.BLOB_URL=l.createObjectURL(new Blob(["var global = (function() { if (typeof self !== 'undefined') { return self; } if (typeof window !== 'undefined') { return window; } if (typeof global !== 'undefined') { return global; } return {}; })(); global.IS_PAPA_WORKER=true; ","(",h,")();"],{type:"text/javascript"}))),(n=new i.Worker(n)).onmessage=g,n.id=s++,a[n.id]=n)).userStep=r.step,f.userChunk=r.chunk,f.userComplete=r.complete,f.userError=r.error,r.step=S(r.step),r.chunk=S(r.chunk),r.complete=S(r.complete),r.error=S(r.error),delete r.worker,f.postMessage({input:t,config:r,workerId:f.id})},o.unparse=function(e,t){var i=!1,r=!0,n=",",a="\r\n",s='"',l=s+s,u=!1,c=null,d=!1,p=((()=>{if("object"==typeof t){if("string"!=typeof t.delimiter||o.BAD_DELIMITERS.filter(function(e){return -1!==t.delimiter.indexOf(e)}).length||(n=t.delimiter),("boolean"==typeof t.quotes||"function"==typeof t.quotes||Array.isArray(t.quotes))&&(i=t.quotes),"boolean"!=typeof t.skipEmptyLines&&"string"!=typeof t.skipEmptyLines||(u=t.skipEmptyLines),"string"==typeof t.newline&&(a=t.newline),"string"==typeof t.quoteChar&&(s=t.quoteChar),"boolean"==typeof t.header&&(r=t.header),Array.isArray(t.columns)){if(0===t.columns.length)throw Error("Option columns is empty");c=t.columns}void 0!==t.escapeChar&&(l=t.escapeChar+s),t.escapeFormulae instanceof RegExp?d=t.escapeFormulae:"boolean"==typeof t.escapeFormulae&&t.escapeFormulae&&(d=/^[=+\-@\t\r].*$/)}})(),RegExp(f(s),"g"));if("string"==typeof e&&(e=JSON.parse(e)),Array.isArray(e)){if(!e.length||Array.isArray(e[0]))return h(null,e,u);if("object"==typeof e[0])return h(c||Object.keys(e[0]),e,u)}else if("object"==typeof e)return"string"==typeof e.data&&(e.data=JSON.parse(e.data)),Array.isArray(e.data)&&(e.fields||(e.fields=e.meta&&e.meta.fields||c),e.fields||(e.fields=Array.isArray(e.data[0])?e.fields:"object"==typeof e.data[0]?Object.keys(e.data[0]):[]),Array.isArray(e.data[0])||"object"==typeof e.data[0]||(e.data=[e.data])),h(e.fields||[],e.data||[],u);throw Error("Unable to serialize unrecognized input");function h(e,t,i){var s="",o=("string"==typeof e&&(e=JSON.parse(e)),"string"==typeof t&&(t=JSON.parse(t)),Array.isArray(e)&&0<e.length),l=!Array.isArray(t[0]);if(o&&r){for(var u=0;u<e.length;u++)0<u&&(s+=n),s+=m(e[u],u);0<t.length&&(s+=a)}for(var c=0;c<t.length;c++){var d=(o?e:t[c]).length,p=!1,h=o?0===Object.keys(t[c]).length:0===t[c].length;if(i&&!o&&(p="greedy"===i?""===t[c].join("").trim():1===t[c].length&&0===t[c][0].length),"greedy"===i&&o){for(var f=[],g=0;g<d;g++){var _=l?e[g]:g;f.push(t[c][_])}p=""===f.join("").trim()}if(!p){for(var y=0;y<d;y++){0<y&&!h&&(s+=n);var v=o&&l?e[y]:y;s+=m(t[c][v],y)}c<t.length-1&&(!i||0<d&&!h)&&(s+=a)}}return s}function m(e,t){var r,a;return null==e?"":e.constructor===Date?JSON.stringify(e).slice(1,25):(a=!1,d&&"string"==typeof e&&d.test(e)&&(e="'"+e,a=!0),r=e.toString().replace(p,l),(a=a||!0===i||"function"==typeof i&&i(e,t)||Array.isArray(i)&&i[t]||((e,t)=>{for(var i=0;i<t.length;i++)if(-1<e.indexOf(t[i]))return!0;return!1})(r,o.BAD_DELIMITERS)||-1<r.indexOf(n)||" "===r.charAt(0)||" "===r.charAt(r.length-1))?s+r+s:r)}},o.RECORD_SEP="\x1e",o.UNIT_SEP="\x1f",o.BYTE_ORDER_MARK="\uFEFF",o.BAD_DELIMITERS=["\r","\n",'"',o.BYTE_ORDER_MARK],o.WORKERS_SUPPORTED=!r&&!!i.Worker,o.NODE_STREAM_INPUT=1,o.LocalChunkSize=0xa00000,o.RemoteChunkSize=5242880,o.DefaultDelimiter=",",o.Parser=m,o.ParserHandle=h,o.NetworkStreamer=u,o.FileStreamer=c,o.StringStreamer=d,o.ReadableStreamStreamer=p,i.jQuery&&((t=i.jQuery).fn.parse=function(e){var r=e.config||{},n=[];return this.each(function(e){if(!("INPUT"===t(this).prop("tagName").toUpperCase()&&"file"===t(this).attr("type").toLowerCase()&&i.FileReader)||!this.files||0===this.files.length)return!0;for(var a=0;a<this.files.length;a++)n.push({file:this.files[a],inputElem:this,instanceConfig:t.extend({},r)})}),a(),this;function a(){if(0===n.length)S(e.complete)&&e.complete();else{var i,r,a,l=n[0];if(S(e.before)){var u=e.before(l.file,l.inputElem);if("object"==typeof u){if("abort"===u.action)return i=l.file,r=l.inputElem,a=u.reason,void(S(e.error)&&e.error({name:"AbortError"},i,r,a));if("skip"===u.action)return void s();"object"==typeof u.config&&(l.instanceConfig=t.extend(l.instanceConfig,u.config))}else if("skip"===u)return void s()}var c=l.instanceConfig.complete;l.instanceConfig.complete=function(e){S(c)&&c(e,l.file,l.inputElem),s()},o.parse(l.file,l.instanceConfig)}}function s(){n.splice(0,1),a()}}),n&&(i.onmessage=function(e){e=e.data,void 0===o.WORKER_ID&&e&&(o.WORKER_ID=e.workerId),"string"==typeof e.input?i.postMessage({workerId:o.WORKER_ID,results:o.parse(e.input,e.config),finished:!0}):(i.File&&e.input instanceof File||e.input instanceof Object)&&(e=o.parse(e.input,e.config))&&i.postMessage({workerId:o.WORKER_ID,results:e,finished:!0})}),(u.prototype=Object.create(l.prototype)).constructor=u,(c.prototype=Object.create(l.prototype)).constructor=c,(d.prototype=Object.create(d.prototype)).constructor=d,(p.prototype=Object.create(l.prototype)).constructor=p,o},"function"==typeof define&&define.amd?void 0!==(n=r())&&e.v(n):t.exports=r()},891547,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),n=e.i(764205);e.s(["default",0,({onChange:e,value:a,className:s,accessToken:o,disabled:l})=>{let[u,c]=(0,i.useState)([]),[d,p]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(o){p(!0);try{let e=await (0,n.getGuardrailsList)(o);console.log("Guardrails response:",e),e.guardrails&&(console.log("Guardrails data:",e.guardrails),c(e.guardrails))}catch(e){console.error("Error fetching guardrails:",e)}finally{p(!1)}}})()},[o]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onChange:t=>{console.log("Selected guardrails:",t),e(t)},value:a,loading:d,className:s,allowClear:!0,options:u.map(e=>(console.log("Mapping guardrail:",e),{label:`${e.guardrail_name}`,value:e.guardrail_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},921511,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),n=e.i(764205);e.s(["default",0,({onChange:e,value:a,className:s,accessToken:o,disabled:l})=>{let[u,c]=(0,i.useState)([]),[d,p]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(o){p(!0);try{let e=await (0,n.getPoliciesList)(o);console.log("Policies response:",e),e.policies&&(console.log("Policies data:",e.policies),c(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{p(!1)}}})()},[o]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting policies is a premium feature.":"Select policies",onChange:t=>{console.log("Selected policies:",t),e(t)},value:a,loading:d,className:s,allowClear:!0,options:u.map(e=>(console.log("Mapping policy:",e),{label:`${e.policy_name}${e.description?` - ${e.description}`:""}`,value:e.policy_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},367240,54943,555436,e=>{"use strict";var t=e.i(475254);let i=(0,t.default)("rotate-ccw",[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]]);e.s(["RotateCcw",()=>i],367240);let r=(0,t.default)("search",[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]]);e.s(["default",()=>r],54943),e.s(["Search",()=>r],555436)},646563,e=>{"use strict";var t=e.i(959013);e.s(["PlusOutlined",()=>t.default])},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["CodeOutlined",0,a],245094)},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["CheckCircleOutlined",0,a],245704)},431343,569074,e=>{"use strict";var t=e.i(475254);let i=(0,t.default)("play",[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]]);e.s(["Play",()=>i],431343);let r=(0,t.default)("upload",[["path",{d:"M12 3v12",key:"1x0j5s"}],["path",{d:"m17 8-5-5-5 5",key:"7q97r8"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}]]);e.s(["Upload",()=>r],569074)},98919,e=>{"use strict";var t=e.i(918549);e.s(["Shield",()=>t.default])},727612,e=>{"use strict";let t=(0,e.i(475254).default)("trash-2",[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]]);e.s(["Trash2",()=>t],727612)},918549,e=>{"use strict";let t=(0,e.i(475254).default)("shield",[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}]]);e.s(["default",()=>t])},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),r=e.i(914949),n=e.i(404948);let a=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,a],836938);var s=e.i(613541),o=e.i(763731),l=e.i(242064),u=e.i(491816);e.i(793154);var c=e.i(880476),d=e.i(183293),p=e.i(717356),h=e.i(320560),f=e.i(307358),m=e.i(246422),g=e.i(838378),_=e.i(617933);let y=(0,m.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:i}=e,r=(0,g.mergeToken)(e,{popoverBg:t,popoverColor:i});return[(e=>{let{componentCls:t,popoverColor:i,titleMinWidth:r,fontWeightStrong:n,innerPadding:a,boxShadowSecondary:s,colorTextHeading:o,borderRadiusLG:l,zIndexPopup:u,titleMarginBottom:c,colorBgElevated:p,popoverBg:f,titleBorderBottom:m,innerContentPadding:g,titlePadding:_}=e;return[{[t]:Object.assign(Object.assign({},(0,d.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:u,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":p,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:f,backgroundClip:"padding-box",borderRadius:l,boxShadow:s,padding:a},[`${t}-title`]:{minWidth:r,marginBottom:c,color:o,fontWeight:n,borderBottom:m,padding:_},[`${t}-inner-content`]:{color:i,padding:g}})},(0,h.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(r),(e=>{let{componentCls:t}=e;return{[t]:_.PresetColors.map(i=>{let r=e[`${i}6`];return{[`&${t}-${i}`]:{"--antd-arrow-background-color":r,[`${t}-inner`]:{backgroundColor:r},[`${t}-arrow`]:{background:"transparent"}}}})}})(r),(0,p.initZoomMotion)(r,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:i,fontHeight:r,padding:n,wireframe:a,zIndexPopupBase:s,borderRadiusLG:o,marginXS:l,lineType:u,colorSplit:c,paddingSM:d}=e,p=i-r;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:s+30},(0,f.getArrowToken)(e)),(0,h.getArrowOffsetToken)({contentRadius:o,limitVerticalRadius:!0})),{innerPadding:12*!a,titleMarginBottom:a?0:l,titlePadding:a?`${p/2}px ${n}px ${p/2-t}px`:0,titleBorderBottom:a?`${t}px ${u} ${c}`:"none",innerContentPadding:a?`${d}px ${n}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var v=function(e,t){var i={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(i[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,r=Object.getOwnPropertySymbols(e);n<r.length;n++)0>t.indexOf(r[n])&&Object.prototype.propertyIsEnumerable.call(e,r[n])&&(i[r[n]]=e[r[n]]);return i};let b=({title:e,content:i,prefixCls:r})=>e||i?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${r}-title`},e),i&&t.createElement("div",{className:`${r}-inner-content`},i)):null,S=e=>{let{hashId:r,prefixCls:n,className:s,style:o,placement:l="top",title:u,content:d,children:p}=e,h=a(u),f=a(d),m=(0,i.default)(r,n,`${n}-pure`,`${n}-placement-${l}`,s);return t.createElement("div",{className:m,style:o},t.createElement("div",{className:`${n}-arrow`}),t.createElement(c.Popup,Object.assign({},e,{className:r,prefixCls:n}),p||t.createElement(b,{prefixCls:n,title:h,content:f})))},w=e=>{let{prefixCls:r,className:n}=e,a=v(e,["prefixCls","className"]),{getPrefixCls:s}=t.useContext(l.ConfigContext),o=s("popover",r),[u,c,d]=y(o);return u(t.createElement(S,Object.assign({},a,{prefixCls:o,hashId:c,className:(0,i.default)(n,d)})))};e.s(["Overlay",0,b,"default",0,w],310730);var E=function(e,t){var i={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(i[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,r=Object.getOwnPropertySymbols(e);n<r.length;n++)0>t.indexOf(r[n])&&Object.prototype.propertyIsEnumerable.call(e,r[n])&&(i[r[n]]=e[r[n]]);return i};let x=t.forwardRef((e,c)=>{var d,p;let{prefixCls:h,title:f,content:m,overlayClassName:g,placement:_="top",trigger:v="hover",children:S,mouseEnterDelay:w=.1,mouseLeaveDelay:x=.1,onOpenChange:k,overlayStyle:O={},styles:C,classNames:R}=e,I=E(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:j,className:A,style:T,classNames:z,styles:F}=(0,l.useComponentConfig)("popover"),M=j("popover",h),[P,L,N]=y(M),D=j(),$=(0,i.default)(g,L,N,A,z.root,null==R?void 0:R.root),H=(0,i.default)(z.body,null==R?void 0:R.body),[U,B]=(0,r.default)(!1,{value:null!=(d=e.open)?d:e.visible,defaultValue:null!=(p=e.defaultOpen)?p:e.defaultVisible}),q=(e,t)=>{B(e,!0),null==k||k(e,t)},V=a(f),W=a(m);return P(t.createElement(u.default,Object.assign({placement:_,trigger:v,mouseEnterDelay:w,mouseLeaveDelay:x},I,{prefixCls:M,classNames:{root:$,body:H},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},F.root),T),O),null==C?void 0:C.root),body:Object.assign(Object.assign({},F.body),null==C?void 0:C.body)},ref:c,open:U,onOpenChange:e=>{q(e)},overlay:V||W?t.createElement(b,{prefixCls:M,title:V,content:W}):null,transitionName:(0,s.getTransitionName)(D,"zoom-big",I.transitionName),"data-popover-inject":!0}),(0,o.cloneElement)(S,{onKeyDown:e=>{var i,r;(0,t.isValidElement)(S)&&(null==(r=null==S?void 0:(i=S.props).onKeyDown)||r.call(i,e)),e.keyCode===n.default.ESC&&q(!1,e)}})))});x._InternalPanelDoNotUseOrYouWillBeFired=w,e.s(["default",0,x],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},872934,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM770.87 199.13l-52.2-52.2a8.01 8.01 0 014.7-13.6l179.4-21c5.1-.6 9.5 3.7 8.9 8.9l-21 179.4c-.8 6.6-8.9 9.4-13.6 4.7l-52.4-52.4-256.2 256.2a8.03 8.03 0 01-11.3 0l-42.4-42.4a8.03 8.03 0 010-11.3l256.1-256.3z"}}]},name:"export",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["ExportOutlined",0,a],872934)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["DollarOutlined",0,a],458505)},903446,e=>{"use strict";let t=(0,e.i(475254).default)("settings",[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["default",()=>t])},190272,785913,e=>{"use strict";var t,i,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i);let a={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=a[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:r,apiKey:a,inputMessage:s,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:c,selectedPolicies:d,selectedMCPServers:p,mcpServers:h,mcpServerToolRestrictions:f,selectedVoice:m,endpointType:g,selectedModel:_,selectedSdk:y,proxySettings:v}=e,b="session"===i?r:a,S=window.location.origin,w=v?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?S=w:v?.PROXY_BASE_URL&&(S=v.PROXY_BASE_URL);let E=s||"Your prompt here",x=E.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),k=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),O={};l.length>0&&(O.tags=l),u.length>0&&(O.vector_stores=u),c.length>0&&(O.guardrails=c),d.length>0&&(O.policies=d);let C=_||"your-model-name",R="azure"===y?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${S}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${S}"
)`;switch(g){case n.CHAT:{let e=Object.keys(O).length>0,i="";if(e){let e=JSON.stringify({metadata:O},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=k.length>0?k:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(r,null,4)}${i}
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
#     ]${i}
# )
# print(response_with_file)
`;break}case n.RESPONSES:{let e=Object.keys(O).length>0,i="";if(e){let e=JSON.stringify({metadata:O},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=k.length>0?k:[{role:"user",content:E}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(r,null,4)}${i}
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
#                 {"type": "input_text", "text": "${x}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===y?`
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
prompt = "${x}"

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
`;break;case n.IMAGE_EDITS:t="azure"===y?`
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
prompt = "${x}"

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
	input="${s||"Your string here"}",
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
	file=audio_file${s?`,
	prompt="${s.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${s||"Your text to convert to speech here"}",
	voice="${m}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${R}
${t}`}],190272)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["LinkOutlined",0,a],596239)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var n=e.i(9583),a=i.forwardRef(function(e,a){return i.createElement(n.default,(0,t.default)({},e,{ref:a,icon:r}))});e.s(["ClockCircleOutlined",0,a],637235)},516015,(e,t,i)=>{},898547,(e,t,i)=>{var r=e.i(247167);e.r(516015);var n=e.r(271645),a=n&&"object"==typeof n&&"default"in n?n:{default:n},s=void 0!==r.default&&r.default.env&&!0,o=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,i=t.name,r=void 0===i?"stylesheet":i,n=t.optimizeForSpeed,a=void 0===n?s:n;u(o(r),"`name` must be a string"),this._name=r,this._deletedRulePlaceholder="#"+r+"-deleted-rule____{}",u("boolean"==typeof a,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=a,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,i=e.prototype;return i.setOptimizeForSpeed=function(e){u("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),u(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},i.isOptimizeForSpeed=function(){return this._optimizeForSpeed},i.inject=function(){var e=this;if(u(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(s||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,i){return"number"==typeof i?e._serverSheet.cssRules[i]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),i},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},i.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},i.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},i.insertRule=function(e,t){if(u(o(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var i=this.getSheet();"number"!=typeof t&&(t=i.cssRules.length);try{i.insertRule(e,t)}catch(t){return s||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var r=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,r))}return this._rulesCount++},i.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var i="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!i.cssRules[e])return e;i.deleteRule(e);try{i.insertRule(t,e)}catch(r){s||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),i.insertRule(this._deletedRulePlaceholder,e)}}else{var r=this._tags[e];u(r,"old rule at index `"+e+"` not found"),r.textContent=t}return e},i.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];u(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},i.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},i.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,i){return i?t=t.concat(Array.prototype.map.call(e.getSheetForTag(i).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},i.makeStyleTag=function(e,t,i){t&&u(o(t),"makeStyleTag accepts only strings as second parameter");var r=document.createElement("style");this._nonce&&r.setAttribute("nonce",this._nonce),r.type="text/css",r.setAttribute("data-"+e,""),t&&r.appendChild(document.createTextNode(t));var n=document.head||document.getElementsByTagName("head")[0];return i?n.insertBefore(r,i):n.appendChild(r),r},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var i=0;i<t.length;i++){var r=t[i];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}(e.prototype,t),e}();function u(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var c=function(e){for(var t=5381,i=e.length;i;)t=33*t^e.charCodeAt(--i);return t>>>0},d={};function p(e,t){if(!t)return"jsx-"+e;var i=String(t),r=e+i;return d[r]||(d[r]="jsx-"+c(e+"-"+i)),d[r]}function h(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var i=e+t;return d[i]||(d[i]=t.replace(/__jsx-style-dynamic-selector/g,e)),d[i]}var f=function(){function e(e){var t=void 0===e?{}:e,i=t.styleSheet,r=void 0===i?null:i,n=t.optimizeForSpeed,a=void 0!==n&&n;this._sheet=r||new l({name:"styled-jsx",optimizeForSpeed:a}),this._sheet.inject(),r&&"boolean"==typeof a&&(this._sheet.setOptimizeForSpeed(a),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var i=this.getIdAndRules(e),r=i.styleId,n=i.rules;if(r in this._instancesCounts){this._instancesCounts[r]+=1;return}var a=n.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[r]=a,this._instancesCounts[r]=1},t.remove=function(e){var t=this,i=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(i in this._instancesCounts,"styleId: `"+i+"` not found"),this._instancesCounts[i]-=1,this._instancesCounts[i]<1){var r=this._fromServer&&this._fromServer[i];r?(r.parentNode.removeChild(r),delete this._fromServer[i]):(this._indices[i].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[i]),delete this._instancesCounts[i]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],i=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return i[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,i;return t=this.cssRules(),void 0===(i=e)&&(i={}),t.map(function(e){var t=e[0],r=e[1];return a.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:i.nonce?i.nonce:void 0,dangerouslySetInnerHTML:{__html:r}})})},t.getIdAndRules=function(e){var t=e.children,i=e.dynamic,r=e.id;if(i){var n=p(r,i);return{styleId:n,rules:Array.isArray(t)?t.map(function(e){return h(n,e)}):[h(n,t)]}}return{styleId:p(r),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),m=n.createContext(null);function g(){return new f}function _(){return n.useContext(m)}m.displayName="StyleSheetContext";var y=a.default.useInsertionEffect||a.default.useLayoutEffect,v="u">typeof window?g():void 0;function b(e){var t=v||_();return t&&("u"<typeof window?t.add(e):y(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}b.dynamic=function(e){return e.map(function(e){return p(e[0],e[1])}).join(" ")},i.StyleRegistry=function(e){var t=e.registry,i=e.children,r=n.useContext(m),s=n.useState(function(){return r||t||g()})[0];return a.default.createElement(m.Provider,{value:s},i)},i.createStyleRegistry=g,i.style=b,i.useStyleRegistry=_},437902,(e,t,i)=>{t.exports=e.r(898547).style}]);