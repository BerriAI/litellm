(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},246349,e=>{"use strict";let t=(0,e.i(475254).default)("chevron-right",[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]]);e.s(["default",()=>t])},841947,e=>{"use strict";let t=(0,e.i(475254).default)("x",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]]);e.s(["default",()=>t])},603908,e=>{"use strict";let t=(0,e.i(475254).default)("plus",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]]);e.s(["default",()=>t])},107233,37727,e=>{"use strict";var t=e.i(603908);e.s(["Plus",()=>t.default],107233);var i=e.i(841947);e.s(["X",()=>i.default],37727)},689020,e=>{"use strict";var t=e.i(764205);let i=async e=>{try{let i=await (0,t.modelHubCall)(e);if(console.log("model_info:",i),i?.data.length>0){let e=i.data.map(e=>({model_group:e.model_group,mode:e?.mode}));return e.sort((e,t)=>e.model_group.localeCompare(t.model_group)),e}return[]}catch(e){throw console.error("Error fetching model info:",e),e}};e.s(["fetchAvailableModels",0,i])},737434,e=>{"use strict";var t=e.i(184163);e.s(["DownloadOutlined",()=>t.default])},916940,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),a=e.i(764205);e.s(["default",0,({onChange:e,value:o,className:n,accessToken:s,placeholder:l="Select vector stores",disabled:c=!1})=>{let[d,u]=(0,i.useState)([]),[h,p]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(s){p(!0);try{let e=await (0,a.vectorStoreListCall)(s);e.data&&u(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{p(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",placeholder:l,onChange:e,value:o,loading:h,className:n,allowClear:!0,options:d.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,title:e.vector_store_description||e.vector_store_id})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"},disabled:c})})}])},59935,(e,t,i)=>{var r;let a;e.e,r=function e(){var t,i="u">typeof self?self:"u">typeof window?window:void 0!==i?i:{},r=!i.document&&!!i.postMessage,a=i.IS_PAPA_WORKER||!1,o={},n=0,s={};function l(e){this._handle=null,this._finished=!1,this._completed=!1,this._halted=!1,this._input=null,this._baseIndex=0,this._partialLine="",this._rowCount=0,this._start=0,this._nextChunk=null,this.isFirstChunk=!0,this._completeResults={data:[],errors:[],meta:{}},(function(e){var t=v(e);t.chunkSize=parseInt(t.chunkSize),e.step||e.chunk||(t.chunkSize=null),this._handle=new p(t),(this._handle.streamer=this)._config=t}).call(this,e),this.parseChunk=function(e,t){var r=parseInt(this._config.skipFirstNLines)||0;if(this.isFirstChunk&&0<r){let t=this._config.newline;t||(o=this._config.quoteChar||'"',t=this._handle.guessLineEndings(e,o)),e=[...e.split(t).slice(r)].join(t)}this.isFirstChunk&&x(this._config.beforeFirstChunk)&&void 0!==(o=this._config.beforeFirstChunk(e))&&(e=o),this.isFirstChunk=!1,this._halted=!1;var r=this._partialLine+e,o=(this._partialLine="",this._handle.parse(r,this._baseIndex,!this._finished));if(!this._handle.paused()&&!this._handle.aborted()){if(e=o.meta.cursor,this._finished||(this._partialLine=r.substring(e-this._baseIndex),this._baseIndex=e),o&&o.data&&(this._rowCount+=o.data.length),r=this._finished||this._config.preview&&this._rowCount>=this._config.preview,a)i.postMessage({results:o,workerId:s.WORKER_ID,finished:r});else if(x(this._config.chunk)&&!t){if(this._config.chunk(o,this._handle),this._handle.paused()||this._handle.aborted())return void(this._halted=!0);this._completeResults=o=void 0}return this._config.step||this._config.chunk||(this._completeResults.data=this._completeResults.data.concat(o.data),this._completeResults.errors=this._completeResults.errors.concat(o.errors),this._completeResults.meta=o.meta),this._completed||!r||!x(this._config.complete)||o&&o.meta.aborted||(this._config.complete(this._completeResults,this._input),this._completed=!0),r||o&&o.meta.paused||this._nextChunk(),o}this._halted=!0},this._sendError=function(e){x(this._config.error)?this._config.error(e):a&&this._config.error&&i.postMessage({workerId:s.WORKER_ID,error:e,finished:!1})}}function c(e){var t;(e=e||{}).chunkSize||(e.chunkSize=s.RemoteChunkSize),l.call(this,e),this._nextChunk=r?function(){this._readChunk(),this._chunkLoaded()}:function(){this._readChunk()},this.stream=function(e){this._input=e,this._nextChunk()},this._readChunk=function(){if(this._finished)this._chunkLoaded();else{if(t=new XMLHttpRequest,this._config.withCredentials&&(t.withCredentials=this._config.withCredentials),r||(t.onload=k(this._chunkLoaded,this),t.onerror=k(this._chunkError,this)),t.open(this._config.downloadRequestBody?"POST":"GET",this._input,!r),this._config.downloadRequestHeaders){var e,i,a=this._config.downloadRequestHeaders;for(i in a)t.setRequestHeader(i,a[i])}this._config.chunkSize&&(e=this._start+this._config.chunkSize-1,t.setRequestHeader("Range","bytes="+this._start+"-"+e));try{t.send(this._config.downloadRequestBody)}catch(e){this._chunkError(e.message)}r&&0===t.status&&this._chunkError()}},this._chunkLoaded=function(){let e;4===t.readyState&&(t.status<200||400<=t.status?this._chunkError():(this._start+=this._config.chunkSize||t.responseText.length,this._finished=!this._config.chunkSize||this._start>=(null!==(e=(e=t).getResponseHeader("Content-Range"))?parseInt(e.substring(e.lastIndexOf("/")+1)):-1),this.parseChunk(t.responseText)))},this._chunkError=function(e){e=t.statusText||e,this._sendError(Error(e))}}function d(e){(e=e||{}).chunkSize||(e.chunkSize=s.LocalChunkSize),l.call(this,e);var t,i,r="u">typeof FileReader;this.stream=function(e){this._input=e,i=e.slice||e.webkitSlice||e.mozSlice,r?((t=new FileReader).onload=k(this._chunkLoaded,this),t.onerror=k(this._chunkError,this)):t=new FileReaderSync,this._nextChunk()},this._nextChunk=function(){this._finished||this._config.preview&&!(this._rowCount<this._config.preview)||this._readChunk()},this._readChunk=function(){var e=this._input,a=(this._config.chunkSize&&(a=Math.min(this._start+this._config.chunkSize,this._input.size),e=i.call(e,this._start,a)),t.readAsText(e,this._config.encoding));r||this._chunkLoaded({target:{result:a}})},this._chunkLoaded=function(e){this._start+=this._config.chunkSize,this._finished=!this._config.chunkSize||this._start>=this._input.size,this.parseChunk(e.target.result)},this._chunkError=function(){this._sendError(t.error)}}function u(e){var t;l.call(this,e=e||{}),this.stream=function(e){return t=e,this._nextChunk()},this._nextChunk=function(){var e,i;if(!this._finished)return t=(e=this._config.chunkSize)?(i=t.substring(0,e),t.substring(e)):(i=t,""),this._finished=!t,this.parseChunk(i)}}function h(e){l.call(this,e=e||{});var t=[],i=!0,r=!1;this.pause=function(){l.prototype.pause.apply(this,arguments),this._input.pause()},this.resume=function(){l.prototype.resume.apply(this,arguments),this._input.resume()},this.stream=function(e){this._input=e,this._input.on("data",this._streamData),this._input.on("end",this._streamEnd),this._input.on("error",this._streamError)},this._checkIsFinished=function(){r&&1===t.length&&(this._finished=!0)},this._nextChunk=function(){this._checkIsFinished(),t.length?this.parseChunk(t.shift()):i=!0},this._streamData=k(function(e){try{t.push("string"==typeof e?e:e.toString(this._config.encoding)),i&&(i=!1,this._checkIsFinished(),this.parseChunk(t.shift()))}catch(e){this._streamError(e)}},this),this._streamError=k(function(e){this._streamCleanUp(),this._sendError(e)},this),this._streamEnd=k(function(){this._streamCleanUp(),r=!0,this._streamData("")},this),this._streamCleanUp=k(function(){this._input.removeListener("data",this._streamData),this._input.removeListener("end",this._streamEnd),this._input.removeListener("error",this._streamError)},this)}function p(e){var t,i,r,a,o=/^\s*-?(\d+\.?|\.\d+|\d+\.\d+)([eE][-+]?\d+)?\s*$/,n=/^((\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z)))$/,l=this,c=0,d=0,u=!1,h=!1,p=[],g={data:[],errors:[],meta:{}};function _(t){return"greedy"===e.skipEmptyLines?""===t.join("").trim():1===t.length&&0===t[0].length}function b(){if(g&&r&&(y("Delimiter","UndetectableDelimiter","Unable to auto-detect delimiting character; defaulted to '"+s.DefaultDelimiter+"'"),r=!1),e.skipEmptyLines&&(g.data=g.data.filter(function(e){return!_(e)})),k()){if(g)if(Array.isArray(g.data[0])){for(var t,i=0;k()&&i<g.data.length;i++)g.data[i].forEach(a);g.data.splice(0,1)}else g.data.forEach(a);function a(t,i){x(e.transformHeader)&&(t=e.transformHeader(t,i)),p.push(t)}}function l(t,i){for(var r=e.header?{}:[],a=0;a<t.length;a++){var s=a,l=t[a],l=((t,i)=>(e.dynamicTypingFunction&&void 0===e.dynamicTyping[t]&&(e.dynamicTyping[t]=e.dynamicTypingFunction(t)),!0===(e.dynamicTyping[t]||e.dynamicTyping))?"true"===i||"TRUE"===i||"false"!==i&&"FALSE"!==i&&((e=>{if(o.test(e)&&-0x20000000000000<(e=parseFloat(e))&&e<0x20000000000000)return 1})(i)?parseFloat(i):n.test(i)?new Date(i):""===i?null:i):i)(s=e.header?a>=p.length?"__parsed_extra":p[a]:s,l=e.transform?e.transform(l,s):l);"__parsed_extra"===s?(r[s]=r[s]||[],r[s].push(l)):r[s]=l}return e.header&&(a>p.length?y("FieldMismatch","TooManyFields","Too many fields: expected "+p.length+" fields but parsed "+a,d+i):a<p.length&&y("FieldMismatch","TooFewFields","Too few fields: expected "+p.length+" fields but parsed "+a,d+i)),r}g&&(e.header||e.dynamicTyping||e.transform)&&(t=1,!g.data.length||Array.isArray(g.data[0])?(g.data=g.data.map(l),t=g.data.length):g.data=l(g.data,0),e.header&&g.meta&&(g.meta.fields=p),d+=t)}function k(){return e.header&&0===p.length}function y(e,t,i,r){e={type:e,code:t,message:i},void 0!==r&&(e.row=r),g.errors.push(e)}x(e.step)&&(a=e.step,e.step=function(t){g=t,k()?b():(b(),0!==g.data.length&&(c+=t.data.length,e.preview&&c>e.preview?i.abort():(g.data=g.data[0],a(g,l))))}),this.parse=function(a,o,n){var l=e.quoteChar||'"',l=(e.newline||(e.newline=this.guessLineEndings(a,l)),r=!1,e.delimiter?x(e.delimiter)&&(e.delimiter=e.delimiter(a),g.meta.delimiter=e.delimiter):((l=((t,i,r,a,o)=>{var n,l,c,d;o=o||[",","	","|",";",s.RECORD_SEP,s.UNIT_SEP];for(var u=0;u<o.length;u++){for(var h,p=o[u],m=0,g=0,b=0,v=(c=void 0,new f({comments:a,delimiter:p,newline:i,preview:10}).parse(t)),k=0;k<v.data.length;k++)r&&_(v.data[k])?b++:(g+=h=v.data[k].length,void 0===c?c=h:0<h&&(m+=Math.abs(h-c),c=h));0<v.data.length&&(g/=v.data.length-b),(void 0===l||m<=l)&&(void 0===d||d<g)&&1.99<g&&(l=m,n=p,d=g)}return{successful:!!(e.delimiter=n),bestDelimiter:n}})(a,e.newline,e.skipEmptyLines,e.comments,e.delimitersToGuess)).successful?e.delimiter=l.bestDelimiter:(r=!0,e.delimiter=s.DefaultDelimiter),g.meta.delimiter=e.delimiter),v(e));return e.preview&&e.header&&l.preview++,t=a,g=(i=new f(l)).parse(t,o,n),b(),u?{meta:{paused:!0}}:g||{meta:{paused:!1}}},this.paused=function(){return u},this.pause=function(){u=!0,i.abort(),t=x(e.chunk)?"":t.substring(i.getCharIndex())},this.resume=function(){l.streamer._halted?(u=!1,l.streamer.parseChunk(t,!0)):setTimeout(l.resume,3)},this.aborted=function(){return h},this.abort=function(){h=!0,i.abort(),g.meta.aborted=!0,x(e.complete)&&e.complete(g),t=""},this.guessLineEndings=function(e,t){e=e.substring(0,1048576);var t=RegExp(m(t)+"([^]*?)"+m(t),"gm"),i=(e=e.replace(t,"")).split("\r"),t=e.split("\n"),e=1<t.length&&t[0].length<i[0].length;if(1===i.length||e)return"\n";for(var r=0,a=0;a<i.length;a++)"\n"===i[a][0]&&r++;return r>=i.length/2?"\r\n":"\r"}}function m(e){return e.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}function f(e){var t=(e=e||{}).delimiter,i=e.newline,r=e.comments,a=e.step,o=e.preview,n=e.fastMode,l=null,c=!1,d=null==e.quoteChar?'"':e.quoteChar,u=d;if(void 0!==e.escapeChar&&(u=e.escapeChar),("string"!=typeof t||-1<s.BAD_DELIMITERS.indexOf(t))&&(t=","),r===t)throw Error("Comment character same as delimiter");!0===r?r="#":("string"!=typeof r||-1<s.BAD_DELIMITERS.indexOf(r))&&(r=!1),"\n"!==i&&"\r"!==i&&"\r\n"!==i&&(i="\n");var h=0,p=!1;this.parse=function(s,f,g){if("string"!=typeof s)throw Error("Input must be a string");var _=s.length,b=t.length,v=i.length,k=r.length,y=x(a),w=[],C=[],j=[],E=h=0;if(!s)return P();if(n||!1!==n&&-1===s.indexOf(d)){for(var S=s.split(i),O=0;O<S.length;O++){if(j=S[O],h+=j.length,O!==S.length-1)h+=i.length;else if(g)break;if(!r||j.substring(0,k)!==r){if(y){if(w=[],N(j.split(t)),D(),p)return P()}else N(j.split(t));if(o&&o<=O)return w=w.slice(0,o),P(!0)}}return P()}for(var R=s.indexOf(t,h),I=s.indexOf(i,h),T=RegExp(m(u)+m(d),"g"),M=s.indexOf(d,h);;)if(s[h]===d)for(M=h,h++;;){if(-1===(M=s.indexOf(d,M+1)))return g||C.push({type:"Quotes",code:"MissingQuotes",message:"Quoted field unterminated",row:w.length,index:h}),L();if(M===_-1)return L(s.substring(h,M).replace(T,d));if(d===u&&s[M+1]===u)M++;else if(d===u||0===M||s[M-1]!==u){-1!==R&&R<M+1&&(R=s.indexOf(t,M+1));var z=A(-1===(I=-1!==I&&I<M+1?s.indexOf(i,M+1):I)?R:Math.min(R,I));if(s.substr(M+1+z,b)===t){j.push(s.substring(h,M).replace(T,d)),s[h=M+1+z+b]!==d&&(M=s.indexOf(d,h)),R=s.indexOf(t,h),I=s.indexOf(i,h);break}if(z=A(I),s.substring(M+1+z,M+1+z+v)===i){if(j.push(s.substring(h,M).replace(T,d)),H(M+1+z+v),R=s.indexOf(t,h),M=s.indexOf(d,h),y&&(D(),p))return P();if(o&&w.length>=o)return P(!0);break}C.push({type:"Quotes",code:"InvalidQuotes",message:"Trailing quote on quoted field is malformed",row:w.length,index:h}),M++}}else if(r&&0===j.length&&s.substring(h,h+k)===r){if(-1===I)return P();h=I+v,I=s.indexOf(i,h),R=s.indexOf(t,h)}else if(-1!==R&&(R<I||-1===I))j.push(s.substring(h,R)),h=R+b,R=s.indexOf(t,h);else{if(-1===I)break;if(j.push(s.substring(h,I)),H(I+v),y&&(D(),p))return P();if(o&&w.length>=o)return P(!0)}return L();function N(e){w.push(e),E=h}function A(e){return -1!==e&&(e=s.substring(M+1,e))&&""===e.trim()?e.length:0}function L(e){return g||(void 0===e&&(e=s.substring(h)),j.push(e),h=_,N(j),y&&D()),P()}function H(e){h=e,N(j),j=[],I=s.indexOf(i,h)}function P(r){if(e.header&&!f&&w.length&&!c){var a=w[0],o=Object.create(null),n=new Set(a);let t=!1;for(let i=0;i<a.length;i++){let r=a[i];if(o[r=x(e.transformHeader)?e.transformHeader(r,i):r]){let e,s=o[r];for(;e=r+"_"+s,s++,n.has(e););n.add(e),a[i]=e,o[r]++,t=!0,(l=null===l?{}:l)[e]=r}else o[r]=1,a[i]=r;n.add(r)}t&&console.warn("Duplicate headers found and renamed."),c=!0}return{data:w,errors:C,meta:{delimiter:t,linebreak:i,aborted:p,truncated:!!r,cursor:E+(f||0),renamedHeaders:l}}}function D(){a(P()),w=[],C=[]}},this.abort=function(){p=!0},this.getCharIndex=function(){return h}}function g(e){var t=e.data,i=o[t.workerId],r=!1;if(t.error)i.userError(t.error,t.file);else if(t.results&&t.results.data){var a={abort:function(){r=!0,_(t.workerId,{data:[],errors:[],meta:{aborted:!0}})},pause:b,resume:b};if(x(i.userStep)){for(var n=0;n<t.results.data.length&&(i.userStep({data:t.results.data[n],errors:t.results.errors,meta:t.results.meta},a),!r);n++);delete t.results}else x(i.userChunk)&&(i.userChunk(t.results,a,t.file),delete t.results)}t.finished&&!r&&_(t.workerId,t.results)}function _(e,t){var i=o[e];x(i.userComplete)&&i.userComplete(t),i.terminate(),delete o[e]}function b(){throw Error("Not implemented.")}function v(e){if("object"!=typeof e||null===e)return e;var t,i=Array.isArray(e)?[]:{};for(t in e)i[t]=v(e[t]);return i}function k(e,t){return function(){e.apply(t,arguments)}}function x(e){return"function"==typeof e}return s.parse=function(t,r){var a,l,p,m=(r=r||{}).dynamicTyping||!1;if(x(m)&&(r.dynamicTypingFunction=m,m={}),r.dynamicTyping=m,r.transform=!!x(r.transform)&&r.transform,!r.worker||!s.WORKERS_SUPPORTED){let e;return m=null,s.NODE_STREAM_INPUT,"string"==typeof t?(t=65279!==(e=t).charCodeAt(0)?e:e.slice(1),m=new(r.download?c:u)(r)):!0===t.readable&&x(t.read)&&x(t.on)?m=new h(r):(i.File&&t instanceof File||t instanceof Object)&&(m=new d(r)),m.stream(t)}(m=!!s.WORKERS_SUPPORTED&&(l=i.URL||i.webkitURL||null,p=e.toString(),a=s.BLOB_URL||(s.BLOB_URL=l.createObjectURL(new Blob(["var global = (function() { if (typeof self !== 'undefined') { return self; } if (typeof window !== 'undefined') { return window; } if (typeof global !== 'undefined') { return global; } return {}; })(); global.IS_PAPA_WORKER=true; ","(",p,")();"],{type:"text/javascript"}))),(a=new i.Worker(a)).onmessage=g,a.id=n++,o[a.id]=a)).userStep=r.step,m.userChunk=r.chunk,m.userComplete=r.complete,m.userError=r.error,r.step=x(r.step),r.chunk=x(r.chunk),r.complete=x(r.complete),r.error=x(r.error),delete r.worker,m.postMessage({input:t,config:r,workerId:m.id})},s.unparse=function(e,t){var i=!1,r=!0,a=",",o="\r\n",n='"',l=n+n,c=!1,d=null,u=!1,h=((()=>{if("object"==typeof t){if("string"!=typeof t.delimiter||s.BAD_DELIMITERS.filter(function(e){return -1!==t.delimiter.indexOf(e)}).length||(a=t.delimiter),("boolean"==typeof t.quotes||"function"==typeof t.quotes||Array.isArray(t.quotes))&&(i=t.quotes),"boolean"!=typeof t.skipEmptyLines&&"string"!=typeof t.skipEmptyLines||(c=t.skipEmptyLines),"string"==typeof t.newline&&(o=t.newline),"string"==typeof t.quoteChar&&(n=t.quoteChar),"boolean"==typeof t.header&&(r=t.header),Array.isArray(t.columns)){if(0===t.columns.length)throw Error("Option columns is empty");d=t.columns}void 0!==t.escapeChar&&(l=t.escapeChar+n),t.escapeFormulae instanceof RegExp?u=t.escapeFormulae:"boolean"==typeof t.escapeFormulae&&t.escapeFormulae&&(u=/^[=+\-@\t\r].*$/)}})(),RegExp(m(n),"g"));if("string"==typeof e&&(e=JSON.parse(e)),Array.isArray(e)){if(!e.length||Array.isArray(e[0]))return p(null,e,c);if("object"==typeof e[0])return p(d||Object.keys(e[0]),e,c)}else if("object"==typeof e)return"string"==typeof e.data&&(e.data=JSON.parse(e.data)),Array.isArray(e.data)&&(e.fields||(e.fields=e.meta&&e.meta.fields||d),e.fields||(e.fields=Array.isArray(e.data[0])?e.fields:"object"==typeof e.data[0]?Object.keys(e.data[0]):[]),Array.isArray(e.data[0])||"object"==typeof e.data[0]||(e.data=[e.data])),p(e.fields||[],e.data||[],c);throw Error("Unable to serialize unrecognized input");function p(e,t,i){var n="",s=("string"==typeof e&&(e=JSON.parse(e)),"string"==typeof t&&(t=JSON.parse(t)),Array.isArray(e)&&0<e.length),l=!Array.isArray(t[0]);if(s&&r){for(var c=0;c<e.length;c++)0<c&&(n+=a),n+=f(e[c],c);0<t.length&&(n+=o)}for(var d=0;d<t.length;d++){var u=(s?e:t[d]).length,h=!1,p=s?0===Object.keys(t[d]).length:0===t[d].length;if(i&&!s&&(h="greedy"===i?""===t[d].join("").trim():1===t[d].length&&0===t[d][0].length),"greedy"===i&&s){for(var m=[],g=0;g<u;g++){var _=l?e[g]:g;m.push(t[d][_])}h=""===m.join("").trim()}if(!h){for(var b=0;b<u;b++){0<b&&!p&&(n+=a);var v=s&&l?e[b]:b;n+=f(t[d][v],b)}d<t.length-1&&(!i||0<u&&!p)&&(n+=o)}}return n}function f(e,t){var r,o;return null==e?"":e.constructor===Date?JSON.stringify(e).slice(1,25):(o=!1,u&&"string"==typeof e&&u.test(e)&&(e="'"+e,o=!0),r=e.toString().replace(h,l),(o=o||!0===i||"function"==typeof i&&i(e,t)||Array.isArray(i)&&i[t]||((e,t)=>{for(var i=0;i<t.length;i++)if(-1<e.indexOf(t[i]))return!0;return!1})(r,s.BAD_DELIMITERS)||-1<r.indexOf(a)||" "===r.charAt(0)||" "===r.charAt(r.length-1))?n+r+n:r)}},s.RECORD_SEP="\x1e",s.UNIT_SEP="\x1f",s.BYTE_ORDER_MARK="\uFEFF",s.BAD_DELIMITERS=["\r","\n",'"',s.BYTE_ORDER_MARK],s.WORKERS_SUPPORTED=!r&&!!i.Worker,s.NODE_STREAM_INPUT=1,s.LocalChunkSize=0xa00000,s.RemoteChunkSize=5242880,s.DefaultDelimiter=",",s.Parser=f,s.ParserHandle=p,s.NetworkStreamer=c,s.FileStreamer=d,s.StringStreamer=u,s.ReadableStreamStreamer=h,i.jQuery&&((t=i.jQuery).fn.parse=function(e){var r=e.config||{},a=[];return this.each(function(e){if(!("INPUT"===t(this).prop("tagName").toUpperCase()&&"file"===t(this).attr("type").toLowerCase()&&i.FileReader)||!this.files||0===this.files.length)return!0;for(var o=0;o<this.files.length;o++)a.push({file:this.files[o],inputElem:this,instanceConfig:t.extend({},r)})}),o(),this;function o(){if(0===a.length)x(e.complete)&&e.complete();else{var i,r,o,l=a[0];if(x(e.before)){var c=e.before(l.file,l.inputElem);if("object"==typeof c){if("abort"===c.action)return i=l.file,r=l.inputElem,o=c.reason,void(x(e.error)&&e.error({name:"AbortError"},i,r,o));if("skip"===c.action)return void n();"object"==typeof c.config&&(l.instanceConfig=t.extend(l.instanceConfig,c.config))}else if("skip"===c)return void n()}var d=l.instanceConfig.complete;l.instanceConfig.complete=function(e){x(d)&&d(e,l.file,l.inputElem),n()},s.parse(l.file,l.instanceConfig)}}function n(){a.splice(0,1),o()}}),a&&(i.onmessage=function(e){e=e.data,void 0===s.WORKER_ID&&e&&(s.WORKER_ID=e.workerId),"string"==typeof e.input?i.postMessage({workerId:s.WORKER_ID,results:s.parse(e.input,e.config),finished:!0}):(i.File&&e.input instanceof File||e.input instanceof Object)&&(e=s.parse(e.input,e.config))&&i.postMessage({workerId:s.WORKER_ID,results:e,finished:!0})}),(c.prototype=Object.create(l.prototype)).constructor=c,(d.prototype=Object.create(l.prototype)).constructor=d,(u.prototype=Object.create(u.prototype)).constructor=u,(h.prototype=Object.create(l.prototype)).constructor=h,s},"function"==typeof define&&define.amd?void 0!==(a=r())&&e.v(a):t.exports=r()},921511,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),a=e.i(764205);function o(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let i=e.version_number??1,r=e.version_status??"draft";return{label:`${e.policy_name} — v${i} (${r})${e.description?` — ${e.description}`:""}`,value:"production"===r?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["default",0,({onChange:e,value:n,className:s,accessToken:l,disabled:c,onPoliciesLoaded:d})=>{let[u,h]=(0,i.useState)([]),[p,m]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(l){m(!0);try{let e=await (0,a.getPoliciesList)(l);e.policies&&(h(e.policies),d?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{m(!1)}}})()},[l,d]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",disabled:c,placeholder:c?"Setting policies is a premium feature.":"Select policies (production or published versions)",onChange:t=>{e(t)},value:n,loading:p,className:s,allowClear:!0,options:o(u),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})},"getPolicyOptionEntries",()=>o])},891547,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(199133),a=e.i(764205);e.s(["default",0,({onChange:e,value:o,className:n,accessToken:s,disabled:l})=>{let[c,d]=(0,i.useState)([]),[u,h]=(0,i.useState)(!1);return(0,i.useEffect)(()=>{(async()=>{if(s){h(!0);try{let e=await (0,a.getGuardrailsList)(s);console.log("Guardrails response:",e),e.guardrails&&(console.log("Guardrails data:",e.guardrails),d(e.guardrails))}catch(e){console.error("Error fetching guardrails:",e)}finally{h(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(r.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onChange:t=>{console.log("Selected guardrails:",t),e(t)},value:o,loading:u,className:n,allowClear:!0,options:c.map(e=>(console.log("Mapping guardrail:",e),{label:`${e.guardrail_name}`,value:e.guardrail_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["ClockCircleOutlined",0,o],637235)},646563,e=>{"use strict";var t=e.i(959013);e.s(["PlusOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["ArrowLeftOutlined",0,o],447566)},367240,555436,e=>{"use strict";let t=(0,e.i(475254).default)("rotate-ccw",[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]]);e.s(["RotateCcw",()=>t],367240);var i=e.i(54943);e.s(["Search",()=>i.default],555436)},678784,678745,e=>{"use strict";let t=(0,e.i(475254).default)("check",[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]]);e.s(["default",()=>t],678745),e.s(["CheckIcon",()=>t],678784)},54943,e=>{"use strict";let t=(0,e.i(475254).default)("search",[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]]);e.s(["default",()=>t])},987432,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M893.3 293.3L730.7 130.7c-7.5-7.5-16.7-13-26.7-16V112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V338.5c0-17-6.7-33.2-18.7-45.2zM384 184h256v104H384V184zm456 656H184V184h136v136c0 17.7 14.3 32 32 32h320c17.7 0 32-14.3 32-32V205.8l136 136V840zM512 442c-79.5 0-144 64.5-144 144s64.5 144 144 144 144-64.5 144-144-64.5-144-144-144zm0 224c-44.2 0-80-35.8-80-80s35.8-80 80-80 80 35.8 80 80-35.8 80-80 80z"}}]},name:"save",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["SaveOutlined",0,o],987432)},431343,569074,e=>{"use strict";var t=e.i(475254);let i=(0,t.default)("play",[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]]);e.s(["Play",()=>i],431343);let r=(0,t.default)("upload",[["path",{d:"M12 3v12",key:"1x0j5s"}],["path",{d:"m17 8-5-5-5 5",key:"7q97r8"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}]]);e.s(["Upload",()=>r],569074)},531245,657150,e=>{"use strict";let t=(0,e.i(475254).default)("bot",[["path",{d:"M12 8V4H8",key:"hb8ula"}],["rect",{width:"16",height:"12",x:"4",y:"8",rx:"2",key:"enze0r"}],["path",{d:"M2 14h2",key:"vft8re"}],["path",{d:"M20 14h2",key:"4cs60a"}],["path",{d:"M15 13v2",key:"1xurst"}],["path",{d:"M9 13v2",key:"rq6x2g"}]]);e.s(["default",()=>t],657150),e.s(["Bot",()=>t],531245)},98919,e=>{"use strict";var t=e.i(918549);e.s(["Shield",()=>t.default])},727612,e=>{"use strict";let t=(0,e.i(475254).default)("trash-2",[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]]);e.s(["Trash2",()=>t],727612)},918549,e=>{"use strict";let t=(0,e.i(475254).default)("shield",[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}]]);e.s(["default",()=>t])},673709,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(678784);let a=(0,e.i(475254).default)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var o=e.i(650056);let n={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};e.s(["default",0,({code:e,language:s})=>{let[l,c]=(0,i.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-gray-200 overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),c(!0),setTimeout(()=>c(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 z-10","aria-label":"Copy code",children:l?(0,t.jsx)(r.CheckIcon,{size:16}):(0,t.jsx)(a,{size:16})}),(0,t.jsx)(o.Prism,{language:s,style:n,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",backgroundColor:"#fafafa"},showLineNumbers:!0,children:e})]})}],673709)},903446,e=>{"use strict";let t=(0,e.i(475254).default)("settings",[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["default",()=>t])},132104,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M868 545.5L536.1 163a31.96 31.96 0 00-48.3 0L156 545.5a7.97 7.97 0 006 13.2h81c4.6 0 9-2 12.1-5.5L474 300.9V864c0 4.4 3.6 8 8 8h60c4.4 0 8-3.6 8-8V300.9l218.9 252.3c3 3.5 7.4 5.5 12.1 5.5h81c6.8 0 10.5-8 6-13.2z"}}]},name:"arrow-up",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["ArrowUpOutlined",0,o],132104)},447593,989022,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645),r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6a25.95 25.95 0 0025.6 30.4h723c1.5 0 3-.1 4.4-.4a25.88 25.88 0 0021.2-30zM204 390h272V182h72v208h272v104H204V390zm468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z"}}]},name:"clear",theme:"outlined"},a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["ClearOutlined",0,o],447593);var n=e.i(843476),s=e.i(592968),l=e.i(637235);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 394c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H400V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v236H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h228v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h164c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V394h164zM628 630H400V394h228v236z"}}]},name:"number",theme:"outlined"};var d=i.forwardRef(function(e,r){return i.createElement(a.default,(0,t.default)({},e,{ref:r,icon:c}))});let u={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM653.3 424.6l52.2 52.2a8.01 8.01 0 01-4.7 13.6l-179.4 21c-5.1.6-9.5-3.7-8.9-8.9l21-179.4c.8-6.6 8.9-9.4 13.6-4.7l52.4 52.4 256.2-256.2c3.1-3.1 8.2-3.1 11.3 0l42.4 42.4c3.1 3.1 3.1 8.2 0 11.3L653.3 424.6z"}}]},name:"import",theme:"outlined"};var h=i.forwardRef(function(e,r){return i.createElement(a.default,(0,t.default)({},e,{ref:r,icon:u}))}),p=e.i(872934),m=e.i(812618),f=e.i(366308),g=e.i(458505);e.s(["default",0,({timeToFirstToken:e,totalLatency:t,usage:i,toolName:r})=>e||t||i?(0,n.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3",children:[void 0!==e&&(0,n.jsx)(s.Tooltip,{title:"Time to first token",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["TTFT: ",(e/1e3).toFixed(2),"s"]})]})}),void 0!==t&&(0,n.jsx)(s.Tooltip,{title:"Total latency",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(l.ClockCircleOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Total Latency: ",(t/1e3).toFixed(2),"s"]})]})}),i?.promptTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Prompt tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(h,{className:"mr-1"}),(0,n.jsxs)("span",{children:["In: ",i.promptTokens]})]})}),i?.completionTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Completion tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(p.ExportOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Out: ",i.completionTokens]})]})}),i?.reasoningTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Reasoning tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(m.BulbOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Reasoning: ",i.reasoningTokens]})]})}),i?.totalTokens!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Total tokens",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(d,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Total: ",i.totalTokens]})]})}),i?.cost!==void 0&&(0,n.jsx)(s.Tooltip,{title:"Cost",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(g.DollarOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["$",i.cost.toFixed(6)]})]})}),r&&(0,n.jsx)(s.Tooltip,{title:"Tool used",children:(0,n.jsxs)("div",{className:"flex items-center",children:[(0,n.jsx)(f.ToolOutlined,{className:"mr-1"}),(0,n.jsxs)("span",{children:["Tool: ",r]})]})})]}):null],989022)},190272,785913,e=>{"use strict";var t,i,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i);let o={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=o[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:r,apiKey:o,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:h,mcpServers:p,mcpServerToolRestrictions:m,selectedVoice:f,endpointType:g,selectedModel:_,selectedSdk:b,proxySettings:v}=e,k="session"===i?r:o,x=window.location.origin,y=v?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?x=y:v?.PROXY_BASE_URL&&(x=v.PROXY_BASE_URL);let w=n||"Your prompt here",C=w.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),c.length>0&&(E.vector_stores=c),d.length>0&&(E.guardrails=d),u.length>0&&(E.policies=u);let S=_||"your-model-name",O="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${k||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${k||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"
)`;switch(g){case a.CHAT:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=j.length>0?j:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${S}",
    messages=${JSON.stringify(r,null,4)}${i}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${S}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${C}"
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
`;break}case a.RESPONSES:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=j.length>0?j:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${S}",
    input=${JSON.stringify(r,null,4)}${i}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${S}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${C}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===b?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${S}",
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
prompt = "${C}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
`;break;case a.IMAGE_EDITS:t="azure"===b?`
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
prompt = "${C}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
prompt = "${C}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${S}",
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
`;break;case a.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${S}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${S}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${S}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${S}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${O}
${t}`}],190272)},84899,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645),r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M931.4 498.9L94.9 79.5c-3.4-1.7-7.3-2.1-11-1.2a15.99 15.99 0 00-11.7 19.3l86.2 352.2c1.3 5.3 5.2 9.6 10.4 11.3l147.7 50.7-147.6 50.7c-5.2 1.8-9.1 6-10.3 11.3L72.2 926.5c-.9 3.7-.5 7.6 1.2 10.9 3.9 7.9 13.5 11.1 21.5 7.2l836.5-417c3.1-1.5 5.6-4.1 7.2-7.1 3.9-8 .7-17.6-7.2-21.6zM170.8 826.3l50.3-205.6 295.2-101.3c2.3-.8 4.2-2.6 5-5 1.4-4.2-.8-8.7-5-10.2L221.1 403 171 198.2l628 314.9-628.2 313.2z"}}]},name:"send",theme:"outlined"},a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["SendOutlined",0,o],84899)},782273,793916,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M625.9 115c-5.9 0-11.9 1.6-17.4 5.3L254 352H90c-8.8 0-16 7.2-16 16v288c0 8.8 7.2 16 16 16h164l354.5 231.7c5.5 3.6 11.6 5.3 17.4 5.3 16.7 0 32.1-13.3 32.1-32.1V147.1c0-18.8-15.4-32.1-32.1-32.1zM586 803L293.4 611.7l-18-11.7H146V424h129.4l17.9-11.7L586 221v582zm348-327H806c-8.8 0-16 7.2-16 16v40c0 8.8 7.2 16 16 16h128c8.8 0 16-7.2 16-16v-40c0-8.8-7.2-16-16-16zm-41.9 261.8l-110.3-63.7a15.9 15.9 0 00-21.7 5.9l-19.9 34.5c-4.4 7.6-1.8 17.4 5.8 21.8L856.3 800a15.9 15.9 0 0021.7-5.9l19.9-34.5c4.4-7.6 1.7-17.4-5.8-21.8zM760 344a15.9 15.9 0 0021.7 5.9L892 286.2c7.6-4.4 10.2-14.2 5.8-21.8L878 230a15.9 15.9 0 00-21.7-5.9L746 287.8a15.99 15.99 0 00-5.8 21.8L760 344z"}}]},name:"sound",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["SoundOutlined",0,o],782273);let n={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M842 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 140.3-113.7 254-254 254S258 594.3 258 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 168.7 126.6 307.9 290 327.6V884H326.7c-13.7 0-24.7 14.3-24.7 32v36c0 4.4 2.8 8 6.2 8h407.6c3.4 0 6.2-3.6 6.2-8v-36c0-17.7-11-32-24.7-32H548V782.1c165.3-18 294-158 294-328.1zM512 624c93.9 0 170-75.2 170-168V232c0-92.8-76.1-168-170-168s-170 75.2-170 168v224c0 92.8 76.1 168 170 168zm-94-392c0-50.6 41.9-92 94-92s94 41.4 94 92v224c0 50.6-41.9 92-94 92s-94-41.4-94-92V232z"}}]},name:"audio",theme:"outlined"};var s=i.forwardRef(function(e,r){return i.createElement(a.default,(0,t.default)({},e,{ref:r,icon:n}))});e.s(["AudioOutlined",0,s],793916)},518617,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm0 76c-205.4 0-372 166.6-372 372s166.6 372 372 372 372-166.6 372-372-166.6-372-372-372zm128.01 198.83c.03 0 .05.01.09.06l45.02 45.01a.2.2 0 01.05.09.12.12 0 010 .07c0 .02-.01.04-.05.08L557.25 512l127.87 127.86a.27.27 0 01.05.06v.02a.12.12 0 010 .07c0 .03-.01.05-.05.09l-45.02 45.02a.2.2 0 01-.09.05.12.12 0 01-.07 0c-.02 0-.04-.01-.08-.05L512 557.25 384.14 685.12c-.04.04-.06.05-.08.05a.12.12 0 01-.07 0c-.03 0-.05-.01-.09-.05l-45.02-45.02a.2.2 0 01-.05-.09.12.12 0 010-.07c0-.02.01-.04.06-.08L466.75 512 338.88 384.14a.27.27 0 01-.05-.06l-.01-.02a.12.12 0 010-.07c0-.03.01-.05.05-.09l45.02-45.02a.2.2 0 01.09-.05.12.12 0 01.07 0c.02 0 .04.01.08.06L512 466.75l127.86-127.86c.04-.05.06-.06.08-.06a.12.12 0 01.07 0z"}}]},name:"close-circle",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["CloseCircleOutlined",0,o],518617)},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["CheckCircleOutlined",0,o],245704)},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["CodeOutlined",0,o],245094)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["LinkOutlined",0,o],596239)},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},149192,e=>{"use strict";var t=e.i(864517);e.s(["CloseOutlined",()=>t.default])},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var a=e.i(9583),o=i.forwardRef(function(e,o){return i.createElement(a.default,(0,t.default)({},e,{ref:o,icon:r}))});e.s(["DollarOutlined",0,o],458505)},611052,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(212931),a=e.i(311451),o=e.i(790848),n=e.i(888259),s=e.i(438957);e.i(247167);var l=e.i(931067);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240zm460 600H232V536h560v304zM484 701v53c0 4.4 3.6 8 8 8h40c4.4 0 8-3.6 8-8v-53a48.01 48.01 0 10-56 0z"}}]},name:"lock",theme:"outlined"};var d=e.i(9583),u=i.forwardRef(function(e,t){return i.createElement(d.default,(0,l.default)({},e,{ref:t,icon:c}))}),h=e.i(492030),p=e.i(266537),m=e.i(447566),f=e.i(149192),g=e.i(596239);e.s(["ByokCredentialModal",0,({server:e,open:l,onClose:c,onSuccess:d,accessToken:_})=>{let[b,v]=(0,i.useState)(1),[k,x]=(0,i.useState)(""),[y,w]=(0,i.useState)(!0),[C,j]=(0,i.useState)(!1),E=e.alias||e.server_name||"Service",S=E.charAt(0).toUpperCase(),O=()=>{v(1),x(""),w(!0),j(!1),c()},R=async()=>{if(!k.trim())return void n.default.error("Please enter your API key");j(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${_}`},body:JSON.stringify({credential:k.trim(),save:y})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}n.default.success(`Connected to ${E}`),d(e.server_id),O()}catch(e){n.default.error(e.message||"Failed to connect")}finally{j(!1)}};return(0,t.jsx)(r.Modal,{open:l,onCancel:O,footer:null,width:480,closeIcon:null,className:"byok-modal",children:(0,t.jsxs)("div",{className:"relative p-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===b?(0,t.jsxs)("button",{onClick:()=>v(1),className:"flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm",children:[(0,t.jsx)(m.ArrowLeftOutlined,{})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${1===b?"bg-blue-500":"bg-gray-300"}`}),(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${2===b?"bg-blue-500":"bg-gray-300"}`})]}),(0,t.jsx)("button",{onClick:O,className:"text-gray-400 hover:text-gray-600",children:(0,t.jsx)(f.CloseOutlined,{})})]}),1===b?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(p.ArrowRightOutlined,{className:"text-gray-400 text-lg"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:S})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:["Connect ",E]}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["LiteLLM needs access to ",E," to complete your request."]}),(0,t.jsx)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-gray-800 mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-gray-500 text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to"," ",E,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-green-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,i)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-gray-700",children:[(0,t.jsx)(h.CheckOutlined,{className:"text-green-500 flex-shrink-0"}),e]},i))})]}),(0,t.jsxs)("button",{onClick:()=>v(2),className:"w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:["Continue to Authentication ",(0,t.jsx)(p.ArrowRightOutlined,{})]}),(0,t.jsx)("button",{onClick:O,className:"mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-4",children:(0,t.jsx)(s.KeyOutlined,{className:"text-blue-400 text-xl"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["Enter your ",E," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsxs)("label",{className:"block text-sm font-semibold text-gray-800 mb-2",children:[E," API Key"]}),(0,t.jsx)(a.Input.Password,{placeholder:"Enter your API key",value:k,onChange:e=>x(e.target.value),size:"large",className:"rounded-lg"}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-500 hover:text-blue-700 text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key? ",(0,t.jsx)(g.LinkOutlined,{})]})]}),(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-gray-800",children:"Save key for future use"})]}),(0,t.jsx)(o.Switch,{checked:y,onChange:w})]}),(0,t.jsxs)("div",{className:"bg-blue-50 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(u,{className:"text-blue-400 mt-0.5 flex-shrink-0"}),(0,t.jsx)("p",{className:"text-sm text-blue-700",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)("button",{onClick:R,disabled:C,className:"w-full bg-blue-500 hover:bg-blue-600 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:[(0,t.jsx)(u,{})," Connect & Authorize"]})]})]})})}],611052)}]);