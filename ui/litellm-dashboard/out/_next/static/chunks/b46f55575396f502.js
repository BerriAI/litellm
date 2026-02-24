(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,233525,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"warnOnce",{enumerable:!0,get:function(){return a}});let a=e=>{}},349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},992571,e=>{"use strict";var t=e.i(619273);function i(e){return{onFetch:(i,s)=>{let n=i.options,o=i.fetchOptions?.meta?.fetchMore?.direction,l=i.state.data?.pages||[],u=i.state.data?.pageParams||[],c={pages:[],pageParams:[]},d=0,p=async()=>{let s=!1,p=(0,t.ensureQueryFn)(i.options,i.fetchOptions),h=async(e,a,r)=>{let n;if(s)return Promise.reject();if(null==a&&e.pages.length)return Promise.resolve(e);let o=(n={client:i.client,queryKey:i.queryKey,pageParam:a,direction:r?"backward":"forward",meta:i.options.meta},(0,t.addConsumeAwareSignal)(n,()=>i.signal,()=>s=!0),n),l=await p(o),{maxPages:u}=i.options,c=r?t.addToStart:t.addToEnd;return{pages:c(e.pages,l,u),pageParams:c(e.pageParams,a,u)}};if(o&&l.length){let e="backward"===o,t={pages:l,pageParams:u},i=(e?r:a)(n,t);c=await h(t,i,e)}else{let t=e??l.length;do{let e=0===d?u[0]??n.initialPageParam:a(n,c);if(d>0&&null==e)break;c=await h(c,e),d++}while(d<t)}return c};i.options.persister?i.fetchFn=()=>i.options.persister?.(p,{client:i.client,queryKey:i.queryKey,meta:i.options.meta,signal:i.signal},s):i.fetchFn=p}}}function a(e,{pages:t,pageParams:i}){let a=t.length-1;return t.length>0?e.getNextPageParam(t[a],t,i[a],i):void 0}function r(e,{pages:t,pageParams:i}){return t.length>0?e.getPreviousPageParam?.(t[0],t,i[0],i):void 0}function s(e,t){return!!t&&null!=a(e,t)}function n(e,t){return!!t&&!!e.getPreviousPageParam&&null!=r(e,t)}e.s(["hasNextPage",()=>s,"hasPreviousPage",()=>n,"infiniteQueryBehavior",()=>i])},114272,e=>{"use strict";var t=e.i(540143),i=e.i(88587),a=e.i(936553),r=class extends i.Removable{#e;#t;#i;#a;constructor(e){super(),this.#e=e.client,this.mutationId=e.mutationId,this.#i=e.mutationCache,this.#t=[],this.state=e.state||s(),this.setOptions(e.options),this.scheduleGc()}setOptions(e){this.options=e,this.updateGcTime(this.options.gcTime)}get meta(){return this.options.meta}addObserver(e){this.#t.includes(e)||(this.#t.push(e),this.clearGcTimeout(),this.#i.notify({type:"observerAdded",mutation:this,observer:e}))}removeObserver(e){this.#t=this.#t.filter(t=>t!==e),this.scheduleGc(),this.#i.notify({type:"observerRemoved",mutation:this,observer:e})}optionalRemove(){this.#t.length||("pending"===this.state.status?this.scheduleGc():this.#i.remove(this))}continue(){return this.#a?.continue()??this.execute(this.state.variables)}async execute(e){let t=()=>{this.#r({type:"continue"})},i={client:this.#e,meta:this.options.meta,mutationKey:this.options.mutationKey};this.#a=(0,a.createRetryer)({fn:()=>this.options.mutationFn?this.options.mutationFn(e,i):Promise.reject(Error("No mutationFn found")),onFail:(e,t)=>{this.#r({type:"failed",failureCount:e,error:t})},onPause:()=>{this.#r({type:"pause"})},onContinue:t,retry:this.options.retry??0,retryDelay:this.options.retryDelay,networkMode:this.options.networkMode,canRun:()=>this.#i.canRun(this)});let r="pending"===this.state.status,s=!this.#a.canStart();try{if(r)t();else{this.#r({type:"pending",variables:e,isPaused:s}),this.#i.config.onMutate&&await this.#i.config.onMutate(e,this,i);let t=await this.options.onMutate?.(e,i);t!==this.state.context&&this.#r({type:"pending",context:t,variables:e,isPaused:s})}let a=await this.#a.start();return await this.#i.config.onSuccess?.(a,e,this.state.context,this,i),await this.options.onSuccess?.(a,e,this.state.context,i),await this.#i.config.onSettled?.(a,null,this.state.variables,this.state.context,this,i),await this.options.onSettled?.(a,null,e,this.state.context,i),this.#r({type:"success",data:a}),a}catch(t){try{await this.#i.config.onError?.(t,e,this.state.context,this,i)}catch(e){Promise.reject(e)}try{await this.options.onError?.(t,e,this.state.context,i)}catch(e){Promise.reject(e)}try{await this.#i.config.onSettled?.(void 0,t,this.state.variables,this.state.context,this,i)}catch(e){Promise.reject(e)}try{await this.options.onSettled?.(void 0,t,e,this.state.context,i)}catch(e){Promise.reject(e)}throw this.#r({type:"error",error:t}),t}finally{this.#i.runNext(this)}}#r(e){this.state=(t=>{switch(e.type){case"failed":return{...t,failureCount:e.failureCount,failureReason:e.error};case"pause":return{...t,isPaused:!0};case"continue":return{...t,isPaused:!1};case"pending":return{...t,context:e.context,data:void 0,failureCount:0,failureReason:null,error:null,isPaused:e.isPaused,status:"pending",variables:e.variables,submittedAt:Date.now()};case"success":return{...t,data:e.data,failureCount:0,failureReason:null,error:null,status:"success",isPaused:!1};case"error":return{...t,data:void 0,error:e.error,failureCount:t.failureCount+1,failureReason:e.error,isPaused:!1,status:"error"}}})(this.state),t.notifyManager.batch(()=>{this.#t.forEach(t=>{t.onMutationUpdate(e)}),this.#i.notify({mutation:this,type:"updated",action:e})})}};function s(){return{context:void 0,data:void 0,error:null,failureCount:0,failureReason:null,isPaused:!1,status:"idle",variables:void 0,submittedAt:0}}e.s(["Mutation",()=>r,"getDefaultState",()=>s])},317751,e=>{"use strict";var t=e.i(619273),i=e.i(286491),a=e.i(540143),r=e.i(915823),s=class extends r.Subscribable{constructor(e={}){super(),this.config=e,this.#s=new Map}#s;build(e,a,r){let s=a.queryKey,n=a.queryHash??(0,t.hashQueryKeyByOptions)(s,a),o=this.get(n);return o||(o=new i.Query({client:e,queryKey:s,queryHash:n,options:e.defaultQueryOptions(a),state:r,defaultOptions:e.getQueryDefaults(s)}),this.add(o)),o}add(e){this.#s.has(e.queryHash)||(this.#s.set(e.queryHash,e),this.notify({type:"added",query:e}))}remove(e){let t=this.#s.get(e.queryHash);t&&(e.destroy(),t===e&&this.#s.delete(e.queryHash),this.notify({type:"removed",query:e}))}clear(){a.notifyManager.batch(()=>{this.getAll().forEach(e=>{this.remove(e)})})}get(e){return this.#s.get(e)}getAll(){return[...this.#s.values()]}find(e){let i={exact:!0,...e};return this.getAll().find(e=>(0,t.matchQuery)(i,e))}findAll(e={}){let i=this.getAll();return Object.keys(e).length>0?i.filter(i=>(0,t.matchQuery)(e,i)):i}notify(e){a.notifyManager.batch(()=>{this.listeners.forEach(t=>{t(e)})})}onFocus(){a.notifyManager.batch(()=>{this.getAll().forEach(e=>{e.onFocus()})})}onOnline(){a.notifyManager.batch(()=>{this.getAll().forEach(e=>{e.onOnline()})})}},n=e.i(114272),o=r,l=class extends o.Subscribable{constructor(e={}){super(),this.config=e,this.#n=new Set,this.#o=new Map,this.#l=0}#n;#o;#l;build(e,t,i){let a=new n.Mutation({client:e,mutationCache:this,mutationId:++this.#l,options:e.defaultMutationOptions(t),state:i});return this.add(a),a}add(e){this.#n.add(e);let t=u(e);if("string"==typeof t){let i=this.#o.get(t);i?i.push(e):this.#o.set(t,[e])}this.notify({type:"added",mutation:e})}remove(e){if(this.#n.delete(e)){let t=u(e);if("string"==typeof t){let i=this.#o.get(t);if(i)if(i.length>1){let t=i.indexOf(e);-1!==t&&i.splice(t,1)}else i[0]===e&&this.#o.delete(t)}}this.notify({type:"removed",mutation:e})}canRun(e){let t=u(e);if("string"!=typeof t)return!0;{let i=this.#o.get(t),a=i?.find(e=>"pending"===e.state.status);return!a||a===e}}runNext(e){let t=u(e);if("string"!=typeof t)return Promise.resolve();{let i=this.#o.get(t)?.find(t=>t!==e&&t.state.isPaused);return i?.continue()??Promise.resolve()}}clear(){a.notifyManager.batch(()=>{this.#n.forEach(e=>{this.notify({type:"removed",mutation:e})}),this.#n.clear(),this.#o.clear()})}getAll(){return Array.from(this.#n)}find(e){let i={exact:!0,...e};return this.getAll().find(e=>(0,t.matchMutation)(i,e))}findAll(e={}){return this.getAll().filter(i=>(0,t.matchMutation)(e,i))}notify(e){a.notifyManager.batch(()=>{this.listeners.forEach(t=>{t(e)})})}resumePausedMutations(){let e=this.getAll().filter(e=>e.state.isPaused);return a.notifyManager.batch(()=>Promise.all(e.map(e=>e.continue().catch(t.noop))))}};function u(e){return e.options.scope?.id}var c=e.i(175555),d=e.i(814448),p=e.i(992571),h=class{#u;#i;#c;#d;#p;#h;#m;#g;constructor(e={}){this.#u=e.queryCache||new s,this.#i=e.mutationCache||new l,this.#c=e.defaultOptions||{},this.#d=new Map,this.#p=new Map,this.#h=0}mount(){this.#h++,1===this.#h&&(this.#m=c.focusManager.subscribe(async e=>{e&&(await this.resumePausedMutations(),this.#u.onFocus())}),this.#g=d.onlineManager.subscribe(async e=>{e&&(await this.resumePausedMutations(),this.#u.onOnline())}))}unmount(){this.#h--,0===this.#h&&(this.#m?.(),this.#m=void 0,this.#g?.(),this.#g=void 0)}isFetching(e){return this.#u.findAll({...e,fetchStatus:"fetching"}).length}isMutating(e){return this.#i.findAll({...e,status:"pending"}).length}getQueryData(e){let t=this.defaultQueryOptions({queryKey:e});return this.#u.get(t.queryHash)?.state.data}ensureQueryData(e){let i=this.defaultQueryOptions(e),a=this.#u.build(this,i),r=a.state.data;return void 0===r?this.fetchQuery(e):(e.revalidateIfStale&&a.isStaleByTime((0,t.resolveStaleTime)(i.staleTime,a))&&this.prefetchQuery(i),Promise.resolve(r))}getQueriesData(e){return this.#u.findAll(e).map(({queryKey:e,state:t})=>[e,t.data])}setQueryData(e,i,a){let r=this.defaultQueryOptions({queryKey:e}),s=this.#u.get(r.queryHash),n=s?.state.data,o=(0,t.functionalUpdate)(i,n);if(void 0!==o)return this.#u.build(this,r).setData(o,{...a,manual:!0})}setQueriesData(e,t,i){return a.notifyManager.batch(()=>this.#u.findAll(e).map(({queryKey:e})=>[e,this.setQueryData(e,t,i)]))}getQueryState(e){let t=this.defaultQueryOptions({queryKey:e});return this.#u.get(t.queryHash)?.state}removeQueries(e){let t=this.#u;a.notifyManager.batch(()=>{t.findAll(e).forEach(e=>{t.remove(e)})})}resetQueries(e,t){let i=this.#u;return a.notifyManager.batch(()=>(i.findAll(e).forEach(e=>{e.reset()}),this.refetchQueries({type:"active",...e},t)))}cancelQueries(e,i={}){let r={revert:!0,...i};return Promise.all(a.notifyManager.batch(()=>this.#u.findAll(e).map(e=>e.cancel(r)))).then(t.noop).catch(t.noop)}invalidateQueries(e,t={}){return a.notifyManager.batch(()=>(this.#u.findAll(e).forEach(e=>{e.invalidate()}),e?.refetchType==="none")?Promise.resolve():this.refetchQueries({...e,type:e?.refetchType??e?.type??"active"},t))}refetchQueries(e,i={}){let r={...i,cancelRefetch:i.cancelRefetch??!0};return Promise.all(a.notifyManager.batch(()=>this.#u.findAll(e).filter(e=>!e.isDisabled()&&!e.isStatic()).map(e=>{let i=e.fetch(void 0,r);return r.throwOnError||(i=i.catch(t.noop)),"paused"===e.state.fetchStatus?Promise.resolve():i}))).then(t.noop)}fetchQuery(e){let i=this.defaultQueryOptions(e);void 0===i.retry&&(i.retry=!1);let a=this.#u.build(this,i);return a.isStaleByTime((0,t.resolveStaleTime)(i.staleTime,a))?a.fetch(i):Promise.resolve(a.state.data)}prefetchQuery(e){return this.fetchQuery(e).then(t.noop).catch(t.noop)}fetchInfiniteQuery(e){return e.behavior=(0,p.infiniteQueryBehavior)(e.pages),this.fetchQuery(e)}prefetchInfiniteQuery(e){return this.fetchInfiniteQuery(e).then(t.noop).catch(t.noop)}ensureInfiniteQueryData(e){return e.behavior=(0,p.infiniteQueryBehavior)(e.pages),this.ensureQueryData(e)}resumePausedMutations(){return d.onlineManager.isOnline()?this.#i.resumePausedMutations():Promise.resolve()}getQueryCache(){return this.#u}getMutationCache(){return this.#i}getDefaultOptions(){return this.#c}setDefaultOptions(e){this.#c=e}setQueryDefaults(e,i){this.#d.set((0,t.hashKey)(e),{queryKey:e,defaultOptions:i})}getQueryDefaults(e){let i=[...this.#d.values()],a={};return i.forEach(i=>{(0,t.partialMatchKey)(e,i.queryKey)&&Object.assign(a,i.defaultOptions)}),a}setMutationDefaults(e,i){this.#p.set((0,t.hashKey)(e),{mutationKey:e,defaultOptions:i})}getMutationDefaults(e){let i=[...this.#p.values()],a={};return i.forEach(i=>{(0,t.partialMatchKey)(e,i.mutationKey)&&Object.assign(a,i.defaultOptions)}),a}defaultQueryOptions(e){if(e._defaulted)return e;let i={...this.#c.queries,...this.getQueryDefaults(e.queryKey),...e,_defaulted:!0};return i.queryHash||(i.queryHash=(0,t.hashQueryKeyByOptions)(i.queryKey,i)),void 0===i.refetchOnReconnect&&(i.refetchOnReconnect="always"!==i.networkMode),void 0===i.throwOnError&&(i.throwOnError=!!i.suspense),!i.networkMode&&i.persister&&(i.networkMode="offlineFirst"),i.queryFn===t.skipToken&&(i.enabled=!1),i}defaultMutationOptions(e){return e?._defaulted?e:{...this.#c.mutations,...e?.mutationKey&&this.getMutationDefaults(e.mutationKey),...e,_defaulted:!0}}clear(){this.#u.clear(),this.#i.clear()}};e.s(["QueryClient",()=>h],317751)},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},209261,e=>{"use strict";e.s(["extractCategories",0,e=>{let t=new Set;return e.forEach(e=>{e.category&&""!==e.category.trim()&&t.add(e.category)}),["All",...Array.from(t).sort(),"Other"]},"filterPluginsByCategory",0,(e,t)=>"All"===t?e:"Other"===t?e.filter(e=>!e.category||""===e.category.trim()):e.filter(e=>e.category===t),"filterPluginsBySearch",0,(e,t)=>{if(!t||""===t.trim())return e;let i=t.toLowerCase().trim();return e.filter(e=>{let t=e.name.toLowerCase().includes(i),a=e.description?.toLowerCase().includes(i)||!1,r=e.keywords?.some(e=>e.toLowerCase().includes(i))||!1;return t||a||r})},"formatDateString",0,e=>{if(!e)return"N/A";try{return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"})}catch(e){return"Invalid date"}},"formatInstallCommand",0,e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"url"===e.source&&e.url?e.url:"Unknown source","getSourceLink",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:"url"===e.source&&e.url?e.url:null,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return r}});let a=e.r(271645);function r(e,t){let i=(0,a.useRef)(null),r=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=i.current;e&&(i.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(i.current=s(e,a)),t&&(r.current=s(t,a))},[e,t])}function s(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},62478,e=>{"use strict";var t=e.i(764205);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),s=i.forwardRef(function(e,s){return i.createElement(r.default,(0,t.default)({},e,{ref:s,icon:a}))});e.s(["SafetyOutlined",0,s],602073)},190272,785913,e=>{"use strict";var t,i,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i);let s={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=s[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:a,apiKey:s,inputMessage:n,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:c,selectedPolicies:d,selectedMCPServers:p,mcpServers:h,mcpServerToolRestrictions:m,selectedVoice:g,endpointType:f,selectedModel:y,selectedSdk:_,proxySettings:b}=e,v="session"===i?a:s,w=window.location.origin,x=b?.LITELLM_UI_API_DOC_BASE_URL;x&&x.trim()?w=x:b?.PROXY_BASE_URL&&(w=b.PROXY_BASE_URL);let C=n||"Your prompt here",S=C.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),O=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),u.length>0&&(E.vector_stores=u),c.length>0&&(E.guardrails=c),d.length>0&&(E.policies=d);let P=y||"your-model-name",I="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${w}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${w}"
)`;switch(f){case r.CHAT:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=O.length>0?O:[{role:"user",content:C}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${P}",
    messages=${JSON.stringify(a,null,4)}${i}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${P}",
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
#     ]${i}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=O.length>0?O:[{role:"user",content:C}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${P}",
    input=${JSON.stringify(a,null,4)}${i}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${P}",
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
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===_?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${P}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${P}",
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
`;break;case r.IMAGE_EDITS:t="azure"===_?`
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
	model="${P}",
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
	model="${P}",
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
	model="${P}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${P}",
	file=audio_file${n?`,
	prompt="${n.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${P}",
	input="${n||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${P}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},798496,e=>{"use strict";var t=e.i(843476),i=e.i(152990),a=e.i(682830),r=e.i(271645),s=e.i(269200),n=e.i(427612),o=e.i(64848),l=e.i(942232),u=e.i(496020),c=e.i(977572),d=e.i(94629),p=e.i(360820),h=e.i(871943);function m({data:e=[],columns:m,isLoading:g=!1,defaultSorting:f=[],pagination:y,onPaginationChange:_,enablePagination:b=!1}){let[v,w]=r.default.useState(f),[x]=r.default.useState("onChange"),[C,S]=r.default.useState({}),[O,E]=r.default.useState({}),P=(0,i.useReactTable)({data:e,columns:m,state:{sorting:v,columnSizing:C,columnVisibility:O,...b&&y?{pagination:y}:{}},columnResizeMode:x,onSortingChange:w,onColumnSizingChange:S,onColumnVisibilityChange:E,...b&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),...b?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(s.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:P.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:P.getHeaderGroups().map(e=>(0,t.jsx)(u.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(o.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,i.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(p.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(h.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(d.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(l.TableBody,{children:g?(0,t.jsx)(u.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):P.getRowModel().rows.length>0?P.getRowModel().rows.map(e=>(0,t.jsx)(u.TableRow,{children:e.getVisibleCells().map(e=>(0,t.jsx)(c.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,i.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(u.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:m.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>m])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var r=e.i(9583),s=i.forwardRef(function(e,s){return i.createElement(r.default,(0,t.default)({},e,{ref:s,icon:a}))});e.s(["UserOutlined",0,s],771674)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var r=e.i(9583),s=i.forwardRef(function(e,s){return i.createElement(r.default,(0,t.default)({},e,{ref:s,icon:a}))});e.s(["CrownOutlined",0,s],100486)},115571,371401,e=>{"use strict";let t="local-storage-change";function i(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function a(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function r(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function s(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>i,"getLocalStorageItem",()=>a,"removeLocalStorageItem",()=>s,"setLocalStorageItem",()=>r],115571);var n=e.i(271645);function o(e){let i=t=>{"disableUsageIndicator"===t.key&&e()},a=t=>{let{key:i}=t.detail;"disableUsageIndicator"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t,a),()=>{window.removeEventListener("storage",i),window.removeEventListener(t,a)}}function l(){return"true"===a("disableUsageIndicator")}function u(){return(0,n.useSyncExternalStore)(o,l)}e.s(["useDisableUsageIndicator",()=>u],371401)},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),a=e.i(764205);let r=(0,i.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:s})=>{let[n,o]=(0,i.useState)(null),[l,u]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,a.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let e=await i.json();e.values?.logo_url&&o(e.values.logo_url),e.values?.favicon_url&&u(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,i.useEffect)(()=>{if(l){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=l});else{let e=document.createElement("link");e.rel="icon",e.href=l,document.head.appendChild(e)}}},[l]),(0,t.jsx)(r.Provider,{value:{logoUrl:n,setLogoUrl:o,faviconUrl:l,setFaviconUrl:u},children:e})},"useTheme",0,()=>{let e=(0,i.useContext)(r);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},86408,e=>{"use strict";var t=e.i(843476),i=e.i(271645),a=e.i(618566),r=e.i(934879),s=e.i(317751),n=e.i(912598);let o=new s.QueryClient;function l(){let e=(0,a.useSearchParams)().get("key"),[s,l]=(0,i.useState)(null);return console.log("PublicModelHubTable accessToken:",s),(0,i.useEffect)(()=>{e&&l(e)},[e]),(0,t.jsx)(n.QueryClientProvider,{client:o,children:(0,t.jsx)(r.default,{accessToken:s,publicPage:!0,premiumUser:!1,userRole:null})})}function u(){return(0,t.jsx)(i.Suspense,{fallback:(0,t.jsx)("div",{className:"flex items-center justify-center min-h-screen",children:"Loading..."}),children:(0,t.jsx)(l,{})})}e.s(["default",()=>u])}]);