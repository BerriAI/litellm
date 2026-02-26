(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,233525,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"warnOnce",{enumerable:!0,get:function(){return r}});let r=e=>{}},349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},992571,e=>{"use strict";var t=e.i(619273);function i(e){return{onFetch:(i,n)=>{let o=i.options,s=i.fetchOptions?.meta?.fetchMore?.direction,l=i.state.data?.pages||[],u=i.state.data?.pageParams||[],c={pages:[],pageParams:[]},d=0,p=async()=>{let n=!1,p=(0,t.ensureQueryFn)(i.options,i.fetchOptions),h=async(e,r,a)=>{let o;if(n)return Promise.reject();if(null==r&&e.pages.length)return Promise.resolve(e);let s=(o={client:i.client,queryKey:i.queryKey,pageParam:r,direction:a?"backward":"forward",meta:i.options.meta},(0,t.addConsumeAwareSignal)(o,()=>i.signal,()=>n=!0),o),l=await p(s),{maxPages:u}=i.options,c=a?t.addToStart:t.addToEnd;return{pages:c(e.pages,l,u),pageParams:c(e.pageParams,r,u)}};if(s&&l.length){let e="backward"===s,t={pages:l,pageParams:u},i=(e?a:r)(o,t);c=await h(t,i,e)}else{let t=e??l.length;do{let e=0===d?u[0]??o.initialPageParam:r(o,c);if(d>0&&null==e)break;c=await h(c,e),d++}while(d<t)}return c};i.options.persister?i.fetchFn=()=>i.options.persister?.(p,{client:i.client,queryKey:i.queryKey,meta:i.options.meta,signal:i.signal},n):i.fetchFn=p}}}function r(e,{pages:t,pageParams:i}){let r=t.length-1;return t.length>0?e.getNextPageParam(t[r],t,i[r],i):void 0}function a(e,{pages:t,pageParams:i}){return t.length>0?e.getPreviousPageParam?.(t[0],t,i[0],i):void 0}function n(e,t){return!!t&&null!=r(e,t)}function o(e,t){return!!t&&!!e.getPreviousPageParam&&null!=a(e,t)}e.s(["hasNextPage",()=>n,"hasPreviousPage",()=>o,"infiniteQueryBehavior",()=>i])},114272,e=>{"use strict";var t=e.i(540143),i=e.i(88587),r=e.i(936553),a=class extends i.Removable{#e;#t;#i;#r;constructor(e){super(),this.#e=e.client,this.mutationId=e.mutationId,this.#i=e.mutationCache,this.#t=[],this.state=e.state||n(),this.setOptions(e.options),this.scheduleGc()}setOptions(e){this.options=e,this.updateGcTime(this.options.gcTime)}get meta(){return this.options.meta}addObserver(e){this.#t.includes(e)||(this.#t.push(e),this.clearGcTimeout(),this.#i.notify({type:"observerAdded",mutation:this,observer:e}))}removeObserver(e){this.#t=this.#t.filter(t=>t!==e),this.scheduleGc(),this.#i.notify({type:"observerRemoved",mutation:this,observer:e})}optionalRemove(){this.#t.length||("pending"===this.state.status?this.scheduleGc():this.#i.remove(this))}continue(){return this.#r?.continue()??this.execute(this.state.variables)}async execute(e){let t=()=>{this.#a({type:"continue"})},i={client:this.#e,meta:this.options.meta,mutationKey:this.options.mutationKey};this.#r=(0,r.createRetryer)({fn:()=>this.options.mutationFn?this.options.mutationFn(e,i):Promise.reject(Error("No mutationFn found")),onFail:(e,t)=>{this.#a({type:"failed",failureCount:e,error:t})},onPause:()=>{this.#a({type:"pause"})},onContinue:t,retry:this.options.retry??0,retryDelay:this.options.retryDelay,networkMode:this.options.networkMode,canRun:()=>this.#i.canRun(this)});let a="pending"===this.state.status,n=!this.#r.canStart();try{if(a)t();else{this.#a({type:"pending",variables:e,isPaused:n}),this.#i.config.onMutate&&await this.#i.config.onMutate(e,this,i);let t=await this.options.onMutate?.(e,i);t!==this.state.context&&this.#a({type:"pending",context:t,variables:e,isPaused:n})}let r=await this.#r.start();return await this.#i.config.onSuccess?.(r,e,this.state.context,this,i),await this.options.onSuccess?.(r,e,this.state.context,i),await this.#i.config.onSettled?.(r,null,this.state.variables,this.state.context,this,i),await this.options.onSettled?.(r,null,e,this.state.context,i),this.#a({type:"success",data:r}),r}catch(t){try{await this.#i.config.onError?.(t,e,this.state.context,this,i)}catch(e){Promise.reject(e)}try{await this.options.onError?.(t,e,this.state.context,i)}catch(e){Promise.reject(e)}try{await this.#i.config.onSettled?.(void 0,t,this.state.variables,this.state.context,this,i)}catch(e){Promise.reject(e)}try{await this.options.onSettled?.(void 0,t,e,this.state.context,i)}catch(e){Promise.reject(e)}throw this.#a({type:"error",error:t}),t}finally{this.#i.runNext(this)}}#a(e){this.state=(t=>{switch(e.type){case"failed":return{...t,failureCount:e.failureCount,failureReason:e.error};case"pause":return{...t,isPaused:!0};case"continue":return{...t,isPaused:!1};case"pending":return{...t,context:e.context,data:void 0,failureCount:0,failureReason:null,error:null,isPaused:e.isPaused,status:"pending",variables:e.variables,submittedAt:Date.now()};case"success":return{...t,data:e.data,failureCount:0,failureReason:null,error:null,status:"success",isPaused:!1};case"error":return{...t,data:void 0,error:e.error,failureCount:t.failureCount+1,failureReason:e.error,isPaused:!1,status:"error"}}})(this.state),t.notifyManager.batch(()=>{this.#t.forEach(t=>{t.onMutationUpdate(e)}),this.#i.notify({mutation:this,type:"updated",action:e})})}};function n(){return{context:void 0,data:void 0,error:null,failureCount:0,failureReason:null,isPaused:!1,status:"idle",variables:void 0,submittedAt:0}}e.s(["Mutation",()=>a,"getDefaultState",()=>n])},317751,e=>{"use strict";var t=e.i(619273),i=e.i(286491),r=e.i(540143),a=e.i(915823),n=class extends a.Subscribable{constructor(e={}){super(),this.config=e,this.#n=new Map}#n;build(e,r,a){let n=r.queryKey,o=r.queryHash??(0,t.hashQueryKeyByOptions)(n,r),s=this.get(o);return s||(s=new i.Query({client:e,queryKey:n,queryHash:o,options:e.defaultQueryOptions(r),state:a,defaultOptions:e.getQueryDefaults(n)}),this.add(s)),s}add(e){this.#n.has(e.queryHash)||(this.#n.set(e.queryHash,e),this.notify({type:"added",query:e}))}remove(e){let t=this.#n.get(e.queryHash);t&&(e.destroy(),t===e&&this.#n.delete(e.queryHash),this.notify({type:"removed",query:e}))}clear(){r.notifyManager.batch(()=>{this.getAll().forEach(e=>{this.remove(e)})})}get(e){return this.#n.get(e)}getAll(){return[...this.#n.values()]}find(e){let i={exact:!0,...e};return this.getAll().find(e=>(0,t.matchQuery)(i,e))}findAll(e={}){let i=this.getAll();return Object.keys(e).length>0?i.filter(i=>(0,t.matchQuery)(e,i)):i}notify(e){r.notifyManager.batch(()=>{this.listeners.forEach(t=>{t(e)})})}onFocus(){r.notifyManager.batch(()=>{this.getAll().forEach(e=>{e.onFocus()})})}onOnline(){r.notifyManager.batch(()=>{this.getAll().forEach(e=>{e.onOnline()})})}},o=e.i(114272),s=a,l=class extends s.Subscribable{constructor(e={}){super(),this.config=e,this.#o=new Set,this.#s=new Map,this.#l=0}#o;#s;#l;build(e,t,i){let r=new o.Mutation({client:e,mutationCache:this,mutationId:++this.#l,options:e.defaultMutationOptions(t),state:i});return this.add(r),r}add(e){this.#o.add(e);let t=u(e);if("string"==typeof t){let i=this.#s.get(t);i?i.push(e):this.#s.set(t,[e])}this.notify({type:"added",mutation:e})}remove(e){if(this.#o.delete(e)){let t=u(e);if("string"==typeof t){let i=this.#s.get(t);if(i)if(i.length>1){let t=i.indexOf(e);-1!==t&&i.splice(t,1)}else i[0]===e&&this.#s.delete(t)}}this.notify({type:"removed",mutation:e})}canRun(e){let t=u(e);if("string"!=typeof t)return!0;{let i=this.#s.get(t),r=i?.find(e=>"pending"===e.state.status);return!r||r===e}}runNext(e){let t=u(e);if("string"!=typeof t)return Promise.resolve();{let i=this.#s.get(t)?.find(t=>t!==e&&t.state.isPaused);return i?.continue()??Promise.resolve()}}clear(){r.notifyManager.batch(()=>{this.#o.forEach(e=>{this.notify({type:"removed",mutation:e})}),this.#o.clear(),this.#s.clear()})}getAll(){return Array.from(this.#o)}find(e){let i={exact:!0,...e};return this.getAll().find(e=>(0,t.matchMutation)(i,e))}findAll(e={}){return this.getAll().filter(i=>(0,t.matchMutation)(e,i))}notify(e){r.notifyManager.batch(()=>{this.listeners.forEach(t=>{t(e)})})}resumePausedMutations(){let e=this.getAll().filter(e=>e.state.isPaused);return r.notifyManager.batch(()=>Promise.all(e.map(e=>e.continue().catch(t.noop))))}};function u(e){return e.options.scope?.id}var c=e.i(175555),d=e.i(814448),p=e.i(992571),h=class{#u;#i;#c;#d;#p;#h;#m;#g;constructor(e={}){this.#u=e.queryCache||new n,this.#i=e.mutationCache||new l,this.#c=e.defaultOptions||{},this.#d=new Map,this.#p=new Map,this.#h=0}mount(){this.#h++,1===this.#h&&(this.#m=c.focusManager.subscribe(async e=>{e&&(await this.resumePausedMutations(),this.#u.onFocus())}),this.#g=d.onlineManager.subscribe(async e=>{e&&(await this.resumePausedMutations(),this.#u.onOnline())}))}unmount(){this.#h--,0===this.#h&&(this.#m?.(),this.#m=void 0,this.#g?.(),this.#g=void 0)}isFetching(e){return this.#u.findAll({...e,fetchStatus:"fetching"}).length}isMutating(e){return this.#i.findAll({...e,status:"pending"}).length}getQueryData(e){let t=this.defaultQueryOptions({queryKey:e});return this.#u.get(t.queryHash)?.state.data}ensureQueryData(e){let i=this.defaultQueryOptions(e),r=this.#u.build(this,i),a=r.state.data;return void 0===a?this.fetchQuery(e):(e.revalidateIfStale&&r.isStaleByTime((0,t.resolveStaleTime)(i.staleTime,r))&&this.prefetchQuery(i),Promise.resolve(a))}getQueriesData(e){return this.#u.findAll(e).map(({queryKey:e,state:t})=>[e,t.data])}setQueryData(e,i,r){let a=this.defaultQueryOptions({queryKey:e}),n=this.#u.get(a.queryHash),o=n?.state.data,s=(0,t.functionalUpdate)(i,o);if(void 0!==s)return this.#u.build(this,a).setData(s,{...r,manual:!0})}setQueriesData(e,t,i){return r.notifyManager.batch(()=>this.#u.findAll(e).map(({queryKey:e})=>[e,this.setQueryData(e,t,i)]))}getQueryState(e){let t=this.defaultQueryOptions({queryKey:e});return this.#u.get(t.queryHash)?.state}removeQueries(e){let t=this.#u;r.notifyManager.batch(()=>{t.findAll(e).forEach(e=>{t.remove(e)})})}resetQueries(e,t){let i=this.#u;return r.notifyManager.batch(()=>(i.findAll(e).forEach(e=>{e.reset()}),this.refetchQueries({type:"active",...e},t)))}cancelQueries(e,i={}){let a={revert:!0,...i};return Promise.all(r.notifyManager.batch(()=>this.#u.findAll(e).map(e=>e.cancel(a)))).then(t.noop).catch(t.noop)}invalidateQueries(e,t={}){return r.notifyManager.batch(()=>(this.#u.findAll(e).forEach(e=>{e.invalidate()}),e?.refetchType==="none")?Promise.resolve():this.refetchQueries({...e,type:e?.refetchType??e?.type??"active"},t))}refetchQueries(e,i={}){let a={...i,cancelRefetch:i.cancelRefetch??!0};return Promise.all(r.notifyManager.batch(()=>this.#u.findAll(e).filter(e=>!e.isDisabled()&&!e.isStatic()).map(e=>{let i=e.fetch(void 0,a);return a.throwOnError||(i=i.catch(t.noop)),"paused"===e.state.fetchStatus?Promise.resolve():i}))).then(t.noop)}fetchQuery(e){let i=this.defaultQueryOptions(e);void 0===i.retry&&(i.retry=!1);let r=this.#u.build(this,i);return r.isStaleByTime((0,t.resolveStaleTime)(i.staleTime,r))?r.fetch(i):Promise.resolve(r.state.data)}prefetchQuery(e){return this.fetchQuery(e).then(t.noop).catch(t.noop)}fetchInfiniteQuery(e){return e.behavior=(0,p.infiniteQueryBehavior)(e.pages),this.fetchQuery(e)}prefetchInfiniteQuery(e){return this.fetchInfiniteQuery(e).then(t.noop).catch(t.noop)}ensureInfiniteQueryData(e){return e.behavior=(0,p.infiniteQueryBehavior)(e.pages),this.ensureQueryData(e)}resumePausedMutations(){return d.onlineManager.isOnline()?this.#i.resumePausedMutations():Promise.resolve()}getQueryCache(){return this.#u}getMutationCache(){return this.#i}getDefaultOptions(){return this.#c}setDefaultOptions(e){this.#c=e}setQueryDefaults(e,i){this.#d.set((0,t.hashKey)(e),{queryKey:e,defaultOptions:i})}getQueryDefaults(e){let i=[...this.#d.values()],r={};return i.forEach(i=>{(0,t.partialMatchKey)(e,i.queryKey)&&Object.assign(r,i.defaultOptions)}),r}setMutationDefaults(e,i){this.#p.set((0,t.hashKey)(e),{mutationKey:e,defaultOptions:i})}getMutationDefaults(e){let i=[...this.#p.values()],r={};return i.forEach(i=>{(0,t.partialMatchKey)(e,i.mutationKey)&&Object.assign(r,i.defaultOptions)}),r}defaultQueryOptions(e){if(e._defaulted)return e;let i={...this.#c.queries,...this.getQueryDefaults(e.queryKey),...e,_defaulted:!0};return i.queryHash||(i.queryHash=(0,t.hashQueryKeyByOptions)(i.queryKey,i)),void 0===i.refetchOnReconnect&&(i.refetchOnReconnect="always"!==i.networkMode),void 0===i.throwOnError&&(i.throwOnError=!!i.suspense),!i.networkMode&&i.persister&&(i.networkMode="offlineFirst"),i.queryFn===t.skipToken&&(i.enabled=!1),i}defaultMutationOptions(e){return e?._defaulted?e:{...this.#c.mutations,...e?.mutationKey&&this.getMutationDefaults(e.mutationKey),...e,_defaulted:!0}}clear(){this.#u.clear(),this.#i.clear()}};e.s(["QueryClient",()=>h],317751)},502547,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,i],502547)},262218,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),r=e.i(529681),a=e.i(702779),n=e.i(563113),o=e.i(763731),s=e.i(121872),l=e.i(242064);e.i(296059);var u=e.i(915654);e.i(262370);var c=e.i(135551),d=e.i(183293),p=e.i(246422),h=e.i(838378);let m=e=>{let{lineWidth:t,fontSizeIcon:i,calc:r}=e,a=e.fontSizeSM;return(0,h.mergeToken)(e,{tagFontSize:a,tagLineHeight:(0,u.unit)(r(e.lineHeightSM).mul(a).equal()),tagIconSize:r(i).sub(r(t).mul(2)).equal(),tagPaddingHorizontal:8,tagBorderlessBg:e.defaultBg})},g=e=>({defaultBg:new c.FastColor(e.colorFillQuaternary).onBackground(e.colorBgContainer).toHexString(),defaultColor:e.colorText}),f=(0,p.genStyleHooks)("Tag",e=>(e=>{let{paddingXXS:t,lineWidth:i,tagPaddingHorizontal:r,componentCls:a,calc:n}=e,o=n(r).sub(i).equal(),s=n(t).sub(i).equal();return{[a]:Object.assign(Object.assign({},(0,d.resetComponent)(e)),{display:"inline-block",height:"auto",marginInlineEnd:e.marginXS,paddingInline:o,fontSize:e.tagFontSize,lineHeight:e.tagLineHeight,whiteSpace:"nowrap",background:e.defaultBg,border:`${(0,u.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,opacity:1,transition:`all ${e.motionDurationMid}`,textAlign:"start",position:"relative",[`&${a}-rtl`]:{direction:"rtl"},"&, a, a:hover":{color:e.defaultColor},[`${a}-close-icon`]:{marginInlineStart:s,fontSize:e.tagIconSize,color:e.colorIcon,cursor:"pointer",transition:`all ${e.motionDurationMid}`,"&:hover":{color:e.colorTextHeading}},[`&${a}-has-color`]:{borderColor:"transparent",[`&, a, a:hover, ${e.iconCls}-close, ${e.iconCls}-close:hover`]:{color:e.colorTextLightSolid}},"&-checkable":{backgroundColor:"transparent",borderColor:"transparent",cursor:"pointer",[`&:not(${a}-checkable-checked):hover`]:{color:e.colorPrimary,backgroundColor:e.colorFillSecondary},"&:active, &-checked":{color:e.colorTextLightSolid},"&-checked":{backgroundColor:e.colorPrimary,"&:hover":{backgroundColor:e.colorPrimaryHover}},"&:active":{backgroundColor:e.colorPrimaryActive}},"&-hidden":{display:"none"},[`> ${e.iconCls} + span, > span + ${e.iconCls}`]:{marginInlineStart:o}}),[`${a}-borderless`]:{borderColor:"transparent",background:e.tagBorderlessBg}}})(m(e)),g);var y=function(e,t){var i={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(i[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,r=Object.getOwnPropertySymbols(e);a<r.length;a++)0>t.indexOf(r[a])&&Object.prototype.propertyIsEnumerable.call(e,r[a])&&(i[r[a]]=e[r[a]]);return i};let b=t.forwardRef((e,r)=>{let{prefixCls:a,style:n,className:o,checked:s,children:u,icon:c,onChange:d,onClick:p}=e,h=y(e,["prefixCls","style","className","checked","children","icon","onChange","onClick"]),{getPrefixCls:m,tag:g}=t.useContext(l.ConfigContext),b=m("tag",a),[_,v,w]=f(b),C=(0,i.default)(b,`${b}-checkable`,{[`${b}-checkable-checked`]:s},null==g?void 0:g.className,o,v,w);return _(t.createElement("span",Object.assign({},h,{ref:r,style:Object.assign(Object.assign({},n),null==g?void 0:g.style),className:C,onClick:e=>{null==d||d(!s),null==p||p(e)}}),c,t.createElement("span",null,u)))});var _=e.i(403541);let v=(0,p.genSubStyleComponent)(["Tag","preset"],e=>{let t;return t=m(e),(0,_.genPresetColor)(t,(e,{textColor:i,lightBorderColor:r,lightColor:a,darkColor:n})=>({[`${t.componentCls}${t.componentCls}-${e}`]:{color:i,background:a,borderColor:r,"&-inverse":{color:t.colorTextLightSolid,background:n,borderColor:n},[`&${t.componentCls}-borderless`]:{borderColor:"transparent"}}}))},g),w=(e,t,i)=>{let r="string"!=typeof i?i:i.charAt(0).toUpperCase()+i.slice(1);return{[`${e.componentCls}${e.componentCls}-${t}`]:{color:e[`color${i}`],background:e[`color${r}Bg`],borderColor:e[`color${r}Border`],[`&${e.componentCls}-borderless`]:{borderColor:"transparent"}}}},C=(0,p.genSubStyleComponent)(["Tag","status"],e=>{let t=m(e);return[w(t,"success","Success"),w(t,"processing","Info"),w(t,"error","Error"),w(t,"warning","Warning")]},g);var x=function(e,t){var i={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(i[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,r=Object.getOwnPropertySymbols(e);a<r.length;a++)0>t.indexOf(r[a])&&Object.prototype.propertyIsEnumerable.call(e,r[a])&&(i[r[a]]=e[r[a]]);return i};let O=t.forwardRef((e,u)=>{let{prefixCls:c,className:d,rootClassName:p,style:h,children:m,icon:g,color:y,onClose:b,bordered:_=!0,visible:w}=e,O=x(e,["prefixCls","className","rootClassName","style","children","icon","color","onClose","bordered","visible"]),{getPrefixCls:S,direction:E,tag:$}=t.useContext(l.ConfigContext),[I,P]=t.useState(!0),k=(0,r.default)(O,["closeIcon","closable"]);t.useEffect(()=>{void 0!==w&&P(w)},[w]);let M=(0,a.isPresetColor)(y),A=(0,a.isPresetStatusColor)(y),j=M||A,T=Object.assign(Object.assign({backgroundColor:y&&!j?y:void 0},null==$?void 0:$.style),h),q=S("tag",c),[D,R,N]=f(q),L=(0,i.default)(q,null==$?void 0:$.className,{[`${q}-${y}`]:j,[`${q}-has-color`]:y&&!j,[`${q}-hidden`]:!I,[`${q}-rtl`]:"rtl"===E,[`${q}-borderless`]:!_},d,p,R,N),B=e=>{e.stopPropagation(),null==b||b(e),e.defaultPrevented||P(!1)},[,Q]=(0,n.useClosable)((0,n.pickClosable)(e),(0,n.pickClosable)($),{closable:!1,closeIconRender:e=>{let r=t.createElement("span",{className:`${q}-close-icon`,onClick:B},e);return(0,o.replaceElement)(e,r,e=>({onClick:t=>{var i;null==(i=null==e?void 0:e.onClick)||i.call(e,t),B(t)},className:(0,i.default)(null==e?void 0:e.className,`${q}-close-icon`)}))}}),z="function"==typeof O.onClick||m&&"a"===m.type,H=g||null,F=H?t.createElement(t.Fragment,null,H,m&&t.createElement("span",null,m)):m,U=t.createElement("span",Object.assign({},k,{ref:u,className:L,style:T}),F,Q,M&&t.createElement(v,{key:"preset",prefixCls:q}),A&&t.createElement(C,{key:"status",prefixCls:q}));return D(z?t.createElement(s.default,{component:"Tag"},U):U)});O.CheckableTag=b,e.s(["Tag",0,O],262218)},801312,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M724 218.3V141c0-6.7-7.7-10.4-12.9-6.3L260.3 486.8a31.86 31.86 0 000 50.3l450.8 352.1c5.3 4.1 12.9.4 12.9-6.3v-77.3c0-4.9-2.3-9.6-6.1-12.6l-360-281 360-281.1c3.8-3 6.1-7.7 6.1-12.6z"}}]},name:"left",theme:"outlined"};var a=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(a.default,(0,t.default)({},e,{ref:n,icon:r}))});e.s(["default",0,n],801312)},475254,e=>{"use strict";var t=e.i(271645);let i=e=>{let t=e.replace(/^([A-Z])|[\s-_]+(\w)/g,(e,t,i)=>i?i.toUpperCase():t.toLowerCase());return t.charAt(0).toUpperCase()+t.slice(1)},r=(...e)=>e.filter((e,t,i)=>!!e&&""!==e.trim()&&i.indexOf(e)===t).join(" ").trim();var a={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};let n=(0,t.forwardRef)(({color:e="currentColor",size:i=24,strokeWidth:n=2,absoluteStrokeWidth:o,className:s="",children:l,iconNode:u,...c},d)=>(0,t.createElement)("svg",{ref:d,...a,width:i,height:i,stroke:e,strokeWidth:o?24*Number(n)/Number(i):n,className:r("lucide",s),...!l&&!(e=>{for(let t in e)if(t.startsWith("aria-")||"role"===t||"title"===t)return!0})(c)&&{"aria-hidden":"true"},...c},[...u.map(([e,i])=>(0,t.createElement)(e,i)),...Array.isArray(l)?l:[l]])),o=(e,a)=>{let o=(0,t.forwardRef)(({className:o,...s},l)=>(0,t.createElement)(n,{ref:l,iconNode:a,className:r(`lucide-${i(e).replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase()}`,`lucide-${e}`,o),...s}));return o.displayName=i(e),o};e.s(["default",()=>o],475254)},312361,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),r=e.i(242064),a=e.i(517455);e.i(296059);var n=e.i(915654),o=e.i(183293),s=e.i(246422),l=e.i(838378);let u=(0,s.genStyleHooks)("Divider",e=>{let t=(0,l.mergeToken)(e,{dividerHorizontalWithTextGutterMargin:e.margin,sizePaddingEdgeHorizontal:0});return[(e=>{let{componentCls:t,sizePaddingEdgeHorizontal:i,colorSplit:r,lineWidth:a,textPaddingInline:s,orientationMargin:l,verticalMarginInline:u}=e;return{[t]:Object.assign(Object.assign({},(0,o.resetComponent)(e)),{borderBlockStart:`${(0,n.unit)(a)} solid ${r}`,"&-vertical":{position:"relative",top:"-0.06em",display:"inline-block",height:"0.9em",marginInline:u,marginBlock:0,verticalAlign:"middle",borderTop:0,borderInlineStart:`${(0,n.unit)(a)} solid ${r}`},"&-horizontal":{display:"flex",clear:"both",width:"100%",minWidth:"100%",margin:`${(0,n.unit)(e.marginLG)} 0`},[`&-horizontal${t}-with-text`]:{display:"flex",alignItems:"center",margin:`${(0,n.unit)(e.dividerHorizontalWithTextGutterMargin)} 0`,color:e.colorTextHeading,fontWeight:500,fontSize:e.fontSizeLG,whiteSpace:"nowrap",textAlign:"center",borderBlockStart:`0 ${r}`,"&::before, &::after":{position:"relative",width:"50%",borderBlockStart:`${(0,n.unit)(a)} solid transparent`,borderBlockStartColor:"inherit",borderBlockEnd:0,transform:"translateY(50%)",content:"''"}},[`&-horizontal${t}-with-text-start`]:{"&::before":{width:`calc(${l} * 100%)`},"&::after":{width:`calc(100% - ${l} * 100%)`}},[`&-horizontal${t}-with-text-end`]:{"&::before":{width:`calc(100% - ${l} * 100%)`},"&::after":{width:`calc(${l} * 100%)`}},[`${t}-inner-text`]:{display:"inline-block",paddingBlock:0,paddingInline:s},"&-dashed":{background:"none",borderColor:r,borderStyle:"dashed",borderWidth:`${(0,n.unit)(a)} 0 0`},[`&-horizontal${t}-with-text${t}-dashed`]:{"&::before, &::after":{borderStyle:"dashed none none"}},[`&-vertical${t}-dashed`]:{borderInlineStartWidth:a,borderInlineEnd:0,borderBlockStart:0,borderBlockEnd:0},"&-dotted":{background:"none",borderColor:r,borderStyle:"dotted",borderWidth:`${(0,n.unit)(a)} 0 0`},[`&-horizontal${t}-with-text${t}-dotted`]:{"&::before, &::after":{borderStyle:"dotted none none"}},[`&-vertical${t}-dotted`]:{borderInlineStartWidth:a,borderInlineEnd:0,borderBlockStart:0,borderBlockEnd:0},[`&-plain${t}-with-text`]:{color:e.colorText,fontWeight:"normal",fontSize:e.fontSize},[`&-horizontal${t}-with-text-start${t}-no-default-orientation-margin-start`]:{"&::before":{width:0},"&::after":{width:"100%"},[`${t}-inner-text`]:{paddingInlineStart:i}},[`&-horizontal${t}-with-text-end${t}-no-default-orientation-margin-end`]:{"&::before":{width:"100%"},"&::after":{width:0},[`${t}-inner-text`]:{paddingInlineEnd:i}}})}})(t),(e=>{let{componentCls:t}=e;return{[t]:{"&-horizontal":{[`&${t}`]:{"&-sm":{marginBlock:e.marginXS},"&-md":{marginBlock:e.margin}}}}}})(t)]},e=>({textPaddingInline:"1em",orientationMargin:.05,verticalMarginInline:e.marginXS}),{unitless:{orientationMargin:!0}});var c=function(e,t){var i={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&0>t.indexOf(r)&&(i[r]=e[r]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,r=Object.getOwnPropertySymbols(e);a<r.length;a++)0>t.indexOf(r[a])&&Object.prototype.propertyIsEnumerable.call(e,r[a])&&(i[r[a]]=e[r[a]]);return i};let d={small:"sm",middle:"md"};e.s(["Divider",0,e=>{let{getPrefixCls:n,direction:o,className:s,style:l}=(0,r.useComponentConfig)("divider"),{prefixCls:p,type:h="horizontal",orientation:m="center",orientationMargin:g,className:f,rootClassName:y,children:b,dashed:_,variant:v="solid",plain:w,style:C,size:x}=e,O=c(e,["prefixCls","type","orientation","orientationMargin","className","rootClassName","children","dashed","variant","plain","style","size"]),S=n("divider",p),[E,$,I]=u(S),P=d[(0,a.default)(x)],k=!!b,M=t.useMemo(()=>"left"===m?"rtl"===o?"end":"start":"right"===m?"rtl"===o?"start":"end":m,[o,m]),A="start"===M&&null!=g,j="end"===M&&null!=g,T=(0,i.default)(S,s,$,I,`${S}-${h}`,{[`${S}-with-text`]:k,[`${S}-with-text-${M}`]:k,[`${S}-dashed`]:!!_,[`${S}-${v}`]:"solid"!==v,[`${S}-plain`]:!!w,[`${S}-rtl`]:"rtl"===o,[`${S}-no-default-orientation-margin-start`]:A,[`${S}-no-default-orientation-margin-end`]:j,[`${S}-${P}`]:!!P},f,y),q=t.useMemo(()=>"number"==typeof g?g:/^\d+$/.test(g)?Number(g):g,[g]);return E(t.createElement("div",Object.assign({className:T,style:Object.assign(Object.assign({},l),C)},O,{role:"separator"}),b&&"vertical"!==h&&t.createElement("span",{className:`${S}-inner-text`,style:{marginInlineStart:A?q:void 0,marginInlineEnd:j?q:void 0}},b)))}],312361)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var a=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(a.default,(0,t.default)({},e,{ref:n,icon:r}))});e.s(["UserOutlined",0,n],771674)},250980,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,i],250980)},948401,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M928 160H96c-17.7 0-32 14.3-32 32v640c0 17.7 14.3 32 32 32h832c17.7 0 32-14.3 32-32V192c0-17.7-14.3-32-32-32zm-40 110.8V792H136V270.8l-27.6-21.5 39.3-50.5 42.8 33.3h643.1l42.8-33.3 39.3 50.5-27.7 21.5zM833.6 232L512 482 190.4 232l-42.8-33.3-39.3 50.5 27.6 21.5 341.6 265.6a55.99 55.99 0 0068.7 0L888 270.8l27.6-21.5-39.3-50.5-42.7 33.2z"}}]},name:"mail",theme:"outlined"};var a=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(a.default,(0,t.default)({},e,{ref:n,icon:r}))});e.s(["MailOutlined",0,n],948401)},209261,e=>{"use strict";e.s(["extractCategories",0,e=>{let t=new Set;return e.forEach(e=>{e.category&&""!==e.category.trim()&&t.add(e.category)}),["All",...Array.from(t).sort(),"Other"]},"filterPluginsByCategory",0,(e,t)=>"All"===t?e:"Other"===t?e.filter(e=>!e.category||""===e.category.trim()):e.filter(e=>e.category===t),"filterPluginsBySearch",0,(e,t)=>{if(!t||""===t.trim())return e;let i=t.toLowerCase().trim();return e.filter(e=>{let t=e.name.toLowerCase().includes(i),r=e.description?.toLowerCase().includes(i)||!1,a=e.keywords?.some(e=>e.toLowerCase().includes(i))||!1;return t||r||a})},"formatDateString",0,e=>{if(!e)return"N/A";try{return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"})}catch(e){return"Invalid date"}},"formatInstallCommand",0,e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"url"===e.source&&e.url?e.url:"Unknown source","getSourceLink",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:"url"===e.source&&e.url?e.url:null,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return a}});let r=e.r(271645);function a(e,t){let i=(0,r.useRef)(null),a=(0,r.useRef)(null);return(0,r.useCallback)(r=>{if(null===r){let e=i.current;e&&(i.current=null,e());let t=a.current;t&&(a.current=null,t())}else e&&(i.current=n(e,r)),t&&(a.current=n(t,r))},[e,t])}function n(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},62478,e=>{"use strict";var t=e.i(764205);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},56456,e=>{"use strict";var t=e.i(739295);e.s(["LoadingOutlined",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let r={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var a=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(a.default,(0,t.default)({},e,{ref:n,icon:r}))});e.s(["SafetyOutlined",0,n],602073)},190272,785913,e=>{"use strict";var t,i,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i);let n={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=n[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:r,apiKey:n,inputMessage:o,chatHistory:s,selectedTags:l,selectedVectorStores:u,selectedGuardrails:c,selectedPolicies:d,selectedMCPServers:p,mcpServers:h,mcpServerToolRestrictions:m,selectedVoice:g,endpointType:f,selectedModel:y,selectedSdk:b,proxySettings:_}=e,v="session"===i?r:n,w=window.location.origin,C=_?.LITELLM_UI_API_DOC_BASE_URL;C&&C.trim()?w=C:_?.PROXY_BASE_URL&&(w=_.PROXY_BASE_URL);let x=o||"Your prompt here",O=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),u.length>0&&(E.vector_stores=u),c.length>0&&(E.guardrails=c),d.length>0&&(E.policies=d);let $=y||"your-model-name",I="azure"===b?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${w}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${w}"
)`;switch(f){case a.CHAT:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=S.length>0?S:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${$}",
    messages=${JSON.stringify(r,null,4)}${i}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${$}",
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
`;break}case a.RESPONSES:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let r=S.length>0?S:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${$}",
    input=${JSON.stringify(r,null,4)}${i}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${$}",
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
	model="${$}",
	prompt="${o}",
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
	model="${$}",
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
prompt = "${O}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${$}",
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
	model="${$}",
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
	input="${o||"Your string here"}",
	model="${$}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${$}",
	file=audio_file${o?`,
	prompt="${o.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${$}",
	input="${o||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${$}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},115571,371401,e=>{"use strict";let t="local-storage-change";function i(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function r(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function a(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function n(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>i,"getLocalStorageItem",()=>r,"removeLocalStorageItem",()=>n,"setLocalStorageItem",()=>a],115571);var o=e.i(271645);function s(e){let i=t=>{"disableUsageIndicator"===t.key&&e()},r=t=>{let{key:i}=t.detail;"disableUsageIndicator"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t,r),()=>{window.removeEventListener("storage",i),window.removeEventListener(t,r)}}function l(){return"true"===r("disableUsageIndicator")}function u(){return(0,o.useSyncExternalStore)(s,l)}e.s(["useDisableUsageIndicator",()=>u],371401)},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(764205);let a=(0,i.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:n})=>{let[o,s]=(0,i.useState)(null),[l,u]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,r.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let e=await i.json();e.values?.logo_url&&s(e.values.logo_url),e.values?.favicon_url&&u(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,i.useEffect)(()=>{if(l){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=l});else{let e=document.createElement("link");e.rel="icon",e.href=l,document.head.appendChild(e)}}},[l]),(0,t.jsx)(a.Provider,{value:{logoUrl:o,setLogoUrl:s,faviconUrl:l,setFaviconUrl:u},children:e})},"useTheme",0,()=>{let e=(0,i.useContext)(a);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},86408,e=>{"use strict";var t=e.i(843476),i=e.i(271645),r=e.i(618566),a=e.i(934879),n=e.i(317751),o=e.i(912598);let s=new n.QueryClient;function l(){let e=(0,r.useSearchParams)().get("key"),[n,l]=(0,i.useState)(null);return console.log("PublicModelHubTable accessToken:",n),(0,i.useEffect)(()=>{e&&l(e)},[e]),(0,t.jsx)(o.QueryClientProvider,{client:s,children:(0,t.jsx)(a.default,{accessToken:n,publicPage:!0,premiumUser:!1,userRole:null})})}function u(){return(0,t.jsx)(i.Suspense,{fallback:(0,t.jsx)("div",{className:"flex items-center justify-center min-h-screen",children:"Loading..."}),children:(0,t.jsx)(l,{})})}e.s(["default",()=>u])}]);