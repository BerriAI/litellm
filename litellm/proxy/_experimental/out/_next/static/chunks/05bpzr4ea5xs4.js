(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,871943,502547,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["ChevronDownIcon",0,o],871943);let n=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,n],502547)},360820,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["ChevronUpIcon",0,o],360820)},434626,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,o],434626)},250980,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,o],250980)},278587,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["RefreshIcon",0,o],278587)},68155,e=>{"use strict";var t=e.i(271645);let o=t.forwardRef(function(e,o){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:o},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"}))});e.s(["TrashIcon",0,o],68155)},655063,e=>{"use strict";var t=e.i(540626),o=e.i(271645);e.s(["useDebouncedValue",0,function(e,n,i){let[r,s,a]=function(e,n,i){let[r,s]=(0,o.useState)(e),a=(0,t.useDebouncer)(s,n,i);return[r,a.maybeExecute,a]}(e,n,i);return(0,o.useEffect)(()=>{s(e)},[e,s]),[r,a]}],655063)},540626,e=>{"use strict";let t;var o=e.i(271645);let n=(0,o.createContext)(null);function i(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[o,n]of e)if(!t.has(o)||!Object.is(n,t.get(o)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let o of e)if(!t.has(o))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let o=r(e);if(o.length!==r(t).length)return!1;for(let n=0;n<o.length;n++)if(!Object.prototype.hasOwnProperty.call(t,o[n])||!Object.is(e[o[n]],t[o[n]]))return!1;return!0}function r(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var s=e.i(430224);function a(e,t){return e===t}function l(e,t=e=>e,n){let i=n?.compare??a,r=(0,o.useCallback)(t=>{let{unsubscribe:o}=e.subscribe(t);return o},[e]),u=(0,o.useCallback)(()=>e.get(),[e]);return(0,s.useSyncExternalStoreWithSelector)(r,u,u,t,i)}function u(e,...t){return"function"==typeof e?e(...t):e}var c=class{#e=!0;#t;#o;#n;#i;#r;#s;#a;#l=0;#u=5;#c=!1;#d=!1;#p=null;#h=()=>{this.debugLog("Connected to event bus"),this.#r=!0,this.#c=!1,this.debugLog("Emitting queued events",this.#i),this.#i.forEach(e=>this.emitEventToBus(e)),this.#i=[],this.stopConnectLoop(),this.#o().removeEventListener("tanstack-connect-success",this.#h)};#g=()=>{if(this.#l<this.#u){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#o().removeEventListener("tanstack-connect",this.#g),this.#d=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#m=()=>{this.#c||(this.#c=!0,this.#o().addEventListener("tanstack-connect-success",this.#h),this.#g())};constructor({pluginId:e,debug:t=!1,enabled:o=!0,reconnectEveryMs:n=300}){this.#t=e,this.#e=o,this.#o=this.getGlobalTarget,this.#n=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#i=[],this.#r=!1,this.#d=!1,this.#s=null,this.#a=n}startConnectLoop(){null!==this.#s||this.#r||(this.debugLog(`Starting connect loop (every ${this.#a}ms)`),this.#s=setInterval(this.#g,this.#a))}stopConnectLoop(){this.#c=!1,null!==this.#s&&(clearInterval(this.#s),this.#s=null,this.#i=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#n&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let o=new Event(e,{detail:t});this.#o().dispatchEvent(o)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#o().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(o){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#p&&(this.debugLog("Emitting event to internal event target",e,t),this.#p.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#d)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#r){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#i.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#c&&(this.#m(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,o){let n=o?.withEventTarget??!1,i=`${this.#t}:${e}`;if(n&&(this.#p||(this.#p=new EventTarget),this.#p.addEventListener(i,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",i),()=>{};let r=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#o().addEventListener(i,r),this.debugLog("Registered event to bus",i),()=>{n&&this.#p?.removeEventListener(i,r),this.#o().removeEventListener(i,r)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#o().addEventListener("tanstack-devtools-global",t),()=>this.#o().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let o=t.detail;this.#t&&o.pluginId!==this.#t||e(o)};return this.#o().addEventListener("tanstack-devtools-global",t),()=>this.#o().removeEventListener("tanstack-devtools-global",t)}};let d=new Map;function p(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let h=new class extends c{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function g(e,t,o){let n="object"==typeof e,i=n?e:void 0;return{next:(n?e.next:e)?.bind(i),error:(n?e.error:t)?.bind(i),complete:(n?e.complete:o)?.bind(i)}}let m=[],b=0,{link:f,unlink:v,propagate:_,checkDirty:k,shallowPropagate:E}=function({update:e,notify:t,unwatched:o}){return{link:function(e,t,o){let n=t.depsTail;if(void 0!==n&&n.dep===e)return;let i=void 0!==n?n.nextDep:t.deps;if(void 0!==i&&i.dep===e){i.version=o,t.depsTail=i;return}let r=e.subsTail;if(void 0!==r&&r.version===o&&r.sub===t)return;let s=t.depsTail=e.subsTail={version:o,dep:e,sub:t,prevDep:n,nextDep:i,prevSub:r,nextSub:void 0};void 0!==i&&(i.prevDep=s),void 0!==n?n.nextDep=s:t.deps=s,void 0!==r?r.nextSub=s:e.subs=s},unlink:function(e,t=e.sub){let n=e.dep,i=e.prevDep,r=e.nextDep,s=e.nextSub,a=e.prevSub;return void 0!==r?r.prevDep=i:t.depsTail=i,void 0!==i?i.nextDep=r:t.deps=r,void 0!==s?s.prevSub=a:n.subsTail=a,void 0!==a?a.nextSub=s:void 0===(n.subs=s)&&o(n),r},propagate:function(e){let o,n=e.nextSub;e:for(;;){let i=e.sub,r=i.flags;if(60&r?12&r?4&r?!(48&r)&&function(e,t){let o=t.depsTail;for(;void 0!==o;){if(o===e)return!0;o=o.prevDep}return!1}(e,i)?(i.flags=40|r,r&=1):r=0:i.flags=-9&r|32:r=0:i.flags=32|r,2&r&&t(i),1&r){let t=i.subs;if(void 0!==t){let i=(e=t).nextSub;void 0!==i&&(o={value:n,prev:o},n=i);continue}}if(void 0!==(e=n)){n=e.nextSub;continue}for(;void 0!==o;)if(e=o.value,o=o.prev,void 0!==e){n=e.nextSub;continue e}break}},checkDirty:function(t,o){let i,r=0,s=!1;e:for(;;){let a=t.dep,l=a.flags;if(16&o.flags)s=!0;else if((17&l)==17){if(e(a)){let e=a.subs;void 0!==e.nextSub&&n(e),s=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(i={value:t,prev:i}),t=a.deps,o=a,++r;continue}if(!s){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;r--;){let r=o.subs,a=void 0!==r.nextSub;if(a?(t=i.value,i=i.prev):t=r,s){if(e(o)){a&&n(r),o=t.sub;continue}s=!1}else o.flags&=-33;o=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return s}},shallowPropagate:n};function n(e){do{let o=e.sub,n=o.flags;(48&n)==32&&(o.flags=16|n,(6&n)==2&&t(o))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){m[w++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,x(e))}}),y=0,w=0;function x(e){let t=e.depsTail,o=void 0!==t?t.nextDep:e.deps;for(;void 0!==o;)o=v(o,e)}var T=class{constructor(e,o){this.atom=function(e){let o="function"==typeof e,n={_snapshot:o?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!o,get:()=>(void 0!==t&&f(n,t,b),n._snapshot),subscribe(e){var o;let i,r,s=g(e),a={current:!1},l=(o=()=>{n.get(),a.current?s.next?.(n._snapshot):a.current=!0},i=()=>{let e=t;t=r,++b,r.depsTail=void 0,r.flags=6;try{return o()}finally{t=e,r.flags&=-5,x(r)}},r={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&k(this.deps,this)?i():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,x(this)}},i(),r);return{unsubscribe:()=>{l.stop()}}},_update(i){let r=t,s=(void 0)??Object.is;if(o)t=n,++b,n.depsTail=void 0;else if(void 0===i)return!1;o&&(n.flags=5);try{let t=n._snapshot,r="function"==typeof i?i(t):void 0===i&&o?e(t):i;if(void 0===t||!s(t,r))return n._snapshot=r,!0;return!1}finally{t=r,o&&(n.flags&=-5),x(n)}}};return o?(n.flags=17,n.get=function(){let e=n.flags;if(16&e||32&e&&k(n.deps,n)){if(n._update()){let e=n.subs;void 0!==e&&E(e)}}else 32&e&&(n.flags=-33&e);return void 0!==t&&f(n,t,b),n._snapshot}):n.set=function(e){if(n._update(e)){let e=n.subs;if(void 0!==e&&(_(e),E(e),1)){for(;y<w;){let e=m[y];m[y++]=void 0,e.notify()}y=0,w=0}}},n}(e),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),o&&(this.actions=o(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(g(e))}};function C(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let S={enabled:!0,leading:!1,trailing:!0,wait:0};var O=class{#b;constructor(e,t){this.fn=e,this.store=new T(C()),this.setOptions=e=>{this.options={...this.options,...e},this.#f()||this.cancel()},this.#v=e=>{this.store.setState(t=>{let o={...t,...e},{isPending:n}=o;return{...o,status:this.#f()?n?"pending":"idle":"disabled"}}),((e,t)=>{let o=t.key;if(o){var n,i;d.set(o,t),h.emit(e,{key:(n={...t,key:o}).key,store:{state:p("function"==typeof(i=n.store).get?i.get():i.state)},options:p(n.options)})}})("Debouncer",this)},this.#f=()=>!!u(this.options.enabled,this),this.#_=()=>u(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#f())return;this.#v({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#v({canLeadingExecute:!1}),t=!0,this.#k(...e)),this.options.trailing&&this.#v({isPending:!0,lastArgs:e}),this.#b&&clearTimeout(this.#b),this.#b=setTimeout(()=>{this.#v({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#k(...e)},this.#_())},this.#k=(...e)=>{this.#f()&&(this.fn(...e),this.#v({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#E(),this.#k(...this.store.state.lastArgs))},this.#E=()=>{this.#b&&(clearTimeout(this.#b),this.#b=void 0)},this.cancel=()=>{this.#E(),this.#v({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#v(C())},this.key=t.key,this.options={...S,...t},this.#v(this.options.initialState??{}),this.key&&h.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#v(e.payload.store.state),this.setOptions(e.payload.options))})}#v;#f;#_;#k;#E};e.s(["useDebouncer",0,function(e,t,r=()=>({})){let s={...((0,o.useContext)(n)?.defaultOptions??{}).debouncer,...t},[a]=(0,o.useState)(()=>{let t=new O(e,s);return t.Subscribe=function(e){let o=l(t.store,e.selector,{compare:i});return"function"==typeof e.children?e.children(o):e.children},t});a.fn=e,a.setOptions(s),(0,o.useEffect)(()=>()=>{s.onUnmount?s.onUnmount(a):a.cancel()},[]);let u=l(a.store,r,{compare:i});return(0,o.useMemo)(()=>({...a,state:u}),[a,u])}],540626)},586448,e=>{"use strict";var t=e.i(271645),o=e.i(540143),n=e.i(869230),i=e.i(915823),r=e.i(619273);function s(e,t){let o=new Set(t);return e.filter(e=>!o.has(e))}var a=class extends i.Subscribable{#y;#w;#x;#T;#C;#S;#O;#I;#L;#A=[];constructor(e,t,o){super(),this.#y=e,this.#T=o,this.#x=[],this.#C=[],this.#w=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#C.forEach(e=>{e.subscribe(t=>{this.#j(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#C.forEach(e=>{e.destroy()})}setQueries(e,t){this.#x=e,this.#T=t,o.notifyManager.batch(()=>{let e=this.#C,t=this.#R(this.#x);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let o=t.map(e=>e.observer),n=o.map(e=>e.getCurrentResult()),i=e.length!==o.length,a=o.some((t,o)=>t!==e[o]),l=i||a,u=!!l||n.some((e,t)=>{let o=this.#w[t];return!o||!(0,r.shallowEqualObjects)(e,o)});(l||u)&&(l&&(this.#A=t,this.#C=o),this.#w=n,this.hasListeners()&&(l&&(s(e,o).forEach(e=>{e.destroy()}),s(o,e).forEach(e=>{e.subscribe(t=>{this.#j(e,t)})})),this.#M()))})}getCurrentResult(){return this.#w}getQueries(){return this.#C.map(e=>e.getCurrentQuery())}getObservers(){return this.#C}getOptimisticResult(e,t){let o=this.#R(e),n=o.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),i=o.map(e=>e.defaultedQueryOptions.queryHash);return[n,e=>this.#N(e??n,t,i),()=>this.#H(n,o)]}#H(e,t){return t.map((o,n)=>{let i=e[n];return o.defaultedQueryOptions.notifyOnChangeProps?i:o.observer.trackResult(i,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#N(e,t,o){if(t){let n=this.#L,i=void 0!==o&&void 0!==n&&(n.length!==o.length||o.some((e,t)=>e!==n[t]));return(!this.#S||this.#w!==this.#I||i||t!==this.#O)&&(this.#O=t,this.#I=this.#w,void 0!==o&&(this.#L=o),this.#S=(0,r.replaceEqualDeep)(this.#S,t(e))),this.#S}return e}#P(){return this.#T?.combine!==void 0&&this.#C.some((e,t)=>e.options.suspense&&this.#w[t]?.data===void 0)}#R(e){let t=new Map;this.#C.forEach(e=>{let o=e.options.queryHash;if(!o)return;let n=t.get(o);n?n.push(e):t.set(o,[e])});let o=[];return e.forEach(e=>{let i=this.#y.defaultQueryOptions(e),r=t.get(i.queryHash)?.shift()??new n.QueryObserver(this.#y,i);o.push({defaultedQueryOptions:i,observer:r})}),o}#j(e,t){let o=this.#C.indexOf(e);if(-1!==o){var n;let e;this.#w=(n=this.#w,(e=n.slice(0))[o]=t,e),this.#M()}}#M(){if(this.hasListeners()){let e=this.#H(this.#w,this.#A),t=this.#P(),n=this.#S,i=t?n:this.#N(e,this.#T?.combine);(t||n!==i)&&o.notifyManager.batch(()=>{this.listeners.forEach(e=>{e(this.#w)})})}}},l=e.i(912598),u=e.i(381384),c=e.i(673664),d=e.i(427001),p=e.i(254440);e.s(["useQueries",0,function({queries:e,...i},s){let h=(0,l.useQueryClient)(s),g=(0,u.useIsRestoring)(),m=(0,c.useQueryErrorResetBoundary)(),b=t.useMemo(()=>e.map(e=>{let t=h.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,h,g]);b.forEach(e=>{(0,p.ensureSuspenseTimers)(e);let t=h.getQueryCache().get(e.queryHash);(0,d.ensurePreventErrorBoundaryRetry)(e,m,t)}),(0,d.useClearResetErrorBoundary)(m);let[f]=t.useState(()=>new a(h,b,i)),[v,_,k]=f.getOptimisticResult(b,i.combine),E=!g&&!1!==i.subscribed;t.useSyncExternalStore(t.useCallback(e=>E?f.subscribe(o.notifyManager.batchCalls(e)):r.noop,[f,E]),()=>f.getCurrentResult(),()=>f.getCurrentResult()),t.useEffect(()=>{f.setQueries(b,i)},[b,i,f]);let y=v.some((e,t)=>(0,p.shouldSuspend)(b[t],e))?v.flatMap((e,t)=>{let o=b[t];if(o&&(0,p.shouldSuspend)(o,e)){let e=new n.QueryObserver(h,o);return(0,p.fetchOptimistic)(o,e,m)}return[]}):[];if(y.length>0)throw Promise.all(y);let w=v.find((e,t)=>{let o=b[t];return o&&(0,d.getHasError)({result:e,errorResetBoundary:m,throwOnError:o.throwOnError,query:h.getQueryCache().get(o.queryHash),suspense:o.suspense})});if(w?.error)throw w.error;return _(k())}],586448)},180127,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["default",0,t])},871689,e=>{"use strict";var t=e.i(180127);e.s(["ArrowLeft",()=>t.default])},455037,e=>{"use strict";var t=e.i(494144);e.s(["prism",()=>t.default])},198458,e=>{"use strict";var t=e.i(655063),o=e.i(266027),n=e.i(271645),i=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:r,fetchPage:s,serializeFilters:a,defaultSorting:l,defaultPageSize:u,enabled:c}=e,[d,p]=(0,n.useState)(l),[h,g]=(0,n.useState)({pageIndex:0,pageSize:u}),[m,b]=(0,n.useState)([]),[f,v]=(0,n.useState)(""),[_]=(0,t.useDebouncedValue)(f,{wait:i.DEBOUNCE_WAIT_MS}),k=(0,n.useMemo)(()=>{let e=d.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=_.trim();return{page:h.pageIndex+1,page_size:h.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...a(m)}},[d,h.pageIndex,h.pageSize,_,m,a]),E={queryKey:[...r,k],queryFn:({signal:e})=>s(k,e),enabled:c,placeholderData:e=>e},{data:y,isLoading:w,isPlaceholderData:x,isFetching:T,error:C,refetch:S}=(0,o.useQuery)(E),O=(0,n.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),I=(0,n.useCallback)(e=>{p(e),O()},[O]),L=(0,n.useCallback)(e=>{b(e),O()},[O]),A=(0,n.useCallback)(e=>{v(e),O()},[O]),j=(0,n.useCallback)(()=>{S()},[S]);return{rows:(0,n.useMemo)(()=>y?.data??[],[y]),rowCount:y?.meta.total_count??0,isLoading:w||x,isFetching:T,error:C,refetch:j,sorting:d,onSortingChange:I,pagination:h,onPaginationChange:g,columnFilters:m,onColumnFiltersChange:L,searchValue:f,onSearchChange:A}}])},466828,e=>{"use strict";var t=e.i(843476),o=e.i(271645),n=e.i(678784);let i=(0,e.i(475254).default)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var r=e.i(650056);let s={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};var a=e.i(488012);e.s(["default",0,({code:e,language:l})=>{let u=(0,a.useSyntaxTheme)(s),[c,d]=(0,o.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-border bg-muted overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),d(!0),setTimeout(()=>d(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground z-raised","aria-label":"Copy code",children:c?(0,t.jsx)(n.CheckIcon,{size:16}):(0,t.jsx)(i,{size:16})}),(0,t.jsx)(r.Prism,{language:l,style:u,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",background:"transparent"},codeTagProps:{style:{background:"transparent"}},showLineNumbers:!0,children:e})]})}],466828)},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let o,{apiKeySource:n,accessToken:i,apiKey:r,inputMessage:s,chatHistory:a,selectedTags:l,selectedVectorStores:u,selectedGuardrails:c,selectedPolicies:d,selectedVoice:p,endpointType:h,selectedModel:g,selectedSdk:m,proxySettings:b,customHeaders:f}=e,v="session"===n?i:r,_=window.location.origin,k=b?.LITELLM_UI_API_DOC_BASE_URL;k&&k.trim()?_=k:b?.PROXY_BASE_URL&&(_=b.PROXY_BASE_URL);let E=s||"Your prompt here",y=E.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=a.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),x={};l.length>0&&(x.tags=l),u.length>0&&(x.vector_stores=u),c.length>0&&(x.guardrails=c),d.length>0&&(x.policies=d);let T=g||"your-model-name",C=f&&Object.keys(f).length>0?`,
	default_headers=${JSON.stringify(f,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===m?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${C}
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${C}
)`;switch(h){case t.EndpointType.CHAT:{let e=Object.keys(x).length>0,t="";if(e){let e=JSON.stringify({metadata:x},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:E}];o=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${T}",
    messages=${JSON.stringify(n,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${T}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${y}"
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
`;break}case t.EndpointType.RESPONSES:{let e=Object.keys(x).length>0,t="";if(e){let e=JSON.stringify({metadata:x},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:E}];o=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${T}",
    input=${JSON.stringify(n,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${T}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${y}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.EndpointType.IMAGE:o="azure"===m?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${T}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
`;break;case t.EndpointType.IMAGE_EDITS:o="azure"===m?`
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${T}",
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
`;break;case t.EndpointType.EMBEDDINGS:o=`
response = client.embeddings.create(
	input="${s||"Your string here"}",
	model="${T}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:o=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${T}",
	file=audio_file${s?`,
	prompt="${s.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:o=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${T}",
	input="${s||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${T}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:o="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${o}`}])},902555,e=>{"use strict";var t=e.i(843476),o=e.i(746798),n=e.i(271645);let i=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))}),r=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});var s=e.i(278587),a=e.i(68155),l=e.i(360820),u=e.i(871943),c=e.i(434626);let d=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});var p=e.i(196631);function h({icon:e,onClick:o,className:n,disabled:i,dataTestId:r}){return i?(0,t.jsx)("span",{className:"inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50","data-testid":r,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})}):(0,t.jsx)("span",{className:(0,p.cx)("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5",n),onClick:o,"data-testid":r,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})})}let g={Edit:{icon:i,className:"hover:text-info"},Delete:{icon:a.TrashIcon,className:"hover:text-destructive"},Test:{icon:r,className:"hover:text-info"},Regenerate:{icon:s.RefreshIcon,className:"hover:text-success"},Reset:{icon:s.RefreshIcon,className:"hover:text-info"},Up:{icon:l.ChevronUpIcon,className:"hover:text-info"},Down:{icon:u.ChevronDownIcon,className:"hover:text-info"},Open:{icon:c.ExternalLinkIcon,className:"hover:text-success"},Copy:{icon:d,className:"hover:text-info"}};e.s(["default",0,function({onClick:e,tooltipText:n,disabled:i=!1,disabledTooltipText:r,dataTestId:s,variant:a}){let{icon:l,className:u}=g[a],c=i?r:n,d=(0,t.jsx)(h,{icon:l,onClick:e,className:u,disabled:i,dataTestId:s});return c?(0,t.jsx)(o.TooltipProvider,{children:(0,t.jsxs)(o.Tooltip,{children:[(0,t.jsx)(o.TooltipTrigger,{render:(0,t.jsx)("span",{tabIndex:i?0:void 0}),children:d}),(0,t.jsx)(o.TooltipContent,{children:c})]})}):(0,t.jsx)("span",{children:d})}],902555)},292335,e=>{"use strict";let t={NONE:"none",API_KEY:"api_key",BEARER_TOKEN:"bearer_token",TOKEN:"token",BASIC:"basic",OAUTH2:"oauth2",OAUTH2_TOKEN_EXCHANGE:"oauth2_token_exchange",OAUTH2_ID_JAG:"oauth2_id_jag",AWS_SIGV4:"aws_sigv4",TRUE_PASSTHROUGH:"true_passthrough",OAUTH_DELEGATE:"oauth_delegate"},o=[{value:t.NONE,label:"None"},{value:t.API_KEY,label:"API Key"},{value:t.BEARER_TOKEN,label:"Bearer Token"},{value:t.TOKEN,label:"Token"},{value:t.BASIC,label:"Basic Auth"},{value:t.OAUTH2,label:"OAuth"},{value:t.OAUTH2_TOKEN_EXCHANGE,label:"OAuth Token Exchange (OBO)"},{value:t.OAUTH2_ID_JAG,label:"ID-JAG (Okta Cross App Access)"},{value:t.AWS_SIGV4,label:"AWS SigV4 (Bedrock AgentCore MCPs)"},{value:t.TRUE_PASSTHROUGH,label:"True Passthrough (no LiteLLM auth)"},{value:t.OAUTH_DELEGATE,label:"OAuth Delegate (client-supplied upstream token)"}],n=e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE,i={INTERACTIVE:"interactive",M2M:"m2m"},r=e=>{let t=e.credentials??{};return JSON.stringify({url:"string"==typeof e.url?e.url:null,spec_path:"string"==typeof e.spec_path?e.spec_path:null,auth_type:e.auth_type??null,oauth_flow_type:e.oauth_flow_type??null,client_id:t.client_id??null,client_secret:t.client_secret??null,scopes:t.scopes??null,upstream_resource:t.upstream_resource??null,issuer:e.issuer??null,authorization_url:e.authorization_url??null,token_url:e.token_url??null,registration_url:e.registration_url??null})},s=["client_id","client_secret"],a=["upstream_resource","upstream_token_header"],l=["access_token","refresh_token","expires_in","scope"],u=(e,t)=>{if(!e)return;let o=Object.fromEntries(t.filter(t=>"string"==typeof e[t]&&""!==e[t]).map(t=>[t,e[t]]));return Object.keys(o).length>0?o:void 0},c="client_credentials",d={SSE:"sse",HTTP:"http",STDIO:"stdio",OPENAPI:"openapi"},p=[{value:d.HTTP,label:"Streamable HTTP (Recommended)"},{value:d.SSE,label:"Server-Sent Events (SSE)"},{value:d.STDIO,label:"Standard Input/Output (stdio)"},{value:d.OPENAPI,label:"OpenAPI Spec"}];e.s(["ADMIN_CONFIG_CREDENTIAL_KEYS",0,a,"AUTH_TYPE",0,t,"AUTH_TYPE_ITEMS",0,o,"CLEARED_ON_INVALIDATION",0,["credentials"],"MCP_OAUTH2_FLOW_INTERACTIVE",0,"authorization_code","MCP_OAUTH2_FLOW_M2M",0,c,"MCP_REACHABLE_DESCRIPTION",0,"Server responded. Authentication and tools were not checked","OAUTH_FLOW",0,i,"TRANSPORT",0,d,"TRANSPORT_ITEMS",0,p,"credentialAuthClass",0,e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE?"client_forwarded":e??null,"gatewayMintsClientFor",0,e=>e.auth_type===t.TRUE_PASSTHROUGH||e.auth_type===t.OAUTH_DELEGATE&&!e.dcr_bridge,"getMcpOAuthMode",0,function(e){return e.auth_type===t.OAUTH2_TOKEN_EXCHANGE?"token_exchange":e.auth_type!==t.OAUTH2?null:e.oauth2_flow===c?"m2m":e.delegate_auth_to_upstream?"passthrough":"authorization_code"},"getOAuthAuthorizationIdentity",0,r,"handleAuth",0,e=>null==e?t.NONE:e,"handleTransport",0,(e,t)=>null==e?d.SSE:t&&e!==d.STDIO?d.OPENAPI:e,"isClientForwardedTokenMode",0,n,"isHeldOAuthTokenStale",0,(e,t)=>void 0!==t&&r(e)!==t,"isUnsupportedOnGatewayConnect",0,e=>n(e)||e===t.OAUTH2_TOKEN_EXCHANGE,"oauth2FlowToFormValue",0,function(e){return e===c?i.M2M:e?i.INTERACTIVE:void 0},"preservedAdminCredentials",0,e=>u(e,[...s,...a]),"preservedDeclaredAppCredentials",0,e=>u(e,s),"withoutMintedTokenCredentials",0,e=>{if(!e)return;let t=Object.fromEntries(Object.entries(e).filter(([e])=>!l.includes(e)));return Object.keys(t).length>0?t:void 0}])},845150,e=>{"use strict";var t=e.i(843476),o=e.i(271645),n=e.i(131792);let i=(e,t)=>{let o=t.trim().toLowerCase();return!o||e.label.toLowerCase().includes(o)||e.value.toLowerCase().includes(o)||(e.description?.toLowerCase().includes(o)??!1)};e.s(["MultiSelect",0,function({id:e,options:r,value:s=[],onValueChange:a,placeholder:l="Select options",emptyText:u="No options found",disabled:c=!1,loading:d=!1,allowCustomValues:p=!1,className:h}){let g=(0,n.useComboboxAnchor)(),[m,b]=(0,o.useState)(""),f=r.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),v=s.filter(e=>"string"==typeof e&&e.length>0).map(e=>f.find(t=>t.value===e)??{label:e,value:e}),_=m.trim(),k=f.some(e=>e.value.toLowerCase()===_.toLowerCase()),E=p&&_&&!k?[...f,{label:`Create "${_}"`,value:_}]:f;return(0,t.jsxs)(n.Combobox,{multiple:!0,items:E,value:v,onValueChange:e=>{a(Array.from(new Set(p?e.flatMap(e=>s.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),b("")},inputValue:m,onInputValueChange:b,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:i,disabled:c||d,children:[(0,t.jsx)(n.ComboboxChips,{render:(0,t.jsx)("div",{ref:g}),className:`min-h-8 py-1 text-sm ${h??""}`,children:(0,t.jsx)(n.ComboboxValue,{children:o=>(0,t.jsxs)(t.Fragment,{children:[o.map(e=>(0,t.jsx)(n.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(n.ComboboxChipsInput,{id:e,placeholder:d?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),o.length>0&&!c&&!d&&(0,t.jsx)(n.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(n.ComboboxContent,{anchor:g,children:[(0,t.jsx)(n.ComboboxEmpty,{children:u}),(0,t.jsx)(n.ComboboxList,{children:e=>(0,t.jsx)(n.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},741466,e=>{"use strict";e.s(["DEBOUNCE_WAIT_MS",0,300])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function o(e,o){let n=t(e);if(""===n)return!0;let i=o.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!i.some(e=>e.includes(n))||n.split(/\s+/).every(e=>i.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,n){return e.filter(e=>o(t,n(e)))},"matchesSearchTerm",0,o,"rankBySearchRelevance",0,function(e,o,n){let i=t(o);if(""===i)return[...e];let r=e=>{let t=n(e).toLowerCase();return 1e3*(t===i)+100*!!t.startsWith(i)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}]);