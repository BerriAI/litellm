(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,434626,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,i],434626)},655063,e=>{"use strict";var t=e.i(540626),i=e.i(271645);e.s(["useDebouncedValue",0,function(e,s,n){let[r,a,o]=function(e,s,n){let[r,a]=(0,i.useState)(e),o=(0,t.useDebouncer)(a,s,n);return[r,o.maybeExecute,o]}(e,s,n);return(0,i.useEffect)(()=>{a(e)},[e,a]),[r,o]}],655063)},540626,e=>{"use strict";let t;var i=e.i(271645);let s=(0,i.createContext)(null);function n(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[i,s]of e)if(!t.has(i)||!Object.is(s,t.get(i)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let i of e)if(!t.has(i))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let i=r(e);if(i.length!==r(t).length)return!1;for(let s=0;s<i.length;s++)if(!Object.prototype.hasOwnProperty.call(t,i[s])||!Object.is(e[i[s]],t[i[s]]))return!1;return!0}function r(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var a=e.i(430224);function o(e,t){return e===t}function l(e,t=e=>e,s){let n=s?.compare??o,r=(0,i.useCallback)(t=>{let{unsubscribe:i}=e.subscribe(t);return i},[e]),u=(0,i.useCallback)(()=>e.get(),[e]);return(0,a.useSyncExternalStoreWithSelector)(r,u,u,t,n)}function u(e,...t){return"function"==typeof e?e(...t):e}var d=class{#e=!0;#t;#i;#s;#n;#r;#a;#o;#l=0;#u=5;#d=!1;#c=!1;#p=null;#m=()=>{this.debugLog("Connected to event bus"),this.#r=!0,this.#d=!1,this.debugLog("Emitting queued events",this.#n),this.#n.forEach(e=>this.emitEventToBus(e)),this.#n=[],this.stopConnectLoop(),this.#i().removeEventListener("tanstack-connect-success",this.#m)};#g=()=>{if(this.#l<this.#u){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#i().removeEventListener("tanstack-connect",this.#g),this.#c=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#h=()=>{this.#d||(this.#d=!0,this.#i().addEventListener("tanstack-connect-success",this.#m),this.#g())};constructor({pluginId:e,debug:t=!1,enabled:i=!0,reconnectEveryMs:s=300}){this.#t=e,this.#e=i,this.#i=this.getGlobalTarget,this.#s=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#n=[],this.#r=!1,this.#c=!1,this.#a=null,this.#o=s}startConnectLoop(){null!==this.#a||this.#r||(this.debugLog(`Starting connect loop (every ${this.#o}ms)`),this.#a=setInterval(this.#g,this.#o))}stopConnectLoop(){this.#d=!1,null!==this.#a&&(clearInterval(this.#a),this.#a=null,this.#n=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#s&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let i=new Event(e,{detail:t});this.#i().dispatchEvent(i)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#i().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(i){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#p&&(this.debugLog("Emitting event to internal event target",e,t),this.#p.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#c)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#r){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#n.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#d&&(this.#h(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,i){let s=i?.withEventTarget??!1,n=`${this.#t}:${e}`;if(s&&(this.#p||(this.#p=new EventTarget),this.#p.addEventListener(n,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",n),()=>{};let r=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#i().addEventListener(n,r),this.debugLog("Registered event to bus",n),()=>{s&&this.#p?.removeEventListener(n,r),this.#i().removeEventListener(n,r)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let i=t.detail;this.#t&&i.pluginId!==this.#t||e(i)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}};let c=new Map;function p(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let m=new class extends d{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function g(e,t,i){let s="object"==typeof e,n=s?e:void 0;return{next:(s?e.next:e)?.bind(n),error:(s?e.error:t)?.bind(n),complete:(s?e.complete:i)?.bind(n)}}let h=[],f=0,{link:b,unlink:x,propagate:v,checkDirty:_,shallowPropagate:y}=function({update:e,notify:t,unwatched:i}){return{link:function(e,t,i){let s=t.depsTail;if(void 0!==s&&s.dep===e)return;let n=void 0!==s?s.nextDep:t.deps;if(void 0!==n&&n.dep===e){n.version=i,t.depsTail=n;return}let r=e.subsTail;if(void 0!==r&&r.version===i&&r.sub===t)return;let a=t.depsTail=e.subsTail={version:i,dep:e,sub:t,prevDep:s,nextDep:n,prevSub:r,nextSub:void 0};void 0!==n&&(n.prevDep=a),void 0!==s?s.nextDep=a:t.deps=a,void 0!==r?r.nextSub=a:e.subs=a},unlink:function(e,t=e.sub){let s=e.dep,n=e.prevDep,r=e.nextDep,a=e.nextSub,o=e.prevSub;return void 0!==r?r.prevDep=n:t.depsTail=n,void 0!==n?n.nextDep=r:t.deps=r,void 0!==a?a.prevSub=o:s.subsTail=o,void 0!==o?o.nextSub=a:void 0===(s.subs=a)&&i(s),r},propagate:function(e){let i,s=e.nextSub;e:for(;;){let n=e.sub,r=n.flags;if(60&r?12&r?4&r?!(48&r)&&function(e,t){let i=t.depsTail;for(;void 0!==i;){if(i===e)return!0;i=i.prevDep}return!1}(e,n)?(n.flags=40|r,r&=1):r=0:n.flags=-9&r|32:r=0:n.flags=32|r,2&r&&t(n),1&r){let t=n.subs;if(void 0!==t){let n=(e=t).nextSub;void 0!==n&&(i={value:s,prev:i},s=n);continue}}if(void 0!==(e=s)){s=e.nextSub;continue}for(;void 0!==i;)if(e=i.value,i=i.prev,void 0!==e){s=e.nextSub;continue e}break}},checkDirty:function(t,i){let n,r=0,a=!1;e:for(;;){let o=t.dep,l=o.flags;if(16&i.flags)a=!0;else if((17&l)==17){if(e(o)){let e=o.subs;void 0!==e.nextSub&&s(e),a=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(n={value:t,prev:n}),t=o.deps,i=o,++r;continue}if(!a){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;r--;){let r=i.subs,o=void 0!==r.nextSub;if(o?(t=n.value,n=n.prev):t=r,a){if(e(i)){o&&s(r),i=t.sub;continue}a=!1}else i.flags&=-33;i=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return a}},shallowPropagate:s};function s(e){do{let i=e.sub,s=i.flags;(48&s)==32&&(i.flags=16|s,(6&s)==2&&t(i))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){h[j++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,E(e))}}),w=0,j=0;function E(e){let t=e.depsTail,i=void 0!==t?t.nextDep:e.deps;for(;void 0!==i;)i=x(i,e)}var k=class{constructor(e,i){this.atom=function(e){let i="function"==typeof e,s={_snapshot:i?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!i,get:()=>(void 0!==t&&b(s,t,f),s._snapshot),subscribe(e){var i;let n,r,a=g(e),o={current:!1},l=(i=()=>{s.get(),o.current?a.next?.(s._snapshot):o.current=!0},n=()=>{let e=t;t=r,++f,r.depsTail=void 0,r.flags=6;try{return i()}finally{t=e,r.flags&=-5,E(r)}},r={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&_(this.deps,this)?n():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,E(this)}},n(),r);return{unsubscribe:()=>{l.stop()}}},_update(n){let r=t,a=(void 0)??Object.is;if(i)t=s,++f,s.depsTail=void 0;else if(void 0===n)return!1;i&&(s.flags=5);try{let t=s._snapshot,r="function"==typeof n?n(t):void 0===n&&i?e(t):n;if(void 0===t||!a(t,r))return s._snapshot=r,!0;return!1}finally{t=r,i&&(s.flags&=-5),E(s)}}};return i?(s.flags=17,s.get=function(){let e=s.flags;if(16&e||32&e&&_(s.deps,s)){if(s._update()){let e=s.subs;void 0!==e&&y(e)}}else 32&e&&(s.flags=-33&e);return void 0!==t&&b(s,t,f),s._snapshot}):s.set=function(e){if(s._update(e)){let e=s.subs;if(void 0!==e&&(v(e),y(e),1)){for(;w<j;){let e=h[w];h[w++]=void 0,e.notify()}w=0,j=0}}},s}(e),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),i&&(this.actions=i(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(g(e))}};function C(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let S={enabled:!0,leading:!1,trailing:!0,wait:0};var N=class{#f;constructor(e,t){this.fn=e,this.store=new k(C()),this.setOptions=e=>{this.options={...this.options,...e},this.#b()||this.cancel()},this.#x=e=>{this.store.setState(t=>{let i={...t,...e},{isPending:s}=i;return{...i,status:this.#b()?s?"pending":"idle":"disabled"}}),((e,t)=>{let i=t.key;if(i){var s,n;c.set(i,t),m.emit(e,{key:(s={...t,key:i}).key,store:{state:p("function"==typeof(n=s.store).get?n.get():n.state)},options:p(s.options)})}})("Debouncer",this)},this.#b=()=>!!u(this.options.enabled,this),this.#v=()=>u(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#b())return;this.#x({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#x({canLeadingExecute:!1}),t=!0,this.#_(...e)),this.options.trailing&&this.#x({isPending:!0,lastArgs:e}),this.#f&&clearTimeout(this.#f),this.#f=setTimeout(()=>{this.#x({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#_(...e)},this.#v())},this.#_=(...e)=>{this.#b()&&(this.fn(...e),this.#x({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#y(),this.#_(...this.store.state.lastArgs))},this.#y=()=>{this.#f&&(clearTimeout(this.#f),this.#f=void 0)},this.cancel=()=>{this.#y(),this.#x({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#x(C())},this.key=t.key,this.options={...S,...t},this.#x(this.options.initialState??{}),this.key&&m.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#x(e.payload.store.state),this.setOptions(e.payload.options))})}#x;#b;#v;#_;#y};e.s(["useDebouncer",0,function(e,t,r=()=>({})){let a={...((0,i.useContext)(s)?.defaultOptions??{}).debouncer,...t},[o]=(0,i.useState)(()=>{let t=new N(e,a);return t.Subscribe=function(e){let i=l(t.store,e.selector,{compare:n});return"function"==typeof e.children?e.children(i):e.children},t});o.fn=e,o.setOptions(a),(0,i.useEffect)(()=>()=>{a.onUnmount?a.onUnmount(o):o.cancel()},[]);let u=l(o.store,r,{compare:n});return(0,i.useMemo)(()=>({...o,state:u}),[o,u])}],540626)},586448,e=>{"use strict";var t=e.i(271645),i=e.i(540143),s=e.i(869230),n=e.i(915823),r=e.i(619273);function a(e,t){let i=new Set(t);return e.filter(e=>!i.has(e))}var o=class extends n.Subscribable{#w;#j;#E;#k;#C;#S;#N;#I;#T;#$=[];constructor(e,t,i){super(),this.#w=e,this.#k=i,this.#E=[],this.#C=[],this.#j=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#C.forEach(e=>{e.subscribe(t=>{this.#L(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#C.forEach(e=>{e.destroy()})}setQueries(e,t){this.#E=e,this.#k=t,i.notifyManager.batch(()=>{let e=this.#C,t=this.#O(this.#E);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let i=t.map(e=>e.observer),s=i.map(e=>e.getCurrentResult()),n=e.length!==i.length,o=i.some((t,i)=>t!==e[i]),l=n||o,u=!!l||s.some((e,t)=>{let i=this.#j[t];return!i||!(0,r.shallowEqualObjects)(e,i)});(l||u)&&(l&&(this.#$=t,this.#C=i),this.#j=s,this.hasListeners()&&(l&&(a(e,i).forEach(e=>{e.destroy()}),a(i,e).forEach(e=>{e.subscribe(t=>{this.#L(e,t)})})),this.#M()))})}getCurrentResult(){return this.#j}getQueries(){return this.#C.map(e=>e.getCurrentQuery())}getObservers(){return this.#C}getOptimisticResult(e,t){let i=this.#O(e),s=i.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),n=i.map(e=>e.defaultedQueryOptions.queryHash);return[s,e=>this.#R(e??s,t,n),()=>this.#z(s,i)]}#z(e,t){return t.map((i,s)=>{let n=e[s];return i.defaultedQueryOptions.notifyOnChangeProps?n:i.observer.trackResult(n,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#R(e,t,i){if(t){let s=this.#T,n=void 0!==i&&void 0!==s&&(s.length!==i.length||i.some((e,t)=>e!==s[t]));return(!this.#S||this.#j!==this.#I||n||t!==this.#N)&&(this.#N=t,this.#I=this.#j,void 0!==i&&(this.#T=i),this.#S=(0,r.replaceEqualDeep)(this.#S,t(e))),this.#S}return e}#A(){return this.#k?.combine!==void 0&&this.#C.some((e,t)=>e.options.suspense&&this.#j[t]?.data===void 0)}#O(e){let t=new Map;this.#C.forEach(e=>{let i=e.options.queryHash;if(!i)return;let s=t.get(i);s?s.push(e):t.set(i,[e])});let i=[];return e.forEach(e=>{let n=this.#w.defaultQueryOptions(e),r=t.get(n.queryHash)?.shift()??new s.QueryObserver(this.#w,n);i.push({defaultedQueryOptions:n,observer:r})}),i}#L(e,t){let i=this.#C.indexOf(e);if(-1!==i){var s;let e;this.#j=(s=this.#j,(e=s.slice(0))[i]=t,e),this.#M()}}#M(){if(this.hasListeners()){let e=this.#z(this.#j,this.#$),t=this.#A(),s=this.#S,n=t?s:this.#R(e,this.#k?.combine);(t||s!==n)&&i.notifyManager.batch(()=>{this.listeners.forEach(e=>{e(this.#j)})})}}},l=e.i(912598),u=e.i(381384),d=e.i(673664),c=e.i(427001),p=e.i(254440);e.s(["useQueries",0,function({queries:e,...n},a){let m=(0,l.useQueryClient)(a),g=(0,u.useIsRestoring)(),h=(0,d.useQueryErrorResetBoundary)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,m,g]);f.forEach(e=>{(0,p.ensureSuspenseTimers)(e);let t=m.getQueryCache().get(e.queryHash);(0,c.ensurePreventErrorBoundaryRetry)(e,h,t)}),(0,c.useClearResetErrorBoundary)(h);let[b]=t.useState(()=>new o(m,f,n)),[x,v,_]=b.getOptimisticResult(f,n.combine),y=!g&&!1!==n.subscribed;t.useSyncExternalStore(t.useCallback(e=>y?b.subscribe(i.notifyManager.batchCalls(e)):r.noop,[b,y]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(f,n)},[f,n,b]);let w=x.some((e,t)=>(0,p.shouldSuspend)(f[t],e))?x.flatMap((e,t)=>{let i=f[t];if(i&&(0,p.shouldSuspend)(i,e)){let e=new s.QueryObserver(m,i);return(0,p.fetchOptimistic)(i,e,h)}return[]}):[];if(w.length>0)throw Promise.all(w);let j=x.find((e,t)=>{let i=f[t];return i&&(0,c.getHasError)({result:e,errorResetBoundary:h,throwOnError:i.throwOnError,query:m.getQueryCache().get(i.queryHash),suspense:i.suspense})});if(j?.error)throw j.error;return v(_())}],586448)},180127,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["default",0,t])},871689,e=>{"use strict";var t=e.i(180127);e.s(["ArrowLeft",()=>t.default])},233565,e=>{"use strict";var t=e.i(246349);e.s(["ChevronRightIcon",()=>t.default])},373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t])},541071,e=>{"use strict";var t=e.i(373488);e.s(["MoreHorizontal",()=>t.default])},546467,e=>{"use strict";let t=(0,e.i(475254).default)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["default",0,t])},778917,e=>{"use strict";var t=e.i(546467);e.s(["ExternalLink",()=>t.default])},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},198458,e=>{"use strict";var t=e.i(655063),i=e.i(266027),s=e.i(271645),n=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:r,fetchPage:a,serializeFilters:o,defaultSorting:l,defaultPageSize:u,enabled:d}=e,[c,p]=(0,s.useState)(l),[m,g]=(0,s.useState)({pageIndex:0,pageSize:u}),[h,f]=(0,s.useState)([]),[b,x]=(0,s.useState)(""),[v]=(0,t.useDebouncedValue)(b,{wait:n.DEBOUNCE_WAIT_MS}),_=(0,s.useMemo)(()=>{let e=c.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=v.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(h)}},[c,m.pageIndex,m.pageSize,v,h,o]),y={queryKey:[...r,_],queryFn:({signal:e})=>a(_,e),enabled:d,placeholderData:e=>e},{data:w,isLoading:j,isPlaceholderData:E,isFetching:k,error:C,refetch:S}=(0,i.useQuery)(y),N=(0,s.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),I=(0,s.useCallback)(e=>{p(e),N()},[N]),T=(0,s.useCallback)(e=>{f(e),N()},[N]),$=(0,s.useCallback)(e=>{x(e),N()},[N]),L=(0,s.useCallback)(()=>{S()},[S]);return{rows:(0,s.useMemo)(()=>w?.data??[],[w]),rowCount:w?.meta.total_count??0,isLoading:j||E,isFetching:k,error:C,refetch:L,sorting:c,onSortingChange:I,pagination:m,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:T,searchValue:b,onSearchChange:$}}])},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let i,{apiKeySource:s,accessToken:n,apiKey:r,inputMessage:a,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:d,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,x="session"===s?n:r,v=window.location.origin,_=f?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?v=_:f?.PROXY_BASE_URL&&(v=f.PROXY_BASE_URL);let y=a||"Your prompt here",w=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),u.length>0&&(E.vector_stores=u),d.length>0&&(E.guardrails=d),c.length>0&&(E.policies=c);let k=g||"your-model-name",C=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"${C}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"${C}
)`;switch(m){case t.EndpointType.CHAT:{let e=Object.keys(E).length>0,t="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=j.length>0?j:[{role:"user",content:y}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(s,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.EndpointType.RESPONSES:{let e=Object.keys(E).length>0,t="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=j.length>0?j:[{role:"user",content:y}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(s,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.EndpointType.IMAGE:i="azure"===h?`
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
`;break;case t.EndpointType.IMAGE_EDITS:i="azure"===h?`
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
`;break;case t.EndpointType.EMBEDDINGS:i=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${a||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${i}`}])},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),s=e.i(871689),n=e.i(643531),r=e.i(174886),a=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,u=e=>e.trim().replace(/\/+$/,""),d=/\.(md|markdown|txt|json|ya?ml|toml)$/i,c=/\.zip$/i,p=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,g=/^[A-Za-z0-9-]+$/,h=/^[A-Za-z0-9._-]+$/,f=/^https?:\/\//i,b="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,v=e=>e.pathname.split("/").filter(e=>""!==e),_=e=>{try{return new URL(e)}catch{return null}},y=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),w=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},j=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),E=(e,t,i,s)=>{let n=u(s??"");return""!==n?l.test(n)?{parsed:{source:"git-subdir",url:t,path:n},label:`${e} subdir — ${t} @ ${n}`,suggestedName:j(w(n))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:j(i)}},k=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),C=e=>`/plugin install ${e.name}@litellm`,S=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",N=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&f.test(e.url)?e.url:null;e.s(["buildMarketplaceSettingsSnippet",0,k,"formatInstallCommand",0,C,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,S,"getSourceLink",0,N,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSha256",0,e=>""===e.trim()||p.test(e.trim()),"isValidSubPath",0,e=>{let t=u(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let i=((e,t)=>{let i=e.trim(),s=x.exec(i),n=s?`${b}${s[1]}@${s[2]}/${s[3]}`:i;if(!n.toLowerCase().startsWith(b))return null;let r=_(n);if(!r||""===r.username||""!==r.password||!y(r))return null;let a=n.indexOf("/",b.length);return -1===a||r.pathname!==n.slice(a)||v(r).length<2?null:E("SSH",i,w(r.pathname).replace(/\.git$/i,""),t)})(e,t);if(i)return i;let s=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=_(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&y(i)?i:null})(e);if(!s)return null;if(c.test(s.pathname))return{parsed:{source:"archive",url:s.href},label:`Zip archive — ${s.host}${s.pathname}`,suggestedName:j(w(s.pathname).replace(c,""))};if("github.com"===s.hostname.replace(/^www\./,""))return((e,t)=>{let i=v(e);if(i.length<2)return null;let s=i[0],n=i[1].replace(/\.git$/,"");if(!g.test(s)||!h.test(n))return null;let r=`${s}/${n}`,a=`https://github.com/${r}`,o={parsed:{source:"github",repo:r},label:`GitHub repo — ${r}`,suggestedName:j(n)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=w(e.join("/")),s=d.test(t)?e.slice(0,-1):e;if(0===s.length)return o;let n=u(s.join("/"));return l.test(n)?{parsed:{source:"git-subdir",url:a,path:n},label:`GitHub subdir — ${r} @ ${n}`,suggestedName:j(w(n))}:null}if(2!==i.length)return null;let c=u(t??"");return""!==c?l.test(c)?{parsed:{source:"git-subdir",url:a,path:c},label:`GitHub subdir — ${r} @ ${c}`,suggestedName:j(w(c))}:null:o})(s,t);if(v(s).length<2)return null;let n=w(s.pathname).replace(/\.git$/,"");return E("Git",`${s.protocol}//${s.host}${s.pathname.replace(/\/+$/,"")}`,n,t)},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let I=({source:e})=>{let i=N(e),s=i&&"git-subdir"===e.source&&e.path?`${i}/tree/main/${e.path}`:i;return s?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:s,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[s.replace("https://",""),(0,t.jsx)(a.Link2,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:S(e)})]}):null};e.s(["default",0,({skill:e,onBack:a})=>{let[l,u]=(0,i.useState)("overview"),[d,c]=(0,i.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),c(t),setTimeout(()=>c(null),2e3)},m=C(e),g=k(window.location.origin),h=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:a,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(s.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>u(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:h.map((e,i)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(I,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(m,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===d?"text-success":"text-info"),children:["install"===d?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"install"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>u("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;p(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===d?"text-success":"text-info"),children:["marketplace-cmd"===d?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"marketplace-cmd"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>p(g,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===d?"text-success":"text-info"),children:["settings"===d?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"settings"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:g})]})]})]})}],652272)},845150,e=>{"use strict";var t=e.i(843476),i=e.i(271645),s=e.i(131792);let n=(e,t)=>{let i=t.trim().toLowerCase();return!i||e.label.toLowerCase().includes(i)||e.value.toLowerCase().includes(i)||(e.description?.toLowerCase().includes(i)??!1)};e.s(["MultiSelect",0,function({id:e,options:r,value:a=[],onValueChange:o,placeholder:l="Select options",emptyText:u="No options found",disabled:d=!1,loading:c=!1,allowCustomValues:p=!1,className:m}){let g=(0,s.useComboboxAnchor)(),[h,f]=(0,i.useState)(""),b=r.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=a.filter(e=>"string"==typeof e&&e.length>0).map(e=>b.find(t=>t.value===e)??{label:e,value:e}),v=h.trim(),_=b.some(e=>e.value.toLowerCase()===v.toLowerCase()),y=p&&v&&!_?[...b,{label:`Create "${v}"`,value:v}]:b;return(0,t.jsxs)(s.Combobox,{multiple:!0,items:y,value:x,onValueChange:e=>{o(Array.from(new Set(p?e.flatMap(e=>a.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),f("")},inputValue:h,onInputValueChange:f,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:n,disabled:d||c,children:[(0,t.jsx)(s.ComboboxChips,{render:(0,t.jsx)("div",{ref:g}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(s.ComboboxValue,{children:i=>(0,t.jsxs)(t.Fragment,{children:[i.map(e=>(0,t.jsx)(s.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(s.ComboboxChipsInput,{id:e,placeholder:c?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),i.length>0&&!d&&!c&&(0,t.jsx)(s.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(s.ComboboxContent,{anchor:g,children:[(0,t.jsx)(s.ComboboxEmpty,{children:u}),(0,t.jsx)(s.ComboboxList,{children:e=>(0,t.jsx)(s.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},755146,e=>{"use strict";var t=e.i(843476),i=e.i(451512),s=e.i(196631);e.i(233565);var n=e.i(678784);e.s(["DropdownMenu",0,function({...e}){return(0,t.jsx)(i.Menu.Root,{"data-slot":"dropdown-menu",...e})},"DropdownMenuCheckboxItem",0,function({className:e,children:r,checked:a,inset:o,...l}){return(0,t.jsxs)(i.Menu.CheckboxItem,{"data-slot":"dropdown-menu-checkbox-item","data-inset":o,className:(0,s.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),checked:a,...l,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-checkbox-item-indicator",children:(0,t.jsx)(i.Menu.CheckboxItemIndicator,{children:(0,t.jsx)(n.CheckIcon,{})})}),r]})},"DropdownMenuContent",0,function({align:e="start",alignOffset:n=0,side:r="bottom",sideOffset:a=4,className:o,...l}){return(0,t.jsx)(i.Menu.Portal,{children:(0,t.jsx)(i.Menu.Positioner,{className:"isolate z-popup outline-none",align:e,alignOffset:n,side:r,sideOffset:a,children:(0,t.jsx)(i.Menu.Popup,{"data-slot":"dropdown-menu-content",className:(0,s.cn)("z-popup max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-md bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95",o),...l})})})},"DropdownMenuItem",0,function({className:e,inset:n,variant:r="default",...a}){return(0,t.jsx)(i.Menu.Item,{"data-slot":"dropdown-menu-item","data-inset":n,"data-variant":r,className:(0,s.cn)("group/dropdown-menu-item relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground not-data-[variant=destructive]:focus:**:text-accent-foreground data-inset:pl-8 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[variant=destructive]:*:[svg]:text-destructive",e),...a})},"DropdownMenuRadioGroup",0,function({...e}){return(0,t.jsx)(i.Menu.RadioGroup,{"data-slot":"dropdown-menu-radio-group",...e})},"DropdownMenuRadioItem",0,function({className:e,children:r,inset:a,...o}){return(0,t.jsxs)(i.Menu.RadioItem,{"data-slot":"dropdown-menu-radio-item","data-inset":a,className:(0,s.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),...o,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-radio-item-indicator",children:(0,t.jsx)(i.Menu.RadioItemIndicator,{children:(0,t.jsx)(n.CheckIcon,{})})}),r]})},"DropdownMenuSeparator",0,function({className:e,...n}){return(0,t.jsx)(i.Menu.Separator,{"data-slot":"dropdown-menu-separator",className:(0,s.cn)("-mx-1 my-1 h-px bg-border",e),...n})},"DropdownMenuTrigger",0,function({...e}){return(0,t.jsx)(i.Menu.Trigger,{"data-slot":"dropdown-menu-trigger",...e})}])},283713,e=>{"use strict";var t=e.i(271645),i=e.i(602869),s=e.i(612256);let n="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,s.useUIConfig)(),r=e?.is_control_plane??!1,a=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(n));(0,t.useEffect)(()=>{if(!o||0===a.length)return;let e=a.find(e=>e.worker_id===o);e&&(0,i.switchToWorkerUrl)(e.url)},[o,a]);let u=a.find(e=>e.worker_id===o)??null,d=(0,t.useCallback)(e=>{let t=a.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(n,e),(0,i.switchToWorkerUrl)(t.url))},[a]);return{isControlPlane:r,workers:a,selectedWorkerId:o,selectedWorker:u,selectWorker:d,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(n),(0,i.switchToWorkerUrl)(null)},[])}}])},741466,e=>{"use strict";e.s(["DEBOUNCE_WAIT_MS",0,300])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function i(e,i){let s=t(e);if(""===s)return!0;let n=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!n.some(e=>e.includes(s))||s.split(/\s+/).every(e=>n.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,s){return e.filter(e=>i(t,s(e)))},"matchesSearchTerm",0,i,"rankBySearchRelevance",0,function(e,i,s){let n=t(i);if(""===n)return[...e];let r=e=>{let t=s(e).toLowerCase();return 1e3*(t===n)+100*!!t.startsWith(n)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}]);