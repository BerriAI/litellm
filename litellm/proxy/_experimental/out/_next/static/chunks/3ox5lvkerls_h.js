(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,434626,e=>{var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["f",0,i],434626)},655063,e=>{var t=e.i(540626),i=e.i(271645);e.s(["f",0,function(e,s,n){let[r,a,o]=function(e,s,n){let[r,a]=(0,i.useState)(e),o=(0,t.f)(a,s,n);return[r,o.maybeExecute,o]}(e,s,n);return(0,i.useEffect)(()=>{a(e)},[e,a]),[r,o]}],655063)},540626,e=>{let t;var i=e.i(271645);let s=(0,i.createContext)(null);function n(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[i,s]of e)if(!t.has(i)||!Object.is(s,t.get(i)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let i of e)if(!t.has(i))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let i=r(e);if(i.length!==r(t).length)return!1;for(let s=0;s<i.length;s++)if(!Object.prototype.hasOwnProperty.call(t,i[s])||!Object.is(e[i[s]],t[i[s]]))return!1;return!0}function r(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var a=e.i(430224);function o(e,t){return e===t}function l(e,t=e=>e,s){let n=s?.compare??o,r=(0,i.useCallback)(t=>{let{unsubscribe:i}=e.subscribe(t);return i},[e]),u=(0,i.useCallback)(()=>e.get(),[e]);return(0,a.useSyncExternalStoreWithSelector)(r,u,u,t,n)}function u(e,...t){return"function"==typeof e?e(...t):e}var d=class{#e=!0;#t;#i;#s;#n;#r;#a;#o;#l=0;#u=5;#d=!1;#p=!1;#c=null;#m=()=>{this.debugLog("Connected to event bus"),this.#r=!0,this.#d=!1,this.debugLog("Emitting queued events",this.#n),this.#n.forEach(e=>this.emitEventToBus(e)),this.#n=[],this.stopConnectLoop(),this.#i().removeEventListener("tanstack-connect-success",this.#m)};#g=()=>{if(this.#l<this.#u){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#i().removeEventListener("tanstack-connect",this.#g),this.#p=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#h=()=>{this.#d||(this.#d=!0,this.#i().addEventListener("tanstack-connect-success",this.#m),this.#g())};constructor({pluginId:e,debug:t=!1,enabled:i=!0,reconnectEveryMs:s=300}){this.#t=e,this.#e=i,this.#i=this.getGlobalTarget,this.#s=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#n=[],this.#r=!1,this.#p=!1,this.#a=null,this.#o=s}startConnectLoop(){null!==this.#a||this.#r||(this.debugLog(`Starting connect loop (every ${this.#o}ms)`),this.#a=setInterval(this.#g,this.#o))}stopConnectLoop(){this.#d=!1,null!==this.#a&&(clearInterval(this.#a),this.#a=null,this.#n=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#s&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let i=new Event(e,{detail:t});this.#i().dispatchEvent(i)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#i().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(i){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#c&&(this.debugLog("Emitting event to internal event target",e,t),this.#c.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#p)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#r){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#n.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#d&&(this.#h(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,i){let s=i?.withEventTarget??!1,n=`${this.#t}:${e}`;if(s&&(this.#c||(this.#c=new EventTarget),this.#c.addEventListener(n,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",n),()=>{};let r=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#i().addEventListener(n,r),this.debugLog("Registered event to bus",n),()=>{s&&this.#c?.removeEventListener(n,r),this.#i().removeEventListener(n,r)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let i=t.detail;this.#t&&i.pluginId!==this.#t||e(i)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}};let p=new Map;function c(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let m=new class extends d{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function g(e,t,i){let s="object"==typeof e,n=s?e:void 0;return{next:(s?e.next:e)?.bind(n),error:(s?e.error:t)?.bind(n),complete:(s?e.complete:i)?.bind(n)}}let h=[],f=0,{link:b,unlink:x,propagate:v,checkDirty:_,shallowPropagate:y}=function({update:e,notify:t,unwatched:i}){return{link:function(e,t,i){let s=t.depsTail;if(void 0!==s&&s.dep===e)return;let n=void 0!==s?s.nextDep:t.deps;if(void 0!==n&&n.dep===e){n.version=i,t.depsTail=n;return}let r=e.subsTail;if(void 0!==r&&r.version===i&&r.sub===t)return;let a=t.depsTail=e.subsTail={version:i,dep:e,sub:t,prevDep:s,nextDep:n,prevSub:r,nextSub:void 0};void 0!==n&&(n.prevDep=a),void 0!==s?s.nextDep=a:t.deps=a,void 0!==r?r.nextSub=a:e.subs=a},unlink:function(e,t=e.sub){let s=e.dep,n=e.prevDep,r=e.nextDep,a=e.nextSub,o=e.prevSub;return void 0!==r?r.prevDep=n:t.depsTail=n,void 0!==n?n.nextDep=r:t.deps=r,void 0!==a?a.prevSub=o:s.subsTail=o,void 0!==o?o.nextSub=a:void 0===(s.subs=a)&&i(s),r},propagate:function(e){let i,s=e.nextSub;e:for(;;){let n=e.sub,r=n.flags;if(60&r?12&r?4&r?!(48&r)&&function(e,t){let i=t.depsTail;for(;void 0!==i;){if(i===e)return!0;i=i.prevDep}return!1}(e,n)?(n.flags=40|r,r&=1):r=0:n.flags=-9&r|32:r=0:n.flags=32|r,2&r&&t(n),1&r){let t=n.subs;if(void 0!==t){let n=(e=t).nextSub;void 0!==n&&(i={value:s,prev:i},s=n);continue}}if(void 0!==(e=s)){s=e.nextSub;continue}for(;void 0!==i;)if(e=i.value,i=i.prev,void 0!==e){s=e.nextSub;continue e}break}},checkDirty:function(t,i){let n,r=0,a=!1;e:for(;;){let o=t.dep,l=o.flags;if(16&i.flags)a=!0;else if((17&l)==17){if(e(o)){let e=o.subs;void 0!==e.nextSub&&s(e),a=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(n={value:t,prev:n}),t=o.deps,i=o,++r;continue}if(!a){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;r--;){let r=i.subs,o=void 0!==r.nextSub;if(o?(t=n.value,n=n.prev):t=r,a){if(e(i)){o&&s(r),i=t.sub;continue}a=!1}else i.flags&=-33;i=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return a}},shallowPropagate:s};function s(e){do{let i=e.sub,s=i.flags;(48&s)==32&&(i.flags=16|s,(6&s)==2&&t(i))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){h[w++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,E(e))}}),j=0,w=0;function E(e){let t=e.depsTail,i=void 0!==t?t.nextDep:e.deps;for(;void 0!==i;)i=x(i,e)}var k=class{constructor(e,i){let s,n;this.atom=(n={_snapshot:(s="function"==typeof e)?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!s,get:()=>(void 0!==t&&b(n,t,f),n._snapshot),subscribe(e){var i;let s,r,a=g(e),o={current:!1},l=(i=()=>{n.get(),o.current?a.next?.(n._snapshot):o.current=!0},s=()=>{let e=t;t=r,++f,r.depsTail=void 0,r.flags=6;try{return i()}finally{t=e,r.flags&=-5,E(r)}},r={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&_(this.deps,this)?s():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,E(this)}},s(),r);return{unsubscribe:()=>{l.stop()}}},_update(i){let r=t,a=(void 0)??Object.is;if(s)t=n,++f,n.depsTail=void 0;else if(void 0===i)return!1;s&&(n.flags=5);try{let t=n._snapshot,r="function"==typeof i?i(t):void 0===i&&s?e(t):i;if(void 0===t||!a(t,r))return n._snapshot=r,!0;return!1}finally{t=r,s&&(n.flags&=-5),E(n)}}},s?(n.flags=17,n.get=function(){let e=n.flags;if(16&e||32&e&&_(n.deps,n)){if(n._update()){let e=n.subs;void 0!==e&&y(e)}}else 32&e&&(n.flags=-33&e);return void 0!==t&&b(n,t,f),n._snapshot}):n.set=function(e){if(n._update(e)){let e=n.subs;if(void 0!==e&&(v(e),y(e),1)){for(;j<w;){let e=h[j];h[j++]=void 0,e.notify()}j=0,w=0}}},n),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),i&&(this.actions=i(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(g(e))}};function C(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let N={enabled:!0,leading:!1,trailing:!0,wait:0};var $=class{#f;constructor(e,t){this.fn=e,this.store=new k(C()),this.setOptions=e=>{this.options={...this.options,...e},this.#b()||this.cancel()},this.#x=e=>{this.store.setState(t=>{let i={...t,...e},{isPending:s}=i;return{...i,status:this.#b()?s?"pending":"idle":"disabled"}}),((e,t)=>{let i=t.key;if(i){var s,n;p.set(i,t),m.emit(e,{key:(s={...t,key:i}).key,store:{state:c("function"==typeof(n=s.store).get?n.get():n.state)},options:c(s.options)})}})("Debouncer",this)},this.#b=()=>!!u(this.options.enabled,this),this.#v=()=>u(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#b())return;this.#x({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#x({canLeadingExecute:!1}),t=!0,this.#_(...e)),this.options.trailing&&this.#x({isPending:!0,lastArgs:e}),this.#f&&clearTimeout(this.#f),this.#f=setTimeout(()=>{this.#x({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#_(...e)},this.#v())},this.#_=(...e)=>{this.#b()&&(this.fn(...e),this.#x({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#y(),this.#_(...this.store.state.lastArgs))},this.#y=()=>{this.#f&&(clearTimeout(this.#f),this.#f=void 0)},this.cancel=()=>{this.#y(),this.#x({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#x(C())},this.key=t.key,this.options={...N,...t},this.#x(this.options.initialState??{}),this.key&&m.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#x(e.payload.store.state),this.setOptions(e.payload.options))})}#x;#b;#v;#_;#y};e.s(["f",0,function(e,t,r=()=>({})){let a={...((0,i.useContext)(s)?.defaultOptions??{}).debouncer,...t},[o]=(0,i.useState)(()=>{let t=new $(e,a);return t.Subscribe=function(e){let i=l(t.store,e.selector,{compare:n});return"function"==typeof e.children?e.children(i):e.children},t});o.fn=e,o.setOptions(a),(0,i.useEffect)(()=>()=>{a.onUnmount?a.onUnmount(o):o.cancel()},[]);let u=l(o.store,r,{compare:n});return(0,i.useMemo)(()=>({...o,state:u}),[o,u])}],540626)},586448,e=>{var t=e.i(271645),i=e.i(540143),s=e.i(869230),n=e.i(915823),r=e.i(619273);function a(e,t){let i=new Set(t);return e.filter(e=>!i.has(e))}var o=class extends n.f{#j;#w;#E;#k;#C;#N;#$;#S;#T;#I=[];constructor(e,t,i){super(),this.#j=e,this.#k=i,this.#E=[],this.#C=[],this.#w=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#C.forEach(e=>{e.subscribe(t=>{this.#L(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#C.forEach(e=>{e.destroy()})}setQueries(e,t){this.#E=e,this.#k=t,i.f.batch(()=>{let e=this.#C,t=this.#O(this.#E);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let i=t.map(e=>e.observer),s=i.map(e=>e.getCurrentResult()),n=e.length!==i.length,o=i.some((t,i)=>t!==e[i]),l=n||o,u=!!l||s.some((e,t)=>{let i=this.#w[t];return!i||!(0,r.U)(e,i)});(l||u)&&(l&&(this.#I=t,this.#C=i),this.#w=s,this.hasListeners()&&(l&&(a(e,i).forEach(e=>{e.destroy()}),a(i,e).forEach(e=>{e.subscribe(t=>{this.#L(e,t)})})),this.#R()))})}getCurrentResult(){return this.#w}getQueries(){return this.#C.map(e=>e.getCurrentQuery())}getObservers(){return this.#C}getOptimisticResult(e,t){let i=this.#O(e),s=i.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),n=i.map(e=>e.defaultedQueryOptions.queryHash);return[s,e=>this.#A(e??s,t,n),()=>this.#z(s,i)]}#z(e,t){return t.map((i,s)=>{let n=e[s];return i.defaultedQueryOptions.notifyOnChangeProps?n:i.observer.trackResult(n,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#A(e,t,i){if(t){let s=this.#T,n=void 0!==i&&void 0!==s&&(s.length!==i.length||i.some((e,t)=>e!==s[t]));return(!this.#N||this.#w!==this.#S||n||t!==this.#$)&&(this.#$=t,this.#S=this.#w,void 0!==i&&(this.#T=i),this.#N=(0,r.B)(this.#N,t(e))),this.#N}return e}#M(){return this.#k?.combine!==void 0&&this.#C.some((e,t)=>e.options.suspense&&this.#w[t]?.data===void 0)}#O(e){let t=new Map;this.#C.forEach(e=>{let i=e.options.queryHash;if(!i)return;let s=t.get(i);s?s.push(e):t.set(i,[e])});let i=[];return e.forEach(e=>{let n=this.#j.defaultQueryOptions(e),r=t.get(n.queryHash)?.shift()??new s.f(this.#j,n);i.push({defaultedQueryOptions:n,observer:r})}),i}#L(e,t){let i=this.#C.indexOf(e);if(-1!==i){var s;let e;this.#w=(s=this.#w,(e=s.slice(0))[i]=t,e),this.#R()}}#R(){if(this.hasListeners()){let e=this.#z(this.#w,this.#I),t=this.#M(),s=this.#N,n=t?s:this.#A(e,this.#k?.combine);(t||s!==n)&&i.f.batch(()=>{this.listeners.forEach(e=>{e(this.#w)})})}}},l=e.i(912598),u=e.i(381384),d=e.i(673664),p=e.i(427001),c=e.i(254440);e.s(["f",0,function({queries:e,...n},a){let m=(0,l.d)(a),g=(0,u.f)(),h=(0,d.p)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,m,g]);f.forEach(e=>{(0,c.p)(e);let t=m.getQueryCache().get(e.queryHash);(0,p.l)(e,h,t)}),(0,p.E)(h);let[b]=t.useState(()=>new o(m,f,n)),[x,v,_]=b.getOptimisticResult(f,n.combine),y=!g&&!1!==n.subscribed;t.useSyncExternalStore(t.useCallback(e=>y?b.subscribe(i.f.batchCalls(e)):r.z,[b,y]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(f,n)},[f,n,b]);let j=x.some((e,t)=>(0,c.Q)(f[t],e))?x.flatMap((e,t)=>{let i=f[t];if(i&&(0,c.Q)(i,e)){let e=new s.f(m,i);return(0,c.X)(i,e,h)}return[]}):[];if(j.length>0)throw Promise.all(j);let w=x.find((e,t)=>{let i=f[t];return i&&(0,p.P)({result:e,errorResetBoundary:h,throwOnError:i.throwOnError,query:m.getQueryCache().get(i.queryHash),suspense:i.suspense})});if(w?.error)throw w.error;return v(_())}],586448)},546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},198458,e=>{var t=e.i(655063),i=e.i(266027),s=e.i(271645),n=e.i(741466);e.s(["f",0,function(e){let{queryKey:r,fetchPage:a,serializeFilters:o,defaultSorting:l,defaultPageSize:u,enabled:d}=e,[p,c]=(0,s.useState)(l),[m,g]=(0,s.useState)({pageIndex:0,pageSize:u}),[h,f]=(0,s.useState)([]),[b,x]=(0,s.useState)(""),[v]=(0,t.f)(b,{wait:n.f}),_=(0,s.useMemo)(()=>{let e=p.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=v.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(h)}},[p,m.pageIndex,m.pageSize,v,h,o]),y={queryKey:[...r,_],queryFn:({signal:e})=>a(_,e),enabled:d,placeholderData:e=>e},{data:j,isLoading:w,isPlaceholderData:E,isFetching:k,error:C,refetch:N}=(0,i.f)(y),$=(0,s.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),S=(0,s.useCallback)(e=>{c(e),$()},[$]),T=(0,s.useCallback)(e=>{f(e),$()},[$]),I=(0,s.useCallback)(e=>{x(e),$()},[$]),L=(0,s.useCallback)(()=>{N()},[N]);return{rows:(0,s.useMemo)(()=>j?.data??[],[j]),rowCount:j?.meta.total_count??0,isLoading:w||E,isFetching:k,error:C,refetch:L,sorting:p,onSortingChange:S,pagination:m,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:T,searchValue:b,onSearchChange:I}}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:s,accessToken:n,apiKey:r,inputMessage:a,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:d,selectedPolicies:p,selectedVoice:c,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,x="session"===s?n:r,v=window.location.origin,_=f?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?v=_:f?.PROXY_BASE_URL&&(v=f.PROXY_BASE_URL);let y=a||"Your prompt here",j=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),u.length>0&&(E.vector_stores=u),d.length>0&&(E.guardrails=d),p.length>0&&(E.policies=p);let k=g||"your-model-name",C=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",N="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"${C}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"${C}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(E).length>0,t="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:y}];i=`
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
#                     "text": "${j}"
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
`;break}case t.Q.RESPONSES:{let e=Object.keys(E).length>0,t="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:y}];i=`
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.Q.IMAGE:i="azure"===h?`
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
prompt = "${j}"

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
`;break;case t.Q.IMAGE_EDITS:i="azure"===h?`
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
prompt = "${j}"

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
prompt = "${j}"

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
`;break;case t.Q.EMBEDDINGS:i=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${a||"Your text to convert to speech here"}",
	voice="${c}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${N}
${i}`}])},652272,209261,e=>{var t=e.i(843476),i=e.i(271645),s=e.i(871689),n=e.i(643531),r=e.i(174886),a=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,u=e=>e.trim().replace(/\/+$/,""),d=/\.(md|markdown|txt|json|ya?ml|toml)$/i,p=/\.zip$/i,c=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,g=/^[A-Za-z0-9-]+$/,h=/^[A-Za-z0-9._-]+$/,f=/^https?:\/\//i,b="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,v=e=>e.pathname.split("/").filter(e=>""!==e),_=e=>{try{return new URL(e)}catch{return null}},y=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),j=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},w=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),E=(e,t,i,s)=>{let n=u(s??"");return""!==n?l.test(n)?{parsed:{source:"git-subdir",url:t,path:n},label:`${e} subdir — ${t} @ ${n}`,suggestedName:w(j(n))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:w(i)}},k=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),C=e=>`/plugin install ${e.name}@litellm`,N=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",$=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&f.test(e.url)?e.url:null;e.s(["U",0,k,"a",0,C,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"_",0,N,"E",0,$,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||c.test(e.trim()),"P",0,e=>{let t=u(e);return""!==t&&l.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,t)=>{let i=((e,t)=>{let i=e.trim(),s=x.exec(i),n=s?`${b}${s[1]}@${s[2]}/${s[3]}`:i;if(!n.toLowerCase().startsWith(b))return null;let r=_(n);if(!r||""===r.username||""!==r.password||!y(r))return null;let a=n.indexOf("/",b.length);return -1===a||r.pathname!==n.slice(a)||v(r).length<2?null:E("SSH",i,j(r.pathname).replace(/\.git$/i,""),t)})(e,t);if(i)return i;let s=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=_(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&y(i)?i:null})(e);if(!s)return null;if(p.test(s.pathname))return{parsed:{source:"archive",url:s.href},label:`Zip archive — ${s.host}${s.pathname}`,suggestedName:w(j(s.pathname).replace(p,""))};if("github.com"===s.hostname.replace(/^www\./,""))return((e,t)=>{let i=v(e);if(i.length<2)return null;let s=i[0],n=i[1].replace(/\.git$/,"");if(!g.test(s)||!h.test(n))return null;let r=`${s}/${n}`,a=`https://github.com/${r}`,o={parsed:{source:"github",repo:r},label:`GitHub repo — ${r}`,suggestedName:w(n)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=j(e.join("/")),s=d.test(t)?e.slice(0,-1):e;if(0===s.length)return o;let n=u(s.join("/"));return l.test(n)?{parsed:{source:"git-subdir",url:a,path:n},label:`GitHub subdir — ${r} @ ${n}`,suggestedName:w(j(n))}:null}if(2!==i.length)return null;let p=u(t??"");return""!==p?l.test(p)?{parsed:{source:"git-subdir",url:a,path:p},label:`GitHub subdir — ${r} @ ${p}`,suggestedName:w(j(p))}:null:o})(s,t);if(v(s).length<2)return null;let n=j(s.pathname).replace(/\.git$/,"");return E("Git",`${s.protocol}//${s.host}${s.pathname.replace(/\/+$/,"")}`,n,t)},"e",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let S=({source:e})=>{let i=$(e),s=i&&"git-subdir"===e.source&&e.path?`${i}/tree/main/${e.path}`:i;return s?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:s,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[s.replace("https://",""),(0,t.jsx)(a.f,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:N(e)})]}):null};e.s(["f",0,({skill:e,onBack:a})=>{let[l,u]=(0,i.useState)("overview"),[d,p]=(0,i.useState)(null),c=(e,t)=>{navigator.clipboard.writeText(e),p(t),setTimeout(()=>p(null),2e3)},m=C(e),g=k(window.location.origin),h=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:a,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(s.f,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>u(e.key),className:(0,o.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:h.map((e,i)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(S,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>c(m,"install"),className:(0,o.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===d?"text-success":"text-info"),children:["install"===d?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),"install"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>u("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;c(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===d?"text-success":"text-info"),children:["marketplace-cmd"===d?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),"marketplace-cmd"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>c(g,"settings"),className:(0,o.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===d?"text-success":"text-info"),children:["settings"===d?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),"settings"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:g})]})]})]})}],652272)},845150,e=>{var t=e.i(843476),i=e.i(271645),s=e.i(131792);let n=(e,t)=>{let i=t.trim().toLowerCase();return!i||e.label.toLowerCase().includes(i)||e.value.toLowerCase().includes(i)||(e.description?.toLowerCase().includes(i)??!1)};e.s(["f",0,function({id:e,options:r,value:a=[],onValueChange:o,placeholder:l="Select options",emptyText:u="No options found",disabled:d=!1,loading:p=!1,allowCustomValues:c=!1,className:m}){let g=(0,s.d)(),[h,f]=(0,i.useState)(""),b=r.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=a.filter(e=>"string"==typeof e&&e.length>0).map(e=>b.find(t=>t.value===e)??{label:e,value:e}),v=h.trim(),_=b.some(e=>e.value.toLowerCase()===v.toLowerCase()),y=c&&v&&!_?[...b,{label:`Create "${v}"`,value:v}]:b;return(0,t.jsxs)(s.x,{multiple:!0,items:y,value:x,onValueChange:e=>{o(Array.from(new Set(c?e.flatMap(e=>a.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),f("")},inputValue:h,onInputValueChange:f,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:n,disabled:d||p,children:[(0,t.jsx)(s.V,{render:(0,t.jsx)("div",{ref:g}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(s.l,{children:i=>(0,t.jsxs)(t.Fragment,{children:[i.map(e=>(0,t.jsx)(s.Q,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(s.z,{id:e,placeholder:p?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),i.length>0&&!d&&!p&&(0,t.jsx)(s.y,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(s.R,{anchor:g,children:[(0,t.jsx)(s.S,{children:u}),(0,t.jsx)(s.k,{children:e=>(0,t.jsx)(s.T,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},741466,e=>{e.s(["f",0,300])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let s=t(e);if(""===s)return!0;let n=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!n.some(e=>e.includes(s))||s.split(/\s+/).every(e=>n.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,s){return e.filter(e=>i(t,s(e)))},"h",0,i,"e",0,function(e,i,s){let n=t(i);if(""===n)return[...e];let r=e=>{let t=s(e).toLowerCase();return 1e3*(t===n)+100*!!t.startsWith(n)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}])})();