(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,541071,373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t],373488),e.s(["MoreHorizontal",0,t],541071)},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},655063,e=>{"use strict";var t=e.i(540626),i=e.i(271645);e.s(["useDebouncedValue",0,function(e,n,s){let[r,a,o]=function(e,n,s){let[r,a]=(0,i.useState)(e),o=(0,t.useDebouncer)(a,n,s);return[r,o.maybeExecute,o]}(e,n,s);return(0,i.useEffect)(()=>{a(e)},[e,a]),[r,o]}],655063)},68155,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"}))});e.s(["TrashIcon",0,i],68155)},250980,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,i],250980)},180127,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["default",0,t])},871689,e=>{"use strict";var t=e.i(180127);e.s(["ArrowLeft",()=>t.default])},845150,e=>{"use strict";var t=e.i(843476),i=e.i(271645),n=e.i(131792);let s=(e,t)=>{let i=t.trim().toLowerCase();return!i||e.label.toLowerCase().includes(i)||e.value.toLowerCase().includes(i)||(e.description?.toLowerCase().includes(i)??!1)};e.s(["MultiSelect",0,function({id:e,options:r,value:a=[],onValueChange:o,placeholder:l="Select options",emptyText:d="No options found",disabled:u=!1,loading:c=!1,allowCustomValues:p=!1,className:m}){let g=(0,n.useComboboxAnchor)(),[h,f]=(0,i.useState)(""),b=r.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=a.filter(e=>"string"==typeof e&&e.length>0).map(e=>b.find(t=>t.value===e)??{label:e,value:e}),v=h.trim(),_=b.some(e=>e.value.toLowerCase()===v.toLowerCase()),y=p&&v&&!_?[...b,{label:`Create "${v}"`,value:v}]:b;return(0,t.jsxs)(n.Combobox,{multiple:!0,items:y,value:x,onValueChange:e=>{o(Array.from(new Set(p?e.flatMap(e=>a.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),f("")},inputValue:h,onInputValueChange:f,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:s,disabled:u||c,children:[(0,t.jsx)(n.ComboboxChips,{render:(0,t.jsx)("div",{ref:g}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(n.ComboboxValue,{children:i=>(0,t.jsxs)(t.Fragment,{children:[i.map(e=>(0,t.jsx)(n.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(n.ComboboxChipsInput,{id:e,placeholder:c?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),i.length>0&&!u&&!c&&(0,t.jsx)(n.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(n.ComboboxContent,{anchor:g,children:[(0,t.jsx)(n.ComboboxEmpty,{children:d}),(0,t.jsx)(n.ComboboxList,{children:e=>(0,t.jsx)(n.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},540626,e=>{"use strict";let t;var i=e.i(271645);let n=(0,i.createContext)(null);function s(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[i,n]of e)if(!t.has(i)||!Object.is(n,t.get(i)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let i of e)if(!t.has(i))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let i=r(e);if(i.length!==r(t).length)return!1;for(let n=0;n<i.length;n++)if(!Object.prototype.hasOwnProperty.call(t,i[n])||!Object.is(e[i[n]],t[i[n]]))return!1;return!0}function r(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var a=e.i(430224);function o(e,t){return e===t}function l(e,t=e=>e,n){let s=n?.compare??o,r=(0,i.useCallback)(t=>{let{unsubscribe:i}=e.subscribe(t);return i},[e]),d=(0,i.useCallback)(()=>e.get(),[e]);return(0,a.useSyncExternalStoreWithSelector)(r,d,d,t,s)}function d(e,...t){return"function"==typeof e?e(...t):e}var u=class{#e=!0;#t;#i;#n;#s;#r;#a;#o;#l=0;#d=5;#u=!1;#c=!1;#p=null;#m=()=>{this.debugLog("Connected to event bus"),this.#r=!0,this.#u=!1,this.debugLog("Emitting queued events",this.#s),this.#s.forEach(e=>this.emitEventToBus(e)),this.#s=[],this.stopConnectLoop(),this.#i().removeEventListener("tanstack-connect-success",this.#m)};#g=()=>{if(this.#l<this.#d){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#i().removeEventListener("tanstack-connect",this.#g),this.#c=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#h=()=>{this.#u||(this.#u=!0,this.#i().addEventListener("tanstack-connect-success",this.#m),this.#g())};constructor({pluginId:e,debug:t=!1,enabled:i=!0,reconnectEveryMs:n=300}){this.#t=e,this.#e=i,this.#i=this.getGlobalTarget,this.#n=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#s=[],this.#r=!1,this.#c=!1,this.#a=null,this.#o=n}startConnectLoop(){null!==this.#a||this.#r||(this.debugLog(`Starting connect loop (every ${this.#o}ms)`),this.#a=setInterval(this.#g,this.#o))}stopConnectLoop(){this.#u=!1,null!==this.#a&&(clearInterval(this.#a),this.#a=null,this.#s=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#n&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let i=new Event(e,{detail:t});this.#i().dispatchEvent(i)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#i().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(i){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#p&&(this.debugLog("Emitting event to internal event target",e,t),this.#p.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#c)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#r){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#s.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#u&&(this.#h(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,i){let n=i?.withEventTarget??!1,s=`${this.#t}:${e}`;if(n&&(this.#p||(this.#p=new EventTarget),this.#p.addEventListener(s,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",s),()=>{};let r=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#i().addEventListener(s,r),this.debugLog("Registered event to bus",s),()=>{n&&this.#p?.removeEventListener(s,r),this.#i().removeEventListener(s,r)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let i=t.detail;this.#t&&i.pluginId!==this.#t||e(i)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}};let c=new Map;function p(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let m=new class extends u{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function g(e,t,i){let n="object"==typeof e,s=n?e:void 0;return{next:(n?e.next:e)?.bind(s),error:(n?e.error:t)?.bind(s),complete:(n?e.complete:i)?.bind(s)}}let h=[],f=0,{link:b,unlink:x,propagate:v,checkDirty:_,shallowPropagate:y}=function({update:e,notify:t,unwatched:i}){return{link:function(e,t,i){let n=t.depsTail;if(void 0!==n&&n.dep===e)return;let s=void 0!==n?n.nextDep:t.deps;if(void 0!==s&&s.dep===e){s.version=i,t.depsTail=s;return}let r=e.subsTail;if(void 0!==r&&r.version===i&&r.sub===t)return;let a=t.depsTail=e.subsTail={version:i,dep:e,sub:t,prevDep:n,nextDep:s,prevSub:r,nextSub:void 0};void 0!==s&&(s.prevDep=a),void 0!==n?n.nextDep=a:t.deps=a,void 0!==r?r.nextSub=a:e.subs=a},unlink:function(e,t=e.sub){let n=e.dep,s=e.prevDep,r=e.nextDep,a=e.nextSub,o=e.prevSub;return void 0!==r?r.prevDep=s:t.depsTail=s,void 0!==s?s.nextDep=r:t.deps=r,void 0!==a?a.prevSub=o:n.subsTail=o,void 0!==o?o.nextSub=a:void 0===(n.subs=a)&&i(n),r},propagate:function(e){let i,n=e.nextSub;e:for(;;){let s=e.sub,r=s.flags;if(60&r?12&r?4&r?!(48&r)&&function(e,t){let i=t.depsTail;for(;void 0!==i;){if(i===e)return!0;i=i.prevDep}return!1}(e,s)?(s.flags=40|r,r&=1):r=0:s.flags=-9&r|32:r=0:s.flags=32|r,2&r&&t(s),1&r){let t=s.subs;if(void 0!==t){let s=(e=t).nextSub;void 0!==s&&(i={value:n,prev:i},n=s);continue}}if(void 0!==(e=n)){n=e.nextSub;continue}for(;void 0!==i;)if(e=i.value,i=i.prev,void 0!==e){n=e.nextSub;continue e}break}},checkDirty:function(t,i){let s,r=0,a=!1;e:for(;;){let o=t.dep,l=o.flags;if(16&i.flags)a=!0;else if((17&l)==17){if(e(o)){let e=o.subs;void 0!==e.nextSub&&n(e),a=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(s={value:t,prev:s}),t=o.deps,i=o,++r;continue}if(!a){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;r--;){let r=i.subs,o=void 0!==r.nextSub;if(o?(t=s.value,s=s.prev):t=r,a){if(e(i)){o&&n(r),i=t.sub;continue}a=!1}else i.flags&=-33;i=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return a}},shallowPropagate:n};function n(e){do{let i=e.sub,n=i.flags;(48&n)==32&&(i.flags=16|n,(6&n)==2&&t(i))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){h[E++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,j(e))}}),w=0,E=0;function j(e){let t=e.depsTail,i=void 0!==t?t.nextDep:e.deps;for(;void 0!==i;)i=x(i,e)}var k=class{constructor(e,i){this.atom=function(e){let i="function"==typeof e,n={_snapshot:i?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!i,get:()=>(void 0!==t&&b(n,t,f),n._snapshot),subscribe(e){var i;let s,r,a=g(e),o={current:!1},l=(i=()=>{n.get(),o.current?a.next?.(n._snapshot):o.current=!0},s=()=>{let e=t;t=r,++f,r.depsTail=void 0,r.flags=6;try{return i()}finally{t=e,r.flags&=-5,j(r)}},r={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&_(this.deps,this)?s():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,j(this)}},s(),r);return{unsubscribe:()=>{l.stop()}}},_update(s){let r=t,a=(void 0)??Object.is;if(i)t=n,++f,n.depsTail=void 0;else if(void 0===s)return!1;i&&(n.flags=5);try{let t=n._snapshot,r="function"==typeof s?s(t):void 0===s&&i?e(t):s;if(void 0===t||!a(t,r))return n._snapshot=r,!0;return!1}finally{t=r,i&&(n.flags&=-5),j(n)}}};return i?(n.flags=17,n.get=function(){let e=n.flags;if(16&e||32&e&&_(n.deps,n)){if(n._update()){let e=n.subs;void 0!==e&&y(e)}}else 32&e&&(n.flags=-33&e);return void 0!==t&&b(n,t,f),n._snapshot}):n.set=function(e){if(n._update(e)){let e=n.subs;if(void 0!==e&&(v(e),y(e),1)){for(;w<E;){let e=h[w];h[w++]=void 0,e.notify()}w=0,E=0}}},n}(e),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),i&&(this.actions=i(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(g(e))}};function C(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let N={enabled:!0,leading:!1,trailing:!0,wait:0};var I=class{#f;constructor(e,t){this.fn=e,this.store=new k(C()),this.setOptions=e=>{this.options={...this.options,...e},this.#b()||this.cancel()},this.#x=e=>{this.store.setState(t=>{let i={...t,...e},{isPending:n}=i;return{...i,status:this.#b()?n?"pending":"idle":"disabled"}}),((e,t)=>{let i=t.key;if(i){var n,s;c.set(i,t),m.emit(e,{key:(n={...t,key:i}).key,store:{state:p("function"==typeof(s=n.store).get?s.get():s.state)},options:p(n.options)})}})("Debouncer",this)},this.#b=()=>!!d(this.options.enabled,this),this.#v=()=>d(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#b())return;this.#x({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#x({canLeadingExecute:!1}),t=!0,this.#_(...e)),this.options.trailing&&this.#x({isPending:!0,lastArgs:e}),this.#f&&clearTimeout(this.#f),this.#f=setTimeout(()=>{this.#x({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#_(...e)},this.#v())},this.#_=(...e)=>{this.#b()&&(this.fn(...e),this.#x({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#y(),this.#_(...this.store.state.lastArgs))},this.#y=()=>{this.#f&&(clearTimeout(this.#f),this.#f=void 0)},this.cancel=()=>{this.#y(),this.#x({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#x(C())},this.key=t.key,this.options={...N,...t},this.#x(this.options.initialState??{}),this.key&&m.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#x(e.payload.store.state),this.setOptions(e.payload.options))})}#x;#b;#v;#_;#y};e.s(["useDebouncer",0,function(e,t,r=()=>({})){let a={...((0,i.useContext)(n)?.defaultOptions??{}).debouncer,...t},[o]=(0,i.useState)(()=>{let t=new I(e,a);return t.Subscribe=function(e){let i=l(t.store,e.selector,{compare:s});return"function"==typeof e.children?e.children(i):e.children},t});o.fn=e,o.setOptions(a),(0,i.useEffect)(()=>()=>{a.onUnmount?a.onUnmount(o):o.cancel()},[]);let d=l(o.store,r,{compare:s});return(0,i.useMemo)(()=>({...o,state:d}),[o,d])}],540626)},741466,e=>{"use strict";e.s(["DEBOUNCE_WAIT_MS",0,300])},871943,502547,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["ChevronDownIcon",0,i],871943);let n=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,n],502547)},278587,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["RefreshIcon",0,i],278587)},360820,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["ChevronUpIcon",0,i],360820)},434626,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,i],434626)},902555,e=>{"use strict";var t=e.i(843476),i=e.i(746798),n=e.i(271645);let s=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))}),r=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});var a=e.i(278587),o=e.i(68155),l=e.i(360820),d=e.i(871943),u=e.i(434626);let c=n.forwardRef(function(e,t){return n.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),n.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});var p=e.i(196631);function m({icon:e,onClick:i,className:n,disabled:s,dataTestId:r}){return s?(0,t.jsx)("span",{className:"inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50","data-testid":r,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})}):(0,t.jsx)("span",{className:(0,p.cx)("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5",n),onClick:i,"data-testid":r,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})})}let g={Edit:{icon:s,className:"hover:text-info"},Delete:{icon:o.TrashIcon,className:"hover:text-destructive"},Test:{icon:r,className:"hover:text-info"},Regenerate:{icon:a.RefreshIcon,className:"hover:text-success"},Up:{icon:l.ChevronUpIcon,className:"hover:text-info"},Down:{icon:d.ChevronDownIcon,className:"hover:text-info"},Open:{icon:u.ExternalLinkIcon,className:"hover:text-success"},Copy:{icon:c,className:"hover:text-info"}};e.s(["default",0,function({onClick:e,tooltipText:n,disabled:s=!1,disabledTooltipText:r,dataTestId:a,variant:o}){let{icon:l,className:d}=g[o],u=s?r:n,c=(0,t.jsx)(m,{icon:l,onClick:e,className:d,disabled:s,dataTestId:a});return u?(0,t.jsx)(i.TooltipProvider,{children:(0,t.jsxs)(i.Tooltip,{children:[(0,t.jsx)(i.TooltipTrigger,{render:(0,t.jsx)("span",{}),children:c}),(0,t.jsx)(i.TooltipContent,{children:u})]})}):(0,t.jsx)("span",{children:c})}],902555)},198458,e=>{"use strict";var t=e.i(655063),i=e.i(266027),n=e.i(271645),s=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:r,fetchPage:a,serializeFilters:o,defaultSorting:l,defaultPageSize:d,enabled:u}=e,[c,p]=(0,n.useState)(l),[m,g]=(0,n.useState)({pageIndex:0,pageSize:d}),[h,f]=(0,n.useState)([]),[b,x]=(0,n.useState)(""),[v]=(0,t.useDebouncedValue)(b,{wait:s.DEBOUNCE_WAIT_MS}),_=(0,n.useMemo)(()=>{let e=c.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=v.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(h)}},[c,m.pageIndex,m.pageSize,v,h,o]),y={queryKey:[...r,_],queryFn:({signal:e})=>a(_,e),enabled:u,placeholderData:e=>e},{data:w,isLoading:E,isFetching:j,error:k,refetch:C}=(0,i.useQuery)(y),N=(0,n.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),I=(0,n.useCallback)(e=>{p(e),N()},[N]),T=(0,n.useCallback)(e=>{f(e),N()},[N]),S=(0,n.useCallback)(e=>{x(e),N()},[N]),L=(0,n.useCallback)(()=>{C()},[C]);return{rows:(0,n.useMemo)(()=>w?.data??[],[w]),rowCount:w?.meta.total_count??0,isLoading:E,isFetching:j,error:k,refetch:L,sorting:c,onSortingChange:I,pagination:m,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:T,searchValue:b,onSearchChange:S}}])},455037,e=>{"use strict";var t=e.i(494144);e.s(["prism",()=>t.default])},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),n=e.i(871689),s=e.i(643531),r=e.i(174886),a=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,d=e=>e.trim().replace(/\/+$/,""),u=/\.(md|markdown|txt|json|ya?ml|toml)$/i,c=/^\d{1,3}(\.\d{1,3}){3}$/,p=/^[A-Za-z0-9-]+$/,m=/^[A-Za-z0-9._-]+$/,g=e=>e.pathname.split("/").filter(e=>""!==e),h=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},f=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),b=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),x=e=>`/plugin install ${e.name}@litellm`;e.s(["buildMarketplaceSettingsSnippet",0,b,"formatInstallCommand",0,x,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=d(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let i=(e=>{let t,i=e.trim();if(""===i||i.startsWith("//"))return null;let n=/^[a-z][a-z0-9+.-]*:\/\//i.test(i)?i:`https://${i}`;try{t=new URL(n)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||c.test(t.hostname)?null:t})(e);if(!i)return null;if("github.com"===i.hostname.replace(/^www\./,""))return((e,t)=>{let i=g(e);if(i.length<2)return null;let n=i[0],s=i[1].replace(/\.git$/,"");if(!p.test(n)||!m.test(s))return null;let r=`${n}/${s}`,a=`https://github.com/${r}`,o={parsed:{source:"github",repo:r},label:`GitHub repo — ${r}`,suggestedName:f(s)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=h(e.join("/")),n=u.test(t)?e.slice(0,-1):e;if(0===n.length)return o;let s=d(n.join("/"));return l.test(s)?{parsed:{source:"git-subdir",url:a,path:s},label:`GitHub subdir — ${r} @ ${s}`,suggestedName:f(h(s))}:null}if(2!==i.length)return null;let c=d(t??"");return""!==c?l.test(c)?{parsed:{source:"git-subdir",url:a,path:c},label:`GitHub subdir — ${r} @ ${c}`,suggestedName:f(h(c))}:null:o})(i,t);if(g(i).length<2)return null;let n=`${i.protocol}//${i.host}${i.pathname.replace(/\/+$/,"")}`,s=d(t??"");return""!==s?l.test(s)?{parsed:{source:"git-subdir",url:n,path:s},label:`Git subdir — ${n} @ ${s}`,suggestedName:f(h(s))}:null:{parsed:{source:"url",url:n},label:`Git repo — ${n}`,suggestedName:f(h(i.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:l})=>{let d,[u,c]=(0,i.useState)("overview"),[p,m]=(0,i.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},h="github"===(d=e.source).source&&d.repo?`https://github.com/${d.repo}`:"git-subdir"===d.source&&d.url?d.path?`${d.url}/tree/main/${d.path}`:d.url:"url"===d.source&&d.url?d.url:null,f=x(e),v=b(window.location.origin),_=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:l,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(n.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>c(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",u===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===u&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:_.map((e,i)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),h&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:h,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[h.replace("https://",""),(0,t.jsx)(a.Link2,{className:"size-3 shrink-0"})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===u&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(f,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===p?"text-success":"text-info"),children:["install"===p?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:f})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,'not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>c("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===u&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;g(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===p?"text-success":"text-info"),children:["marketplace-cmd"===p?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"marketplace-cmd"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>g(v,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===p?"text-success":"text-info"),children:["settings"===p?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(r.Copy,{className:"size-3"}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:v})]})]})]})}],652272)},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function i(e,i){let n=t(e);if(""===n)return!0;let s=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!s.some(e=>e.includes(n))||n.split(/\s+/).every(e=>s.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,n){return e.filter(e=>i(t,n(e)))},"matchesSearchTerm",0,i,"rankBySearchRelevance",0,function(e,i,n){let s=t(i);if(""===s)return[...e];let r=e=>{let t=n(e).toLowerCase();return 1e3*(t===s)+100*!!t.startsWith(s)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])},909947,865361,e=>{"use strict";var t,i,n=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.COMPLETION="completion",t.RESPONSES="responses",t.IMAGE_EDITS="image_edit",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t.REALTIME="realtime",t),s=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i.INTERACTIONS="interactions",i);let r={image_generation:"image",video_generation:"video",chat:"chat",completion:"chat",responses:"responses",image_edit:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings",realtime:"realtime"};e.s(["EndpointType",()=>s,"ModelMode",()=>n,"getEndpointType",0,e=>Object.values(n).includes(e)?r[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:n,apiKey:r,inputMessage:a,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f}=e,b="session"===i?n:r,x=window.location.origin,v=f?.LITELLM_UI_API_DOC_BASE_URL;v&&v.trim()?x=v:f?.PROXY_BASE_URL&&(x=f.PROXY_BASE_URL);let _=a||"Your prompt here",y=_.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),d.length>0&&(E.vector_stores=d),u.length>0&&(E.guardrails=u),c.length>0&&(E.policies=c);let j=g||"your-model-name",k="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"
)`;switch(m){case s.CHAT:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:_}];t=`
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
#     ]${i}
# )
# print(response_with_file)
`;break}case s.RESPONSES:{let e=Object.keys(E).length>0,i="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:_}];t=`
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
#                 {"type": "input_text", "text": "${y}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case s.IMAGE:t="azure"===h?`
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
prompt = "${y}"

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
`;break;case s.IMAGE_EDITS:t="azure"===h?`
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
prompt = "${y}"

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
`;break;case s.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${j}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case s.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${j}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case s.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${j}",
	input="${a||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${k}
${t}`}],909947)},157058,e=>{"use strict";var t=e.i(843476),i=e.i(934879),n=e.i(976883),s=e.i(135214),r=e.i(708347);e.s(["default",0,function(){let{accessToken:e,userRole:a,premiumUser:o}=(0,s.default)();return(0,r.isAdminRole)(a)?(0,t.jsx)(i.default,{accessToken:e,publicPage:!1,premiumUser:o,userRole:a}):(0,t.jsx)(n.default,{accessToken:e,isEmbedded:!0})}])}]);