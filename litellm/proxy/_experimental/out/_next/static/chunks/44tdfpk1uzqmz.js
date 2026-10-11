(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,749099,e=>{"use strict";e.s([],924305),e.i(924305);var t,i=e.i(271645),n=e.i(951437),s=e.i(828918),a=e.i(146376),r=e.i(502077),o=e.i(956789),l=e.i(333848),d=e.i(552245),u=e.i(176782),c=e.i(788015),p=e.i(540886),m=e.i(733332);let g=i.createContext(void 0);var h=e.i(875812);let f=((t={}).checked="data-checked",t.unchecked="data-unchecked",t.disabled="data-disabled",t.readonly="data-readonly",t.required="data-required",t.valid="data-valid",t.invalid="data-invalid",t.touched="data-touched",t.dirty="data-dirty",t.filled="data-filled",t.focused="data-focused",t),b={...h.fieldValidityMapping,checked:e=>e?{[f.checked]:""}:{[f.unchecked]:""}};var x=e.i(469690),v=e.i(381104),_=e.i(884708),y=e.i(247778),k=e.i(31421),w=e.i(538489),E=e.i(675606),j=e.i(56434),C=e.i(606039),S=e.i(843476);let T=i.forwardRef(function(e,t){let{checked:m,className:h,defaultChecked:f,"aria-labelledby":T,form:N,id:I,inputRef:L,name:$,nativeButton:A=!1,onCheckedChange:z,readOnly:O=!1,required:M=!1,disabled:P=!1,render:D,uncheckedValue:R,value:q,style:B,...U}=e,{clearErrors:F}=(0,_.useFormContext)(),{state:H,setTouched:V,setDirty:K,validityData:W,setFilled:G,setFocused:Y,validationMode:Z,disabled:J,name:Q,validation:X}=(0,x.useFieldRootContext)(),{labelId:ee}=(0,y.useLabelableContext)(),et=J||P,ei=Q??$,en=i.useRef(null),es=(0,s.useMergedRefs)(en,L,X.inputRef),ea=i.useRef(null),er=(0,c.useBaseUiId)(),eo=(0,w.useLabelableId)({id:I,implicit:!1,controlRef:ea}),el=A?void 0:eo,[ed,eu]=(0,n.useControlled)({controlled:m,default:!!f,name:"Switch",state:"checked"});(0,v.useRegisterFieldControl)(ea,er,ed,void 0,!et,$),(0,a.useIsoLayoutEffect)(()=>{en.current&&G(en.current.checked)},[en,G]),(0,C.useValueChanged)(ed,()=>{F(ei),K(ed!==W.initialValue),G(ed),X.change(ed)});let{getButtonProps:ec,buttonRef:ep}=(0,p.useButton)({disabled:et,native:A}),em=(0,k.useAriaLabelledBy)(T,ee,en,!A,el),eg=(0,u.mergeProps)({checked:ed,disabled:et,form:N,id:el,name:ei,required:M,style:ei?r.visuallyHiddenInput:r.visuallyHidden,tabIndex:-1,type:"checkbox","aria-hidden":!0,ref:es,onChange(e){if(e.nativeEvent.defaultPrevented)return;if(O)return void e.preventDefault();let t=e.currentTarget.checked,i=(0,E.createChangeEventDetails)(j.REASONS.none,e.nativeEvent);z?.(t,i),i.isCanceled||eu(t)},onFocus(){ea.current?.focus()}},e=>X.getValidationProps(et,e),void 0!==q?{value:q}:o.EMPTY_OBJECT),eh=i.useMemo(()=>({...H,checked:ed,disabled:et,readOnly:O,required:M}),[H,ed,et,O,M]),ef=(0,d.useRenderElement)("span",e,{state:eh,ref:[t,ea,ep],props:[{id:A?eo:er,role:"switch","aria-checked":ed,"aria-readonly":O||void 0,"aria-required":M||void 0,"aria-labelledby":em,onFocus(){et||Y(!0)},onBlur(){let e=en.current;e&&!et&&(V(!0),Y(!1),"onBlur"===Z&&X.commit(e.checked))},onClick(e){if(O||et)return;e.preventDefault();let t=en.current;t&&t.dispatchEvent(new((0,l.ownerWindow)(t)).PointerEvent("click",{bubbles:!0,shiftKey:e.shiftKey,ctrlKey:e.ctrlKey,altKey:e.altKey,metaKey:e.metaKey}))}},U,ec,e=>X.getValidationProps(et,e)],stateAttributesMapping:b});return(0,S.jsxs)(g.Provider,{value:eh,children:[ef,!ed&&ei&&void 0!==R&&(0,S.jsx)("input",{type:"hidden",form:N,name:ei,value:R,disabled:et}),(0,S.jsx)("input",{...eg,suppressHydrationWarning:!0})]})}),N=i.forwardRef(function(e,t){let{render:n,className:s,style:a,...r}=e,o=function(){let e=i.useContext(g);if(void 0===e)throw Error((0,m.default)(63));return e}();return(0,d.useRenderElement)("span",e,{state:o,ref:t,stateAttributesMapping:b,props:r})});e.s(["Root",0,T,"Thumb",0,N],450994);var I=e.i(450994);e.s(["Switch",0,I],749099)},434626,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,i],434626)},655063,e=>{"use strict";var t=e.i(540626),i=e.i(271645);e.s(["useDebouncedValue",0,function(e,n,s){let[a,r,o]=function(e,n,s){let[a,r]=(0,i.useState)(e),o=(0,t.useDebouncer)(r,n,s);return[a,o.maybeExecute,o]}(e,n,s);return(0,i.useEffect)(()=>{r(e)},[e,r]),[a,o]}],655063)},540626,e=>{"use strict";let t;var i=e.i(271645);let n=(0,i.createContext)(null);function s(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[i,n]of e)if(!t.has(i)||!Object.is(n,t.get(i)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let i of e)if(!t.has(i))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let i=a(e);if(i.length!==a(t).length)return!1;for(let n=0;n<i.length;n++)if(!Object.prototype.hasOwnProperty.call(t,i[n])||!Object.is(e[i[n]],t[i[n]]))return!1;return!0}function a(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var r=e.i(430224);function o(e,t){return e===t}function l(e,t=e=>e,n){let s=n?.compare??o,a=(0,i.useCallback)(t=>{let{unsubscribe:i}=e.subscribe(t);return i},[e]),d=(0,i.useCallback)(()=>e.get(),[e]);return(0,r.useSyncExternalStoreWithSelector)(a,d,d,t,s)}function d(e,...t){return"function"==typeof e?e(...t):e}var u=class{#e=!0;#t;#i;#n;#s;#a;#r;#o;#l=0;#d=5;#u=!1;#c=!1;#p=null;#m=()=>{this.debugLog("Connected to event bus"),this.#a=!0,this.#u=!1,this.debugLog("Emitting queued events",this.#s),this.#s.forEach(e=>this.emitEventToBus(e)),this.#s=[],this.stopConnectLoop(),this.#i().removeEventListener("tanstack-connect-success",this.#m)};#g=()=>{if(this.#l<this.#d){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#i().removeEventListener("tanstack-connect",this.#g),this.#c=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#h=()=>{this.#u||(this.#u=!0,this.#i().addEventListener("tanstack-connect-success",this.#m),this.#g())};constructor({pluginId:e,debug:t=!1,enabled:i=!0,reconnectEveryMs:n=300}){this.#t=e,this.#e=i,this.#i=this.getGlobalTarget,this.#n=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#s=[],this.#a=!1,this.#c=!1,this.#r=null,this.#o=n}startConnectLoop(){null!==this.#r||this.#a||(this.debugLog(`Starting connect loop (every ${this.#o}ms)`),this.#r=setInterval(this.#g,this.#o))}stopConnectLoop(){this.#u=!1,null!==this.#r&&(clearInterval(this.#r),this.#r=null,this.#s=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#n&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let i=new Event(e,{detail:t});this.#i().dispatchEvent(i)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#i().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(i){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#p&&(this.debugLog("Emitting event to internal event target",e,t),this.#p.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#c)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#a){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#s.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#u&&(this.#h(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,i){let n=i?.withEventTarget??!1,s=`${this.#t}:${e}`;if(n&&(this.#p||(this.#p=new EventTarget),this.#p.addEventListener(s,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",s),()=>{};let a=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#i().addEventListener(s,a),this.debugLog("Registered event to bus",s),()=>{n&&this.#p?.removeEventListener(s,a),this.#i().removeEventListener(s,a)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let i=t.detail;this.#t&&i.pluginId!==this.#t||e(i)};return this.#i().addEventListener("tanstack-devtools-global",t),()=>this.#i().removeEventListener("tanstack-devtools-global",t)}};let c=new Map;function p(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let m=new class extends u{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function g(e,t,i){let n="object"==typeof e,s=n?e:void 0;return{next:(n?e.next:e)?.bind(s),error:(n?e.error:t)?.bind(s),complete:(n?e.complete:i)?.bind(s)}}let h=[],f=0,{link:b,unlink:x,propagate:v,checkDirty:_,shallowPropagate:y}=function({update:e,notify:t,unwatched:i}){return{link:function(e,t,i){let n=t.depsTail;if(void 0!==n&&n.dep===e)return;let s=void 0!==n?n.nextDep:t.deps;if(void 0!==s&&s.dep===e){s.version=i,t.depsTail=s;return}let a=e.subsTail;if(void 0!==a&&a.version===i&&a.sub===t)return;let r=t.depsTail=e.subsTail={version:i,dep:e,sub:t,prevDep:n,nextDep:s,prevSub:a,nextSub:void 0};void 0!==s&&(s.prevDep=r),void 0!==n?n.nextDep=r:t.deps=r,void 0!==a?a.nextSub=r:e.subs=r},unlink:function(e,t=e.sub){let n=e.dep,s=e.prevDep,a=e.nextDep,r=e.nextSub,o=e.prevSub;return void 0!==a?a.prevDep=s:t.depsTail=s,void 0!==s?s.nextDep=a:t.deps=a,void 0!==r?r.prevSub=o:n.subsTail=o,void 0!==o?o.nextSub=r:void 0===(n.subs=r)&&i(n),a},propagate:function(e){let i,n=e.nextSub;e:for(;;){let s=e.sub,a=s.flags;if(60&a?12&a?4&a?!(48&a)&&function(e,t){let i=t.depsTail;for(;void 0!==i;){if(i===e)return!0;i=i.prevDep}return!1}(e,s)?(s.flags=40|a,a&=1):a=0:s.flags=-9&a|32:a=0:s.flags=32|a,2&a&&t(s),1&a){let t=s.subs;if(void 0!==t){let s=(e=t).nextSub;void 0!==s&&(i={value:n,prev:i},n=s);continue}}if(void 0!==(e=n)){n=e.nextSub;continue}for(;void 0!==i;)if(e=i.value,i=i.prev,void 0!==e){n=e.nextSub;continue e}break}},checkDirty:function(t,i){let s,a=0,r=!1;e:for(;;){let o=t.dep,l=o.flags;if(16&i.flags)r=!0;else if((17&l)==17){if(e(o)){let e=o.subs;void 0!==e.nextSub&&n(e),r=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(s={value:t,prev:s}),t=o.deps,i=o,++a;continue}if(!r){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;a--;){let a=i.subs,o=void 0!==a.nextSub;if(o?(t=s.value,s=s.prev):t=a,r){if(e(i)){o&&n(a),i=t.sub;continue}r=!1}else i.flags&=-33;i=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return r}},shallowPropagate:n};function n(e){do{let i=e.sub,n=i.flags;(48&n)==32&&(i.flags=16|n,(6&n)==2&&t(i))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){h[w++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,E(e))}}),k=0,w=0;function E(e){let t=e.depsTail,i=void 0!==t?t.nextDep:e.deps;for(;void 0!==i;)i=x(i,e)}var j=class{constructor(e,i){this.atom=function(e){let i="function"==typeof e,n={_snapshot:i?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!i,get:()=>(void 0!==t&&b(n,t,f),n._snapshot),subscribe(e){var i;let s,a,r=g(e),o={current:!1},l=(i=()=>{n.get(),o.current?r.next?.(n._snapshot):o.current=!0},s=()=>{let e=t;t=a,++f,a.depsTail=void 0,a.flags=6;try{return i()}finally{t=e,a.flags&=-5,E(a)}},a={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&_(this.deps,this)?s():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,E(this)}},s(),a);return{unsubscribe:()=>{l.stop()}}},_update(s){let a=t,r=(void 0)??Object.is;if(i)t=n,++f,n.depsTail=void 0;else if(void 0===s)return!1;i&&(n.flags=5);try{let t=n._snapshot,a="function"==typeof s?s(t):void 0===s&&i?e(t):s;if(void 0===t||!r(t,a))return n._snapshot=a,!0;return!1}finally{t=a,i&&(n.flags&=-5),E(n)}}};return i?(n.flags=17,n.get=function(){let e=n.flags;if(16&e||32&e&&_(n.deps,n)){if(n._update()){let e=n.subs;void 0!==e&&y(e)}}else 32&e&&(n.flags=-33&e);return void 0!==t&&b(n,t,f),n._snapshot}):n.set=function(e){if(n._update(e)){let e=n.subs;if(void 0!==e&&(v(e),y(e),1)){for(;k<w;){let e=h[k];h[k++]=void 0,e.notify()}k=0,w=0}}},n}(e),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),i&&(this.actions=i(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(g(e))}};function C(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let S={enabled:!0,leading:!1,trailing:!0,wait:0};var T=class{#f;constructor(e,t){this.fn=e,this.store=new j(C()),this.setOptions=e=>{this.options={...this.options,...e},this.#b()||this.cancel()},this.#x=e=>{this.store.setState(t=>{let i={...t,...e},{isPending:n}=i;return{...i,status:this.#b()?n?"pending":"idle":"disabled"}}),((e,t)=>{let i=t.key;if(i){var n,s;c.set(i,t),m.emit(e,{key:(n={...t,key:i}).key,store:{state:p("function"==typeof(s=n.store).get?s.get():s.state)},options:p(n.options)})}})("Debouncer",this)},this.#b=()=>!!d(this.options.enabled,this),this.#v=()=>d(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#b())return;this.#x({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#x({canLeadingExecute:!1}),t=!0,this.#_(...e)),this.options.trailing&&this.#x({isPending:!0,lastArgs:e}),this.#f&&clearTimeout(this.#f),this.#f=setTimeout(()=>{this.#x({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#_(...e)},this.#v())},this.#_=(...e)=>{this.#b()&&(this.fn(...e),this.#x({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#y(),this.#_(...this.store.state.lastArgs))},this.#y=()=>{this.#f&&(clearTimeout(this.#f),this.#f=void 0)},this.cancel=()=>{this.#y(),this.#x({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#x(C())},this.key=t.key,this.options={...S,...t},this.#x(this.options.initialState??{}),this.key&&m.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#x(e.payload.store.state),this.setOptions(e.payload.options))})}#x;#b;#v;#_;#y};e.s(["useDebouncer",0,function(e,t,a=()=>({})){let r={...((0,i.useContext)(n)?.defaultOptions??{}).debouncer,...t},[o]=(0,i.useState)(()=>{let t=new T(e,r);return t.Subscribe=function(e){let i=l(t.store,e.selector,{compare:s});return"function"==typeof e.children?e.children(i):e.children},t});o.fn=e,o.setOptions(r),(0,i.useEffect)(()=>()=>{r.onUnmount?r.onUnmount(o):o.cancel()},[]);let d=l(o.store,a,{compare:s});return(0,i.useMemo)(()=>({...o,state:d}),[o,d])}],540626)},373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t])},541071,e=>{"use strict";var t=e.i(373488);e.s(["MoreHorizontal",()=>t.default])},546467,e=>{"use strict";let t=(0,e.i(475254).default)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["default",0,t])},778917,e=>{"use strict";var t=e.i(546467);e.s(["ExternalLink",()=>t.default])},350682,991799,897250,e=>{"use strict";var t=e.i(475254);let i=(0,t.default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]);e.s(["Github",0,i],350682);let n=(0,t.default)("moon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]);e.s(["Moon",0,n],991799);let s=(0,t.default)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);e.s(["Sun",0,s],897250)},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},972518,799647,e=>{"use strict";var t=e.i(475254);let i=(0,t.default)("panel-left-close",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m16 15-3-3 3-3",key:"14y99z"}]]);e.s(["PanelLeftClose",0,i],972518);let n=(0,t.default)("panel-left-open",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m14 9 3 3-3 3",key:"8010ee"}]]);e.s(["PanelLeftOpen",0,n],799647)},45483,e=>{e.q("/litellm-asset-prefix/_next/static/media/moyai-head.3-tcfui7g9zwr.svg")},492941,e=>{"use strict";var t=e.i(602869),i=e.i(266027);async function n(){let e=(0,t.getProxyBaseUrl)(),i=await fetch(`${e}/public/litellm_blog_posts`);if(!i.ok)throw Error(`Failed to fetch blog posts: ${i.statusText}`);return i.json()}e.s(["useBlogPosts",0,()=>(0,i.useQuery)({queryKey:["blogPosts"],queryFn:n,staleTime:36e5,retry:1,retryDelay:0})])},198458,e=>{"use strict";var t=e.i(655063),i=e.i(266027),n=e.i(271645),s=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:a,fetchPage:r,serializeFilters:o,defaultSorting:l,defaultPageSize:d,enabled:u}=e,[c,p]=(0,n.useState)(l),[m,g]=(0,n.useState)({pageIndex:0,pageSize:d}),[h,f]=(0,n.useState)([]),[b,x]=(0,n.useState)(""),[v]=(0,t.useDebouncedValue)(b,{wait:s.DEBOUNCE_WAIT_MS}),_=(0,n.useMemo)(()=>{let e=c.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=v.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(h)}},[c,m.pageIndex,m.pageSize,v,h,o]),y={queryKey:[...a,_],queryFn:({signal:e})=>r(_,e),enabled:u,placeholderData:e=>e},{data:k,isLoading:w,isPlaceholderData:E,isFetching:j,error:C,refetch:S}=(0,i.useQuery)(y),T=(0,n.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),N=(0,n.useCallback)(e=>{p(e),T()},[T]),I=(0,n.useCallback)(e=>{f(e),T()},[T]),L=(0,n.useCallback)(e=>{x(e),T()},[T]),$=(0,n.useCallback)(()=>{S()},[S]);return{rows:(0,n.useMemo)(()=>k?.data??[],[k]),rowCount:k?.meta.total_count??0,isLoading:w||E,isFetching:j,error:C,refetch:$,sorting:c,onSortingChange:N,pagination:m,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:I,searchValue:b,onSearchChange:L}}])},731565,e=>{"use strict";var t=e.i(115571),i=e.i(271645);function n(e){let i=t=>{"disableBlogPosts"===t.key&&e()},n=t=>{let{key:i}=t.detail;"disableBlogPosts"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t.LOCAL_STORAGE_EVENT,n),()=>{window.removeEventListener("storage",i),window.removeEventListener(t.LOCAL_STORAGE_EVENT,n)}}function s(){return"true"===(0,t.getLocalStorageItem)("disableBlogPosts")}e.s(["useDisableBlogPosts",0,function(){return(0,i.useSyncExternalStore)(n,s,()=>!1)}])},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let i,{apiKeySource:n,accessToken:s,apiKey:a,inputMessage:r,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,x="session"===n?s:a,v=window.location.origin,_=f?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?v=_:f?.PROXY_BASE_URL&&(v=f.PROXY_BASE_URL);let y=r||"Your prompt here",k=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),d.length>0&&(E.vector_stores=d),u.length>0&&(E.guardrails=u),c.length>0&&(E.policies=c);let j=g||"your-model-name",C=b&&Object.keys(b).length>0?`,
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
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:y}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${j}",
    messages=${JSON.stringify(n,null,4)}${t}
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
#                     "text": "${k}"
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
    extra_body=${e}`}let n=w.length>0?w:[{role:"user",content:y}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${j}",
    input=${JSON.stringify(n,null,4)}${t}
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
#                 {"type": "input_text", "text": "${k}"},
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
	model="${j}",
	prompt="${r}",
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
prompt = "${k}"

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
prompt = "${k}"

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
prompt = "${k}"

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
`;break;case t.EndpointType.EMBEDDINGS:i=`
response = client.embeddings.create(
	input="${r||"Your string here"}",
	model="${j}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${j}",
	file=audio_file${r?`,
	prompt="${r.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${j}",
	input="${r||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${j}",
#     input="${r||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${i}`}])},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),n=e.i(871689),s=e.i(643531),a=e.i(174886),r=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,d=e=>e.trim().replace(/\/+$/,""),u=/\.(md|markdown|txt|json|ya?ml|toml)$/i,c=/\.zip$/i,p=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,g=/^[A-Za-z0-9-]+$/,h=/^[A-Za-z0-9._-]+$/,f=/^https?:\/\//i,b="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,v=e=>e.pathname.split("/").filter(e=>""!==e),_=e=>{try{return new URL(e)}catch{return null}},y=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),k=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},w=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),E=(e,t,i,n)=>{let s=d(n??"");return""!==s?l.test(s)?{parsed:{source:"git-subdir",url:t,path:s},label:`${e} subdir — ${t} @ ${s}`,suggestedName:w(k(s))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:w(i)}},j=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),C=e=>`/plugin install ${e.name}@litellm`,S=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",T=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&f.test(e.url)?e.url:null;e.s(["buildMarketplaceSettingsSnippet",0,j,"formatInstallCommand",0,C,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,S,"getSourceLink",0,T,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSha256",0,e=>""===e.trim()||p.test(e.trim()),"isValidSubPath",0,e=>{let t=d(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let i=((e,t)=>{let i=e.trim(),n=x.exec(i),s=n?`${b}${n[1]}@${n[2]}/${n[3]}`:i;if(!s.toLowerCase().startsWith(b))return null;let a=_(s);if(!a||""===a.username||""!==a.password||!y(a))return null;let r=s.indexOf("/",b.length);return -1===r||a.pathname!==s.slice(r)||v(a).length<2?null:E("SSH",i,k(a.pathname).replace(/\.git$/i,""),t)})(e,t);if(i)return i;let n=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=_(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&y(i)?i:null})(e);if(!n)return null;if(c.test(n.pathname))return{parsed:{source:"archive",url:n.href},label:`Zip archive — ${n.host}${n.pathname}`,suggestedName:w(k(n.pathname).replace(c,""))};if("github.com"===n.hostname.replace(/^www\./,""))return((e,t)=>{let i=v(e);if(i.length<2)return null;let n=i[0],s=i[1].replace(/\.git$/,"");if(!g.test(n)||!h.test(s))return null;let a=`${n}/${s}`,r=`https://github.com/${a}`,o={parsed:{source:"github",repo:a},label:`GitHub repo — ${a}`,suggestedName:w(s)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=k(e.join("/")),n=u.test(t)?e.slice(0,-1):e;if(0===n.length)return o;let s=d(n.join("/"));return l.test(s)?{parsed:{source:"git-subdir",url:r,path:s},label:`GitHub subdir — ${a} @ ${s}`,suggestedName:w(k(s))}:null}if(2!==i.length)return null;let c=d(t??"");return""!==c?l.test(c)?{parsed:{source:"git-subdir",url:r,path:c},label:`GitHub subdir — ${a} @ ${c}`,suggestedName:w(k(c))}:null:o})(n,t);if(v(n).length<2)return null;let s=k(n.pathname).replace(/\.git$/,"");return E("Git",`${n.protocol}//${n.host}${n.pathname.replace(/\/+$/,"")}`,s,t)},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let N=({source:e})=>{let i=T(e),n=i&&"git-subdir"===e.source&&e.path?`${i}/tree/main/${e.path}`:i;return n?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:n,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[n.replace("https://",""),(0,t.jsx)(r.Link2,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:S(e)})]}):null};e.s(["default",0,({skill:e,onBack:r})=>{let[l,d]=(0,i.useState)("overview"),[u,c]=(0,i.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),c(t),setTimeout(()=>c(null),2e3)},m=C(e),g=j(window.location.origin),h=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:r,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(n.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:h.map((e,i)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(N,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(m,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===u?"text-success":"text-info"),children:["install"===u?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;p(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===u?"text-success":"text-info"),children:["marketplace-cmd"===u?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"marketplace-cmd"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>p(g,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===u?"text-success":"text-info"),children:["settings"===u?(0,t.jsx)(s.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:g})]})]})]})}],652272)},845150,e=>{"use strict";var t=e.i(843476),i=e.i(271645),n=e.i(131792);let s=(e,t)=>{let i=t.trim().toLowerCase();return!i||e.label.toLowerCase().includes(i)||e.value.toLowerCase().includes(i)||(e.description?.toLowerCase().includes(i)??!1)};e.s(["MultiSelect",0,function({id:e,options:a,value:r=[],onValueChange:o,placeholder:l="Select options",emptyText:d="No options found",disabled:u=!1,loading:c=!1,allowCustomValues:p=!1,className:m}){let g=(0,n.useComboboxAnchor)(),[h,f]=(0,i.useState)(""),b=a.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=r.filter(e=>"string"==typeof e&&e.length>0).map(e=>b.find(t=>t.value===e)??{label:e,value:e}),v=h.trim(),_=b.some(e=>e.value.toLowerCase()===v.toLowerCase()),y=p&&v&&!_?[...b,{label:`Create "${v}"`,value:v}]:b;return(0,t.jsxs)(n.Combobox,{multiple:!0,items:y,value:x,onValueChange:e=>{o(Array.from(new Set(p?e.flatMap(e=>r.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),f("")},inputValue:h,onInputValueChange:f,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:s,disabled:u||c,children:[(0,t.jsx)(n.ComboboxChips,{render:(0,t.jsx)("div",{ref:g}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(n.ComboboxValue,{children:i=>(0,t.jsxs)(t.Fragment,{children:[i.map(e=>(0,t.jsx)(n.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(n.ComboboxChipsInput,{id:e,placeholder:c?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),i.length>0&&!u&&!c&&(0,t.jsx)(n.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(n.ComboboxContent,{anchor:g,children:[(0,t.jsx)(n.ComboboxEmpty,{children:d}),(0,t.jsx)(n.ComboboxList,{children:e=>(0,t.jsx)(n.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},699375,e=>{"use strict";var t=e.i(843476),i=e.i(749099),n=e.i(196631);e.s(["Switch",0,function({className:e,size:s="default",...a}){return(0,t.jsx)(i.Switch.Root,{"data-slot":"switch","data-size":s,className:(0,n.cn)("peer group/switch relative inline-flex shrink-0 items-center rounded-full border border-transparent shadow-xs transition-all outline-none after:absolute after:-inset-x-3 after:-inset-y-2 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 data-[size=default]:h-[18.4px] data-[size=default]:w-[32px] data-[size=sm]:h-[14px] data-[size=sm]:w-[24px] dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 data-checked:bg-primary data-unchecked:bg-input dark:data-unchecked:bg-input/80 data-disabled:cursor-not-allowed data-disabled:opacity-50",e),...a,children:(0,t.jsx)(i.Switch.Thumb,{"data-slot":"switch-thumb",className:"pointer-events-none block rounded-full bg-background ring-0 transition-transform group-data-[size=default]/switch:size-4 group-data-[size=sm]/switch:size-3 group-data-[size=default]/switch:data-checked:translate-x-[calc(100%-2px)] group-data-[size=sm]/switch:data-checked:translate-x-[calc(100%-2px)] dark:data-checked:bg-primary-foreground group-data-[size=default]/switch:data-unchecked:translate-x-0 group-data-[size=sm]/switch:data-unchecked:translate-x-0 dark:data-unchecked:bg-foreground"})})}])},283713,e=>{"use strict";var t=e.i(271645),i=e.i(602869),n=e.i(612256);let s="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,n.useUIConfig)(),a=e?.is_control_plane??!1,r=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(s));(0,t.useEffect)(()=>{if(!o||0===r.length)return;let e=r.find(e=>e.worker_id===o);e&&(0,i.switchToWorkerUrl)(e.url)},[o,r]);let d=r.find(e=>e.worker_id===o)??null,u=(0,t.useCallback)(e=>{let t=r.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(s,e),(0,i.switchToWorkerUrl)(t.url))},[r]);return{isControlPlane:a,workers:r,selectedWorkerId:o,selectedWorker:d,selectWorker:u,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(s),(0,i.switchToWorkerUrl)(null)},[])}}])},741466,e=>{"use strict";e.s(["DEBOUNCE_WAIT_MS",0,300])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function i(e,i){let n=t(e);if(""===n)return!0;let s=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!s.some(e=>e.includes(n))||n.split(/\s+/).every(e=>s.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,n){return e.filter(e=>i(t,n(e)))},"matchesSearchTerm",0,i,"rankBySearchRelevance",0,function(e,i,n){let s=t(i);if(""===s)return[...e];let a=e=>{let t=n(e).toLowerCase();return 1e3*(t===s)+100*!!t.startsWith(s)+(1e3-t.length)};return[...e].sort((e,t)=>a(t)-a(e))}])}]);