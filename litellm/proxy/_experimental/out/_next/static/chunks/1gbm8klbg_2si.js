(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,655063,e=>{var t=e.i(540626),r=e.i(271645);e.s(["f",0,function(e,i,n){let[s,a,o]=function(e,i,n){let[s,a]=(0,r.useState)(e),o=(0,t.f)(a,i,n);return[s,o.maybeExecute,o]}(e,i,n);return(0,r.useEffect)(()=>{a(e)},[e,a]),[s,o]}],655063)},540626,e=>{let t;var r=e.i(271645);let i=(0,r.createContext)(null);function n(e,t){if(Object.is(e,t))return!0;if("object"!=typeof e||null===e||"object"!=typeof t||null===t)return!1;if(e instanceof Map&&t instanceof Map){if(e.size!==t.size)return!1;for(let[r,i]of e)if(!t.has(r)||!Object.is(i,t.get(r)))return!1;return!0}if(e instanceof Set&&t instanceof Set){if(e.size!==t.size)return!1;for(let r of e)if(!t.has(r))return!1;return!0}if(e instanceof Date&&t instanceof Date)return e.getTime()===t.getTime();let r=s(e);if(r.length!==s(t).length)return!1;for(let i=0;i<r.length;i++)if(!Object.prototype.hasOwnProperty.call(t,r[i])||!Object.is(e[r[i]],t[r[i]]))return!1;return!0}function s(e){return Object.keys(e).concat(Object.getOwnPropertySymbols(e))}var a=e.i(430224);function o(e,t){return e===t}function l(e,t=e=>e,i){let n=i?.compare??o,s=(0,r.useCallback)(t=>{let{unsubscribe:r}=e.subscribe(t);return r},[e]),u=(0,r.useCallback)(()=>e.get(),[e]);return(0,a.useSyncExternalStoreWithSelector)(s,u,u,t,n)}function u(e,...t){return"function"==typeof e?e(...t):e}var d=class{#e=!0;#t;#r;#i;#n;#s;#a;#o;#l=0;#u=5;#d=!1;#c=!1;#p=null;#f=()=>{this.debugLog("Connected to event bus"),this.#s=!0,this.#d=!1,this.debugLog("Emitting queued events",this.#n),this.#n.forEach(e=>this.emitEventToBus(e)),this.#n=[],this.stopConnectLoop(),this.#r().removeEventListener("tanstack-connect-success",this.#f)};#h=()=>{if(this.#l<this.#u){this.#l++,this.dispatchCustomEvent("tanstack-connect",{});return}this.#r().removeEventListener("tanstack-connect",this.#h),this.#c=!0,this.debugLog("Max retries reached, giving up on connection"),this.stopConnectLoop()};#g=()=>{this.#d||(this.#d=!0,this.#r().addEventListener("tanstack-connect-success",this.#f),this.#h())};constructor({pluginId:e,debug:t=!1,enabled:r=!0,reconnectEveryMs:i=300}){this.#t=e,this.#e=r,this.#r=this.getGlobalTarget,this.#i=t,this.debugLog(" Initializing event subscription for plugin",this.#t),this.#n=[],this.#s=!1,this.#c=!1,this.#a=null,this.#o=i}startConnectLoop(){null!==this.#a||this.#s||(this.debugLog(`Starting connect loop (every ${this.#o}ms)`),this.#a=setInterval(this.#h,this.#o))}stopConnectLoop(){this.#d=!1,null!==this.#a&&(clearInterval(this.#a),this.#a=null,this.#n=[],this.debugLog("Stopped connect loop"))}debugLog(...e){this.#i&&console.log(`🌴 [tanstack-devtools:${this.#t}-plugin]`,...e)}getGlobalTarget(){if("u">typeof globalThis&&globalThis.__TANSTACK_EVENT_TARGET__)return this.debugLog("Using global event target"),globalThis.__TANSTACK_EVENT_TARGET__;if("u">typeof window&&void 0!==window.addEventListener)return this.debugLog("Using window as event target"),window;let e="u">typeof EventTarget?new EventTarget:void 0;return void 0===e||void 0===e.addEventListener?(this.debugLog("No event mechanism available, running in non-web environment"),{addEventListener:()=>{},removeEventListener:()=>{},dispatchEvent:()=>!1}):(this.debugLog("Using new EventTarget as fallback"),e)}getPluginId(){return this.#t}dispatchCustomEventShim(e,t){try{let r=new Event(e,{detail:t});this.#r().dispatchEvent(r)}catch(e){this.debugLog("Failed to dispatch shim event")}}dispatchCustomEvent(e,t){try{this.#r().dispatchEvent(new CustomEvent(e,{detail:t}))}catch(r){this.dispatchCustomEventShim(e,t)}}emitEventToBus(e){this.debugLog("Emitting event to client bus",e),this.dispatchCustomEvent("tanstack-dispatch-event",e)}createEventPayload(e,t){return{type:`${this.#t}:${e}`,payload:t,pluginId:this.#t}}emit(e,t){if(!this.#e)return void this.debugLog("Event bus client is disabled, not emitting event",e,t);if(this.#p&&(this.debugLog("Emitting event to internal event target",e,t),this.#p.dispatchEvent(new CustomEvent(`${this.#t}:${e}`,{detail:this.createEventPayload(e,t)}))),this.#c)return void this.debugLog("Previously failed to connect, not emitting to bus");if(!this.#s){this.debugLog("Bus not available, will be pushed as soon as connected"),this.#n.push(this.createEventPayload(e,t)),"u">typeof CustomEvent&&!this.#d&&(this.#g(),this.startConnectLoop());return}return this.emitEventToBus(this.createEventPayload(e,t))}on(e,t,r){let i=r?.withEventTarget??!1,n=`${this.#t}:${e}`;if(i&&(this.#p||(this.#p=new EventTarget),this.#p.addEventListener(n,e=>{t(e.detail)})),!this.#e)return this.debugLog("Event bus client is disabled, not registering event",n),()=>{};let s=e=>{this.debugLog("Received event from bus",e.detail),t(e.detail)};return this.#r().addEventListener(n,s),this.debugLog("Registered event to bus",n),()=>{i&&this.#p?.removeEventListener(n,s),this.#r().removeEventListener(n,s)}}onAll(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{e(t.detail)};return this.#r().addEventListener("tanstack-devtools-global",t),()=>this.#r().removeEventListener("tanstack-devtools-global",t)}onAllPluginEvents(e){if(!this.#e)return this.debugLog("Event bus client is disabled, not registering event"),()=>{};let t=t=>{let r=t.detail;this.#t&&r.pluginId!==this.#t||e(r)};return this.#r().addEventListener("tanstack-devtools-global",t),()=>this.#r().removeEventListener("tanstack-devtools-global",t)}};let c=new Map;function p(e){if(void 0!==e)try{return JSON.parse(JSON.stringify(e))}catch{return null}}let f=new class extends d{constructor(e){super({pluginId:"pacer",debug:e?.debug,reconnectEveryMs:1e3})}};function h(e,t,r){let i="object"==typeof e,n=i?e:void 0;return{next:(i?e.next:e)?.bind(n),error:(i?e.error:t)?.bind(n),complete:(i?e.complete:r)?.bind(n)}}let g=[],m=0,{link:v,unlink:b,propagate:y,checkDirty:_,shallowPropagate:x}=function({update:e,notify:t,unwatched:r}){return{link:function(e,t,r){let i=t.depsTail;if(void 0!==i&&i.dep===e)return;let n=void 0!==i?i.nextDep:t.deps;if(void 0!==n&&n.dep===e){n.version=r,t.depsTail=n;return}let s=e.subsTail;if(void 0!==s&&s.version===r&&s.sub===t)return;let a=t.depsTail=e.subsTail={version:r,dep:e,sub:t,prevDep:i,nextDep:n,prevSub:s,nextSub:void 0};void 0!==n&&(n.prevDep=a),void 0!==i?i.nextDep=a:t.deps=a,void 0!==s?s.nextSub=a:e.subs=a},unlink:function(e,t=e.sub){let i=e.dep,n=e.prevDep,s=e.nextDep,a=e.nextSub,o=e.prevSub;return void 0!==s?s.prevDep=n:t.depsTail=n,void 0!==n?n.nextDep=s:t.deps=s,void 0!==a?a.prevSub=o:i.subsTail=o,void 0!==o?o.nextSub=a:void 0===(i.subs=a)&&r(i),s},propagate:function(e){let r,i=e.nextSub;e:for(;;){let n=e.sub,s=n.flags;if(60&s?12&s?4&s?!(48&s)&&function(e,t){let r=t.depsTail;for(;void 0!==r;){if(r===e)return!0;r=r.prevDep}return!1}(e,n)?(n.flags=40|s,s&=1):s=0:n.flags=-9&s|32:s=0:n.flags=32|s,2&s&&t(n),1&s){let t=n.subs;if(void 0!==t){let n=(e=t).nextSub;void 0!==n&&(r={value:i,prev:r},i=n);continue}}if(void 0!==(e=i)){i=e.nextSub;continue}for(;void 0!==r;)if(e=r.value,r=r.prev,void 0!==e){i=e.nextSub;continue e}break}},checkDirty:function(t,r){let n,s=0,a=!1;e:for(;;){let o=t.dep,l=o.flags;if(16&r.flags)a=!0;else if((17&l)==17){if(e(o)){let e=o.subs;void 0!==e.nextSub&&i(e),a=!0}}else if((33&l)==33){(void 0!==t.nextSub||void 0!==t.prevSub)&&(n={value:t,prev:n}),t=o.deps,r=o,++s;continue}if(!a){let e=t.nextDep;if(void 0!==e){t=e;continue}}for(;s--;){let s=r.subs,o=void 0!==s.nextSub;if(o?(t=n.value,n=n.prev):t=s,a){if(e(r)){o&&i(s),r=t.sub;continue}a=!1}else r.flags&=-33;r=t.sub;let l=t.nextDep;if(void 0!==l){t=l;continue e}}return a}},shallowPropagate:i};function i(e){do{let r=e.sub,i=r.flags;(48&i)==32&&(r.flags=16|i,(6&i)==2&&t(r))}while(void 0!==(e=e.nextSub))}}({update:e=>e._update(),notify(e){g[E++]=e,e.flags&=-3},unwatched(e){void 0!==e.depsTail&&(e.depsTail=void 0,e.flags=17,S(e))}}),w=0,E=0;function S(e){let t=e.depsTail,r=void 0!==t?t.nextDep:e.deps;for(;void 0!==r;)r=b(r,e)}var k=class{constructor(e,r){let i,n;this.atom=(n={_snapshot:(i="function"==typeof e)?void 0:e,subs:void 0,subsTail:void 0,deps:void 0,depsTail:void 0,flags:+!i,get:()=>(void 0!==t&&v(n,t,m),n._snapshot),subscribe(e){var r;let i,s,a=h(e),o={current:!1},l=(r=()=>{n.get(),o.current?a.next?.(n._snapshot):o.current=!0},i=()=>{let e=t;t=s,++m,s.depsTail=void 0,s.flags=6;try{return r()}finally{t=e,s.flags&=-5,S(s)}},s={deps:void 0,depsTail:void 0,subs:void 0,subsTail:void 0,flags:6,notify(){let e=this.flags;16&e||32&e&&_(this.deps,this)?i():this.flags=2},stop(){this.flags=0,this.depsTail=void 0,S(this)}},i(),s);return{unsubscribe:()=>{l.stop()}}},_update(r){let s=t,a=(void 0)??Object.is;if(i)t=n,++m,n.depsTail=void 0;else if(void 0===r)return!1;i&&(n.flags=5);try{let t=n._snapshot,s="function"==typeof r?r(t):void 0===r&&i?e(t):r;if(void 0===t||!a(t,s))return n._snapshot=s,!0;return!1}finally{t=s,i&&(n.flags&=-5),S(n)}}},i?(n.flags=17,n.get=function(){let e=n.flags;if(16&e||32&e&&_(n.deps,n)){if(n._update()){let e=n.subs;void 0!==e&&x(e)}}else 32&e&&(n.flags=-33&e);return void 0!==t&&v(n,t,m),n._snapshot}):n.set=function(e){if(n._update(e)){let e=n.subs;if(void 0!==e&&(y(e),x(e),1)){for(;w<E;){let e=g[w];g[w++]=void 0,e.notify()}w=0,E=0}}},n),this.get=this.get.bind(this),this.setState=this.setState.bind(this),this.subscribe=this.subscribe.bind(this),r&&(this.actions=r(this))}setState(e){this.atom.set(e)}get state(){return this.atom.get()}get(){return this.state}subscribe(e){return this.atom.subscribe(h(e))}};function $(){return{canLeadingExecute:!0,executionCount:0,isPending:!1,lastArgs:void 0,status:"idle",maybeExecuteCount:0}}let C={enabled:!0,leading:!1,trailing:!0,wait:0};var T=class{#m;constructor(e,t){this.fn=e,this.store=new k($()),this.setOptions=e=>{this.options={...this.options,...e},this.#v()||this.cancel()},this.#b=e=>{this.store.setState(t=>{let r={...t,...e},{isPending:i}=r;return{...r,status:this.#v()?i?"pending":"idle":"disabled"}}),((e,t)=>{let r=t.key;if(r){var i,n;c.set(r,t),f.emit(e,{key:(i={...t,key:r}).key,store:{state:p("function"==typeof(n=i.store).get?n.get():n.state)},options:p(i.options)})}})("Debouncer",this)},this.#v=()=>!!u(this.options.enabled,this),this.#y=()=>u(this.options.wait,this),this.maybeExecute=(...e)=>{if(!this.#v())return;this.#b({maybeExecuteCount:this.store.state.maybeExecuteCount+1});let t=!1;this.options.leading&&this.store.state.canLeadingExecute&&(this.#b({canLeadingExecute:!1}),t=!0,this.#_(...e)),this.options.trailing&&this.#b({isPending:!0,lastArgs:e}),this.#m&&clearTimeout(this.#m),this.#m=setTimeout(()=>{this.#b({canLeadingExecute:!0}),this.options.trailing&&!t&&this.#_(...e)},this.#y())},this.#_=(...e)=>{this.#v()&&(this.fn(...e),this.#b({executionCount:this.store.state.executionCount+1,isPending:!1,lastArgs:void 0}),this.options.onExecute?.(e,this))},this.flush=()=>{this.store.state.isPending&&this.store.state.lastArgs&&(this.#x(),this.#_(...this.store.state.lastArgs))},this.#x=()=>{this.#m&&(clearTimeout(this.#m),this.#m=void 0)},this.cancel=()=>{this.#x(),this.#b({canLeadingExecute:!0,isPending:!1})},this.reset=()=>{this.#b($())},this.key=t.key,this.options={...C,...t},this.#b(this.options.initialState??{}),this.key&&f.on("d-Debouncer",e=>{e.payload.key===this.key&&(this.#b(e.payload.store.state),this.setOptions(e.payload.options))})}#b;#v;#y;#_;#x};e.s(["f",0,function(e,t,s=()=>({})){let a={...((0,r.useContext)(i)?.defaultOptions??{}).debouncer,...t},[o]=(0,r.useState)(()=>{let t=new T(e,a);return t.Subscribe=function(e){let r=l(t.store,e.selector,{compare:n});return"function"==typeof e.children?e.children(r):e.children},t});o.fn=e,o.setOptions(a),(0,r.useEffect)(()=>()=>{a.onUnmount?a.onUnmount(o):o.cancel()},[]);let u=l(o.store,s,{compare:n});return(0,r.useMemo)(()=>({...o,state:u}),[o,u])}],540626)},586448,e=>{var t=e.i(271645),r=e.i(540143),i=e.i(869230),n=e.i(915823),s=e.i(619273);function a(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var o=class extends n.f{#w;#E;#S;#k;#$;#C;#T;#O;#I;#L=[];constructor(e,t,r){super(),this.#w=e,this.#k=r,this.#S=[],this.#$=[],this.#E=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#$.forEach(e=>{e.subscribe(t=>{this.#z(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#$.forEach(e=>{e.destroy()})}setQueries(e,t){this.#S=e,this.#k=t,r.f.batch(()=>{let e=this.#$,t=this.#M(this.#S);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),i=r.map(e=>e.getCurrentResult()),n=e.length!==r.length,o=r.some((t,r)=>t!==e[r]),l=n||o,u=!!l||i.some((e,t)=>{let r=this.#E[t];return!r||!(0,s.U)(e,r)});(l||u)&&(l&&(this.#L=t,this.#$=r),this.#E=i,this.hasListeners()&&(l&&(a(e,r).forEach(e=>{e.destroy()}),a(r,e).forEach(e=>{e.subscribe(t=>{this.#z(e,t)})})),this.#j()))})}getCurrentResult(){return this.#E}getQueries(){return this.#$.map(e=>e.getCurrentQuery())}getObservers(){return this.#$}getOptimisticResult(e,t){let r=this.#M(e),i=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),n=r.map(e=>e.defaultedQueryOptions.queryHash);return[i,e=>this.#P(e??i,t,n),()=>this.#A(i,r)]}#A(e,t){return t.map((r,i)=>{let n=e[i];return r.defaultedQueryOptions.notifyOnChangeProps?n:r.observer.trackResult(n,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#P(e,t,r){if(t){let i=this.#I,n=void 0!==r&&void 0!==i&&(i.length!==r.length||r.some((e,t)=>e!==i[t]));return(!this.#C||this.#E!==this.#O||n||t!==this.#T)&&(this.#T=t,this.#O=this.#E,void 0!==r&&(this.#I=r),this.#C=(0,s.B)(this.#C,t(e))),this.#C}return e}#R(){return this.#k?.combine!==void 0&&this.#$.some((e,t)=>e.options.suspense&&this.#E[t]?.data===void 0)}#M(e){let t=new Map;this.#$.forEach(e=>{let r=e.options.queryHash;if(!r)return;let i=t.get(r);i?i.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let n=this.#w.defaultQueryOptions(e),s=t.get(n.queryHash)?.shift()??new i.f(this.#w,n);r.push({defaultedQueryOptions:n,observer:s})}),r}#z(e,t){let r=this.#$.indexOf(e);if(-1!==r){var i;let e;this.#E=(i=this.#E,(e=i.slice(0))[r]=t,e),this.#j()}}#j(){if(this.hasListeners()){let e=this.#A(this.#E,this.#L),t=this.#R(),i=this.#C,n=t?i:this.#P(e,this.#k?.combine);(t||i!==n)&&r.f.batch(()=>{this.listeners.forEach(e=>{e(this.#E)})})}}},l=e.i(912598),u=e.i(381384),d=e.i(673664),c=e.i(427001),p=e.i(254440);e.s(["f",0,function({queries:e,...n},a){let f=(0,l.d)(a),h=(0,u.f)(),g=(0,d.p)(),m=t.useMemo(()=>e.map(e=>{let t=f.defaultQueryOptions(e);return t._optimisticResults=h?"isRestoring":"optimistic",t}),[e,f,h]);m.forEach(e=>{(0,p.p)(e);let t=f.getQueryCache().get(e.queryHash);(0,c.l)(e,g,t)}),(0,c.E)(g);let[v]=t.useState(()=>new o(f,m,n)),[b,y,_]=v.getOptimisticResult(m,n.combine),x=!h&&!1!==n.subscribed;t.useSyncExternalStore(t.useCallback(e=>x?v.subscribe(r.f.batchCalls(e)):s.z,[v,x]),()=>v.getCurrentResult(),()=>v.getCurrentResult()),t.useEffect(()=>{v.setQueries(m,n)},[m,n,v]);let w=b.some((e,t)=>(0,p.Q)(m[t],e))?b.flatMap((e,t)=>{let r=m[t];if(r&&(0,p.Q)(r,e)){let e=new i.f(f,r);return(0,p.X)(r,e,g)}return[]}):[];if(w.length>0)throw Promise.all(w);let E=b.find((e,t)=>{let r=m[t];return r&&(0,c.P)({result:e,errorResetBoundary:g,throwOnError:r.throwOnError,query:f.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(E?.error)throw E.error;return y(_())}],586448)},810980,e=>{let t=(0,e.i(475254).f)("book-open",[["path",{d:"M12 7v14",key:"1akyts"}],["path",{d:"M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z",key:"ruj8y"}]]);e.s(["f",0,t],810980)},828579,e=>{let t=(0,e.i(475254).f)("boxes",[["path",{d:"M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z",key:"lc1i9w"}],["path",{d:"m7 16.5-4.74-2.85",key:"1o9zyk"}],["path",{d:"m7 16.5 5-3",key:"va8pkn"}],["path",{d:"M7 16.5v5.17",key:"jnp8gn"}],["path",{d:"M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z",key:"8zsnat"}],["path",{d:"m17 16.5-5-3",key:"8arw3v"}],["path",{d:"m17 16.5 4.74-2.85",key:"8rfmw"}],["path",{d:"M17 16.5v5.17",key:"k6z78m"}],["path",{d:"M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z",key:"1xygjf"}],["path",{d:"M12 8 7.26 5.15",key:"1vbdud"}],["path",{d:"m12 8 4.74-2.85",key:"3rx089"}],["path",{d:"M12 13.5V8",key:"1io7kd"}]]);e.s(["f",0,t],828579)},101048,e=>{var t=e.i(123287);e.s(["f",()=>t.f])},280566,e=>{let t=(0,e.i(475254).f)("circle-dashed",[["path",{d:"M10.1 2.182a10 10 0 0 1 3.8 0",key:"5ilxe3"}],["path",{d:"M13.9 21.818a10 10 0 0 1-3.8 0",key:"11zvb9"}],["path",{d:"M17.609 3.721a10 10 0 0 1 2.69 2.7",key:"1iw5b2"}],["path",{d:"M2.182 13.9a10 10 0 0 1 0-3.8",key:"c0bmvh"}],["path",{d:"M20.279 17.609a10 10 0 0 1-2.7 2.69",key:"1ruxm7"}],["path",{d:"M21.818 10.1a10 10 0 0 1 0 3.8",key:"qkgqxc"}],["path",{d:"M3.721 6.391a10 10 0 0 1 2.7-2.69",key:"1mcia2"}],["path",{d:"M6.391 20.279a10 10 0 0 1-2.69-2.7",key:"1fvljs"}]]);e.s(["f",0,t],280566)},828917,e=>{let t=(0,e.i(475254).f)("circle-plus",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M8 12h8",key:"1wcyev"}],["path",{d:"M12 8v8",key:"napkw2"}]]);e.s(["f",0,t],828917)},798031,e=>{let t=(0,e.i(475254).f)("circle-x",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m15 9-6 6",key:"1uzhvr"}],["path",{d:"m9 9 6 6",key:"z0biqf"}]]);e.s(["f",0,t])},832724,e=>{var t=e.i(798031);e.s(["f",()=>t.f])},546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},373264,e=>{let t=(0,e.i(475254).f)("layout-grid",[["rect",{width:"7",height:"7",x:"3",y:"3",rx:"1",key:"1g98yp"}],["rect",{width:"7",height:"7",x:"14",y:"3",rx:"1",key:"6d4xhi"}],["rect",{width:"7",height:"7",x:"14",y:"14",rx:"1",key:"nxv5o0"}],["rect",{width:"7",height:"7",x:"3",y:"14",rx:"1",key:"1bb6yr"}]]);e.s(["f",0,t],373264)},138620,e=>{var t=e.i(722219);e.s(["z",0,t])},198458,e=>{var t=e.i(655063),r=e.i(266027),i=e.i(271645),n=e.i(741466);e.s(["f",0,function(e){let{queryKey:s,fetchPage:a,serializeFilters:o,defaultSorting:l,defaultPageSize:u,enabled:d}=e,[c,p]=(0,i.useState)(l),[f,h]=(0,i.useState)({pageIndex:0,pageSize:u}),[g,m]=(0,i.useState)([]),[v,b]=(0,i.useState)(""),[y]=(0,t.f)(v,{wait:n.f}),_=(0,i.useMemo)(()=>{let e=c.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=y.trim();return{page:f.pageIndex+1,page_size:f.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(g)}},[c,f.pageIndex,f.pageSize,y,g,o]),x={queryKey:[...s,_],queryFn:({signal:e})=>a(_,e),enabled:d,placeholderData:e=>e},{data:w,isLoading:E,isPlaceholderData:S,isFetching:k,error:$,refetch:C}=(0,r.f)(x),T=(0,i.useCallback)(()=>h(e=>({...e,pageIndex:0})),[]),O=(0,i.useCallback)(e=>{p(e),T()},[T]),I=(0,i.useCallback)(e=>{m(e),T()},[T]),L=(0,i.useCallback)(e=>{b(e),T()},[T]),z=(0,i.useCallback)(()=>{C()},[C]);return{rows:(0,i.useMemo)(()=>w?.data??[],[w]),rowCount:w?.meta.total_count??0,isLoading:E||S,isFetching:k,error:$,refetch:z,sorting:c,onSortingChange:O,pagination:f,onPaginationChange:h,columnFilters:g,onColumnFiltersChange:I,searchValue:v,onSearchChange:L}}])},66330,e=>{var t=e.i(735450);e.s(["f",0,(e,r,i,n)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${r}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(n)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(i)},
        "state": "Invoice #4471 from Acme Tools. Total $174.00, due in 30 days.",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
            "total": {"type": "number", "unit": "USD"}
        }
    }
)

print(response.json()["answers"])`;return(0,t._)(e)?`import requests

response = requests.post(
    ${JSON.stringify(`${r}/v1/systemone`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(n)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(i)},
        "state": "I was charged twice for my subscription this month.",
        "questions": {
            "wants_refund": {
                "type": "noul",
                "instructions": "Is the customer asking for money back?"
            }
        }
    }
)

print(response.json()["answers"])`:`import openai

client = openai.OpenAI(
    api_key="your_api_key",
    base_url=${JSON.stringify(r)}  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model=${JSON.stringify(i)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let r,{apiKeySource:i,accessToken:n,apiKey:s,inputMessage:a,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:d,selectedPolicies:c,selectedVoice:p,endpointType:f,selectedModel:h,selectedSdk:g,proxySettings:m,customHeaders:v}=e,b="session"===i?n:s,y=window.location.origin,_=m?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?y=_:m?.PROXY_BASE_URL&&(y=m.PROXY_BASE_URL);let x=a||"Your prompt here",w=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),E=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};l.length>0&&(S.tags=l),u.length>0&&(S.vector_stores=u),d.length>0&&(S.guardrails=d),c.length>0&&(S.policies=c);let k=h||"your-model-name",$=v&&Object.keys(v).length>0?`,
	default_headers=${JSON.stringify(v,null,2).replace(/\n/g,"\n	")}`:"",C="azure"===g?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${$}
)`;switch(f){case t.Q.CHAT:{let e=Object.keys(S).length>0,t="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=E.length>0?E:[{role:"user",content:x}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(i,null,4)}${t}
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
`;break}case t.Q.RESPONSES:{let e=Object.keys(S).length>0,t="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=E.length>0?E:[{role:"user",content:x}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(i,null,4)}${t}
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
#                     "detail": "auto",
#                 },
#             ],
#         }
#     ]${t}
# )
# print(response_with_file.output_text)
`;break}case t.Q.IMAGE:r="azure"===g?`
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
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
`;break;case t.Q.IMAGE_EDITS:r="azure"===g?`
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
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
									"detail": "auto",
							},
							{
									"type": "input_image",
									"image_url": f"data:image/jpeg;base64,{base64_image2}",
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id1,
									"detail": "auto",
							},
							{
									"type": "input_image",
									"file_id": file_id2,
									"detail": "auto",
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
`;break;case t.Q.EMBEDDINGS:r=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:r=`
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
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${r}`}])},209261,e=>{var t=e.i(201787);let r=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,i=e=>e.trim().replace(/\/+$/,""),n=/\.(md|markdown|txt|json|ya?ml|toml)$/i,s=/\.zip$/i,a=/^[0-9a-fA-F]{64}$/,o=/^\d{1,3}(\.\d{1,3}){3}$/,l=/^[A-Za-z0-9-]+$/,u=/^[A-Za-z0-9._-]+$/,d=/^https?:\/\//i,c="ssh://",p=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,f=e=>e.pathname.split("/").filter(e=>""!==e),h=e=>{try{return new URL(e)}catch{return null}},g=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!o.test(e.hostname),m=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},v=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),b=(e,n,s,a)=>{let o=i(a??"");return""!==o?r.test(o)?{parsed:{source:"git-subdir",url:n,path:o},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:n,path:o}),suggestedName:v(m(o))}:null:{parsed:{source:"url",url:n},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:n}),suggestedName:v(s)}},y="<your-litellm-key>",_=e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e);e.s(["i",0,y,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${y}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"v",0,(e,t,r,i)=>{if("archive"!==t.source||!_(e)||!t.sha256||!a.test(t.sha256))return null;let n=`/claude-code/plugins/${e}/archive/${t.sha256}.zip`;try{let e=new URL(`${r.replace(/\/+$/,"")}${n}`,i);return/^https?:$/.test(e.protocol)&&t.url===e.href?n:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&d.test(e.url)?e.url:null,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||a.test(e.trim()),"P",0,e=>{let t=i(e);return""!==t&&r.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,a)=>{let o=((e,t)=>{let r=e.trim(),i=p.exec(r),n=i?`${c}${i[1]}@${i[2]}/${i[3]}`:r;if(!n.toLowerCase().startsWith(c))return null;let s=h(n);if(!s||""===s.username||""!==s.password||!g(s))return null;let a=n.indexOf("/",c.length);return -1===a||s.pathname!==n.slice(a)||f(s).length<2?null:b("SSH",r,m(s.pathname).replace(/\.git$/i,""),t)})(e,a);if(o)return o;let d=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=h(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&g(r)?r:null})(e);if(!d)return null;if(s.test(d.pathname))return{parsed:{source:"archive",url:d.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${d.host}${d.pathname}`}),suggestedName:v(m(d.pathname).replace(s,""))};if("github.com"===d.hostname.replace(/^www\./,""))return((e,s)=>{let a=f(e);if(a.length<2)return null;let o=a[0],d=a[1].replace(/\.git$/,"");if(!l.test(o)||!u.test(d))return null;let c=`${o}/${d}`,p=`https://github.com/${c}`,h={parsed:{source:"github",repo:c},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:c}),suggestedName:v(d)};if(a.length>=4&&("tree"===a[2]||"blob"===a[2])){let e=a.slice(4),s=m(e.join("/")),o=n.test(s)?e.slice(0,-1):e;if(0===o.length)return h;let l=i(o.join("/"));return r.test(l)?{parsed:{source:"git-subdir",url:p,path:l},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:l}),suggestedName:v(m(l))}:null}if(2!==a.length)return null;let g=i(s??"");return""!==g?r.test(g)?{parsed:{source:"git-subdir",url:p,path:g},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:g}),suggestedName:v(m(g))}:null:h})(d,a);if(f(d).length<2)return null;let y=m(d.pathname).replace(/\.git$/,"");return b("Git",`${d.protocol}//${d.host}${d.pathname.replace(/\/+$/,"")}`,y,a)},"e",0,_])},466098,348594,831538,e=>{var t=e.i(586448),r=e.i(602869),i=e.i(271645),n=e.i(198458),s=e.i(735450);let a="mode",o="providers",l="features",u=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],d=(e,t)=>{let r,i;return r=`filter[${e}][in]`,""===(i=u(t).join(","))?[]:[[r,i]]},c=e=>{switch(e.id){case a:return d(e.id,(0,s.q)(u(e.value)));case o:case l:return d(e.id,e.value);default:return[]}},p=e=>Object.fromEntries(e.flatMap(c)),f=(e,t)=>u(e.find(e=>e.id===t)?.value),h=(e,t,r)=>{let i=e.filter(e=>e.id!==t);return(Array.isArray(r)?0===r.length:""===r.trim())?i:[...i,{id:t,value:r}]};e.s(["g",0,l,"e",0,a,"T",0,o,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,f,"$",0,p,"R",0,h],348594);let g="/public/v1/model_hub",m=["publicModelHub","list"],v=[{id:"model_group",desc:!1}],b=async(e,t)=>{try{return await r.Mw.get(g,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,g,"t",0,e=>{let t=(0,n.f)({queryKey:m,fetchPage:b,serializeFilters:p,defaultSorting:v,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:r}=t,s=(0,i.useCallback)((e,t)=>r(r=>h(r,e,t)),[r]),u=(0,i.useCallback)(e=>s(o,e),[s]),d=(0,i.useCallback)(e=>s(a,e),[s]),c=(0,i.useCallback)(e=>s(l,e),[s]);return{...t,providerValues:f(t.columnFilters,o),onProvidersChange:u,modeValues:f(t.columnFilters,a),onModesChange:d,featureValues:f(t.columnFilters,l),onFeaturesChange:c,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let y=["providers","modes","features"];e.s(["f",0,e=>{let[i,n,s]=(0,t.f)({queries:y.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>r.Mw.get(`${g}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:i,modes:n,features:s}}],466098)},82612,e=>{var t=e.i(843476),r=e.i(225913),i=e.i(196631);let n=(0,r.f)("mb-2 flex shrink-0 items-center justify-center [&_svg]:pointer-events-none [&_svg]:shrink-0",{variants:{variant:{default:"bg-transparent",icon:"flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted text-foreground [&_svg:not([class*='size-'])]:size-6"}},defaultVariants:{variant:"default"}});e.s(["e",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"empty",className:(0,i.c)("flex w-full min-w-0 flex-1 flex-col items-center justify-center gap-4 rounded-lg border-dashed p-12 text-center text-balance",e),...r})},"X",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"empty-content",className:(0,i.c)("flex w-full max-w-sm min-w-0 flex-col items-center gap-4 text-sm text-balance",e),...r})},"Y",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"empty-description",className:(0,i.c)("text-sm/relaxed text-muted-foreground [&>a]:underline [&>a]:underline-offset-4 [&>a:hover]:text-primary",e),...r})},"c",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"empty-header",className:(0,i.c)("flex max-w-sm flex-col items-center gap-2",e),...r})},"B",0,function({className:e,variant:r="default",...s}){return(0,t.jsx)("div",{"data-slot":"empty-icon","data-variant":r,className:(0,i.c)(n({variant:r,className:e})),...s})},"p",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"empty-title",className:(0,i.c)("cn-font-heading text-lg font-medium tracking-tight",e),...r})}])},906197,e=>{var t=e.i(843476),r=e.i(176782),i=e.i(911825),n=e.i(225913),s=e.i(196631);e.i(772436);let a=(0,n.f)("group/item flex w-full flex-wrap items-center rounded-md border text-sm transition-colors duration-100 outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [a]:transition-colors [a]:hover:bg-muted",{variants:{variant:{default:"border-transparent",outline:"border-border",muted:"border-transparent bg-muted/50"},size:{default:"gap-3.5 px-4 py-3.5",sm:"gap-2.5 px-3 py-2.5",xs:"gap-2 px-2.5 py-2 in-data-[slot=dropdown-menu-content]:p-0"}},defaultVariants:{variant:"default",size:"default"}}),o=(0,n.f)("flex shrink-0 items-center justify-center gap-2 group-has-data-[slot=item-description]/item:translate-y-0.5 group-has-data-[slot=item-description]/item:self-start [&_svg]:pointer-events-none",{variants:{variant:{default:"bg-transparent",icon:"[&_svg:not([class*='size-'])]:size-4",image:"size-10 overflow-hidden rounded-sm group-data-[size=sm]/item:size-8 group-data-[size=xs]/item:size-6 [&_img]:size-full [&_img]:object-cover"}},defaultVariants:{variant:"default"}});e.s(["M",0,function({className:e,variant:t="default",size:n="default",render:o,...l}){return(0,i.f)({defaultTagName:"div",props:(0,r.w)({className:(0,s.c)(a({variant:t,size:n,className:e}))},l),render:o,state:{slot:"item",variant:t,size:n}})},"P",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"item-actions",className:(0,s.c)("flex items-center gap-2",e),...r})},"H",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"item-content",className:(0,s.c)("flex flex-1 flex-col gap-1 group-data-[size=xs]/item:gap-0 [&+[data-slot=item-content]]:flex-none",e),...r})},"B",0,function({className:e,...r}){return(0,t.jsx)("p",{"data-slot":"item-description",className:(0,s.c)("line-clamp-2 text-left text-sm leading-normal font-normal text-muted-foreground group-data-[size=xs]/item:text-xs [&>a]:underline [&>a]:underline-offset-4 [&>a:hover]:text-primary",e),...r})},"t",0,function({className:e,...r}){return(0,t.jsx)("div",{role:"list","data-slot":"item-group",className:(0,s.c)("group/item-group flex w-full flex-col gap-4 has-data-[size=sm]:gap-2.5 has-data-[size=xs]:gap-2",e),...r})},"Y",0,function({className:e,variant:r="default",...i}){return(0,t.jsx)("div",{"data-slot":"item-media","data-variant":r,className:(0,s.c)(o({variant:r,className:e})),...i})},"$",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"item-title",className:(0,s.c)("line-clamp-1 flex w-fit items-center gap-2 text-sm leading-snug font-medium underline-offset-4",e),...r})}])},759684,e=>{var t,r,i,n,s,a=e.i(843476),o=e.i(271645),l=e.i(667865),u=e.i(439957),d=e.i(733332);let c=o.createContext(void 0);function p(){let e=o.useContext(c);if(void 0===e)throw Error((0,d.f)(53));return e}var f=e.i(552245);let h=((t={}).scrollAreaCornerHeight="--scroll-area-corner-height",t.scrollAreaCornerWidth="--scroll-area-corner-width",t);function g(e,t,r){if(!e)return 0;let i=getComputedStyle(e),n="x"===r?"Inline":"Block";return"x"===r&&"margin"===t?2*parseFloat(i[`${t}InlineStart`]):parseFloat(i[`${t}${n}Start`])+parseFloat(i[`${t}${n}End`])}let m=((r={}).orientation="data-orientation",r.hovering="data-hovering",r.scrolling="data-scrolling",r.hasOverflowX="data-has-overflow-x",r.hasOverflowY="data-has-overflow-y",r.overflowXStart="data-overflow-x-start",r.overflowXEnd="data-overflow-x-end",r.overflowYStart="data-overflow-y-start",r.overflowYEnd="data-overflow-y-end",r);var v=e.i(60837),b=e.i(788015);let y=((i={}).scrolling="data-scrolling",i.hasOverflowX="data-has-overflow-x",i.hasOverflowY="data-has-overflow-y",i.overflowXStart="data-overflow-x-start",i.overflowXEnd="data-overflow-x-end",i.overflowYStart="data-overflow-y-start",i.overflowYEnd="data-overflow-y-end",i),_={hasOverflowX:e=>e?{[y.hasOverflowX]:""}:null,hasOverflowY:e=>e?{[y.hasOverflowY]:""}:null,overflowXStart:e=>e?{[y.overflowXStart]:""}:null,overflowXEnd:e=>e?{[y.overflowXEnd]:""}:null,overflowYStart:e=>e?{[y.overflowYStart]:""}:null,overflowYEnd:e=>e?{[y.overflowYEnd]:""}:null,cornerHidden:()=>null};var x=e.i(647554),w=e.i(172410);let E={x:0,y:0},S={width:0,height:0},k={xStart:!1,xEnd:!1,yStart:!1,yEnd:!1},$={x:!0,y:!0,corner:!0},C=o.forwardRef(function(e,t){let{render:r,className:i,overflowEdgeThreshold:n,style:s,...d}=e,{xStart:p,xEnd:y,yStart:C,yEnd:T}=function(e){if("number"==typeof e){let t=Math.max(0,e);return{xStart:t,xEnd:t,yStart:t,yEnd:t}}return{xStart:Math.max(0,e?.xStart||0),xEnd:Math.max(0,e?.xEnd||0),yStart:Math.max(0,e?.yStart||0),yEnd:Math.max(0,e?.yEnd||0)}}(n),O=(0,b.f)(),I=(0,u.e)(),L=(0,u.e)(),{nonce:z,disableStyleElements:M}=(0,w.f)(),[j,P]=o.useState(!1),[A,R]=o.useState(!1),[N,H]=o.useState(!1),[q,Y]=o.useState(!1),[D,W]=o.useState(!1),[X,U]=o.useState(S),[Q,F]=o.useState(S),[B,V]=o.useState(k),[J,Z]=o.useState($),K=o.useRef(null),G=o.useRef(null),ee=o.useRef(null),et=o.useRef(null),er=o.useRef(null),ei=o.useRef(null),en=o.useRef(null),es=o.useRef(!1),ea=o.useRef(0),eo=o.useRef(0),el=o.useRef(0),eu=o.useRef(0),ed=o.useRef("vertical"),ec=o.useRef(E),ep=(0,l.f)(e=>{let t=e.x-ec.current.x,r=e.y-ec.current.y;ec.current=e,0!==r&&(H(!0),I.start(500,()=>{H(!1)})),0!==t&&(R(!0),L.start(500,()=>{R(!1)}))}),ef=(0,l.f)(e=>{0===e.button&&(es.current=!0,ea.current=e.clientY,eo.current=e.clientX,ed.current=e.currentTarget.getAttribute(m.orientation),G.current&&(el.current=G.current.scrollTop,eu.current=G.current.scrollLeft),er.current&&"vertical"===ed.current&&er.current.setPointerCapture(e.pointerId),ei.current&&"horizontal"===ed.current&&ei.current.setPointerCapture(e.pointerId))}),eh=(0,l.f)(e=>{if(!es.current)return;let t=e.clientY-ea.current,r=e.clientX-eo.current;if(G.current){let i=G.current.scrollHeight,n=G.current.clientHeight,s=G.current.scrollWidth,a=G.current.clientWidth;if(er.current&&ee.current&&"vertical"===ed.current){let r=g(ee.current,"padding","y"),s=g(er.current,"margin","y"),a=er.current.offsetHeight,o=ee.current.offsetHeight-a-r-s;G.current.scrollTop=el.current+t/o*(i-n),e.preventDefault(),H(!0),I.start(500,()=>{H(!1)})}if(ei.current&&et.current&&"horizontal"===ed.current){let t=g(et.current,"padding","x"),i=g(ei.current,"margin","x"),n=ei.current.offsetWidth,o=et.current.offsetWidth-n-t-i;G.current.scrollLeft=eu.current+r/o*(s-a),e.preventDefault(),R(!0),L.start(500,()=>{R(!1)})}}}),eg=(0,l.f)(e=>{es.current=!1,er.current&&"vertical"===ed.current&&er.current.hasPointerCapture(e.pointerId)&&er.current.releasePointerCapture(e.pointerId),ei.current&&"horizontal"===ed.current&&ei.current.hasPointerCapture(e.pointerId)&&ei.current.releasePointerCapture(e.pointerId)});function em(e){Y("touch"===e.pointerType)}function ev(e){em(e),"touch"!==e.pointerType&&P((0,x.I)(K.current,e.target))}let eb=o.useMemo(()=>({scrolling:A||N,hasOverflowX:!J.x,hasOverflowY:!J.y,overflowXStart:B.xStart,overflowXEnd:B.xEnd,overflowYStart:B.yStart,overflowYEnd:B.yEnd,cornerHidden:J.corner}),[A,N,J.x,J.y,J.corner,B]),ey={role:"presentation",onPointerEnter:ev,onPointerMove:ev,onPointerDown:em,onPointerLeave(){P(!1)},style:{position:"relative",[h.scrollAreaCornerHeight]:`${X.height}px`,[h.scrollAreaCornerWidth]:`${X.width}px`}},e_=(0,f.f)("div",e,{state:eb,ref:[t,K],props:[ey,d],stateAttributesMapping:_}),ex=o.useMemo(()=>({handlePointerDown:ef,handlePointerMove:eh,handlePointerUp:eg,handleScroll:ep,cornerSize:X,setCornerSize:U,thumbSize:Q,setThumbSize:F,hasMeasuredScrollbar:D,setHasMeasuredScrollbar:W,touchModality:q,cornerRef:en,scrollingX:A,setScrollingX:R,scrollingY:N,setScrollingY:H,hovering:j,setHovering:P,viewportRef:G,rootRef:K,scrollbarYRef:ee,scrollbarXRef:et,thumbYRef:er,thumbXRef:ei,rootId:O,hiddenState:J,setHiddenState:Z,overflowEdges:B,setOverflowEdges:V,viewportState:eb,overflowEdgeThreshold:{xStart:p,xEnd:y,yStart:C,yEnd:T}}),[ef,eh,eg,ep,X,Q,D,q,A,R,N,H,j,P,O,J,B,eb,p,y,C,T]);return(0,a.jsxs)(c.Provider,{value:ex,children:[!M&&v.f.getElement(z),e_]})});var T=e.i(146376),O=e.i(818663);let I=o.createContext(void 0);var L=e.i(872855),z=e.i(201675);let M=((n={}).scrollAreaOverflowXStart="--scroll-area-overflow-x-start",n.scrollAreaOverflowXEnd="--scroll-area-overflow-x-end",n.scrollAreaOverflowYStart="--scroll-area-overflow-y-start",n.scrollAreaOverflowYEnd="--scroll-area-overflow-y-end",n);var j=e.i(550896);let P=!1,A=o.forwardRef(function(e,t){let{render:r,className:i,style:n,...s}=e,{viewportRef:d,scrollbarYRef:c,scrollbarXRef:h,thumbYRef:m,thumbXRef:b,cornerRef:y,cornerSize:x,setCornerSize:w,setThumbSize:E,rootId:S,setHiddenState:k,hiddenState:$,setHasMeasuredScrollbar:C,handleScroll:A,setHovering:R,setOverflowEdges:N,overflowEdges:H,overflowEdgeThreshold:q,scrollingX:Y,scrollingY:D}=p(),W=(0,L.f)(),X=o.useRef(!0),U=o.useRef([0/0,0/0,0/0,0/0]),Q=(0,u.e)(),F=(0,u.e)(),B=(0,l.f)(()=>{var e;let t,r,i=d.current,n=c.current,s=h.current,a=m.current,o=b.current,l=y.current;if(!i)return;let u=i.scrollHeight,p=i.scrollWidth,f=i.clientHeight,v=i.clientWidth,_=i.scrollTop,S=i.scrollLeft,$=U.current,T=Number.isNaN($[0]);if($[0]=f,$[1]=u,$[2]=v,$[3]=p,T&&C(!0),0===u||0===p)return;let O=(t=(e=i).clientHeight>=e.scrollHeight,{y:t,x:r=e.clientWidth>=e.scrollWidth,corner:t||r}),I=O.y,L=O.x,P=v/p,A=f/u,R=Math.max(0,p-v),H=Math.max(0,u-f),Y=0,D=0;if(!L){let e=0;e="rtl"===W?(0,z.f)(-S,0,R):(0,z.f)(S,0,R),Y=(0,j.k)(e,R),D=R-Y}let X=I?0:(0,z.f)(_,0,H),Q=I?0:(0,j.k)(X,H),F=I?0:H-Q,B=L?0:v,V=I?0:f,J=0,Z=0;L||I||(J=n?.offsetWidth||0,Z=s?.offsetHeight||0);let K=0===x.width&&0===x.height,G=K?J:0,ee=K?Z:0,et=g(s,"padding","x"),er=g(n,"padding","y"),ei=g(o,"margin","x"),en=g(a,"margin","y"),es=B-et-ei,ea=V-er-en,eo=s?Math.min(s.offsetWidth-G,es):es,el=n?Math.min(n.offsetHeight-ee,ea):ea,eu=Math.max(16,eo*P),ed=Math.max(16,el*A);if(E(e=>e.height===ed&&e.width===eu?e:{width:eu,height:ed}),n&&a){let e=n.offsetHeight-ed-er-en,t=u-f,r=Math.min(e,Math.max(0,(0===t?0:_/t)*e));a.style.transform=`translate3d(0,${r}px,0)`}if(s&&o){let e=s.offsetWidth-eu-et-ei,t=p-v,r=0===t?0:S/t,i="rtl"===W?(0,z.f)(r*e,-e,0):(0,z.f)(r*e,0,e);o.style.transform=`translate3d(${i}px,0,0)`}for(let[e,t]of[[M.scrollAreaOverflowXStart,Y],[M.scrollAreaOverflowXEnd,D],[M.scrollAreaOverflowYStart,Q],[M.scrollAreaOverflowYEnd,F]])i.style.setProperty(e,`${t}px`);l&&(L||I?w({width:0,height:0}):L||I||w({width:J,height:Z})),k(e=>{var t,r;return t=e,r=O,t.y===r.y&&t.x===r.x&&t.corner===r.corner?t:r});let ec={xStart:!L&&Y>q.xStart,xEnd:!L&&D>q.xEnd,yStart:!I&&Q>q.yStart,yEnd:!I&&F>q.yEnd};N(e=>e.xStart===ec.xStart&&e.xEnd===ec.xEnd&&e.yStart===ec.yStart&&e.yEnd===ec.yEnd?e:ec)});function V(){X.current=!1}(0,T.f)(()=>{d.current&&(P||O.f.engine.webkit||("u">typeof CSS&&"registerProperty"in CSS&&[M.scrollAreaOverflowXStart,M.scrollAreaOverflowXEnd,M.scrollAreaOverflowYStart,M.scrollAreaOverflowYEnd].forEach(e=>{try{CSS.registerProperty({name:e,syntax:"<length>",inherits:!1,initialValue:"0px"})}catch{}}),P=!0))},[d]),(0,T.f)(()=>{queueMicrotask(B)},[B,$,W,q.xStart,q.xEnd,q.yStart,q.yEnd]),(0,T.f)(()=>{d.current?.matches(":hover")&&R(!0)},[d,R]),(0,T.f)(()=>{let e=d.current;if("u"<typeof ResizeObserver||!e)return;let t=!1,r=new ResizeObserver(()=>{if(!t){t=!0;let r=U.current;if(r[0]===e.clientHeight&&r[1]===e.scrollHeight&&r[2]===e.clientWidth&&r[3]===e.scrollWidth)return}B()});return r.observe(e),F.start(0,()=>{let t=e.getAnimations({subtree:!0});0!==t.length&&Promise.allSettled(t.map(e=>e.finished)).then(B).catch(()=>{})}),()=>{r.disconnect(),F.clear()}},[B,d,F]);let J={role:"presentation",...S&&{"data-id":`${S}-viewport`},tabIndex:$.x&&$.y?-1:0,className:v.f.className,style:{overflow:"scroll"},onScroll(){d.current&&(B(),X.current||A({x:d.current.scrollLeft,y:d.current.scrollTop}),Q.start(100,()=>{X.current=!0}))},onWheel:V,onTouchMove:V,onPointerMove:V,onPointerEnter:V,onKeyDown:V},Z=o.useMemo(()=>({scrolling:Y||D,hasOverflowX:!$.x,hasOverflowY:!$.y,overflowXStart:H.xStart,overflowXEnd:H.xEnd,overflowYStart:H.yStart,overflowYEnd:H.yEnd,cornerHidden:$.corner}),[Y,D,$.x,$.y,$.corner,H]),K=(0,f.f)("div",e,{ref:[t,d],state:Z,props:[J,s],stateAttributesMapping:_}),G=o.useMemo(()=>({computeThumbPosition:B}),[B]);return(0,a.jsx)(I.Provider,{value:G,children:K})});var R=e.i(574735);let N=o.createContext(void 0),H=((s={}).scrollAreaThumbHeight="--scroll-area-thumb-height",s.scrollAreaThumbWidth="--scroll-area-thumb-width",s),q=o.forwardRef(function(e,t){let{render:r,className:i,orientation:n="vertical",keepMounted:s=!1,style:l,...u}=e,{hovering:d,scrollingX:c,scrollingY:m,hiddenState:v,overflowEdges:b,scrollbarYRef:y,scrollbarXRef:w,viewportRef:E,thumbYRef:S,thumbXRef:k,handlePointerDown:$,handlePointerUp:C,handleScroll:T,rootId:O,thumbSize:I,hasMeasuredScrollbar:z}=p(),M={hovering:d,scrolling:{horizontal:c,vertical:m}[n],orientation:n,hasOverflowX:!v.x,hasOverflowY:!v.y,overflowXStart:b.xStart,overflowXEnd:b.xEnd,overflowYStart:b.yStart,overflowYEnd:b.yEnd,cornerHidden:v.corner},j=(0,L.f)(),P=!z&&!s,A="vertical"===n?v.y:v.x,q=s||!A;o.useEffect(()=>{if(!q)return;let e=E.current,t="vertical"===n?y.current:w.current;if(t)return(0,R.f)(t,"wheel",function(r){if(!e||!t||r.ctrlKey)return;let i="horizontal"===n,s=i?"scrollLeft":"scrollTop",a=i?r.deltaX:r.deltaY;if(0===a)return;let o=i?e.scrollWidth-e.clientWidth:e.scrollHeight-e.clientHeight,l=i&&"rtl"===j?-o:0,u=i&&"rtl"===j?0:o,d=e[s];d<=l&&a<0||d>=u&&a>0||(r.preventDefault(),e[s]=Math.min(u,Math.max(l,d+a)),T({x:e.scrollLeft,y:e.scrollTop}))},{passive:!1})},[j,T,n,w,y,q,E]);let Y={...O&&{"data-id":`${O}-scrollbar`},onPointerDown(e){if(0!==e.button)return;let t=(0,x.Z)(e.nativeEvent),r="vertical"===n?S.current:k.current;if(!(r&&(0,x.I)(r,t))&&E.current){if(S.current&&y.current&&"vertical"===n){let t=g(S.current,"margin","y"),r=g(y.current,"padding","y"),i=S.current.offsetHeight,n=y.current.getBoundingClientRect(),s=e.clientY-n.top-i/2-r+t/2,a=E.current.scrollHeight,o=E.current.clientHeight,l=y.current.offsetHeight-i-r-t;E.current.scrollTop=s/l*(a-o)}if(k.current&&w.current&&"horizontal"===n){let t,r=g(k.current,"margin","x"),i=g(w.current,"padding","x"),n=k.current.offsetWidth,s=w.current.getBoundingClientRect(),a=e.clientX-s.left-n/2-i+r/2,o=E.current.scrollWidth,l=E.current.clientWidth,u=a/(w.current.offsetWidth-n-i-r);"rtl"===j?(t=(1-u)*(o-l),E.current.scrollLeft<=0&&(t=-t)):t=u*(o-l),E.current.scrollLeft=t}T({x:E.current.scrollLeft,y:E.current.scrollTop}),$(e)}},onPointerUp:C,onPointerCancel:C,style:{position:"absolute",touchAction:"none",WebkitUserSelect:"none",userSelect:"none",visibility:P?"hidden":void 0,..."vertical"===n&&{top:0,bottom:`var(${h.scrollAreaCornerHeight})`,insetInlineEnd:0,[H.scrollAreaThumbHeight]:`${I.height}px`},..."horizontal"===n&&{insetInlineStart:0,insetInlineEnd:`var(${h.scrollAreaCornerWidth})`,bottom:0,[H.scrollAreaThumbWidth]:`${I.width}px`}}},D=(0,f.f)("div",e,{ref:[t,"vertical"===n?y:w],state:M,props:[Y,u],stateAttributesMapping:_}),W=o.useMemo(()=>({orientation:n}),[n]);return q?(0,a.jsx)(N.Provider,{value:W,children:D}):null}),Y=o.forwardRef(function(e,t){let{render:r,className:i,style:n,...s}=e,{thumbYRef:a,thumbXRef:l,handlePointerDown:u,handlePointerMove:c,handlePointerUp:h,setScrollingX:g,setScrollingY:m,scrollingX:v,scrollingY:b,hasMeasuredScrollbar:y}=p(),{orientation:_}=function(){let e=o.useContext(N);if(void 0===e)throw Error((0,d.f)(54));return e}();function x(e){"vertical"===_&&m(!1),"horizontal"===_&&g(!1),h(e)}return(0,f.f)("div",e,{ref:[t,"vertical"===_?a:l],state:{scrolling:"horizontal"===_?v:b,orientation:_},props:[{onPointerDown:u,onPointerMove:c,onPointerUp:x,onPointerCancel:x,style:{visibility:y?void 0:"hidden",..."vertical"===_&&{height:`var(${H.scrollAreaThumbHeight})`},..."horizontal"===_&&{width:`var(${H.scrollAreaThumbWidth})`}}},s]})}),D=o.forwardRef(function(e,t){let{render:r,className:i,style:n,...s}=e,{cornerRef:a,cornerSize:o,hiddenState:l}=p(),u=(0,f.f)("div",e,{ref:[t,a],props:[{style:{position:"absolute",bottom:0,insetInlineEnd:0,width:o.width,height:o.height}},s]});return l.corner?null:u});e.s(["Corner",0,D,"Root",0,C,"Scrollbar",0,q,"Thumb",0,Y,"Viewport",0,A],236093);var W=e.i(236093),W=W,X=W,U=W,Q=W,F=W,B=e.i(196631);function V({className:e,orientation:t="vertical",...r}){return(0,a.jsx)(Q.Scrollbar,{"data-slot":"scroll-area-scrollbar","data-orientation":t,orientation:t,className:(0,B.c)("flex touch-none p-px transition-colors select-none data-horizontal:h-2.5 data-horizontal:flex-col data-horizontal:border-t data-horizontal:border-t-transparent data-vertical:h-full data-vertical:w-2.5 data-vertical:border-l data-vertical:border-l-transparent",e),...r,children:(0,a.jsx)(F.Thumb,{"data-slot":"scroll-area-thumb",className:"relative flex-1 rounded-full bg-border"})})}e.s(["D",0,function({className:e,children:t,...r}){return(0,a.jsxs)(W.Root,{"data-slot":"scroll-area",className:(0,B.c)("relative",e),...r,children:[(0,a.jsx)(X.Viewport,{"data-slot":"scroll-area-viewport",className:"size-full rounded-[inherit] transition-[color,box-shadow] outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-1",children:t}),(0,a.jsx)(V,{}),(0,a.jsx)(U.Corner,{})]})},"c",0,V],759684)},170962,31052,363785,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(951437),n=e.i(788015),s=e.i(552245),a=e.i(733332);let o=r.createContext(void 0);function l(e=!0){let t=r.useContext(o);if(void 0===t&&!e)throw Error((0,a.f)(7));return t}e.s(["U",0,o,"J",0,l],31052);var u=e.i(540886),d=e.i(348990),c=e.i(675606),p=e.i(380025);let f=r.forwardRef(function(e,a){let{className:o,defaultPressed:f=!1,disabled:h=!1,form:g,onPressedChange:m,pressed:v,render:b,type:y,value:_,nativeButton:x=!0,style:w,...E}=e,S=(0,n.f)(_||void 0),k=l(),$=k?.value??[],C=k?void 0:f,T=(h||k?.disabled)??!1,[O,I]=(0,i.f)({controlled:k?void 0!==S&&$.indexOf(S)>-1:v,default:C,name:"Toggle",state:"pressed"}),{getButtonProps:L,buttonRef:z}=(0,u.f)({disabled:T,native:x}),M={disabled:T,pressed:O},j=[z,a],P=[{"aria-pressed":O,onClick(e){let t=!O,r=(0,c.j)(p.f.none,e.nativeEvent);m?.(t,r),r.isCanceled||(S&&k?.setGroupValue?.(S,t,r),r.isCanceled||I(t))}},E,L],A=(0,s.f)("button",e,{enabled:!k,state:M,ref:j,props:P}),R=r.useMemo(()=>({disabled:T,focusableWhenDisabled:!1}),[T]);return k?(0,t.jsx)(d.f,{tag:"button",render:b,className:o,style:w,metadata:R,state:M,refs:j,props:P}):A});e.s(["f",0,f],363785);var h=e.i(225913),g=e.i(196631);let m=(0,h.f)("group/toggle inline-flex items-center justify-center gap-1 rounded-md text-sm font-medium whitespace-nowrap transition-[color,box-shadow] outline-none hover:bg-muted hover:text-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-destructive/20 aria-pressed:bg-muted dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",{variants:{variant:{default:"bg-transparent",outline:"border border-input bg-transparent shadow-xs hover:bg-muted"},size:{default:"h-9 min-w-9 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",sm:"h-8 min-w-8 px-2.5 has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5",lg:"h-10 min-w-10 px-2.5 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2"}},defaultVariants:{variant:"default",size:"default"}});e.s(["z",0,function({className:e,variant:r="default",size:i="default",...n}){return(0,t.jsx)(f,{"data-slot":"toggle",className:(0,g.c)(m({variant:r,size:i,className:e})),...n})},"F",0,m],170962)},802954,e=>{var t=e.i(618566),r=e.i(438847),i=e.i(271645),n=e.i(759525),s=e.i(782066);e.s(["f",0,function(e,a,o="tab"){let[l,u]=(0,r.o)(o,r.d.withDefault(a)),d=(0,s.I)((0,t.usePathname)()??"")||"home",c=e.find(e=>e===l)??a;return(0,i.useEffect)(()=>{l!==c&&u(null)},[l,c,u]),[c,(0,i.useCallback)(e=>{(0,n.T)({page:d,action:"click",target:`${o}=${e}`}),u(e)},[d,o,u])]}])},735450,e=>{var t=e.i(411382),r=e.i(772325),i=e.i(54105),n=e.i(240192),s=e.i(138620);let a="https://docs.litellm.ai/docs/decisions",o=`${a}#supported-providers`,l=`${a}#self-hosted-laya-and-nimble`,u=new Set(["/v1/decisions","/v1/systemone"]),d=t.z.object({litellm_provider:i.z.string(),mode:i.z.string().nullish(),supported_endpoints:r.z.array(i.z.string()).nullish()}),c="decisions",p=["evaluation",c],f=e=>null!=e&&p.includes(e),h=e=>f(e.mode)||(e.supported_endpoints??[]).some(e=>u.has(e)),g=e=>{let t=n.z.record(i.z.string(),s.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let r=d.safeParse(t);return r.success?[[e,r.data]]:[]}):[]},m={decisions:0,"self-hosted":1,"needs-upgrade":2},v=t.z.object({decision_model_providers:r.z.array(i.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),y=(e,t,r)=>({row:e,routeOrder:m[e.route],rankInProvider:r.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,a,"d",0,c,"Z",0,o,"L",0,l,"B",0,"playground?tab=system-one","F",0,function(e){let t=g(e),r=t.filter(([,e])=>h(e)),i=new Map([...new Set(r.map(([,e])=>e.litellm_provider))].map(e=>{let r=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:r.every(([,e])=>h(e))}]}));return{models:new Set(r.map(([e])=>e)),providers:i}},"W",0,(e,t)=>g(e).filter(([,e])=>h(e)).map(([e,r])=>({model:e,provider:r.litellm_provider,route:t?.has(r.litellm_provider)?"decisions":(r.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(y).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>f(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?p:[e]),"m",0,h,"_",0,f,"r",0,(e,t,r)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||r.some(t=>e.models.has(t)),"P",0,e=>{let t=v.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},759525,e=>{var t=e.i(867271),r=e.i(768371);let i={queryKey:["telemetry","pageNavigationEnabled"],queryFn:async()=>{let{data:e}=await r.N.GET("/telemetry/ui_events/enabled");return e?.enabled??!1},staleTime:3e5,retry:!1};e.s(["F",0,3e5,"T",0,e=>t.queryClient.fetchQuery(i).then(t=>t?r.N.POST("/telemetry/ui_events",{body:e}):void 0).then(()=>void 0).catch(()=>void 0)])},741466,e=>{e.s(["f",0,300])},899426,e=>{let t=e=>e.trim().toLowerCase();function r(e,r){let i=t(e);if(""===i)return!0;let n=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!n.some(e=>e.includes(i))||i.split(/\s+/).every(e=>n.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,i){return e.filter(e=>r(t,i(e)))},"h",0,r,"e",0,function(e,r,i){let n=t(r);if(""===n)return[...e];let s=e=>{let t=i(e).toLowerCase();return 1e3*(t===n)+100*!!t.startsWith(n)+(1e3-t.length)};return[...e].sort((e,t)=>s(t)-s(e))}])}])})();