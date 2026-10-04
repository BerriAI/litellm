(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,416224,353155,e=>{"use strict";var t=e.i(989257);let r=new Map;e.s(["formatNumber",0,function(e,n,i){return null==e?"":(function(e,n){let i=JSON.stringify({locale:(0,t.stringifyLocale)(e),options:n}),a=r.get(i);if(a)return a;let s=new Intl.NumberFormat(e,n);return r.set(i,s),s})(n,i).format(e)}],416224),e.s(["valueToPercent",0,function(e,t,r){return(e-t)*100/(r-t)}],353155)},586448,e=>{"use strict";var t=e.i(271645),r=e.i(540143),n=e.i(869230),i=e.i(915823),a=e.i(619273);function s(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var o=class extends i.Subscribable{#e;#t;#r;#n;#i;#a;#s;#o;#l;#u=[];constructor(e,t,r){super(),this.#e=e,this.#n=r,this.#r=[],this.#i=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#i.forEach(e=>{e.subscribe(t=>{this.#d(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#i.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#n=t,r.notifyManager.batch(()=>{let e=this.#i,t=this.#c(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),n=r.map(e=>e.getCurrentResult()),i=e.length!==r.length,o=r.some((t,r)=>t!==e[r]),l=i||o,u=!!l||n.some((e,t)=>{let r=this.#t[t];return!r||!(0,a.shallowEqualObjects)(e,r)});(l||u)&&(l&&(this.#u=t,this.#i=r),this.#t=n,this.hasListeners()&&(l&&(s(e,r).forEach(e=>{e.destroy()}),s(r,e).forEach(e=>{e.subscribe(t=>{this.#d(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#i.map(e=>e.getCurrentQuery())}getObservers(){return this.#i}getOptimisticResult(e,t){let r=this.#c(e),n=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),i=r.map(e=>e.defaultedQueryOptions.queryHash);return[n,e=>this.#m(e??n,t,i),()=>this.#f(n,r)]}#f(e,t){return t.map((r,n)=>{let i=e[n];return r.defaultedQueryOptions.notifyOnChangeProps?i:r.observer.trackResult(i,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,r){if(t){let n=this.#l,i=void 0!==r&&void 0!==n&&(n.length!==r.length||r.some((e,t)=>e!==n[t]));return(!this.#a||this.#t!==this.#o||i||t!==this.#s)&&(this.#s=t,this.#o=this.#t,void 0!==r&&(this.#l=r),this.#a=(0,a.replaceEqualDeep)(this.#a,t(e))),this.#a}return e}#g(){return this.#n?.combine!==void 0&&this.#i.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#c(e){let t=new Map;this.#i.forEach(e=>{let r=e.options.queryHash;if(!r)return;let n=t.get(r);n?n.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let i=this.#e.defaultQueryOptions(e),a=t.get(i.queryHash)?.shift()??new n.QueryObserver(this.#e,i);r.push({defaultedQueryOptions:i,observer:a})}),r}#d(e,t){let r=this.#i.indexOf(e);if(-1!==r){var n;let e;this.#t=(n=this.#t,(e=n.slice(0))[r]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#f(this.#t,this.#u),t=this.#g(),n=this.#a,i=t?n:this.#m(e,this.#n?.combine);(t||n!==i)&&r.notifyManager.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},l=e.i(912598),u=e.i(381384),d=e.i(673664),c=e.i(427001),p=e.i(254440);e.s(["useQueries",0,function({queries:e,...i},s){let m=(0,l.useQueryClient)(s),f=(0,u.useIsRestoring)(),g=(0,d.useQueryErrorResetBoundary)(),h=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=f?"isRestoring":"optimistic",t}),[e,m,f]);h.forEach(e=>{(0,p.ensureSuspenseTimers)(e);let t=m.getQueryCache().get(e.queryHash);(0,c.ensurePreventErrorBoundaryRetry)(e,g,t)}),(0,c.useClearResetErrorBoundary)(g);let[b]=t.useState(()=>new o(m,h,i)),[x,y,_]=b.getOptimisticResult(h,i.combine),v=!f&&!1!==i.subscribed;t.useSyncExternalStore(t.useCallback(e=>v?b.subscribe(r.notifyManager.batchCalls(e)):a.noop,[b,v]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(h,i)},[h,i,b]);let w=x.some((e,t)=>(0,p.shouldSuspend)(h[t],e))?x.flatMap((e,t)=>{let r=h[t];if(r&&(0,p.shouldSuspend)(r,e)){let e=new n.QueryObserver(m,r);return(0,p.fetchOptimistic)(r,e,g)}return[]}):[];if(w.length>0)throw Promise.all(w);let j=x.find((e,t)=>{let r=h[t];return r&&(0,c.getHasError)({result:e,errorResetBoundary:g,throwOnError:r.throwOnError,query:m.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(j?.error)throw j.error;return y(_())}],586448)},245423,e=>{"use strict";let t=(0,e.i(475254).default)("bell",[["path",{d:"M10.268 21a2 2 0 0 0 3.464 0",key:"vwvbt9"}],["path",{d:"M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326",key:"11g9vi"}]]);e.s(["Bell",0,t],245423)},233565,e=>{"use strict";var t=e.i(246349);e.s(["ChevronRightIcon",()=>t.default])},243553,e=>{"use strict";let t=(0,e.i(475254).default)("crown",[["path",{d:"M11.562 3.266a.5.5 0 0 1 .876 0L15.39 8.87a1 1 0 0 0 1.516.294L21.183 5.5a.5.5 0 0 1 .798.519l-2.834 10.246a1 1 0 0 1-.956.734H5.81a1 1 0 0 1-.957-.734L2.02 6.02a.5.5 0 0 1 .798-.519l4.276 3.664a1 1 0 0 0 1.516-.294z",key:"1vdc57"}],["path",{d:"M5 21h14",key:"11awu3"}]]);e.s(["Crown",0,t],243553)},541071,373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t],373488),e.s(["MoreHorizontal",0,t],541071)},546467,e=>{"use strict";let t=(0,e.i(475254).default)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["default",0,t])},778917,e=>{"use strict";var t=e.i(546467);e.s(["ExternalLink",()=>t.default])},350682,e=>{"use strict";let t=(0,e.i(475254).default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]);e.s(["Github",0,t],350682)},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},373264,e=>{"use strict";let t=(0,e.i(475254).default)("layout-grid",[["rect",{width:"7",height:"7",x:"3",y:"3",rx:"1",key:"1g98yp"}],["rect",{width:"7",height:"7",x:"14",y:"3",rx:"1",key:"6d4xhi"}],["rect",{width:"7",height:"7",x:"14",y:"14",rx:"1",key:"nxv5o0"}],["rect",{width:"7",height:"7",x:"3",y:"14",rx:"1",key:"1bb6yr"}]]);e.s(["LayoutGrid",0,t],373264)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},292270,263488,e=>{"use strict";var t=e.i(475254);let r=(0,t.default)("log-out",[["path",{d:"m16 17 5-5-5-5",key:"1bji2h"}],["path",{d:"M21 12H9",key:"dn1m92"}],["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}]]);e.s(["LogOut",0,r],292270);let n=(0,t.default)("mail",[["path",{d:"m22 7-8.991 5.727a2 2 0 0 1-2.009 0L2 7",key:"132q7q"}],["rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",key:"izxlao"}]]);e.s(["Mail",0,n],263488)},972518,799647,731565,e=>{"use strict";var t=e.i(475254);let r=(0,t.default)("panel-left-close",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m16 15-3-3 3-3",key:"14y99z"}]]);e.s(["PanelLeftClose",0,r],972518);let n=(0,t.default)("panel-left-open",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m14 9 3 3-3 3",key:"8010ee"}]]);e.s(["PanelLeftOpen",0,n],799647);var i=e.i(115571),a=e.i(271645);function s(e){let t=t=>{"disableBlogPosts"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableBlogPosts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(i.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(i.LOCAL_STORAGE_EVENT,r)}}function o(){return"true"===(0,i.getLocalStorageItem)("disableBlogPosts")}e.s(["useDisableBlogPosts",0,function(){return(0,a.useSyncExternalStore)(s,o)}],731565)},953651,e=>{"use strict";let t=(0,e.i(475254).default)("server",[["rect",{width:"20",height:"8",x:"2",y:"2",rx:"2",ry:"2",key:"ngkwjq"}],["rect",{width:"20",height:"8",x:"2",y:"14",rx:"2",ry:"2",key:"iecqi9"}],["line",{x1:"6",x2:"6.01",y1:"6",y2:"6",key:"16zg32"}],["line",{x1:"6",x2:"6.01",y1:"18",y2:"18",key:"nzw8ys"}]]);e.s(["default",0,t])},618393,e=>{"use strict";var t=e.i(953651);e.s(["Server",()=>t.default])},522016,(e,t,r)=>{"use strict";e.i(247167),Object.defineProperty(r,"__esModule",{value:!0});var n={default:function(){return b},useLinkStatus:function(){return y}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=e.r(190809),s=e.r(843476),o=a._(e.r(271645)),l=e.r(195057),u=e.r(8372),d=e.r(818581),c=e.r(718967),p=e.r(405550),m=e.r(388540),f=e.r(91949),g=e.r(573668),h=e.r(509396);function b(t){var r;let n,i,a,[b,y]=(0,o.useOptimistic)(f.IDLE_LINK_STATUS),_=(0,o.useRef)(null),{href:v,as:w,children:j,prefetch:E=null,passHref:k,replace:N,shallow:S,scroll:C,onClick:R,onMouseEnter:O,onTouchStart:I,legacyBehavior:L=!1,onNavigate:T,transitionTypes:$,ref:P,unstable_dynamicOnHover:M,...A}=t;n=j,L&&("string"==typeof n||"number"==typeof n)&&(n=(0,s.jsx)("a",{children:n}));let z=o.default.useContext(u.AppRouterContext),D=!1!==E,U=!1===E?"none":!0===E?"full":"auto",B="none"!==U?"auto"===U?h.FetchStrategy.PPR:h.FetchStrategy.Full:h.FetchStrategy.PPR,H="string"==typeof(r=w||v)?r:(0,l.formatUrl)(r);if(L){if(n?.$$typeof===Symbol.for("react.lazy"))throw Object.defineProperty(Error("`<Link legacyBehavior>` received a direct child that is either a Server Component, or JSX that was loaded with React.lazy(). This is not supported. Either remove legacyBehavior, or make the direct child a Client Component that renders the Link's `<a>` tag."),"__NEXT_ERROR_CODE",{value:"E863",enumerable:!1,configurable:!0});i=o.default.Children.only(n)}let G=L?i&&"object"==typeof i&&i.ref:P,q,F=o.default.useCallback(e=>(null!==z&&(_.current=(0,f.mountLinkInstance)(e,H,z,B,D,y,q)),()=>{_.current&&((0,f.unmountLinkForCurrentNavigation)(_.current),_.current=null),(0,f.unmountPrefetchableInstance)(e)}),[D,H,z,B,y,q]),Q={ref:(0,d.useMergedRef)(F,G),onClick(t){L||"function"!=typeof R||R(t),L&&i.props&&"function"==typeof i.props.onClick&&i.props.onClick(t),!z||t.defaultPrevented||function(t,r,n,i,a,s,l,u="none"){if("u">typeof window){let d,{nodeName:c}=t.currentTarget;if("A"===c.toUpperCase()&&((d=t.currentTarget.getAttribute("target"))&&"_self"!==d||t.metaKey||t.ctrlKey||t.shiftKey||t.altKey||t.nativeEvent&&2===t.nativeEvent.which)||t.currentTarget.hasAttribute("download"))return;if(!(0,g.isLocalURL)(r)){i&&(t.preventDefault(),location.replace(r));return}if(t.preventDefault(),s){let e=!1;if(s({preventDefault:()=>{e=!0}}),e)return}let{dispatchNavigateAction:p}=e.r(699781);o.default.startTransition(()=>{p(r,i?"replace":"push",!1===a?m.ScrollBehavior.NoScroll:m.ScrollBehavior.Default,n.current,l,u)})}}(t,H,_,N,C,T,$,U)},onMouseEnter(e){L||"function"!=typeof O||O(e),L&&i.props&&"function"==typeof i.props.onMouseEnter&&i.props.onMouseEnter(e),z&&D&&(0,f.onNavigationIntent)(e.currentTarget,!0===M)},onTouchStart:function(e){L||"function"!=typeof I||I(e),L&&i.props&&"function"==typeof i.props.onTouchStart&&i.props.onTouchStart(e),z&&D&&(0,f.onNavigationIntent)(e.currentTarget,!0===M)}};return(0,c.isAbsoluteUrl)(H)?Q.href=H:L&&!k&&("a"!==i.type||"href"in i.props)||(Q.href=(0,p.addBasePath)(H)),a=L?o.default.cloneElement(i,Q):(0,s.jsx)("a",{...A,...Q,children:n}),(0,s.jsx)(x.Provider,{value:b,children:a})}let x=(0,o.createContext)(f.IDLE_LINK_STATUS),y=()=>(0,o.useContext)(x);("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},818581,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"useMergedRef",{enumerable:!0,get:function(){return i}});let n=e.r(271645);function i(e,t){let r=(0,n.useRef)(null),i=(0,n.useRef)(null);return(0,n.useCallback)(n=>{if(null===n){let e=r.current;e&&(r.current=null,e());let t=i.current;t&&(i.current=null,t())}else e&&(r.current=a(e,n)),t&&(i.current=a(t,n))},[e,t])}function a(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let r=e(t);return"function"==typeof r?r:()=>e(null)}}("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},573668,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"isLocalURL",{enumerable:!0,get:function(){return a}});let n=e.r(718967),i=e.r(652817);function a(e){if(!(0,n.isAbsoluteUrl)(e))return!0;try{let t=(0,n.getLocationOrigin)(),r=new URL(e,t);return r.origin===t&&(0,i.hasBasePath)(r.pathname)}catch(e){return!1}}},998183,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var n={assign:function(){return l},searchParamsToUrlQuery:function(){return a},urlQueryToSearchParams:function(){return o}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});function a(e){let t={};for(let[r,n]of e.entries()){let e=t[r];void 0===e?t[r]=n:Array.isArray(e)?e.push(n):t[r]=[e,n]}return t}function s(e){return"string"==typeof e?e:("number"!=typeof e||isNaN(e))&&"boolean"!=typeof e?"":String(e)}function o(e){let t=new URLSearchParams;for(let[r,n]of Object.entries(e))if(Array.isArray(n))for(let e of n)t.append(r,s(e));else t.set(r,s(n));return t}function l(e,...t){for(let r of t){for(let t of r.keys())e.delete(t);for(let[t,n]of r.entries())e.append(t,n)}return e}},195057,(e,t,r)=>{"use strict";e.i(247167),Object.defineProperty(r,"__esModule",{value:!0});var n={formatUrl:function(){return o},formatWithValidation:function(){return u},urlObjectKeys:function(){return l}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=e.r(190809)._(e.r(998183)),s=/https?|ftp|gopher|file/;function o(e){let{auth:t,hostname:r}=e,n=e.protocol||"",i=e.pathname||"",o=e.hash||"",l=e.query||"",u=!1;t=t?encodeURIComponent(t).replace(/%3A/i,":")+"@":"",e.host?u=t+e.host:r&&(u=t+(~r.indexOf(":")?`[${r}]`:r),e.port&&(u+=":"+e.port)),l&&"object"==typeof l&&(l=String(a.urlQueryToSearchParams(l)));let d=e.search||l&&`?${l}`||"";return n&&!n.endsWith(":")&&(n+=":"),e.slashes||(!n||s.test(n))&&!1!==u?(u="//"+(u||""),i&&"/"!==i[0]&&(i="/"+i)):u||(u=""),o&&"#"!==o[0]&&(o="#"+o),d&&"?"!==d[0]&&(d="?"+d),i=i.replace(/[?#]/g,encodeURIComponent),d=d.replace("#","%23"),`${n}${u}${i}${d}${o}`}let l=["auth","hash","host","hostname","href","path","pathname","port","protocol","query","search","slashes"];function u(e){return o(e)}},718967,(e,t,r)=>{"use strict";e.i(247167),Object.defineProperty(r,"__esModule",{value:!0});var n={DecodeError:function(){return b},MiddlewareNotFoundError:function(){return v},MissingStaticPage:function(){return _},NormalizeError:function(){return x},PageNotFoundError:function(){return y},SP:function(){return g},ST:function(){return h},WEB_VITALS:function(){return a},execOnce:function(){return s},getDisplayName:function(){return c},getLocationOrigin:function(){return u},getURL:function(){return d},isAbsoluteUrl:function(){return l},isResSent:function(){return p},loadGetInitialProps:function(){return f},normalizeRepeatedSlashes:function(){return m},stringifyError:function(){return w}};for(var i in n)Object.defineProperty(r,i,{enumerable:!0,get:n[i]});let a=["CLS","FCP","FID","INP","LCP","TTFB"];function s(e){let t,r=!1;return(...n)=>(r||(r=!0,t=e(...n)),t)}let o=/^[a-zA-Z][a-zA-Z\d+\-.]*?:/,l=e=>{let t=e.charCodeAt(0);return!!(t>=65&&t<=90||t>=97&&t<=122)&&o.test(e)};function u(){let{protocol:e,hostname:t,port:r}=window.location;return`${e}//${t}${r?":"+r:""}`}function d(){let{href:e}=window.location,t=u();return e.substring(t.length)}function c(e){return"string"==typeof e?e:e.displayName||e.name||"Unknown"}function p(e){return e.finished||e.headersSent}function m(e){let t=e.split("?");return t[0].replace(/\\/g,"/").replace(/\/\/+/g,"/")+(t[1]?`?${t.slice(1).join("?")}`:"")}async function f(e,t){let r=t.res||t.ctx&&t.ctx.res;if(!e.getInitialProps)return t.ctx&&t.Component?{pageProps:await f(t.Component,t.ctx)}:{};let n=await e.getInitialProps(t);if(r&&p(r))return n;if(!n)throw Object.defineProperty(Error(`"${c(e)}.getInitialProps()" should resolve to an object. But found "${n}" instead.`),"__NEXT_ERROR_CODE",{value:"E1025",enumerable:!1,configurable:!0});return n}let g="u">typeof performance,h=g&&["mark","measure","getEntriesByName"].every(e=>"function"==typeof performance[e]);class b extends Error{}class x extends Error{}class y extends Error{constructor(e){super(),this.code="ENOENT",this.name="PageNotFoundError",this.message=`Cannot find module for page: ${e}`}}class _ extends Error{constructor(e,t){super(),this.message=`Failed to load static file for page: ${e} ${t}`}}class v extends Error{constructor(){super(),this.code="ENOENT",this.message="Cannot find the middleware module"}}function w(e){return JSON.stringify({message:e.message,stack:e.stack})}},143488,e=>{"use strict";var t=e.i(266027),r=e.i(602869);let n=(0,e.i(243652).createQueryKeys)("healthReadinessDetails"),i=async e=>{let t=(0,r.getProxyBaseUrl)(),n=await fetch(`${t}/health/readiness/details`,{method:"GET",headers:{[(0,r.getGlobalLitellmHeaderName)()]:`Bearer ${e}`,"Content-Type":"application/json"}});if(!n.ok)throw Error(`Failed to fetch health readiness details: ${n.statusText}`);return n.json()};e.s(["useHealthReadinessDetails",0,e=>(0,t.useQuery)({queryKey:n.detail("readiness"),queryFn:()=>i(e),enabled:!!e,staleTime:3e5,retry:!1})])},592392,e=>{"use strict";var t=e.i(602869),r=e.i(266027);let n=(0,e.i(243652).createQueryKeys)("proxySettings"),i={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};function a(e){let i=(0,t.getProxyBaseUrl)();return(0,r.useQuery)({queryKey:[...n.all,i,e],queryFn:()=>{if((0,t.getProxyBaseUrl)()!==i)throw Error("Gateway changed while loading settings.");return e?(0,t.getProxyUISettings)(e):null},enabled:!!e})}e.s(["default",0,function(e){let{data:t}=a(e);return t??i},"useProxySettingsQuery",0,a])},912089,e=>{"use strict";var t=e.i(115571),r=e.i(271645);function n(e){let r=t=>{"disableBouncingIcon"===t.key&&e()},n=t=>{let{key:r}=t.detail;"disableBouncingIcon"===r&&e()};return window.addEventListener("storage",r),window.addEventListener(t.LOCAL_STORAGE_EVENT,n),()=>{window.removeEventListener("storage",r),window.removeEventListener(t.LOCAL_STORAGE_EVENT,n)}}function i(){return"true"===(0,t.getLocalStorageItem)("disableBouncingIcon")}e.s(["useDisableBouncingIcon",0,function(){return(0,r.useSyncExternalStore)(n,i)}])},556260,e=>{"use strict";var t=e.i(271645),r=e.i(602869),n=e.i(115571);function i(e){return window.addEventListener("storage",e),window.addEventListener(n.LOCAL_STORAGE_EVENT,e),()=>{window.removeEventListener("storage",e),window.removeEventListener(n.LOCAL_STORAGE_EVENT,e)}}e.s(["useDisableLiteAdmin",0,function(e){let a=e?`disableLiteAdmin:${JSON.stringify([(0,r.getProxyBaseUrl)(),e])}`:null;return[(0,t.useSyncExternalStore)(i,()=>null!==a&&"true"===(0,n.getLocalStorageItem)(a),()=>!1),e=>{null!==a&&(e?(0,n.setLocalStorageItem)(a,"true"):(0,n.removeLocalStorageItem)(a),(0,n.emitLocalStorageChange)(a))}]}])},636772,e=>{"use strict";var t=e.i(271645),r=e.i(115571);function n(e){let t=t=>{"disableShowPrompts"===t.key&&e()},n=t=>{let{key:r}=t.detail;"disableShowPrompts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(r.LOCAL_STORAGE_EVENT,n),()=>{window.removeEventListener("storage",t),window.removeEventListener(r.LOCAL_STORAGE_EVENT,n)}}function i(){return"true"===(0,r.getLocalStorageItem)("disableShowPrompts")}e.s(["useDisableShowPrompts",0,function(){return(0,t.useSyncExternalStore)(n,i)}])},886400,e=>{"use strict";var t=e.i(602869),r=e.i(268004),n=e.i(321836),i=e.i(592392);async function a(e){if(e)try{await (0,t.sessionLogoutCall)(e)}catch{}(0,r.clearTokenCookies)(),(0,n.clearStoredReturnUrl)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url")}e.s(["revokeSessionAndClearClientState",0,a,"useLogout",0,function(e){let t=(0,i.default)(e);return()=>{a(e).finally(()=>{window.location.href=t.PROXY_LOGOUT_URL||""})}}])},222038,e=>{"use strict";e.s(["navAccountDisplayName",0,function(e,t){let r=e?.trim();if(r)return r;let n=t?.trim();return!n||/^default[_\s-]?user[_\s-]?id$/i.test(n)?"Account":n}])},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let r,{apiKeySource:n,accessToken:i,apiKey:a,inputMessage:s,chatHistory:o,selectedTags:l,selectedVectorStores:u,selectedGuardrails:d,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:f,selectedSdk:g,proxySettings:h,customHeaders:b}=e,x="session"===n?i:a,y=window.location.origin,_=h?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?y=_:h?.PROXY_BASE_URL&&(y=h.PROXY_BASE_URL);let v=s||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),j=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),E={};l.length>0&&(E.tags=l),u.length>0&&(E.vector_stores=u),d.length>0&&(E.guardrails=d),c.length>0&&(E.policies=c);let k=f||"your-model-name",N=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===g?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${N}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${N}
)`;switch(m){case t.EndpointType.CHAT:{let e=Object.keys(E).length>0,t="";if(e){let e=JSON.stringify({metadata:E},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let n=j.length>0?j:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(n,null,4)}${t}
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
    extra_body=${e}`}let n=j.length>0?j:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(n,null,4)}${t}
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
`;break}case t.EndpointType.IMAGE:r="azure"===g?`
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
`;break;case t.EndpointType.IMAGE_EDITS:r="azure"===g?`
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
`;break;case t.EndpointType.EMBEDDINGS:r=`
response = client.embeddings.create(
	input="${s||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${s?`,
	prompt="${s.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:r=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${s||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${r}`}])},865361,e=>{"use strict";var t,r,n=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.COMPLETION="completion",t.RESPONSES="responses",t.IMAGE_EDITS="image_edit",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t.REALTIME="realtime",t),i=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r.INTERACTIONS="interactions",r);let a={image_generation:"image",video_generation:"video",chat:"chat",completion:"chat",responses:"responses",image_edit:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings",realtime:"realtime"},s=e=>Object.values(n).includes(e)?a[e]:"chat";e.s(["EndpointType",()=>i,"getEndpointType",0,s,"isModeCompatibleWithEndpoint",0,(e,t)=>{if(!e)return!0;if(!Object.values(n).includes(e))return!1;let r=s(e);return"responses"===t||"anthropic_messages"===t||"interactions"===t?r===t||"chat"===r:"image_edits"===t?r===t||"image"===r:r===t}])},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),n=e.i(871689),i=e.i(643531),a=e.i(174886),s=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,u=e=>e.trim().replace(/\/+$/,""),d=/\.(md|markdown|txt|json|ya?ml|toml)$/i,c=/\.zip$/i,p=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,f=/^[A-Za-z0-9-]+$/,g=/^[A-Za-z0-9._-]+$/,h=/^https?:\/\//i,b="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,y=e=>e.pathname.split("/").filter(e=>""!==e),_=e=>{try{return new URL(e)}catch{return null}},v=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),w=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},j=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),E=(e,t,r,n)=>{let i=u(n??"");return""!==i?l.test(i)?{parsed:{source:"git-subdir",url:t,path:i},label:`${e} subdir — ${t} @ ${i}`,suggestedName:j(w(i))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:j(r)}},k=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),N=e=>`/plugin install ${e.name}@litellm`,S=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",C=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&h.test(e.url)?e.url:null;e.s(["buildMarketplaceSettingsSnippet",0,k,"formatInstallCommand",0,N,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,S,"getSourceLink",0,C,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSha256",0,e=>""===e.trim()||p.test(e.trim()),"isValidSubPath",0,e=>{let t=u(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let r=((e,t)=>{let r=e.trim(),n=x.exec(r),i=n?`${b}${n[1]}@${n[2]}/${n[3]}`:r;if(!i.toLowerCase().startsWith(b))return null;let a=_(i);if(!a||""===a.username||""!==a.password||!v(a))return null;let s=i.indexOf("/",b.length);return -1===s||a.pathname!==i.slice(s)||y(a).length<2?null:E("SSH",r,w(a.pathname).replace(/\.git$/i,""),t)})(e,t);if(r)return r;let n=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=_(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&v(r)?r:null})(e);if(!n)return null;if(c.test(n.pathname))return{parsed:{source:"archive",url:n.href},label:`Zip archive — ${n.host}${n.pathname}`,suggestedName:j(w(n.pathname).replace(c,""))};if("github.com"===n.hostname.replace(/^www\./,""))return((e,t)=>{let r=y(e);if(r.length<2)return null;let n=r[0],i=r[1].replace(/\.git$/,"");if(!f.test(n)||!g.test(i))return null;let a=`${n}/${i}`,s=`https://github.com/${a}`,o={parsed:{source:"github",repo:a},label:`GitHub repo — ${a}`,suggestedName:j(i)};if(r.length>=4&&("tree"===r[2]||"blob"===r[2])){let e=r.slice(4),t=w(e.join("/")),n=d.test(t)?e.slice(0,-1):e;if(0===n.length)return o;let i=u(n.join("/"));return l.test(i)?{parsed:{source:"git-subdir",url:s,path:i},label:`GitHub subdir — ${a} @ ${i}`,suggestedName:j(w(i))}:null}if(2!==r.length)return null;let c=u(t??"");return""!==c?l.test(c)?{parsed:{source:"git-subdir",url:s,path:c},label:`GitHub subdir — ${a} @ ${c}`,suggestedName:j(w(c))}:null:o})(n,t);if(y(n).length<2)return null;let i=w(n.pathname).replace(/\.git$/,"");return E("Git",`${n.protocol}//${n.host}${n.pathname.replace(/\/+$/,"")}`,i,t)},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let R=({source:e})=>{let r=C(e),n=r&&"git-subdir"===e.source&&e.path?`${r}/tree/main/${e.path}`:r;return n?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:n,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[n.replace("https://",""),(0,t.jsx)(s.Link2,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:S(e)})]}):null};e.s(["default",0,({skill:e,onBack:s})=>{let[l,u]=(0,r.useState)("overview"),[d,c]=(0,r.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),c(t),setTimeout(()=>c(null),2e3)},m=N(e),f=k(window.location.origin),g=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:s,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(n.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>u(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:g.map((e,r)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(R,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(m,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===d?"text-success":"text-info"),children:["install"===d?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"install"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>u("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;p(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===d?"text-success":"text-info"),children:["marketplace-cmd"===d?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"marketplace-cmd"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>p(f,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===d?"text-success":"text-info"),children:["settings"===d?(0,t.jsx)(i.Check,{className:"size-3"}):(0,t.jsx)(a.Copy,{className:"size-3"}),"settings"===d?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:f})]})]})]})}],652272)},67488,e=>{"use strict";var t=e.i(843476),r=e.i(463059),n=e.i(618566),i=e.i(196631);function a(e){let t=(0,n.useRouter)();return r=>{r.metaKey||r.ctrlKey||r.shiftKey||1===r.button||(r.preventDefault(),t.push(e))}}function s({href:e,className:n,children:o}){let l=a(e);return(0,t.jsxs)("a",{href:e,onClick:l,className:(0,i.cn)("group inline-flex min-w-0 max-w-full items-center gap-0.5 font-semibold underline-offset-4 hover:underline",n),children:[(0,t.jsx)("span",{className:"min-w-0 truncate",children:o}),(0,t.jsx)(r.ChevronRight,{className:"size-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground"})]})}e.s(["EntityLink",0,function({href:e,className:r,children:n}){return e?(0,t.jsx)(s,{href:e,className:r,children:n}):(0,t.jsx)("span",{className:(0,i.cn)("inline-block min-w-0 max-w-full truncate font-semibold",r),children:n})},"useEntityLinkClick",0,a])},936557,e=>{"use strict";var t=e.i(843476);e.s([],876013),e.i(876013);var r=e.i(271645),n=e.i(502077),i=e.i(733332);let a=r.createContext(void 0);function s(){let e=r.useContext(a);if(void 0===e)throw Error((0,i.default)(38));return e}var o=e.i(416224),l=e.i(353155),u=e.i(201675),d=e.i(552245);let c=r.forwardRef(function(e,i){let{format:s,getAriaValueText:c,locale:p,max:m=100,min:f=0,value:g,render:h,className:b,children:x,style:y,..._}=e,[v,w]=r.useState(),j=(0,l.valueToPercent)(g,f,m),E=(0,u.clamp)(Number.isNaN(j)?0:j,0,100),k=(0,u.clamp)(Number.isNaN(g)?f:g,f,m),N=s?(0,o.formatNumber)(g,p,s):(0,o.formatNumber)(E/100,p,{style:"percent"}),S=N;c&&(S=c(N,g));let C={"aria-labelledby":v,"aria-valuemax":m,"aria-valuemin":f,"aria-valuenow":k,"aria-valuetext":S,role:"meter",children:(0,t.jsxs)(r.Fragment,{children:[x,(0,t.jsx)("span",{role:"presentation",style:n.visuallyHidden,children:"x"})]})},R=r.useMemo(()=>({formattedValue:N,max:m,min:f,percentageValue:E,setLabelId:w,value:g}),[N,m,f,E,w,g]),O=(0,d.useRenderElement)("div",e,{ref:i,props:[C,_]});return(0,t.jsx)(a.Provider,{value:R,children:O})}),p=r.forwardRef(function(e,t){let{render:r,className:n,style:i,...a}=e;return(0,d.useRenderElement)("div",e,{ref:t,props:a})}),m=r.forwardRef(function(e,t){let{render:r,className:n,style:i,...a}=e,{percentageValue:o}=s();return(0,d.useRenderElement)("div",e,{ref:t,props:[{style:{insetInlineStart:0,height:"inherit",width:`${o}%`}},a]})}),f=r.forwardRef(function(e,t){let{className:r,render:n,children:i,style:a,...o}=e,{value:l,formattedValue:u}=s();return(0,d.useRenderElement)("span",e,{ref:t,props:[{"aria-hidden":!0,children:"function"==typeof i?i(u,l):u},o]})});var g=e.i(757337);let h=r.forwardRef(function(e,t){let{render:r,className:n,style:i,id:a,...o}=e,{setLabelId:l}=s(),u=(0,g.useRegisteredLabelId)(a,l);return(0,d.useRenderElement)("span",e,{ref:t,props:[{id:u,role:"presentation"},o]})});e.s(["Indicator",0,m,"Label",0,h,"Root",0,c,"Track",0,p,"Value",0,f],6256);var b=e.i(6256),b=b,x=e.i(225913),y=e.i(196631);let _=(0,x.cva)("h-full rounded-full transition-[width] duration-300",{variants:{tone:{default:"bg-primary",warning:"bg-warning",over:"bg-destructive"}},defaultVariants:{tone:"default"}}),v=r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(b.Root,{ref:n,"data-slot":"meter",className:(0,y.cn)("flex w-full flex-col gap-1.5",e),...r}));v.displayName="Meter";let w=r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(b.Label,{ref:n,"data-slot":"meter-label",className:(0,y.cn)("text-xs text-muted-foreground",e),...r}));w.displayName="MeterLabel",r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(b.Value,{ref:n,"data-slot":"meter-value",className:(0,y.cn)("text-xs font-medium tabular-nums",e),...r})).displayName="MeterValue";let j=r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(b.Track,{ref:n,"data-slot":"meter-track",className:(0,y.cn)("h-1.5 w-full overflow-hidden rounded-full bg-muted",e),...r}));j.displayName="MeterTrack";let E=r.forwardRef(({className:e,tone:r,...n},i)=>(0,t.jsx)(b.Indicator,{ref:i,"data-slot":"meter-indicator",className:(0,y.cn)(_({tone:r,className:e})),...n}));E.displayName="MeterIndicator",e.s(["Meter",0,v,"MeterIndicator",0,E,"MeterLabel",0,w,"MeterTrack",0,j],936557)},581070,e=>{"use strict";var t=e.i(843476),r=e.i(746798);e.s(["CellTooltip",0,function({content:e,trigger:n}){return(0,t.jsx)(r.TooltipProvider,{delay:300,children:(0,t.jsxs)(r.Tooltip,{children:[(0,t.jsx)(r.TooltipTrigger,{render:n}),(0,t.jsx)(r.TooltipContent,{children:e})]})})}])},112179,e=>{"use strict";var t=e.i(843476),r=e.i(67488),n=e.i(487486),i=e.i(196631),a=e.i(581070);let s={success:"border-success/20 bg-success/10 text-success",error:"border-destructive/20 bg-destructive/10 text-destructive",warning:"border-warning/20 bg-warning/10 text-warning",neutral:"border-border bg-muted text-muted-foreground",info:"border-info/20 bg-info/10 text-info"};function o({href:e,dataTestId:a,className:s,children:l}){let u=(0,r.useEntityLinkClick)(e);return(0,t.jsx)(n.Badge,{variant:"outline","data-testid":a,className:(0,i.cn)("cursor-pointer hover:underline",s),render:(0,t.jsx)("a",{href:e,onClick:u}),children:l})}e.s(["StatusBadge",0,function({tone:e,label:r,tooltip:l,dataTestId:u,className:d,href:c}){let p=(0,i.cn)("whitespace-nowrap font-normal",s[e],d),m=c?(0,t.jsx)(o,{href:c,dataTestId:u,className:p,children:r}):(0,t.jsx)(n.Badge,{variant:"outline","data-testid":u,className:p,children:r});return l?(0,t.jsx)(a.CellTooltip,{content:l,trigger:m}):m}])},799676,e=>{"use strict";var t=e.i(843476);e.s([],704824),e.i(704824);var r=e.i(271645),n=e.i(552245),i=e.i(733332);let a=r.createContext(void 0);function s(){let e=r.useContext(a);if(void 0===e)throw Error((0,i.default)(13));return e}let o={imageLoadingStatus:()=>null},l=r.forwardRef(function(e,i){let{className:s,render:l,style:u,...d}=e,[c,p]=r.useState("idle"),m=r.useMemo(()=>({imageLoadingStatus:c,setImageLoadingStatus:p}),[c,p]),f=(0,n.useRenderElement)("span",e,{state:{imageLoadingStatus:c},ref:i,props:d,stateAttributesMapping:o});return(0,t.jsx)(a.Provider,{value:m,children:f})});var u=e.i(667865),d=e.i(146376),c=e.i(137584),p=e.i(209407),m=e.i(223910),f=e.i(956789);let g={...o,...p.transitionStatusMapping},h=r.forwardRef(function(e,t){let{className:i,render:a,onLoadingStatusChange:o,style:l,...p}=e,{setImageLoadingStatus:h}=s(),b=function(e,{referrerPolicy:t,crossOrigin:n,sizes:i,srcSet:a}){let[s,o]=r.useState("idle");return(0,d.useIsoLayoutEffect)(()=>{if(!e&&!a)return o("error"),f.NOOP;let r=!0,s=new window.Image,l=e=>()=>{r&&o(e)};return o("loading"),s.onload=l("loaded"),s.onerror=l("error"),t&&(s.referrerPolicy=t),s.crossOrigin=n??null,i&&(s.sizes=i),a&&(s.srcset=a),e&&(s.src=e),s.complete&&o(s.naturalWidth>0?"loaded":"error"),()=>{r=!1}},[e,a,i,n,t]),s}(p.src,p),x="loaded"===b,{mounted:y,transitionStatus:_,setMounted:v}=(0,m.useTransitionStatus)(x),w=r.useRef(null),j=(0,u.useStableCallback)(e=>{o?.(e),h(e)});(0,d.useIsoLayoutEffect)(()=>{"idle"!==b&&j(b)},[b,j]),(0,d.useIsoLayoutEffect)(()=>()=>h("idle"),[h]),(0,c.useOpenChangeComplete)({open:x,ref:w,onComplete(){x||v(!1)}});let E=(0,n.useRenderElement)("img",e,{state:{imageLoadingStatus:b,transitionStatus:_},ref:[t,w],props:p,stateAttributesMapping:g,enabled:y});return y?E:null});var b=e.i(439957);let x=r.forwardRef(function(e,t){let{className:i,render:a,delay:l,style:u,...d}=e,{imageLoadingStatus:c}=s(),[p,m]=r.useState(void 0===l),f=(0,b.useTimeout)();return r.useEffect(()=>(void 0!==l?f.start(l,()=>m(!0)):m(!0),f.clear),[f,l]),(0,n.useRenderElement)("span",e,{state:{imageLoadingStatus:c},ref:t,props:d,stateAttributesMapping:o,enabled:"loaded"!==c&&(void 0===l||p)})});e.s(["Fallback",0,x,"Image",0,h,"Root",0,l],514751);var y=e.i(514751),y=y,_=e.i(196631);let v=r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(y.Root,{ref:n,"data-slot":"avatar",className:(0,_.cn)("relative flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full",e),...r}));v.displayName="Avatar",r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(y.Image,{ref:n,"data-slot":"avatar-image",className:(0,_.cn)("size-full object-cover",e),...r})).displayName="AvatarImage";let w=r.forwardRef(({className:e,...r},n)=>(0,t.jsx)(y.Fallback,{ref:n,"data-slot":"avatar-fallback",className:(0,_.cn)("flex size-full items-center justify-center rounded-full text-xs font-medium",e),...r}));w.displayName="AvatarFallback",e.s(["Avatar",0,v,"AvatarFallback",0,w],799676)},755146,e=>{"use strict";var t=e.i(843476),r=e.i(451512),n=e.i(196631);e.i(233565);var i=e.i(678784);e.s(["DropdownMenu",0,function({...e}){return(0,t.jsx)(r.Menu.Root,{"data-slot":"dropdown-menu",...e})},"DropdownMenuCheckboxItem",0,function({className:e,children:a,checked:s,inset:o,...l}){return(0,t.jsxs)(r.Menu.CheckboxItem,{"data-slot":"dropdown-menu-checkbox-item","data-inset":o,className:(0,n.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),checked:s,...l,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-checkbox-item-indicator",children:(0,t.jsx)(r.Menu.CheckboxItemIndicator,{children:(0,t.jsx)(i.CheckIcon,{})})}),a]})},"DropdownMenuContent",0,function({align:e="start",alignOffset:i=0,side:a="bottom",sideOffset:s=4,className:o,...l}){return(0,t.jsx)(r.Menu.Portal,{children:(0,t.jsx)(r.Menu.Positioner,{className:"isolate z-popup outline-none",align:e,alignOffset:i,side:a,sideOffset:s,children:(0,t.jsx)(r.Menu.Popup,{"data-slot":"dropdown-menu-content",className:(0,n.cn)("z-popup max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-md bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95",o),...l})})})},"DropdownMenuItem",0,function({className:e,inset:i,variant:a="default",...s}){return(0,t.jsx)(r.Menu.Item,{"data-slot":"dropdown-menu-item","data-inset":i,"data-variant":a,className:(0,n.cn)("group/dropdown-menu-item relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground not-data-[variant=destructive]:focus:**:text-accent-foreground data-inset:pl-8 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[variant=destructive]:*:[svg]:text-destructive",e),...s})},"DropdownMenuRadioGroup",0,function({...e}){return(0,t.jsx)(r.Menu.RadioGroup,{"data-slot":"dropdown-menu-radio-group",...e})},"DropdownMenuRadioItem",0,function({className:e,children:a,inset:s,...o}){return(0,t.jsxs)(r.Menu.RadioItem,{"data-slot":"dropdown-menu-radio-item","data-inset":s,className:(0,n.cn)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),...o,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-radio-item-indicator",children:(0,t.jsx)(r.Menu.RadioItemIndicator,{children:(0,t.jsx)(i.CheckIcon,{})})}),a]})},"DropdownMenuSeparator",0,function({className:e,...i}){return(0,t.jsx)(r.Menu.Separator,{"data-slot":"dropdown-menu-separator",className:(0,n.cn)("-mx-1 my-1 h-px bg-border",e),...i})},"DropdownMenuTrigger",0,function({...e}){return(0,t.jsx)(r.Menu.Trigger,{"data-slot":"dropdown-menu-trigger",...e})}])},275144,e=>{"use strict";var t=e.i(843476),r=e.i(271645),n=e.i(602869);let i=(0,r.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:a})=>{let[s,o]=(0,r.useState)(null),[l,u]=(0,r.useState)(null),[d,c]=(0,r.useState)(null);return(0,r.useEffect)(()=>{(async()=>{try{let e=(0,n.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",r=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(r.ok){let e=await r.json();e.values?.logo_url&&o(e.values.logo_url),e.values?.logo_url_dark&&u(e.values.logo_url_dark),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,r.useEffect)(()=>{if(d){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=d});else{let e=document.createElement("link");e.rel="icon",e.href=d,document.head.appendChild(e)}}},[d]),(0,t.jsx)(i.Provider,{value:{logoUrl:s,setLogoUrl:o,logoUrlDark:l,setLogoUrlDark:u,faviconUrl:d,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,r.useContext)(i);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},283713,e=>{"use strict";var t=e.i(271645),r=e.i(602869),n=e.i(612256);let i="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,n.useUIConfig)(),a=e?.is_control_plane??!1,s=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(i));(0,t.useEffect)(()=>{if(!o||0===s.length)return;let e=s.find(e=>e.worker_id===o);e&&(0,r.switchToWorkerUrl)(e.url)},[o,s]);let u=s.find(e=>e.worker_id===o)??null,d=(0,t.useCallback)(e=>{let t=s.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(i,e),(0,r.switchToWorkerUrl)(t.url))},[s]);return{isControlPlane:a,workers:s,selectedWorkerId:o,selectedWorker:u,selectWorker:d,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(i),(0,r.switchToWorkerUrl)(null)},[])}}])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function r(e,r){let n=t(e);if(""===n)return!0;let i=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!i.some(e=>e.includes(n))||n.split(/\s+/).every(e=>i.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,n){return e.filter(e=>r(t,n(e)))},"matchesSearchTerm",0,r,"rankBySearchRelevance",0,function(e,r,n){let i=t(r);if(""===i)return[...e];let a=e=>{let t=n(e).toLowerCase();return 1e3*(t===i)+100*!!t.startsWith(i)+(1e3-t.length)};return[...e].sort((e,t)=>a(t)-a(e))}])}]);