(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,871943,502547,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["ChevronDownIcon",0,r],871943);let o=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,o],502547)},360820,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["ChevronUpIcon",0,r],360820)},434626,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,r],434626)},250980,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,r],250980)},278587,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["RefreshIcon",0,r],278587)},68155,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"}))});e.s(["TrashIcon",0,r],68155)},586448,e=>{"use strict";var t=e.i(271645),r=e.i(540143),o=e.i(869230),n=e.i(915823),s=e.i(619273);function i(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var a=class extends n.Subscribable{#e;#t;#r;#o;#n;#s;#i;#a;#l;#c=[];constructor(e,t,r){super(),this.#e=e,this.#o=r,this.#r=[],this.#n=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#n.forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#n.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#o=t,r.notifyManager.batch(()=>{let e=this.#n,t=this.#d(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),o=r.map(e=>e.getCurrentResult()),n=e.length!==r.length,a=r.some((t,r)=>t!==e[r]),l=n||a,c=!!l||o.some((e,t)=>{let r=this.#t[t];return!r||!(0,s.shallowEqualObjects)(e,r)});(l||c)&&(l&&(this.#c=t,this.#n=r),this.#t=o,this.hasListeners()&&(l&&(i(e,r).forEach(e=>{e.destroy()}),i(r,e).forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#n.map(e=>e.getCurrentQuery())}getObservers(){return this.#n}getOptimisticResult(e,t){let r=this.#d(e),o=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),n=r.map(e=>e.defaultedQueryOptions.queryHash);return[o,e=>this.#m(e??o,t,n),()=>this.#h(o,r)]}#h(e,t){return t.map((r,o)=>{let n=e[o];return r.defaultedQueryOptions.notifyOnChangeProps?n:r.observer.trackResult(n,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,r){if(t){let o=this.#l,n=void 0!==r&&void 0!==o&&(o.length!==r.length||r.some((e,t)=>e!==o[t]));return(!this.#s||this.#t!==this.#a||n||t!==this.#i)&&(this.#i=t,this.#a=this.#t,void 0!==r&&(this.#l=r),this.#s=(0,s.replaceEqualDeep)(this.#s,t(e))),this.#s}return e}#g(){return this.#o?.combine!==void 0&&this.#n.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#d(e){let t=new Map;this.#n.forEach(e=>{let r=e.options.queryHash;if(!r)return;let o=t.get(r);o?o.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let n=this.#e.defaultQueryOptions(e),s=t.get(n.queryHash)?.shift()??new o.QueryObserver(this.#e,n);r.push({defaultedQueryOptions:n,observer:s})}),r}#u(e,t){let r=this.#n.indexOf(e);if(-1!==r){var o;let e;this.#t=(o=this.#t,(e=o.slice(0))[r]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#h(this.#t,this.#c),t=this.#g(),o=this.#s,n=t?o:this.#m(e,this.#o?.combine);(t||o!==n)&&r.notifyManager.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},l=e.i(912598),c=e.i(381384),u=e.i(673664),d=e.i(427001),p=e.i(254440);e.s(["useQueries",0,function({queries:e,...n},i){let m=(0,l.useQueryClient)(i),h=(0,c.useIsRestoring)(),g=(0,u.useQueryErrorResetBoundary)(),b=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=h?"isRestoring":"optimistic",t}),[e,m,h]);b.forEach(e=>{(0,p.ensureSuspenseTimers)(e);let t=m.getQueryCache().get(e.queryHash);(0,d.ensurePreventErrorBoundaryRetry)(e,g,t)}),(0,d.useClearResetErrorBoundary)(g);let[f]=t.useState(()=>new a(m,b,n)),[x,k,_]=f.getOptimisticResult(b,n.combine),v=!h&&!1!==n.subscribed;t.useSyncExternalStore(t.useCallback(e=>v?f.subscribe(r.notifyManager.batchCalls(e)):s.noop,[f,v]),()=>f.getCurrentResult(),()=>f.getCurrentResult()),t.useEffect(()=>{f.setQueries(b,n)},[b,n,f]);let w=x.some((e,t)=>(0,p.shouldSuspend)(b[t],e))?x.flatMap((e,t)=>{let r=b[t];if(r&&(0,p.shouldSuspend)(r,e)){let e=new o.QueryObserver(m,r);return(0,p.fetchOptimistic)(r,e,g)}return[]}):[];if(w.length>0)throw Promise.all(w);let y=x.find((e,t)=>{let r=b[t];return r&&(0,d.getHasError)({result:e,errorResetBoundary:g,throwOnError:r.throwOnError,query:m.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(y?.error)throw y.error;return k(_())}],586448)},180127,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["default",0,t])},871689,e=>{"use strict";var t=e.i(180127);e.s(["ArrowLeft",()=>t.default])},541071,e=>{"use strict";var t=e.i(373488);e.s(["MoreHorizontal",()=>t.default])},332102,e=>{"use strict";let t=(0,e.i(475254).default)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["Inbox",0,t],332102)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},16715,e=>{"use strict";let t=(0,e.i(475254).default)("refresh-cw",[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]]);e.s(["RefreshCw",0,t],16715)},455037,e=>{"use strict";var t=e.i(494144);e.s(["prism",()=>t.default])},198458,e=>{"use strict";var t=e.i(655063),r=e.i(266027),o=e.i(271645),n=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:s,fetchPage:i,serializeFilters:a,defaultSorting:l,defaultPageSize:c,enabled:u}=e,[d,p]=(0,o.useState)(l),[m,h]=(0,o.useState)({pageIndex:0,pageSize:c}),[g,b]=(0,o.useState)([]),[f,x]=(0,o.useState)(""),[k]=(0,t.useDebouncedValue)(f,{wait:n.DEBOUNCE_WAIT_MS}),_=(0,o.useMemo)(()=>{let e=d.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=k.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...a(g)}},[d,m.pageIndex,m.pageSize,k,g,a]),v={queryKey:[...s,_],queryFn:({signal:e})=>i(_,e),enabled:u,placeholderData:e=>e},{data:w,isLoading:y,isPlaceholderData:j,isFetching:C,error:N,refetch:E}=(0,r.useQuery)(v),O=(0,o.useCallback)(()=>h(e=>({...e,pageIndex:0})),[]),S=(0,o.useCallback)(e=>{p(e),O()},[O]),T=(0,o.useCallback)(e=>{b(e),O()},[O]),A=(0,o.useCallback)(e=>{x(e),O()},[O]),R=(0,o.useCallback)(()=>{E()},[E]);return{rows:(0,o.useMemo)(()=>w?.data??[],[w]),rowCount:w?.meta.total_count??0,isLoading:y||j,isFetching:C,error:N,refetch:R,sorting:d,onSortingChange:S,pagination:m,onPaginationChange:h,columnFilters:g,onColumnFiltersChange:T,searchValue:f,onSearchChange:A}}])},466828,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(678784);let n=(0,e.i(475254).default)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var s=e.i(650056);let i={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};var a=e.i(488012);e.s(["default",0,({code:e,language:l})=>{let c=(0,a.useSyntaxTheme)(i),[u,d]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-border bg-muted overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),d(!0),setTimeout(()=>d(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground z-raised","aria-label":"Copy code",children:u?(0,t.jsx)(o.CheckIcon,{size:16}):(0,t.jsx)(n,{size:16})}),(0,t.jsx)(s.Prism,{language:l,style:c,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",background:"transparent"},codeTagProps:{style:{background:"transparent"}},showLineNumbers:!0,children:e})]})}],466828)},909947,e=>{"use strict";var t=e.i(865361);e.s(["generateCodeSnippet",0,e=>{let r,{apiKeySource:o,accessToken:n,apiKey:s,inputMessage:i,chatHistory:a,selectedTags:l,selectedVectorStores:c,selectedGuardrails:u,selectedPolicies:d,selectedVoice:p,endpointType:m,selectedModel:h,selectedSdk:g,proxySettings:b,customHeaders:f}=e,x="session"===o?n:s,k=window.location.origin,_=b?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?k=_:b?.PROXY_BASE_URL&&(k=b.PROXY_BASE_URL);let v=i||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),y=a.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};l.length>0&&(j.tags=l),c.length>0&&(j.vector_stores=c),u.length>0&&(j.guardrails=u),d.length>0&&(j.policies=d);let C=h||"your-model-name",N=f&&Object.keys(f).length>0?`,
	default_headers=${JSON.stringify(f,null,2).replace(/\n/g,"\n	")}`:"",E="azure"===g?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${k}",
	api_version="2024-02-01"${N}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${k}"${N}
)`;switch(m){case t.EndpointType.CHAT:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let o=y.length>0?y:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(o,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${C}",
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
`;break}case t.EndpointType.RESPONSES:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let o=y.length>0?y:[{role:"user",content:v}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(o,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${C}",
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
	model="${C}",
	prompt="${i}",
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
	model="${C}",
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
	model="${C}",
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
	model="${C}",
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
	input="${i||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.EndpointType.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${i?`,
	prompt="${i.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.EndpointType.SPEECH:r=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${i||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${i||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${E}
${r}`}])},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(871689),n=e.i(643531),s=e.i(174886),i=e.i(306228),a=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,c=e=>e.trim().replace(/\/+$/,""),u=/\.(md|markdown|txt|json|ya?ml|toml)$/i,d=/\.zip$/i,p=/^[0-9a-fA-F]{64}$/,m=/^\d{1,3}(\.\d{1,3}){3}$/,h=/^[A-Za-z0-9-]+$/,g=/^[A-Za-z0-9._-]+$/,b=/^https?:\/\//i,f="ssh://",x=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,k=e=>e.pathname.split("/").filter(e=>""!==e),_=e=>{try{return new URL(e)}catch{return null}},v=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!m.test(e.hostname),w=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},y=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),j=(e,t,r,o)=>{let n=c(o??"");return""!==n?l.test(n)?{parsed:{source:"git-subdir",url:t,path:n},label:`${e} subdir — ${t} @ ${n}`,suggestedName:y(w(n))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:y(r)}},C=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),N=e=>`/plugin install ${e.name}@litellm`,E=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",O=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&b.test(e.url)?e.url:null;e.s(["buildMarketplaceSettingsSnippet",0,C,"formatInstallCommand",0,N,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,E,"getSourceLink",0,O,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSha256",0,e=>""===e.trim()||p.test(e.trim()),"isValidSubPath",0,e=>{let t=c(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let r=((e,t)=>{let r=e.trim(),o=x.exec(r),n=o?`${f}${o[1]}@${o[2]}/${o[3]}`:r;if(!n.toLowerCase().startsWith(f))return null;let s=_(n);if(!s||""===s.username||""!==s.password||!v(s))return null;let i=n.indexOf("/",f.length);return -1===i||s.pathname!==n.slice(i)||k(s).length<2?null:j("SSH",r,w(s.pathname).replace(/\.git$/i,""),t)})(e,t);if(r)return r;let o=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=_(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&v(r)?r:null})(e);if(!o)return null;if(d.test(o.pathname))return{parsed:{source:"archive",url:o.href},label:`Zip archive — ${o.host}${o.pathname}`,suggestedName:y(w(o.pathname).replace(d,""))};if("github.com"===o.hostname.replace(/^www\./,""))return((e,t)=>{let r=k(e);if(r.length<2)return null;let o=r[0],n=r[1].replace(/\.git$/,"");if(!h.test(o)||!g.test(n))return null;let s=`${o}/${n}`,i=`https://github.com/${s}`,a={parsed:{source:"github",repo:s},label:`GitHub repo — ${s}`,suggestedName:y(n)};if(r.length>=4&&("tree"===r[2]||"blob"===r[2])){let e=r.slice(4),t=w(e.join("/")),o=u.test(t)?e.slice(0,-1):e;if(0===o.length)return a;let n=c(o.join("/"));return l.test(n)?{parsed:{source:"git-subdir",url:i,path:n},label:`GitHub subdir — ${s} @ ${n}`,suggestedName:y(w(n))}:null}if(2!==r.length)return null;let d=c(t??"");return""!==d?l.test(d)?{parsed:{source:"git-subdir",url:i,path:d},label:`GitHub subdir — ${s} @ ${d}`,suggestedName:y(w(d))}:null:a})(o,t);if(k(o).length<2)return null;let n=w(o.pathname).replace(/\.git$/,"");return j("Git",`${o.protocol}//${o.host}${o.pathname.replace(/\/+$/,"")}`,n,t)},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let S=({source:e})=>{let r=O(e),o=r&&"git-subdir"===e.source&&e.path?`${r}/tree/main/${e.path}`:r;return o?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:o,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[o.replace("https://",""),(0,t.jsx)(i.Link2,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:E(e)})]}):null};e.s(["default",0,({skill:e,onBack:i})=>{let[l,c]=(0,r.useState)("overview"),[u,d]=(0,r.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),d(t),setTimeout(()=>d(null),2e3)},m=N(e),h=C(window.location.origin),g=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:i,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(o.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>c(e.key),className:(0,a.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",l===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===l&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:g.map((e,r)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,a.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(S,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(m,"install"),className:(0,a.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===u?"text-success":"text-info"),children:["install"===u?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(s.Copy,{className:"size-3"}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:m})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>c("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===l&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;p(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,a.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===u?"text-success":"text-info"),children:["marketplace-cmd"===u?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(s.Copy,{className:"size-3"}),"marketplace-cmd"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>p(h,"settings"),className:(0,a.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===u?"text-success":"text-info"),children:["settings"===u?(0,t.jsx)(n.Check,{className:"size-3"}):(0,t.jsx)(s.Copy,{className:"size-3"}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:h})]})]})]})}],652272)},902555,e=>{"use strict";var t=e.i(843476),r=e.i(746798),o=e.i(271645);let n=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))}),s=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});var i=e.i(278587),a=e.i(68155),l=e.i(360820),c=e.i(871943),u=e.i(434626);let d=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});var p=e.i(196631);function m({icon:e,onClick:r,className:o,disabled:n,dataTestId:s}){return n?(0,t.jsx)("span",{className:"inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50","data-testid":s,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})}):(0,t.jsx)("span",{className:(0,p.cx)("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5",o),onClick:r,"data-testid":s,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})})}let h={Edit:{icon:n,className:"hover:text-info"},Delete:{icon:a.TrashIcon,className:"hover:text-destructive"},Test:{icon:s,className:"hover:text-info"},Regenerate:{icon:i.RefreshIcon,className:"hover:text-success"},Reset:{icon:i.RefreshIcon,className:"hover:text-info"},Up:{icon:l.ChevronUpIcon,className:"hover:text-info"},Down:{icon:c.ChevronDownIcon,className:"hover:text-info"},Open:{icon:u.ExternalLinkIcon,className:"hover:text-success"},Copy:{icon:d,className:"hover:text-info"}};e.s(["default",0,function({onClick:e,tooltipText:o,disabled:n=!1,disabledTooltipText:s,dataTestId:i,variant:a}){let{icon:l,className:c}=h[a],u=n?s:o,d=(0,t.jsx)(m,{icon:l,onClick:e,className:c,disabled:n,dataTestId:i});return u?(0,t.jsx)(r.TooltipProvider,{children:(0,t.jsxs)(r.Tooltip,{children:[(0,t.jsx)(r.TooltipTrigger,{render:(0,t.jsx)("span",{tabIndex:n?0:void 0}),children:d}),(0,t.jsx)(r.TooltipContent,{children:u})]})}):(0,t.jsx)("span",{children:d})}],902555)},292335,e=>{"use strict";let t={NONE:"none",API_KEY:"api_key",BEARER_TOKEN:"bearer_token",TOKEN:"token",BASIC:"basic",OAUTH2:"oauth2",OAUTH2_TOKEN_EXCHANGE:"oauth2_token_exchange",OAUTH2_ID_JAG:"oauth2_id_jag",AWS_SIGV4:"aws_sigv4",TRUE_PASSTHROUGH:"true_passthrough",OAUTH_DELEGATE:"oauth_delegate"},r=[{value:t.NONE,label:"None"},{value:t.API_KEY,label:"API Key"},{value:t.BEARER_TOKEN,label:"Bearer Token"},{value:t.TOKEN,label:"Token"},{value:t.BASIC,label:"Basic Auth"},{value:t.OAUTH2,label:"OAuth"},{value:t.OAUTH2_TOKEN_EXCHANGE,label:"OAuth Token Exchange (OBO)"},{value:t.OAUTH2_ID_JAG,label:"ID-JAG (Okta Cross App Access)"},{value:t.AWS_SIGV4,label:"AWS SigV4 (Bedrock AgentCore MCPs)"},{value:t.TRUE_PASSTHROUGH,label:"True Passthrough (no LiteLLM auth)"},{value:t.OAUTH_DELEGATE,label:"OAuth Delegate (client-supplied upstream token)"}],o=e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE,n={INTERACTIVE:"interactive",M2M:"m2m"},s=e=>{let t=e.credentials??{};return JSON.stringify({url:"string"==typeof e.url?e.url:null,spec_path:"string"==typeof e.spec_path?e.spec_path:null,auth_type:e.auth_type??null,oauth_flow_type:e.oauth_flow_type??null,client_id:t.client_id??null,client_secret:t.client_secret??null,scopes:t.scopes??null,upstream_resource:t.upstream_resource??null,issuer:e.issuer??null,authorization_url:e.authorization_url??null,token_url:e.token_url??null,registration_url:e.registration_url??null})},i=["client_id","client_secret"],a=["upstream_resource","upstream_token_header"],l=["access_token","refresh_token","expires_in","scope"],c=(e,t)=>{if(!e)return;let r=Object.fromEntries(t.filter(t=>"string"==typeof e[t]&&""!==e[t]).map(t=>[t,e[t]]));return Object.keys(r).length>0?r:void 0},u="client_credentials",d={SSE:"sse",HTTP:"http",STDIO:"stdio",OPENAPI:"openapi"},p=[{value:d.HTTP,label:"Streamable HTTP (Recommended)"},{value:d.SSE,label:"Server-Sent Events (SSE)"},{value:d.STDIO,label:"Standard Input/Output (stdio)"},{value:d.OPENAPI,label:"OpenAPI Spec"}];e.s(["ADMIN_CONFIG_CREDENTIAL_KEYS",0,a,"AUTH_TYPE",0,t,"AUTH_TYPE_ITEMS",0,r,"CLEARED_ON_INVALIDATION",0,["credentials"],"MCP_OAUTH2_FLOW_INTERACTIVE",0,"authorization_code","MCP_OAUTH2_FLOW_M2M",0,u,"MCP_REACHABLE_DESCRIPTION",0,"Server responded. Authentication and tools were not checked","OAUTH_FLOW",0,n,"TRANSPORT",0,d,"TRANSPORT_ITEMS",0,p,"credentialAuthClass",0,e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE?"client_forwarded":e??null,"gatewayMintsClientFor",0,e=>e.auth_type===t.TRUE_PASSTHROUGH||e.auth_type===t.OAUTH_DELEGATE&&!e.dcr_bridge,"getMcpOAuthMode",0,function(e){return e.auth_type===t.OAUTH2_TOKEN_EXCHANGE?"token_exchange":e.auth_type!==t.OAUTH2?null:e.oauth2_flow===u?"m2m":e.delegate_auth_to_upstream?"passthrough":"authorization_code"},"getOAuthAuthorizationIdentity",0,s,"handleAuth",0,e=>null==e?t.NONE:e,"handleTransport",0,(e,t)=>null==e?d.SSE:t&&e!==d.STDIO?d.OPENAPI:e,"isClientForwardedTokenMode",0,o,"isHeldOAuthTokenStale",0,(e,t)=>void 0!==t&&s(e)!==t,"isUnsupportedOnGatewayConnect",0,e=>o(e)||e===t.OAUTH2_TOKEN_EXCHANGE,"oauth2FlowToFormValue",0,function(e){return e===u?n.M2M:e?n.INTERACTIVE:void 0},"preservedAdminCredentials",0,e=>c(e,[...i,...a]),"preservedDeclaredAppCredentials",0,e=>c(e,i),"withoutMintedTokenCredentials",0,e=>{if(!e)return;let t=Object.fromEntries(Object.entries(e).filter(([e])=>!l.includes(e)));return Object.keys(t).length>0?t:void 0}])},67488,e=>{"use strict";var t=e.i(843476),r=e.i(463059),o=e.i(618566),n=e.i(196631);function s(e){let t=(0,o.useRouter)();return r=>{r.metaKey||r.ctrlKey||r.shiftKey||1===r.button||(r.preventDefault(),t.push(e))}}function i({href:e,className:o,children:a}){let l=s(e);return(0,t.jsxs)("a",{href:e,onClick:l,className:(0,n.cn)("group inline-flex min-w-0 max-w-full items-center gap-0.5 font-semibold underline-offset-4 hover:underline",o),children:[(0,t.jsx)("span",{className:"min-w-0 truncate",children:a}),(0,t.jsx)(r.ChevronRight,{className:"size-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground"})]})}e.s(["EntityLink",0,function({href:e,className:r,children:o}){return e?(0,t.jsx)(i,{href:e,className:r,children:o}):(0,t.jsx)("span",{className:(0,n.cn)("inline-block min-w-0 max-w-full truncate font-semibold",r),children:o})},"useEntityLinkClick",0,s])},845150,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(131792);let n=(e,t)=>{let r=t.trim().toLowerCase();return!r||e.label.toLowerCase().includes(r)||e.value.toLowerCase().includes(r)||(e.description?.toLowerCase().includes(r)??!1)};e.s(["MultiSelect",0,function({id:e,options:s,value:i=[],onValueChange:a,placeholder:l="Select options",emptyText:c="No options found",disabled:u=!1,loading:d=!1,allowCustomValues:p=!1,className:m}){let h=(0,o.useComboboxAnchor)(),[g,b]=(0,r.useState)(""),f=s.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=i.filter(e=>"string"==typeof e&&e.length>0).map(e=>f.find(t=>t.value===e)??{label:e,value:e}),k=g.trim(),_=f.some(e=>e.value.toLowerCase()===k.toLowerCase()),v=p&&k&&!_?[...f,{label:`Create "${k}"`,value:k}]:f;return(0,t.jsxs)(o.Combobox,{multiple:!0,items:v,value:x,onValueChange:e=>{a(Array.from(new Set(p?e.flatMap(e=>i.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),b("")},inputValue:g,onInputValueChange:b,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:n,disabled:u||d,children:[(0,t.jsx)(o.ComboboxChips,{render:(0,t.jsx)("div",{ref:h}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(o.ComboboxValue,{children:r=>(0,t.jsxs)(t.Fragment,{children:[r.map(e=>(0,t.jsx)(o.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(o.ComboboxChipsInput,{id:e,placeholder:d?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),r.length>0&&!u&&!d&&(0,t.jsx)(o.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(o.ComboboxContent,{anchor:h,children:[(0,t.jsx)(o.ComboboxEmpty,{children:c}),(0,t.jsx)(o.ComboboxList,{children:e=>(0,t.jsx)(o.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},581070,e=>{"use strict";var t=e.i(843476),r=e.i(746798);e.s(["CellTooltip",0,function({content:e,trigger:o}){return(0,t.jsx)(r.TooltipProvider,{delay:300,children:(0,t.jsxs)(r.Tooltip,{children:[(0,t.jsx)(r.TooltipTrigger,{render:o}),(0,t.jsx)(r.TooltipContent,{children:e})]})})}])},112179,e=>{"use strict";var t=e.i(843476),r=e.i(67488),o=e.i(487486),n=e.i(196631),s=e.i(581070);let i={success:"border-success/20 bg-success/10 text-success",error:"border-destructive/20 bg-destructive/10 text-destructive",warning:"border-warning/20 bg-warning/10 text-warning",neutral:"border-border bg-muted text-muted-foreground",info:"border-info/20 bg-info/10 text-info"};function a({href:e,dataTestId:s,className:i,children:l}){let c=(0,r.useEntityLinkClick)(e);return(0,t.jsx)(o.Badge,{variant:"outline","data-testid":s,className:(0,n.cn)("cursor-pointer hover:underline",i),render:(0,t.jsx)("a",{href:e,onClick:c}),children:l})}e.s(["StatusBadge",0,function({tone:e,label:r,tooltip:l,dataTestId:c,className:u,href:d}){let p=(0,n.cn)("whitespace-nowrap font-normal",i[e],u),m=d?(0,t.jsx)(a,{href:d,dataTestId:c,className:p,children:r}):(0,t.jsx)(o.Badge,{variant:"outline","data-testid":c,className:p,children:r});return l?(0,t.jsx)(s.CellTooltip,{content:l,trigger:m}):m}])},735450,e=>{"use strict";var t=e.i(681307);let r="https://docs.litellm.ai/docs/decisions",o=`${r}#supported-providers`,n=new Set(["/v1/decisions","/v1/systemone"]),s=t.z.object({litellm_provider:t.z.string(),mode:t.z.string().nullish(),supported_endpoints:t.z.array(t.z.string()).nullish()}),i=e=>"decisions"===e||"evaluation"===e,a=e=>i(e.mode)||(e.supported_endpoints??[]).some(e=>n.has(e));e.s(["DECISIONS_DOCS_URL",0,r,"DECISIONS_PROVIDERS_DOCS_URL",0,o,"SYSTEM_ONE_PLAYGROUND_ROUTE",0,"playground?tab=system-one","buildDecisionCatalog",0,function(e){let r,o=(r=t.z.record(t.z.string(),t.z.unknown()).safeParse(e)).success?Object.entries(r.data).flatMap(([e,t])=>{let r=s.safeParse(t);return r.success?[[e,r.data]]:[]}):[],n=o.filter(([,e])=>a(e)),i=new Map([...new Set(n.map(([,e])=>e.litellm_provider))].map(e=>{let t=o.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:t.every(([,e])=>a(e))}]}));return{models:new Set(n.map(([e])=>e)),providers:i}},"isDecisionMode",0,i,"isDecisionSelection",0,(e,t,r)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||r.some(t=>e.models.has(t))])},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function r(e,r){let o=t(e);if(""===o)return!0;let n=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!n.some(e=>e.includes(o))||o.split(/\s+/).every(e=>n.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,o){return e.filter(e=>r(t,o(e)))},"matchesSearchTerm",0,r,"rankBySearchRelevance",0,function(e,r,o){let n=t(r);if(""===n)return[...e];let s=e=>{let t=o(e).toLowerCase();return 1e3*(t===n)+100*!!t.startsWith(n)+(1e3-t.length)};return[...e].sort((e,t)=>s(t)-s(e))}])}]);