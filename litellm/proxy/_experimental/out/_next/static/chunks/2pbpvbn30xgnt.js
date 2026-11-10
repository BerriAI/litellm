(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,871943,502547,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["f",0,r],871943);let o=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["f",0,o],502547)},360820,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["f",0,r],360820)},278587,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["f",0,r],278587)},586448,e=>{var t=e.i(271645),r=e.i(540143),o=e.i(869230),i=e.i(915823),s=e.i(619273);function a(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var l=class extends i.f{#e;#t;#r;#o;#i;#s;#a;#l;#n;#d=[];constructor(e,t,r){super(),this.#e=e,this.#o=r,this.#r=[],this.#i=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#i.forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#i.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#o=t,r.f.batch(()=>{let e=this.#i,t=this.#c(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),o=r.map(e=>e.getCurrentResult()),i=e.length!==r.length,l=r.some((t,r)=>t!==e[r]),n=i||l,d=!!n||o.some((e,t)=>{let r=this.#t[t];return!r||!(0,s.U)(e,r)});(n||d)&&(n&&(this.#d=t,this.#i=r),this.#t=o,this.hasListeners()&&(n&&(a(e,r).forEach(e=>{e.destroy()}),a(r,e).forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#i.map(e=>e.getCurrentQuery())}getObservers(){return this.#i}getOptimisticResult(e,t){let r=this.#c(e),o=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),i=r.map(e=>e.defaultedQueryOptions.queryHash);return[o,e=>this.#m(e??o,t,i),()=>this.#g(o,r)]}#g(e,t){return t.map((r,o)=>{let i=e[o];return r.defaultedQueryOptions.notifyOnChangeProps?i:r.observer.trackResult(i,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,r){if(t){let o=this.#n,i=void 0!==r&&void 0!==o&&(o.length!==r.length||r.some((e,t)=>e!==o[t]));return(!this.#s||this.#t!==this.#l||i||t!==this.#a)&&(this.#a=t,this.#l=this.#t,void 0!==r&&(this.#n=r),this.#s=(0,s.B)(this.#s,t(e))),this.#s}return e}#h(){return this.#o?.combine!==void 0&&this.#i.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#c(e){let t=new Map;this.#i.forEach(e=>{let r=e.options.queryHash;if(!r)return;let o=t.get(r);o?o.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let i=this.#e.defaultQueryOptions(e),s=t.get(i.queryHash)?.shift()??new o.f(this.#e,i);r.push({defaultedQueryOptions:i,observer:s})}),r}#u(e,t){let r=this.#i.indexOf(e);if(-1!==r){var o;let e;this.#t=(o=this.#t,(e=o.slice(0))[r]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#g(this.#t,this.#d),t=this.#h(),o=this.#s,i=t?o:this.#m(e,this.#o?.combine);(t||o!==i)&&r.f.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},n=e.i(912598),d=e.i(381384),u=e.i(673664),c=e.i(427001),p=e.i(254440);e.s(["f",0,function({queries:e,...i},a){let m=(0,n.d)(a),g=(0,d.f)(),h=(0,u.p)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,m,g]);f.forEach(e=>{(0,p.p)(e);let t=m.getQueryCache().get(e.queryHash);(0,c.l)(e,h,t)}),(0,c.E)(h);let[b]=t.useState(()=>new l(m,f,i)),[x,k,_]=b.getOptimisticResult(f,i.combine),v=!g&&!1!==i.subscribed;t.useSyncExternalStore(t.useCallback(e=>v?b.subscribe(r.f.batchCalls(e)):s.z,[b,v]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(f,i)},[f,i,b]);let y=x.some((e,t)=>(0,p.Q)(f[t],e))?x.flatMap((e,t)=>{let r=f[t];if(r&&(0,p.Q)(r,e)){let e=new o.f(m,r);return(0,p.X)(r,e,h)}return[]}):[];if(y.length>0)throw Promise.all(y);let w=x.find((e,t)=>{let r=f[t];return r&&(0,c.P)({result:e,errorResetBoundary:h,throwOnError:r.throwOnError,query:m.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(w?.error)throw w.error;return k(_())}],586448)},180127,e=>{let t=(0,e.i(475254).f)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["f",0,t])},871689,e=>{var t=e.i(180127);e.s(["f",()=>t.f])},810980,e=>{let t=(0,e.i(475254).f)("book-open",[["path",{d:"M12 7v14",key:"1akyts"}],["path",{d:"M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z",key:"ruj8y"}]]);e.s(["f",0,t],810980)},440160,e=>{let t=(0,e.i(475254).f)("download",[["path",{d:"M12 15V3",key:"m9g1x1"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["path",{d:"m7 10 5 5 5-5",key:"brsn70"}]]);e.s(["f",0,t],440160)},541071,373488,e=>{let t=(0,e.i(475254).f)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["f",0,t],373488),e.s(["f",0,t],541071)},546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},332102,e=>{let t=(0,e.i(475254).f)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["f",0,t],332102)},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},531278,e=>{var t=e.i(717521);e.s(["f",()=>t.f])},283086,576144,e=>{let t=(0,e.i(475254).f)("sparkles",[["path",{d:"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",key:"4pj2yx"}],["path",{d:"M20 3v4",key:"1olli1"}],["path",{d:"M22 5h-4",key:"1gvqau"}],["path",{d:"M4 17v2",key:"vumght"}],["path",{d:"M5 18H3",key:"zchphs"}]]);e.s(["f",0,t],576144),e.s(["f",0,t],283086)},455037,e=>{var t=e.i(494144);e.s(["f",()=>t.f])},368670,e=>{var t=e.i(602869),r=e.i(266027);let o=(0,e.i(243652).f)("modelCostMap");e.s(["Q",0,o,"X",0,(e=!0,i=!1)=>(0,r.f)({enabled:e,queryKey:o.list(i?{filters:{catalog_only:"true"}}:{}),queryFn:async()=>await (0,t.T)(i),staleTime:6e4,gcTime:6e4})])},803099,e=>{var t=e.i(602869),r=e.i(266027);let o=(0,e.i(243652).f)("providerFields");e.s(["f",0,()=>(0,r.f)({queryKey:o.list({}),queryFn:async()=>await (0,t.Bg)(),staleTime:864e5,gcTime:864e5})])},66330,e=>{var t=e.i(735450);e.s(["f",0,(e,r,o,i)=>{if("extraction"===e)return`import requests

response = requests.post(
    ${JSON.stringify(`${r}/v1/extraction`)},  # Your LiteLLM Proxy URL
    headers={${JSON.stringify(i)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(o)},
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
    headers={${JSON.stringify(i)}: "Bearer your_api_key"},
    json={
        "model": ${JSON.stringify(o)},
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
    model=${JSON.stringify(o)},
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}])},466828,e=>{var t=e.i(843476),r=e.i(271645),o=e.i(678784);let i=(0,e.i(475254).f)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var s=e.i(322831),a=e.i(650056);let l={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};var n=e.i(488012);e.s(["f",0,({code:e,language:d})=>{let{t:u}=(0,s.f)(),c=(0,n.f)(l),[p,m]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-border bg-muted overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),m(!0),setTimeout(()=>m(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground z-raised","aria-label":u("codeBlock.copyCode"),children:p?(0,t.jsx)(o.f,{size:16}):(0,t.jsx)(i,{size:16})}),(0,t.jsx)(a.f,{language:d,style:c,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",background:"transparent"},codeTagProps:{style:{background:"transparent"}},showLineNumbers:!0,children:e})]})}],466828)},37084,e=>{let t=e=>e?`${e.getFullYear()}-${String(e.getMonth()+1).padStart(2,"0")}-${String(e.getDate()).padStart(2,"0")}`:"all";e.s(["h",0,(e,t)=>{let r=window.URL.createObjectURL(e),o=document.createElement("a");o.href=r,o.download=t,document.body.appendChild(o),o.click(),document.body.removeChild(o),window.URL.revokeObjectURL(r)},"O",0,(e,r,o,i)=>`${e}_usage_${r}_${t(i.from)}_${t(i.to)}.${o}`])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let r,{apiKeySource:o,accessToken:i,apiKey:s,inputMessage:a,chatHistory:l,selectedTags:n,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,x="session"===o?i:s,k=window.location.origin,_=f?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?k=_:f?.PROXY_BASE_URL&&(k=f.PROXY_BASE_URL);let v=a||"Your prompt here",y=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};n.length>0&&(j.tags=n),d.length>0&&(j.vector_stores=d),u.length>0&&(j.guardrails=u),c.length>0&&(j.policies=c);let C=g||"your-model-name",N=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",O="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${k}",
	api_version="2024-02-01"${N}
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${k}"${N}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let o=w.length>0?w:[{role:"user",content:v}];r=`
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
`;break}case t.Q.RESPONSES:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let o=w.length>0?w:[{role:"user",content:v}];r=`
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
#                 {"type": "input_text", "text": "${y}"},
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
`;break}case t.Q.IMAGE:r="azure"===h?`
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
	model="${C}",
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
`;break;case t.Q.IMAGE_EDITS:r="azure"===h?`
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
	model="${C}",
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
prompt = "${y}"

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
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:r=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${a||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${O}
${r}`}])},209261,e=>{var t=e.i(201787);let r=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,o=e=>e.trim().replace(/\/+$/,""),i=/\.(md|markdown|txt|json|ya?ml|toml)$/i,s=/\.zip$/i,a=/^[0-9a-fA-F]{64}$/,l=/^\d{1,3}(\.\d{1,3}){3}$/,n=/^[A-Za-z0-9-]+$/,d=/^[A-Za-z0-9._-]+$/,u=/^https?:\/\//i,c="ssh://",p=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,m=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{try{return new URL(e)}catch{return null}},h=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!l.test(e.hostname),f=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},b=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),x=(e,i,s,a)=>{let l=o(a??"");return""!==l?r.test(l)?{parsed:{source:"git-subdir",url:i,path:l},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:i,path:l}),suggestedName:b(f(l))}:null:{parsed:{source:"url",url:i},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:i}),suggestedName:b(s)}},k="<your-litellm-key>",_=e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e);e.s(["i",0,k,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${k}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"v",0,(e,t,r,o)=>{if("archive"!==t.source||!_(e)||!t.sha256||!a.test(t.sha256))return null;let i=`/claude-code/plugins/${e}/archive/${t.sha256}.zip`;try{let e=new URL(`${r.replace(/\/+$/,"")}${i}`,o);return/^https?:$/.test(e.protocol)&&t.url===e.href?i:null}catch{return null}},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&u.test(e.url)?e.url:null,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||a.test(e.trim()),"P",0,e=>{let t=o(e);return""!==t&&r.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,a)=>{let l=((e,t)=>{let r=e.trim(),o=p.exec(r),i=o?`${c}${o[1]}@${o[2]}/${o[3]}`:r;if(!i.toLowerCase().startsWith(c))return null;let s=g(i);if(!s||""===s.username||""!==s.password||!h(s))return null;let a=i.indexOf("/",c.length);return -1===a||s.pathname!==i.slice(a)||m(s).length<2?null:x("SSH",r,f(s.pathname).replace(/\.git$/i,""),t)})(e,a);if(l)return l;let u=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=g(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&h(r)?r:null})(e);if(!u)return null;if(s.test(u.pathname))return{parsed:{source:"archive",url:u.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${u.host}${u.pathname}`}),suggestedName:b(f(u.pathname).replace(s,""))};if("github.com"===u.hostname.replace(/^www\./,""))return((e,s)=>{let a=m(e);if(a.length<2)return null;let l=a[0],u=a[1].replace(/\.git$/,"");if(!n.test(l)||!d.test(u))return null;let c=`${l}/${u}`,p=`https://github.com/${c}`,g={parsed:{source:"github",repo:c},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:c}),suggestedName:b(u)};if(a.length>=4&&("tree"===a[2]||"blob"===a[2])){let e=a.slice(4),s=f(e.join("/")),l=i.test(s)?e.slice(0,-1):e;if(0===l.length)return g;let n=o(l.join("/"));return r.test(n)?{parsed:{source:"git-subdir",url:p,path:n},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:n}),suggestedName:b(f(n))}:null}if(2!==a.length)return null;let h=o(s??"");return""!==h?r.test(h)?{parsed:{source:"git-subdir",url:p,path:h},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:h}),suggestedName:b(f(h))}:null:g})(u,a);if(m(u).length<2)return null;let k=f(u.pathname).replace(/\.git$/,"");return x("Git",`${u.protocol}//${u.host}${u.pathname.replace(/\/+$/,"")}`,k,a)},"e",0,_])},652272,e=>{var t=e.i(843476),r=e.i(271645),o=e.i(871689),i=e.i(643531),s=e.i(174886),a=e.i(440160),l=e.i(306228),n=e.i(345771),d=e.i(322831),u=e.i(768371),c=e.i(196631),p=e.i(37084),m=e.i(602869),g=e.i(519455),h=e.i(209261);let f=({skill:e,accessToken:o})=>{let{t:i}=(0,d.f)(),[s,n]=(0,r.useState)(!1),[u,c]=(0,r.useState)(null),{source:f}=e,b=(0,h.v)(e.name,f,(0,m._h)(),window.location.origin),x=async()=>{if(b&&!s){n(!0),c(null);try{let t=await m.Mw.getBlob(b,{accessToken:o,redirect:"error"});(0,p.h)(t,`${e.name}.zip`)}catch(e){c(e instanceof Error?e.message:i("claudeCodePlugins.skillDetail.overview.downloadFailed"))}finally{n(!1)}}},k=(0,h.E)(f),_=k&&"git-subdir"===f.source&&f.path?`${k}/tree/main/${f.path}`:k;return _||f.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:i("claudeCodePlugins.skillDetail.overview.source")}),b&&(0,t.jsxs)(g.K,{variant:"link",className:"h-auto p-0 text-[13px]",onClick:x,disabled:s,children:[(0,t.jsx)(a.f,{className:"size-3 shrink-0"}),i(s?"claudeCodePlugins.skillDetail.overview.downloading":"claudeCodePlugins.skillDetail.overview.downloadZip")]}),!b&&_&&(0,t.jsxs)("a",{href:_,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[_.replace("https://",""),(0,t.jsx)(l.f,{className:"size-3 shrink-0"})]}),!_&&(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:(0,h._)(f)}),u&&(0,t.jsx)("p",{role:"alert",className:"mt-2 text-xs text-destructive",children:u})]}):null};e.s(["f",0,({skill:e,onBack:a,accessToken:l})=>{let{t:p}=(0,d.f)(),[m,g]=(0,r.useState)("overview"),[b,x]=(0,r.useState)(null),k=u.q.useQuery("get","/.well-known/litellm-ui-config").data?.claude_code_marketplace_auth_required!==!1,_=(e,t)=>{navigator.clipboard.writeText(e),x(t),setTimeout(()=>x(null),2e3)},v=(0,h.a)(e),y=(0,h.U)(window.location.origin,k),w=[...e.category?[{property:p("claudeCodePlugins.skillDetail.properties.category"),value:e.category}]:[],...e.domain?[{property:p("claudeCodePlugins.skillDetail.properties.domain"),value:e.domain}]:[],...e.namespace?[{property:p("claudeCodePlugins.skillDetail.properties.namespace"),value:e.namespace}]:[],...e.version?[{property:p("claudeCodePlugins.skillDetail.properties.version"),value:e.version}]:[],...e.author?.name?[{property:p("claudeCodePlugins.skillDetail.properties.author"),value:e.author.name}]:[],...e.created_at?[{property:p("claudeCodePlugins.skillDetail.properties.added"),value:new Date(e.created_at).toLocaleDateString()}]:[]],j=[{key:"overview",label:p("claudeCodePlugins.skillDetail.tabs.overview")},{key:"usage",label:p("claudeCodePlugins.skillDetail.tabs.usage")}];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:a,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(o.f,{className:"size-3"}),(0,t.jsx)("span",{children:p("claudeCodePlugins.skillDetail.back")})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:j.map(e=>(0,t.jsx)("div",{onClick:()=>g(e.key),className:(0,c.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",m===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===m&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.overview.title")}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.subtitle")}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.propertyHeader")}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:w.map((e,r)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.status")}),(0,t.jsx)("span",{className:(0,c.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:p(e.enabled?"claudeCodePlugins.skillDetail.overview.public":"claudeCodePlugins.skillDetail.overview.draft")})]}),(0,t.jsx)(f,{skill:e,accessToken:l}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.tags")}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.skillId")}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===m&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.usage.title")}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.usage.intro")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:p("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>_(v,"install"),className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===b?"text-success":"text-info"),children:["install"===b?(0,t.jsx)(i.f,{className:"size-3"}):(0,t.jsx)(s.f,{className:"size-3"}),p("install"===b?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:v})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.usage.staleCatalog",{name:e.name})}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:[p("claudeCodePlugins.skillDetail.usage.notConfigured")," ",(0,t.jsx)("span",{onClick:()=>g("setup"),className:"cursor-pointer text-info",children:p("claudeCodePlugins.skillDetail.usage.seeSetup")})]})]}),"setup"===m&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.setup.title")}),k?(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.setup.removeFirst")}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace remove litellm"})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(n.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.requiresKey",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"}),placeholder:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:h.i})}})})]}):(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.setup.runCommand")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:p("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;_(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===b?"text-success":"text-info"),children:["marketplace-cmd"===b?(0,t.jsx)(i.f,{className:"size-3"}):(0,t.jsx)(s.f,{className:"size-3"}),p("marketplace-cmd"===b?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(n.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.orSettings",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})}})})]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>_(y,"settings"),className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===b?"text-success":"text-info"),children:["settings"===b?(0,t.jsx)(i.f,{className:"size-3"}):(0,t.jsx)(s.f,{className:"size-3"}),p("settings"===b?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:y})]})]})]})}],652272)},902555,434626,e=>{var t=e.i(843476),r=e.i(746798),o=e.i(271645);let i=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))}),s=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});var a=e.i(278587),l=e.i(68155),n=e.i(360820),d=e.i(871943);let u=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["f",0,u],434626);let c=o.forwardRef(function(e,t){return o.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),o.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});var p=e.i(196631);function m({icon:e,onClick:r,className:o,disabled:i,dataTestId:s}){return i?(0,t.jsx)("span",{className:"inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50","data-testid":s,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})}):(0,t.jsx)("span",{className:(0,p.P)("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5",o),onClick:r,"data-testid":s,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})})}let g={Edit:{icon:i,className:"hover:text-info"},Delete:{icon:l.f,className:"hover:text-destructive"},Test:{icon:s,className:"hover:text-info"},Regenerate:{icon:a.f,className:"hover:text-success"},Reset:{icon:a.f,className:"hover:text-info"},Up:{icon:n.f,className:"hover:text-info"},Down:{icon:d.f,className:"hover:text-info"},Open:{icon:u,className:"hover:text-success"},Copy:{icon:c,className:"hover:text-info"}};e.s(["f",0,function({onClick:e,tooltipText:o,disabled:i=!1,disabledTooltipText:s,dataTestId:a,variant:l}){let{icon:n,className:d}=g[l],u=i?s:o,c=(0,t.jsx)(m,{icon:n,onClick:e,className:d,disabled:i,dataTestId:a});return u?(0,t.jsx)(r.g,{children:(0,t.jsxs)(r.W,{children:[(0,t.jsx)(r.y,{render:(0,t.jsx)("span",{tabIndex:i?0:void 0}),children:c}),(0,t.jsx)(r.Y,{children:u})]})}):(0,t.jsx)("span",{children:c})}],902555)},292335,e=>{let t={NONE:"none",API_KEY:"api_key",BEARER_TOKEN:"bearer_token",TOKEN:"token",BASIC:"basic",AUTHORIZATION:"authorization",OAUTH2:"oauth2",OAUTH2_TOKEN_EXCHANGE:"oauth2_token_exchange",OAUTH2_ID_JAG:"oauth2_id_jag",AWS_SIGV4:"aws_sigv4",TRUE_PASSTHROUGH:"true_passthrough",OAUTH_DELEGATE:"oauth_delegate"},r=[{value:t.NONE,label:"None"},{value:t.API_KEY,label:"API Key"},{value:t.BEARER_TOKEN,label:"Bearer Token"},{value:t.TOKEN,label:"Token"},{value:t.BASIC,label:"Basic Auth"},{value:t.AUTHORIZATION,label:"Authorization (raw header)"},{value:t.OAUTH2,label:"OAuth"},{value:t.OAUTH2_TOKEN_EXCHANGE,label:"OAuth Token Exchange (OBO)"},{value:t.OAUTH2_ID_JAG,label:"ID-JAG (Okta Cross App Access)"},{value:t.AWS_SIGV4,label:"AWS SigV4 (Bedrock AgentCore MCPs)"},{value:t.TRUE_PASSTHROUGH,label:"True Passthrough (no LiteLLM auth)"},{value:t.OAUTH_DELEGATE,label:"OAuth Delegate (client-supplied upstream token)"}],o=e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE,i={INTERACTIVE:"interactive",M2M:"m2m"},s=e=>{let t=e.credentials??{};return JSON.stringify({url:"string"==typeof e.url?e.url:null,spec_path:"string"==typeof e.spec_path?e.spec_path:null,auth_type:e.auth_type??null,oauth_flow_type:e.oauth_flow_type??null,client_id:t.client_id??null,client_secret:t.client_secret??null,scopes:t.scopes??null,upstream_resource:t.upstream_resource??null,issuer:e.issuer??null,authorization_url:e.authorization_url??null,token_url:e.token_url??null,registration_url:e.registration_url??null})},a=["client_id","client_secret"],l=["upstream_resource","upstream_token_header"],n=["access_token","refresh_token","expires_in","scope"],d=(e,t)=>{if(!e)return;let r=Object.fromEntries(t.filter(t=>"string"==typeof e[t]&&""!==e[t]).map(t=>[t,e[t]]));return Object.keys(r).length>0?r:void 0},u="client_credentials",c={SSE:"sse",HTTP:"http",STDIO:"stdio",OPENAPI:"openapi"},p=[{value:c.HTTP,label:"Streamable HTTP (Recommended)"},{value:c.SSE,label:"Server-Sent Events (SSE)"},{value:c.STDIO,label:"Standard Input/Output (stdio)"},{value:c.OPENAPI,label:"OpenAPI Spec"}];e.s(["_",0,l,"W",0,t,"B",0,r,"L",0,["credentials"],"V",0,"authorization_code","P",0,u,"S",0,"Server responded. Authentication and tools were not checked","n",0,i,"v",0,c,"m",0,p,"J",0,e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE?"client_forwarded":e??null,"A",0,e=>e.auth_type===t.TRUE_PASSTHROUGH||e.auth_type===t.OAUTH_DELEGATE&&!e.dcr_bridge,"j",0,function(e){return e.auth_type===t.OAUTH2_TOKEN_EXCHANGE?"token_exchange":e.auth_type!==t.OAUTH2?null:e.oauth2_flow===u?"m2m":e.delegate_auth_to_upstream?"passthrough":"authorization_code"},"Z",0,s,"d",0,e=>null==e?t.NONE:e,"K",0,(e,t)=>null==e?c.SSE:t&&e!==c.STDIO?c.OPENAPI:e,"C",0,o,"i",0,(e,t)=>void 0!==t&&s(e)!==t,"z",0,e=>o(e)||e===t.OAUTH2_TOKEN_EXCHANGE,"Y",0,function(e){return e===u?i.M2M:e?i.INTERACTIVE:void 0},"o",0,e=>{let t=d(e,[...a,...l]),r=(e=>{if(!Array.isArray(e))return;let t=e.filter(e=>"string"==typeof e&&""!==e);return t.length>0?t:void 0})(e?.scopes);return void 0===r?t:{...t??{},scopes:r}},"T",0,e=>d(e,a),"f",0,e=>{if(!e)return;let t=Object.fromEntries(Object.entries(e).filter(([e])=>!n.includes(e)));return Object.keys(t).length>0?t:void 0}])},466098,348594,831538,e=>{var t=e.i(586448),r=e.i(602869),o=e.i(271645),i=e.i(198458),s=e.i(735450);let a="mode",l="providers",n="features",d=e=>Array.isArray(e)?e.filter(e=>"string"==typeof e):[],u=(e,t)=>{let r,o;return r=`filter[${e}][in]`,""===(o=d(t).join(","))?[]:[[r,o]]},c=e=>{switch(e.id){case a:return u(e.id,(0,s.q)(d(e.value)));case l:case n:return u(e.id,e.value);default:return[]}},p=e=>Object.fromEntries(e.flatMap(c)),m=(e,t)=>d(e.find(e=>e.id===t)?.value),g=(e,t,r)=>{let o=e.filter(e=>e.id!==t);return(Array.isArray(r)?0===r.length:""===r.trim())?o:[...o,{id:t,value:r}]};e.s(["g",0,n,"e",0,a,"T",0,l,"i",0,["model_group","mode","providers","max_input_tokens","max_output_tokens","input_cost_per_token","output_cost_per_token","rpm","tpm"],"U",0,e=>e.split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),"o",0,m,"$",0,p,"R",0,g],348594);let h="/public/v1/model_hub",f=["publicModelHub","list"],b=[{id:"model_group",desc:!1}],x=async(e,t)=>{try{return await r.Mw.get(h,{query:e,signal:t})}catch(e){throw t.aborted||console.error("There was an error fetching the public model data",e),e}};e.s(["H",0,50,"Z",0,h,"t",0,e=>{let t=(0,i.f)({queryKey:f,fetchPage:x,serializeFilters:p,defaultSorting:b,defaultPageSize:50,enabled:e}),{onColumnFiltersChange:r}=t,s=(0,o.useCallback)((e,t)=>r(r=>g(r,e,t)),[r]),d=(0,o.useCallback)(e=>s(l,e),[s]),u=(0,o.useCallback)(e=>s(a,e),[s]),c=(0,o.useCallback)(e=>s(n,e),[s]);return{...t,providerValues:m(t.columnFilters,l),onProvidersChange:d,modeValues:m(t.columnFilters,a),onModesChange:u,featureValues:m(t.columnFilters,n),onFeaturesChange:c,hasActiveQuery:""!==t.searchValue.trim()||t.columnFilters.length>0}}],831538);let k=["providers","modes","features"];e.s(["f",0,e=>{let[o,i,s]=(0,t.f)({queries:k.map(t=>({queryKey:["publicModelHub","facet",t],queryFn:({signal:e})=>r.Mw.get(`${h}/${t}`,{query:{page_size:100},signal:e}),enabled:e,staleTime:1/0}))}).map(e=>e.data?.data??[]);return{providers:o,modes:i,features:s}}],466098)},735450,e=>{var t=e.i(411382),r=e.i(772325),o=e.i(54105),i=e.i(240192),s=e.i(138620);let a="https://docs.litellm.ai/docs/decisions",l=`${a}#supported-providers`,n=`${a}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),u=t.z.object({litellm_provider:o.z.string(),mode:o.z.string().nullish(),supported_endpoints:r.z.array(o.z.string()).nullish()}),c="decisions",p=["evaluation",c],m=e=>null!=e&&p.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),h=e=>{let t=i.z.record(o.z.string(),s.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let r=u.safeParse(t);return r.success?[[e,r.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},b=t.z.object({decision_model_providers:r.z.array(o.z.string())}),x=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),k=(e,t,r)=>({row:e,routeOrder:f[e.route],rankInProvider:r.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,a,"d",0,c,"Z",0,l,"L",0,n,"B",0,"playground?tab=system-one","F",0,function(e){let t=h(e),r=t.filter(([,e])=>g(e)),o=new Map([...new Set(r.map(([,e])=>e.litellm_provider))].map(e=>{let r=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:r.every(([,e])=>g(e))}]}));return{models:new Set(r.map(([e])=>e)),providers:o}},"W",0,(e,t)=>h(e).filter(([,e])=>g(e)).map(([e,r])=>({model:e,provider:r.litellm_provider,route:t?.has(r.litellm_provider)?"decisions":(r.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(x).map(k).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?p:[e]),"m",0,g,"_",0,m,"r",0,(e,t,r)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||r.some(t=>e.models.has(t)),"P",0,e=>{let t=b.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function r(e,r){let o=t(e);if(""===o)return!0;let i=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!i.some(e=>e.includes(o))||o.split(/\s+/).every(e=>i.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,o){return e.filter(e=>r(t,o(e)))},"h",0,r,"e",0,function(e,r,o){let i=t(r);if(""===i)return[...e];let s=e=>{let t=o(e).toLowerCase();return 1e3*(t===i)+100*!!t.startsWith(i)+(1e3-t.length)};return[...e].sort((e,t)=>s(t)-s(e))}])}])})();