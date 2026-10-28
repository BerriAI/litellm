(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,[586448,e=>{var t=e.i(271645),r=e.i(540143),i=e.i(869230),o=e.i(915823),s=e.i(619273);function a(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var n=class extends o.f{#e;#t;#r;#i;#o;#s;#a;#n;#l;#d=[];constructor(e,t,r){super(),this.#e=e,this.#i=r,this.#r=[],this.#o=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#o.forEach(e=>{e.subscribe(t=>{this.#c(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#o.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#i=t,r.f.batch(()=>{let e=this.#o,t=this.#u(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),i=r.map(e=>e.getCurrentResult()),o=e.length!==r.length,n=r.some((t,r)=>t!==e[r]),l=o||n,d=!!l||i.some((e,t)=>{let r=this.#t[t];return!r||!(0,s.U)(e,r)});(l||d)&&(l&&(this.#d=t,this.#o=r),this.#t=i,this.hasListeners()&&(l&&(a(e,r).forEach(e=>{e.destroy()}),a(r,e).forEach(e=>{e.subscribe(t=>{this.#c(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#o.map(e=>e.getCurrentQuery())}getObservers(){return this.#o}getOptimisticResult(e,t){let r=this.#u(e),i=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),o=r.map(e=>e.defaultedQueryOptions.queryHash);return[i,e=>this.#h(e??i,t,o),()=>this.#f(i,r)]}#f(e,t){return t.map((r,i)=>{let o=e[i];return r.defaultedQueryOptions.notifyOnChangeProps?o:r.observer.trackResult(o,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#h(e,t,r){if(t){let i=this.#l,o=void 0!==r&&void 0!==i&&(i.length!==r.length||r.some((e,t)=>e!==i[t]));return(!this.#s||this.#t!==this.#n||o||t!==this.#a)&&(this.#a=t,this.#n=this.#t,void 0!==r&&(this.#l=r),this.#s=(0,s.B)(this.#s,t(e))),this.#s}return e}#m(){return this.#i?.combine!==void 0&&this.#o.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#u(e){let t=new Map;this.#o.forEach(e=>{let r=e.options.queryHash;if(!r)return;let i=t.get(r);i?i.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let o=this.#e.defaultQueryOptions(e),s=t.get(o.queryHash)?.shift()??new i.f(this.#e,o);r.push({defaultedQueryOptions:o,observer:s})}),r}#c(e,t){let r=this.#o.indexOf(e);if(-1!==r){var i;let e;this.#t=(i=this.#t,(e=i.slice(0))[r]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#f(this.#t,this.#d),t=this.#m(),i=this.#s,o=t?i:this.#h(e,this.#i?.combine);(t||i!==o)&&r.f.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},l=e.i(912598),d=e.i(381384),c=e.i(673664),u=e.i(427001),p=e.i(254440);e.s(["f",0,function({queries:e,...o},a){let h=(0,l.d)(a),f=(0,d.f)(),m=(0,c.p)(),g=t.useMemo(()=>e.map(e=>{let t=h.defaultQueryOptions(e);return t._optimisticResults=f?"isRestoring":"optimistic",t}),[e,h,f]);g.forEach(e=>{(0,p.p)(e);let t=h.getQueryCache().get(e.queryHash);(0,u.l)(e,m,t)}),(0,u.E)(m);let[b]=t.useState(()=>new n(h,g,o)),[_,y,k]=b.getOptimisticResult(g,o.combine),x=!f&&!1!==o.subscribed;t.useSyncExternalStore(t.useCallback(e=>x?b.subscribe(r.f.batchCalls(e)):s.z,[b,x]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(g,o)},[g,o,b]);let v=_.some((e,t)=>(0,p.Q)(g[t],e))?_.flatMap((e,t)=>{let r=g[t];if(r&&(0,p.Q)(r,e)){let e=new i.f(h,r);return(0,p.X)(r,e,m)}return[]}):[];if(v.length>0)throw Promise.all(v);let w=_.find((e,t)=>{let r=g[t];return r&&(0,u.P)({result:e,errorResetBoundary:m,throwOnError:r.throwOnError,query:h.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(w?.error)throw w.error;return y(k())}],586448)},180127,e=>{let t=(0,e.i(475254).f)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["f",0,t])},871689,e=>{var t=e.i(180127);e.s(["f",()=>t.f])},972520,e=>{let t=(0,e.i(475254).f)("arrow-right",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]]);e.s(["f",0,t],972520)},269638,e=>{let t=(0,e.i(475254).f)("circle-check-big",[["path",{d:"M21.801 10A10 10 0 1 1 17 3.335",key:"yps3ct"}],["path",{d:"m9 11 3 3L22 4",key:"1pflzl"}]]);e.s(["f",0,t],269638)},595468,e=>{var t=e.i(123287);e.s(["f",()=>t.f])},101048,e=>{var t=e.i(123287);e.s(["f",()=>t.f])},798031,e=>{let t=(0,e.i(475254).f)("circle-x",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m15 9-6 6",key:"1uzhvr"}],["path",{d:"m9 9 6 6",key:"z0biqf"}]]);e.s(["f",0,t])},832724,e=>{var t=e.i(798031);e.s(["f",()=>t.f])},339402,e=>{let t=(0,e.i(475254).f)("code",[["path",{d:"m16 18 6-6-6-6",key:"eg8j8"}],["path",{d:"m8 6-6 6 6 6",key:"ppft3o"}]]);e.s(["f",0,t])},758472,e=>{var t=e.i(339402);e.s(["f",()=>t.f])},575822,e=>{let t=(0,e.i(475254).f)("coins",[["circle",{cx:"8",cy:"8",r:"6",key:"3yglwk"}],["path",{d:"M18.09 10.37A6 6 0 1 1 10.34 18",key:"t5s6rm"}],["path",{d:"M7 6h1v4",key:"1obek4"}],["path",{d:"m16.71 13.88.7.71-2.82 2.82",key:"1rbuyh"}]]);e.s(["f",0,t],575822)},133073,e=>{let t=(0,e.i(475254).f)("database-zap",[["ellipse",{cx:"12",cy:"5",rx:"9",ry:"3",key:"msslwz"}],["path",{d:"M3 5V19A9 3 0 0 0 15 21.84",key:"14ibmq"}],["path",{d:"M21 5V8",key:"1marbg"}],["path",{d:"M21 12L18 17H22L19 22",key:"zafso"}],["path",{d:"M3 12A9 3 0 0 0 14.59 14.87",key:"1y4wr8"}]]);e.s(["f",0,t],133073)},77705,e=>{let t=(0,e.i(475254).f)("eye-off",[["path",{d:"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49",key:"ct8e1f"}],["path",{d:"M14.084 14.158a3 3 0 0 1-4.242-4.242",key:"151rxh"}],["path",{d:"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143",key:"13bj9a"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]]);e.s(["f",0,t],77705)},286536,e=>{let t=(0,e.i(475254).f)("eye",[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["f",0,t],286536)},803408,e=>{let t=(0,e.i(475254).f)("image",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",ry:"2",key:"1m3agn"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}]]);e.s(["f",0,t])},332102,e=>{let t=(0,e.i(475254).f)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["f",0,t],332102)},181692,e=>{let t=(0,e.i(475254).f)("key",[["path",{d:"m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4",key:"g0fldk"}],["path",{d:"m21 2-9.6 9.6",key:"1j0ho8"}],["circle",{cx:"7.5",cy:"15.5",r:"5.5",key:"yqb3hr"}]]);e.s(["f",0,t])},834161,e=>{var t=e.i(181692);e.s(["f",()=>t.f])},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},221345,e=>{let t=(0,e.i(475254).f)("link",[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]]);e.s(["f",0,t],221345)},63804,e=>{let t=(0,e.i(475254).f)("list-checks",[["path",{d:"m3 17 2 2 4-4",key:"1jhpwq"}],["path",{d:"m3 7 2 2 4-4",key:"1obspn"}],["path",{d:"M13 6h8",key:"15sg57"}],["path",{d:"M13 12h8",key:"h98zly"}],["path",{d:"M13 18h8",key:"oe0vm4"}]]);e.s(["f",0,t],63804)},270756,e=>{let t=(0,e.i(475254).f)("lock",[["rect",{width:"18",height:"11",x:"3",y:"11",rx:"2",ry:"2",key:"1w4ew1"}],["path",{d:"M7 11V7a5 5 0 0 1 10 0v4",key:"fwvmzm"}]]);e.s(["f",0,t],270756)},686311,e=>{let t=(0,e.i(475254).f)("message-square",[["path",{d:"M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",key:"1lielz"}]]);e.s(["f",0,t],686311)},387951,e=>{let t=(0,e.i(475254).f)("mic",[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]]);e.s(["f",0,t],387951)},299023,e=>{let t=(0,e.i(475254).f)("minus",[["path",{d:"M5 12h14",key:"1ays0h"}]]);e.s(["f",0,t],299023)},64763,e=>{let t=(0,e.i(475254).f)("panel-right",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M15 3v18",key:"14nvp0"}]]);e.s(["f",0,t],64763)},82735,e=>{let t=(0,e.i(475254).f)("paperclip",[["path",{d:"m16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551",key:"1miecu"}]]);e.s(["f",0,t],82735)},431343,e=>{let t=(0,e.i(475254).f)("play",[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]]);e.s(["f",0,t],431343)},996781,784875,e=>{var t=e.i(475254);let r=(0,t.f)("repeat",[["path",{d:"m17 2 4 4-4 4",key:"nntrym"}],["path",{d:"M3 11v-1a4 4 0 0 1 4-4h14",key:"84bu3i"}],["path",{d:"m7 22-4-4 4-4",key:"1wqhfi"}],["path",{d:"M21 13v1a4 4 0 0 1-4 4H3",key:"1rx37r"}]]);e.s(["f",0,r],996781);let i=(0,t.f)("scan-search",[["path",{d:"M3 7V5a2 2 0 0 1 2-2h2",key:"aa7l1z"}],["path",{d:"M17 3h2a2 2 0 0 1 2 2v2",key:"4qcy5o"}],["path",{d:"M21 17v2a2 2 0 0 1-2 2h-2",key:"6vwrx8"}],["path",{d:"M7 21H5a2 2 0 0 1-2-2v-2",key:"ioqczr"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}],["path",{d:"m16 16-1.9-1.9",key:"1dq9hf"}]]);e.s(["f",0,i],784875)},991810,e=>{let t=(0,e.i(475254).f)("rotate-cw",[["path",{d:"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8",key:"1p45f6"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}]]);e.s(["f",0,t],991810)},251854,e=>{let t=(0,e.i(475254).f)("save",[["path",{d:"M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z",key:"1c8476"}],["path",{d:"M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7",key:"1ydtos"}],["path",{d:"M7 3v4a1 1 0 0 0 1 1h7",key:"t51u73"}]]);e.s(["f",0,t])},356909,e=>{var t=e.i(251854);e.s(["f",()=>t.f])},514764,e=>{let t=(0,e.i(475254).f)("send",[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]]);e.s(["f",0,t],514764)},661263,e=>{let t=(0,e.i(475254).f)("split",[["path",{d:"M16 3h5v5",key:"1806ms"}],["path",{d:"M8 3H3v5",key:"15dfkv"}],["path",{d:"M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3",key:"1qrqzj"}],["path",{d:"m15 9 6-6",key:"ko1vev"}]]);e.s(["f",0,t],661263)},823429,e=>{let t=(0,e.i(475254).f)("square-pen",[["path",{d:"M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7",key:"1m0v6g"}],["path",{d:"M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z",key:"ohrbg2"}]]);e.s(["f",0,t])},110031,e=>{var t=e.i(823429);e.s(["f",()=>t.f])},417835,e=>{let t=(0,e.i(475254).f)("timer",[["line",{x1:"10",x2:"14",y1:"2",y2:"2",key:"14vaq8"}],["line",{x1:"12",x2:"15",y1:"14",y2:"11",key:"17fdiu"}],["circle",{cx:"12",cy:"14",r:"8",key:"1e1u0o"}]]);e.s(["f",0,t],417835)},878894,e=>{var t=e.i(582458);e.s(["f",()=>t.f])},569074,e=>{let t=(0,e.i(475254).f)("upload",[["path",{d:"M12 3v12",key:"1x0j5s"}],["path",{d:"m17 8-5-5-5 5",key:"7q97r8"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}]]);e.s(["f",0,t],569074)},923426,e=>{let t=(0,e.i(475254).f)("user-round",[["circle",{cx:"12",cy:"8",r:"5",key:"1hypcn"}],["path",{d:"M20 21a8 8 0 0 0-16 0",key:"rfgkzh"}]]);e.s(["f",0,t],923426)},382373,e=>{let t=(0,e.i(475254).f)("volume-2",[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]]);e.s(["f",0,t],382373)},563458,e=>{var t=e.i(722219);e.s(["z",0,t])},835460,e=>{var t=e.i(722219);e.s(["z",0,t])},864261,e=>{var t=e.i(751247),r=e.i(135214),i=e.i(441228);e.s(["f",0,e=>{let{userRole:o}=(0,r.f)(),s=(0,i.f)();return(0,t.q)(o,e,s)}])},466828,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(678784);let o=(0,e.i(475254).f)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var s=e.i(650056);let a={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};var n=e.i(488012);e.s(["f",0,({code:e,language:l})=>{let d=(0,n.f)(a),[c,u]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-border bg-muted overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),u(!0),setTimeout(()=>u(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground z-raised","aria-label":"Copy code",children:c?(0,t.jsx)(i.f,{size:16}):(0,t.jsx)(o,{size:16})}),(0,t.jsx)(s.f,{language:l,style:d,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",background:"transparent"},codeTagProps:{style:{background:"transparent"}},showLineNumbers:!0,children:e})]})}],466828)},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let r,{apiKeySource:i,accessToken:o,apiKey:s,inputMessage:a,chatHistory:n,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedVoice:p,endpointType:h,selectedModel:f,selectedSdk:m,proxySettings:g,customHeaders:b}=e,_="session"===i?o:s,y=window.location.origin,k=g?.LITELLM_UI_API_DOC_BASE_URL;k&&k.trim()?y=k:g?.PROXY_BASE_URL&&(y=g.PROXY_BASE_URL);let x=a||"Your prompt here",v=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};l.length>0&&(j.tags=l),d.length>0&&(j.vector_stores=d),c.length>0&&(j.guardrails=c),u.length>0&&(j.policies=u);let C=f||"your-model-name",E=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",S="azure"===m?`import openai

client = openai.AzureOpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"${E}
)`:`import openai

client = openai.OpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"${E}
)`;switch(h){case t.Q.CHAT:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:x}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(i,null,4)}${t}
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
#                     "text": "${v}"
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
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:x}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(i,null,4)}${t}
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
#                 {"type": "input_text", "text": "${v}"},
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
`;break}case t.Q.IMAGE:r="azure"===m?`
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
prompt = "${v}"

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
`;break;case t.Q.IMAGE_EDITS:r="azure"===m?`
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
prompt = "${v}"

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
prompt = "${v}"

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
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${S}
${r}`}])},891547,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(602869),o=e.i(845150);e.s(["f",0,({onChange:e,value:s,className:a,accessToken:n,disabled:l})=>{let[d,c]=(0,r.useState)([]),[u,p]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(n){p(!0);try{let e=await (0,i.gx)(n);e.guardrails&&c(e.guardrails)}catch(e){console.error("Error fetching guardrails:",e)}finally{p(!1)}}})()},[n]),(0,t.jsx)("div",{className:"min-w-0",children:(0,t.jsx)(o.f,{disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onValueChange:t=>{e(t)},value:s,loading:u,className:a,options:d.flatMap(e=>{let t=e.guardrail_name;return null==t||""===t?[]:[{label:t,value:t}]})})})}])},248467,e=>{var t=e.i(843476),r=e.i(299023),i=e.i(107233),o=e.i(519455),s=e.i(793479);e.s(["f",0,({value:e=[],onChange:a})=>{let n=(t,r)=>a?.(e.map((e,i)=>i===t?r:e));return(0,t.jsxs)("div",{className:"space-y-2",children:[e.map(([i,l],d)=>(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(s.f,{placeholder:"Header Name",value:i,onChange:e=>n(d,[e.target.value,l])}),(0,t.jsx)(s.f,{placeholder:"Header Value",value:l,onChange:e=>n(d,[i,e.target.value])}),(0,t.jsx)(o.K,{type:"button",variant:"ghost",size:"icon-sm",onClick:()=>a?.(e.filter((e,t)=>t!==d)),"aria-label":`Remove header ${d+1}`,children:(0,t.jsx)(r.f,{})})]},d)),(0,t.jsxs)(o.K,{type:"button",variant:"outline",onClick:()=>a?.([...e,["",""]]),children:[(0,t.jsx)(i.f,{}),"Add Header"]})]})}])},459161,e=>{var t=e.i(223305);e.i(356449);var r=e.i(857223),i=e.i(602869),o=e.i(417385),s=e.i(441773),a=e.i(892034),n=e.i(757625);async function l(e,d,c,u,p=[],h,f,m,g,b,_,y,k,x,v,w,j,C,E,S,R,z,O,M=!0,N,I,A){if(!u)throw Error("Virtual Key is required");if(!c||""===c.trim())throw Error("Model is required. Please select a model before sending a request.");console.log=function(){};let L=S||(0,i.V5)(),T=S||(0,i._h)(),q=(0,n.p)(p,I),P=new r.f.OpenAI({apiKey:u,baseURL:L,dangerouslyAllowBrowser:!0,defaultHeaders:q,...A?{fetch:(0,t.f)(A)}:{}});try{let t,r,i,o=Date.now(),n=!1,l=!1,u=e.flatMap(e=>{let t="assistant"===e.role?(e.mcpEvents??[]).flatMap(e=>{let t=e.item,r=t?.id||e.item_id;return"response.output_item.done"===e.type&&t?.type==="mcp_call"&&r&&t.name&&"string"==typeof t.arguments&&"string"==typeof t.output?[{type:"function_call",call_id:r,name:t.name,arguments:t.arguments},{type:"function_call_output",call_id:r,output:t.output}]:[]}):[];return[...t,...t.length>0&&""===e.content?[]:[{role:e.role,content:e.content,type:"message"}]]}),p=[];if(x&&x.length>0)if(x.includes("__all__")){let e={type:"mcp",server_label:"litellm",server_url:`${T}/mcp`,require_approval:"never"};p.push(e)}else x.forEach(e=>{if(e.startsWith("toolset:")){let t=e.slice(8),r=O?.find(e=>e.toolset_id===t),i=r?.toolset_name||t,o={type:"mcp",server_label:i,server_url:`${T}/mcp/${encodeURIComponent(i)}`,require_approval:"never"};p.push(o)}else{let t=R?.find(t=>t.server_id===e),r=t?.server_name||e,i=z?.[e]||[],o={type:"mcp",server_label:r,server_url:`${T}/mcp/${encodeURIComponent(r)}`,require_approval:"never",...i.length>0?{allowed_tools:i}:{}};p.push(o)}});C&&p.push({type:"code_interpreter",container:{type:"auto"}});let S={model:c,input:u,litellm_trace_id:b,...v?{previous_response_id:v}:{},..._?{vector_store_ids:_}:{},...y?{guardrails:y}:{},...k?{policies:k}:{},...p.length>0?{tools:p,tool_choice:"auto"}:{}},I=M?await P.responses.create({...S,stream:!0},{signal:h}):await (async()=>{let e=await P.responses.create({...S,stream:!1},{signal:h}).withResponse();return l=null!==e.response.headers.get("x-litellm-cache-key"),e.data})(),A=M?I:(r=(t=I.output??[]).filter(e=>"message"===e.type).flatMap(e=>e.content??[]).filter(e=>"output_text"===e.type).map(e=>e.text??"").join(""),i=t.filter(e=>"reasoning"===e.type).flatMap(e=>e.summary??[]).map(e=>e.text??"").join(""),[...t.map(e=>({type:"response.output_item.done",item:e})),...i?[{type:"response.reasoning.delta",delta:i}]:[],...r?[{type:"response.output_text.delta",delta:r}]:[],{type:"response.completed",response:I}]),L="",q={code:"",containerId:""};for await(let e of A)if("object"==typeof e&&null!==e){if((e.type?.startsWith("response.mcp_")||"response.output_item.done"===e.type&&(e.item?.type==="mcp_list_tools"||e.item?.type==="mcp_call"))&&j){let t={type:e.type,sequence_number:e.sequence_number,output_index:e.output_index,item_id:e.item_id||e.item?.id,item:e.item,delta:e.delta,arguments:e.arguments,timestamp:Date.now()};j(t)}"response.output_item.done"===e.type&&e.item?.type==="mcp_call"&&e.item?.name&&(L=e.item.name),$=q;var $,D=q="response.output_item.done"===e.type&&e.item?.type==="code_interpreter_call"?{code:e.item.code||"",containerId:e.item.container_id||""}:$;if("response.output_item.done"===e.type&&e.item?.type==="message"&&e.item?.content&&E){for(let t of e.item.content)if("output_text"===t.type&&t.annotations){let e=t.annotations.filter(e=>"container_file_citation"===e.type);(e.length>0||D.code)&&E({code:D.code,containerId:D.containerId,annotations:e})}}if("response.role.delta"===e.type)continue;if("response.output_text.delta"===e.type&&"string"==typeof e.delta){let t=e.delta;if(t.length>0&&(d("assistant",t,c),!n)){n=!0;let e=Date.now()-o;m&&M&&m(e)}}if("response.reasoning.delta"===e.type&&"delta"in e){let t=e.delta;"string"==typeof t&&f&&f(t)}if("response.completed"===e.type&&"response"in e){let t=e.response,r=t.usage;if(t.id&&w&&w(t.id),r&&g){let e={completionTokens:r.output_tokens,promptTokens:r.input_tokens,totalTokens:r.total_tokens,...(0,s.h)(r),...l?{servedFromResponseCache:!0}:{}},t=r.output_tokens_details?.reasoning_tokens??r.completion_tokens_details?.reasoning_tokens;t&&(e.reasoningTokens=t);let i=(0,a.f)(r.cost);void 0!==i&&(e.cost=i),g(e,L)}}}return N&&N(Date.now()-o),I}catch(e){throw h?.aborted||o.f.fromError(`Error occurred while generating model response. Please try again. Error: ${e}`),e}}e.s(["f",0,l],459161)},892034,757625,e=>{e.s(["f",0,function(e){if("number"==typeof e)return Number.isFinite(e)?e:void 0;if("string"!=typeof e)return;let t=e.trim();if(""===t)return;let r=Number(t);return Number.isFinite(r)?r:void 0}],892034);let t=e=>Array.isArray(e)&&2===e.length&&e.every(e=>"string"==typeof e);e.s(["p",0,(e,t)=>({...e&&e.length>0?{"x-litellm-tags":e.join(",")}:{},...t}),"V",0,e=>Object.fromEntries(e.map(([e,t])=>[e.trim(),t]).filter(([e])=>""!==e)),"k",0,e=>{if(!e)return[];try{let r=JSON.parse(e);return Array.isArray(r)?r.filter(t):[]}catch{return[]}},"s",0,(e,t)=>{let r=new Set(Object.keys(t).map(e=>e.toLowerCase()));return{...Object.fromEntries(Object.entries(e).filter(([e])=>!r.has(e.toLowerCase()))),...t}}],757625)},611052,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(417385),o=e.i(768371),s=e.i(431703),a=e.i(871689),n=e.i(972520),l=e.i(643531),d=e.i(834161),c=e.i(306228),u=e.i(270756),p=e.i(37727),h=e.i(776639),f=e.i(450240),m=e.i(699375);e.s(["f",0,({server:e,open:g,onClose:b,onSuccess:_})=>{let[y,k]=(0,r.useState)(1),[x,v]=(0,r.useState)(""),[w,j]=(0,r.useState)(!0),[C,E]=(0,r.useState)(!1),S=(0,r.useId)(),R=e.alias||e.server_name||"Service",z=R.charAt(0).toUpperCase(),O=()=>{k(1),v(""),j(!0),E(!1),b()},M=async()=>{if(!x.trim())return void i.f.error("Please enter your API key");E(!0);try{await o.Q.POST("/v1/mcp/server/{server_id}/user-credential",{params:{path:{server_id:e.server_id}},body:{credential:x.trim(),save:w}}),i.f.success(`Connected to ${R}`),_(e.server_id),O()}catch(e){i.f.error((e=>{if(e instanceof s.a){let t=e.body?.detail?.error;if(t)return t}return e instanceof Error&&e.message?e.message:"Failed to connect"})(e))}finally{E(!1)}};return(0,t.jsx)(h.N,{open:g,onOpenChange:e=>!e&&O(),children:(0,t.jsx)(h.d,{className:"max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[480px] byok-modal",showCloseButton:!1,children:(0,t.jsxs)("div",{className:"relative p-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===y?(0,t.jsxs)("button",{onClick:()=>k(1),className:"flex items-center gap-1 text-muted-foreground hover:text-foreground text-sm",children:[(0,t.jsx)(a.f,{className:"size-3.5"})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${1===y?"bg-info":"bg-border"}`}),(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${2===y?"bg-info":"bg-border"}`})]}),(0,t.jsx)("button",{onClick:O,className:"text-muted-foreground hover:text-foreground",children:(0,t.jsx)(p.X,{className:"size-4"})})]}),1===y?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-linear-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow-sm",children:"L"}),(0,t.jsx)(n.f,{className:"size-4.5 text-muted-foreground"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-linear-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow-sm",children:z})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:["Connect ",R]}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["LiteLLM needs access to ",R," to complete your request."]}),(0,t.jsx)("div",{className:"bg-muted rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-foreground mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-muted-foreground text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to"," ",R,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-success",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,r)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-foreground",children:[(0,t.jsx)(l.f,{className:"size-3.5 shrink-0 text-success"}),e]},r))})]}),(0,t.jsxs)("button",{onClick:()=>k(2),className:"w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:["Continue to Authentication ",(0,t.jsx)(n.f,{className:"size-4"})]}),(0,t.jsx)("button",{onClick:O,className:"mt-3 w-full text-muted-foreground hover:text-foreground text-sm py-2",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-info/10 flex items-center justify-center mb-4",children:(0,t.jsx)(d.f,{className:"size-5 text-info"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["Enter your ",R," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsxs)("label",{htmlFor:S,className:"block text-sm font-semibold text-foreground mb-2",children:[R," API Key"]}),(0,t.jsx)(f.f,{id:S,placeholder:"Enter your API key",value:x,onChange:e=>v(e.target.value),groupClassName:"rounded-lg"}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-info hover:text-info/80 text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key? ",(0,t.jsx)(c.f,{className:"size-3.5"})]})]}),(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-foreground",children:"Save key for future use"})]}),(0,t.jsx)(m.f,{checked:w,onCheckedChange:j,"aria-label":"Save key for future use"})]}),(0,t.jsxs)("div",{className:"bg-info/10 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(u.f,{className:"mt-0.5 size-4 shrink-0 text-info"}),(0,t.jsx)("p",{className:"text-sm text-info",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)("button",{onClick:M,disabled:C,className:"w-full bg-info hover:bg-info/80 disabled:opacity-60 text-info-foreground font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:[(0,t.jsx)(u.f,{className:"size-4"})," Connect & Authorize"]})]})]})})})}])},921511,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(864261),o=e.i(602869),s=e.i(845150);function a(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let r=e.version_number??1,i=e.version_status??"draft";return{label:`${e.policy_name} — v${r} (${i})${e.description?` — ${e.description}`:""}`,value:"production"===i?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["U",0,({onChange:e,value:n,className:l,accessToken:d,disabled:c,onPoliciesLoaded:u})=>{let p=(0,i.f)("viewPolicies"),[h,f]=(0,r.useState)([]),[m,g]=(0,r.useState)(!1);return((0,r.useEffect)(()=>{(async()=>{if(d&&p){g(!0);try{let e=await (0,o.$)(d);e.policies&&(f(e.policies),u?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{g(!1)}}})()},[d,p,u]),p)?(0,t.jsx)("div",{className:"min-w-0",children:(0,t.jsx)(s.f,{disabled:c,placeholder:c?"Setting policies is a premium feature.":"Select policies (production or published versions)",onValueChange:t=>{e(t)},value:n,loading:m,className:l,options:a(h)})}):null},"A",0,a])},710540,e=>{var t=e.i(843476),r=e.i(677572),i=e.i(196631);e.s(["w",0,function({className:e,...r}){return(0,t.jsx)("div",{className:(0,i.c)("flex min-h-0 min-w-0 w-full flex-col gap-6 p-4 sm:p-6",e),...r})},"g",0,function({className:e,...r}){return(0,t.jsx)("div",{className:(0,i.c)("flex min-h-0 min-w-0 flex-1 flex-col gap-6",e),...r})},"j",0,function({className:e,...o}){return(0,t.jsx)(r.i,{className:(0,i.c)("min-h-0 min-w-0 flex-1 gap-6",e),...o})},"m",0,function({className:e,...o}){return(0,t.jsx)(r.S,{className:(0,i.c)("flex min-h-0 min-w-0 flex-1 flex-col",e),...o})},"_",0,function({className:e,...o}){return(0,t.jsx)(r.U,{variant:"line",className:(0,i.c)("h-10 w-full shrink-0 justify-start gap-6 overflow-x-auto border-b border-border p-0",e),...o})},"H",0,function({className:e,...o}){return(0,t.jsx)(r.k,{className:(0,i.c)("flex-none px-0 font-normal group-data-horizontal/tabs:after:bottom-0",e),...o})}])},263005,e=>{var t=e.i(843476),r=e.i(196631);e.s(["K",0,function({className:e,...i}){return(0,t.jsx)("div",{className:(0,r.c)("shrink-0",e),...i})},"V",0,function({className:e,...i}){return(0,t.jsx)("div",{role:"group","aria-label":"Page controls",className:(0,r.c)("mt-5 flex min-h-9 items-center gap-2",e),...i})},"h",0,function({className:e,...i}){return(0,t.jsx)("p",{className:(0,r.c)("mt-1.5 text-sm text-muted-foreground",e),...i})},"I",0,function({className:e,...i}){return(0,t.jsx)("h1",{className:(0,r.c)("flex items-center gap-2.5 text-2xl font-semibold tracking-tight text-foreground [&_svg]:size-5 [&_svg]:flex-none [&_svg]:stroke-[1.75]",e),...i})}])},450240,e=>{var t=e.i(843476),r=e.i(286536),i=e.i(77705),o=e.i(271645),s=e.i(950594);let a=o.forwardRef(({className:e,groupClassName:a,disabled:n,...l},d)=>{let[c,u]=o.useState(!1);return(0,t.jsxs)(s.H,{className:a,children:[(0,t.jsx)(s.M,{...l,ref:d,type:c?"text":"password",disabled:n,className:e}),(0,t.jsx)(s.B,{align:"inline-end",children:(0,t.jsx)(s.w,{size:"icon-xs",disabled:n,"aria-label":c?"Hide password":"Show password",onClick:()=>u(e=>!e),children:c?(0,t.jsx)(i.f,{}):(0,t.jsx)(r.f,{})})})]})});a.displayName="PasswordInput",e.s(["f",0,a])},868499,e=>{var t=e.i(843476),r=e.i(366250),i=e.i(581239),o=e.i(156736),s=e.i(209793),a=e.i(784324),n=e.i(264951),l=e.i(77173);let d=e.i(313488).f;e.s(["Backdrop",()=>i.f,"Close",()=>o.f,"Description",()=>s.f,"Popup",()=>a.f,"Portal",()=>n.f,"Root",0,function(e){return(0,r.f)(e,"alert-dialog")},"Title",()=>l.f,"Trigger",0,d],734604);var c=e.i(734604),c=c,u=c,p=c,h=c,f=c,m=c,g=c,b=c,_=e.i(196631),y=e.i(519455);function k({...e}){return(0,t.jsx)(p.Portal,{"data-slot":"alert-dialog-portal",...e})}function x({className:e,...r}){return(0,t.jsx)(h.Backdrop,{"data-slot":"alert-dialog-overlay",className:(0,_.c)("fixed inset-0 isolate z-popup bg-black/10 duration-100 supports-backdrop-filter:backdrop-blur-xs data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",e),...r})}e.s(["m",0,function({...e}){return(0,t.jsx)(c.Root,{"data-slot":"alert-dialog",...e})},"s",0,function({className:e,variant:r="default",size:i="default",...o}){return(0,t.jsx)(b.Close,{"data-slot":"alert-dialog-action",className:(0,_.c)(e),render:(0,t.jsx)(y.K,{variant:r,size:i}),...o})},"D",0,function({className:e,variant:r="outline",size:i="default",...o}){return(0,t.jsx)(b.Close,{"data-slot":"alert-dialog-cancel",className:(0,_.c)(e),render:(0,t.jsx)(y.K,{variant:r,size:i}),...o})},"B",0,function({className:e,size:r="default",...i}){return(0,t.jsxs)(k,{children:[(0,t.jsx)(x,{}),(0,t.jsx)(f.Popup,{"data-slot":"alert-dialog-content","data-size":r,className:(0,_.c)("group/alert-dialog-content fixed top-1/2 left-1/2 z-popup grid w-full -translate-x-1/2 -translate-y-1/2 gap-6 rounded-xl bg-popover p-6 text-popover-foreground ring-1 ring-foreground/10 duration-100 outline-none data-[size=default]:max-w-xs data-[size=sm]:max-w-xs data-[size=default]:sm:max-w-lg data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",e),...i})]})},"I",0,function({className:e,...r}){return(0,t.jsx)(g.Description,{"data-slot":"alert-dialog-description",className:(0,_.c)("text-sm text-balance text-muted-foreground md:text-pretty *:[a]:underline *:[a]:underline-offset-3 *:[a]:hover:text-foreground",e),...r})},"_",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"alert-dialog-footer",className:(0,_.c)("flex flex-col-reverse gap-2 group-data-[size=sm]/alert-dialog-content:grid group-data-[size=sm]/alert-dialog-content:grid-cols-2 sm:flex-row sm:justify-end",e),...r})},"O",0,function({className:e,...r}){return(0,t.jsx)("div",{"data-slot":"alert-dialog-header",className:(0,_.c)("grid grid-rows-[auto_1fr] place-items-center gap-1.5 text-center has-data-[slot=alert-dialog-media]:grid-rows-[auto_auto_1fr] has-data-[slot=alert-dialog-media]:gap-x-6 sm:group-data-[size=default]/alert-dialog-content:place-items-start sm:group-data-[size=default]/alert-dialog-content:text-left sm:group-data-[size=default]/alert-dialog-content:has-data-[slot=alert-dialog-media]:grid-rows-[auto_1fr]",e),...r})},"g",0,function({className:e,...r}){return(0,t.jsx)(m.Title,{"data-slot":"alert-dialog-title",className:(0,_.c)("text-lg font-medium sm:group-data-[size=default]/alert-dialog-content:group-has-data-[slot=alert-dialog-media]/alert-dialog-content:col-start-2",e),...r})},"t",0,function({...e}){return(0,t.jsx)(u.Trigger,{"data-slot":"alert-dialog-trigger",...e})}],868499)},916940,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(602869),o=e.i(845150);e.s(["f",0,({onChange:e,value:s,className:a,accessToken:n,placeholder:l="Select vector stores",disabled:d=!1})=>{let[c,u]=(0,r.useState)([]),[p,h]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(n){h(!0);try{let e=await (0,i.jn)(n);e.data&&u(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{h(!1)}}})()},[n]),(0,t.jsx)("div",{className:"min-w-0",children:(0,t.jsx)(o.f,{placeholder:l,onValueChange:e,value:s,loading:p,className:a,disabled:d,options:c.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,description:e.vector_store_description||void 0}))})})}])},802954,e=>{var t=e.i(618566),r=e.i(438847),i=e.i(271645),o=e.i(759525),s=e.i(782066);e.s(["f",0,function(e,a,n="tab"){let[l,d]=(0,r.o)(n,r.d.withDefault(a)),c=(0,s.I)((0,t.usePathname)()??"")||"home",u=e.find(e=>e===l)??a;return(0,i.useEffect)(()=>{l!==u&&d(null)},[l,u,d]),[u,(0,i.useCallback)(e=>{(0,o.T)({page:c,action:"click",target:`${n}=${e}`}),d(e)},[c,n,d])]}])},695420,e=>{var t=e.i(271645);e.s(["f",0,function(e){let[r,i]=(0,t.useState)(()=>new Set([e]));return{onTabChange:(0,t.useCallback)(e=>{i(t=>new Set(t).add(String(e)))},[]),hasVisited:(0,t.useCallback)(e=>r.has(e),[r])}}])},735450,e=>{var t=e.i(411382),r=e.i(772325),i=e.i(54105),o=e.i(240192),s=e.i(138620);let a="https://docs.litellm.ai/docs/decisions",n=`${a}#supported-providers`,l=`${a}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:i.z.string(),mode:i.z.string().nullish(),supported_endpoints:r.z.array(i.z.string()).nullish()}),u="decisions",p=["evaluation",u],h=e=>null!=e&&p.includes(e),f=e=>h(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),m=e=>{let t=o.z.record(i.z.string(),s.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let r=c.safeParse(t);return r.success?[[e,r.data]]:[]}):[]},g={decisions:0,"self-hosted":1,"needs-upgrade":2},b=t.z.object({decision_model_providers:r.z.array(i.z.string())}),_=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),y=(e,t,r)=>({row:e,routeOrder:g[e.route],rankInProvider:r.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,a,"d",0,u,"Z",0,n,"L",0,l,"B",0,"playground?tab=system-one","F",0,function(e){let t=m(e),r=t.filter(([,e])=>f(e)),i=new Map([...new Set(r.map(([,e])=>e.litellm_provider))].map(e=>{let r=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:r.every(([,e])=>f(e))}]}));return{models:new Set(r.map(([e])=>e)),providers:i}},"W",0,(e,t)=>m(e).filter(([,e])=>f(e)).map(([e,r])=>({model:e,provider:r.litellm_provider,route:t?.has(r.litellm_provider)?"decisions":(r.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(_).map(y).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>h(e)?u:e??void 0,"q",0,e=>e.flatMap(e=>e===u?p:[e]),"m",0,f,"_",0,h,"r",0,(e,t,r)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||r.some(t=>e.models.has(t)),"P",0,e=>{let t=b.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},223305,e=>{e.s(["f",0,e=>async(t,r)=>{let i=await fetch(t,r);return e(i.headers),i}])},841238,e=>{let t=e=>"object"==typeof e&&null!==e;e.s(["f",0,e=>{if(!t(e)||!e.error)return null;let{error:r}=e;return"string"==typeof r?r:t(r)&&"string"==typeof r.message?r.message:JSON.stringify(r)}])}],59935,(e,t,r)=>{var i;let o;e.e,i=function e(){var t,r="u">typeof self?self:"u">typeof window?window:void 0!==r?r:{},i=!r.document&&!!r.postMessage,o=r.IS_PAPA_WORKER||!1,s={},a=0,n={};function l(e){this._handle=null,this._finished=!1,this._completed=!1,this._halted=!1,this._input=null,this._baseIndex=0,this._partialLine="",this._rowCount=0,this._start=0,this._nextChunk=null,this.isFirstChunk=!0,this._completeResults={data:[],errors:[],meta:{}},(function(e){var t=y(e);t.chunkSize=parseInt(t.chunkSize),e.step||e.chunk||(t.chunkSize=null),this._handle=new h(t),(this._handle.streamer=this)._config=t}).call(this,e),this.parseChunk=function(e,t){var i=parseInt(this._config.skipFirstNLines)||0;if(this.isFirstChunk&&0<i){let t=this._config.newline;t||(s=this._config.quoteChar||'"',t=this._handle.guessLineEndings(e,s)),e=[...e.split(t).slice(i)].join(t)}this.isFirstChunk&&x(this._config.beforeFirstChunk)&&void 0!==(s=this._config.beforeFirstChunk(e))&&(e=s),this.isFirstChunk=!1,this._halted=!1;var i=this._partialLine+e,s=(this._partialLine="",this._handle.parse(i,this._baseIndex,!this._finished));if(!this._handle.paused()&&!this._handle.aborted()){if(e=s.meta.cursor,this._finished||(this._partialLine=i.substring(e-this._baseIndex),this._baseIndex=e),s&&s.data&&(this._rowCount+=s.data.length),i=this._finished||this._config.preview&&this._rowCount>=this._config.preview,o)r.postMessage({results:s,workerId:n.WORKER_ID,finished:i});else if(x(this._config.chunk)&&!t){if(this._config.chunk(s,this._handle),this._handle.paused()||this._handle.aborted())return void(this._halted=!0);this._completeResults=s=void 0}return this._config.step||this._config.chunk||(this._completeResults.data=this._completeResults.data.concat(s.data),this._completeResults.errors=this._completeResults.errors.concat(s.errors),this._completeResults.meta=s.meta),this._completed||!i||!x(this._config.complete)||s&&s.meta.aborted||(this._config.complete(this._completeResults,this._input),this._completed=!0),i||s&&s.meta.paused||this._nextChunk(),s}this._halted=!0},this._sendError=function(e){x(this._config.error)?this._config.error(e):o&&this._config.error&&r.postMessage({workerId:n.WORKER_ID,error:e,finished:!1})}}function d(e){var t;(e=e||{}).chunkSize||(e.chunkSize=n.RemoteChunkSize),l.call(this,e),this._nextChunk=i?function(){this._readChunk(),this._chunkLoaded()}:function(){this._readChunk()},this.stream=function(e){this._input=e,this._nextChunk()},this._readChunk=function(){if(this._finished)this._chunkLoaded();else{if(t=new XMLHttpRequest,this._config.withCredentials&&(t.withCredentials=this._config.withCredentials),i||(t.onload=k(this._chunkLoaded,this),t.onerror=k(this._chunkError,this)),t.open(this._config.downloadRequestBody?"POST":"GET",this._input,!i),this._config.downloadRequestHeaders){var e,r,o=this._config.downloadRequestHeaders;for(r in o)t.setRequestHeader(r,o[r])}this._config.chunkSize&&(e=this._start+this._config.chunkSize-1,t.setRequestHeader("Range","bytes="+this._start+"-"+e));try{t.send(this._config.downloadRequestBody)}catch(e){this._chunkError(e.message)}i&&0===t.status&&this._chunkError()}},this._chunkLoaded=function(){let e;4===t.readyState&&(t.status<200||400<=t.status?this._chunkError():(this._start+=this._config.chunkSize||t.responseText.length,this._finished=!this._config.chunkSize||this._start>=(null!==(e=(e=t).getResponseHeader("Content-Range"))?parseInt(e.substring(e.lastIndexOf("/")+1)):-1),this.parseChunk(t.responseText)))},this._chunkError=function(e){e=t.statusText||e,this._sendError(Error(e))}}function c(e){(e=e||{}).chunkSize||(e.chunkSize=n.LocalChunkSize),l.call(this,e);var t,r,i="u">typeof FileReader;this.stream=function(e){this._input=e,r=e.slice||e.webkitSlice||e.mozSlice,i?((t=new FileReader).onload=k(this._chunkLoaded,this),t.onerror=k(this._chunkError,this)):t=new FileReaderSync,this._nextChunk()},this._nextChunk=function(){this._finished||this._config.preview&&!(this._rowCount<this._config.preview)||this._readChunk()},this._readChunk=function(){var e=this._input,o=(this._config.chunkSize&&(o=Math.min(this._start+this._config.chunkSize,this._input.size),e=r.call(e,this._start,o)),t.readAsText(e,this._config.encoding));i||this._chunkLoaded({target:{result:o}})},this._chunkLoaded=function(e){this._start+=this._config.chunkSize,this._finished=!this._config.chunkSize||this._start>=this._input.size,this.parseChunk(e.target.result)},this._chunkError=function(){this._sendError(t.error)}}function u(e){var t;l.call(this,e=e||{}),this.stream=function(e){return t=e,this._nextChunk()},this._nextChunk=function(){var e,r;if(!this._finished)return t=(e=this._config.chunkSize)?(r=t.substring(0,e),t.substring(e)):(r=t,""),this._finished=!t,this.parseChunk(r)}}function p(e){l.call(this,e=e||{});var t=[],r=!0,i=!1;this.pause=function(){l.prototype.pause.apply(this,arguments),this._input.pause()},this.resume=function(){l.prototype.resume.apply(this,arguments),this._input.resume()},this.stream=function(e){this._input=e,this._input.on("data",this._streamData),this._input.on("end",this._streamEnd),this._input.on("error",this._streamError)},this._checkIsFinished=function(){i&&1===t.length&&(this._finished=!0)},this._nextChunk=function(){this._checkIsFinished(),t.length?this.parseChunk(t.shift()):r=!0},this._streamData=k(function(e){try{t.push("string"==typeof e?e:e.toString(this._config.encoding)),r&&(r=!1,this._checkIsFinished(),this.parseChunk(t.shift()))}catch(e){this._streamError(e)}},this),this._streamError=k(function(e){this._streamCleanUp(),this._sendError(e)},this),this._streamEnd=k(function(){this._streamCleanUp(),i=!0,this._streamData("")},this),this._streamCleanUp=k(function(){this._input.removeListener("data",this._streamData),this._input.removeListener("end",this._streamEnd),this._input.removeListener("error",this._streamError)},this)}function h(e){var t,r,i,o,s=/^\s*-?(\d+\.?|\.\d+|\d+\.\d+)([eE][-+]?\d+)?\s*$/,a=/^((\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z)))$/,l=this,d=0,c=0,u=!1,p=!1,h=[],g={data:[],errors:[],meta:{}};function b(t){return"greedy"===e.skipEmptyLines?""===t.join("").trim():1===t.length&&0===t[0].length}function _(){if(g&&i&&(v("Delimiter","UndetectableDelimiter","Unable to auto-detect delimiting character; defaulted to '"+n.DefaultDelimiter+"'"),i=!1),e.skipEmptyLines&&(g.data=g.data.filter(function(e){return!b(e)})),k()){if(g)if(Array.isArray(g.data[0])){for(var t,r=0;k()&&r<g.data.length;r++)g.data[r].forEach(o);g.data.splice(0,1)}else g.data.forEach(o);function o(t,r){x(e.transformHeader)&&(t=e.transformHeader(t,r)),h.push(t)}}function l(t,r){for(var i=e.header?{}:[],o=0;o<t.length;o++){var n=o,l=t[o],l=((t,r)=>(e.dynamicTypingFunction&&void 0===e.dynamicTyping[t]&&(e.dynamicTyping[t]=e.dynamicTypingFunction(t)),!0===(e.dynamicTyping[t]||e.dynamicTyping))?"true"===r||"TRUE"===r||"false"!==r&&"FALSE"!==r&&((e=>{if(s.test(e)&&-0x20000000000000<(e=parseFloat(e))&&e<0x20000000000000)return 1})(r)?parseFloat(r):a.test(r)?new Date(r):""===r?null:r):r)(n=e.header?o>=h.length?"__parsed_extra":h[o]:n,l=e.transform?e.transform(l,n):l);"__parsed_extra"===n?(i[n]=i[n]||[],i[n].push(l)):i[n]=l}return e.header&&(o>h.length?v("FieldMismatch","TooManyFields","Too many fields: expected "+h.length+" fields but parsed "+o,c+r):o<h.length&&v("FieldMismatch","TooFewFields","Too few fields: expected "+h.length+" fields but parsed "+o,c+r)),i}g&&(e.header||e.dynamicTyping||e.transform)&&(t=1,!g.data.length||Array.isArray(g.data[0])?(g.data=g.data.map(l),t=g.data.length):g.data=l(g.data,0),e.header&&g.meta&&(g.meta.fields=h),c+=t)}function k(){return e.header&&0===h.length}function v(e,t,r,i){e={type:e,code:t,message:r},void 0!==i&&(e.row=i),g.errors.push(e)}x(e.step)&&(o=e.step,e.step=function(t){g=t,k()?_():(_(),0!==g.data.length&&(d+=t.data.length,e.preview&&d>e.preview?r.abort():(g.data=g.data[0],o(g,l))))}),this.parse=function(o,s,a){var l=e.quoteChar||'"',l=(e.newline||(e.newline=this.guessLineEndings(o,l)),i=!1,e.delimiter?x(e.delimiter)&&(e.delimiter=e.delimiter(o),g.meta.delimiter=e.delimiter):((l=((t,r,i,o,s)=>{var a,l,d,c;s=s||[",","	","|",";",n.RECORD_SEP,n.UNIT_SEP];for(var u=0;u<s.length;u++){for(var p,h=s[u],f=0,g=0,_=0,y=(d=void 0,new m({comments:o,delimiter:h,newline:r,preview:10}).parse(t)),k=0;k<y.data.length;k++)i&&b(y.data[k])?_++:(g+=p=y.data[k].length,void 0===d?d=p:0<p&&(f+=Math.abs(p-d),d=p));0<y.data.length&&(g/=y.data.length-_),(void 0===l||f<=l)&&(void 0===c||c<g)&&1.99<g&&(l=f,a=h,c=g)}return{successful:!!(e.delimiter=a),bestDelimiter:a}})(o,e.newline,e.skipEmptyLines,e.comments,e.delimitersToGuess)).successful?e.delimiter=l.bestDelimiter:(i=!0,e.delimiter=n.DefaultDelimiter),g.meta.delimiter=e.delimiter),y(e));return e.preview&&e.header&&l.preview++,t=o,g=(r=new m(l)).parse(t,s,a),_(),u?{meta:{paused:!0}}:g||{meta:{paused:!1}}},this.paused=function(){return u},this.pause=function(){u=!0,r.abort(),t=x(e.chunk)?"":t.substring(r.getCharIndex())},this.resume=function(){l.streamer._halted?(u=!1,l.streamer.parseChunk(t,!0)):setTimeout(l.resume,3)},this.aborted=function(){return p},this.abort=function(){p=!0,r.abort(),g.meta.aborted=!0,x(e.complete)&&e.complete(g),t=""},this.guessLineEndings=function(e,t){e=e.substring(0,1048576);var t=RegExp(f(t)+"([^]*?)"+f(t),"gm"),r=(e=e.replace(t,"")).split("\r"),t=e.split("\n"),e=1<t.length&&t[0].length<r[0].length;if(1===r.length||e)return"\n";for(var i=0,o=0;o<r.length;o++)"\n"===r[o][0]&&i++;return i>=r.length/2?"\r\n":"\r"}}function f(e){return e.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}function m(e){var t=(e=e||{}).delimiter,r=e.newline,i=e.comments,o=e.step,s=e.preview,a=e.fastMode,l=null,d=!1,c=null==e.quoteChar?'"':e.quoteChar,u=c;if(void 0!==e.escapeChar&&(u=e.escapeChar),("string"!=typeof t||-1<n.BAD_DELIMITERS.indexOf(t))&&(t=","),i===t)throw Error("Comment character same as delimiter");!0===i?i="#":("string"!=typeof i||-1<n.BAD_DELIMITERS.indexOf(i))&&(i=!1),"\n"!==r&&"\r"!==r&&"\r\n"!==r&&(r="\n");var p=0,h=!1;this.parse=function(n,m,g){if("string"!=typeof n)throw Error("Input must be a string");var b=n.length,_=t.length,y=r.length,k=i.length,v=x(o),w=[],j=[],C=[],E=p=0;if(!n)return P();if(a||!1!==a&&-1===n.indexOf(c)){for(var S=n.split(r),R=0;R<S.length;R++){if(C=S[R],p+=C.length,R!==S.length-1)p+=r.length;else if(g)break;if(!i||C.substring(0,k)!==i){if(v){if(w=[],A(C.split(t)),$(),h)return P()}else A(C.split(t));if(s&&s<=R)return w=w.slice(0,s),P(!0)}}return P()}for(var z=n.indexOf(t,p),O=n.indexOf(r,p),M=RegExp(f(u)+f(c),"g"),N=n.indexOf(c,p);;)if(n[p]===c)for(N=p,p++;;){if(-1===(N=n.indexOf(c,N+1)))return g||j.push({type:"Quotes",code:"MissingQuotes",message:"Quoted field unterminated",row:w.length,index:p}),T();if(N===b-1)return T(n.substring(p,N).replace(M,c));if(c===u&&n[N+1]===u)N++;else if(c===u||0===N||n[N-1]!==u){-1!==z&&z<N+1&&(z=n.indexOf(t,N+1));var I=L(-1===(O=-1!==O&&O<N+1?n.indexOf(r,N+1):O)?z:Math.min(z,O));if(n.substr(N+1+I,_)===t){C.push(n.substring(p,N).replace(M,c)),n[p=N+1+I+_]!==c&&(N=n.indexOf(c,p)),z=n.indexOf(t,p),O=n.indexOf(r,p);break}if(I=L(O),n.substring(N+1+I,N+1+I+y)===r){if(C.push(n.substring(p,N).replace(M,c)),q(N+1+I+y),z=n.indexOf(t,p),N=n.indexOf(c,p),v&&($(),h))return P();if(s&&w.length>=s)return P(!0);break}j.push({type:"Quotes",code:"InvalidQuotes",message:"Trailing quote on quoted field is malformed",row:w.length,index:p}),N++}}else if(i&&0===C.length&&n.substring(p,p+k)===i){if(-1===O)return P();p=O+y,O=n.indexOf(r,p),z=n.indexOf(t,p)}else if(-1!==z&&(z<O||-1===O))C.push(n.substring(p,z)),p=z+_,z=n.indexOf(t,p);else{if(-1===O)break;if(C.push(n.substring(p,O)),q(O+y),v&&($(),h))return P();if(s&&w.length>=s)return P(!0)}return T();function A(e){w.push(e),E=p}function L(e){return -1!==e&&(e=n.substring(N+1,e))&&""===e.trim()?e.length:0}function T(e){return g||(void 0===e&&(e=n.substring(p)),C.push(e),p=b,A(C),v&&$()),P()}function q(e){p=e,A(C),C=[],O=n.indexOf(r,p)}function P(i){if(e.header&&!m&&w.length&&!d){var o=w[0],s=Object.create(null),a=new Set(o);let t=!1;for(let r=0;r<o.length;r++){let i=o[r];if(s[i=x(e.transformHeader)?e.transformHeader(i,r):i]){let e,n=s[i];for(;e=i+"_"+n,n++,a.has(e););a.add(e),o[r]=e,s[i]++,t=!0,(l=null===l?{}:l)[e]=i}else s[i]=1,o[r]=i;a.add(i)}t&&console.warn("Duplicate headers found and renamed."),d=!0}return{data:w,errors:j,meta:{delimiter:t,linebreak:r,aborted:h,truncated:!!i,cursor:E+(m||0),renamedHeaders:l}}}function $(){o(P()),w=[],j=[]}},this.abort=function(){h=!0},this.getCharIndex=function(){return p}}function g(e){var t=e.data,r=s[t.workerId],i=!1;if(t.error)r.userError(t.error,t.file);else if(t.results&&t.results.data){var o={abort:function(){i=!0,b(t.workerId,{data:[],errors:[],meta:{aborted:!0}})},pause:_,resume:_};if(x(r.userStep)){for(var a=0;a<t.results.data.length&&(r.userStep({data:t.results.data[a],errors:t.results.errors,meta:t.results.meta},o),!i);a++);delete t.results}else x(r.userChunk)&&(r.userChunk(t.results,o,t.file),delete t.results)}t.finished&&!i&&b(t.workerId,t.results)}function b(e,t){var r=s[e];x(r.userComplete)&&r.userComplete(t),r.terminate(),delete s[e]}function _(){throw Error("Not implemented.")}function y(e){if("object"!=typeof e||null===e)return e;var t,r=Array.isArray(e)?[]:{};for(t in e)r[t]=y(e[t]);return r}function k(e,t){return function(){e.apply(t,arguments)}}function x(e){return"function"==typeof e}return n.parse=function(t,i){var o,l,h,f=(i=i||{}).dynamicTyping||!1;if(x(f)&&(i.dynamicTypingFunction=f,f={}),i.dynamicTyping=f,i.transform=!!x(i.transform)&&i.transform,!i.worker||!n.WORKERS_SUPPORTED){let e;return f=null,n.NODE_STREAM_INPUT,"string"==typeof t?(t=65279!==(e=t).charCodeAt(0)?e:e.slice(1),f=new(i.download?d:u)(i)):!0===t.readable&&x(t.read)&&x(t.on)?f=new p(i):(r.File&&t instanceof File||t instanceof Object)&&(f=new c(i)),f.stream(t)}(f=!!n.WORKERS_SUPPORTED&&(l=r.URL||r.webkitURL||null,h=e.toString(),o=n.BLOB_URL||(n.BLOB_URL=l.createObjectURL(new Blob(["var global = (function() { if (typeof self !== 'undefined') { return self; } if (typeof window !== 'undefined') { return window; } if (typeof global !== 'undefined') { return global; } return {}; })(); global.IS_PAPA_WORKER=true; ","(",h,")();"],{type:"text/javascript"}))),(o=new r.Worker(o)).onmessage=g,o.id=a++,s[o.id]=o)).userStep=i.step,f.userChunk=i.chunk,f.userComplete=i.complete,f.userError=i.error,i.step=x(i.step),i.chunk=x(i.chunk),i.complete=x(i.complete),i.error=x(i.error),delete i.worker,f.postMessage({input:t,config:i,workerId:f.id})},n.unparse=function(e,t){var r=!1,i=!0,o=",",s="\r\n",a='"',l=a+a,d=!1,c=null,u=!1,p=((()=>{if("object"==typeof t){if("string"!=typeof t.delimiter||n.BAD_DELIMITERS.filter(function(e){return -1!==t.delimiter.indexOf(e)}).length||(o=t.delimiter),("boolean"==typeof t.quotes||"function"==typeof t.quotes||Array.isArray(t.quotes))&&(r=t.quotes),"boolean"!=typeof t.skipEmptyLines&&"string"!=typeof t.skipEmptyLines||(d=t.skipEmptyLines),"string"==typeof t.newline&&(s=t.newline),"string"==typeof t.quoteChar&&(a=t.quoteChar),"boolean"==typeof t.header&&(i=t.header),Array.isArray(t.columns)){if(0===t.columns.length)throw Error("Option columns is empty");c=t.columns}void 0!==t.escapeChar&&(l=t.escapeChar+a),t.escapeFormulae instanceof RegExp?u=t.escapeFormulae:"boolean"==typeof t.escapeFormulae&&t.escapeFormulae&&(u=/^[=+\-@\t\r].*$/)}})(),RegExp(f(a),"g"));if("string"==typeof e&&(e=JSON.parse(e)),Array.isArray(e)){if(!e.length||Array.isArray(e[0]))return h(null,e,d);if("object"==typeof e[0])return h(c||Object.keys(e[0]),e,d)}else if("object"==typeof e)return"string"==typeof e.data&&(e.data=JSON.parse(e.data)),Array.isArray(e.data)&&(e.fields||(e.fields=e.meta&&e.meta.fields||c),e.fields||(e.fields=Array.isArray(e.data[0])?e.fields:"object"==typeof e.data[0]?Object.keys(e.data[0]):[]),Array.isArray(e.data[0])||"object"==typeof e.data[0]||(e.data=[e.data])),h(e.fields||[],e.data||[],d);throw Error("Unable to serialize unrecognized input");function h(e,t,r){var a="",n=("string"==typeof e&&(e=JSON.parse(e)),"string"==typeof t&&(t=JSON.parse(t)),Array.isArray(e)&&0<e.length),l=!Array.isArray(t[0]);if(n&&i){for(var d=0;d<e.length;d++)0<d&&(a+=o),a+=m(e[d],d);0<t.length&&(a+=s)}for(var c=0;c<t.length;c++){var u=(n?e:t[c]).length,p=!1,h=n?0===Object.keys(t[c]).length:0===t[c].length;if(r&&!n&&(p="greedy"===r?""===t[c].join("").trim():1===t[c].length&&0===t[c][0].length),"greedy"===r&&n){for(var f=[],g=0;g<u;g++){var b=l?e[g]:g;f.push(t[c][b])}p=""===f.join("").trim()}if(!p){for(var _=0;_<u;_++){0<_&&!h&&(a+=o);var y=n&&l?e[_]:_;a+=m(t[c][y],_)}c<t.length-1&&(!r||0<u&&!h)&&(a+=s)}}return a}function m(e,t){var i,s;return null==e?"":e.constructor===Date?JSON.stringify(e).slice(1,25):(s=!1,u&&"string"==typeof e&&u.test(e)&&(e="'"+e,s=!0),i=e.toString().replace(p,l),(s=s||!0===r||"function"==typeof r&&r(e,t)||Array.isArray(r)&&r[t]||((e,t)=>{for(var r=0;r<t.length;r++)if(-1<e.indexOf(t[r]))return!0;return!1})(i,n.BAD_DELIMITERS)||-1<i.indexOf(o)||" "===i.charAt(0)||" "===i.charAt(i.length-1))?a+i+a:i)}},n.RECORD_SEP="\x1e",n.UNIT_SEP="\x1f",n.BYTE_ORDER_MARK="\uFEFF",n.BAD_DELIMITERS=["\r","\n",'"',n.BYTE_ORDER_MARK],n.WORKERS_SUPPORTED=!i&&!!r.Worker,n.NODE_STREAM_INPUT=1,n.LocalChunkSize=0xa00000,n.RemoteChunkSize=5242880,n.DefaultDelimiter=",",n.Parser=m,n.ParserHandle=h,n.NetworkStreamer=d,n.FileStreamer=c,n.StringStreamer=u,n.ReadableStreamStreamer=p,r.jQuery&&((t=r.jQuery).fn.parse=function(e){var i=e.config||{},o=[];return this.each(function(e){if(!("INPUT"===t(this).prop("tagName").toUpperCase()&&"file"===t(this).attr("type").toLowerCase()&&r.FileReader)||!this.files||0===this.files.length)return!0;for(var s=0;s<this.files.length;s++)o.push({file:this.files[s],inputElem:this,instanceConfig:t.extend({},i)})}),s(),this;function s(){if(0===o.length)x(e.complete)&&e.complete();else{var r,i,s,l=o[0];if(x(e.before)){var d=e.before(l.file,l.inputElem);if("object"==typeof d){if("abort"===d.action)return r=l.file,i=l.inputElem,s=d.reason,void(x(e.error)&&e.error({name:"AbortError"},r,i,s));if("skip"===d.action)return void a();"object"==typeof d.config&&(l.instanceConfig=t.extend(l.instanceConfig,d.config))}else if("skip"===d)return void a()}var c=l.instanceConfig.complete;l.instanceConfig.complete=function(e){x(c)&&c(e,l.file,l.inputElem),a()},n.parse(l.file,l.instanceConfig)}}function a(){o.splice(0,1),s()}}),o&&(r.onmessage=function(e){e=e.data,void 0===n.WORKER_ID&&e&&(n.WORKER_ID=e.workerId),"string"==typeof e.input?r.postMessage({workerId:n.WORKER_ID,results:n.parse(e.input,e.config),finished:!0}):(r.File&&e.input instanceof File||e.input instanceof Object)&&(e=n.parse(e.input,e.config))&&r.postMessage({workerId:n.WORKER_ID,results:e,finished:!0})}),(d.prototype=Object.create(l.prototype)).constructor=d,(c.prototype=Object.create(l.prototype)).constructor=c,(u.prototype=Object.create(u.prototype)).constructor=u,(p.prototype=Object.create(l.prototype)).constructor=p,n},"function"==typeof define&&define.amd?void 0!==(o=i())&&e.v(o):t.exports=i()}]);