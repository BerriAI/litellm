(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,286536,e=>{"use strict";let t=(0,e.i(475254).default)("eye",[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["Eye",()=>t],286536)},77705,e=>{"use strict";let t=(0,e.i(475254).default)("eye-off",[["path",{d:"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49",key:"ct8e1f"}],["path",{d:"M14.084 14.158a3 3 0 0 1-4.242-4.242",key:"151rxh"}],["path",{d:"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143",key:"13bj9a"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]]);e.s(["EyeOff",()=>t],77705)},611052,270756,e=>{"use strict";var t=e.i(843476),o=e.i(271645),r=e.i(776639),a=e.i(519455),i=e.i(793479),s=e.i(110204),n=e.i(699375),l=e.i(888259),d=e.i(871689),c=e.i(972520),p=e.i(643531),m=e.i(286536),u=e.i(77705),g=e.i(834161),f=e.i(221345);let h=(0,e.i(475254).default)("lock",[["rect",{width:"18",height:"11",x:"3",y:"11",rx:"2",ry:"2",key:"1w4ew1"}],["path",{d:"M7 11V7a5 5 0 0 1 10 0v4",key:"fwvmzm"}]]);e.s(["Lock",()=>h],270756);var _=e.i(37727),x=e.i(975157);e.s(["ByokCredentialModal",0,({server:e,open:b,onClose:y,onSuccess:v,accessToken:w})=>{let[k,j]=(0,o.useState)(1),[N,S]=(0,o.useState)(""),[C,I]=(0,o.useState)(!1),[T,E]=(0,o.useState)(!0),[M,P]=(0,o.useState)(!1),A=e.alias||e.server_name||"Service",R=A.charAt(0).toUpperCase(),D=()=>{j(1),S(""),I(!1),E(!0),P(!1),y()},O=async()=>{if(!N.trim())return void l.default.error("Please enter your API key");P(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${w}`},body:JSON.stringify({credential:N.trim(),save:T})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}l.default.success(`Connected to ${A}`),v(e.server_id),D()}catch(e){l.default.error(e.message||"Failed to connect")}finally{P(!1)}};return(0,t.jsx)(r.Dialog,{open:b,onOpenChange:e=>e?void 0:D(),children:(0,t.jsx)(r.DialogContent,{className:"max-w-md p-0 overflow-hidden",children:(0,t.jsxs)("div",{className:"relative p-6",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===k?(0,t.jsxs)("button",{onClick:()=>j(1),className:"flex items-center gap-1 text-muted-foreground hover:text-foreground text-sm",children:[(0,t.jsx)(d.ArrowLeft,{className:"h-3.5 w-3.5"})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:(0,x.cn)("w-2 h-2 rounded-full",1===k?"bg-primary":"bg-muted")}),(0,t.jsx)("div",{className:(0,x.cn)("w-2 h-2 rounded-full",2===k?"bg-primary":"bg-muted")})]}),(0,t.jsx)("button",{onClick:D,className:"text-muted-foreground hover:text-foreground","aria-label":"Close",children:(0,t.jsx)(_.X,{className:"h-4 w-4"})})]}),1===k?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(c.ArrowRight,{className:"text-muted-foreground h-4 w-4"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:R})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:["Connect ",A]}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["LiteLLM needs access to ",A," to complete your request."]}),(0,t.jsx)("div",{className:"bg-muted rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-foreground mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-muted-foreground text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to ",A,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-emerald-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,o)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-foreground",children:[(0,t.jsx)(p.Check,{className:"text-emerald-500 flex-shrink-0 h-4 w-4"}),e]},o))})]}),(0,t.jsxs)(a.Button,{onClick:()=>j(2),className:"w-full",children:["Continue to Authentication ",(0,t.jsx)(c.ArrowRight,{className:"h-4 w-4"})]}),(0,t.jsx)(a.Button,{onClick:D,variant:"ghost",className:"mt-3 w-full text-muted-foreground",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4",children:(0,t.jsx)(g.Key,{className:"text-primary h-5 w-5"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-foreground mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-muted-foreground mb-6",children:["Enter your ",A," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4 space-y-2",children:[(0,t.jsxs)(s.Label,{htmlFor:"byok-api-key",children:[A," API Key"]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(i.Input,{id:"byok-api-key",type:C?"text":"password",placeholder:"Enter your API key",value:N,onChange:e=>S(e.target.value)}),(0,t.jsx)("button",{type:"button",onClick:()=>I(!C),className:"absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground","aria-label":C?"Hide api key":"Show api key",children:C?(0,t.jsx)(u.EyeOff,{size:14}):(0,t.jsx)(m.Eye,{size:14})})]}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-primary hover:underline text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key?"," ",(0,t.jsx)(f.Link,{className:"h-3.5 w-3.5"})]})]}),(0,t.jsxs)("div",{className:"bg-muted rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-muted-foreground",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-foreground",children:"Save key for future use"})]}),(0,t.jsx)(n.Switch,{checked:T,onCheckedChange:E})]}),(0,t.jsxs)("div",{className:"bg-primary/5 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(h,{className:"text-primary mt-0.5 flex-shrink-0 h-4 w-4"}),(0,t.jsx)("p",{className:"text-sm text-primary",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)(a.Button,{onClick:O,disabled:M,className:"w-full",children:[(0,t.jsx)(h,{className:"h-4 w-4"})," Connect & Authorize"]})]})]})})})}],611052)},367692,e=>{"use strict";var t=e.i(843476),o=e.i(271645),r=e.i(470152),a=e.i(981140),i=e.i(820783),s=e.i(30030),n=e.i(369340),l=e.i(586318),d=e.i(999682),c=e.i(635804),p=e.i(248425),m=e.i(75830),u=["PageUp","PageDown"],g=["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"],f={"from-left":["Home","PageDown","ArrowDown","ArrowLeft"],"from-right":["Home","PageDown","ArrowDown","ArrowRight"],"from-bottom":["Home","PageDown","ArrowDown","ArrowLeft"],"from-top":["Home","PageDown","ArrowUp","ArrowLeft"]},h="Slider",[_,x,b]=(0,m.createCollection)(h),[y,v]=(0,s.createContextScope)(h,[b]),[w,k]=y(h),j=o.forwardRef((e,i)=>{let{name:s,min:l=0,max:d=100,step:c=1,orientation:p="horizontal",disabled:m=!1,minStepsBetweenThumbs:f=0,defaultValue:h=[l],value:x,onValueChange:b=()=>{},onValueCommit:y=()=>{},inverted:v=!1,form:k,...j}=e,N=o.useRef(new Set),S=o.useRef(0),T="horizontal"===p,[E=[],M]=(0,n.useControllableState)({prop:x,defaultProp:h,onChange:e=>{let t=[...N.current];t[S.current]?.focus(),b(e)}}),P=o.useRef(E);function A(e,t,{commit:o}={commit:!1}){let a,i=(String(c).split(".")[1]||"").length,s=Math.round((Math.round((e-l)/c)*c+l)*(a=Math.pow(10,i)))/a,n=(0,r.clamp)(s,[l,d]);M((e=[])=>{let r=function(e=[],t,o){let r=[...e];return r[o]=t,r.sort((e,t)=>e-t)}(e,n,t);if(!function(e,t){if(t>0)return Math.min(...e.slice(0,-1).map((t,o)=>e[o+1]-t))>=t;return!0}(r,f*c))return e;{S.current=r.indexOf(n);let t=String(r)!==String(e);return t&&o&&y(r),t?r:e}})}return(0,t.jsx)(w,{scope:e.__scopeSlider,name:s,disabled:m,min:l,max:d,valueIndexToChangeRef:S,thumbs:N.current,values:E,orientation:p,form:k,children:(0,t.jsx)(_.Provider,{scope:e.__scopeSlider,children:(0,t.jsx)(_.Slot,{scope:e.__scopeSlider,children:(0,t.jsx)(T?C:I,{"aria-disabled":m,"data-disabled":m?"":void 0,...j,ref:i,onPointerDown:(0,a.composeEventHandlers)(j.onPointerDown,()=>{m||(P.current=E)}),min:l,max:d,inverted:v,onSlideStart:m?void 0:function(e){let t=function(e,t){if(1===e.length)return 0;let o=e.map(e=>Math.abs(e-t)),r=Math.min(...o);return o.indexOf(r)}(E,e);A(e,t)},onSlideMove:m?void 0:function(e){A(e,S.current)},onSlideEnd:m?void 0:function(){let e=P.current[S.current];E[S.current]!==e&&y(E)},onHomeKeyDown:()=>!m&&A(l,0,{commit:!0}),onEndKeyDown:()=>!m&&A(d,E.length-1,{commit:!0}),onStepKeyDown:({event:e,direction:t})=>{if(!m){let o=u.includes(e.key)||e.shiftKey&&g.includes(e.key),r=S.current;A(E[r]+c*(o?10:1)*t,r,{commit:!0})}}})})})})});j.displayName=h;var[N,S]=y(h,{startEdge:"left",endEdge:"right",size:"width",direction:1}),C=o.forwardRef((e,r)=>{let{min:a,max:s,dir:n,inverted:d,onSlideStart:c,onSlideMove:p,onSlideEnd:m,onStepKeyDown:u,...g}=e,[h,_]=o.useState(null),x=(0,i.useComposedRefs)(r,e=>_(e)),b=o.useRef(void 0),y=(0,l.useDirection)(n),v="ltr"===y,w=v&&!d||!v&&d;function k(e){let t=b.current||h.getBoundingClientRect(),o=z([0,t.width],w?[a,s]:[s,a]);return b.current=t,o(e-t.left)}return(0,t.jsx)(N,{scope:e.__scopeSlider,startEdge:w?"left":"right",endEdge:w?"right":"left",direction:w?1:-1,size:"width",children:(0,t.jsx)(T,{dir:y,"data-orientation":"horizontal",...g,ref:x,style:{...g.style,"--radix-slider-thumb-transform":"translateX(-50%)"},onSlideStart:e=>{let t=k(e.clientX);c?.(t)},onSlideMove:e=>{let t=k(e.clientX);p?.(t)},onSlideEnd:()=>{b.current=void 0,m?.()},onStepKeyDown:e=>{let t=f[w?"from-left":"from-right"].includes(e.key);u?.({event:e,direction:t?-1:1})}})})}),I=o.forwardRef((e,r)=>{let{min:a,max:s,inverted:n,onSlideStart:l,onSlideMove:d,onSlideEnd:c,onStepKeyDown:p,...m}=e,u=o.useRef(null),g=(0,i.useComposedRefs)(r,u),h=o.useRef(void 0),_=!n;function x(e){let t=h.current||u.current.getBoundingClientRect(),o=z([0,t.height],_?[s,a]:[a,s]);return h.current=t,o(e-t.top)}return(0,t.jsx)(N,{scope:e.__scopeSlider,startEdge:_?"bottom":"top",endEdge:_?"top":"bottom",size:"height",direction:_?1:-1,children:(0,t.jsx)(T,{"data-orientation":"vertical",...m,ref:g,style:{...m.style,"--radix-slider-thumb-transform":"translateY(50%)"},onSlideStart:e=>{let t=x(e.clientY);l?.(t)},onSlideMove:e=>{let t=x(e.clientY);d?.(t)},onSlideEnd:()=>{h.current=void 0,c?.()},onStepKeyDown:e=>{let t=f[_?"from-bottom":"from-top"].includes(e.key);p?.({event:e,direction:t?-1:1})}})})}),T=o.forwardRef((e,o)=>{let{__scopeSlider:r,onSlideStart:i,onSlideMove:s,onSlideEnd:n,onHomeKeyDown:l,onEndKeyDown:d,onStepKeyDown:c,...m}=e,f=k(h,r);return(0,t.jsx)(p.Primitive.span,{...m,ref:o,onKeyDown:(0,a.composeEventHandlers)(e.onKeyDown,e=>{"Home"===e.key?(l(e),e.preventDefault()):"End"===e.key?(d(e),e.preventDefault()):u.concat(g).includes(e.key)&&(c(e),e.preventDefault())}),onPointerDown:(0,a.composeEventHandlers)(e.onPointerDown,e=>{let t=e.target;t.setPointerCapture(e.pointerId),e.preventDefault(),f.thumbs.has(t)?t.focus():i(e)}),onPointerMove:(0,a.composeEventHandlers)(e.onPointerMove,e=>{e.target.hasPointerCapture(e.pointerId)&&s(e)}),onPointerUp:(0,a.composeEventHandlers)(e.onPointerUp,e=>{let t=e.target;t.hasPointerCapture(e.pointerId)&&(t.releasePointerCapture(e.pointerId),n(e))})})}),E="SliderTrack",M=o.forwardRef((e,o)=>{let{__scopeSlider:r,...a}=e,i=k(E,r);return(0,t.jsx)(p.Primitive.span,{"data-disabled":i.disabled?"":void 0,"data-orientation":i.orientation,...a,ref:o})});M.displayName=E;var P="SliderRange",A=o.forwardRef((e,r)=>{let{__scopeSlider:a,...s}=e,n=k(P,a),l=S(P,a),d=o.useRef(null),c=(0,i.useComposedRefs)(r,d),m=n.values.length,u=n.values.map(e=>L(e,n.min,n.max)),g=m>1?Math.min(...u):0,f=100-Math.max(...u);return(0,t.jsx)(p.Primitive.span,{"data-orientation":n.orientation,"data-disabled":n.disabled?"":void 0,...s,ref:c,style:{...e.style,[l.startEdge]:g+"%",[l.endEdge]:f+"%"}})});A.displayName=P;var R="SliderThumb",D=o.forwardRef((e,r)=>{let a=x(e.__scopeSlider),[s,n]=o.useState(null),l=(0,i.useComposedRefs)(r,e=>n(e)),d=o.useMemo(()=>s?a().findIndex(e=>e.ref.current===s):-1,[a,s]);return(0,t.jsx)(O,{...e,ref:l,index:d})}),O=o.forwardRef((e,r)=>{var s,n,l,d,m;let u,g,{__scopeSlider:f,index:h,name:x,...b}=e,y=k(R,f),v=S(R,f),[w,j]=o.useState(null),N=(0,i.useComposedRefs)(r,e=>j(e)),C=!w||y.form||!!w.closest("form"),I=(0,c.useSize)(w),T=y.values[h],E=void 0===T?0:L(T,y.min,y.max),M=(s=h,(n=y.values.length)>2?`Value ${s+1} of ${n}`:2===n?["Minimum","Maximum"][s]:void 0),P=I?.[v.size],A=P?(l=P,d=E,m=v.direction,g=z([0,50],[0,u=l/2]),(u-g(d)*m)*m):0;return o.useEffect(()=>{if(w)return y.thumbs.add(w),()=>{y.thumbs.delete(w)}},[w,y.thumbs]),(0,t.jsxs)("span",{style:{transform:"var(--radix-slider-thumb-transform)",position:"absolute",[v.startEdge]:`calc(${E}% + ${A}px)`},children:[(0,t.jsx)(_.ItemSlot,{scope:e.__scopeSlider,children:(0,t.jsx)(p.Primitive.span,{role:"slider","aria-label":e["aria-label"]||M,"aria-valuemin":y.min,"aria-valuenow":T,"aria-valuemax":y.max,"aria-orientation":y.orientation,"data-orientation":y.orientation,"data-disabled":y.disabled?"":void 0,tabIndex:y.disabled?void 0:0,...b,ref:N,style:void 0===T?{display:"none"}:e.style,onFocus:(0,a.composeEventHandlers)(e.onFocus,()=>{y.valueIndexToChangeRef.current=h})})}),C&&(0,t.jsx)($,{name:x??(y.name?y.name+(y.values.length>1?"[]":""):void 0),form:y.form,value:T},h)]})});D.displayName=R;var $=o.forwardRef(({__scopeSlider:e,value:r,...a},s)=>{let n=o.useRef(null),l=(0,i.useComposedRefs)(n,s),c=(0,d.usePrevious)(r);return o.useEffect(()=>{let e=n.current;if(!e)return;let t=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set;if(c!==r&&t){let o=new Event("input",{bubbles:!0});t.call(e,r),e.dispatchEvent(o)}},[c,r]),(0,t.jsx)(p.Primitive.input,{style:{display:"none"},...a,ref:l,defaultValue:r})});function L(e,t,o){return(0,r.clamp)(100/(o-t)*(e-t),[0,100])}function z(e,t){return o=>{if(e[0]===e[1]||t[0]===t[1])return t[0];let r=(t[1]-t[0])/(e[1]-e[0]);return t[0]+r*(o-e[0])}}$.displayName="RadioBubbleInput";var H=e.i(975157);let B=o.forwardRef(({className:e,...o},r)=>(0,t.jsxs)(j,{ref:r,className:(0,H.cn)("relative flex w-full touch-none select-none items-center",e),...o,children:[(0,t.jsx)(M,{className:"relative h-2 w-full grow overflow-hidden rounded-full bg-secondary",children:(0,t.jsx)(A,{className:"absolute h-full bg-primary"})}),(0,t.jsx)(D,{className:"block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"})]}));B.displayName=j.displayName,e.s(["Slider",()=>B],367692)},514764,e=>{"use strict";let t=(0,e.i(475254).default)("send",[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]]);e.s(["Send",()=>t],514764)},975558,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-up",[["path",{d:"m5 12 7-7 7 7",key:"hav0vg"}],["path",{d:"M12 19V5",key:"x0mq9r"}]]);e.s(["ArrowUp",()=>t],975558)},989359,989022,e=>{"use strict";var t=e.i(475254);let o=(0,t.default)("eraser",[["path",{d:"M21 21H8a2 2 0 0 1-1.42-.587l-3.994-3.999a2 2 0 0 1 0-2.828l10-10a2 2 0 0 1 2.829 0l5.999 6a2 2 0 0 1 0 2.828L12.834 21",key:"g5wo59"}],["path",{d:"m5.082 11.09 8.828 8.828",key:"1wx5vj"}]]);e.s(["Eraser",()=>o],989359);var r=e.i(843476),a=e.i(746798),i=e.i(503116),s=e.i(212426);let n=(0,t.default)("hash",[["line",{x1:"4",x2:"20",y1:"9",y2:"9",key:"4lhtct"}],["line",{x1:"4",x2:"20",y1:"15",y2:"15",key:"vyu0kd"}],["line",{x1:"10",x2:"8",y1:"3",y2:"21",key:"1ggp8o"}],["line",{x1:"16",x2:"14",y1:"3",y2:"21",key:"weycgp"}]]);var l=e.i(204997),d=e.i(292270),c=e.i(341240),p=e.i(195116);let m=({tooltip:e,icon:t,children:o})=>(0,r.jsx)(a.TooltipProvider,{children:(0,r.jsxs)(a.Tooltip,{children:[(0,r.jsx)(a.TooltipTrigger,{asChild:!0,children:(0,r.jsxs)("div",{className:"flex items-center",children:[t,(0,r.jsx)("span",{children:o})]})}),(0,r.jsx)(a.TooltipContent,{children:e})]})});e.s(["default",0,({timeToFirstToken:e,totalLatency:t,usage:o,toolName:a})=>e||t||o?(0,r.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-border text-xs text-muted-foreground flex flex-wrap gap-3",children:[void 0!==e&&(0,r.jsxs)(m,{tooltip:"Time to first token",icon:(0,r.jsx)(i.Clock,{className:"h-3 w-3 mr-1"}),children:["TTFT: ",(e/1e3).toFixed(2),"s"]}),void 0!==t&&(0,r.jsxs)(m,{tooltip:"Total latency",icon:(0,r.jsx)(i.Clock,{className:"h-3 w-3 mr-1"}),children:["Total Latency: ",(t/1e3).toFixed(2),"s"]}),o?.promptTokens!==void 0&&(0,r.jsxs)(m,{tooltip:"Prompt tokens",icon:(0,r.jsx)(l.LogIn,{className:"h-3 w-3 mr-1"}),children:["In: ",o.promptTokens]}),o?.completionTokens!==void 0&&(0,r.jsxs)(m,{tooltip:"Completion tokens",icon:(0,r.jsx)(d.LogOut,{className:"h-3 w-3 mr-1"}),children:["Out: ",o.completionTokens]}),o?.reasoningTokens!==void 0&&(0,r.jsxs)(m,{tooltip:"Reasoning tokens",icon:(0,r.jsx)(c.Lightbulb,{className:"h-3 w-3 mr-1"}),children:["Reasoning: ",o.reasoningTokens]}),o?.totalTokens!==void 0&&(0,r.jsxs)(m,{tooltip:"Total tokens",icon:(0,r.jsx)(n,{className:"h-3 w-3 mr-1"}),children:["Total: ",o.totalTokens]}),o?.cost!==void 0&&(0,r.jsxs)(m,{tooltip:"Cost",icon:(0,r.jsx)(s.DollarSign,{className:"h-3 w-3 mr-1"}),children:["$",o.cost.toFixed(6)]}),a&&(0,r.jsxs)(m,{tooltip:"Tool used",icon:(0,r.jsx)(p.Wrench,{className:"h-3 w-3 mr-1"}),children:["Tool: ",a]})]}):null],989022)},212426,e=>{"use strict";let t=(0,e.i(475254).default)("dollar-sign",[["line",{x1:"12",x2:"12",y1:"2",y2:"22",key:"7eqyqh"}],["path",{d:"M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6",key:"1b0p4s"}]]);e.s(["DollarSign",()=>t],212426)},204997,e=>{"use strict";let t=(0,e.i(475254).default)("log-in",[["path",{d:"m10 17 5-5-5-5",key:"1bsop3"}],["path",{d:"M15 12H3",key:"6jk70r"}],["path",{d:"M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4",key:"u53s6r"}]]);e.s(["LogIn",()=>t],204997)},219470,341240,e=>{"use strict";e.s(["coy",0,{'code[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",maxHeight:"inherit",height:"inherit",padding:"0 1em",display:"block",overflow:"auto"},'pre[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",position:"relative",margin:".5em 0",overflow:"visible",padding:"1px",backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em"},'pre[class*="language-"] > code':{position:"relative",zIndex:"1",borderLeft:"10px solid #358ccb",boxShadow:"-1px 0px 0px 0px #358ccb, 0px 0px 0px 1px #dfdfdf",backgroundColor:"#fdfdfd",backgroundImage:"linear-gradient(transparent 50%, rgba(69, 142, 209, 0.04) 50%)",backgroundSize:"3em 3em",backgroundOrigin:"content-box",backgroundAttachment:"local"},':not(pre) > code[class*="language-"]':{backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em",position:"relative",padding:".2em",borderRadius:"0.3em",color:"#c92c2c",border:"1px solid rgba(0, 0, 0, 0.1)",display:"inline",whiteSpace:"normal"},'pre[class*="language-"]:before':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"0.18em",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(-2deg)",MozTransform:"rotate(-2deg)",msTransform:"rotate(-2deg)",OTransform:"rotate(-2deg)",transform:"rotate(-2deg)"},'pre[class*="language-"]:after':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"auto",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(2deg)",MozTransform:"rotate(2deg)",msTransform:"rotate(2deg)",OTransform:"rotate(2deg)",transform:"rotate(2deg)",right:"0.75em"},comment:{color:"#7D8B99"},"block-comment":{color:"#7D8B99"},prolog:{color:"#7D8B99"},doctype:{color:"#7D8B99"},cdata:{color:"#7D8B99"},punctuation:{color:"#5F6364"},property:{color:"#c92c2c"},tag:{color:"#c92c2c"},boolean:{color:"#c92c2c"},number:{color:"#c92c2c"},"function-name":{color:"#c92c2c"},constant:{color:"#c92c2c"},symbol:{color:"#c92c2c"},deleted:{color:"#c92c2c"},selector:{color:"#2f9c0a"},"attr-name":{color:"#2f9c0a"},string:{color:"#2f9c0a"},char:{color:"#2f9c0a"},function:{color:"#2f9c0a"},builtin:{color:"#2f9c0a"},inserted:{color:"#2f9c0a"},operator:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},entity:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)",cursor:"help"},url:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},variable:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},atrule:{color:"#1990b8"},"attr-value":{color:"#1990b8"},keyword:{color:"#1990b8"},"class-name":{color:"#1990b8"},regex:{color:"#e90"},important:{color:"#e90",fontWeight:"normal"},".language-css .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},".style .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:".7"},'pre[class*="language-"].line-numbers.line-numbers':{paddingLeft:"0"},'pre[class*="language-"].line-numbers.line-numbers code':{paddingLeft:"3.8em"},'pre[class*="language-"].line-numbers.line-numbers .line-numbers-rows':{left:"0"},'pre[class*="language-"][data-line]':{paddingTop:"0",paddingBottom:"0",paddingLeft:"0"},"pre[data-line] code":{position:"relative",paddingLeft:"4em"},"pre .line-highlight":{marginTop:"0"}}],219470);let t=(0,e.i(475254).default)("lightbulb",[["path",{d:"M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5",key:"1gvzjb"}],["path",{d:"M9 18h6",key:"x1upvd"}],["path",{d:"M10 22h4",key:"ceow96"}]]);e.s(["Lightbulb",()=>t],341240)},221345,e=>{"use strict";let t=(0,e.i(475254).default)("link",[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]]);e.s(["Link",()=>t],221345)},190272,785913,e=>{"use strict";var t,o,r=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((o={}).IMAGE="image",o.VIDEO="video",o.CHAT="chat",o.RESPONSES="responses",o.IMAGE_EDITS="image_edits",o.ANTHROPIC_MESSAGES="anthropic_messages",o.EMBEDDINGS="embeddings",o.SPEECH="speech",o.TRANSCRIPTION="transcription",o.A2A_AGENTS="a2a_agents",o.MCP="mcp",o.REALTIME="realtime",o);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(r).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:o,accessToken:r,apiKey:i,inputMessage:s,chatHistory:n,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:p,selectedMCPServers:m,mcpServers:u,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:_,selectedSdk:x,proxySettings:b}=e,y="session"===o?r:i,v=window.location.origin,w=b?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:b?.PROXY_BASE_URL&&(v=b.PROXY_BASE_URL);let k=s||"Your prompt here",j=k.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),N=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};l.length>0&&(S.tags=l),d.length>0&&(S.vector_stores=d),c.length>0&&(S.guardrails=c),p.length>0&&(S.policies=p);let C=_||"your-model-name",I="azure"===x?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case a.CHAT:{let e=Object.keys(S).length>0,o="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let r=N.length>0?N:[{role:"user",content:k}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(r,null,4)}${o}
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
#     ]${o}
# )
# print(response_with_file)
`;break}case a.RESPONSES:{let e=Object.keys(S).length>0,o="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();o=`,
    extra_body=${e}`}let r=N.length>0?N:[{role:"user",content:k}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(r,null,4)}${o}
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${o}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===x?`
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
prompt = "${j}"

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
`;break;case a.IMAGE_EDITS:t="azure"===x?`
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
prompt = "${j}"

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
`;break;case a.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${s||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${s?`,
	prompt="${s.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${s||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${I}
${t}`}],190272)},629569,e=>{"use strict";var t=e.i(290571),o=e.i(95779),r=e.i(444755),a=e.i(673706),i=e.i(271645);let s=i.default.forwardRef((e,s)=>{let{color:n,children:l,className:d}=e,c=(0,t.__rest)(e,["color","children","className"]);return i.default.createElement("p",Object.assign({ref:s,className:(0,r.tremorTwMerge)("font-medium text-tremor-title",n?(0,a.getColorClassNames)(n,o.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",d)},c),l)});s.displayName="Title",e.s(["Title",()=>s],629569)},592968,e=>{"use strict";var t=e.i(491816);e.s(["Tooltip",()=>t.default])},936325,e=>{"use strict";var t=e.i(95779),o=e.i(444755),r=e.i(673706),a=e.i(271645);let i=a.default.forwardRef((e,i)=>{let{color:s,className:n,children:l}=e;return a.default.createElement("p",{ref:i,className:(0,o.tremorTwMerge)("text-tremor-default",s?(0,r.getColorClassNames)(s,t.colorPalette.text).textColor:(0,o.tremorTwMerge)("text-tremor-content","dark:text-dark-tremor-content"),n)},l)});i.displayName="Text",e.s(["default",()=>i])},599724,e=>{"use strict";var t=e.i(936325);e.s(["Text",()=>t.default])},304967,e=>{"use strict";var t=e.i(290571),o=e.i(271645),r=e.i(480731),a=e.i(95779),i=e.i(444755),s=e.i(673706);let n=(0,s.makeClassName)("Card"),l=o.default.forwardRef((e,l)=>{let{decoration:d="",decorationColor:c,children:p,className:m}=e,u=(0,t.__rest)(e,["decoration","decorationColor","children","className"]);return o.default.createElement("div",Object.assign({ref:l,className:(0,i.tremorTwMerge)(n("root"),"relative w-full text-left ring-1 rounded-tremor-default p-6","bg-tremor-background ring-tremor-ring shadow-tremor-card","dark:bg-dark-tremor-background dark:ring-dark-tremor-ring dark:shadow-dark-tremor-card",c?(0,s.getColorClassNames)(c,a.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",(e=>{if(!e)return"";switch(e){case r.HorizontalPositions.Left:return"border-l-4";case r.VerticalPositions.Top:return"border-t-4";case r.HorizontalPositions.Right:return"border-r-4";case r.VerticalPositions.Bottom:return"border-b-4";default:return""}})(d),m)},u),p)});l.displayName="Card",e.s(["Card",()=>l],304967)},503116,e=>{"use strict";let t=(0,e.i(475254).default)("clock",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["polyline",{points:"12 6 12 12 16 14",key:"68esgv"}]]);e.s(["Clock",()=>t],503116)},891547,e=>{"use strict";var t=e.i(843476),o=e.i(271645),r=e.i(487486),a=e.i(793479),i=e.i(337822),s=e.i(37727),n=e.i(975157),l=e.i(764205);e.s(["default",0,({onChange:e,value:d,className:c,accessToken:p,disabled:m})=>{let[u,g]=(0,o.useState)([]),[f,h]=(0,o.useState)(!1),[_,x]=(0,o.useState)(""),b=(0,o.useMemo)(()=>d??[],[d]);(0,o.useEffect)(()=>{(async()=>{if(p)try{let e=await (0,l.getGuardrailsList)(p);e.guardrails&&g(e.guardrails)}catch(e){console.error("Error fetching guardrails:",e)}})()},[p]);let y=(0,o.useMemo)(()=>u.filter(e=>e.guardrail_name&&!b.includes(e.guardrail_name)).filter(e=>!_||(e.guardrail_name??"").toLowerCase().includes(_.toLowerCase())),[u,b,_]);return(0,t.jsxs)(i.Popover,{open:f,onOpenChange:h,children:[(0,t.jsx)(i.PopoverTrigger,{asChild:!0,children:(0,t.jsx)("button",{type:"button",disabled:m,className:(0,n.cn)("min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",c),children:0===b.length?(0,t.jsx)("span",{className:"text-muted-foreground px-1",children:m?"Setting guardrails is a premium feature.":"Select guardrails"}):b.map(o=>(0,t.jsxs)(r.Badge,{variant:"secondary",className:"gap-1",children:[o,(0,t.jsx)("span",{role:"button",tabIndex:0,onClick:t=>{t.stopPropagation(),e(b.filter(e=>e!==o))},className:"inline-flex items-center","aria-label":`Remove ${o}`,children:(0,t.jsx)(s.X,{size:12})})]},o))})}),(0,t.jsxs)(i.PopoverContent,{align:"start",className:"w-[var(--radix-popover-trigger-width)] p-2",children:[(0,t.jsx)(a.Input,{autoFocus:!0,placeholder:"Search guardrails…",value:_,onChange:e=>x(e.target.value),className:"h-8 mb-2"}),(0,t.jsx)("div",{className:"max-h-60 overflow-y-auto",children:0===y.length?(0,t.jsx)("div",{className:"py-2 px-3 text-sm text-muted-foreground",children:"No matches"}):y.map(o=>{let r=o.guardrail_name;return(0,t.jsx)("button",{type:"button",className:"w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent",onClick:()=>e([...b,r]),children:r},r)})})]})]})}])},921511,e=>{"use strict";var t=e.i(843476),o=e.i(271645),r=e.i(487486),a=e.i(793479),i=e.i(337822),s=e.i(37727),n=e.i(975157),l=e.i(764205);function d(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let o=e.version_number??1,r=e.version_status??"draft";return{label:`${e.policy_name} — v${o} (${r})${e.description?` — ${e.description}`:""}`,value:"production"===r?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["default",0,({onChange:e,value:c,className:p,accessToken:m,disabled:u,onPoliciesLoaded:g})=>{let[f,h]=(0,o.useState)([]),[_,x]=(0,o.useState)(!1),[b,y]=(0,o.useState)(!1),[v,w]=(0,o.useState)("");(0,o.useEffect)(()=>{(async()=>{if(m){x(!0);try{let e=await (0,l.getPoliciesList)(m);e.policies&&(h(e.policies),g?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{x(!1)}}})()},[m,g]);let k=(0,o.useMemo)(()=>d(f),[f]),j=c??[],N=(0,o.useMemo)(()=>k.filter(e=>!j.includes(e.value)).filter(e=>!v||e.label.toLowerCase().includes(v.toLowerCase())),[k,j,v]),S=e=>k.find(t=>t.value===e)?.label??e;return(0,t.jsxs)(i.Popover,{open:b,onOpenChange:y,children:[(0,t.jsx)(i.PopoverTrigger,{asChild:!0,children:(0,t.jsx)("button",{type:"button",disabled:u||_,className:(0,n.cn)("min-h-9 w-full flex flex-wrap items-center gap-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-left disabled:opacity-50",p),children:0===j.length?(0,t.jsx)("span",{className:"text-muted-foreground px-1",children:u?"Setting policies is a premium feature.":_?"Loading policies…":"Select policies (production or published versions)"}):j.map(o=>(0,t.jsxs)(r.Badge,{variant:"secondary",className:"gap-1 inline-flex items-center",children:[S(o),(0,t.jsx)("span",{role:"button",tabIndex:0,onClick:t=>{t.stopPropagation(),e(j.filter(e=>e!==o))},className:"inline-flex items-center","aria-label":`Remove ${S(o)}`,children:(0,t.jsx)(s.X,{size:12})})]},o))})}),(0,t.jsxs)(i.PopoverContent,{align:"start",className:"w-[var(--radix-popover-trigger-width)] p-2",children:[(0,t.jsx)(a.Input,{autoFocus:!0,placeholder:"Search policies…",value:v,onChange:e=>w(e.target.value),className:"h-8 mb-2"}),(0,t.jsx)("div",{className:"max-h-60 overflow-y-auto",children:0===N.length?(0,t.jsx)("div",{className:"py-2 px-3 text-sm text-muted-foreground",children:"No matches"}):N.map(o=>(0,t.jsx)("button",{type:"button",className:"w-full text-left px-2 py-1.5 text-sm rounded hover:bg-accent",onClick:()=>e([...j,o.value]),children:o.label},o.value))})]})]})},"getPolicyOptionEntries",()=>d])},382373,387951,e=>{"use strict";var t=e.i(475254);let o=(0,t.default)("volume-2",[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]]);e.s(["Volume2",()=>o],382373);let r=(0,t.default)("mic",[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]]);e.s(["Mic",()=>r],387951)},434166,e=>{"use strict";function t(e,t){window.sessionStorage.setItem(e,btoa(encodeURIComponent(t).replace(/%([0-9A-F]{2})/g,(e,t)=>String.fromCharCode(parseInt(t,16)))))}function o(e){try{let t=window.sessionStorage.getItem(e);if(null===t)return null;return decodeURIComponent(atob(t).split("").map(e=>"%"+e.charCodeAt(0).toString(16).padStart(2,"0")).join(""))}catch{return null}}e.s(["getSecureItem",()=>o,"setSecureItem",()=>t])},254530,452598,e=>{"use strict";e.i(247167);var t=e.i(356449),o=e.i(764205);async function r(e,r,a,i,s,n,l,d,c,p,m,u,g,f,h,_,x,b,y,v,w,k,j,N,S){console.log=function(){},console.log("isLocal:",!1);let C=v||(0,o.getProxyBaseUrl)(),I={};s&&s.length>0&&(I["x-litellm-tags"]=s.join(","));let T=new t.default.OpenAI({apiKey:i,baseURL:C,dangerouslyAllowBrowser:!0,defaultHeaders:I});try{let t,o=Date.now(),i=!1,s={},v=!1,C=[];for await(let y of(f&&f.length>0&&(f.includes("__all__")?C.push({type:"mcp",server_label:"litellm",server_url:"litellm_proxy/mcp",require_approval:"never"}):f.forEach(e=>{if(e.startsWith("toolset:")){let t=e.slice(8),o=S?.find(e=>e.toolset_id===t),r=o?.toolset_name||t;C.push({type:"mcp",server_label:r,server_url:`litellm_proxy/mcp/${encodeURIComponent(r)}`,require_approval:"never"})}else{let t=w?.find(t=>t.server_id===e),o=t?.alias||t?.server_name||e,r=k?.[e]||[];C.push({type:"mcp",server_label:"litellm",server_url:`litellm_proxy/mcp/${o}`,require_approval:"never",...r.length>0?{allowed_tools:r}:{}})}})),await T.chat.completions.create({model:a,stream:!0,stream_options:{include_usage:!0},litellm_trace_id:p,messages:e,...m?{vector_store_ids:m}:{},...u?{guardrails:u}:{},...g?{policies:g}:{},...C.length>0?{tools:C,tool_choice:"auto"}:{},...void 0!==x?{temperature:x}:{},...void 0!==b?{max_tokens:b}:{},...N?{mock_testing_fallbacks:!0}:{}},{signal:n}))){console.log("Stream chunk:",y);let e=y.choices[0]?.delta;if(console.log("Delta content:",y.choices[0]?.delta?.content),console.log("Delta reasoning content:",e?.reasoning_content),!i&&(y.choices[0]?.delta?.content||e&&e.reasoning_content)&&(i=!0,t=Date.now()-o,console.log("First token received! Time:",t,"ms"),d?(console.log("Calling onTimingData with:",t),d(t)):console.log("onTimingData callback is not defined!")),y.choices[0]?.delta?.content){let e=y.choices[0].delta.content;r(e,y.model)}if(e&&e.image&&h&&(console.log("Image generated:",e.image),h(e.image.url,y.model)),e&&e.reasoning_content){let t=e.reasoning_content;l&&l(t)}if(e&&e.provider_specific_fields?.search_results&&_&&(console.log("Search results found:",e.provider_specific_fields.search_results),_(e.provider_specific_fields.search_results)),e&&e.provider_specific_fields){let t=e.provider_specific_fields;if(t.mcp_list_tools&&!s.mcp_list_tools&&(s.mcp_list_tools=t.mcp_list_tools,j&&!v)){v=!0;let e={type:"response.output_item.done",item_id:"mcp_list_tools",item:{type:"mcp_list_tools",tools:t.mcp_list_tools.map(e=>({name:e.function?.name||e.name||"",description:e.function?.description||e.description||"",input_schema:e.function?.parameters||e.input_schema||{}}))},timestamp:Date.now()};j(e),console.log("MCP list_tools event sent:",e)}t.mcp_tool_calls&&(s.mcp_tool_calls=t.mcp_tool_calls),t.mcp_call_results&&(s.mcp_call_results=t.mcp_call_results),(t.mcp_list_tools||t.mcp_tool_calls||t.mcp_call_results)&&console.log("MCP metadata found in chunk:",{mcp_list_tools:t.mcp_list_tools?"present":"absent",mcp_tool_calls:t.mcp_tool_calls?"present":"absent",mcp_call_results:t.mcp_call_results?"present":"absent"})}if(y.usage&&c){console.log("Usage data found:",y.usage);let e={completionTokens:y.usage.completion_tokens,promptTokens:y.usage.prompt_tokens,totalTokens:y.usage.total_tokens};y.usage.completion_tokens_details?.reasoning_tokens&&(e.reasoningTokens=y.usage.completion_tokens_details.reasoning_tokens),void 0!==y.usage.cost&&null!==y.usage.cost&&(e.cost=parseFloat(y.usage.cost)),c(e)}}j&&(s.mcp_tool_calls||s.mcp_call_results)&&s.mcp_tool_calls&&s.mcp_tool_calls.length>0&&s.mcp_tool_calls.forEach((e,t)=>{let o=e.function?.name||e.name||"",r=e.function?.arguments||e.arguments||"{}",a=s.mcp_call_results?.find(t=>t.tool_call_id===e.id||t.tool_call_id===e.call_id)||s.mcp_call_results?.[t],i={type:"response.output_item.done",item:{type:"mcp_call",name:o,arguments:"string"==typeof r?r:JSON.stringify(r),output:a?.result?"string"==typeof a.result?a.result:JSON.stringify(a.result):void 0},item_id:e.id||e.call_id,timestamp:Date.now()};j(i),console.log("MCP call event sent:",i)});let I=Date.now();y&&y(I-o)}catch(e){throw n?.aborted&&console.log("Chat completion request was cancelled"),e}}e.s(["makeOpenAIChatCompletionRequest",()=>r],254530);var a=e.i(727749);async function i(e,r,s,n,l=[],d,c,p,m,u,g,f,h,_,x,b,y,v,w,k,j,N,S){if(!n)throw Error("Virtual Key is required");if(!s||""===s.trim())throw Error("Model is required. Please select a model before sending a request.");console.log=function(){};let C=k||(0,o.getProxyBaseUrl)(),I={};l&&l.length>0&&(I["x-litellm-tags"]=l.join(","));let T=new t.default.OpenAI({apiKey:n,baseURL:C,dangerouslyAllowBrowser:!0,defaultHeaders:I});try{let t=Date.now(),o=!1,a=e.map(e=>(Array.isArray(e.content),{role:e.role,content:e.content,type:"message"})),i=[];_&&_.length>0&&(_.includes("__all__")?i.push({type:"mcp",server_label:"litellm",server_url:`${C}/mcp`,require_approval:"never"}):_.forEach(e=>{if(e.startsWith("toolset:")){let t=e.slice(8),o=S?.find(e=>e.toolset_id===t),r=o?.toolset_name||t;i.push({type:"mcp",server_label:r,server_url:`${C}/mcp/${encodeURIComponent(r)}`,require_approval:"never"})}else{let t=j?.find(t=>t.server_id===e),o=t?.server_name||e,r=N?.[e]||[];i.push({type:"mcp",server_label:o,server_url:`${C}/mcp/${encodeURIComponent(o)}`,require_approval:"never",...r.length>0?{allowed_tools:r}:{}})}})),v&&i.push({type:"code_interpreter",container:{type:"auto"}});let n=await T.responses.create({model:s,input:a,stream:!0,litellm_trace_id:u,...x?{previous_response_id:x}:{},...g?{vector_store_ids:g}:{},...f?{guardrails:f}:{},...h?{policies:h}:{},...i.length>0?{tools:i,tool_choice:"auto"}:{}},{signal:d}),l="",k={code:"",containerId:""};for await(let e of n)if(console.log("Response event:",e),"object"==typeof e&&null!==e){if((e.type?.startsWith("response.mcp_")||"response.output_item.done"===e.type&&(e.item?.type==="mcp_list_tools"||e.item?.type==="mcp_call"))&&(console.log("MCP event received:",e),y)){let t={type:e.type,sequence_number:e.sequence_number,output_index:e.output_index,item_id:e.item_id||e.item?.id,item:e.item,delta:e.delta,arguments:e.arguments,timestamp:Date.now()};y(t)}"response.output_item.done"===e.type&&e.item?.type==="mcp_call"&&e.item?.name&&(l=e.item.name,console.log("MCP tool used:",l)),E=k;var E,M=k="response.output_item.done"===e.type&&e.item?.type==="code_interpreter_call"?(console.log("Code interpreter call completed:",e.item),{code:e.item.code||"",containerId:e.item.container_id||""}):E;if("response.output_item.done"===e.type&&e.item?.type==="message"&&e.item?.content&&w){for(let t of e.item.content)if("output_text"===t.type&&t.annotations){let e=t.annotations.filter(e=>"container_file_citation"===e.type);(e.length>0||M.code)&&w({code:M.code,containerId:M.containerId,annotations:e})}}if("response.role.delta"===e.type)continue;if("response.output_text.delta"===e.type&&"string"==typeof e.delta){let a=e.delta;if(console.log("Text delta",a),a.length>0&&(r("assistant",a,s),!o)){o=!0;let e=Date.now()-t;console.log("First token received! Time:",e,"ms"),p&&p(e)}}if("response.reasoning.delta"===e.type&&"delta"in e){let t=e.delta;"string"==typeof t&&c&&c(t)}if("response.completed"===e.type&&"response"in e){let t=e.response,o=t.usage;if(console.log("Usage data:",o),console.log("Response completed event:",t),t.id&&b&&(console.log("Response ID for session management:",t.id),b(t.id)),o&&m){console.log("Usage data:",o);let e={completionTokens:o.output_tokens,promptTokens:o.input_tokens,totalTokens:o.total_tokens};o.completion_tokens_details?.reasoning_tokens&&(e.reasoningTokens=o.completion_tokens_details.reasoning_tokens),m(e,l)}}}return n}catch(e){throw d?.aborted?console.log("Responses API request was cancelled"):a.default.fromBackend(`Error occurred while generating model response. Please try again. Error: ${e}`),e}}e.s(["makeOpenAIResponsesRequest",()=>i],452598)},355343,e=>{"use strict";var t=e.i(843476),o=e.i(273443),r=e.i(643531);e.s(["default",0,({events:e,className:a})=>{if(!e||0===e.length)return null;let i=e.find(e=>"response.output_item.done"===e.type&&e.item?.type==="mcp_list_tools"&&e.item.tools&&e.item.tools.length>0),s=e.filter(e=>"response.output_item.done"===e.type&&e.item?.type==="mcp_call");if(!i&&0===s.length)return null;let n=i?["list-tools"]:s.map((e,t)=>`mcp-call-${t}`);return(0,t.jsx)("div",{className:`mcp-events-display ${a||""}`,children:(0,t.jsxs)("div",{className:"relative pl-5",children:[(0,t.jsx)("div",{className:"absolute left-[9px] top-[18px] bottom-0 w-[0.5px] bg-border opacity-80"}),(0,t.jsxs)(o.Accordion,{type:"multiple",defaultValue:n,className:"w-full",children:[i&&(0,t.jsxs)(o.AccordionItem,{value:"list-tools",className:"border-none",children:[(0,t.jsx)(o.AccordionTrigger,{className:"py-1 text-sm text-muted-foreground hover:text-foreground hover:no-underline",children:"List tools"}),(0,t.jsx)(o.AccordionContent,{className:"pt-1",children:(0,t.jsx)("div",{children:i.item?.tools?.map((e,o)=>(0,t.jsx)("div",{className:"font-mono text-sm text-foreground/80 bg-background relative z-[1] py-0",children:e.name},o))})})]}),s.map((e,a)=>(0,t.jsxs)(o.AccordionItem,{value:`mcp-call-${a}`,className:"border-none",children:[(0,t.jsx)(o.AccordionTrigger,{className:"py-1 text-sm text-muted-foreground hover:text-foreground hover:no-underline",children:e.item?.name||"Tool call"}),(0,t.jsx)(o.AccordionContent,{className:"pt-1",children:(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"mb-3 bg-background relative z-[1]",children:[(0,t.jsx)("div",{className:"text-sm text-muted-foreground font-medium mb-1",children:"Request"}),(0,t.jsx)("div",{className:"bg-muted border border-border rounded-md p-2 text-xs",children:e.item?.arguments&&(0,t.jsx)("pre",{className:"font-mono text-foreground m-0 whitespace-pre-wrap break-words",children:(()=>{try{return JSON.stringify(JSON.parse(e.item.arguments),null,2)}catch(t){return e.item.arguments}})()})})]}),(0,t.jsx)("div",{className:"mb-3 bg-background relative z-[1]",children:(0,t.jsxs)("div",{className:"flex items-center text-sm text-muted-foreground",children:[(0,t.jsx)(r.Check,{className:"h-4 w-4 mr-1.5 text-emerald-500"}),"Approved"]})}),e.item?.output&&(0,t.jsxs)("div",{className:"mb-0 bg-background relative z-[1]",children:[(0,t.jsx)("div",{className:"text-sm text-muted-foreground font-medium mb-1",children:"Response"}),(0,t.jsx)("div",{className:"text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap font-mono",children:e.item.output})]})]})})]},`mcp-call-${a}`))]})]})})}])},966988,e=>{"use strict";var t=e.i(843476),o=e.i(271645),r=e.i(519455),a=e.i(918789),i=e.i(650056),s=e.i(219470),n=e.i(664659),l=e.i(463059),d=e.i(341240);e.s(["default",0,({reasoningContent:e})=>{let[c,p]=(0,o.useState)(!0);return e?(0,t.jsxs)("div",{className:"reasoning-content mt-1 mb-2",children:[(0,t.jsxs)(r.Button,{variant:"ghost",size:"sm",className:"flex items-center text-xs text-muted-foreground hover:text-foreground",onClick:()=>p(!c),children:[(0,t.jsx)(d.Lightbulb,{className:"h-3.5 w-3.5"}),c?"Hide reasoning":"Show reasoning",c?(0,t.jsx)(n.ChevronDown,{className:"h-3 w-3 ml-1"}):(0,t.jsx)(l.ChevronRight,{className:"h-3 w-3 ml-1"})]}),c&&(0,t.jsx)("div",{className:"mt-2 p-3 bg-muted border border-border rounded-md text-sm text-foreground",children:(0,t.jsx)(a.default,{components:{code({inline:e,className:o,children:r,...a}){let n=/language-(\w+)/.exec(o||"");return!e&&n?(0,t.jsx)(i.Prism,{style:s.coy,language:n[1],PreTag:"div",className:"rounded-md my-2",...a,children:String(r).replace(/\n$/,"")}):(0,t.jsx)("code",{className:`${o} px-1.5 py-0.5 rounded bg-muted text-sm font-mono`,...a,children:r})}},children:e})})]}):null}])}]);