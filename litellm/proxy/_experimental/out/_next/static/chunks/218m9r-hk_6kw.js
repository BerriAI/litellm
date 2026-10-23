(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,972520,e=>{let r=(0,e.i(475254).f)("arrow-right",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]]);e.s(["f",0,r],972520)},411929,e=>{var r=e.i(843476),s=e.i(271645),t=e.i(972520),a=e.i(174886),o=e.i(519455),i=e.i(515288),d=e.i(624687),n=e.i(571303),l=e.i(602869),c=e.i(417385);let m=({accessToken:e})=>{let[m,u]=(0,s.useState)(`{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
}`),[f,p]=(0,s.useState)(""),[x,h]=(0,s.useState)(!1),g=async()=>{h(!0);try{let a;try{a=JSON.parse(m)}catch(e){c.f.fromError("Invalid JSON in request body"),h(!1);return}let o={call_type:"completion",request_body:a};if(!e){c.f.fromError("No access token found"),h(!1);return}let i=await (0,l.DU)(e,o);if(i.raw_request_api_base&&i.raw_request_body){var r,s,t;let e,a,o=(r=i.raw_request_api_base,s=i.raw_request_body,t=i.raw_request_headers||{},e=JSON.stringify(s,null,2).split("\n").map(e=>`  ${e}`).join("\n"),a=Object.entries(t).map(([e,r])=>`-H '${e}: ${r}'`).join(" \\\n  "),`curl -X POST \\
  ${r} \\
  ${a?`${a} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);p(o),c.f.success("Request transformed successfully")}else{let e="string"==typeof i?i:JSON.stringify(i);p(e),c.f.info("Transformed request received in unexpected format")}}catch(e){console.error("Error transforming request:",e),c.f.fromError("Failed to transform request")}finally{h(!1)}};return(0,r.jsxs)("div",{className:"p-2",children:[(0,r.jsx)("h1",{className:"text-lg font-medium text-foreground",children:"Playground"}),(0,r.jsx)("p",{className:"text-sm text-muted-foreground",children:"See how LiteLLM transforms your request for the specified provider."}),(0,r.jsxs)("div",{className:"mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2",children:[(0,r.jsxs)(i.N,{children:[(0,r.jsxs)(i.Y,{children:[(0,r.jsx)(i.v,{className:"text-2xl font-bold",children:"Original Request"}),(0,r.jsx)(i.u,{children:"The request you would send to LiteLLM /chat/completions endpoint."})]}),(0,r.jsx)(i.O,{children:(0,r.jsx)(d.f,{className:"h-72 resize-none p-4 font-mono text-sm field-sizing-fixed",value:m,onChange:e=>u(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),g())},placeholder:"Press Cmd/Ctrl + Enter to transform"})}),(0,r.jsx)(i.S,{className:"justify-end",children:(0,r.jsxs)(o.K,{onClick:g,disabled:x,children:[(0,r.jsx)("span",{children:"Transform"}),x?(0,r.jsx)(n.f,{className:"size-4"}):(0,r.jsx)(t.f,{})]})})]}),(0,r.jsxs)(i.N,{children:[(0,r.jsxs)(i.Y,{children:[(0,r.jsx)(i.v,{className:"text-2xl font-bold",children:"Transformed Request"}),(0,r.jsx)(i.u,{children:"How LiteLLM transforms your request for the specified provider."}),(0,r.jsx)("p",{className:"mt-2 text-xs text-muted-foreground",children:"Note: Sensitive headers are not shown."})]}),(0,r.jsx)(i.O,{children:(0,r.jsxs)("div",{className:"relative rounded-md bg-muted",children:[(0,r.jsx)("pre",{className:"h-72 overflow-auto p-4 font-mono text-sm",children:f||`curl -X POST \\
  https://api.openai.com/v1/chat/completions \\
  -H 'Authorization: Bearer sk-xxx' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    }
  ],
  "temperature": 0.7
  }'`}),(0,r.jsx)(o.K,{variant:"ghost",size:"icon-sm","aria-label":"Copy to clipboard",className:"absolute top-2 right-2",onClick:()=>{navigator.clipboard.writeText(f||""),c.f.success("Copied to clipboard")},children:(0,r.jsx)(a.f,{})})]})})]})]}),(0,r.jsx)("div",{className:"mt-4 text-right",children:(0,r.jsxs)("p",{className:"text-sm text-muted-foreground",children:["Found an error? File an issue"," ",(0,r.jsx)("a",{className:"underline underline-offset-4",href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer",children:"here"}),"."]})})]})};var u=e.i(135214);e.s(["default",0,function(){let{accessToken:e}=(0,u.f)();return(0,r.jsx)(m,{accessToken:e})}],411929)},515288,e=>{var r=e.i(843476),s=e.i(271645),t=e.i(196631);let a=s.forwardRef(({className:e,size:s="default",...a},o)=>(0,r.jsx)("div",{ref:o,"data-slot":"card","data-size":s,className:(0,t.c)("group/card flex flex-col gap-(--card-spacing) rounded-xl bg-card py-(--card-spacing) text-sm text-card-foreground shadow-xs ring-1 ring-foreground/10 [--card-spacing:--spacing(6)] has-[>img:first-child]:pt-0 data-[size=sm]:[--card-spacing:--spacing(4)] *:[img:first-child]:rounded-t-xl *:[img:last-child]:rounded-b-xl",e),...a}));a.displayName="Card";let o=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-header",className:(0,t.c)("group/card-header @container/card-header grid auto-rows-min items-start gap-1 rounded-t-xl px-(--card-spacing) has-data-[slot=card-action]:grid-cols-[1fr_auto] has-data-[slot=card-description]:grid-rows-[auto_auto] [.border-b]:pb-(--card-spacing)",e),...s}));o.displayName="CardHeader";let i=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-title",className:(0,t.c)("text-base leading-normal font-medium group-data-[size=sm]/card:text-sm",e),...s}));i.displayName="CardTitle";let d=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-description",className:(0,t.c)("text-sm text-muted-foreground",e),...s}));d.displayName="CardDescription";let n=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-action",className:(0,t.c)("col-start-2 row-span-2 row-start-1 self-start justify-self-end",e),...s}));n.displayName="CardAction";let l=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-content",className:(0,t.c)("px-(--card-spacing)",e),...s}));l.displayName="CardContent";let c=s.forwardRef(({className:e,...s},a)=>(0,r.jsx)("div",{ref:a,"data-slot":"card-footer",className:(0,t.c)("flex items-center rounded-b-xl px-(--card-spacing) [.border-t]:pt-(--card-spacing)",e),...s}));c.displayName="CardFooter",e.s(["N",0,a,"e",0,n,"O",0,l,"u",0,d,"S",0,c,"Y",0,o,"v",0,i])}])})();