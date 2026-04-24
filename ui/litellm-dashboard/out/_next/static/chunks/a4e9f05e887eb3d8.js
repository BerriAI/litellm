(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},174886,e=>{"use strict";var t=e.i(991124);e.s(["Copy",()=>t.default])},662316,e=>{"use strict";var t=e.i(843476),r=e.i(271645),s=e.i(519455),a=e.i(174886),l=e.i(764205),o=e.i(727749);e.s(["default",0,({accessToken:e})=>{let[n,i]=(0,r.useState)(`{
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
}`),[d,c]=(0,r.useState)(""),[u,m]=(0,r.useState)(!1),f=async()=>{m(!0);try{let a;try{a=JSON.parse(n)}catch(e){o.default.fromBackend("Invalid JSON in request body"),m(!1);return}let i={call_type:"completion",request_body:a};if(!e){o.default.fromBackend("No access token found"),m(!1);return}let d=await (0,l.transformRequestCall)(e,i);if(d.raw_request_api_base&&d.raw_request_body){var t,r,s;let e,a,l=(t=d.raw_request_api_base,r=d.raw_request_body,s=d.raw_request_headers||{},e=JSON.stringify(r,null,2).split("\n").map(e=>`  ${e}`).join("\n"),a=Object.entries(s).map(([e,t])=>`-H '${e}: ${t}'`).join(" \\\n  "),`curl -X POST \\
  ${t} \\
  ${a?`${a} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);c(l),o.default.success("Request transformed successfully")}else{let e="string"==typeof d?d:JSON.stringify(d);c(e),o.default.info("Transformed request received in unexpected format")}}catch(e){console.error("Error transforming request:",e),o.default.fromBackend("Failed to transform request")}finally{m(!1)}};return(0,t.jsxs)("div",{className:"w-full m-2 overflow-hidden",children:[(0,t.jsx)("h1",{className:"text-2xl font-semibold",children:"Playground"}),(0,t.jsx)("p",{className:"text-sm text-muted-foreground",children:"See how LiteLLM transforms your request for the specified provider."}),(0,t.jsxs)("div",{className:"flex gap-4 w-full min-w-0 overflow-hidden mt-4",children:[(0,t.jsxs)("div",{className:"flex-1 basis-1/2 flex flex-col border border-border rounded-lg p-6 overflow-hidden max-h-[600px] min-w-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("h2",{className:"text-2xl font-bold mb-1",children:"Original Request"}),(0,t.jsx)("p",{className:"text-muted-foreground m-0",children:"The request you would send to LiteLLM /chat/completions endpoint."})]}),(0,t.jsx)("textarea",{className:"flex-1 w-full min-h-[240px] p-4 border border-border rounded-md font-mono text-sm resize-none mb-6 overflow-auto",value:n,onChange:e=>i(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),f())},placeholder:"Press Cmd/Ctrl + Enter to transform"}),(0,t.jsx)("div",{className:"flex justify-end mt-auto",children:(0,t.jsxs)(s.Button,{onClick:f,disabled:u,className:"bg-black text-white hover:bg-black/90",children:[(0,t.jsx)("span",{children:u?"Transforming...":"Transform"}),(0,t.jsx)("span",{children:"→"})]})})]}),(0,t.jsxs)("div",{className:"flex-1 basis-1/2 flex flex-col border border-border rounded-lg p-6 overflow-hidden max-h-[800px] min-w-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("h2",{className:"text-2xl font-bold mb-1",children:"Transformed Request"}),(0,t.jsx)("p",{className:"text-muted-foreground m-0",children:"How LiteLLM transforms your request for the specified provider."}),(0,t.jsx)("br",{}),(0,t.jsx)("p",{className:"text-muted-foreground m-0 text-xs",children:"Note: Sensitive headers are not shown."})]}),(0,t.jsxs)("div",{className:"relative bg-muted rounded-md flex-1 flex flex-col overflow-hidden",children:[(0,t.jsx)("pre",{className:"p-4 font-mono text-sm m-0 overflow-auto flex-1",children:d||`curl -X POST \\
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
  }'`}),(0,t.jsx)(s.Button,{variant:"ghost",size:"icon",className:"absolute right-2 top-2 h-7 w-7",onClick:()=>{navigator.clipboard.writeText(d||""),o.default.success("Copied to clipboard")},"aria-label":"Copy",children:(0,t.jsx)(a.Copy,{className:"h-3.5 w-3.5"})})]})]})]}),(0,t.jsx)("div",{className:"mt-4 text-right w-full",children:(0,t.jsxs)("p",{className:"text-sm text-muted-foreground",children:["Found an error? File an issue"," ",(0,t.jsx)("a",{href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer",className:"text-primary hover:text-primary/80 underline",children:"here"}),"."]})})]})}])},715288,e=>{"use strict";var t=e.i(843476),r=e.i(662316),s=e.i(135214);e.s(["default",0,()=>{let{accessToken:e}=(0,s.default)();return(0,t.jsx)(r.default,{accessToken:e})}])}]);