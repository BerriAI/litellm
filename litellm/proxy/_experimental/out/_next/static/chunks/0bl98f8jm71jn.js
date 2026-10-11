(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,972520,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-right",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]]);e.s(["ArrowRight",0,t],972520)},411929,e=>{"use strict";var t=e.i(843476),r=e.i(271645),s=e.i(972520),a=e.i(174886),o=e.i(519455),n=e.i(515288),i=e.i(624687),l=e.i(571303),d=e.i(602869),c=e.i(417385);let u=({accessToken:e})=>{let[u,m]=(0,r.useState)(`{
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
}`),[p,h]=(0,r.useState)(""),[x,f]=(0,r.useState)(!1),g=async()=>{f(!0);try{let a;try{a=JSON.parse(u)}catch(e){c.toast.fromError("Invalid JSON in request body"),f(!1);return}let o={call_type:"completion",request_body:a};if(!e){c.toast.fromError("No access token found"),f(!1);return}let n=await (0,d.transformRequestCall)(e,o);if(n.raw_request_api_base&&n.raw_request_body){var t,r,s;let e,a,o=(t=n.raw_request_api_base,r=n.raw_request_body,s=n.raw_request_headers||{},e=JSON.stringify(r,null,2).split("\n").map(e=>`  ${e}`).join("\n"),a=Object.entries(s).map(([e,t])=>`-H '${e}: ${t}'`).join(" \\\n  "),`curl -X POST \\
  ${t} \\
  ${a?`${a} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);h(o),c.toast.success("Request transformed successfully")}else{let e="string"==typeof n?n:JSON.stringify(n);h(e),c.toast.info("Transformed request received in unexpected format")}}catch(e){console.error("Error transforming request:",e),c.toast.fromError("Failed to transform request")}finally{f(!1)}};return(0,t.jsxs)("div",{className:"p-2",children:[(0,t.jsx)("h1",{className:"text-lg font-medium text-foreground",children:"Playground"}),(0,t.jsx)("p",{className:"text-sm text-muted-foreground",children:"See how LiteLLM transforms your request for the specified provider."}),(0,t.jsxs)("div",{className:"mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2",children:[(0,t.jsxs)(n.Card,{children:[(0,t.jsxs)(n.CardHeader,{children:[(0,t.jsx)(n.CardTitle,{className:"text-2xl font-bold",children:"Original Request"}),(0,t.jsx)(n.CardDescription,{children:"The request you would send to LiteLLM /chat/completions endpoint."})]}),(0,t.jsx)(n.CardContent,{children:(0,t.jsx)(i.Textarea,{className:"h-72 resize-none p-4 font-mono text-sm field-sizing-fixed",value:u,onChange:e=>m(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),g())},placeholder:"Press Cmd/Ctrl + Enter to transform"})}),(0,t.jsx)(n.CardFooter,{className:"justify-end",children:(0,t.jsxs)(o.Button,{onClick:g,disabled:x,children:[(0,t.jsx)("span",{children:"Transform"}),x?(0,t.jsx)(l.UiLoadingSpinner,{className:"size-4"}):(0,t.jsx)(s.ArrowRight,{})]})})]}),(0,t.jsxs)(n.Card,{children:[(0,t.jsxs)(n.CardHeader,{children:[(0,t.jsx)(n.CardTitle,{className:"text-2xl font-bold",children:"Transformed Request"}),(0,t.jsx)(n.CardDescription,{children:"How LiteLLM transforms your request for the specified provider."}),(0,t.jsx)("p",{className:"mt-2 text-xs text-muted-foreground",children:"Note: Sensitive headers are not shown."})]}),(0,t.jsx)(n.CardContent,{children:(0,t.jsxs)("div",{className:"relative rounded-md bg-muted",children:[(0,t.jsx)("pre",{className:"h-72 overflow-auto p-4 font-mono text-sm",children:p||`curl -X POST \\
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
  }'`}),(0,t.jsx)(o.Button,{variant:"ghost",size:"icon-sm","aria-label":"Copy to clipboard",className:"absolute top-2 right-2",onClick:()=>{navigator.clipboard.writeText(p||""),c.toast.success("Copied to clipboard")},children:(0,t.jsx)(a.Copy,{})})]})})]})]}),(0,t.jsx)("div",{className:"mt-4 text-right",children:(0,t.jsxs)("p",{className:"text-sm text-muted-foreground",children:["Found an error? File an issue"," ",(0,t.jsx)("a",{className:"underline underline-offset-4",href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer",children:"here"}),"."]})})]})};var m=e.i(135214);e.s(["default",0,function(){let{accessToken:e}=(0,m.default)();return(0,t.jsx)(u,{accessToken:e})}],411929)}]);