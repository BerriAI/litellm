(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,972520,e=>{let s=(0,e.i(475254).f)("arrow-right",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]]);e.s(["f",0,s],972520)},411929,e=>{var s=e.i(843476),t=e.i(271645),r=e.i(972520),o=e.i(174886),n=e.i(519455),a=e.i(515288),i=e.i(624687),l=e.i(571303),c=e.i(602869),d=e.i(417385);let u=({accessToken:e})=>{let[u,m]=(0,t.useState)(`{
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
}`),[f,p]=(0,t.useState)(""),[h,x]=(0,t.useState)(!1),j=async()=>{x(!0);try{let o;try{o=JSON.parse(u)}catch(e){d.f.fromError("Invalid JSON in request body"),x(!1);return}let n={call_type:"completion",request_body:o};if(!e){d.f.fromError("No access token found"),x(!1);return}let a=await (0,c.DU)(e,n);if(a.raw_request_api_base&&a.raw_request_body){var s,t,r;let e,o,n=(s=a.raw_request_api_base,t=a.raw_request_body,r=a.raw_request_headers||{},e=JSON.stringify(t,null,2).split("\n").map(e=>`  ${e}`).join("\n"),o=Object.entries(r).map(([e,s])=>`-H '${e}: ${s}'`).join(" \\\n  "),`curl -X POST \\
  ${s} \\
  ${o?`${o} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);p(n),d.f.success("Request transformed successfully")}else{let e="string"==typeof a?a:JSON.stringify(a);p(e),d.f.info("Transformed request received in unexpected format")}}catch(e){console.error("Error transforming request:",e),d.f.fromError("Failed to transform request")}finally{x(!1)}};return(0,s.jsxs)("div",{className:"p-2",children:[(0,s.jsx)("h1",{className:"text-lg font-medium text-foreground",children:"Playground"}),(0,s.jsx)("p",{className:"text-sm text-muted-foreground",children:"See how LiteLLM transforms your request for the specified provider."}),(0,s.jsxs)("div",{className:"mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2",children:[(0,s.jsxs)(a.N,{children:[(0,s.jsxs)(a.Y,{children:[(0,s.jsx)(a.v,{className:"text-2xl font-bold",children:"Original Request"}),(0,s.jsx)(a.u,{children:"The request you would send to LiteLLM /chat/completions endpoint."})]}),(0,s.jsx)(a.O,{children:(0,s.jsx)(i.f,{className:"h-72 resize-none p-4 font-mono text-sm field-sizing-fixed",value:u,onChange:e=>m(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),j())},placeholder:"Press Cmd/Ctrl + Enter to transform"})}),(0,s.jsx)(a.S,{className:"justify-end",children:(0,s.jsxs)(n.K,{onClick:j,disabled:h,children:[(0,s.jsx)("span",{children:"Transform"}),h?(0,s.jsx)(l.f,{className:"size-4"}):(0,s.jsx)(r.f,{})]})})]}),(0,s.jsxs)(a.N,{children:[(0,s.jsxs)(a.Y,{children:[(0,s.jsx)(a.v,{className:"text-2xl font-bold",children:"Transformed Request"}),(0,s.jsx)(a.u,{children:"How LiteLLM transforms your request for the specified provider."}),(0,s.jsx)("p",{className:"mt-2 text-xs text-muted-foreground",children:"Note: Sensitive headers are not shown."})]}),(0,s.jsx)(a.O,{children:(0,s.jsxs)("div",{className:"relative rounded-md bg-muted",children:[(0,s.jsx)("pre",{className:"h-72 overflow-auto p-4 font-mono text-sm",children:f||`curl -X POST \\
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
  }'`}),(0,s.jsx)(n.K,{variant:"ghost",size:"icon-sm","aria-label":"Copy to clipboard",className:"absolute top-2 right-2",onClick:()=>{navigator.clipboard.writeText(f||""),d.f.success("Copied to clipboard")},children:(0,s.jsx)(o.f,{})})]})})]})]}),(0,s.jsx)("div",{className:"mt-4 text-right",children:(0,s.jsxs)("p",{className:"text-sm text-muted-foreground",children:["Found an error? File an issue"," ",(0,s.jsx)("a",{className:"underline underline-offset-4",href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer",children:"here"}),"."]})})]})};var m=e.i(135214);e.s(["default",0,function(){let{accessToken:e}=(0,m.f)();return(0,s.jsx)(u,{accessToken:e})}],411929)}])})();