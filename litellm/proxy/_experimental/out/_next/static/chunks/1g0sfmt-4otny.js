(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,972520,e=>{let s=(0,e.i(475254).f)("arrow-right",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]]);e.s(["f",0,s],972520)},411929,e=>{var s=e.i(843476),t=e.i(271645),r=e.i(345771),a=e.i(322831),n=e.i(972520),o=e.i(174886),l=e.i(519455),i=e.i(515288),m=e.i(624687),u=e.i(571303),c=e.i(602869),f=e.i(417385);let d=({accessToken:e})=>{let{t:d}=(0,a.f)(),[p,x]=(0,t.useState)(`{
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
}`),[h,q]=(0,t.useState)(""),[j,g]=(0,t.useState)(!1),R=async()=>{g(!0);try{let a;try{a=JSON.parse(p)}catch(e){f.f.fromError(d("transformRequest.transformRequestPanel.toasts.invalidJson")),g(!1);return}let n={call_type:"completion",request_body:a};if(!e){f.f.fromError(d("transformRequest.transformRequestPanel.toasts.noAccessToken")),g(!1);return}let o=await (0,c.DU)(e,n);if(o.raw_request_api_base&&o.raw_request_body){var s,t,r;let e,a,n=(s=o.raw_request_api_base,t=o.raw_request_body,r=o.raw_request_headers||{},e=JSON.stringify(t,null,2).split("\n").map(e=>`  ${e}`).join("\n"),a=Object.entries(r).map(([e,s])=>`-H '${e}: ${s}'`).join(" \\\n  "),`curl -X POST \\
  ${s} \\
  ${a?`${a} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);q(n),f.f.success(d("transformRequest.transformRequestPanel.toasts.transformed"))}else{let e="string"==typeof o?o:JSON.stringify(o);q(e),f.f.info(d("transformRequest.transformRequestPanel.toasts.unexpectedFormat"))}}catch(e){console.error("Error transforming request:",e),f.f.fromError(d("transformRequest.transformRequestPanel.toasts.failed"))}finally{g(!1)}};return(0,s.jsxs)("div",{className:"p-2",children:[(0,s.jsx)("h1",{className:"text-lg font-medium text-foreground",children:d("transformRequest.transformRequestPanel.title")}),(0,s.jsx)("p",{className:"text-sm text-muted-foreground",children:d("transformRequest.transformRequestPanel.description")}),(0,s.jsxs)("div",{className:"mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2",children:[(0,s.jsxs)(i.N,{children:[(0,s.jsxs)(i.Y,{children:[(0,s.jsx)(i.v,{className:"text-2xl font-bold",children:d("transformRequest.transformRequestPanel.original.title")}),(0,s.jsx)(i.u,{children:d("transformRequest.transformRequestPanel.original.description")})]}),(0,s.jsx)(i.O,{children:(0,s.jsx)(m.f,{className:"h-72 resize-none p-4 font-mono text-sm field-sizing-fixed",value:p,onChange:e=>x(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),R())},placeholder:d("transformRequest.transformRequestPanel.original.placeholder")})}),(0,s.jsx)(i.S,{className:"justify-end",children:(0,s.jsxs)(l.K,{onClick:R,disabled:j,children:[(0,s.jsx)("span",{children:d("transformRequest.transformRequestPanel.original.transform")}),j?(0,s.jsx)(u.f,{className:"size-4"}):(0,s.jsx)(n.f,{})]})})]}),(0,s.jsxs)(i.N,{children:[(0,s.jsxs)(i.Y,{children:[(0,s.jsx)(i.v,{className:"text-2xl font-bold",children:d("transformRequest.transformRequestPanel.transformed.title")}),(0,s.jsx)(i.u,{children:d("transformRequest.transformRequestPanel.transformed.description")}),(0,s.jsx)("p",{className:"mt-2 text-xs text-muted-foreground",children:d("transformRequest.transformRequestPanel.transformed.sensitiveHeaders")})]}),(0,s.jsx)(i.O,{children:(0,s.jsxs)("div",{className:"relative rounded-md bg-muted",children:[(0,s.jsx)("pre",{className:"h-72 overflow-auto p-4 font-mono text-sm",children:h||`curl -X POST \\
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
  }'`}),(0,s.jsx)(l.K,{variant:"ghost",size:"icon-sm","aria-label":d("transformRequest.transformRequestPanel.transformed.copy"),className:"absolute top-2 right-2",onClick:()=>{navigator.clipboard.writeText(h||""),f.f.success(d("transformRequest.transformRequestPanel.toasts.copied"))},children:(0,s.jsx)(o.f,{})})]})})]})]}),(0,s.jsx)("div",{className:"mt-4 text-right",children:(0,s.jsx)("p",{className:"text-sm text-muted-foreground",children:(0,s.jsx)(r.f,{i18nKey:"transformRequest.transformRequestPanel.fileIssue",components:{issues:(0,s.jsx)("a",{className:"underline underline-offset-4",href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer"})}})})})]})};var p=e.i(135214);e.s(["default",0,function(){let{accessToken:e}=(0,p.f)();return(0,s.jsx)(d,{accessToken:e})}],411929)}])})();