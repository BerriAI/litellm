(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,629569,e=>{"use strict";var t=e.i(290571),r=e.i(95779),s=e.i(444755),i=e.i(673706),o=e.i(271645);let l=o.default.forwardRef((e,l)=>{let{color:a,children:n,className:d}=e,c=(0,t.__rest)(e,["color","children","className"]);return o.default.createElement("p",Object.assign({ref:l,className:(0,s.tremorTwMerge)("font-medium text-tremor-title",a?(0,i.getColorClassNames)(a,r.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",d)},c),n)});l.displayName="Title",e.s(["Title",()=>l],629569)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},662316,e=>{"use strict";var t=e.i(843476),r=e.i(271645),s=e.i(464571),i=e.i(166406),o=e.i(629569),l=e.i(764205),a=e.i(727749);e.s(["default",0,({accessToken:e})=>{let[n,d]=(0,r.useState)(`{
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
}`),[c,p]=(0,r.useState)(""),[u,m]=(0,r.useState)(!1),x=async()=>{m(!0);try{let i;try{i=JSON.parse(n)}catch(e){a.default.fromBackend("Invalid JSON in request body"),m(!1);return}let o={call_type:"completion",request_body:i};if(!e){a.default.fromBackend("No access token found"),m(!1);return}let d=await (0,l.transformRequestCall)(e,o);if(d.raw_request_api_base&&d.raw_request_body){var t,r,s;let e,i,o=(t=d.raw_request_api_base,r=d.raw_request_body,s=d.raw_request_headers||{},e=JSON.stringify(r,null,2).split("\n").map(e=>`  ${e}`).join("\n"),i=Object.entries(s).map(([e,t])=>`-H '${e}: ${t}'`).join(" \\\n  "),`curl -X POST \\
  ${t} \\
  ${i?`${i} \\
  `:""}-H 'Content-Type: application/json' \\
  -d '{
${e}
  }'`);p(o),a.default.success("Request transformed successfully")}else{let e="string"==typeof d?d:JSON.stringify(d);p(e),a.default.info("Transformed request received in unexpected format")}}catch(e){console.error("Error transforming request:",e),a.default.fromBackend("Failed to transform request")}finally{m(!1)}};return(0,t.jsxs)("div",{className:"w-full m-2",style:{overflow:"hidden"},children:[(0,t.jsx)(o.Title,{children:"Playground"}),(0,t.jsx)("p",{className:"text-sm text-gray-500",children:"See how LiteLLM transforms your request for the specified provider."}),(0,t.jsxs)("div",{style:{display:"flex",gap:"16px",width:"100%",minWidth:0,overflow:"hidden"},className:"mt-4",children:[(0,t.jsxs)("div",{style:{flex:"1 1 50%",display:"flex",flexDirection:"column",border:"1px solid #e8e8e8",borderRadius:"8px",padding:"24px",overflow:"hidden",maxHeight:"600px",minWidth:0},children:[(0,t.jsxs)("div",{style:{marginBottom:"24px"},children:[(0,t.jsx)("h2",{style:{fontSize:"24px",fontWeight:"bold",margin:"0 0 4px 0"},children:"Original Request"}),(0,t.jsx)("p",{style:{color:"#666",margin:0},children:"The request you would send to LiteLLM /chat/completions endpoint."})]}),(0,t.jsx)("textarea",{style:{flex:"1 1 auto",width:"100%",minHeight:"240px",padding:"16px",border:"1px solid #e8e8e8",borderRadius:"6px",fontFamily:"monospace",fontSize:"14px",resize:"none",marginBottom:"24px",overflow:"auto"},value:n,onChange:e=>d(e.target.value),onKeyDown:e=>{(e.metaKey||e.ctrlKey)&&"Enter"===e.key&&(e.preventDefault(),x())},placeholder:"Press Cmd/Ctrl + Enter to transform"}),(0,t.jsx)("div",{style:{display:"flex",justifyContent:"flex-end",marginTop:"auto"},children:(0,t.jsxs)(s.Button,{type:"primary",style:{backgroundColor:"#000",display:"flex",alignItems:"center",gap:"8px"},onClick:x,loading:u,children:[(0,t.jsx)("span",{children:"Transform"}),(0,t.jsx)("span",{children:"→"})]})})]}),(0,t.jsxs)("div",{style:{flex:"1 1 50%",display:"flex",flexDirection:"column",border:"1px solid #e8e8e8",borderRadius:"8px",padding:"24px",overflow:"hidden",maxHeight:"800px",minWidth:0},children:[(0,t.jsxs)("div",{style:{marginBottom:"24px"},children:[(0,t.jsx)("h2",{style:{fontSize:"24px",fontWeight:"bold",margin:"0 0 4px 0"},children:"Transformed Request"}),(0,t.jsx)("p",{style:{color:"#666",margin:0},children:"How LiteLLM transforms your request for the specified provider."}),(0,t.jsx)("br",{}),(0,t.jsx)("p",{style:{color:"#666",margin:0},className:"text-xs",children:"Note: Sensitive headers are not shown."})]}),(0,t.jsxs)("div",{style:{position:"relative",backgroundColor:"#f5f5f5",borderRadius:"6px",flex:"1 1 auto",display:"flex",flexDirection:"column",overflow:"hidden"},children:[(0,t.jsx)("pre",{style:{padding:"16px",fontFamily:"monospace",fontSize:"14px",margin:0,overflow:"auto",flex:"1 1 auto"},children:c||`curl -X POST \\
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
  }'`}),(0,t.jsx)(s.Button,{type:"text",icon:(0,t.jsx)(i.CopyOutlined,{}),style:{position:"absolute",right:"8px",top:"8px"},size:"small",onClick:()=>{navigator.clipboard.writeText(c||""),a.default.success("Copied to clipboard")}})]})]})]}),(0,t.jsx)("div",{className:"mt-4 text-right w-full",children:(0,t.jsxs)("p",{className:"text-sm text-gray-500",children:["Found an error? File an issue"," ",(0,t.jsx)("a",{href:"https://github.com/BerriAI/litellm/issues",target:"_blank",rel:"noopener noreferrer",children:"here"}),"."]})})]})}])},715288,e=>{"use strict";var t=e.i(843476),r=e.i(662316),s=e.i(135214);e.s(["default",0,()=>{let{accessToken:e}=(0,s.default)();return(0,t.jsx)(r.default,{accessToken:e})}])}]);