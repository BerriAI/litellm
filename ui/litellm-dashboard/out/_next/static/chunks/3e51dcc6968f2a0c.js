(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,976883,737033,e=>{"use strict";var t=e.i(843476),r=e.i(275144),a=e.i(778917),s=e.i(555436),i=e.i(994388),l=e.i(304967),n=e.i(599724),o=e.i(629569),c=e.i(212931),d=e.i(199133),m=e.i(653496),u=e.i(262218),p=e.i(592968),g=e.i(174886),h=e.i(952571),x=e.i(271645),f=e.i(798496),_=e.i(727749),b=e.i(402874),y=e.i(764205),v=e.i(793479),j=e.i(967489),N=e.i(487486),w=e.i(746798),A=e.i(221345),S=e.i(652272);let T="__all__",C=({skills:e,isLoading:r,isAdmin:a,accessToken:i,publicPage:l=!1,onPublishSuccess:n})=>{let[o,c]=(0,x.useState)(""),[d,m]=(0,x.useState)(void 0),[u,p]=(0,x.useState)(null),h=e.length,_=(0,x.useMemo)(()=>[...new Set(e.map(e=>e.domain).filter(Boolean))],[e]),b=(0,x.useMemo)(()=>[...new Set(e.map(e=>e.namespace).filter(Boolean))],[e]),y=(0,x.useMemo)(()=>{let t=e;if(d&&(t=t.filter(e=>(e.domain||"General")===d)),o.trim()){let e=o.toLowerCase();t=t.filter(t=>t.name.toLowerCase().includes(e)||t.description?.toLowerCase().includes(e)||t.domain?.toLowerCase().includes(e)||t.namespace?.toLowerCase().includes(e)||t.keywords?.some(t=>t.toLowerCase().includes(e)))}return t},[e,o,d]);return u?(0,t.jsx)(S.default,{skill:u,onBack:()=>p(null),isAdmin:a,accessToken:i,onPublishClick:n}):r?(0,t.jsx)("div",{className:"text-center py-16 text-muted-foreground",children:"Loading skills..."}):(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{className:"grid grid-cols-3 gap-4",children:[(0,t.jsxs)("div",{className:"border border-border rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-muted-foreground mb-1",children:"Total Skills"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-foreground",children:h})]}),(0,t.jsxs)("div",{className:"border border-border rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-muted-foreground mb-1",children:"Namespaces"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-foreground",children:b.length})]}),(0,t.jsxs)("div",{className:"border border-border rounded-lg p-4",children:[(0,t.jsx)("div",{className:"text-xs text-muted-foreground mb-1",children:"Domains"}),(0,t.jsx)("div",{className:"text-2xl font-semibold text-foreground",children:_.length})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-3",children:[(0,t.jsxs)("h3",{className:"text-sm font-semibold text-foreground",children:["All ",l?"Public ":"","Skills"]}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsxs)(j.Select,{value:d??T,onValueChange:e=>m(e===T?void 0:e),children:[(0,t.jsx)(j.SelectTrigger,{className:"w-40",children:(0,t.jsx)(j.SelectValue,{placeholder:"All Domains"})}),(0,t.jsxs)(j.SelectContent,{children:[(0,t.jsx)(j.SelectItem,{value:T,children:"All Domains"}),_.map(e=>(0,t.jsx)(j.SelectItem,{value:e,children:e},e))]})]}),(0,t.jsxs)("div",{className:"relative w-72",children:[(0,t.jsx)(s.Search,{className:"absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none"}),(0,t.jsx)(v.Input,{placeholder:"Search by name, namespace, or tag…",value:o,onChange:e=>c(e.target.value),className:"pl-8 h-9"})]})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:((e,r,a=!1)=>[{header:"Skill Name",accessorKey:"name",enableSorting:!0,sortingFn:"alphanumeric",cell:({row:a})=>{let s=a.original;return(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("button",{type:"button",className:"font-medium text-sm cursor-pointer text-primary hover:underline bg-transparent border-none p-0",onClick:()=>e(s),children:s.name}),(0,t.jsx)(w.TooltipProvider,{children:(0,t.jsxs)(w.Tooltip,{children:[(0,t.jsx)(w.TooltipTrigger,{asChild:!0,children:(0,t.jsx)("button",{type:"button",onClick:()=>r(s.name),className:"cursor-pointer text-muted-foreground hover:text-primary","aria-label":"Copy skill name",children:(0,t.jsx)(g.Copy,{className:"h-3 w-3"})})}),(0,t.jsx)(w.TooltipContent,{children:"Copy skill name"})]})})]}),s.description&&(0,t.jsx)("p",{className:"text-xs text-muted-foreground line-clamp-1 md:hidden",children:s.description})]})}},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>(0,t.jsx)("p",{className:"text-xs line-clamp-2",children:e.original.description||"-"})},{header:"Category",accessorKey:"category",enableSorting:!0,cell:({row:e})=>{let r=e.original.category;return r?(0,t.jsx)(N.Badge,{className:"bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 text-xs",children:r}):(0,t.jsx)("span",{className:"text-xs text-muted-foreground",children:"-"})}},{header:"Domain",accessorKey:"domain",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("span",{className:"text-xs",children:e.original.domain||"-"})},{header:"Source",accessorKey:"source",enableSorting:!1,cell:({row:e})=>{let r=e.original.source,a=null,s="-";return(r?.source==="github"&&r.repo?(a=`https://github.com/${r.repo}`,s=r.repo):r?.source==="git-subdir"&&r.url?s=(a=r.path?`${r.url}/tree/main/${r.path}`:r.url).replace("https://github.com/",""):r?.source==="url"&&r.url&&(a=r.url,s=r.url.replace(/^https?:\/\//,"")),a)?(0,t.jsxs)("a",{href:a,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 text-xs text-primary hover:underline truncate max-w-[180px]",title:s,children:[(0,t.jsx)("span",{className:"truncate",children:s}),(0,t.jsx)(A.Link,{className:"h-2.5 w-2.5 shrink-0"})]}):(0,t.jsx)("span",{className:"text-xs text-muted-foreground",children:"-"})}},{header:"Status",accessorKey:"enabled",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(N.Badge,{className:e.original.enabled?"bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300 text-xs":"bg-muted text-muted-foreground text-xs",children:e.original.enabled?"Public":"Draft"})}])(e=>p(e),e=>{navigator.clipboard.writeText(e)},l),data:y,isLoading:!1,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-3 text-center",children:(0,t.jsxs)("span",{className:"text-sm text-muted-foreground",children:["Showing ",y.length," of ",h," skill",1!==h?"s":""]})})]})]})};e.s(["default",0,C],737033);var k=e.i(190272),I=e.i(785913),E=e.i(916925);let{TabPane:L}=m.Tabs;e.s(["default",0,({accessToken:e,isEmbedded:v=!1})=>{let j,N,w,A,S,T,O,[M,P]=(0,x.useState)(null),[R,$]=(0,x.useState)(null),[D,z]=(0,x.useState)(null),[B,U]=(0,x.useState)("LiteLLM Gateway"),[H,F]=(0,x.useState)(null),[G,V]=(0,x.useState)(""),[K,W]=(0,x.useState)({}),[q,X]=(0,x.useState)(!0),[Y,J]=(0,x.useState)(!0),[Z,Q]=(0,x.useState)(!0),[ee,et]=(0,x.useState)(""),[er,ea]=(0,x.useState)(""),[es,ei]=(0,x.useState)(""),[el,en]=(0,x.useState)([]),[eo,ec]=(0,x.useState)([]),[ed,em]=(0,x.useState)([]),[eu,ep]=(0,x.useState)([]),[eg,eh]=(0,x.useState)([]),[ex,ef]=(0,x.useState)("I'm alive! ✓"),[e_,eb]=(0,x.useState)(!1),[ey,ev]=(0,x.useState)(!1),[ej,eN]=(0,x.useState)(!1),[ew,eA]=(0,x.useState)(null),[eS,eT]=(0,x.useState)(null),[eC,ek]=(0,x.useState)(null),[eI,eE]=(0,x.useState)({}),[eL,eO]=(0,x.useState)("models"),[eM,eP]=(0,x.useState)([]),[eR,e$]=(0,x.useState)(!1);(0,x.useEffect)(()=>{(async()=>{try{await (0,y.getUiConfig)()}catch(e){console.error("Failed to get UI config:",e)}let e=async()=>{try{X(!0);let e=await (0,y.modelHubPublicModelsCall)();console.log("ModelHubData:",e),P(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public model data",e),ef("Service unavailable")}finally{X(!1)}},t=async()=>{try{J(!0);let e=await (0,y.agentHubPublicModelsCall)();console.log("AgentHubData:",e),$(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public agent data",e)}finally{J(!1)}},r=async()=>{try{Q(!0);let e=await (0,y.mcpHubPublicServersCall)();console.log("MCPHubData:",e),z(Array.isArray(e)?e:[])}catch(e){console.error("There was an error fetching the public MCP server data",e)}finally{Q(!1)}},a=async()=>{try{e$(!0);let e=await (0,y.skillHubPublicCall)();eP(e.plugins??[])}catch(e){console.error("There was an error fetching the public skill data",e)}finally{e$(!1)}};(async()=>{let e=await (0,y.getPublicModelHubInfo)();console.log("Public Model Hub Info:",e),U(e.docs_title),F(e.custom_docs_description),V(e.litellm_version),W(e.useful_links||{})})(),e(),t(),r(),a()})()},[]),(0,x.useEffect)(()=>{},[ee,el,eo,ed]);let eD=(0,x.useMemo)(()=>{if(!M||!Array.isArray(M))return[];let e=M;if(ee.trim()){let t=ee.toLowerCase(),r=t.split(/\s+/),a=M.filter(e=>{let a=e.model_group.toLowerCase();return!!a.includes(t)||r.every(e=>a.includes(e))});a.length>0&&(e=a.sort((e,r)=>{let a=e.model_group.toLowerCase(),s=r.model_group.toLowerCase(),i=1e3*(a===t),l=1e3*(s===t),n=100*!!a.startsWith(t),o=100*!!s.startsWith(t),c=50*!!t.split(/\s+/).every(e=>a.includes(e)),d=50*!!t.split(/\s+/).every(e=>s.includes(e)),m=a.length;return l+o+d+(1e3-s.length)-(i+n+c+(1e3-m))}))}return e.filter(e=>{let t=0===el.length||el.some(t=>e.providers.includes(t)),r=0===eo.length||eo.includes(e.mode||""),a=0===ed.length||Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).some(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");return ed.includes(t)});return t&&r&&a})},[M,ee,el,eo,ed]),ez=(0,x.useMemo)(()=>{if(!R||!Array.isArray(R))return[];let e=R;if(er.trim()){let t=er.toLowerCase(),r=t.split(/\s+/);e=(e=R.filter(e=>{let a=e.name.toLowerCase(),s=e.description.toLowerCase();return!!(a.includes(t)||s.includes(t))||r.every(e=>a.includes(e)||s.includes(e))})).sort((e,r)=>{let a=e.name.toLowerCase(),s=r.name.toLowerCase(),i=1e3*(a===t),l=1e3*(s===t),n=100*!!a.startsWith(t),o=100*!!s.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-s.length)-c})}return e.filter(e=>0===eu.length||e.skills?.some(e=>e.tags?.some(e=>eu.includes(e))))},[R,er,eu]),eB=(0,x.useMemo)(()=>{if(!D||!Array.isArray(D))return[];let e=D;if(es.trim()){let t=es.toLowerCase(),r=t.split(/\s+/);e=(e=D.filter(e=>{let a=e.server_name.toLowerCase(),s=(e.mcp_info?.description||"").toLowerCase();return!!(a.includes(t)||s.includes(t))||r.every(e=>a.includes(e)||s.includes(e))})).sort((e,r)=>{let a=e.server_name.toLowerCase(),s=r.server_name.toLowerCase(),i=1e3*(a===t),l=1e3*(s===t),n=100*!!a.startsWith(t),o=100*!!s.startsWith(t),c=i+n+(1e3-a.length);return l+o+(1e3-s.length)-c})}return e.filter(e=>0===eg.length||eg.includes(e.transport))},[D,es,eg]),eU=e=>{navigator.clipboard.writeText(e),_.default.success("Copied to clipboard!")},eH=e=>e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" "),eF=e=>`$${(1e6*e).toFixed(4)}`,eG=e=>e?e>=1e3?`${(e/1e3).toFixed(0)}K`:e.toString():"N/A";return(0,t.jsx)(r.ThemeProvider,{accessToken:e,children:(0,t.jsxs)("div",{className:v?"w-full":"min-h-screen bg-white",children:[!v&&(0,t.jsx)(b.default,{userID:null,userEmail:null,userRole:null,premiumUser:!1,setProxySettings:eE,proxySettings:eI,accessToken:e||null,isPublicPage:!0,isDarkMode:!1,toggleDarkMode:()=>{}}),(0,t.jsxs)("div",{className:v?"w-full p-6":"w-full px-8 py-12",children:[v&&(0,t.jsx)("div",{className:"mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg",children:(0,t.jsx)("p",{className:"text-sm text-gray-700",children:"These are models, agents, and MCP servers your proxy admin has indicated are available in your company."})}),!v&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"About"}),(0,t.jsx)("p",{className:"text-gray-700 mb-6 text-base leading-relaxed",children:H||"Proxy Server to call 100+ LLMs in the OpenAI format."}),(0,t.jsx)("div",{className:"flex items-center space-x-3 text-sm text-gray-600",children:(0,t.jsxs)("span",{className:"flex items-center",children:[(0,t.jsx)("span",{className:"w-4 h-4 mr-2",children:"🔧"}),"Built with litellm: v",G]})})]}),K&&Object.keys(K).length>0&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Useful Links"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",children:Object.entries(K||{}).map(([e,t])=>({title:e,url:"string"==typeof t?t:t.url,index:"string"==typeof t?0:t.index??0})).sort((e,t)=>e.index-t.index).map(({title:e,url:r})=>(0,t.jsxs)("button",{onClick:()=>window.open(r,"_blank"),className:"flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200",children:[(0,t.jsx)(a.ExternalLink,{className:"w-4 h-4"}),(0,t.jsx)(n.Text,{className:"text-sm font-medium",children:e})]},e))})]}),!v&&(0,t.jsxs)(l.Card,{className:"mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:[(0,t.jsx)(o.Title,{className:"text-2xl font-semibold mb-6 text-gray-900",children:"Health and Endpoint Status"}),(0,t.jsx)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6",children:(0,t.jsxs)(n.Text,{className:"text-green-600 font-medium text-sm",children:["Service status: ",ex]})})]}),(0,t.jsx)(l.Card,{className:"p-8 bg-white border border-gray-200 rounded-lg shadow-sm",children:(0,t.jsxs)(m.Tabs,{activeKey:eL,onChange:eO,size:"large",className:"public-hub-tabs",children:[(0,t.jsxs)(L,{tab:"Model Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Models"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Models:"}),(0,t.jsx)(p.Tooltip,{title:"Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'",placement:"top",children:(0,t.jsx)(h.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(s.Search,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search model names... (smart search enabled)",value:ee,onChange:e=>et(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Provider:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:el,onChange:e=>en(e),placeholder:"Select providers",className:"w-full",size:"large",allowClear:!0,optionRender:e=>{let{logo:r}=(0,E.getProviderLogoAndName)(e.value);return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[r&&(0,t.jsx)("img",{src:r,alt:e.label,className:"w-5 h-5 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e.label})]})},children:M&&Array.isArray(M)&&(j=new Set,M.forEach(e=>{(e.providers??[]).forEach(e=>j.add(e))}),Array.from(j)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Mode:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eo,onChange:e=>ec(e),placeholder:"Select modes",className:"w-full",size:"large",allowClear:!0,children:M&&Array.isArray(M)&&(N=new Set,M.forEach(e=>{e.mode&&N.add(e.mode)}),Array.from(N)).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Features:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:ed,onChange:e=>em(e),placeholder:"Select features",className:"w-full",size:"large",allowClear:!0,children:M&&Array.isArray(M)&&(w=new Set,M.forEach(e=>{Object.entries(e).filter(([e,t])=>e.startsWith("supports_")&&!0===t).forEach(([e])=>{let t=e.replace(/^supports_/,"").split("_").map(e=>e.charAt(0).toUpperCase()+e.slice(1)).join(" ");w.add(t)})}),Array.from(w).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Model Name",accessorKey:"model_group",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(p.Tooltip,{title:e.original.model_group,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eA(e.original),eb(!0)},children:e.original.model_group})})}),size:150},{header:"Providers",accessorKey:"providers",enableSorting:!0,cell:({row:e})=>{let r=e.original.providers??[];return(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:r.map(e=>{let{logo:r}=(0,E.getProviderLogoAndName)(e);return(0,t.jsxs)("div",{className:"flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs",children:[r&&(0,t.jsx)("img",{src:r,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]},e)})})},size:120},{header:"Mode",accessorKey:"mode",enableSorting:!0,cell:({row:e})=>{let r=e.original.mode;return(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:(e=>{switch(e?.toLowerCase()){case"chat":return"💬";case"rerank":return"🔄";case"embedding":return"📄";default:return"🤖"}})(r||"")}),(0,t.jsx)(n.Text,{children:r||"Chat"})]})},size:100},{header:"Max Input",accessorKey:"max_input_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eG(e.original.max_input_tokens)}),size:100,meta:{className:"text-center"}},{header:"Max Output",accessorKey:"max_output_tokens",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-center",children:eG(e.original.max_output_tokens)}),size:100,meta:{className:"text-center"}},{header:"Input $/1M",accessorKey:"input_cost_per_token",enableSorting:!0,cell:({row:e})=>{let r=e.original.input_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:r?eF(r):"Free"})},size:100,meta:{className:"text-center"}},{header:"Output $/1M",accessorKey:"output_cost_per_token",enableSorting:!0,cell:({row:e})=>{let r=e.original.output_cost_per_token;return(0,t.jsx)(n.Text,{className:"text-center",children:r?eF(r):"Free"})},size:100,meta:{className:"text-center"}},{header:"Features",accessorKey:"supports_vision",enableSorting:!1,cell:({row:e})=>{let r=Object.entries(e.original).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>eH(e));return 0===r.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===r.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:r[0]})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs",children:r[0]}),(0,t.jsx)(p.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Features:"}),r.map((e,r)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e]},r))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",r.length-1]})})]})},size:120},{header:"Health Status",accessorKey:"health_status",enableSorting:!0,cell:({row:e})=>{let r=e.original,a="healthy"===r.health_status?"green":"unhealthy"===r.health_status?"red":"default",s=r.health_response_time?`Response Time: ${Number(r.health_response_time).toFixed(2)}ms`:"N/A",i=r.health_checked_at?`Last Checked: ${new Date(r.health_checked_at).toLocaleString()}`:"N/A";return(0,t.jsx)(p.Tooltip,{title:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("div",{children:s}),(0,t.jsx)("div",{children:i})]}),children:(0,t.jsx)(u.Tag,{color:a,children:(0,t.jsx)("span",{className:"capitalize",children:r.health_status??"Unknown"})},r.model_group)})},size:100},{header:"Limits",accessorKey:"rpm",enableSorting:!0,cell:({row:e})=>{var r,a;let s,i=e.original;return(0,t.jsx)(n.Text,{className:"text-xs text-gray-600",children:(r=i.rpm,a=i.tpm,s=[],r&&s.push(`RPM: ${r.toLocaleString()}`),a&&s.push(`TPM: ${a.toLocaleString()}`),s.length>0?s.join(", "):"N/A")})},size:150}],data:eD,isLoading:q,defaultSorting:[{id:"model_group",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eD.length," of ",M?.length||0," models"]})})]},"models"),R&&Array.isArray(R)&&R.length>0&&(0,t.jsxs)(L,{tab:"Agent Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available Agents"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search Agents:"}),(0,t.jsx)(p.Tooltip,{title:"Search agents by name or description",placement:"top",children:(0,t.jsx)(h.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(s.Search,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search agent names or descriptions...",value:er,onChange:e=>ea(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Skills:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eu,onChange:e=>ep(e),placeholder:"Select skills",className:"w-full",size:"large",allowClear:!0,children:R&&Array.isArray(R)&&(A=new Set,R.forEach(e=>{e.skills?.forEach(e=>{e.tags?.forEach(e=>A.add(e))})}),Array.from(A).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Agent Name",accessorKey:"name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(p.Tooltip,{title:e.original.name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{eT(e.original),ev(!0)},children:e.original.name})})}),size:150},{header:"Description",accessorKey:"description",enableSorting:!1,cell:({row:e})=>{let r=e.original.description??"",a=r.length>80?r.substring(0,80)+"...":r;return(0,t.jsx)(p.Tooltip,{title:r,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"Version",accessorKey:"version",enableSorting:!0,cell:({row:e})=>(0,t.jsx)(n.Text,{className:"text-sm",children:e.original.version}),size:80},{header:"Provider",accessorKey:"provider",enableSorting:!1,cell:({row:e})=>{let r=e.original.provider;return r?(0,t.jsx)("div",{className:"text-sm",children:(0,t.jsx)(n.Text,{className:"font-medium",children:r.organization})}):(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"})},size:120},{header:"Skills",accessorKey:"skills",enableSorting:!1,cell:({row:e})=>{let r=e.original.skills||[];return 0===r.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):1===r.length?(0,t.jsx)("div",{className:"h-6 flex items-center",children:(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:r[0].name})}):(0,t.jsxs)("div",{className:"h-6 flex items-center space-x-1",children:[(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:r[0].name}),(0,t.jsx)(p.Tooltip,{title:(0,t.jsxs)("div",{className:"space-y-1",children:[(0,t.jsx)("div",{className:"font-medium",children:"All Skills:"}),r.map((e,r)=>(0,t.jsxs)("div",{className:"text-xs",children:["• ",e.name]},r))]}),trigger:"click",placement:"topLeft",children:(0,t.jsxs)("span",{className:"text-xs text-purple-600 cursor-pointer hover:text-purple-800 hover:underline",onClick:e=>e.stopPropagation(),children:["+",r.length-1]})})]})},size:150},{header:"Capabilities",accessorKey:"capabilities",enableSorting:!1,cell:({row:e})=>{let r=Object.entries(e.original.capabilities||{}).filter(([e,t])=>!0===t).map(([e])=>e);return 0===r.length?(0,t.jsx)(n.Text,{className:"text-gray-400",children:"-"}):(0,t.jsx)("div",{className:"flex flex-wrap gap-1",children:r.map(e=>(0,t.jsx)(u.Tag,{color:"green",className:"text-xs capitalize",children:e},e))})},size:150}],data:ez,isLoading:Y,defaultSorting:[{id:"name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",ez.length," of ",R?.length||0," agents"]})})]},"agents"),D&&Array.isArray(D)&&D.length>0&&(0,t.jsxs)(L,{tab:"MCP Hub",children:[(0,t.jsx)("div",{className:"flex justify-between items-center mb-8",children:(0,t.jsx)(o.Title,{className:"text-2xl font-semibold text-gray-900",children:"Available MCP Servers"})}),(0,t.jsxs)("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200",children:[(0,t.jsxs)("div",{children:[(0,t.jsxs)("div",{className:"flex items-center space-x-2 mb-3",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium text-gray-700",children:"Search MCP Servers:"}),(0,t.jsx)(p.Tooltip,{title:"Search MCP servers by name or description",placement:"top",children:(0,t.jsx)(h.Info,{className:"w-4 h-4 text-gray-400 cursor-help"})})]}),(0,t.jsxs)("div",{className:"relative",children:[(0,t.jsx)(s.Search,{className:"w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2"}),(0,t.jsx)("input",{type:"text",placeholder:"Search MCP server names or descriptions...",value:es,onChange:e=>ei(e.target.value),className:"border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-3 text-gray-700",children:"Transport:"}),(0,t.jsx)(d.Select,{mode:"multiple",value:eg,onChange:e=>eh(e),placeholder:"Select transport types",className:"w-full",size:"large",allowClear:!0,children:D&&Array.isArray(D)&&(S=new Set,D.forEach(e=>{e.transport&&S.add(e.transport)}),Array.from(S).sort()).map(e=>(0,t.jsx)(d.Select.Option,{value:e,children:e},e))})]})]}),(0,t.jsx)(f.ModelDataTable,{columns:[{header:"Server Name",accessorKey:"server_name",enableSorting:!0,cell:({row:e})=>(0,t.jsx)("div",{className:"overflow-hidden",children:(0,t.jsx)(p.Tooltip,{title:e.original.server_name,children:(0,t.jsx)(i.Button,{size:"xs",variant:"light",className:"font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left",onClick:()=>{ek(e.original),eN(!0)},children:e.original.server_name})})}),size:150},{header:"Description",accessorKey:"mcp_info.description",enableSorting:!1,cell:({row:e})=>{let r=String(e.original.mcp_info?.description??"-"),a=r.length>80?r.substring(0,80)+"...":r;return(0,t.jsx)(p.Tooltip,{title:r,children:(0,t.jsx)(n.Text,{className:"text-sm text-gray-700",children:a})})},size:250},{header:"URL",accessorKey:"url",enableSorting:!1,cell:({row:e})=>{let r=e.original.url??"",a=r.length>40?r.substring(0,40)+"...":r;return(0,t.jsx)(p.Tooltip,{title:r,children:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)(n.Text,{className:"text-xs font-mono",children:a}),(0,t.jsx)(g.Copy,{onClick:()=>eU(r),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-3 h-3"})]})})},size:200},{header:"Transport",accessorKey:"transport",enableSorting:!0,cell:({row:e})=>{let r=e.original.transport;return(0,t.jsx)(u.Tag,{color:"blue",className:"text-xs uppercase",children:r})},size:100},{header:"Auth Type",accessorKey:"auth_type",enableSorting:!0,cell:({row:e})=>{let r=e.original.auth_type;return(0,t.jsx)(u.Tag,{color:"none"===r?"gray":"green",className:"text-xs capitalize",children:r})},size:100}],data:eB,isLoading:Z,defaultSorting:[{id:"server_name",desc:!1}]}),(0,t.jsx)("div",{className:"mt-8 text-center",children:(0,t.jsxs)(n.Text,{className:"text-sm text-gray-600",children:["Showing ",eB.length," of ",D?.length||0," MCP servers"]})})]},"mcp"),(0,t.jsx)(L,{tab:"Skill Hub",children:(0,t.jsx)(C,{skills:eM,isLoading:eR,publicPage:!0})},"skills")]})})]}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:ew?.model_group||"Model Details"}),ew&&(0,t.jsx)(p.Tooltip,{title:"Copy model name",children:(0,t.jsx)(g.Copy,{onClick:()=>eU(ew.model_group),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:e_,footer:null,onOk:()=>{eb(!1),eA(null)},onCancel:()=>{eb(!1),eA(null)},children:ew&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Model Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Model Name:"}),(0,t.jsx)(n.Text,{children:ew.model_group})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Mode:"}),(0,t.jsx)(n.Text,{children:ew.mode||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Providers:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(ew.providers??[]).map(e=>{let{logo:r}=(0,E.getProviderLogoAndName)(e);return(0,t.jsx)(u.Tag,{color:"blue",children:(0,t.jsxs)("div",{className:"flex items-center space-x-1",children:[r&&(0,t.jsx)("img",{src:r,alt:e,className:"w-3 h-3 flex-shrink-0 object-contain",onError:e=>{e.target.style.display="none"}}),(0,t.jsx)("span",{className:"capitalize",children:e})]})},e)})})]})]}),ew.model_group.includes("*")&&(0,t.jsx)("div",{className:"bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4",children:(0,t.jsxs)("div",{className:"flex items-start space-x-2",children:[(0,t.jsx)(h.Info,{className:"w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0"}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-blue-900 mb-2",children:"Wildcard Routing"}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800 mb-2",children:["This model uses wildcard routing. You can pass any value where you see the"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:"*"})," symbol."]}),(0,t.jsxs)(n.Text,{className:"text-sm text-blue-800",children:["For example, with"," ",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:ew.model_group}),", you can use any string (",(0,t.jsx)("code",{className:"bg-blue-100 px-1 py-0.5 rounded text-xs",children:ew.model_group.replaceAll("*","my-custom-value")}),") that matches this pattern."]})]})]})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Token & Cost Information"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Input Tokens:"}),(0,t.jsx)(n.Text,{children:ew.max_input_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Max Output Tokens:"}),(0,t.jsx)(n.Text,{children:ew.max_output_tokens?.toLocaleString()||"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:ew.input_cost_per_token?eF(ew.input_cost_per_token):"Not specified"})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Cost per 1M Tokens:"}),(0,t.jsx)(n.Text,{children:ew.output_cost_per_token?eF(ew.output_cost_per_token):"Not specified"})]})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:(T=Object.entries(ew).filter(([e,t])=>e.startsWith("supports_")&&!0===t).map(([e])=>e),O=["green","blue","purple","orange","red","yellow"],0===T.length?(0,t.jsx)(n.Text,{className:"text-gray-500",children:"No special capabilities listed"}):T.map((e,r)=>(0,t.jsx)(u.Tag,{color:O[r%O.length],children:eH(e)},e)))})]}),(ew.tpm||ew.rpm)&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Rate Limits"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[ew.tpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Tokens per Minute:"}),(0,t.jsx)(n.Text,{children:ew.tpm.toLocaleString()})]}),ew.rpm&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Requests per Minute:"}),(0,t.jsx)(n.Text,{children:ew.rpm.toLocaleString()})]})]})]}),ew.supported_openai_params&&ew.supported_openai_params.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Supported OpenAI Parameters"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:ew.supported_openai_params.map(e=>(0,t.jsx)(u.Tag,{color:"green",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:(0,k.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,I.getEndpointType)(ew.mode||"chat"),selectedModel:ew.model_group,selectedSdk:"openai"})})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eU((0,k.generateCodeSnippet)({apiKeySource:"custom",accessToken:null,apiKey:"your_api_key",inputMessage:"Hello, how are you?",chatHistory:[{role:"user",content:"Hello, how are you?",isImage:!1}],selectedTags:[],selectedVectorStores:[],selectedGuardrails:[],selectedPolicies:[],selectedMCPServers:[],endpointType:(0,I.getEndpointType)(ew.mode||"chat"),selectedModel:ew.model_group,selectedSdk:"openai"}))},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eS?.name||"Agent Details"}),eS&&(0,t.jsx)(p.Tooltip,{title:"Copy agent name",children:(0,t.jsx)(g.Copy,{onClick:()=>eU(eS.name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ey,footer:null,onOk:()=>{ev(!1),eT(null)},onCancel:()=>{ev(!1),eT(null)},children:eS&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Agent Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Name:"}),(0,t.jsx)(n.Text,{children:eS.name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Version:"}),(0,t.jsx)(n.Text,{children:eS.version})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:eS.description})]}),eS.url&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsx)("a",{href:eS.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all",children:eS.url})]})]})]}),eS.capabilities&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Capabilities"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-2",children:Object.entries(eS.capabilities).filter(([e,t])=>!0===t).map(([e])=>(0,t.jsx)(u.Tag,{color:"green",className:"capitalize",children:e},e))})]}),eS.skills&&eS.skills.length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Skills"}),(0,t.jsx)("div",{className:"space-y-4",children:eS.skills.map((e,r)=>(0,t.jsxs)("div",{className:"border border-gray-200 rounded-lg p-4",children:[(0,t.jsx)("div",{className:"flex items-start justify-between mb-2",children:(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium text-base",children:e.name}),(0,t.jsx)(n.Text,{className:"text-sm text-gray-600",children:e.description})]})}),e.tags&&e.tags.length>0&&(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-2",children:e.tags.map(e=>(0,t.jsx)(u.Tag,{color:"purple",className:"text-xs",children:e},e))})]},r))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Input/Output Modes"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Input Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eS.defaultInputModes??[]).map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Output Modes:"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1 mt-1",children:(eS.defaultOutputModes??[]).map(e=>(0,t.jsx)(u.Tag,{color:"blue",children:e},e))})]})]})]}),eS.documentationUrl&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Documentation"}),(0,t.jsxs)("a",{href:eS.documentationUrl,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 flex items-center space-x-2",children:[(0,t.jsx)(a.ExternalLink,{className:"w-4 h-4"}),(0,t.jsx)("span",{children:"View Documentation"})]})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example (A2A Protocol)"}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 1: Retrieve Agent Card"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`base_url = '${eS.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eU(`from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

base_url = '${eS.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-sm font-medium mb-2 text-gray-700",children:"Step 2: Call the Agent"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-xs",children:`client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eU(`client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})]})}),(0,t.jsx)(c.Modal,{title:(0,t.jsxs)("div",{className:"flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eC?.server_name||"MCP Server Details"}),eC&&(0,t.jsx)(p.Tooltip,{title:"Copy server name",children:(0,t.jsx)(g.Copy,{onClick:()=>eU(eC.server_name),className:"cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"})})]}),width:1e3,open:ej,footer:null,onOk:()=>{eN(!1),ek(null)},onCancel:()=>{eN(!1),ek(null)},children:eC&&(0,t.jsxs)("div",{className:"space-y-6",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Server Overview"}),(0,t.jsxs)("div",{className:"grid grid-cols-2 gap-4 mb-4",children:[(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Server Name:"}),(0,t.jsx)(n.Text,{children:eC.server_name})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Transport:"}),(0,t.jsx)(u.Tag,{color:"blue",children:eC.transport})]}),eC.alias&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Alias:"}),(0,t.jsx)(n.Text,{children:eC.alias})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Auth Type:"}),(0,t.jsx)(u.Tag,{color:"none"===eC.auth_type?"gray":"green",children:eC.auth_type})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"Description:"}),(0,t.jsx)(n.Text,{children:eC.mcp_info?.description||"-"})]}),(0,t.jsxs)("div",{className:"col-span-2",children:[(0,t.jsx)(n.Text,{className:"font-medium",children:"URL:"}),(0,t.jsxs)("a",{href:eC.url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-600 hover:text-blue-800 text-sm break-all flex items-center space-x-2",children:[(0,t.jsx)("span",{children:eC.url}),(0,t.jsx)(a.ExternalLink,{className:"w-4 h-4"})]})]})]})]}),eC.mcp_info&&Object.keys(eC.mcp_info).length>0&&(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Additional Information"}),(0,t.jsx)("div",{className:"bg-gray-50 p-4 rounded-lg",children:(0,t.jsx)("pre",{className:"text-xs overflow-x-auto",children:JSON.stringify(eC.mcp_info,null,2)})})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)(n.Text,{className:"text-lg font-semibold mb-4",children:"Usage Example"}),(0,t.jsx)("div",{className:"bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto",children:(0,t.jsx)("pre",{className:"text-sm",children:`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eC.server_name}": {
            "url": "http://localhost:4000/${eC.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call a tool
        response = await client.call_tool(
            name="tool_name", 
            arguments={"arg": "value"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())`})}),(0,t.jsx)("div",{className:"mt-2 text-right",children:(0,t.jsx)("button",{onClick:()=>{eU(`# Using MCP Server with Python FastMCP

from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${eC.server_name}": {
            "url": "http://localhost:4000/${eC.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call a tool
        response = await client.call_tool(
            name="tool_name", 
            arguments={"arg": "value"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())`)},className:"text-sm text-blue-600 hover:text-blue-800 cursor-pointer",children:"Copy to clipboard"})})]})]})})]})})}],976883)},62478,e=>{"use strict";var t=e.i(764205);let r=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,r])},292270,e=>{"use strict";let t=(0,e.i(475254).default)("log-out",[["path",{d:"m16 17 5-5-5-5",key:"1bji2h"}],["path",{d:"M21 12H9",key:"dn1m92"}],["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}]]);e.s(["LogOut",()=>t],292270)},818581,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"useMergedRef",{enumerable:!0,get:function(){return s}});let a=e.r(271645);function s(e,t){let r=(0,a.useRef)(null),s=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=r.current;e&&(r.current=null,e());let t=s.current;t&&(s.current=null,t())}else e&&(r.current=i(e,a)),t&&(s.current=i(t,a))},[e,t])}function i(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let r=e(t);return"function"==typeof r?r:()=>e(null)}}("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},190272,785913,e=>{"use strict";var t,r,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),s=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>s,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:a,apiKey:i,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:m,selectedMCPServers:u,mcpServers:p,mcpServerToolRestrictions:g,selectedVoice:h,endpointType:x,selectedModel:f,selectedSdk:_,proxySettings:b}=e,y="session"===r?a:i,v=window.location.origin,j=b?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?v=j:b?.PROXY_BASE_URL&&(v=b.PROXY_BASE_URL);let N=l||"Your prompt here",w=N.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),A=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};o.length>0&&(S.tags=o),c.length>0&&(S.vector_stores=c),d.length>0&&(S.guardrails=d),m.length>0&&(S.policies=m);let T=f||"your-model-name",C="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${y||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(x){case s.CHAT:{let e=Object.keys(S).length>0,r="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let a=A.length>0?A:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${T}",
    messages=${JSON.stringify(a,null,4)}${r}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${T}",
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
#     ]${r}
# )
# print(response_with_file)
`;break}case s.RESPONSES:{let e=Object.keys(S).length>0,r="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let a=A.length>0?A:[{role:"user",content:N}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${T}",
    input=${JSON.stringify(a,null,4)}${r}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${T}",
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
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case s.IMAGE:t="azure"===_?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${T}",
	prompt="${l}",
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
	model="${T}",
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
`;break;case s.IMAGE_EDITS:t="azure"===_?`
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
	model="${T}",
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
	model="${T}",
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
`;break;case s.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${l||"Your string here"}",
	model="${T}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case s.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${T}",
	file=audio_file${l?`,
	prompt="${l.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case s.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${T}",
	input="${l||"Your text to convert to speech here"}",
	voice="${h}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${T}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${t}`}],190272)},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),a=e.i(871689),s=e.i(174886),i=e.i(643531),l=e.i(221345);let n=e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`;e.s(["formatInstallCommand",0,n,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:o})=>{let c,[d,m]=(0,r.useState)("overview"),[u,p]=(0,r.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),p(t),setTimeout(()=>p(null),2e3)},h="github"===(c=e.source).source&&c.repo?`https://github.com/${c.repo}`:"git-subdir"===c.source&&c.url?c.path?`${c.url}/tree/main/${c.path}`:c.url:"url"===c.source&&c.url?c.url:null,x=n(e),f=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:o,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(a.ArrowLeft,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>m(e.key),style:{padding:"12px 20px",fontSize:14,color:d===e.key?"#1a73e8":"#5f6368",borderBottom:d===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:d===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===d&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:f.map((e,r)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),h&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:h,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[h.replace("https://",""),(0,t.jsx)(l.Link,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(x,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===u?(0,t.jsx)(i.Check,{}):(0,t.jsx)(s.Copy,{}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:x})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>m("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{g(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===u?(0,t.jsx)(i.Check,{}):(0,t.jsx)(s.Copy,{}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},221345,e=>{"use strict";let t=(0,e.i(475254).default)("link",[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]]);e.s(["Link",()=>t],221345)},115571,e=>{"use strict";let t="local-storage-change";function r(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function a(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function s(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function i(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>r,"getLocalStorageItem",()=>a,"removeLocalStorageItem",()=>i,"setLocalStorageItem",()=>s])},916925,e=>{"use strict";var t,r=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t);let a={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference"},s="../ui/assets/logos/",i={"A2A Agent":`${s}a2a_agent.png`,Ai21:`${s}ai21.svg`,"Ai21 Chat":`${s}ai21.svg`,"AI/ML API":`${s}aiml_api.svg`,"Aiohttp Openai":`${s}openai_small.svg`,Anthropic:`${s}anthropic.svg`,"Anthropic Text":`${s}anthropic.svg`,AssemblyAI:`${s}assemblyai_small.png`,Azure:`${s}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${s}microsoft_azure.svg`,"Azure Text":`${s}microsoft_azure.svg`,Baseten:`${s}baseten.svg`,"Amazon Bedrock":`${s}bedrock.svg`,"Amazon Bedrock Mantle":`${s}bedrock.svg`,"AWS SageMaker":`${s}bedrock.svg`,Cerebras:`${s}cerebras.svg`,Cloudflare:`${s}cloudflare.svg`,Codestral:`${s}mistral.svg`,Cohere:`${s}cohere.svg`,"Cohere Chat":`${s}cohere.svg`,Cometapi:`${s}cometapi.svg`,Cursor:`${s}cursor.svg`,"Databricks (Qwen API)":`${s}databricks.svg`,Dashscope:`${s}dashscope.svg`,Deepseek:`${s}deepseek.svg`,Deepgram:`${s}deepgram.png`,DeepInfra:`${s}deepinfra.png`,ElevenLabs:`${s}elevenlabs.png`,"Fal AI":`${s}fal_ai.jpg`,"Featherless Ai":`${s}featherless.svg`,"Fireworks AI":`${s}fireworks.svg`,Friendliai:`${s}friendli.svg`,"Github Copilot":`${s}github_copilot.svg`,"Google AI Studio":`${s}google.svg`,GradientAI:`${s}gradientai.svg`,Groq:`${s}groq.svg`,vllm:`${s}vllm.png`,Huggingface:`${s}huggingface.svg`,Hyperbolic:`${s}hyperbolic.svg`,Infinity:`${s}infinity.png`,"Jina AI":`${s}jina.png`,"Lambda Ai":`${s}lambda.svg`,"Lm Studio":`${s}lmstudio.svg`,"Meta Llama":`${s}meta_llama.svg`,MiniMax:`${s}minimax.svg`,"Mistral AI":`${s}mistral.svg`,Moonshot:`${s}moonshot.svg`,Morph:`${s}morph.svg`,Nebius:`${s}nebius.svg`,Novita:`${s}novita.svg`,"Nvidia Nim":`${s}nvidia_nim.svg`,Ollama:`${s}ollama.svg`,"Ollama Chat":`${s}ollama.svg`,Oobabooga:`${s}openai_small.svg`,OpenAI:`${s}openai_small.svg`,"Openai Like":`${s}openai_small.svg`,"OpenAI Text Completion":`${s}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${s}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${s}openai_small.svg`,Openrouter:`${s}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${s}oracle.svg`,Perplexity:`${s}perplexity-ai.svg`,Recraft:`${s}recraft.svg`,Replicate:`${s}replicate.svg`,RunwayML:`${s}runwayml.png`,Sagemaker:`${s}bedrock.svg`,Sambanova:`${s}sambanova.svg`,"SAP Generative AI Hub":`${s}sap.png`,Snowflake:`${s}snowflake.svg`,"Text-Completion-Codestral":`${s}mistral.svg`,TogetherAI:`${s}togetherai.svg`,Topaz:`${s}topaz.svg`,Triton:`${s}nvidia_triton.png`,V0:`${s}v0.svg`,"Vercel Ai Gateway":`${s}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${s}google.svg`,"Vertex Ai Beta":`${s}google.svg`,Vllm:`${s}vllm.png`,VolcEngine:`${s}volcengine.png`,"Voyage AI":`${s}voyage.webp`,Watsonx:`${s}watsonx.svg`,"Watsonx Text":`${s}watsonx.svg`,xAI:`${s}xai.svg`,Xinference:`${s}xinference.svg`};e.s(["Providers",()=>r,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:i[e],displayName:e}}let t=Object.keys(a).find(t=>a[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let s=r[t];return{logo:i[s],displayName:s}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let r=a[e];console.log(`Provider mapped to: ${r}`);let s=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let a=t.litellm_provider;(a===r||"string"==typeof a&&a.includes(r))&&s.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&s.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&s.push(e)}))),s},"providerLogoMap",0,i,"provider_map",0,a])},798496,e=>{"use strict";var t=e.i(843476),r=e.i(152990),a=e.i(682830),s=e.i(271645),i=e.i(784774),l=e.i(899602),n=e.i(664659),o=e.i(655900);function c({data:e=[],columns:c,isLoading:d=!1,defaultSorting:m=[],pagination:u,onPaginationChange:p,enablePagination:g=!1,onRowClick:h}){let[x,f]=s.default.useState(m),[_]=s.default.useState("onChange"),[b,y]=s.default.useState({}),[v,j]=s.default.useState({}),N=(0,r.useReactTable)({data:e,columns:c,state:{sorting:x,columnSizing:b,columnVisibility:v,...g&&u?{pagination:u}:{}},columnResizeMode:_,onSortingChange:f,onColumnSizingChange:y,onColumnVisibilityChange:j,...g&&p?{onPaginationChange:p}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),...g?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(i.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:N.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(i.TableHeader,{children:N.getHeaderGroups().map(e=>(0,t.jsx)(i.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(i.TableHead,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,r.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(o.ChevronUp,{className:"h-4 w-4 text-primary"}),desc:(0,t.jsx)(n.ChevronDown,{className:"h-4 w-4 text-primary"})})[e.column.getIsSorted()]:(0,t.jsx)(l.ArrowUpDown,{className:"h-4 w-4 text-muted-foreground"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-primary":"hover:bg-primary/20"}`})]},e.id))},e.id))}),(0,t.jsx)(i.TableBody,{children:d?(0,t.jsx)(i.TableRow,{children:(0,t.jsx)(i.TableCell,{colSpan:c.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-muted-foreground",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):N.getRowModel().rows.length>0?N.getRowModel().rows.map(e=>(0,t.jsx)(i.TableRow,{onClick:()=>h?.(e.original),className:h?"cursor-pointer hover:bg-muted":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(i.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-background shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,r.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(i.TableRow,{children:(0,t.jsx)(i.TableCell,{colSpan:c.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-muted-foreground",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>c])},972518,e=>{"use strict";let t=(0,e.i(475254).default)("panel-left-close",[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M9 3v18",key:"fh3hqa"}],["path",{d:"m16 15-3-3 3-3",key:"14y99z"}]]);e.s(["PanelLeftClose",()=>t],972518)},371401,e=>{"use strict";var t=e.i(115571),r=e.i(271645);function a(e){let r=t=>{"disableUsageIndicator"===t.key&&e()},a=t=>{let{key:r}=t.detail;"disableUsageIndicator"===r&&e()};return window.addEventListener("storage",r),window.addEventListener(t.LOCAL_STORAGE_EVENT,a),()=>{window.removeEventListener("storage",r),window.removeEventListener(t.LOCAL_STORAGE_EVENT,a)}}function s(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}function i(){return(0,r.useSyncExternalStore)(a,s)}e.s(["useDisableUsageIndicator",()=>i])},283713,e=>{"use strict";var t=e.i(271645),r=e.i(764205),a=e.i(612256);let s="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,a.useUIConfig)(),i=e?.is_control_plane??!1,l=e?.workers??[],[n,o]=(0,t.useState)(()=>localStorage.getItem(s));(0,t.useEffect)(()=>{if(!n||0===l.length)return;let e=l.find(e=>e.worker_id===n);e&&(0,r.switchToWorkerUrl)(e.url)},[n,l]);let c=l.find(e=>e.worker_id===n)??null,d=(0,t.useCallback)(e=>{let t=l.find(t=>t.worker_id===e);t&&(o(e),localStorage.setItem(s,e),(0,r.switchToWorkerUrl)(t.url))},[l]);return{isControlPlane:i,workers:l,selectedWorkerId:n,selectedWorker:c,selectWorker:d,disconnectFromWorker:(0,t.useCallback)(()=>{o(null),localStorage.removeItem(s),(0,r.switchToWorkerUrl)(null)},[])}}])},275144,e=>{"use strict";var t=e.i(843476),r=e.i(271645),a=e.i(764205);let s=(0,r.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:i})=>{let[l,n]=(0,r.useState)(null),[o,c]=(0,r.useState)(null);return(0,r.useEffect)(()=>{(async()=>{try{let e=(0,a.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",r=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(r.ok){let e=await r.json();e.values?.logo_url&&n(e.values.logo_url),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,r.useEffect)(()=>{if(o){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=o});else{let e=document.createElement("link");e.rel="icon",e.href=o,document.head.appendChild(e)}}},[o]),(0,t.jsx)(s.Provider,{value:{logoUrl:l,setLogoUrl:n,faviconUrl:o,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,r.useContext)(s);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},998183,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var a={assign:function(){return o},searchParamsToUrlQuery:function(){return i},urlQueryToSearchParams:function(){return n}};for(var s in a)Object.defineProperty(r,s,{enumerable:!0,get:a[s]});function i(e){let t={};for(let[r,a]of e.entries()){let e=t[r];void 0===e?t[r]=a:Array.isArray(e)?e.push(a):t[r]=[e,a]}return t}function l(e){return"string"==typeof e?e:("number"!=typeof e||isNaN(e))&&"boolean"!=typeof e?"":String(e)}function n(e){let t=new URLSearchParams;for(let[r,a]of Object.entries(e))if(Array.isArray(a))for(let e of a)t.append(r,l(e));else t.set(r,l(a));return t}function o(e,...t){for(let r of t){for(let t of r.keys())e.delete(t);for(let[t,a]of r.entries())e.append(t,a)}return e}},195057,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var a={formatUrl:function(){return n},formatWithValidation:function(){return c},urlObjectKeys:function(){return o}};for(var s in a)Object.defineProperty(r,s,{enumerable:!0,get:a[s]});let i=e.r(151836)._(e.r(998183)),l=/https?|ftp|gopher|file/;function n(e){let{auth:t,hostname:r}=e,a=e.protocol||"",s=e.pathname||"",n=e.hash||"",o=e.query||"",c=!1;t=t?encodeURIComponent(t).replace(/%3A/i,":")+"@":"",e.host?c=t+e.host:r&&(c=t+(~r.indexOf(":")?`[${r}]`:r),e.port&&(c+=":"+e.port)),o&&"object"==typeof o&&(o=String(i.urlQueryToSearchParams(o)));let d=e.search||o&&`?${o}`||"";return a&&!a.endsWith(":")&&(a+=":"),e.slashes||(!a||l.test(a))&&!1!==c?(c="//"+(c||""),s&&"/"!==s[0]&&(s="/"+s)):c||(c=""),n&&"#"!==n[0]&&(n="#"+n),d&&"?"!==d[0]&&(d="?"+d),s=s.replace(/[?#]/g,encodeURIComponent),d=d.replace("#","%23"),`${a}${c}${s}${d}${n}`}let o=["auth","hash","host","hostname","href","path","pathname","port","protocol","query","search","slashes"];function c(e){return n(e)}},718967,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var a={DecodeError:function(){return f},MiddlewareNotFoundError:function(){return v},MissingStaticPage:function(){return y},NormalizeError:function(){return _},PageNotFoundError:function(){return b},SP:function(){return h},ST:function(){return x},WEB_VITALS:function(){return i},execOnce:function(){return l},getDisplayName:function(){return m},getLocationOrigin:function(){return c},getURL:function(){return d},isAbsoluteUrl:function(){return o},isResSent:function(){return u},loadGetInitialProps:function(){return g},normalizeRepeatedSlashes:function(){return p},stringifyError:function(){return j}};for(var s in a)Object.defineProperty(r,s,{enumerable:!0,get:a[s]});let i=["CLS","FCP","FID","INP","LCP","TTFB"];function l(e){let t,r=!1;return(...a)=>(r||(r=!0,t=e(...a)),t)}let n=/^[a-zA-Z][a-zA-Z\d+\-.]*?:/,o=e=>n.test(e);function c(){let{protocol:e,hostname:t,port:r}=window.location;return`${e}//${t}${r?":"+r:""}`}function d(){let{href:e}=window.location,t=c();return e.substring(t.length)}function m(e){return"string"==typeof e?e:e.displayName||e.name||"Unknown"}function u(e){return e.finished||e.headersSent}function p(e){let t=e.split("?");return t[0].replace(/\\/g,"/").replace(/\/\/+/g,"/")+(t[1]?`?${t.slice(1).join("?")}`:"")}async function g(e,t){let r=t.res||t.ctx&&t.ctx.res;if(!e.getInitialProps)return t.ctx&&t.Component?{pageProps:await g(t.Component,t.ctx)}:{};let a=await e.getInitialProps(t);if(r&&u(r))return a;if(!a)throw Object.defineProperty(Error(`"${m(e)}.getInitialProps()" should resolve to an object. But found "${a}" instead.`),"__NEXT_ERROR_CODE",{value:"E394",enumerable:!1,configurable:!0});return a}let h="u">typeof performance,x=h&&["mark","measure","getEntriesByName"].every(e=>"function"==typeof performance[e]);class f extends Error{}class _ extends Error{}class b extends Error{constructor(e){super(),this.code="ENOENT",this.name="PageNotFoundError",this.message=`Cannot find module for page: ${e}`}}class y extends Error{constructor(e,t){super(),this.message=`Failed to load static file for page: ${e} ${t}`}}class v extends Error{constructor(){super(),this.code="ENOENT",this.message="Cannot find the middleware module"}}function j(e){return JSON.stringify({message:e.message,stack:e.stack})}},573668,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"isLocalURL",{enumerable:!0,get:function(){return i}});let a=e.r(718967),s=e.r(652817);function i(e){if(!(0,a.isAbsoluteUrl)(e))return!0;try{let t=(0,a.getLocationOrigin)(),r=new URL(e,t);return r.origin===t&&(0,s.hasBasePath)(r.pathname)}catch(e){return!1}}},284508,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"errorOnce",{enumerable:!0,get:function(){return a}});let a=e=>{}},522016,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0});var a={default:function(){return f},useLinkStatus:function(){return b}};for(var s in a)Object.defineProperty(r,s,{enumerable:!0,get:a[s]});let i=e.r(151836),l=e.r(843476),n=i._(e.r(271645)),o=e.r(195057),c=e.r(8372),d=e.r(818581),m=e.r(718967),u=e.r(405550);e.r(233525);let p=e.r(91949),g=e.r(573668),h=e.r(509396);function x(e){return"string"==typeof e?e:(0,o.formatUrl)(e)}function f(t){var r;let a,s,i,[o,f]=(0,n.useOptimistic)(p.IDLE_LINK_STATUS),b=(0,n.useRef)(null),{href:y,as:v,children:j,prefetch:N=null,passHref:w,replace:A,shallow:S,scroll:T,onClick:C,onMouseEnter:k,onTouchStart:I,legacyBehavior:E=!1,onNavigate:L,ref:O,unstable_dynamicOnHover:M,...P}=t;a=j,E&&("string"==typeof a||"number"==typeof a)&&(a=(0,l.jsx)("a",{children:a}));let R=n.default.useContext(c.AppRouterContext),$=!1!==N,D=!1!==N?null===(r=N)||"auto"===r?h.FetchStrategy.PPR:h.FetchStrategy.Full:h.FetchStrategy.PPR,{href:z,as:B}=n.default.useMemo(()=>{let e=x(y);return{href:e,as:v?x(v):e}},[y,v]);if(E){if(a?.$$typeof===Symbol.for("react.lazy"))throw Object.defineProperty(Error("`<Link legacyBehavior>` received a direct child that is either a Server Component, or JSX that was loaded with React.lazy(). This is not supported. Either remove legacyBehavior, or make the direct child a Client Component that renders the Link's `<a>` tag."),"__NEXT_ERROR_CODE",{value:"E863",enumerable:!1,configurable:!0});s=n.default.Children.only(a)}let U=E?s&&"object"==typeof s&&s.ref:O,H=n.default.useCallback(e=>(null!==R&&(b.current=(0,p.mountLinkInstance)(e,z,R,D,$,f)),()=>{b.current&&((0,p.unmountLinkForCurrentNavigation)(b.current),b.current=null),(0,p.unmountPrefetchableInstance)(e)}),[$,z,R,D,f]),F={ref:(0,d.useMergedRef)(H,U),onClick(t){E||"function"!=typeof C||C(t),E&&s.props&&"function"==typeof s.props.onClick&&s.props.onClick(t),!R||t.defaultPrevented||function(t,r,a,s,i,l,o){if("u">typeof window){let c,{nodeName:d}=t.currentTarget;if("A"===d.toUpperCase()&&((c=t.currentTarget.getAttribute("target"))&&"_self"!==c||t.metaKey||t.ctrlKey||t.shiftKey||t.altKey||t.nativeEvent&&2===t.nativeEvent.which)||t.currentTarget.hasAttribute("download"))return;if(!(0,g.isLocalURL)(r)){i&&(t.preventDefault(),location.replace(r));return}if(t.preventDefault(),o){let e=!1;if(o({preventDefault:()=>{e=!0}}),e)return}let{dispatchNavigateAction:m}=e.r(699781);n.default.startTransition(()=>{m(a||r,i?"replace":"push",l??!0,s.current)})}}(t,z,B,b,A,T,L)},onMouseEnter(e){E||"function"!=typeof k||k(e),E&&s.props&&"function"==typeof s.props.onMouseEnter&&s.props.onMouseEnter(e),R&&$&&(0,p.onNavigationIntent)(e.currentTarget,!0===M)},onTouchStart:function(e){E||"function"!=typeof I||I(e),E&&s.props&&"function"==typeof s.props.onTouchStart&&s.props.onTouchStart(e),R&&$&&(0,p.onNavigationIntent)(e.currentTarget,!0===M)}};return(0,m.isAbsoluteUrl)(B)?F.href=B:E&&!w&&("a"!==s.type||"href"in s.props)||(F.href=(0,u.addBasePath)(B)),i=E?n.default.cloneElement(s,F):(0,l.jsx)("a",{...P,...F,children:a}),(0,l.jsx)(_.Provider,{value:o,children:i})}e.r(284508);let _=(0,n.createContext)(p.IDLE_LINK_STATUS),b=()=>(0,n.useContext)(_);("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},402874,521323,636772,e=>{"use strict";var t=e.i(843476),r=e.i(764205),a=e.i(266027);let s=(0,e.i(243652).createQueryKeys)("healthReadiness"),i=async()=>{let e=(0,r.getProxyBaseUrl)(),t=await fetch(`${e}/health/readiness`);if(!t.ok)throw Error(`Failed to fetch health readiness: ${t.statusText}`);return t.json()},l=()=>(0,a.useQuery)({queryKey:s.detail("readiness"),queryFn:i,staleTime:3e5});e.s(["useHealthReadiness",0,l],521323);var n=e.i(115571),o=e.i(271645);function c(e){let t=t=>{"disableBouncingIcon"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableBouncingIcon"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(n.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(n.LOCAL_STORAGE_EVENT,r)}}function d(){return"true"===(0,n.getLocalStorageItem)("disableBouncingIcon")}function m(){return(0,o.useSyncExternalStore)(c,d)}var u=e.i(275144),p=e.i(268004),g=e.i(321836),h=e.i(62478),x=e.i(487486),f=e.i(519455),_=e.i(699375),b=e.i(475254);let y=(0,b.default)("menu",[["path",{d:"M4 12h16",key:"1lakjw"}],["path",{d:"M4 18h16",key:"19g7jn"}],["path",{d:"M4 6h16",key:"1o0s65"}]]);(0,b.default)("moon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]);var v=e.i(972518);(0,b.default)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);var j=e.i(522016);async function N(){let e=(0,r.getProxyBaseUrl)(),t=await fetch(`${e}/public/litellm_blog_posts`);if(!t.ok)throw Error(`Failed to fetch blog posts: ${t.statusText}`);return t.json()}function w(e){let t=t=>{"disableBlogPosts"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableBlogPosts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(n.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(n.LOCAL_STORAGE_EVENT,r)}}function A(){return"true"===(0,n.getLocalStorageItem)("disableBlogPosts")}function S(){return(0,o.useSyncExternalStore)(w,A)}var T=e.i(755146),C=e.i(531278);let k=()=>{let e=S(),{data:r,isLoading:s,isError:i,refetch:l}=(0,a.useQuery)({queryKey:["blogPosts"],queryFn:N,staleTime:36e5,retry:1,retryDelay:0});return e?null:(0,t.jsxs)(T.DropdownMenu,{children:[(0,t.jsx)(T.DropdownMenuTrigger,{asChild:!0,children:(0,t.jsx)(f.Button,{variant:"ghost",children:"Blog"})}),(0,t.jsx)(T.DropdownMenuContent,{align:"end",className:"w-[420px] p-1",children:s?(0,t.jsx)("div",{className:"flex items-center justify-center py-4",children:(0,t.jsx)(C.Loader2,{className:"h-4 w-4 animate-spin text-muted-foreground"})}):i?(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2 p-3",children:[(0,t.jsx)("span",{className:"text-destructive text-sm",children:"Failed to load posts"}),(0,t.jsx)(f.Button,{size:"sm",variant:"outline",onClick:()=>l(),children:"Retry"})]}):r&&0!==r.posts.length?(0,t.jsxs)(t.Fragment,{children:[r.posts.slice(0,5).map(e=>(0,t.jsx)(T.DropdownMenuItem,{asChild:!0,className:"cursor-pointer",children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",className:"block w-full px-2 py-2",children:[(0,t.jsx)("h5",{className:"font-semibold text-sm mb-0.5",children:e.title}),(0,t.jsx)("div",{className:"text-[11px] text-muted-foreground mb-0.5",children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"text-xs line-clamp-2 text-muted-foreground m-0",children:e.description})]})},e.url)),(0,t.jsx)(T.DropdownMenuSeparator,{}),(0,t.jsx)(T.DropdownMenuItem,{asChild:!0,children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",className:"block w-full px-2 py-2 text-sm",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-3 py-3 text-muted-foreground text-sm",children:"No posts available"})})]})};function I(e){let t=t=>{"disableShowPrompts"===t.key&&e()},r=t=>{let{key:r}=t.detail;"disableShowPrompts"===r&&e()};return window.addEventListener("storage",t),window.addEventListener(n.LOCAL_STORAGE_EVENT,r),()=>{window.removeEventListener("storage",t),window.removeEventListener(n.LOCAL_STORAGE_EVENT,r)}}function E(){return"true"===(0,n.getLocalStorageItem)("disableShowPrompts")}function L(){return(0,o.useSyncExternalStore)(I,E)}e.s(["useDisableShowPrompts",()=>L],636772);let O=(0,b.default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]),M=(0,b.default)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]]),P=()=>L()?null:(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(f.Button,{asChild:!0,variant:"outline",className:"shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow",children:(0,t.jsxs)("a",{href:"https://www.litellm.ai/support",target:"_blank",rel:"noopener noreferrer",children:[(0,t.jsx)(M,{className:"h-4 w-4"}),"Join Slack"]})}),(0,t.jsx)(f.Button,{asChild:!0,variant:"outline",className:"shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow",children:(0,t.jsxs)("a",{href:"https://github.com/BerriAI/litellm",target:"_blank",rel:"noopener noreferrer",children:[(0,t.jsx)(O,{className:"h-4 w-4"}),"Star us on GitHub"]})})]});var R=e.i(135214),$=e.i(371401),D=e.i(746798),z=e.i(664659),B=e.i(243553),U=e.i(292270),H=e.i(263488),F=e.i(98919),G=e.i(284614);let V=({onLogout:e})=>{let{userId:r,userEmail:a,userRole:s,premiumUser:i}=(0,R.default)(),l=L(),c=(0,$.useDisableUsageIndicator)(),d=S(),u=m(),[p,g]=(0,o.useState)(!1);(0,o.useEffect)(()=>{g("true"===(0,n.getLocalStorageItem)("disableShowNewBadge"))},[]);let h=(e,t)=>{t?(0,n.setLocalStorageItem)(e,"true"):(0,n.removeLocalStorageItem)(e),(0,n.emitLocalStorageChange)(e)},b=({label:e,checked:r,onChange:a,ariaLabel:s})=>(0,t.jsxs)("div",{className:"flex items-center justify-between w-full",children:[(0,t.jsx)("span",{className:"text-muted-foreground text-sm",children:e}),(0,t.jsx)(_.Switch,{checked:r,onCheckedChange:a,"aria-label":s??e})]});return(0,t.jsxs)(T.DropdownMenu,{children:[(0,t.jsx)(T.DropdownMenuTrigger,{asChild:!0,children:(0,t.jsxs)(f.Button,{variant:"ghost",className:"gap-2",children:[(0,t.jsx)(G.User,{className:"h-4 w-4"}),(0,t.jsx)("span",{className:"text-sm",children:"User"}),(0,t.jsx)(z.ChevronDown,{className:"h-3 w-3"})]})}),(0,t.jsxs)(T.DropdownMenuContent,{align:"end",className:"w-72 p-3",children:[(0,t.jsxs)("div",{className:"flex flex-col gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(H.Mail,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground truncate max-w-[160px]",children:a||"-"})]}),i?(0,t.jsxs)(x.Badge,{className:"bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 gap-1",children:[(0,t.jsx)(B.Crown,{className:"h-3 w-3"}),"Premium"]}):(0,t.jsx)(D.TooltipProvider,{children:(0,t.jsxs)(D.Tooltip,{children:[(0,t.jsx)(D.TooltipTrigger,{asChild:!0,children:(0,t.jsxs)(x.Badge,{variant:"secondary",className:"gap-1",children:[(0,t.jsx)(B.Crown,{className:"h-3 w-3"}),"Standard"]})}),(0,t.jsx)(D.TooltipContent,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(T.DropdownMenuSeparator,{}),(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(G.User,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsx)("span",{className:"text-sm truncate max-w-[150px]",title:r||"-",children:r||"-"})]}),(0,t.jsxs)("div",{className:"flex items-center justify-between",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2 text-sm",children:[(0,t.jsx)(F.Shield,{className:"h-4 w-4 text-muted-foreground"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{className:"text-sm",children:s})]}),(0,t.jsx)(T.DropdownMenuSeparator,{}),(0,t.jsx)(b,{label:"Hide New Feature Indicators",checked:p,onChange:e=>{g(e),h("disableShowNewBadge",e)},ariaLabel:"Toggle hide new feature indicators"}),(0,t.jsx)(b,{label:"Hide All Prompts",checked:l,onChange:e=>h("disableShowPrompts",e),ariaLabel:"Toggle hide all prompts"}),(0,t.jsx)(b,{label:"Hide Usage Indicator",checked:c,onChange:e=>h("disableUsageIndicator",e),ariaLabel:"Toggle hide usage indicator"}),(0,t.jsx)(b,{label:"Hide Blog Posts",checked:d,onChange:e=>h("disableBlogPosts",e),ariaLabel:"Toggle hide blog posts"}),(0,t.jsx)(b,{label:"Hide Bouncing Icon",checked:u,onChange:e=>h("disableBouncingIcon",e),ariaLabel:"Toggle hide bouncing icon"})]}),(0,t.jsx)(T.DropdownMenuSeparator,{}),(0,t.jsxs)(T.DropdownMenuItem,{onClick:e,children:[(0,t.jsx)(U.LogOut,{className:"h-4 w-4"}),"Logout"]})]})]})};var K=e.i(967489),W=e.i(283713);let q=({onWorkerSwitch:e})=>{let{isControlPlane:r,selectedWorker:a,workers:s}=(0,W.useWorker)();return r&&a?(0,t.jsxs)(K.Select,{value:a.worker_id,onValueChange:e,children:[(0,t.jsx)(K.SelectTrigger,{className:"min-w-[180px]",children:(0,t.jsx)(K.SelectValue,{})}),(0,t.jsx)(K.SelectContent,{children:s.map(e=>(0,t.jsx)(K.SelectItem,{value:e.worker_id,disabled:e.worker_id===a.worker_id,children:e.name},e.worker_id))})]}):null};e.s(["default",0,({userID:e,userEmail:a,userRole:s,premiumUser:i,proxySettings:n,setProxySettings:c,accessToken:d,isPublicPage:_=!1,sidebarCollapsed:b=!1,onToggleSidebar:N,isDarkMode:w,toggleDarkMode:A})=>{let S=(0,r.getProxyBaseUrl)(),[T,C]=(0,o.useState)(""),{logoUrl:I}=(0,u.useTheme)(),{data:E}=l(),L=E?.litellm_version,O=m(),M=I||`${S}/get_image`;return(0,o.useEffect)(()=>{(async()=>{if(d){let e=await (0,h.fetchProxySettings)(d);console.log("response from fetchProxySettings",e),e&&c(e)}})()},[d]),(0,o.useEffect)(()=>{C(n?.PROXY_LOGOUT_URL||"")},[n]),(0,t.jsx)("nav",{className:"bg-background border-b border-border sticky top-0 z-10",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex items-center h-14 px-4",children:[(0,t.jsxs)("div",{className:"flex items-center flex-shrink-0",children:[N&&(0,t.jsx)("button",{onClick:N,className:"flex items-center justify-center w-10 h-10 mr-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors",title:b?"Expand sidebar":"Collapse sidebar","aria-label":b?"Expand sidebar":"Collapse sidebar",children:b?(0,t.jsx)(y,{className:"h-5 w-5"}):(0,t.jsx)(v.PanelLeftClose,{className:"h-5 w-5"})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(j.default,{href:S||"/",className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsx)("div",{className:"h-10 max-w-48 flex items-center justify-center overflow-hidden",children:(0,t.jsx)("img",{src:M,alt:"LiteLLM Brand",className:"max-w-full max-h-full w-auto h-auto object-contain"})})})}),L&&(0,t.jsxs)("div",{className:"relative",children:[!O&&(0,t.jsx)("span",{className:"absolute -top-1 -left-2 text-lg animate-bounce",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(x.Badge,{variant:"outline",className:"relative text-xs font-medium cursor-pointer z-10",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"flex-shrink-0",children:["v",L]})})]})]})]}),(0,t.jsxs)("div",{className:"flex items-center space-x-5 ml-auto",children:[(0,t.jsx)(q,{onWorkerSwitch:e=>{(0,p.clearTokenCookies)(),(0,g.clearStoredReturnUrl)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=`/ui/login?worker=${encodeURIComponent(e)}`}}),(0,t.jsx)(P,{}),!1,(0,t.jsx)(f.Button,{variant:"ghost",asChild:!0,children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",children:"Docs"})}),(0,t.jsx)(k,{}),!_&&(0,t.jsx)(V,{onLogout:()=>{(0,p.clearTokenCookies)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=T}})]})]})})})}],402874)}]);