(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},350682,e=>{let t=(0,e.i(475254).f)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]);e.s(["f",0,t],350682)},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},164668,e=>{var t=e.i(717521);e.s(["f",()=>t.f])},492941,e=>{var t=e.i(602869),s=e.i(266027);async function i(){let e=(0,t._h)(),s=await fetch(`${e}/public/litellm_blog_posts`);if(!s.ok)throw Error(`Failed to fetch blog posts: ${s.statusText}`);return s.json()}e.s(["f",0,()=>(0,s.f)({queryKey:["blogPosts"],queryFn:i,staleTime:36e5,retry:1,retryDelay:0})])},842354,e=>{var t=e.i(768371);e.s(["f",0,()=>t.q.useQuery("get","/.well-known/litellm-ui-config").data?.claude_code_marketplace_auth_required!==!1])},731565,e=>{var t=e.i(115571),s=e.i(271645);function i(e){let s=t=>{"disableBlogPosts"===t.key&&e()},i=t=>{let{key:s}=t.detail;"disableBlogPosts"===s&&e()};return window.addEventListener("storage",s),window.addEventListener(t.M,i),()=>{window.removeEventListener("storage",s),window.removeEventListener(t.M,i)}}function a(){return"true"===(0,t.t)("disableBlogPosts")}e.s(["f",0,function(){return(0,s.useSyncExternalStore)(i,a,()=>!1)}])},556260,319684,e=>{var t=e.i(271645),s=e.i(602869),i=e.i(115571);function a(e){return window.addEventListener("storage",e),window.addEventListener(i.M,e),()=>{window.removeEventListener("storage",e),window.removeEventListener(i.M,e)}}e.s(["f",0,function(e){let r=e?`disableLiteAdmin:${JSON.stringify([(0,s._h)(),e])}`:null;return[(0,t.useSyncExternalStore)(a,()=>null!==r&&"true"===(0,i.t)(r),()=>!1),e=>{null!==r&&(e?(0,i.A)(r,"true"):(0,i.p)(r),(0,i.s)(r))}]}],556260),e.s(["f",0,"Aria Agent"],319684)},251773,423680,771243,895335,e=>{var t=e.i(843476),s=e.i(731565),i=e.i(492941);let a="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var r=e.i(519455),l=e.i(755146),n=e.i(664659),o=e.i(164668);e.s(["f",0,()=>{let e=(0,s.f)(),{data:d,isLoading:c,isError:u,refetch:m}=(0,i.f)();return e?null:(0,t.jsxs)(l.G,{modal:!1,children:[(0,t.jsxs)(l.X,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(r.K,{variant:"ghost",className:`${a} border-0!`}),children:["Blog",(0,t.jsx)(n.f,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(l.L,{align:"end",side:"bottom",className:"w-auto",children:c?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(o.f,{role:"img","aria-label":"loading",className:"size-4 animate-spin"})}):u?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:"Failed to load posts"}),(0,t.jsx)(r.K,{variant:"outline",size:"sm",onClick:()=>m(),children:"Retry"})]}):d&&0!==d.posts.length?(0,t.jsxs)(t.Fragment,{children:[d.posts.slice(0,5).map(e=>(0,t.jsx)(l.r,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(l.U,{}),(0,t.jsx)(l.r,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:"No posts available"})})]})}],251773);let d=()=>(0,t.jsx)(n.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0});e.s(["f",0,()=>(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:a,children:["Docs",(0,t.jsx)(d,{})]})],423680);var c=e.i(636772),u=e.i(747862),m=e.i(746798),p=e.i(196631),g=e.i(350682);let f=[{href:"https://www.litellm.ai/support",label:"Join Slack",tooltip:"LiteLLM Slack community",Icon:(0,e.i(475254).f)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]])},{href:"https://github.com/BerriAI/litellm",label:"LiteLLM on GitHub",tooltip:"LiteLLM on GitHub",Icon:g.f}];e.s(["f",0,()=>(0,c.f)()?null:(0,t.jsx)(m.g,{children:(0,t.jsx)(u.f,{"aria-label":"Community links",children:f.map(({href:e,label:s,tooltip:i,Icon:a})=>(0,t.jsxs)(m.W,{children:[(0,t.jsx)(m.y,{render:(0,t.jsx)("a",{href:e,target:"_blank",rel:"noopener noreferrer","aria-label":s,className:(0,p.c)((0,r.i)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(a,{})}),(0,t.jsx)(m.Y,{children:i})]},e))})})],771243);var x=e.i(143488),h=e.i(768371),b=e.i(271645),j=e.i(115571);function _(e){return window.addEventListener("storage",e),window.addEventListener(j.M,e),()=>{window.removeEventListener("storage",e),window.removeEventListener(j.M,e)}}var v=e.i(487486),w=e.i(337822),y=e.i(772436);let N=/^v?(\d+)\.(\d+)\.(\d+)/,k=e=>{let t=N.exec(e.trim());return t?[Number(t[1]),Number(t[2]),Number(t[3])]:null};var $=e.i(245423);let C={readKey:"litellmHideAutoRouterAnnouncement",title:"LiteLLM Auto Router",description:"Route every request to the cheapest model that can handle it, no prompt changes needed.",linkLabel:"Read the docs",href:"https://docs.litellm.ai/docs/proxy/auto_routing"},P=(e,t,s)=>`${e} ${1===e?t:s}`,z=({announcement:e})=>{var s;let i=(s=e.readKey,(0,b.useSyncExternalStore)(_,()=>"true"===(0,j.t)(s),()=>!1));return(0,t.jsxs)("article",{"aria-label":e.title,className:"flex flex-col gap-2",children:[(0,t.jsx)("p",{className:"text-sm font-medium",children:e.title}),(0,t.jsx)("p",{className:"text-sm leading-snug text-muted-foreground",children:e.description}),(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsx)("a",{className:(0,p.c)((0,r.i)({size:"sm"})),href:e.href,target:"_blank",rel:"noopener noreferrer",children:e.linkLabel}),i?null:(0,t.jsx)(r.K,{variant:"link",size:"sm",className:"px-1!",onClick:()=>{var t;return t=e.readKey,void((0,j.A)(t,"true"),(0,j.s)(t))},children:"Mark as read"})]})]})},L=({currentVersion:e,latestRelease:s})=>{var i;let a=e&&s&&((e,t)=>{let s=k(e),i=k(t);if(!s||!i)return!1;for(let e=0;e<3;e+=1)if(i[e]!==s[e])return i[e]>s[e];return!1})(e,s.version)?{readKey:`litellm:upgradeBannerDismissed:${s.version}`,title:`LiteLLM v${s.version} is available`,description:`${(({new_features:e,bug_fixes:t,other_updates:s})=>[P(e,"new feature","new features"),P(t,"fix","fixes"),`and ${P(s,"other update","other updates")}`].join(", "))(s)}. Your current version is v${e}`,linkLabel:"View release notes",href:s.release_url}:null,r=a?[a,C]:[C],l=(i=r.map(e=>e.readKey),(0,b.useSyncExternalStore)(_,()=>i.filter(e=>"true"!==(0,j.t)(e)).length,()=>i.length));return(0,t.jsxs)(w.V,{children:[(0,t.jsx)(w.J,{className:"flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!","aria-label":l>0?`Notifications, ${l} unread`:"Notifications",children:(0,t.jsxs)("span",{className:"relative inline-flex",children:[(0,t.jsx)($.f,{className:"size-4","aria-hidden":!0}),l>0?(0,t.jsx)(v.f,{className:"absolute -top-0.5 -right-1 size-1.5 p-0","aria-hidden":!0}):null]})}),(0,t.jsx)(w.T,{align:"end","aria-label":"Notifications",children:r.map((e,s)=>(0,t.jsxs)(b.Fragment,{children:[s>0?(0,t.jsx)(y.f,{}):null,(0,t.jsx)(z,{announcement:e})]},e.readKey))})]})};e.s(["f",0,({accessToken:e})=>{let{data:s}=(0,x.f)(e),{data:i}=h.q.useQuery("get","/get/latest_release_info",{},{enabled:!!e,staleTime:36e5,retry:!1});return(0,t.jsx)(L,{currentVersion:s?.litellm_version,latestRelease:i})}],895335)},641141,e=>{var t=e.i(843476),s=e.i(135214),i=e.i(731565),a=e.i(912089),r=e.i(556260),l=e.i(319684),n=e.i(636772),o=e.i(115571),d=e.i(222038),c=e.i(782066),u=e.i(708347),m=e.i(664659),p=e.i(344523),g=e.i(243553),f=e.i(465261),x=e.i(292270),h=e.i(263488),b=e.i(581418),j=e.i(284614),_=e.i(618566),v=e.i(799676),w=e.i(487486),y=e.i(337822),N=e.i(772436),k=e.i(699375),$=e.i(746798),C=e.i(922407),P=e.i(196631),z=e.i(271645);e.s(["f",0,({onLogout:e,variant:L="navbar",collapsed:S=!1})=>{let{userId:D,userEmail:I,userRole:A,userRoleLabel:E,isViewOnly:M,premiumUser:O,loginMethod:T}=(0,s.f)(),B=(0,_.useRouter)(),[R,H]=(0,z.useState)(!1),U=(0,n.f)(),F=(0,i.f)(),q=(0,a.f)(),[K,Z]=(0,r.f)(D),Y=D&&!M&&(0,u.u)(A),[J,V]=(0,z.useState)(!1);(0,z.useEffect)(()=>{V("true"===(0,o.t)("disableShowNewBadge"))},[]);let W=I||D||"user",Q=function(e,t){let s=e?.split("@")[0]?.trim();if(s){let e=s.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(I,D),G=function(e){let t=0;for(let s=0;s<e.length;s+=1)t=e.charCodeAt(s)+((t<<5)-t);return Math.abs(t)%360}(W),X=(0,d.f)(I,D);return(0,t.jsxs)(y.V,{open:R,onOpenChange:H,children:["sidebar"===L?(0,t.jsxs)(y.J,{render:(0,t.jsx)("button",{type:"button",className:(0,P.c)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",S?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":`Account menu — ${E??"Unknown role"} — signed in as ${I||D||"unknown"}`,"aria-haspopup":"dialog",title:S?X:void 0}),children:[(0,t.jsx)(v.E,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(v.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${G} 46% 38%)`},children:Q})}),!S&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:X}),E&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:E})]}),(0,t.jsx)(p.f,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(y.J,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":`Account menu — ${E??"Unknown role"} — signed in as ${I||D||"unknown"}`,"aria-haspopup":"dialog"}),children:[(0,t.jsx)(v.E,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(v.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${G} 46% 38%)`},children:Q})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:X}),(0,t.jsx)(m.f,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(y.T,{align:"sidebar"===L?"start":"end",side:"sidebar"===L?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(h.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:I||"-"})]}),O?(0,t.jsxs)(w.f,{children:[(0,t.jsx)(g.f,{className:"size-3"}),"Premium"]}):(0,t.jsx)($.g,{children:(0,t.jsxs)($.W,{children:[(0,t.jsxs)($.y,{render:(0,t.jsx)(w.f,{variant:"outline"}),children:[(0,t.jsx)(g.f,{className:"size-3"}),"Standard"]}),(0,t.jsx)($.Y,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(N.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(j.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:D||"-",children:D||"-"}),(0,t.jsx)(C.f,{value:D,label:"Copy User ID"})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(b.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{children:E})]}),(0,t.jsx)(N.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide New Feature Indicators"}),(0,t.jsx)(k.f,{size:"sm",checked:J,onCheckedChange:e=>{V(e),e?(0,o.A)("disableShowNewBadge","true"):(0,o.p)("disableShowNewBadge"),(0,o.s)("disableShowNewBadge")},"aria-label":"Toggle hide new feature indicators"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide All Prompts"}),(0,t.jsx)(k.f,{size:"sm",checked:U,onCheckedChange:e=>{e?(0,o.A)("disableShowPrompts","true"):(0,o.p)("disableShowPrompts"),(0,o.s)("disableShowPrompts")},"aria-label":"Toggle hide all prompts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Blog Posts"}),(0,t.jsx)(k.f,{size:"sm",checked:F,onCheckedChange:e=>{e?(0,o.A)("disableBlogPosts","true"):(0,o.p)("disableBlogPosts"),(0,o.s)("disableBlogPosts")},"aria-label":"Toggle hide blog posts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Bouncing Icon"}),(0,t.jsx)(k.f,{size:"sm",checked:q,onCheckedChange:e=>{e?(0,o.A)("disableBouncingIcon","true"):(0,o.p)("disableBouncingIcon"),(0,o.s)("disableBouncingIcon")},"aria-label":"Toggle hide bouncing icon"})]}),!0===O&&Y&&(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("span",{className:"text-muted-foreground",children:["Hide ",l.f]}),(0,t.jsx)(k.f,{size:"sm",checked:K,onCheckedChange:Z,"aria-label":`Toggle hide ${l.f}`})]})]}),(0,t.jsx)(N.f,{}),"username_password"===T&&(0,t.jsxs)("button",{type:"button",onClick:()=>{H(!1),B.push((0,c.B)("change-password"))},className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(f.f,{className:"size-4"}),"Change Password"]}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(x.f,{className:"size-4"}),"Logout"]})]})]})}])},383862,e=>{var t=e.i(843476),s=e.i(618393),i=e.i(131792),a=e.i(950594),r=e.i(283713);e.s(["f",0,({onWorkerSwitch:e})=>{let{isControlPlane:l,selectedWorker:n,workers:o}=(0,r.f)();if(!l||!n)return null;let d=o.map(e=>({label:e.name,value:e.worker_id,disabled:e.worker_id===n.worker_id}));return(0,t.jsxs)(i.x,{items:d,value:d.find(e=>e.value===n.worker_id)??null,itemToStringLabel:e=>e.label,onValueChange:t=>{t&&e(t.value)},children:[(0,t.jsx)(i.c,{className:"min-w-[180px]","aria-label":"Worker",children:(0,t.jsx)(a.B,{align:"inline-start",children:(0,t.jsx)(s.f,{className:"size-4"})})}),(0,t.jsxs)(i.R,{children:[(0,t.jsx)(i.S,{children:"No matching workers"}),(0,t.jsx)(i.k,{children:e=>(0,t.jsx)(i.T,{value:e,disabled:e.disabled,children:e.label},e.value)})]})]})}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let s,{apiKeySource:i,accessToken:a,apiKey:r,inputMessage:l,chatHistory:n,selectedTags:o,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedVoice:m,endpointType:p,selectedModel:g,selectedSdk:f,proxySettings:x,customHeaders:h}=e,b="session"===i?a:r,j=window.location.origin,_=x?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?j=_:x?.PROXY_BASE_URL&&(j=x.PROXY_BASE_URL);let v=l||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),y=n.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),N={};o.length>0&&(N.tags=o),d.length>0&&(N.vector_stores=d),c.length>0&&(N.guardrails=c),u.length>0&&(N.policies=u);let k=g||"your-model-name",$=h&&Object.keys(h).length>0?`,
	default_headers=${JSON.stringify(h,null,2).replace(/\n/g,"\n	")}`:"",C="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${j}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${j}"${$}
)`;switch(p){case t.Q.CHAT:{let e=Object.keys(N).length>0,t="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=y.length>0?y:[{role:"user",content:v}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(i,null,4)}${t}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${k}",
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.Q.RESPONSES:{let e=Object.keys(N).length>0,t="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=y.length>0?y:[{role:"user",content:v}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(i,null,4)}${t}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${k}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${w}"},
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
`;break}case t.Q.IMAGE:s="azure"===f?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${k}",
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
	model="${k}",
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
`;break;case t.Q.IMAGE_EDITS:s="azure"===f?`
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
	model="${k}",
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
prompt = "${w}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${k}",
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
`;break;case t.Q.EMBEDDINGS:s=`
response = client.embeddings.create(
	input="${l||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:s=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${l?`,
	prompt="${l.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:s=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${l||"Your text to convert to speech here"}",
	voice="${m}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:s="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${s}`}])},209261,e=>{var t=e.i(201787);let s=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,i=e=>e.trim().replace(/\/+$/,""),a=/\.(md|markdown|txt|json|ya?ml|toml)$/i,r=/\.zip$/i,l=/^[0-9a-fA-F]{64}$/,n=/^\d{1,3}(\.\d{1,3}){3}$/,o=/^[A-Za-z0-9-]+$/,d=/^[A-Za-z0-9._-]+$/,c=/^https?:\/\//i,u="ssh://",m=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,p=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{try{return new URL(e)}catch{return null}},f=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!n.test(e.hostname),x=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},h=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),b=(e,a,r,l)=>{let n=i(l??"");return""!==n?s.test(n)?{parsed:{source:"git-subdir",url:a,path:n},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:a,path:n}),suggestedName:h(x(n))}:null:{parsed:{source:"url",url:a},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:a}),suggestedName:h(r)}},j="<your-litellm-key>";e.s(["i",0,j,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${j}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&c.test(e.url)?e.url:null,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||l.test(e.trim()),"P",0,e=>{let t=i(e);return""!==t&&s.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,l)=>{let n=((e,t)=>{let s=e.trim(),i=m.exec(s),a=i?`${u}${i[1]}@${i[2]}/${i[3]}`:s;if(!a.toLowerCase().startsWith(u))return null;let r=g(a);if(!r||""===r.username||""!==r.password||!f(r))return null;let l=a.indexOf("/",u.length);return -1===l||r.pathname!==a.slice(l)||p(r).length<2?null:b("SSH",s,x(r.pathname).replace(/\.git$/i,""),t)})(e,l);if(n)return n;let c=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let s=g(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return s&&"https:"===s.protocol&&""===s.username&&""===s.password&&f(s)?s:null})(e);if(!c)return null;if(r.test(c.pathname))return{parsed:{source:"archive",url:c.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${c.host}${c.pathname}`}),suggestedName:h(x(c.pathname).replace(r,""))};if("github.com"===c.hostname.replace(/^www\./,""))return((e,r)=>{let l=p(e);if(l.length<2)return null;let n=l[0],c=l[1].replace(/\.git$/,"");if(!o.test(n)||!d.test(c))return null;let u=`${n}/${c}`,m=`https://github.com/${u}`,g={parsed:{source:"github",repo:u},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:u}),suggestedName:h(c)};if(l.length>=4&&("tree"===l[2]||"blob"===l[2])){let e=l.slice(4),r=x(e.join("/")),n=a.test(r)?e.slice(0,-1):e;if(0===n.length)return g;let o=i(n.join("/"));return s.test(o)?{parsed:{source:"git-subdir",url:m,path:o},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:u,path:o}),suggestedName:h(x(o))}:null}if(2!==l.length)return null;let f=i(r??"");return""!==f?s.test(f)?{parsed:{source:"git-subdir",url:m,path:f},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:u,path:f}),suggestedName:h(x(f))}:null:g})(c,l);if(p(c).length<2)return null;let j=x(c.pathname).replace(/\.git$/,"");return b("Git",`${c.protocol}//${c.host}${c.pathname.replace(/\/+$/,"")}`,j,l)},"e",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},652272,e=>{var t=e.i(843476),s=e.i(271645),i=e.i(871689),a=e.i(643531),r=e.i(174886),l=e.i(306228),n=e.i(345771),o=e.i(322831),d=e.i(842354),c=e.i(196631),u=e.i(209261);let m=({source:e})=>{let{t:s}=(0,o.f)(),i=(0,u.E)(e),a=i&&"git-subdir"===e.source&&e.path?`${i}/tree/main/${e.path}`:i;return a?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:s("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsxs)("a",{href:a,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[a.replace("https://",""),(0,t.jsx)(l.f,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:s("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:(0,u._)(e)})]}):null};e.s(["f",0,({skill:e,onBack:l})=>{let{t:p}=(0,o.f)(),[g,f]=(0,s.useState)("overview"),[x,h]=(0,s.useState)(null),b=(0,d.f)(),j=(e,t)=>{navigator.clipboard.writeText(e),h(t),setTimeout(()=>h(null),2e3)},_=(0,u.a)(e),v=(0,u.U)(window.location.origin,b),w=[...e.category?[{property:p("claudeCodePlugins.skillDetail.properties.category"),value:e.category}]:[],...e.domain?[{property:p("claudeCodePlugins.skillDetail.properties.domain"),value:e.domain}]:[],...e.namespace?[{property:p("claudeCodePlugins.skillDetail.properties.namespace"),value:e.namespace}]:[],...e.version?[{property:p("claudeCodePlugins.skillDetail.properties.version"),value:e.version}]:[],...e.author?.name?[{property:p("claudeCodePlugins.skillDetail.properties.author"),value:e.author.name}]:[],...e.created_at?[{property:p("claudeCodePlugins.skillDetail.properties.added"),value:new Date(e.created_at).toLocaleDateString()}]:[]],y=[{key:"overview",label:p("claudeCodePlugins.skillDetail.tabs.overview")},{key:"usage",label:p("claudeCodePlugins.skillDetail.tabs.usage")}];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:l,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(i.f,{className:"size-3"}),(0,t.jsx)("span",{children:p("claudeCodePlugins.skillDetail.back")})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:y.map(e=>(0,t.jsx)("div",{onClick:()=>f(e.key),className:(0,c.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",g===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===g&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.overview.title")}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.subtitle")}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.propertyHeader")}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:w.map((e,s)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},s))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.status")}),(0,t.jsx)("span",{className:(0,c.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:p(e.enabled?"claudeCodePlugins.skillDetail.overview.public":"claudeCodePlugins.skillDetail.overview.draft")})]}),(0,t.jsx)(m,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.tags")}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:p("claudeCodePlugins.skillDetail.overview.skillId")}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.usage.title")}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.usage.intro")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:p("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>j(_,"install"),className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===x?"text-success":"text-info"),children:["install"===x?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),p("install"===x?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:_})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.usage.staleCatalog",{name:e.name})}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:[p("claudeCodePlugins.skillDetail.usage.notConfigured")," ",(0,t.jsx)("span",{onClick:()=>f("setup"),className:"cursor-pointer text-info",children:p("claudeCodePlugins.skillDetail.usage.seeSetup")})]})]}),"setup"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:p("claudeCodePlugins.skillDetail.setup.title")}),b?(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.setup.removeFirst")}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace remove litellm"})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(n.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.requiresKey",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"}),placeholder:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:u.i})}})})]}):(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:p("claudeCodePlugins.skillDetail.setup.runCommand")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:p("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;j(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===x?"text-success":"text-info"),children:["marketplace-cmd"===x?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),p("marketplace-cmd"===x?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(n.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.orSettings",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})}})})]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>j(v,"settings"),className:(0,c.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===x?"text-success":"text-info"),children:["settings"===x?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(r.f,{className:"size-3"}),p("settings"===x?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:v})]})]})]})}])},402874,e=>{var t=e.i(843476),s=e.i(143488),i=e.i(912089),a=e.i(636772),r=e.i(283713),l=e.i(602869),n=e.i(782066),o=e.i(275144),d=e.i(886400),c=e.i(321836),u=e.i(487486),m=e.i(972518),p=e.i(799647),g=e.i(522016),f=e.i(251773),x=e.i(423680),h=e.i(771243),b=e.i(196631),j=e.i(895335),_=e.i(641141),v=e.i(455880),w=e.i(853295),y=e.i(383862);let N="h-auto max-h-full w-auto max-w-full object-contain";e.s(["f",0,({accessToken:e,isPublicPage:k=!1,sidebarCollapsed:$=!1,onToggleSidebar:C,deploymentHealth:P})=>{let z=(0,l._h)(),{logoUrl:L,logoUrlDark:S}=(0,o.i)(),{data:D}=(0,s.f)(e),I=D?.litellm_version,A=(0,i.f)(),E=(0,a.f)(),{isControlPlane:M,selectedWorker:O}=(0,r.f)(),T=M&&null!==O,B=L||`${z}/get_image`,R=S||L||`${z}/get_image?theme=dark`,H=(0,d.r)(e);return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex h-14 items-center px-4",children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[C&&(0,t.jsx)("button",{onClick:C,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:$?"Expand sidebar":"Collapse sidebar",children:(0,t.jsx)("span",{className:"text-lg",children:$?(0,t.jsx)(p.f,{className:"size-[18px]"}):(0,t.jsx)(m.f,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.default,{href:(0,n.B)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:"flex h-10 max-w-48 items-center justify-center overflow-hidden",children:[(0,t.jsx)("img",{src:B,alt:"LiteLLM Brand",className:(0,b.c)(N,"dark:hidden")}),(0,t.jsx)("img",{src:R,alt:"","aria-hidden":!0,className:(0,b.c)(N,"hidden dark:block")})]})})}),I&&(0,t.jsxs)("div",{className:"relative",children:[!A&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(u.f,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",I]})})]})]})]}),!k&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(w.f,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[T&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(y.f,{onWorkerSwitch:t=>{(0,d.v)(e).finally(()=>{window.location.href=`${(0,c.D)()}?worker=${encodeURIComponent(t)}`})}})}),(0,t.jsxs)("nav",{"aria-label":"Product documentation",className:`flex min-w-0 items-center gap-2 ${T?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(x.f,{}),(0,t.jsx)(f.f,{})]}),!E&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(h.f,{})}),!k&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(v.f,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),P,(0,t.jsx)(j.f,{accessToken:e}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(_.f,{onLogout:H})]})})]})]})})})}])},747862,e=>{var t=e.i(843476);e.i(176782),e.i(911825);var s=e.i(225913),i=e.i(196631);e.i(772436);let a=(0,s.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:s,...r}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":s,className:(0,i.c)(a({orientation:s}),e),...r})}])},283713,e=>{var t=e.i(271645),s=e.i(602869),i=e.i(612256);let a="litellm_selected_worker_id";e.s(["f",0,()=>{let{data:e}=(0,i.f)(),r=e?.is_control_plane??!1,l=e?.workers??[],[n,o]=(0,t.useState)(()=>localStorage.getItem(a));(0,t.useEffect)(()=>{if(!n||0===l.length)return;let e=l.find(e=>e.worker_id===n);e&&(0,s.NZ)(e.url)},[n,l]);let d=l.find(e=>e.worker_id===n)??null,c=(0,t.useCallback)(e=>{let t=l.find(t=>t.worker_id===e);t&&(o(e),localStorage.setItem(a,e),(0,s.NZ)(t.url))},[l]);return{isControlPlane:r,workers:l,selectedWorkerId:n,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{o(null),localStorage.removeItem(a),(0,s.NZ)(null)},[])}}])},735450,e=>{var t=e.i(411382),s=e.i(772325),i=e.i(54105),a=e.i(240192),r=e.i(138620);let l="https://docs.litellm.ai/docs/decisions",n=`${l}#supported-providers`,o=`${l}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:i.z.string(),mode:i.z.string().nullish(),supported_endpoints:s.z.array(i.z.string()).nullish()}),u="decisions",m=["evaluation",u],p=e=>null!=e&&m.includes(e),g=e=>p(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),f=e=>{let t=a.z.record(i.z.string(),r.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let s=c.safeParse(t);return s.success?[[e,s.data]]:[]}):[]},x={decisions:0,"self-hosted":1,"needs-upgrade":2},h=t.z.object({decision_model_providers:s.z.array(i.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),j=(e,t,s)=>({row:e,routeOrder:x[e.route],rankInProvider:s.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,l,"d",0,u,"Z",0,n,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),s=t.filter(([,e])=>g(e)),i=new Map([...new Set(s.map(([,e])=>e.litellm_provider))].map(e=>{let s=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:s.every(([,e])=>g(e))}]}));return{models:new Set(s.map(([e])=>e)),providers:i}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,s])=>({model:e,provider:s.litellm_provider,route:t?.has(s.litellm_provider)?"decisions":(s.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(j).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>p(e)?u:e??void 0,"q",0,e=>e.flatMap(e=>e===u?m:[e]),"m",0,g,"_",0,p,"r",0,(e,t,s)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||s.some(t=>e.models.has(t)),"P",0,e=>{let t=h.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function s(e,s){let i=t(e);if(""===i)return!0;let a=s.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!a.some(e=>e.includes(i))||i.split(/\s+/).every(e=>a.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,i){return e.filter(e=>s(t,i(e)))},"h",0,s,"e",0,function(e,s,i){let a=t(s);if(""===a)return[...e];let r=e=>{let t=i(e).toLowerCase();return 1e3*(t===a)+100*!!t.startsWith(a)+(1e3-t.length)};return[...e].sort((e,t)=>r(t)-r(e))}])}])})();