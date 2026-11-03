(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,350682,e=>{let t=(0,e.i(475254).f)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]);e.s(["f",0,t],350682)},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},164668,e=>{var t=e.i(717521);e.s(["f",()=>t.f])},492941,e=>{var t=e.i(602869),s=e.i(266027);async function r(){let e=(0,t._h)(),s=await fetch(`${e}/public/litellm_blog_posts`);if(!s.ok)throw Error(`Failed to fetch blog posts: ${s.statusText}`);return s.json()}e.s(["f",0,()=>(0,s.f)({queryKey:["blogPosts"],queryFn:r,staleTime:36e5,retry:1,retryDelay:0})])},731565,e=>{var t=e.i(115571),s=e.i(271645);function r(e){let s=t=>{"disableBlogPosts"===t.key&&e()},r=t=>{let{key:s}=t.detail;"disableBlogPosts"===s&&e()};return window.addEventListener("storage",s),window.addEventListener(t.M,r),()=>{window.removeEventListener("storage",s),window.removeEventListener(t.M,r)}}function a(){return"true"===(0,t.t)("disableBlogPosts")}e.s(["f",0,function(){return(0,s.useSyncExternalStore)(r,a,()=>!1)}])},556260,319684,e=>{var t=e.i(271645),s=e.i(602869),r=e.i(115571);function a(e){return window.addEventListener("storage",e),window.addEventListener(r.M,e),()=>{window.removeEventListener("storage",e),window.removeEventListener(r.M,e)}}e.s(["f",0,function(e){let i=e?`disableLiteAdmin:${JSON.stringify([(0,s._h)(),e])}`:null;return[(0,t.useSyncExternalStore)(a,()=>null!==i&&"true"===(0,r.t)(i),()=>!1),e=>{null!==i&&(e?(0,r.A)(i,"true"):(0,r.p)(i),(0,r.s)(i))}]}],556260),e.s(["f",0,"Aria Agent"],319684)},251773,423680,771243,895335,e=>{var t=e.i(843476),s=e.i(731565),r=e.i(492941);let a="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var i=e.i(519455),n=e.i(755146),l=e.i(664659),o=e.i(164668);e.s(["f",0,()=>{let e=(0,s.f)(),{data:d,isLoading:c,isError:m,refetch:p}=(0,r.f)();return e?null:(0,t.jsxs)(n.G,{modal:!1,children:[(0,t.jsxs)(n.X,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(i.K,{variant:"ghost",className:`${a} border-0!`}),children:["Blog",(0,t.jsx)(l.f,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(n.L,{align:"end",side:"bottom",className:"w-auto",children:c?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(o.f,{role:"img","aria-label":"loading",className:"size-4 animate-spin"})}):m?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:"Failed to load posts"}),(0,t.jsx)(i.K,{variant:"outline",size:"sm",onClick:()=>p(),children:"Retry"})]}):d&&0!==d.posts.length?(0,t.jsxs)(t.Fragment,{children:[d.posts.slice(0,5).map(e=>(0,t.jsx)(n.r,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(n.U,{}),(0,t.jsx)(n.r,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:"No posts available"})})]})}],251773);let d=()=>(0,t.jsx)(l.f,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0});e.s(["f",0,()=>(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:a,children:["Docs",(0,t.jsx)(d,{})]})],423680);var c=e.i(636772),m=e.i(747862),p=e.i(746798),u=e.i(196631),g=e.i(350682);let f=[{href:"https://www.litellm.ai/support",label:"Join Slack",tooltip:"LiteLLM Slack community",Icon:(0,e.i(475254).f)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]])},{href:"https://github.com/BerriAI/litellm",label:"LiteLLM on GitHub",tooltip:"LiteLLM on GitHub",Icon:g.f}];e.s(["f",0,()=>(0,c.f)()?null:(0,t.jsx)(p.g,{children:(0,t.jsx)(m.f,{"aria-label":"Community links",children:f.map(({href:e,label:s,tooltip:r,Icon:a})=>(0,t.jsxs)(p.W,{children:[(0,t.jsx)(p.y,{render:(0,t.jsx)("a",{href:e,target:"_blank",rel:"noopener noreferrer","aria-label":s,className:(0,u.c)((0,i.i)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(a,{})}),(0,t.jsx)(p.Y,{children:r})]},e))})})],771243);var h=e.i(143488),x=e.i(768371),b=e.i(271645),_=e.i(115571);function j(e){return window.addEventListener("storage",e),window.addEventListener(_.M,e),()=>{window.removeEventListener("storage",e),window.removeEventListener(_.M,e)}}var v=e.i(487486),w=e.i(337822),y=e.i(772436);let N=/^v?(\d+)\.(\d+)\.(\d+)/,k=e=>{let t=N.exec(e.trim());return t?[Number(t[1]),Number(t[2]),Number(t[3])]:null};var $=e.i(245423);let z={readKey:"litellmHideAutoRouterAnnouncement",title:"LiteLLM Auto Router",description:"Route every request to the cheapest model that can handle it, no prompt changes needed.",linkLabel:"Read the docs",href:"https://docs.litellm.ai/docs/proxy/auto_routing"},C=(e,t,s)=>`${e} ${1===e?t:s}`,S=({announcement:e})=>{var s;let r=(s=e.readKey,(0,b.useSyncExternalStore)(j,()=>"true"===(0,_.t)(s),()=>!1));return(0,t.jsxs)("article",{"aria-label":e.title,className:"flex flex-col gap-2",children:[(0,t.jsx)("p",{className:"text-sm font-medium",children:e.title}),(0,t.jsx)("p",{className:"text-sm leading-snug text-muted-foreground",children:e.description}),(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsx)("a",{className:(0,u.c)((0,i.i)({size:"sm"})),href:e.href,target:"_blank",rel:"noopener noreferrer",children:e.linkLabel}),r?null:(0,t.jsx)(i.K,{variant:"link",size:"sm",className:"px-1!",onClick:()=>{var t;return t=e.readKey,void((0,_.A)(t,"true"),(0,_.s)(t))},children:"Mark as read"})]})]})},L=({currentVersion:e,latestRelease:s})=>{var r;let a=e&&s&&((e,t)=>{let s=k(e),r=k(t);if(!s||!r)return!1;for(let e=0;e<3;e+=1)if(r[e]!==s[e])return r[e]>s[e];return!1})(e,s.version)?{readKey:`litellm:upgradeBannerDismissed:${s.version}`,title:`LiteLLM v${s.version} is available`,description:`${(({new_features:e,bug_fixes:t,other_updates:s})=>[C(e,"new feature","new features"),C(t,"fix","fixes"),`and ${C(s,"other update","other updates")}`].join(", "))(s)}. Your current version is v${e}`,linkLabel:"View release notes",href:s.release_url}:null,i=a?[a,z]:[z],n=(r=i.map(e=>e.readKey),(0,b.useSyncExternalStore)(j,()=>r.filter(e=>"true"!==(0,_.t)(e)).length,()=>r.length));return(0,t.jsxs)(w.V,{children:[(0,t.jsx)(w.J,{className:"flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!","aria-label":n>0?`Notifications, ${n} unread`:"Notifications",children:(0,t.jsxs)("span",{className:"relative inline-flex",children:[(0,t.jsx)($.f,{className:"size-4","aria-hidden":!0}),n>0?(0,t.jsx)(v.f,{className:"absolute -top-0.5 -right-1 size-1.5 p-0","aria-hidden":!0}):null]})}),(0,t.jsx)(w.T,{align:"end","aria-label":"Notifications",children:i.map((e,s)=>(0,t.jsxs)(b.Fragment,{children:[s>0?(0,t.jsx)(y.f,{}):null,(0,t.jsx)(S,{announcement:e})]},e.readKey))})]})};e.s(["f",0,({accessToken:e})=>{let{data:s}=(0,h.f)(e),{data:r}=x.q.useQuery("get","/get/latest_release_info",{},{enabled:!!e,staleTime:36e5,retry:!1});return(0,t.jsx)(L,{currentVersion:s?.litellm_version,latestRelease:r})}],895335)},641141,e=>{var t=e.i(843476),s=e.i(135214),r=e.i(731565),a=e.i(912089),i=e.i(556260),n=e.i(319684),l=e.i(636772),o=e.i(115571),d=e.i(222038),c=e.i(782066),m=e.i(708347),p=e.i(664659),u=e.i(344523),g=e.i(243553),f=e.i(465261),h=e.i(292270),x=e.i(263488),b=e.i(581418),_=e.i(284614),j=e.i(618566),v=e.i(799676),w=e.i(487486),y=e.i(337822),N=e.i(772436),k=e.i(699375),$=e.i(746798),z=e.i(922407),C=e.i(196631),S=e.i(271645);e.s(["f",0,({onLogout:e,variant:L="navbar",collapsed:I=!1})=>{let{userId:A,userEmail:E,userRole:P,userRoleLabel:O,isViewOnly:M,premiumUser:T,loginMethod:R}=(0,s.f)(),B=(0,j.useRouter)(),[D,H]=(0,S.useState)(!1),U=(0,l.f)(),F=(0,r.f)(),K=(0,a.f)(),[q,Z]=(0,i.f)(A),Y=A&&!M&&(0,m.u)(P),[G,J]=(0,S.useState)(!1);(0,S.useEffect)(()=>{J("true"===(0,o.t)("disableShowNewBadge"))},[]);let V=E||A||"user",W=function(e,t){let s=e?.split("@")[0]?.trim();if(s){let e=s.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(E,A),Q=function(e){let t=0;for(let s=0;s<e.length;s+=1)t=e.charCodeAt(s)+((t<<5)-t);return Math.abs(t)%360}(V),X=(0,d.f)(E,A);return(0,t.jsxs)(y.V,{open:D,onOpenChange:H,children:["sidebar"===L?(0,t.jsxs)(y.J,{render:(0,t.jsx)("button",{type:"button",className:(0,C.c)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",I?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":`Account menu — ${O??"Unknown role"} — signed in as ${E||A||"unknown"}`,"aria-haspopup":"dialog",title:I?X:void 0}),children:[(0,t.jsx)(v.E,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(v.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${Q} 46% 38%)`},children:W})}),!I&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:X}),O&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:O})]}),(0,t.jsx)(u.f,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(y.J,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":`Account menu — ${O??"Unknown role"} — signed in as ${E||A||"unknown"}`,"aria-haspopup":"dialog"}),children:[(0,t.jsx)(v.E,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(v.b,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${Q} 46% 38%)`},children:W})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:X}),(0,t.jsx)(p.f,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(y.T,{align:"sidebar"===L?"start":"end",side:"sidebar"===L?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(x.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:E||"-"})]}),T?(0,t.jsxs)(w.f,{children:[(0,t.jsx)(g.f,{className:"size-3"}),"Premium"]}):(0,t.jsx)($.g,{children:(0,t.jsxs)($.W,{children:[(0,t.jsxs)($.y,{render:(0,t.jsx)(w.f,{variant:"outline"}),children:[(0,t.jsx)(g.f,{className:"size-3"}),"Standard"]}),(0,t.jsx)($.Y,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(N.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(_.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:A||"-",children:A||"-"}),(0,t.jsx)(z.f,{value:A,label:"Copy User ID"})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(b.f,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{children:O})]}),(0,t.jsx)(N.f,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide New Feature Indicators"}),(0,t.jsx)(k.f,{size:"sm",checked:G,onCheckedChange:e=>{J(e),e?(0,o.A)("disableShowNewBadge","true"):(0,o.p)("disableShowNewBadge"),(0,o.s)("disableShowNewBadge")},"aria-label":"Toggle hide new feature indicators"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide All Prompts"}),(0,t.jsx)(k.f,{size:"sm",checked:U,onCheckedChange:e=>{e?(0,o.A)("disableShowPrompts","true"):(0,o.p)("disableShowPrompts"),(0,o.s)("disableShowPrompts")},"aria-label":"Toggle hide all prompts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Blog Posts"}),(0,t.jsx)(k.f,{size:"sm",checked:F,onCheckedChange:e=>{e?(0,o.A)("disableBlogPosts","true"):(0,o.p)("disableBlogPosts"),(0,o.s)("disableBlogPosts")},"aria-label":"Toggle hide blog posts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Bouncing Icon"}),(0,t.jsx)(k.f,{size:"sm",checked:K,onCheckedChange:e=>{e?(0,o.A)("disableBouncingIcon","true"):(0,o.p)("disableBouncingIcon"),(0,o.s)("disableBouncingIcon")},"aria-label":"Toggle hide bouncing icon"})]}),!0===T&&Y&&(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("span",{className:"text-muted-foreground",children:["Hide ",n.f]}),(0,t.jsx)(k.f,{size:"sm",checked:q,onCheckedChange:Z,"aria-label":`Toggle hide ${n.f}`})]})]}),(0,t.jsx)(N.f,{}),"username_password"===R&&(0,t.jsxs)("button",{type:"button",onClick:()=>{H(!1),B.push((0,c.B)("change-password"))},className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(f.f,{className:"size-4"}),"Change Password"]}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(h.f,{className:"size-4"}),"Logout"]})]})]})}])},383862,e=>{var t=e.i(843476),s=e.i(618393),r=e.i(131792),a=e.i(950594),i=e.i(283713);e.s(["f",0,({onWorkerSwitch:e})=>{let{isControlPlane:n,selectedWorker:l,workers:o}=(0,i.f)();if(!n||!l)return null;let d=o.map(e=>({label:e.name,value:e.worker_id,disabled:e.worker_id===l.worker_id}));return(0,t.jsxs)(r.x,{items:d,value:d.find(e=>e.value===l.worker_id)??null,itemToStringLabel:e=>e.label,onValueChange:t=>{t&&e(t.value)},children:[(0,t.jsx)(r.c,{className:"min-w-[180px]","aria-label":"Worker",children:(0,t.jsx)(a.B,{align:"inline-start",children:(0,t.jsx)(s.f,{className:"size-4"})})}),(0,t.jsxs)(r.R,{children:[(0,t.jsx)(r.S,{children:"No matching workers"}),(0,t.jsx)(r.k,{children:e=>(0,t.jsx)(r.T,{value:e,disabled:e.disabled,children:e.label},e.value)})]})]})}])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let s,{apiKeySource:r,accessToken:a,apiKey:i,inputMessage:n,chatHistory:l,selectedTags:o,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:m,selectedVoice:p,endpointType:u,selectedModel:g,selectedSdk:f,proxySettings:h,customHeaders:x}=e,b="session"===r?a:i,_=window.location.origin,j=h?.LITELLM_UI_API_DOC_BASE_URL;j&&j.trim()?_=j:h?.PROXY_BASE_URL&&(_=h.PROXY_BASE_URL);let v=n||"Your prompt here",w=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),y=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),N={};o.length>0&&(N.tags=o),d.length>0&&(N.vector_stores=d),c.length>0&&(N.guardrails=c),m.length>0&&(N.policies=m);let k=g||"your-model-name",$=x&&Object.keys(x).length>0?`,
	default_headers=${JSON.stringify(x,null,2).replace(/\n/g,"\n	")}`:"",z="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${$}
)`;switch(u){case t.Q.CHAT:{let e=Object.keys(N).length>0,t="";if(e){let e=JSON.stringify({metadata:N},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=y.length>0?y:[{role:"user",content:v}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(r,null,4)}${t}
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
    extra_body=${e}`}let r=y.length>0?y:[{role:"user",content:v}];s=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(r,null,4)}${t}
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
	prompt="${n}",
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
	input="${n||"Your string here"}",
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
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:s=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${n||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:s="\n# Code generation for this endpoint is not implemented yet."}return`${z}
${s}`}])},652272,209261,e=>{var t=e.i(843476),s=e.i(271645),r=e.i(871689),a=e.i(643531),i=e.i(174886),n=e.i(306228),l=e.i(196631);let o=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,d=e=>e.trim().replace(/\/+$/,""),c=/\.(md|markdown|txt|json|ya?ml|toml)$/i,m=/\.zip$/i,p=/^[0-9a-fA-F]{64}$/,u=/^\d{1,3}(\.\d{1,3}){3}$/,g=/^[A-Za-z0-9-]+$/,f=/^[A-Za-z0-9._-]+$/,h=/^https?:\/\//i,x="ssh://",b=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,_=e=>e.pathname.split("/").filter(e=>""!==e),j=e=>{try{return new URL(e)}catch{return null}},v=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!u.test(e.hostname),w=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},y=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),N=(e,t,s,r)=>{let a=d(r??"");return""!==a?o.test(a)?{parsed:{source:"git-subdir",url:t,path:a},label:`${e} subdir — ${t} @ ${a}`,suggestedName:y(w(a))}:null:{parsed:{source:"url",url:t},label:`${e} repo — ${t}`,suggestedName:y(s)}},k=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),$=e=>`/plugin install ${e.name}@litellm`,z=e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:"Unknown source",C=e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&h.test(e.url)?e.url:null;e.s(["U",0,k,"a",0,$,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"_",0,z,"E",0,C,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||p.test(e.trim()),"P",0,e=>{let t=d(e);return""!==t&&o.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,t)=>{let s=((e,t)=>{let s=e.trim(),r=b.exec(s),a=r?`${x}${r[1]}@${r[2]}/${r[3]}`:s;if(!a.toLowerCase().startsWith(x))return null;let i=j(a);if(!i||""===i.username||""!==i.password||!v(i))return null;let n=a.indexOf("/",x.length);return -1===n||i.pathname!==a.slice(n)||_(i).length<2?null:N("SSH",s,w(i.pathname).replace(/\.git$/i,""),t)})(e,t);if(s)return s;let r=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let s=j(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return s&&"https:"===s.protocol&&""===s.username&&""===s.password&&v(s)?s:null})(e);if(!r)return null;if(m.test(r.pathname))return{parsed:{source:"archive",url:r.href},label:`Zip archive — ${r.host}${r.pathname}`,suggestedName:y(w(r.pathname).replace(m,""))};if("github.com"===r.hostname.replace(/^www\./,""))return((e,t)=>{let s=_(e);if(s.length<2)return null;let r=s[0],a=s[1].replace(/\.git$/,"");if(!g.test(r)||!f.test(a))return null;let i=`${r}/${a}`,n=`https://github.com/${i}`,l={parsed:{source:"github",repo:i},label:`GitHub repo — ${i}`,suggestedName:y(a)};if(s.length>=4&&("tree"===s[2]||"blob"===s[2])){let e=s.slice(4),t=w(e.join("/")),r=c.test(t)?e.slice(0,-1):e;if(0===r.length)return l;let a=d(r.join("/"));return o.test(a)?{parsed:{source:"git-subdir",url:n,path:a},label:`GitHub subdir — ${i} @ ${a}`,suggestedName:y(w(a))}:null}if(2!==s.length)return null;let m=d(t??"");return""!==m?o.test(m)?{parsed:{source:"git-subdir",url:n,path:m},label:`GitHub subdir — ${i} @ ${m}`,suggestedName:y(w(m))}:null:l})(r,t);if(_(r).length<2)return null;let a=w(r.pathname).replace(/\.git$/,"");return N("Git",`${r.protocol}//${r.host}${r.pathname.replace(/\/+$/,"")}`,a,t)},"e",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261);let S=({source:e})=>{let s=C(e),r=s&&"git-subdir"===e.source&&e.path?`${s}/tree/main/${e.path}`:s;return r?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:r,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[r.replace("https://",""),(0,t.jsx)(n.f,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:z(e)})]}):null};e.s(["f",0,({skill:e,onBack:n})=>{let[o,d]=(0,s.useState)("overview"),[c,m]=(0,s.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},u=$(e),g=k(window.location.origin),f=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:n,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(r.f,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),className:(0,l.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",o===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===o&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:f.map((e,s)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},s))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,l.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),(0,t.jsx)(S,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===o&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(u,"install"),className:(0,l.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===c?"text-success":"text-info"),children:["install"===c?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(i.f,{className:"size-3"}),"install"===c?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:u})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,' not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===o&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;p(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,l.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===c?"text-success":"text-info"),children:["marketplace-cmd"===c?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(i.f,{className:"size-3"}),"marketplace-cmd"===c?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>p(g,"settings"),className:(0,l.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===c?"text-success":"text-info"),children:["settings"===c?(0,t.jsx)(a.f,{className:"size-3"}):(0,t.jsx)(i.f,{className:"size-3"}),"settings"===c?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:g})]})]})]})}],652272)},402874,e=>{var t=e.i(843476),s=e.i(143488),r=e.i(912089),a=e.i(636772),i=e.i(283713),n=e.i(602869),l=e.i(782066),o=e.i(275144),d=e.i(886400),c=e.i(321836),m=e.i(487486),p=e.i(972518),u=e.i(799647),g=e.i(522016),f=e.i(251773),h=e.i(423680),x=e.i(771243),b=e.i(196631),_=e.i(895335),j=e.i(641141),v=e.i(455880),w=e.i(853295),y=e.i(383862);let N="h-auto max-h-full w-auto max-w-full object-contain";e.s(["f",0,({accessToken:e,isPublicPage:k=!1,sidebarCollapsed:$=!1,onToggleSidebar:z,deploymentHealth:C})=>{let S=(0,n._h)(),{logoUrl:L,logoUrlDark:I}=(0,o.i)(),{data:A}=(0,s.f)(e),E=A?.litellm_version,P=(0,r.f)(),O=(0,a.f)(),{isControlPlane:M,selectedWorker:T}=(0,i.f)(),R=M&&null!==T,B=L||`${S}/get_image`,D=I||L||`${S}/get_image?theme=dark`,H=(0,d.r)(e);return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex h-14 items-center px-4",children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[z&&(0,t.jsx)("button",{onClick:z,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:$?"Expand sidebar":"Collapse sidebar",children:(0,t.jsx)("span",{className:"text-lg",children:$?(0,t.jsx)(u.f,{className:"size-[18px]"}):(0,t.jsx)(p.f,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.default,{href:(0,l.B)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:"flex h-10 max-w-48 items-center justify-center overflow-hidden",children:[(0,t.jsx)("img",{src:B,alt:"LiteLLM Brand",className:(0,b.c)(N,"dark:hidden")}),(0,t.jsx)("img",{src:D,alt:"","aria-hidden":!0,className:(0,b.c)(N,"hidden dark:block")})]})})}),E&&(0,t.jsxs)("div",{className:"relative",children:[!P&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(m.f,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",E]})})]})]})]}),!k&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(w.f,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[R&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(y.f,{onWorkerSwitch:t=>{(0,d.v)(e).finally(()=>{window.location.href=`${(0,c.D)()}?worker=${encodeURIComponent(t)}`})}})}),(0,t.jsxs)("nav",{"aria-label":"Product documentation",className:`flex min-w-0 items-center gap-2 ${R?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(h.f,{}),(0,t.jsx)(f.f,{})]}),!O&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(x.f,{})}),!k&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(v.f,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),C,(0,t.jsx)(_.f,{accessToken:e}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(j.f,{onLogout:H})]})})]})]})})})}])},747862,e=>{var t=e.i(843476);e.i(176782),e.i(911825);var s=e.i(225913),r=e.i(196631);e.i(772436);let a=(0,s.f)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});e.s(["f",0,function({className:e,orientation:s,...i}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":s,className:(0,r.c)(a({orientation:s}),e),...i})}])},283713,e=>{var t=e.i(271645),s=e.i(602869),r=e.i(612256);let a="litellm_selected_worker_id";e.s(["f",0,()=>{let{data:e}=(0,r.f)(),i=e?.is_control_plane??!1,n=e?.workers??[],[l,o]=(0,t.useState)(()=>localStorage.getItem(a));(0,t.useEffect)(()=>{if(!l||0===n.length)return;let e=n.find(e=>e.worker_id===l);e&&(0,s.NZ)(e.url)},[l,n]);let d=n.find(e=>e.worker_id===l)??null,c=(0,t.useCallback)(e=>{let t=n.find(t=>t.worker_id===e);t&&(o(e),localStorage.setItem(a,e),(0,s.NZ)(t.url))},[n]);return{isControlPlane:i,workers:n,selectedWorkerId:l,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{o(null),localStorage.removeItem(a),(0,s.NZ)(null)},[])}}])},735450,e=>{var t=e.i(411382),s=e.i(772325),r=e.i(54105),a=e.i(240192),i=e.i(138620);let n="https://docs.litellm.ai/docs/decisions",l=`${n}#supported-providers`,o=`${n}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),c=t.z.object({litellm_provider:r.z.string(),mode:r.z.string().nullish(),supported_endpoints:s.z.array(r.z.string()).nullish()}),m="decisions",p=["evaluation",m],u=e=>null!=e&&p.includes(e),g=e=>u(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),f=e=>{let t=a.z.record(r.z.string(),i.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let s=c.safeParse(t);return s.success?[[e,s.data]]:[]}):[]},h={decisions:0,"self-hosted":1,"needs-upgrade":2},x=t.z.object({decision_model_providers:s.z.array(r.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),_=(e,t,s)=>({row:e,routeOrder:h[e.route],rankInProvider:s.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,n,"d",0,m,"Z",0,l,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),s=t.filter(([,e])=>g(e)),r=new Map([...new Set(s.map(([,e])=>e.litellm_provider))].map(e=>{let s=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:s.every(([,e])=>g(e))}]}));return{models:new Set(s.map(([e])=>e)),providers:r}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,s])=>({model:e,provider:s.litellm_provider,route:t?.has(s.litellm_provider)?"decisions":(s.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(_).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>u(e)?m:e??void 0,"q",0,e=>e.flatMap(e=>e===m?p:[e]),"m",0,g,"_",0,u,"r",0,(e,t,s)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||s.some(t=>e.models.has(t)),"P",0,e=>{let t=x.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function s(e,s){let r=t(e);if(""===r)return!0;let a=s.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!a.some(e=>e.includes(r))||r.split(/\s+/).every(e=>a.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,r){return e.filter(e=>s(t,r(e)))},"h",0,s,"e",0,function(e,s,r){let a=t(s);if(""===a)return[...e];let i=e=>{let t=r(e).toLowerCase();return 1e3*(t===a)+100*!!t.startsWith(a)+(1e3-t.length)};return[...e].sort((e,t)=>i(t)-i(e))}])}])})();