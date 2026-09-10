(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},62478,e=>{"use strict";var t=e.i(602869);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},592392,e=>{"use strict";var t=e.i(62478),a=e.i(266027);let s=(0,e.i(243652).createQueryKeys)("proxySettings"),r={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};e.s(["default",0,function(e){let{data:i}=(0,a.useQuery)({queryKey:[...s.all,e],queryFn:()=>(0,t.fetchProxySettings)(e),enabled:!!e});return i??r}])},283713,e=>{"use strict";var t=e.i(271645),a=e.i(602869),s=e.i(612256);let r="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,s.useUIConfig)(),i=e?.is_control_plane??!1,n=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(r));(0,t.useEffect)(()=>{if(!o||0===n.length)return;let e=n.find(e=>e.worker_id===o);e&&(0,a.switchToWorkerUrl)(e.url)},[o,n]);let d=n.find(e=>e.worker_id===o)??null,c=(0,t.useCallback)(e=>{let t=n.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(r,e),(0,a.switchToWorkerUrl)(t.url))},[n]);return{isControlPlane:i,workers:n,selectedWorkerId:o,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(r),(0,a.switchToWorkerUrl)(null)},[])}}])},251773,423680,771243,895335,e=>{"use strict";var t=e.i(843476),a=e.i(731565),s=e.i(602869),r=e.i(266027);async function i(){let e=(0,s.getProxyBaseUrl)(),t=await fetch(`${e}/public/litellm_blog_posts`);if(!t.ok)throw Error(`Failed to fetch blog posts: ${t.statusText}`);return t.json()}let n="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-foreground outline-none transition-colors hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50 ";var o=e.i(519455),l=e.i(755146),d=e.i(664659),c=e.i(164668);e.s(["BlogDropdown",0,()=>{let e=(0,a.useDisableBlogPosts)(),{data:s,isLoading:m,isError:p,refetch:u}=(0,r.useQuery)({queryKey:["blogPosts"],queryFn:i,staleTime:36e5,retry:1,retryDelay:0});return e?null:(0,t.jsxs)(l.DropdownMenu,{modal:!1,children:[(0,t.jsxs)(l.DropdownMenuTrigger,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(o.Button,{variant:"ghost",className:`${n} border-0!`}),children:["Blog",(0,t.jsx)(d.ChevronDown,{className:"size-2.5 text-muted-foreground","aria-hidden":!0})]}),(0,t.jsx)(l.DropdownMenuContent,{align:"end",side:"bottom",className:"w-auto",children:m?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(c.LoaderCircle,{role:"img","aria-label":"loading",className:"size-4 animate-spin"})}):p?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:"Failed to load posts"}),(0,t.jsx)(o.Button,{variant:"outline",size:"sm",onClick:()=>u(),children:"Retry"})]}):s&&0!==s.posts.length?(0,t.jsxs)(t.Fragment,{children:[s.posts.slice(0,5).map(e=>(0,t.jsx)(l.DropdownMenuItem,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(l.DropdownMenuSeparator,{}),(0,t.jsx)(l.DropdownMenuItem,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:"No posts available"})})]})}],251773);let m=()=>(0,t.jsx)(d.ChevronDown,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0});e.s(["DocsLink",0,()=>(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:n,children:["Docs",(0,t.jsx)(m,{})]})],423680);var p=e.i(636772);e.i(176782),e.i(911825);var u=e.i(225913),g=e.i(196631);e.i(772436);let h=(0,u.cva)("flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-raised has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",{variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});function x({className:e,orientation:a,...s}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":a,className:(0,g.cn)(h({orientation:a}),e),...s})}var f=e.i(746798),b=e.i(475254);let _=(0,b.default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]),j=[{href:"https://www.litellm.ai/support",label:"Join Slack",tooltip:"LiteLLM Slack community",Icon:(0,b.default)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]])},{href:"https://github.com/BerriAI/litellm",label:"LiteLLM on GitHub",tooltip:"LiteLLM on GitHub",Icon:_}];e.s(["CommunityEngagementButtons",0,()=>(0,p.useDisableShowPrompts)()?null:(0,t.jsx)(f.TooltipProvider,{children:(0,t.jsx)(x,{"aria-label":"Community links",children:j.map(({href:e,label:a,tooltip:s,Icon:r})=>(0,t.jsxs)(f.Tooltip,{children:[(0,t.jsx)(f.TooltipTrigger,{render:(0,t.jsx)("a",{href:e,target:"_blank",rel:"noopener noreferrer","aria-label":a,className:(0,g.cn)((0,o.buttonVariants)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(r,{})}),(0,t.jsx)(f.TooltipContent,{children:s})]},e))})})],771243);var y=e.i(271645),w=e.i(115571);let v="litellmHideAutoRouterAnnouncement";function N(e){let t=t=>{t.key===v&&e()},a=t=>{let{key:a}=t.detail;a===v&&e()};return window.addEventListener("storage",t),window.addEventListener(w.LOCAL_STORAGE_EVENT,a),()=>{window.removeEventListener("storage",t),window.removeEventListener(w.LOCAL_STORAGE_EVENT,a)}}function k(){return"true"===(0,w.getLocalStorageItem)(v)}var C=e.i(487486),S=e.i(337822),I=e.i(245423);e.s(["NotificationsBell",0,()=>{let e=!(0,y.useSyncExternalStore)(N,k),[a,s]=(0,y.useState)(!1),r=(0,t.jsxs)("div",{className:"max-w-[280px]",children:[(0,t.jsx)(S.PopoverTitle,{className:"mt-0! mb-2!",children:"LiteLLM Auto Router"}),(0,t.jsx)(S.PopoverDescription,{className:"mb-3! text-sm leading-snug",children:"Route every request to the cheapest model that can handle it, no prompt changes needed."}),(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsx)("a",{className:(0,g.cn)((0,o.buttonVariants)({size:"sm"})),href:"https://docs.litellm.ai/docs/proxy/auto_routing",target:"_blank",rel:"noopener noreferrer",children:"Read the docs"}),e?(0,t.jsx)(o.Button,{variant:"link",size:"sm",className:"px-1!",onClick:()=>{(0,w.setLocalStorageItem)(v,"true"),(0,w.emitLocalStorageChange)(v),s(!1)},children:"Mark as read"}):null]})]});return(0,t.jsxs)(S.Popover,{open:a,onOpenChange:s,children:[(0,t.jsx)(S.PopoverTrigger,{className:"flex! h-9! w-9! items-center justify-center rounded-md! text-muted-foreground transition-colors hover:bg-accent! hover:text-foreground!","aria-label":"Notifications",children:(0,t.jsxs)("span",{className:"relative inline-flex",children:[(0,t.jsx)(I.Bell,{className:"size-4","aria-hidden":!0}),e?(0,t.jsx)(C.Badge,{className:"absolute -top-0.5 -right-1 size-1.5 p-0","aria-hidden":!0}):null]})}),(0,t.jsx)(S.PopoverContent,{align:"end",children:r})]})}],895335)},641141,e=>{"use strict";var t=e.i(843476),a=e.i(135214),s=e.i(731565),r=e.i(912089),i=e.i(636772),n=e.i(115571),o=e.i(222038),l=e.i(664659),d=e.i(344523),c=e.i(243553),m=e.i(292270),p=e.i(263488),u=e.i(581418),g=e.i(284614),h=e.i(799676),x=e.i(487486),f=e.i(337822),b=e.i(772436),_=e.i(699375),j=e.i(746798),y=e.i(922407),w=e.i(196631),v=e.i(271645);e.s(["default",0,({onLogout:e,variant:N="navbar",collapsed:k=!1})=>{let{userId:C,userEmail:S,userRoleLabel:I,premiumUser:L}=(0,a.default)(),$=(0,i.useDisableShowPrompts)(),E=(0,s.useDisableBlogPosts)(),A=(0,r.useDisableBouncingIcon)(),[T,P]=(0,v.useState)(!1);(0,v.useEffect)(()=>{P("true"===(0,n.getLocalStorageItem)("disableShowNewBadge"))},[]);let z=S||C||"user",D=function(e,t){let a=e?.split("@")[0]?.trim();if(a){let e=a.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(S,C),M=function(e){let t=0;for(let a=0;a<e.length;a+=1)t=e.charCodeAt(a)+((t<<5)-t);return Math.abs(t)%360}(z),O=(0,o.navAccountDisplayName)(S,C);return(0,t.jsxs)(f.Popover,{children:["sidebar"===N?(0,t.jsxs)(f.PopoverTrigger,{render:(0,t.jsx)("button",{type:"button",className:(0,w.cn)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",k?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":`Account menu — ${I??"Unknown role"} — signed in as ${S||C||"unknown"}`,"aria-haspopup":"dialog",title:k?O:void 0}),children:[(0,t.jsx)(h.Avatar,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(h.AvatarFallback,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${M} 46% 38%)`},children:D})}),!k&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:O}),I&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:I})]}),(0,t.jsx)(d.ChevronsUpDown,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(f.PopoverTrigger,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-accent!","aria-label":`Account menu — ${I??"Unknown role"} — signed in as ${S||C||"unknown"}`,"aria-haspopup":"dialog"}),children:[(0,t.jsx)(h.Avatar,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(h.AvatarFallback,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${M} 46% 38%)`},children:D})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-foreground md:inline",children:O}),(0,t.jsx)(l.ChevronDown,{className:"hidden size-2.5 shrink-0 text-muted-foreground md:inline","aria-hidden":!0})]}),(0,t.jsxs)(f.PopoverContent,{align:"sidebar"===N?"start":"end",side:"sidebar"===N?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-card p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(p.Mail,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:S||"-"})]}),L?(0,t.jsxs)(x.Badge,{children:[(0,t.jsx)(c.Crown,{className:"size-3"}),"Premium"]}):(0,t.jsx)(j.TooltipProvider,{children:(0,t.jsxs)(j.Tooltip,{children:[(0,t.jsxs)(j.TooltipTrigger,{render:(0,t.jsx)(x.Badge,{variant:"outline"}),children:[(0,t.jsx)(c.Crown,{className:"size-3"}),"Standard"]}),(0,t.jsx)(j.TooltipContent,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(b.Separator,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.User,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:C||"-",children:C||"-"}),(0,t.jsx)(y.default,{value:C,label:"Copy User ID"})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(u.ShieldCheck,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{children:I})]}),(0,t.jsx)(b.Separator,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide New Feature Indicators"}),(0,t.jsx)(_.Switch,{size:"sm",checked:T,onCheckedChange:e=>{P(e),e?(0,n.setLocalStorageItem)("disableShowNewBadge","true"):(0,n.removeLocalStorageItem)("disableShowNewBadge"),(0,n.emitLocalStorageChange)("disableShowNewBadge")},"aria-label":"Toggle hide new feature indicators"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide All Prompts"}),(0,t.jsx)(_.Switch,{size:"sm",checked:$,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableShowPrompts","true"):(0,n.removeLocalStorageItem)("disableShowPrompts"),(0,n.emitLocalStorageChange)("disableShowPrompts")},"aria-label":"Toggle hide all prompts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Blog Posts"}),(0,t.jsx)(_.Switch,{size:"sm",checked:E,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableBlogPosts","true"):(0,n.removeLocalStorageItem)("disableBlogPosts"),(0,n.emitLocalStorageChange)("disableBlogPosts")},"aria-label":"Toggle hide blog posts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Bouncing Icon"}),(0,t.jsx)(_.Switch,{size:"sm",checked:A,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableBouncingIcon","true"):(0,n.removeLocalStorageItem)("disableBouncingIcon"),(0,n.emitLocalStorageChange)("disableBouncingIcon")},"aria-label":"Toggle hide bouncing icon"})]})]}),(0,t.jsx)(b.Separator,{}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(m.LogOut,{className:"size-4"}),"Logout"]})]})]})}])},455880,e=>{"use strict";var t=e.i(843476),a=e.i(475254);let s=(0,a.default)("moon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]),r=(0,a.default)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);var i=e.i(363178),n=e.i(519455);e.s(["default",0,()=>{let{setTheme:e,resolvedTheme:a}=(0,i.useTheme)(),o="dark"===a,l=o?"Switch to light mode":"Switch to dark mode (beta)";return(0,t.jsx)(n.Button,{variant:"ghost",size:"icon-sm","aria-label":l,title:l,className:"text-muted-foreground",onClick:()=>e(o?"light":"dark"),children:o?(0,t.jsx)(s,{}):(0,t.jsx)(r,{})})}],455880)},853295,658140,e=>{"use strict";var t=e.i(843476),a=e.i(618566),s=e.i(755146),r=e.i(643531),i=e.i(344523),n=e.i(373264),o=e.i(271645),l=e.i(431703),d=e.i(602869);let c=(0,o.createContext)({mode:"ai-gateway",setMode:()=>{},plugins:[],activePlugin:null}),m="litellm_plugin_mode",p=(0,l.createApiClient)({getBaseUrl:()=>(0,d.getProxyBaseUrl)()??""});function u(){return localStorage.getItem(m)??"ai-gateway"}function g(){return(0,o.useContext)(c)}e.s(["PluginModeProvider",0,function({children:e,accessToken:a}){let[s,r]=(0,o.useState)(u),[i,n]=(0,o.useState)([]),[l,d]=(0,o.useState)(!1);(0,o.useEffect)(()=>{a&&p.get("/api/plugins",{accessToken:a}).then(e=>{n(Array.isArray(e)?e:[])}).catch(()=>{}).finally(()=>d(!0))},[a]);let g="ai-gateway"!==s&&l&&!i.some(e=>e.name===s)?"ai-gateway":s,h=i.find(e=>e.name===g)??null;return(0,t.jsx)(c.Provider,{value:{mode:g,setMode:e=>{r(e),localStorage.setItem(m,e)},plugins:i,activePlugin:h},children:e})},"usePluginMode",0,g],658140);var h=e.i(292639),x=e.i(571353);let f="chat";e.s(["default",0,function(){let{mode:e,setMode:o,plugins:l}=g(),{data:d}=(0,h.useUISettings)(),c=(0,a.usePathname)(),m=!!d?.values?.enable_chat_ui,p=(0,x.migratedHref)(f),u=(c??"").replace(/\/+$/,""),b=m&&(u===p||u.startsWith(`${p}/`)),_=b?"Chat":l.find(t=>t.name===e)?.display_name??"AI Gateway",j=[{key:"ai-gateway",label:"AI Gateway"},...l.map(e=>({key:e.name,label:e.display_name}))],y=m?{key:f,label:(0,t.jsxs)("div",{className:"flex items-center justify-between gap-6 py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:"Chat"}),b&&(0,t.jsx)(r.Check,{className:"size-4 text-info"})]}),onClick:()=>window.location.assign((0,x.migratedHref)(f))}:{key:f,disabled:!0,label:(0,t.jsxs)("div",{className:"flex max-w-[220px] flex-col py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:"Chat"}),(0,t.jsx)("span",{className:"whitespace-normal text-xs leading-snug text-muted-foreground",children:"Admins can enable in Settings"})]})},w=[...j.map(a=>({key:a.key,label:(0,t.jsxs)("div",{className:"flex items-center justify-between gap-6 py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:a.label}),!b&&a.key===e&&(0,t.jsx)(r.Check,{className:"size-4 text-info"})]}),onClick:()=>{o(a.key),b&&window.location.assign((0,x.migratedHref)(""))}})),y];return(0,t.jsxs)(s.DropdownMenu,{children:[(0,t.jsxs)(s.DropdownMenuTrigger,{render:(0,t.jsx)("button",{type:"button",className:"flex h-8 max-w-[220px] items-center gap-1.5 rounded-md border border-border bg-background pl-1.5 pr-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"}),children:[(0,t.jsx)("span",{className:"flex size-5 flex-none items-center justify-center rounded bg-muted text-muted-foreground",children:(0,t.jsx)(n.LayoutGrid,{className:"size-[13px]"})}),(0,t.jsx)("span",{className:"truncate",children:_}),(0,t.jsx)(i.ChevronsUpDown,{className:"size-3.5 flex-none text-muted-foreground"})]}),(0,t.jsx)(s.DropdownMenuContent,{className:"w-auto",children:w.map(e=>(0,t.jsx)(s.DropdownMenuItem,{disabled:e.disabled,onClick:e.onClick,children:e.label},e.key))})]})}],853295)},383862,e=>{"use strict";var t=e.i(843476),a=e.i(618393),s=e.i(131792),r=e.i(950594),i=e.i(283713);e.s(["default",0,({onWorkerSwitch:e})=>{let{isControlPlane:n,selectedWorker:o,workers:l}=(0,i.useWorker)();if(!n||!o)return null;let d=l.map(e=>({label:e.name,value:e.worker_id,disabled:e.worker_id===o.worker_id}));return(0,t.jsxs)(s.Combobox,{items:d,value:d.find(e=>e.value===o.worker_id)??null,itemToStringLabel:e=>e.label,onValueChange:t=>{t&&e(t.value)},children:[(0,t.jsx)(s.ComboboxInput,{className:"min-w-[180px]","aria-label":"Worker",children:(0,t.jsx)(r.InputGroupAddon,{align:"inline-start",children:(0,t.jsx)(a.Server,{className:"size-4"})})}),(0,t.jsxs)(s.ComboboxContent,{children:[(0,t.jsx)(s.ComboboxEmpty,{children:"No matching workers"}),(0,t.jsx)(s.ComboboxList,{children:e=>(0,t.jsx)(s.ComboboxItem,{value:e,disabled:e.disabled,children:e.label},e.value)})]})]})}])},402874,e=>{"use strict";var t=e.i(843476),a=e.i(143488),s=e.i(912089),r=e.i(636772),i=e.i(283713),n=e.i(602869),o=e.i(571353),l=e.i(275144),d=e.i(268004),c=e.i(321836),m=e.i(592392),p=e.i(487486),u=e.i(972518),g=e.i(799647),h=e.i(522016),x=e.i(251773),f=e.i(423680),b=e.i(771243),_=e.i(196631),j=e.i(895335),y=e.i(641141),w=e.i(455880),v=e.i(853295),N=e.i(383862);let k="h-auto max-h-full w-auto max-w-full object-contain";e.s(["default",0,({accessToken:e,isPublicPage:C=!1,sidebarCollapsed:S=!1,onToggleSidebar:I})=>{let L=(0,n.getProxyBaseUrl)(),$=(0,m.default)(e),{logoUrl:E}=(0,l.useTheme)(),{data:A}=(0,a.useHealthReadinessDetails)(e),T=A?.litellm_version,P=(0,s.useDisableBouncingIcon)(),z=(0,r.useDisableShowPrompts)(),{isControlPlane:D,selectedWorker:M}=(0,i.useWorker)(),O=D&&null!==M,B=E||`${L}/get_image`,R=E||`${L}/get_image?theme=dark`;return(0,t.jsx)("nav",{className:"sticky top-0 z-chrome border-b border-border bg-card",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex h-14 items-center px-4",children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[I&&(0,t.jsx)("button",{onClick:I,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",title:S?"Expand sidebar":"Collapse sidebar",children:(0,t.jsx)("span",{className:"text-lg",children:S?(0,t.jsx)(g.PanelLeftOpen,{className:"size-[18px]"}):(0,t.jsx)(u.PanelLeftClose,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(h.default,{href:(0,o.migratedHref)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsxs)("div",{className:"flex h-10 max-w-48 items-center justify-center overflow-hidden",children:[(0,t.jsx)("img",{src:B,alt:"LiteLLM Brand",className:(0,_.cn)(k,"dark:hidden")}),(0,t.jsx)("img",{src:R,alt:"","aria-hidden":!0,className:(0,_.cn)(k,"hidden dark:block")})]})})}),T&&(0,t.jsxs)("div",{className:"relative",children:[!P&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(p.Badge,{variant:"outline",className:"relative z-raised cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",T]})})]})]})]}),!C&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(v.default,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[O&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(N.default,{onWorkerSwitch:e=>{(0,d.clearTokenCookies)(),(0,c.clearStoredReturnUrl)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=`${(0,c.getLoginUrl)()}?worker=${encodeURIComponent(e)}`}})}),(0,t.jsxs)("nav",{"aria-label":"Product documentation",className:`flex min-w-0 items-center gap-2 ${O?"border-l border-border pl-4":""}`,children:[(0,t.jsx)(f.DocsLink,{}),(0,t.jsx)(x.BlogDropdown,{})]}),!z&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsx)(b.CommunityEngagementButtons,{})}),!C&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-border pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-muted px-1 py-0 transition-colors hover:bg-accent",children:[(0,t.jsx)(w.default,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(j.NotificationsBell,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-border","aria-hidden":!0}),(0,t.jsx)(y.default,{onLogout:()=>{(0,d.clearTokenCookies)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=$.PROXY_LOGOUT_URL||""}})]})})]})]})})})}])},434626,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,a],434626)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},909947,865361,e=>{"use strict";var t,a,s=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.COMPLETION="completion",t.RESPONSES="responses",t.IMAGE_EDITS="image_edit",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t.REALTIME="realtime",t),r=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a.INTERACTIONS="interactions",a);let i={image_generation:"image",video_generation:"video",chat:"chat",completion:"chat",responses:"responses",image_edit:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings",realtime:"realtime"};e.s(["EndpointType",()=>r,"ModelMode",()=>s,"getEndpointType",0,e=>Object.values(s).includes(e)?i[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:s,apiKey:i,inputMessage:n,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:m,selectedVoice:p,endpointType:u,selectedModel:g,selectedSdk:h,proxySettings:x}=e,f="session"===a?s:i,b=window.location.origin,_=x?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?b=_:x?.PROXY_BASE_URL&&(b=x.PROXY_BASE_URL);let j=n||"Your prompt here",y=j.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),v={};l.length>0&&(v.tags=l),d.length>0&&(v.vector_stores=d),c.length>0&&(v.guardrails=c),m.length>0&&(v.policies=m);let N=g||"your-model-name",k="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${f||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${b}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${f||"YOUR_LITELLM_API_KEY"}",
	base_url="${b}"
)`;switch(u){case r.CHAT:{let e=Object.keys(v).length>0,a="";if(e){let e=JSON.stringify({metadata:v},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:j}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${N}",
    messages=${JSON.stringify(s,null,4)}${a}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${N}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${y}"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_file}"  # or data:application/pdf;base64,{base64_file}
#                     }
#                 }
#             ]
#         }
#     ]${a}
# )
# print(response_with_file)
`;break}case r.RESPONSES:{let e=Object.keys(v).length>0,a="";if(e){let e=JSON.stringify({metadata:v},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let s=w.length>0?w:[{role:"user",content:j}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${N}",
    input=${JSON.stringify(s,null,4)}${a}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${N}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${y}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===h?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${N}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
`;break;case r.IMAGE_EDITS:t="azure"===h?`
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
prompt = "${y}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${N}",
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
`;break;case r.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${N}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${N}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${N}",
	input="${n||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${N}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${k}
${t}`}],909947)},652272,209261,e=>{"use strict";var t=e.i(843476),a=e.i(271645),s=e.i(871689),r=e.i(643531),i=e.i(174886),n=e.i(306228),o=e.i(196631);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,d=e=>e.trim().replace(/\/+$/,""),c=/\.(md|markdown|txt|json|ya?ml|toml)$/i,m=/^\d{1,3}(\.\d{1,3}){3}$/,p=/^[A-Za-z0-9-]+$/,u=/^[A-Za-z0-9._-]+$/,g=e=>e.pathname.split("/").filter(e=>""!==e),h=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},x=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),f=e=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),b=e=>`/plugin install ${e.name}@litellm`;e.s(["buildMarketplaceSettingsSnippet",0,f,"formatInstallCommand",0,b,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=d(e);return""!==t&&l.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let a=(e=>{let t,a=e.trim();if(""===a||a.startsWith("//"))return null;let s=/^[a-z][a-z0-9+.-]*:\/\//i.test(a)?a:`https://${a}`;try{t=new URL(s)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||m.test(t.hostname)?null:t})(e);if(!a)return null;if("github.com"===a.hostname.replace(/^www\./,""))return((e,t)=>{let a=g(e);if(a.length<2)return null;let s=a[0],r=a[1].replace(/\.git$/,"");if(!p.test(s)||!u.test(r))return null;let i=`${s}/${r}`,n=`https://github.com/${i}`,o={parsed:{source:"github",repo:i},label:`GitHub repo — ${i}`,suggestedName:x(r)};if(a.length>=4&&("tree"===a[2]||"blob"===a[2])){let e=a.slice(4),t=h(e.join("/")),s=c.test(t)?e.slice(0,-1):e;if(0===s.length)return o;let r=d(s.join("/"));return l.test(r)?{parsed:{source:"git-subdir",url:n,path:r},label:`GitHub subdir — ${i} @ ${r}`,suggestedName:x(h(r))}:null}if(2!==a.length)return null;let m=d(t??"");return""!==m?l.test(m)?{parsed:{source:"git-subdir",url:n,path:m},label:`GitHub subdir — ${i} @ ${m}`,suggestedName:x(h(m))}:null:o})(a,t);if(g(a).length<2)return null;let s=`${a.protocol}//${a.host}${a.pathname.replace(/\/+$/,"")}`,r=d(t??"");return""!==r?l.test(r)?{parsed:{source:"git-subdir",url:s,path:r},label:`Git subdir — ${s} @ ${r}`,suggestedName:x(h(r))}:null:{parsed:{source:"url",url:s},label:`Git repo — ${s}`,suggestedName:x(h(a.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:l})=>{let d,[c,m]=(0,a.useState)("overview"),[p,u]=(0,a.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),u(t),setTimeout(()=>u(null),2e3)},h="github"===(d=e.source).source&&d.repo?`https://github.com/${d.repo}`:"git-subdir"===d.source&&d.url?d.path?`${d.url}/tree/main/${d.path}`:d.url:"url"===d.source&&d.url?d.url:null,x=b(e),_=f(window.location.origin),j=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:l,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(s.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>m(e.key),className:(0,o.cn)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",c===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===c&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:"Skill Details"}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:"Property"}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:j.map((e,a)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},a))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Status"}),(0,t.jsx)("span",{className:(0,o.cn)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:e.enabled?"Public":"Draft"})]}),h&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Source"}),(0,t.jsxs)("a",{href:h,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[h.replace("https://",""),(0,t.jsx)(n.Link2,{className:"size-3 shrink-0"})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:"Tags"}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:"Skill ID"}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===c&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"Using this skill"}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(x,"install"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===p?"text-success":"text-info"),children:["install"===p?(0,t.jsx)(r.Check,{className:"size-3"}):(0,t.jsx)(i.Copy,{className:"size-3"}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:x})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsxs)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:['If you see "Plugin ',e.name,'not found in marketplace", update the catalog first:']}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>m("setup"),className:"cursor-pointer text-info",children:"See one-time setup →"})]})]}),"setup"===c&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:"One-time marketplace setup"}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:"Run this command in Claude Code to register the marketplace:"}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;g(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===p?"text-success":"text-info"),children:["marketplace-cmd"===p?(0,t.jsx)(r.Check,{className:"size-3"}):(0,t.jsx)(i.Copy,{className:"size-3"}),"marketplace-cmd"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsxs)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:["Or add this to ",(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})," ","for a persistent configuration:"]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>g(_,"settings"),className:(0,o.cn)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===p?"text-success":"text-info"),children:["settings"===p?(0,t.jsx)(r.Check,{className:"size-3"}):(0,t.jsx)(i.Copy,{className:"size-3"}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:_})]})]})]})}],652272)},899426,e=>{"use strict";let t=e=>e.trim().toLowerCase();function a(e,a){let s=t(e);if(""===s)return!0;let r=a.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!r.some(e=>e.includes(s))||s.split(/\s+/).every(e=>r.some(t=>t.includes(e)))}e.s(["filterBySearchTerm",0,function(e,t,s){return e.filter(e=>a(t,s(e)))},"matchesSearchTerm",0,a,"rankBySearchRelevance",0,function(e,a,s){let r=t(a);if(""===r)return[...e];let i=e=>{let t=s(e).toLowerCase();return 1e3*(t===r)+100*!!t.startsWith(r)+(1e3-t.length)};return[...e].sort((e,t)=>i(t)-i(e))}])},198458,e=>{"use strict";var t=e.i(655063),a=e.i(266027),s=e.i(271645),r=e.i(741466);e.s(["useResourceList",0,function(e){let{queryKey:i,fetchPage:n,serializeFilters:o,defaultSorting:l,defaultPageSize:d,enabled:c}=e,[m,p]=(0,s.useState)(l),[u,g]=(0,s.useState)({pageIndex:0,pageSize:d}),[h,x]=(0,s.useState)([]),[f,b]=(0,s.useState)(""),[_]=(0,t.useDebouncedValue)(f,{wait:r.DEBOUNCE_WAIT_MS}),j=(0,s.useMemo)(()=>{let e=m.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=_.trim();return{page:u.pageIndex+1,page_size:u.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...o(h)}},[m,u.pageIndex,u.pageSize,_,h,o]),y={queryKey:[...i,j],queryFn:({signal:e})=>n(j,e),enabled:c,placeholderData:e=>e},{data:w,isLoading:v,isFetching:N,error:k,refetch:C}=(0,a.useQuery)(y),S=(0,s.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),I=(0,s.useCallback)(e=>{p(e),S()},[S]),L=(0,s.useCallback)(e=>{x(e),S()},[S]),$=(0,s.useCallback)(e=>{b(e),S()},[S]),E=(0,s.useCallback)(()=>{C()},[C]);return{rows:(0,s.useMemo)(()=>w?.data??[],[w]),rowCount:w?.meta.total_count??0,isLoading:v,isFetching:N,error:k,refetch:E,sorting:m,onSortingChange:I,pagination:u,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:L,searchValue:f,onSearchChange:$}}])}]);