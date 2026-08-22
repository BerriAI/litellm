(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,164668,e=>{"use strict";var t=e.i(717521);e.s(["LoaderCircle",()=>t.default])},62478,e=>{"use strict";var t=e.i(602869);let a=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,a])},592392,e=>{"use strict";var t=e.i(62478),a=e.i(266027);let i=(0,e.i(243652).createQueryKeys)("proxySettings"),r={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};e.s(["default",0,function(e){let{data:s}=(0,a.useQuery)({queryKey:[...i.all,e],queryFn:()=>(0,t.fetchProxySettings)(e),enabled:!!e});return s??r}])},283713,e=>{"use strict";var t=e.i(271645),a=e.i(602869),i=e.i(612256);let r="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,i.useUIConfig)(),s=e?.is_control_plane??!1,n=e?.workers??[],[o,l]=(0,t.useState)(()=>localStorage.getItem(r));(0,t.useEffect)(()=>{if(!o||0===n.length)return;let e=n.find(e=>e.worker_id===o);e&&(0,a.switchToWorkerUrl)(e.url)},[o,n]);let d=n.find(e=>e.worker_id===o)??null,c=(0,t.useCallback)(e=>{let t=n.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(r,e),(0,a.switchToWorkerUrl)(t.url))},[n]);return{isControlPlane:s,workers:n,selectedWorkerId:o,selectedWorker:d,selectWorker:c,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(r),(0,a.switchToWorkerUrl)(null)},[])}}])},251773,276701,771243,895335,e=>{"use strict";var t=e.i(843476),a=e.i(731565),i=e.i(602869),r=e.i(266027);async function s(){let e=(0,i.getProxyBaseUrl)(),t=await fetch(`${e}/public/litellm_blog_posts`);if(!t.ok)throw Error(`Failed to fetch blog posts: ${t.statusText}`);return t.json()}let n="inline-flex h-9 shrink-0 items-center justify-center gap-1 rounded-md px-2 text-sm font-medium leading-none text-gray-800 transition-colors hover:bg-gray-100 hover:text-gray-950";e.s(["NAV_PRODUCT_LINK_CLASS",0,n],276701);var o=e.i(519455),l=e.i(755146),d=e.i(664659),c=e.i(164668);e.s(["BlogDropdown",0,()=>{let e=(0,a.useDisableBlogPosts)(),{data:i,isLoading:p,isError:m,refetch:u}=(0,r.useQuery)({queryKey:["blogPosts"],queryFn:s,staleTime:36e5,retry:1,retryDelay:0});return e?null:(0,t.jsxs)(l.DropdownMenu,{modal:!1,children:[(0,t.jsxs)(l.DropdownMenuTrigger,{openOnHover:!0,closeDelay:100,render:(0,t.jsx)(o.Button,{variant:"ghost",className:`${n} border-0! bg-transparent!`}),children:["Blog",(0,t.jsx)(d.ChevronDown,{className:"size-2.5 text-gray-500","aria-hidden":!0})]}),(0,t.jsx)(l.DropdownMenuContent,{align:"end",side:"bottom",className:"w-auto",children:p?(0,t.jsx)("div",{className:"flex items-center px-2 py-1.5 text-sm",children:(0,t.jsx)(c.LoaderCircle,{role:"img","aria-label":"loading",className:"size-4 animate-spin"})}):m?(0,t.jsxs)("div",{className:"flex items-center gap-2 px-2 py-1.5 text-sm",children:[(0,t.jsx)("span",{className:"text-destructive",children:"Failed to load posts"}),(0,t.jsx)(o.Button,{variant:"outline",size:"sm",onClick:()=>u(),children:"Retry"})]}):i&&0!==i.posts.length?(0,t.jsxs)(t.Fragment,{children:[i.posts.slice(0,5).map(e=>(0,t.jsx)(l.DropdownMenuItem,{children:(0,t.jsxs)("a",{href:e.url,target:"_blank",rel:"noopener noreferrer",style:{display:"block",width:380},children:[(0,t.jsx)("h5",{className:"text-sm font-semibold",style:{marginBottom:2},children:e.title}),(0,t.jsx)("span",{className:"text-muted-foreground",style:{fontSize:11},children:new Date(e.date+"T00:00:00").toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}),(0,t.jsx)("p",{className:"line-clamp-2",children:e.description})]})},e.url)),(0,t.jsx)(l.DropdownMenuSeparator,{}),(0,t.jsx)(l.DropdownMenuItem,{children:(0,t.jsx)("a",{href:"https://docs.litellm.ai/blog",target:"_blank",rel:"noopener noreferrer",children:"View all posts"})})]}):(0,t.jsx)("div",{className:"px-2 py-1.5 text-sm text-muted-foreground",children:"No posts available"})})]})}],251773);var p=e.i(636772);e.i(176782),e.i(911825);var m=e.i(115504);e.i(772436);let u=(0,m.cva)({base:"flex w-fit items-stretch *:focus-visible:relative *:focus-visible:z-10 has-[>[data-slot=button-group]]:gap-2 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",variants:{orientation:{horizontal:"*:data-slot:rounded-r-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-r-md! [&>[data-slot]~[data-slot]]:rounded-l-none [&>[data-slot]~[data-slot]]:border-l-0",vertical:"flex-col *:data-slot:rounded-b-none [&>[data-slot]:not(:has(~[data-slot]))]:rounded-b-md! [&>[data-slot]~[data-slot]]:rounded-t-none [&>[data-slot]~[data-slot]]:border-t-0"}},defaultVariants:{orientation:"horizontal"}});function g({className:e,orientation:a,...i}){return(0,t.jsx)("div",{role:"group","data-slot":"button-group","data-orientation":a,className:(0,m.cn)(u({orientation:a}),e),...i})}var f=e.i(746798),h=e.i(475254);let x=(0,h.default)("github",[["path",{d:"M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",key:"tonef"}],["path",{d:"M9 18c-4.51 2-5-2-7-2",key:"9comsn"}]]),b=[{href:"https://www.litellm.ai/support",label:"Join Slack",tooltip:"LiteLLM Slack community",Icon:(0,h.default)("slack",[["rect",{width:"3",height:"8",x:"13",y:"2",rx:"1.5",key:"diqz80"}],["path",{d:"M19 8.5V10h1.5A1.5 1.5 0 1 0 19 8.5",key:"183iwg"}],["rect",{width:"3",height:"8",x:"8",y:"14",rx:"1.5",key:"hqg7r1"}],["path",{d:"M5 15.5V14H3.5A1.5 1.5 0 1 0 5 15.5",key:"76g71w"}],["rect",{width:"8",height:"3",x:"14",y:"13",rx:"1.5",key:"1kmz0a"}],["path",{d:"M15.5 19H14v1.5a1.5 1.5 0 1 0 1.5-1.5",key:"jc4sz0"}],["rect",{width:"8",height:"3",x:"2",y:"8",rx:"1.5",key:"1omvl4"}],["path",{d:"M8.5 5H10V3.5A1.5 1.5 0 1 0 8.5 5",key:"16f3cl"}]])},{href:"https://github.com/BerriAI/litellm",label:"LiteLLM on GitHub",tooltip:"LiteLLM on GitHub",Icon:x}];e.s(["CommunityEngagementButtons",0,()=>(0,p.useDisableShowPrompts)()?null:(0,t.jsx)(f.TooltipProvider,{children:(0,t.jsx)(g,{"aria-label":"Community links",children:b.map(({href:e,label:a,tooltip:i,Icon:r})=>(0,t.jsxs)(f.Tooltip,{children:[(0,t.jsx)(f.TooltipTrigger,{render:(0,t.jsx)("a",{href:e,target:"_blank",rel:"noopener noreferrer","aria-label":a,className:(0,m.cn)((0,o.buttonVariants)({variant:"outline",size:"icon"}),"text-muted-foreground")}),children:(0,t.jsx)(r,{})}),(0,t.jsx)(f.TooltipContent,{children:i})]},e))})})],771243);var _=e.i(271645),y=e.i(115571);let j="litellmHideAutoRouterAnnouncement";function v(e){let t=t=>{t.key===j&&e()},a=t=>{let{key:a}=t.detail;a===j&&e()};return window.addEventListener("storage",t),window.addEventListener(y.LOCAL_STORAGE_EVENT,a),()=>{window.removeEventListener("storage",t),window.removeEventListener(y.LOCAL_STORAGE_EVENT,a)}}function w(){return"true"===(0,y.getLocalStorageItem)(j)}var k=e.i(487486),N=e.i(337822),C=e.i(245423);e.s(["NotificationsBell",0,()=>{let e=!(0,_.useSyncExternalStore)(v,w),[a,i]=(0,_.useState)(!1),r=(0,t.jsxs)("div",{className:"max-w-[280px]",children:[(0,t.jsx)(N.PopoverTitle,{className:"mt-0! mb-2!",children:"LiteLLM Auto Router"}),(0,t.jsx)(N.PopoverDescription,{className:"mb-3! text-sm leading-snug",children:"Route every request to the cheapest model that can handle it, no prompt changes needed."}),(0,t.jsxs)("div",{className:"flex flex-wrap items-center gap-2",children:[(0,t.jsx)("a",{className:(0,m.cn)((0,o.buttonVariants)({size:"sm"})),href:"https://docs.litellm.ai/docs/proxy/auto_routing",target:"_blank",rel:"noopener noreferrer",children:"Read the docs"}),e?(0,t.jsx)(o.Button,{variant:"link",size:"sm",className:"px-1!",onClick:()=>{(0,y.setLocalStorageItem)(j,"true"),(0,y.emitLocalStorageChange)(j),i(!1)},children:"Mark as read"}):null]})]});return(0,t.jsxs)(N.Popover,{open:a,onOpenChange:i,children:[(0,t.jsx)(N.PopoverTrigger,{className:"flex! h-9! w-9! items-center justify-center rounded-md! text-gray-600 transition-colors hover:bg-gray-100! hover:text-gray-900!","aria-label":"Notifications",children:(0,t.jsxs)("span",{className:"relative inline-flex",children:[(0,t.jsx)(C.Bell,{className:"size-4","aria-hidden":!0}),e?(0,t.jsx)(k.Badge,{className:"absolute -top-0.5 -right-1 size-1.5 p-0","aria-hidden":!0}):null]})}),(0,t.jsx)(N.PopoverContent,{align:"end",children:r})]})}],895335)},641141,e=>{"use strict";var t=e.i(843476),a=e.i(135214),i=e.i(731565),r=e.i(912089),s=e.i(636772),n=e.i(115571),o=e.i(222038),l=e.i(664659),d=e.i(344523),c=e.i(243553),p=e.i(292270),m=e.i(263488),u=e.i(581418),g=e.i(284614),f=e.i(799676),h=e.i(487486),x=e.i(337822),b=e.i(772436),_=e.i(699375),y=e.i(746798),j=e.i(922407),v=e.i(115504),w=e.i(271645);e.s(["default",0,({onLogout:e,variant:k="navbar",collapsed:N=!1})=>{let{userId:C,userEmail:S,userRoleLabel:I,premiumUser:z}=(0,a.default)(),L=(0,s.useDisableShowPrompts)(),A=(0,i.useDisableBlogPosts)(),E=(0,r.useDisableBouncingIcon)(),[$,D]=(0,w.useState)(!1);(0,w.useEffect)(()=>{D("true"===(0,n.getLocalStorageItem)("disableShowNewBadge"))},[]);let T=S||C||"user",P=function(e,t){let a=e?.split("@")[0]?.trim();if(a){let e=a.replace(/[^a-zA-Z0-9]+/g," ").trim().split(/\s+/).filter(Boolean);if(e.length>=2)return`${e[0].charAt(0)}${e[1].charAt(0)}`.toUpperCase();if(1===e.length){let t=e[0];return t.length>=2?t.slice(0,2).toUpperCase():`${t.charAt(0)}`.toUpperCase()}}return t&&t.length>=2?t.slice(0,2).toUpperCase():t&&1===t.length?`${t.toUpperCase()}•`:"?"}(S,C),M=function(e){let t=0;for(let a=0;a<e.length;a+=1)t=e.charCodeAt(a)+((t<<5)-t);return Math.abs(t)%360}(T),B=(0,o.navAccountDisplayName)(S,C);return(0,t.jsxs)(x.Popover,{children:["sidebar"===k?(0,t.jsxs)(x.PopoverTrigger,{render:(0,t.jsx)("button",{type:"button",className:(0,v.cn)("flex w-full items-center rounded-lg border border-transparent transition-colors hover:bg-sidebar-accent",N?"justify-center px-0 py-1":"gap-2.5 px-2 py-1.5 text-left"),"aria-label":`Account menu — ${I??"Unknown role"} — signed in as ${S||C||"unknown"}`,"aria-haspopup":"dialog",title:N?B:void 0}),children:[(0,t.jsx)(f.Avatar,{className:"size-[30px] shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(f.AvatarFallback,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${M} 46% 38%)`},children:P})}),!N&&(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("span",{className:"min-w-0 flex-1 leading-tight",children:[(0,t.jsx)("span",{className:"block truncate text-[13px] font-medium text-sidebar-foreground",children:B}),I&&(0,t.jsx)("span",{className:"block truncate text-[11px] text-muted-foreground",children:I})]}),(0,t.jsx)(d.ChevronsUpDown,{size:16,strokeWidth:1.75,className:"shrink-0 text-muted-foreground","aria-hidden":!0})]})]}):(0,t.jsxs)(x.PopoverTrigger,{render:(0,t.jsx)("button",{type:"button",className:"flex! max-w-[min(200px,34vw)] items-center gap-2 rounded-md! py-0.5! pl-1! pr-2! transition-colors hover:bg-gray-100!","aria-label":`Account menu — ${I??"Unknown role"} — signed in as ${S||C||"unknown"}`,"aria-haspopup":"dialog"}),children:[(0,t.jsx)(f.Avatar,{className:"shadow-inner ring-1 ring-black/5","aria-hidden":!0,children:(0,t.jsx)(f.AvatarFallback,{className:"font-semibold text-white",style:{backgroundColor:`hsl(${M} 46% 38%)`},children:P})}),(0,t.jsx)("span",{className:"hidden min-w-0 truncate text-left text-sm font-medium leading-none text-gray-900 md:inline",children:B}),(0,t.jsx)(l.ChevronDown,{className:"hidden size-2.5 shrink-0 text-gray-400 md:inline","aria-hidden":!0})]}),(0,t.jsxs)(x.PopoverContent,{align:"sidebar"===k?"start":"end",side:"sidebar"===k?"top":"bottom",className:"w-auto gap-0 rounded-lg bg-white p-1 shadow-lg","data-testid":"user-dropdown-panel",children:[(0,t.jsxs)("div",{className:"flex w-full flex-col gap-2 p-3 text-sm",children:[(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(m.Mail,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:S||"-"})]}),z?(0,t.jsxs)(h.Badge,{children:[(0,t.jsx)(c.Crown,{className:"size-3"}),"Premium"]}):(0,t.jsx)(y.TooltipProvider,{children:(0,t.jsxs)(y.Tooltip,{children:[(0,t.jsxs)(y.TooltipTrigger,{render:(0,t.jsx)(h.Badge,{variant:"outline"}),children:[(0,t.jsx)(c.Crown,{className:"size-3"}),"Standard"]}),(0,t.jsx)(y.TooltipContent,{side:"left",children:"Upgrade to Premium for advanced features"})]})})]}),(0,t.jsx)(b.Separator,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(g.User,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"User ID"})]}),(0,t.jsxs)("div",{className:"flex items-center gap-1",children:[(0,t.jsx)("span",{className:"max-w-[150px] truncate",title:C||"-",children:C||"-"}),(0,t.jsx)(j.default,{value:C,label:"Copy User ID"})]})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(u.ShieldCheck,{className:"size-4"}),(0,t.jsx)("span",{className:"text-muted-foreground",children:"Role"})]}),(0,t.jsx)("span",{children:I})]}),(0,t.jsx)(b.Separator,{className:"my-2"}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide New Feature Indicators"}),(0,t.jsx)(_.Switch,{size:"sm",checked:$,onCheckedChange:e=>{D(e),e?(0,n.setLocalStorageItem)("disableShowNewBadge","true"):(0,n.removeLocalStorageItem)("disableShowNewBadge"),(0,n.emitLocalStorageChange)("disableShowNewBadge")},"aria-label":"Toggle hide new feature indicators"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide All Prompts"}),(0,t.jsx)(_.Switch,{size:"sm",checked:L,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableShowPrompts","true"):(0,n.removeLocalStorageItem)("disableShowPrompts"),(0,n.emitLocalStorageChange)("disableShowPrompts")},"aria-label":"Toggle hide all prompts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Blog Posts"}),(0,t.jsx)(_.Switch,{size:"sm",checked:A,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableBlogPosts","true"):(0,n.removeLocalStorageItem)("disableBlogPosts"),(0,n.emitLocalStorageChange)("disableBlogPosts")},"aria-label":"Toggle hide blog posts"})]}),(0,t.jsxs)("div",{className:"flex w-full items-center justify-between gap-2",children:[(0,t.jsx)("span",{className:"text-muted-foreground",children:"Hide Bouncing Icon"}),(0,t.jsx)(_.Switch,{size:"sm",checked:E,onCheckedChange:e=>{e?(0,n.setLocalStorageItem)("disableBouncingIcon","true"):(0,n.removeLocalStorageItem)("disableBouncingIcon"),(0,n.emitLocalStorageChange)("disableBouncingIcon")},"aria-label":"Toggle hide bouncing icon"})]})]}),(0,t.jsx)(b.Separator,{}),(0,t.jsxs)("button",{type:"button",onClick:e,className:"flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm hover:bg-accent",children:[(0,t.jsx)(p.LogOut,{className:"size-4"}),"Logout"]})]})]})}])},853295,658140,383862,e=>{"use strict";var t=e.i(843476),a=e.i(618566),i=e.i(755146),r=e.i(643531),s=e.i(344523),n=e.i(373264),o=e.i(271645),l=e.i(431703),d=e.i(602869);let c=(0,o.createContext)({mode:"ai-gateway",setMode:()=>{},plugins:[],activePlugin:null}),p="litellm_plugin_mode",m=(0,l.createApiClient)({getBaseUrl:()=>(0,d.getProxyBaseUrl)()??""});function u(){return localStorage.getItem(p)??"ai-gateway"}function g(){return(0,o.useContext)(c)}e.s(["PluginModeProvider",0,function({children:e,accessToken:a}){let[i,r]=(0,o.useState)(u),[s,n]=(0,o.useState)([]),[l,d]=(0,o.useState)(!1);(0,o.useEffect)(()=>{a&&m.get("/api/plugins",{accessToken:a}).then(e=>{n(Array.isArray(e)?e:[])}).catch(()=>{}).finally(()=>d(!0))},[a]);let g="ai-gateway"!==i&&l&&!s.some(e=>e.name===i)?"ai-gateway":i,f=s.find(e=>e.name===g)??null;return(0,t.jsx)(c.Provider,{value:{mode:g,setMode:e=>{r(e),localStorage.setItem(p,e)},plugins:s,activePlugin:f},children:e})},"usePluginMode",0,g],658140);var f=e.i(292639),h=e.i(571353);let x="chat";e.s(["default",0,function(){let{mode:e,setMode:o,plugins:l}=g(),{data:d}=(0,f.useUISettings)(),c=(0,a.usePathname)(),p=!!d?.values?.enable_chat_ui,m=(0,h.migratedHref)(x),u=(c??"").replace(/\/+$/,""),b=p&&(u===m||u.startsWith(`${m}/`)),_=b?"Chat":l.find(t=>t.name===e)?.display_name??"AI Gateway",y=[{key:"ai-gateway",label:"AI Gateway"},...l.map(e=>({key:e.name,label:e.display_name}))],j=p?{key:x,label:(0,t.jsxs)("div",{className:"flex items-center justify-between gap-6 py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:"Chat"}),b&&(0,t.jsx)(r.Check,{className:"size-4 text-blue-600"})]}),onClick:()=>window.location.assign((0,h.migratedHref)(x))}:{key:x,disabled:!0,label:(0,t.jsxs)("div",{className:"flex max-w-[220px] flex-col py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:"Chat"}),(0,t.jsx)("span",{className:"whitespace-normal text-xs leading-snug text-muted-foreground",children:"Admins can enable in Settings"})]})},v=[...y.map(a=>({key:a.key,label:(0,t.jsxs)("div",{className:"flex items-center justify-between gap-6 py-0.5",children:[(0,t.jsx)("span",{className:"font-medium",children:a.label}),!b&&a.key===e&&(0,t.jsx)(r.Check,{className:"size-4 text-blue-600"})]}),onClick:()=>{o(a.key),b&&window.location.assign((0,h.migratedHref)(""))}})),j];return(0,t.jsxs)(i.DropdownMenu,{children:[(0,t.jsxs)(i.DropdownMenuTrigger,{render:(0,t.jsx)("button",{type:"button",className:"flex h-8 max-w-[220px] items-center gap-1.5 rounded-md border border-border bg-background pl-1.5 pr-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"}),children:[(0,t.jsx)("span",{className:"flex size-5 flex-none items-center justify-center rounded bg-muted text-muted-foreground",children:(0,t.jsx)(n.LayoutGrid,{className:"size-[13px]"})}),(0,t.jsx)("span",{className:"truncate",children:_}),(0,t.jsx)(s.ChevronsUpDown,{className:"size-3.5 flex-none text-muted-foreground"})]}),(0,t.jsx)(i.DropdownMenuContent,{className:"w-auto",children:v.map(e=>(0,t.jsx)(i.DropdownMenuItem,{disabled:e.disabled,onClick:e.onClick,children:e.label},e.key))})]})}],853295);var b=e.i(618393),_=e.i(131792),y=e.i(950594),j=e.i(283713);e.s(["default",0,({onWorkerSwitch:e})=>{let{isControlPlane:a,selectedWorker:i,workers:r}=(0,j.useWorker)();if(!a||!i)return null;let s=r.map(e=>({label:e.name,value:e.worker_id,disabled:e.worker_id===i.worker_id}));return(0,t.jsxs)(_.Combobox,{items:s,value:s.find(e=>e.value===i.worker_id)??null,itemToStringLabel:e=>e.label,onValueChange:t=>{t&&e(t.value)},children:[(0,t.jsx)(_.ComboboxInput,{className:"min-w-[180px]","aria-label":"Worker",children:(0,t.jsx)(y.InputGroupAddon,{align:"inline-start",children:(0,t.jsx)(b.Server,{className:"size-4"})})}),(0,t.jsxs)(_.ComboboxContent,{children:[(0,t.jsx)(_.ComboboxEmpty,{children:"No matching workers"}),(0,t.jsx)(_.ComboboxList,{children:e=>(0,t.jsx)(_.ComboboxItem,{value:e,disabled:e.disabled,children:e.label},e.value)})]})]})}],383862)},402874,e=>{"use strict";var t=e.i(843476),a=e.i(143488),i=e.i(912089),r=e.i(636772),s=e.i(283713),n=e.i(602869),o=e.i(571353),l=e.i(275144),d=e.i(268004),c=e.i(321836),p=e.i(592392),m=e.i(487486),u=e.i(664659),g=e.i(972518),f=e.i(799647),h=e.i(522016),x=e.i(251773),b=e.i(771243),_=e.i(276701),y=e.i(895335),j=e.i(641141),v=e.i(853295),w=e.i(383862);e.s(["default",0,({accessToken:e,isPublicPage:k=!1,sidebarCollapsed:N=!1,onToggleSidebar:C})=>{let S=(0,n.getProxyBaseUrl)(),I=(0,p.default)(e),{logoUrl:z}=(0,l.useTheme)(),{data:L}=(0,a.useHealthReadinessDetails)(e),A=L?.litellm_version,E=(0,i.useDisableBouncingIcon)(),$=(0,r.useDisableShowPrompts)(),{isControlPlane:D,selectedWorker:T}=(0,s.useWorker)(),P=D&&null!==T,M=z||`${S}/get_image`;return(0,t.jsx)("nav",{className:"sticky top-0 z-10 border-b border-gray-200 bg-white",children:(0,t.jsx)("div",{className:"w-full",children:(0,t.jsxs)("div",{className:"flex h-14 items-center px-4",children:[(0,t.jsxs)("div",{className:"flex shrink-0 items-center",children:[C&&(0,t.jsx)("button",{onClick:C,className:"mr-2 flex h-9 w-9 items-center justify-center rounded-md text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900",title:N?"Expand sidebar":"Collapse sidebar",children:(0,t.jsx)("span",{className:"text-lg",children:N?(0,t.jsx)(f.PanelLeftOpen,{className:"size-[18px]"}):(0,t.jsx)(g.PanelLeftClose,{className:"size-[18px]"})})}),(0,t.jsxs)("div",{className:"flex items-center gap-2",children:[(0,t.jsx)(h.default,{href:(0,o.migratedHref)(""),className:"flex items-center",children:(0,t.jsx)("div",{className:"relative",children:(0,t.jsx)("div",{className:"flex h-10 max-w-48 items-center justify-center overflow-hidden",children:(0,t.jsx)("img",{src:M,alt:"LiteLLM Brand",className:"h-auto max-h-full w-auto max-w-full object-contain"})})})}),A&&(0,t.jsxs)("div",{className:"relative",children:[!E&&(0,t.jsx)("span",{className:"absolute -left-2 -top-1 animate-bounce text-lg",style:{animationDuration:"2s"},title:"Thanks for using LiteLLM!",children:"🌑"}),(0,t.jsx)(m.Badge,{variant:"outline",className:"relative z-10 cursor-pointer text-xs font-medium",children:(0,t.jsxs)("a",{href:"https://docs.litellm.ai/release_notes",target:"_blank",rel:"noopener noreferrer",className:"shrink-0",children:["v",A]})})]})]})]}),!k&&(0,t.jsx)("div",{className:"ml-4 flex shrink-0 items-center border-l border-gray-200 pl-4",children:(0,t.jsx)(v.default,{})}),(0,t.jsxs)("div",{className:"ml-auto flex min-w-0 flex-1 items-center justify-end gap-4",children:[P&&(0,t.jsx)("div",{className:"flex shrink-0 items-center",children:(0,t.jsx)(w.default,{onWorkerSwitch:e=>{(0,d.clearTokenCookies)(),(0,c.clearStoredReturnUrl)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=`${(0,c.getLoginUrl)()}?worker=${encodeURIComponent(e)}`}})}),(0,t.jsxs)("nav",{"aria-label":"Product documentation",className:`flex min-w-0 items-center gap-2 ${P?"border-l border-gray-200 pl-4":""}`,children:[(0,t.jsxs)("a",{href:"https://docs.litellm.ai/docs/",target:"_blank",rel:"noopener noreferrer",className:_.NAV_PRODUCT_LINK_CLASS,children:["Docs",(0,t.jsx)(u.ChevronDown,{className:"pointer-events-none size-2.5 opacity-0","aria-hidden":!0})]}),(0,t.jsx)(x.BlogDropdown,{})]}),!$&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-gray-200 pl-4",children:(0,t.jsx)(b.CommunityEngagementButtons,{})}),!k&&(0,t.jsx)("div",{className:"flex shrink-0 items-center border-l border-gray-200 pl-4",children:(0,t.jsxs)("div",{className:"flex items-center gap-0.5 rounded-lg bg-gray-50 px-1 py-0 transition-colors hover:bg-gray-100",children:[(0,t.jsx)(y.NotificationsBell,{}),(0,t.jsx)("span",{className:"mx-0.5 h-6 w-px shrink-0 bg-gray-200","aria-hidden":!0}),(0,t.jsx)(j.default,{onLogout:()=>{(0,d.clearTokenCookies)(),localStorage.removeItem("litellm_selected_worker_id"),localStorage.removeItem("litellm_worker_url"),window.location.href=I.PROXY_LOGOUT_URL||""}})]})})]})]})})})}])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",0,t])},952571,e=>{"use strict";var t=e.i(879664);e.s(["Info",()=>t.default])},233565,e=>{"use strict";var t=e.i(246349);e.s(["ChevronRightIcon",()=>t.default])},755146,e=>{"use strict";var t=e.i(843476),a=e.i(451512),i=e.i(115504);e.i(233565),e.i(678784),e.s(["DropdownMenu",0,function({...e}){return(0,t.jsx)(a.Menu.Root,{"data-slot":"dropdown-menu",...e})},"DropdownMenuContent",0,function({align:e="start",alignOffset:r=0,side:s="bottom",sideOffset:n=4,className:o,...l}){return(0,t.jsx)(a.Menu.Portal,{children:(0,t.jsx)(a.Menu.Positioner,{className:"isolate z-50 outline-none",align:e,alignOffset:r,side:s,sideOffset:n,children:(0,t.jsx)(a.Menu.Popup,{"data-slot":"dropdown-menu-content",className:(0,i.cn)("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-md bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95",o),...l})})})},"DropdownMenuItem",0,function({className:e,inset:r,variant:s="default",...n}){return(0,t.jsx)(a.Menu.Item,{"data-slot":"dropdown-menu-item","data-inset":r,"data-variant":s,className:(0,i.cn)("group/dropdown-menu-item relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground not-data-[variant=destructive]:focus:**:text-accent-foreground data-inset:pl-8 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[variant=destructive]:*:[svg]:text-destructive",e),...n})},"DropdownMenuSeparator",0,function({className:e,...r}){return(0,t.jsx)(a.Menu.Separator,{"data-slot":"dropdown-menu-separator",className:(0,i.cn)("-mx-1 my-1 h-px bg-border",e),...r})},"DropdownMenuTrigger",0,function({...e}){return(0,t.jsx)(a.Menu.Trigger,{"data-slot":"dropdown-menu-trigger",...e})}])},515288,e=>{"use strict";var t=e.i(843476),a=e.i(271645),i=e.i(115504);let r=a.forwardRef(({className:e,size:a="default",...r},s)=>(0,t.jsx)("div",{ref:s,"data-slot":"card","data-size":a,className:(0,i.cn)("group/card flex flex-col gap-(--card-spacing) rounded-xl bg-card py-(--card-spacing) text-sm text-card-foreground shadow-xs ring-1 ring-foreground/10 [--card-spacing:--spacing(6)] has-[>img:first-child]:pt-0 data-[size=sm]:[--card-spacing:--spacing(4)] *:[img:first-child]:rounded-t-xl *:[img:last-child]:rounded-b-xl",e),...r}));r.displayName="Card";let s=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-header",className:(0,i.cn)("group/card-header @container/card-header grid auto-rows-min items-start gap-1 rounded-t-xl px-(--card-spacing) has-data-[slot=card-action]:grid-cols-[1fr_auto] has-data-[slot=card-description]:grid-rows-[auto_auto] [.border-b]:pb-(--card-spacing)",e),...a}));s.displayName="CardHeader";let n=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-title",className:(0,i.cn)("text-base leading-normal font-medium group-data-[size=sm]/card:text-sm",e),...a}));n.displayName="CardTitle";let o=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-description",className:(0,i.cn)("text-sm text-muted-foreground",e),...a}));o.displayName="CardDescription";let l=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-action",className:(0,i.cn)("col-start-2 row-span-2 row-start-1 self-start justify-self-end",e),...a}));l.displayName="CardAction";let d=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-content",className:(0,i.cn)("px-(--card-spacing)",e),...a}));d.displayName="CardContent";let c=a.forwardRef(({className:e,...a},r)=>(0,t.jsx)("div",{ref:r,"data-slot":"card-footer",className:(0,i.cn)("flex items-center rounded-b-xl px-(--card-spacing) [.border-t]:pt-(--card-spacing)",e),...a}));c.displayName="CardFooter",e.s(["Card",0,r,"CardAction",0,l,"CardContent",0,d,"CardDescription",0,o,"CardFooter",0,c,"CardHeader",0,s,"CardTitle",0,n])},776639,e=>{"use strict";var t=e.i(843476),a=e.i(353753),i=e.i(115504),r=e.i(519455),s=e.i(995926);function n({...e}){return(0,t.jsx)(a.Dialog.Portal,{"data-slot":"dialog-portal",...e})}function o({className:e,...r}){return(0,t.jsx)(a.Dialog.Backdrop,{"data-slot":"dialog-overlay",className:(0,i.cn)("fixed inset-0 isolate z-50 bg-black/10 duration-100 supports-backdrop-filter:backdrop-blur-xs data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",e),...r})}e.s(["Dialog",0,function({...e}){return(0,t.jsx)(a.Dialog.Root,{"data-slot":"dialog",...e})},"DialogContent",0,function({className:e,children:l,showCloseButton:d=!0,...c}){return(0,t.jsxs)(n,{children:[(0,t.jsx)(o,{}),(0,t.jsxs)(a.Dialog.Popup,{"data-slot":"dialog-content",className:(0,i.cn)("fixed top-1/2 left-1/2 z-50 grid w-full max-w-[calc(100%-2rem)] -translate-x-1/2 -translate-y-1/2 gap-6 rounded-xl bg-popover p-6 text-sm text-popover-foreground ring-1 ring-foreground/10 duration-100 outline-none sm:max-w-md data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",e),...c,children:[l,d&&(0,t.jsxs)(a.Dialog.Close,{"data-slot":"dialog-close",render:(0,t.jsx)(r.Button,{variant:"ghost",className:"absolute top-4 right-4",size:"icon-sm"}),children:[(0,t.jsx)(s.XIcon,{}),(0,t.jsx)("span",{className:"sr-only",children:"Close"})]})]})]})},"DialogDescription",0,function({className:e,...r}){return(0,t.jsx)(a.Dialog.Description,{"data-slot":"dialog-description",className:(0,i.cn)("text-sm text-muted-foreground *:[a]:underline *:[a]:underline-offset-3 *:[a]:hover:text-foreground",e),...r})},"DialogFooter",0,function({className:e,showCloseButton:s=!1,children:n,...o}){return(0,t.jsxs)("div",{"data-slot":"dialog-footer",className:(0,i.cn)("flex flex-col-reverse gap-2 sm:flex-row sm:justify-end",e),...o,children:[n,s&&(0,t.jsx)(a.Dialog.Close,{render:(0,t.jsx)(r.Button,{variant:"outline"}),children:"Close"})]})},"DialogHeader",0,function({className:e,...a}){return(0,t.jsx)("div",{"data-slot":"dialog-header",className:(0,i.cn)("flex flex-col gap-2",e),...a})},"DialogTitle",0,function({className:e,...r}){return(0,t.jsx)(a.Dialog.Title,{"data-slot":"dialog-title",className:(0,i.cn)("leading-none font-medium",e),...r})}])},541071,373488,e=>{"use strict";let t=(0,e.i(475254).default)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["default",0,t],373488),e.s(["MoreHorizontal",0,t],541071)},434626,e=>{"use strict";var t=e.i(271645);let a=t.forwardRef(function(e,a){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:a},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,a],434626)},339019,865361,e=>{"use strict";var t,a,i=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.COMPLETION="completion",t.RESPONSES="responses",t.IMAGE_EDITS="image_edit",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t.REALTIME="realtime",t),r=((a={}).IMAGE="image",a.VIDEO="video",a.CHAT="chat",a.RESPONSES="responses",a.IMAGE_EDITS="image_edits",a.ANTHROPIC_MESSAGES="anthropic_messages",a.EMBEDDINGS="embeddings",a.SPEECH="speech",a.TRANSCRIPTION="transcription",a.A2A_AGENTS="a2a_agents",a.MCP="mcp",a.REALTIME="realtime",a.INTERACTIONS="interactions",a);let s={image_generation:"image",video_generation:"video",chat:"chat",completion:"chat",responses:"responses",image_edit:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings",realtime:"realtime"};e.s(["EndpointType",()=>r,"ModelMode",()=>i,"getEndpointType",0,e=>Object.values(i).includes(e)?s[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:a,accessToken:i,apiKey:s,inputMessage:n,chatHistory:o,selectedTags:l,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:p,selectedVoice:m,endpointType:u,selectedModel:g,selectedSdk:f,proxySettings:h}=e,x="session"===a?i:s,b=window.location.origin,_=h?.LITELLM_UI_API_DOC_BASE_URL;_&&_.trim()?b=_:h?.PROXY_BASE_URL&&(b=h.PROXY_BASE_URL);let y=n||"Your prompt here",j=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),v=o.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),w={};l.length>0&&(w.tags=l),d.length>0&&(w.vector_stores=d),c.length>0&&(w.guardrails=c),p.length>0&&(w.policies=p);let k=g||"your-model-name",N="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${b}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${b}"
)`;switch(u){case r.CHAT:{let e=Object.keys(w).length>0,a="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=v.length>0?v:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(i,null,4)}${a}
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
#                     "text": "${j}"
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
`;break}case r.RESPONSES:{let e=Object.keys(w).length>0,a="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();a=`,
    extra_body=${e}`}let i=v.length>0?v:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(i,null,4)}${a}
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
#                 {"type": "input_text", "text": "${j}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${a}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===f?`
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
prompt = "${j}"

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
`;break;case r.IMAGE_EDITS:t="azure"===f?`
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
prompt = "${j}"

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
prompt = "${j}"

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
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case r.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${n||"Your text to convert to speech here"}",
	voice="${m}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${N}
${t}`}],339019)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var r=e.i(9583),s=a.forwardRef(function(e,s){return a.createElement(r.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["LinkOutlined",0,s],596239)},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),a=e.i(271645);let i={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var r=e.i(9583),s=a.forwardRef(function(e,s){return a.createElement(r.default,(0,t.default)({},e,{ref:s,icon:i}))});e.s(["ArrowLeftOutlined",0,s],447566)},652272,209261,e=>{"use strict";var t=e.i(843476),a=e.i(271645),i=e.i(447566),r=e.i(166406),s=e.i(492030),n=e.i(596239);let o=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,l=e=>e.trim().replace(/\/+$/,""),d=/\.(md|markdown|txt|json|ya?ml|toml)$/i,c=/^\d{1,3}(\.\d{1,3}){3}$/,p=/^[A-Za-z0-9-]+$/,m=/^[A-Za-z0-9._-]+$/,u=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},f=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),h=e=>JSON.stringify({extraKnownMarketplaces:{"my-org":{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),x=e=>{let{source:t}=e;return"github"===t.source&&t.repo?`/plugin marketplace add ${t.repo}`:("url"===t.source||"git-subdir"===t.source)&&t.url?`/plugin marketplace add ${t.url}`:`/plugin marketplace add ${e.name}`};e.s(["buildMarketplaceSettingsSnippet",0,h,"formatInstallCommand",0,x,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=l(e);return""!==t&&o.test(t)},"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let a=(e=>{let t,a=e.trim();if(""===a||a.startsWith("//"))return null;let i=/^[a-z][a-z0-9+.-]*:\/\//i.test(a)?a:`https://${a}`;try{t=new URL(i)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||c.test(t.hostname)?null:t})(e);if(!a)return null;if("github.com"===a.hostname.replace(/^www\./,""))return((e,t)=>{let a=u(e);if(a.length<2)return null;let i=a[0],r=a[1].replace(/\.git$/,"");if(!p.test(i)||!m.test(r))return null;let s=`${i}/${r}`,n=`https://github.com/${s}`,c={parsed:{source:"github",repo:s},label:`GitHub repo — ${s}`,suggestedName:f(r)};if(a.length>=4&&("tree"===a[2]||"blob"===a[2])){let e=a.slice(4),t=g(e.join("/")),i=d.test(t)?e.slice(0,-1):e;if(0===i.length)return c;let r=l(i.join("/"));return o.test(r)?{parsed:{source:"git-subdir",url:n,path:r},label:`GitHub subdir — ${s} @ ${r}`,suggestedName:f(g(r))}:null}if(2!==a.length)return null;let h=l(t??"");return""!==h?o.test(h)?{parsed:{source:"git-subdir",url:n,path:h},label:`GitHub subdir — ${s} @ ${h}`,suggestedName:f(g(h))}:null:c})(a,t);if(u(a).length<2)return null;let i=`${a.protocol}//${a.host}${a.pathname.replace(/\/+$/,"")}`,r=l(t??"");return""!==r?o.test(r)?{parsed:{source:"git-subdir",url:i,path:r},label:`Git subdir — ${i} @ ${r}`,suggestedName:f(g(r))}:null:{parsed:{source:"url",url:i},label:`Git repo — ${i}`,suggestedName:f(g(a.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:o})=>{let l,[d,c]=(0,a.useState)("overview"),[p,m]=(0,a.useState)(null),u=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},g="github"===(l=e.source).source&&l.repo?`https://github.com/${l.repo}`:"git-subdir"===l.source&&l.url?l.path?`${l.url}/tree/main/${l.path}`:l.url:"url"===l.source&&l.url?l.url:null,f=x(e),b=h(window.location.origin),_=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:o,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(i.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>c(e.key),style:{padding:"12px 20px",fontSize:14,color:d===e.key?"#1a73e8":"#5f6368",borderBottom:d===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:d===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===d&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:_.map((e,a)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},a))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),g&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:g,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[g.replace("https://",""),(0,t.jsx)(n.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>u(f,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===p?(0,t.jsx)(s.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"install"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:f})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>c("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===d&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to"," ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," ","to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>u(b,"settings"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===p?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===p?(0,t.jsx)(s.CheckOutlined,{}):(0,t.jsx)(r.CopyOutlined,{}),"settings"===p?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:b})]})]})]})}],652272)}]);