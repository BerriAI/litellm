(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,695411,e=>{"use strict";var t=e.i(602869);let r=async e=>{try{let r=await (0,t.modelHubCall)(e);if(console.log("model_info:",r),r?.data.length>0){let e=r.data.map(e=>({model_group:e.model_group,mode:e?.mode}));return e.sort((e,t)=>e.model_group.localeCompare(t.model_group)),e}return[]}catch(e){throw console.error("Error fetching model info:",e),e}};e.s(["fetchAvailableModels",0,r])},737434,e=>{"use strict";var t=e.i(184163);e.s(["DownloadOutlined",()=>t.default])},916940,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(199133),a=e.i(602869);e.s(["default",0,({onChange:e,value:i,className:n,accessToken:s,placeholder:l="Select vector stores",disabled:c=!1})=>{let[d,u]=(0,r.useState)([]),[p,m]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(s){m(!0);try{let e=await (0,a.vectorStoreListCall)(s);e.data&&u(e.data)}catch(e){console.error("Error fetching vector stores:",e)}finally{m(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",placeholder:l,onChange:e,value:i,loading:p,className:n,allowClear:!0,options:d.map(e=>({label:`${e.vector_store_name||e.vector_store_id} (${e.vector_store_id})`,value:e.vector_store_id,title:e.vector_store_description||e.vector_store_id})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"},disabled:c})})}])},246349,e=>{"use strict";let t=(0,e.i(475254).default)("chevron-right",[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]]);e.s(["default",0,t])},841947,e=>{"use strict";let t=(0,e.i(475254).default)("x",[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]]);e.s(["default",0,t])},603908,e=>{"use strict";let t=(0,e.i(475254).default)("plus",[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]]);e.s(["default",0,t])},107233,37727,e=>{"use strict";var t=e.i(603908);e.s(["Plus",()=>t.default],107233);var r=e.i(841947);e.s(["X",()=>r.default],37727)},482725,e=>{"use strict";var t=e.i(244451);e.s(["Spin",()=>t.default])},270377,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M464 688a48 48 0 1096 0 48 48 0 10-96 0zm24-112h48c4.4 0 8-3.6 8-8V296c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v272c0 4.4 3.6 8 8 8z"}}]},name:"exclamation-circle",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["ExclamationCircleOutlined",0,i],270377)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["ArrowLeftOutlined",0,i],447566)},637235,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}},{tag:"path",attrs:{d:"M686.7 638.6L544.1 535.5V288c0-4.4-3.6-8-8-8H488c-4.4 0-8 3.6-8 8v275.4c0 2.6 1.2 5 3.3 6.5l165.4 120.6c3.6 2.6 8.6 1.8 11.2-1.7l28.6-39c2.6-3.7 1.8-8.7-1.8-11.2z"}}]},name:"clock-circle",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["ClockCircleOutlined",0,i],637235)},597440,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M360 184h-8c4.4 0 8-3.6 8-8v8h304v-8c0 4.4 3.6 8 8 8h-8v72h72v-80c0-35.3-28.7-64-64-64H352c-35.3 0-64 28.7-64 64v80h72v-72zm504 72H160c-17.7 0-32 14.3-32 32v32c0 4.4 3.6 8 8 8h60.4l24.7 523c1.6 34.1 29.8 61 63.9 61h454c34.2 0 62.3-26.8 63.9-61l24.7-523H888c4.4 0 8-3.6 8-8v-32c0-17.7-14.3-32-32-32zM731.3 840H292.7l-24.2-512h487l-24.2 512z"}}]},name:"delete",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["default",0,i],597440)},955135,e=>{"use strict";var t=e.i(597440);e.s(["DeleteOutlined",()=>t.default])},646563,e=>{"use strict";var t=e.i(959013);e.s(["PlusOutlined",()=>t.default])},599724,936325,e=>{"use strict";var t=e.i(95779),r=e.i(444755),o=e.i(673706),a=e.i(271645);let i=a.default.forwardRef((e,i)=>{let{color:n,className:s,children:l}=e;return a.default.createElement("p",{ref:i,className:(0,r.tremorTwMerge)("text-tremor-default",n?(0,o.getColorClassNames)(n,t.colorPalette.text).textColor:(0,r.tremorTwMerge)("text-tremor-content","dark:text-dark-tremor-content"),s)},l)});i.displayName="Text",e.s(["default",0,i],936325),e.s(["Text",0,i],599724)},994388,e=>{"use strict";var t=e.i(290571),r=e.i(829087),o=e.i(271645);let a=["preEnter","entering","entered","preExit","exiting","exited","unmounted"],i=e=>({_s:e,status:a[e],isEnter:e<3,isMounted:6!==e,isResolved:2===e||e>4}),n=e=>e?6:5,s=(e,t,r,o,a)=>{clearTimeout(o.current);let n=i(e);t(n),r.current=n,a&&a({current:n})};var l=e.i(480731),c=e.i(444755),d=e.i(673706);let u=e=>{var r=(0,t.__rest)(e,[]);return o.default.createElement("svg",Object.assign({},r,{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 24 24",fill:"currentColor"}),o.default.createElement("path",{fill:"none",d:"M0 0h24v24H0z"}),o.default.createElement("path",{d:"M18.364 5.636L16.95 7.05A7 7 0 1 0 19 12h2a9 9 0 1 1-2.636-6.364z"}))};var p=e.i(95779);let m={xs:{height:"h-4",width:"w-4"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-6",width:"w-6"},xl:{height:"h-6",width:"w-6"}},h=(e,t)=>{switch(e){case"primary":return{textColor:t?(0,d.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",hoverTextColor:t?(0,d.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,d.getColorClassNames)(t,p.colorPalette.background).bgColor:"bg-tremor-brand dark:bg-dark-tremor-brand",hoverBgColor:t?(0,d.getColorClassNames)(t,p.colorPalette.darkBackground).hoverBgColor:"hover:bg-tremor-brand-emphasis dark:hover:bg-dark-tremor-brand-emphasis",borderColor:t?(0,d.getColorClassNames)(t,p.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",hoverBorderColor:t?(0,d.getColorClassNames)(t,p.colorPalette.darkBorder).hoverBorderColor:"hover:border-tremor-brand-emphasis dark:hover:border-dark-tremor-brand-emphasis"};case"secondary":return{textColor:t?(0,d.getColorClassNames)(t,p.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,d.getColorClassNames)(t,p.colorPalette.text).textColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,d.getColorClassNames)("transparent").bgColor,hoverBgColor:t?(0,c.tremorTwMerge)((0,d.getColorClassNames)(t,p.colorPalette.background).hoverBgColor,"hover:bg-opacity-20 dark:hover:bg-opacity-20"):"hover:bg-tremor-brand-faint dark:hover:bg-dark-tremor-brand-faint",borderColor:t?(0,d.getColorClassNames)(t,p.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand"};case"light":return{textColor:t?(0,d.getColorClassNames)(t,p.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,d.getColorClassNames)(t,p.colorPalette.darkText).hoverTextColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,d.getColorClassNames)("transparent").bgColor,borderColor:"",hoverBorderColor:""}}},g=(0,d.makeClassName)("Button"),f=({loading:e,iconSize:t,iconPosition:r,Icon:a,needMargin:i,transitionStatus:n})=>{let s=i?r===l.HorizontalPositions.Left?(0,c.tremorTwMerge)("-ml-1","mr-1.5"):(0,c.tremorTwMerge)("-mr-1","ml-1.5"):"",d=(0,c.tremorTwMerge)("w-0 h-0"),p={default:d,entering:d,entered:t,exiting:t,exited:d};return e?o.default.createElement(u,{className:(0,c.tremorTwMerge)(g("icon"),"animate-spin shrink-0",s,p.default,p[n]),style:{transition:"width 150ms"}}):o.default.createElement(a,{className:(0,c.tremorTwMerge)(g("icon"),"shrink-0",t,s)})},b=o.default.forwardRef((e,a)=>{let{icon:u,iconPosition:p=l.HorizontalPositions.Left,size:b=l.Sizes.SM,color:v,variant:x="primary",disabled:_,loading:k=!1,loadingText:y,children:w,tooltip:S,className:C}=e,j=(0,t.__rest)(e,["icon","iconPosition","size","color","variant","disabled","loading","loadingText","children","tooltip","className"]),z=k||_,N=void 0!==u||k,M=k&&y,T=!(!w&&!M),R=(0,c.tremorTwMerge)(m[b].height,m[b].width),E="light"!==x?(0,c.tremorTwMerge)("rounded-tremor-default border","shadow-tremor-input","dark:shadow-dark-tremor-input"):"",O=h(x,v),I=("light"!==x?{xs:{paddingX:"px-2.5",paddingY:"py-1.5",fontSize:"text-xs"},sm:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-sm"},md:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-md"},lg:{paddingX:"px-4",paddingY:"py-2.5",fontSize:"text-lg"},xl:{paddingX:"px-4",paddingY:"py-3",fontSize:"text-xl"}}:{xs:{paddingX:"",paddingY:"",fontSize:"text-xs"},sm:{paddingX:"",paddingY:"",fontSize:"text-sm"},md:{paddingX:"",paddingY:"",fontSize:"text-md"},lg:{paddingX:"",paddingY:"",fontSize:"text-lg"},xl:{paddingX:"",paddingY:"",fontSize:"text-xl"}})[b],{tooltipProps:H,getReferenceProps:A}=(0,r.useTooltip)(300),[P,L]=(({enter:e=!0,exit:t=!0,preEnter:r,preExit:a,timeout:l,initialEntered:c,mountOnEnter:d,unmountOnExit:u,onStateChange:p}={})=>{let[m,h]=(0,o.useState)(()=>i(c?2:n(d))),g=(0,o.useRef)(m),f=(0,o.useRef)(0),[b,v]="object"==typeof l?[l.enter,l.exit]:[l,l],x=(0,o.useCallback)(()=>{let e=((e,t)=>{switch(e){case 1:case 0:return 2;case 4:case 3:return n(t)}})(g.current._s,u);e&&s(e,h,g,f,p)},[p,u]);return[m,(0,o.useCallback)(o=>{let i=e=>{switch(s(e,h,g,f,p),e){case 1:b>=0&&(f.current=((...e)=>setTimeout(...e))(x,b));break;case 4:v>=0&&(f.current=((...e)=>setTimeout(...e))(x,v));break;case 0:case 3:f.current=((...e)=>setTimeout(...e))(()=>{isNaN(document.body.offsetTop)||i(e+1)},0)}},l=g.current.isEnter;"boolean"!=typeof o&&(o=!l),o?l||i(e?+!r:2):l&&i(t?a?3:4:n(u))},[x,p,e,t,r,a,b,v,u]),x]})({timeout:50});return(0,o.useEffect)(()=>{L(k)},[k]),o.default.createElement("button",Object.assign({ref:(0,d.mergeRefs)([a,H.refs.setReference]),className:(0,c.tremorTwMerge)(g("root"),"shrink-0 inline-flex justify-center items-center group font-medium outline-none",E,I.paddingX,I.paddingY,I.fontSize,O.textColor,O.bgColor,O.borderColor,O.hoverBorderColor,z?"opacity-50 cursor-not-allowed":(0,c.tremorTwMerge)(h(x,v).hoverTextColor,h(x,v).hoverBgColor,h(x,v).hoverBorderColor),C),disabled:z},A,j),o.default.createElement(r.default,Object.assign({text:S},H)),N&&p!==l.HorizontalPositions.Right?o.default.createElement(f,{loading:k,iconSize:R,iconPosition:p,Icon:u,transitionStatus:P.status,needMargin:T}):null,M||w?o.default.createElement("span",{className:(0,c.tremorTwMerge)(g("text"),"text-tremor-default whitespace-nowrap")},M?y:w):null,N&&p===l.HorizontalPositions.Right?o.default.createElement(f,{loading:k,iconSize:R,iconPosition:p,Icon:u,transitionStatus:P.status,needMargin:T}):null)});b.displayName="Button",e.s(["Button",0,b],994388)},304967,e=>{"use strict";var t=e.i(290571),r=e.i(271645),o=e.i(480731),a=e.i(95779),i=e.i(444755),n=e.i(673706);let s=(0,n.makeClassName)("Card"),l=r.default.forwardRef((e,l)=>{let{decoration:c="",decorationColor:d,children:u,className:p}=e,m=(0,t.__rest)(e,["decoration","decorationColor","children","className"]);return r.default.createElement("div",Object.assign({ref:l,className:(0,i.tremorTwMerge)(s("root"),"relative w-full text-left ring-1 rounded-tremor-default p-6","bg-tremor-background ring-tremor-ring shadow-tremor-card","dark:bg-dark-tremor-background dark:ring-dark-tremor-ring dark:shadow-dark-tremor-card",d?(0,n.getColorClassNames)(d,a.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",(e=>{if(!e)return"";switch(e){case o.HorizontalPositions.Left:return"border-l-4";case o.VerticalPositions.Top:return"border-t-4";case o.HorizontalPositions.Right:return"border-r-4";case o.VerticalPositions.Bottom:return"border-b-4";default:return""}})(c),p)},m),u)});l.displayName="Card",e.s(["Card",0,l],304967)},629569,e=>{"use strict";var t=e.i(290571),r=e.i(95779),o=e.i(444755),a=e.i(673706),i=e.i(271645);let n=i.default.forwardRef((e,n)=>{let{color:s,children:l,className:c}=e,d=(0,t.__rest)(e,["color","children","className"]);return i.default.createElement("p",Object.assign({ref:n,className:(0,o.tremorTwMerge)("font-medium text-tremor-title",s?(0,a.getColorClassNames)(s,r.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",c)},d),l)});n.displayName="Title",e.s(["Title",0,n],629569)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},536916,e=>{"use strict";var t=e.i(374276);e.s(["Checkbox",()=>t.default])},921511,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(199133),a=e.i(602869);function i(e){return e.filter(e=>(e.version_status??"draft")!=="draft").map(e=>{var t;let r=e.version_number??1,o=e.version_status??"draft";return{label:`${e.policy_name} — v${r} (${o})${e.description?` — ${e.description}`:""}`,value:"production"===o?e.policy_name:e.policy_id?(t=e.policy_id,`policy_${t}`):e.policy_name}})}e.s(["default",0,({onChange:e,value:n,className:s,accessToken:l,disabled:c,onPoliciesLoaded:d})=>{let[u,p]=(0,r.useState)([]),[m,h]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(l){h(!0);try{let e=await (0,a.getPoliciesList)(l);e.policies&&(p(e.policies),d?.(e.policies))}catch(e){console.error("Error fetching policies:",e)}finally{h(!1)}}})()},[l,d]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",disabled:c,placeholder:c?"Setting policies is a premium feature.":"Select policies (production or published versions)",onChange:t=>{e(t)},value:n,loading:m,className:s,allowClear:!0,options:i(u),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})},"getPolicyOptionEntries",0,i])},891547,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(199133),a=e.i(602869);e.s(["default",0,({onChange:e,value:i,className:n,accessToken:s,disabled:l})=>{let[c,d]=(0,r.useState)([]),[u,p]=(0,r.useState)(!1);return(0,r.useEffect)(()=>{(async()=>{if(s){p(!0);try{let e=await (0,a.getGuardrailsList)(s);console.log("Guardrails response:",e),e.guardrails&&(console.log("Guardrails data:",e.guardrails),d(e.guardrails))}catch(e){console.error("Error fetching guardrails:",e)}finally{p(!1)}}})()},[s]),(0,t.jsx)("div",{children:(0,t.jsx)(o.Select,{mode:"multiple",disabled:l,placeholder:l?"Setting guardrails is a premium feature.":"Select guardrails",onChange:t=>{console.log("Selected guardrails:",t),e(t)},value:i,loading:u,className:n,allowClear:!0,options:c.map(e=>(console.log("Mapping guardrail:",e),{label:`${e.guardrail_name}`,value:e.guardrail_name})),optionFilterProp:"label",showSearch:!0,style:{width:"100%"}})})}])},987432,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M893.3 293.3L730.7 130.7c-7.5-7.5-16.7-13-26.7-16V112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V338.5c0-17-6.7-33.2-18.7-45.2zM384 184h256v104H384V184zm456 656H184V184h136v136c0 17.7 14.3 32 32 32h320c17.7 0 32-14.3 32-32V205.8l136 136V840zM512 442c-79.5 0-144 64.5-144 144s64.5 144 144 144 144-64.5 144-144-64.5-144-144-144zm0 224c-44.2 0-80-35.8-80-80s35.8-80 80-80 80 35.8 80 80-35.8 80-80 80z"}}]},name:"save",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["SaveOutlined",0,i],987432)},678784,678745,e=>{"use strict";let t=(0,e.i(475254).default)("check",[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]]);e.s(["default",0,t],678745),e.s(["CheckIcon",0,t],678784)},54943,e=>{"use strict";let t=(0,e.i(475254).default)("search",[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]]);e.s(["default",0,t])},367240,555436,e=>{"use strict";let t=(0,e.i(475254).default)("rotate-ccw",[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}]]);e.s(["RotateCcw",0,t],367240);var r=e.i(54943);e.s(["Search",()=>r.default],555436)},362024,e=>{"use strict";var t=e.i(988122);e.s(["Collapse",()=>t.default])},245704,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M699 353h-46.9c-10.2 0-19.9 4.9-25.9 13.3L469 584.3l-71.2-98.8c-6-8.3-15.6-13.3-25.9-13.3H325c-6.5 0-10.3 7.4-6.5 12.7l124.6 172.8a31.8 31.8 0 0051.7 0l210.6-292c3.9-5.3.1-12.7-6.4-12.7z"}},{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372z"}}]},name:"check-circle",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["CheckCircleOutlined",0,i],245704)},149192,e=>{"use strict";var t=e.i(864517);e.s(["CloseOutlined",()=>t.default])},240647,e=>{"use strict";var t=e.i(286612);e.s(["RightOutlined",()=>t.default])},518617,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64c247.4 0 448 200.6 448 448S759.4 960 512 960 64 759.4 64 512 264.6 64 512 64zm0 76c-205.4 0-372 166.6-372 372s166.6 372 372 372 372-166.6 372-372-166.6-372-372-372zm128.01 198.83c.03 0 .05.01.09.06l45.02 45.01a.2.2 0 01.05.09.12.12 0 010 .07c0 .02-.01.04-.05.08L557.25 512l127.87 127.86a.27.27 0 01.05.06v.02a.12.12 0 010 .07c0 .03-.01.05-.05.09l-45.02 45.02a.2.2 0 01-.09.05.12.12 0 01-.07 0c-.02 0-.04-.01-.08-.05L512 557.25 384.14 685.12c-.04.04-.06.05-.08.05a.12.12 0 01-.07 0c-.03 0-.05-.01-.09-.05l-45.02-45.02a.2.2 0 01-.05-.09.12.12 0 010-.07c0-.02.01-.04.06-.08L466.75 512 338.88 384.14a.27.27 0 01-.05-.06l-.01-.02a.12.12 0 010-.07c0-.03.01-.05.05-.09l45.02-45.02a.2.2 0 01.09-.05.12.12 0 01.07 0c.02 0 .04.01.08.06L512 466.75l127.86-127.86c.04-.05.06-.06.08-.06a.12.12 0 01.07 0z"}}]},name:"close-circle",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["CloseCircleOutlined",0,i],518617)},657150,e=>{"use strict";let t=(0,e.i(475254).default)("bot",[["path",{d:"M12 8V4H8",key:"hb8ula"}],["rect",{width:"16",height:"12",x:"4",y:"8",rx:"2",key:"enze0r"}],["path",{d:"M2 14h2",key:"vft8re"}],["path",{d:"M20 14h2",key:"4cs60a"}],["path",{d:"M15 13v2",key:"1xurst"}],["path",{d:"M9 13v2",key:"rq6x2g"}]]);e.s(["default",0,t])},531245,782273,793916,e=>{"use strict";var t=e.i(657150);e.s(["Bot",()=>t.default],531245),e.i(247167);var r=e.i(931067),o=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M625.9 115c-5.9 0-11.9 1.6-17.4 5.3L254 352H90c-8.8 0-16 7.2-16 16v288c0 8.8 7.2 16 16 16h164l354.5 231.7c5.5 3.6 11.6 5.3 17.4 5.3 16.7 0 32.1-13.3 32.1-32.1V147.1c0-18.8-15.4-32.1-32.1-32.1zM586 803L293.4 611.7l-18-11.7H146V424h129.4l17.9-11.7L586 221v582zm348-327H806c-8.8 0-16 7.2-16 16v40c0 8.8 7.2 16 16 16h128c8.8 0 16-7.2 16-16v-40c0-8.8-7.2-16-16-16zm-41.9 261.8l-110.3-63.7a15.9 15.9 0 00-21.7 5.9l-19.9 34.5c-4.4 7.6-1.8 17.4 5.8 21.8L856.3 800a15.9 15.9 0 0021.7-5.9l19.9-34.5c4.4-7.6 1.7-17.4-5.8-21.8zM760 344a15.9 15.9 0 0021.7 5.9L892 286.2c7.6-4.4 10.2-14.2 5.8-21.8L878 230a15.9 15.9 0 00-21.7-5.9L746 287.8a15.99 15.99 0 00-5.8 21.8L760 344z"}}]},name:"sound",theme:"outlined"};var i=e.i(9583),n=o.forwardRef(function(e,t){return o.createElement(i.default,(0,r.default)({},e,{ref:t,icon:a}))});e.s(["SoundOutlined",0,n],782273);let s={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M842 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 140.3-113.7 254-254 254S258 594.3 258 454c0-4.4-3.6-8-8-8h-60c-4.4 0-8 3.6-8 8 0 168.7 126.6 307.9 290 327.6V884H326.7c-13.7 0-24.7 14.3-24.7 32v36c0 4.4 2.8 8 6.2 8h407.6c3.4 0 6.2-3.6 6.2-8v-36c0-17.7-11-32-24.7-32H548V782.1c165.3-18 294-158 294-328.1zM512 624c93.9 0 170-75.2 170-168V232c0-92.8-76.1-168-170-168s-170 75.2-170 168v224c0 92.8 76.1 168 170 168zm-94-392c0-50.6 41.9-92 94-92s94 41.4 94 92v224c0 50.6-41.9 92-94 92s-94-41.4-94-92V232z"}}]},name:"audio",theme:"outlined"};var l=o.forwardRef(function(e,t){return o.createElement(i.default,(0,r.default)({},e,{ref:t,icon:s}))});e.s(["AudioOutlined",0,l],793916)},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["LinkOutlined",0,i],596239)},339019,865361,e=>{"use strict";var t,r,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r.INTERACTIONS="interactions",r);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:o,apiKey:i,inputMessage:n,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:p,mcpServers:m,mcpServerToolRestrictions:h,selectedVoice:g,endpointType:f,selectedModel:b,selectedSdk:v,proxySettings:x}=e,_="session"===r?o:i,k=window.location.origin,y=x?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?k=y:x?.PROXY_BASE_URL&&(k=x.PROXY_BASE_URL);let w=n||"Your prompt here",S=w.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),C=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};l.length>0&&(j.tags=l),c.length>0&&(j.vector_stores=c),d.length>0&&(j.guardrails=d),u.length>0&&(j.policies=u);let z=b||"your-model-name",N="azure"===v?`import openai

client = openai.AzureOpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${k}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	base_url="${k}"
)`;switch(f){case a.CHAT:{let e=Object.keys(j).length>0,r="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=C.length>0?C:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${z}",
    messages=${JSON.stringify(o,null,4)}${r}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${z}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${S}"
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
`;break}case a.RESPONSES:{let e=Object.keys(j).length>0,r="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=C.length>0?C:[{role:"user",content:w}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${z}",
    input=${JSON.stringify(o,null,4)}${r}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${z}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${S}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===v?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${z}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${z}",
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
`;break;case a.IMAGE_EDITS:t="azure"===v?`
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${z}",
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
prompt = "${S}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${z}",
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
`;break;case a.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${z}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${z}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${z}",
	input="${n||"Your text to convert to speech here"}",
	voice="${g}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${z}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${N}
${t}`}],339019)},516015,(e,t,r)=>{},898547,(e,t,r)=>{var o=e.i(247167);e.r(516015);var a=e.r(271645),i=a&&"object"==typeof a&&"default"in a?a:{default:a},n=void 0!==o.default&&o.default.env&&!0,s=function(e){return"[object String]"===Object.prototype.toString.call(e)},l=function(){function e(e){var t=void 0===e?{}:e,r=t.name,o=void 0===r?"stylesheet":r,a=t.optimizeForSpeed,i=void 0===a?n:a;c(s(o),"`name` must be a string"),this._name=o,this._deletedRulePlaceholder="#"+o+"-deleted-rule____{}",c("boolean"==typeof i,"`optimizeForSpeed` must be a boolean"),this._optimizeForSpeed=i,this._serverSheet=void 0,this._tags=[],this._injected=!1,this._rulesCount=0;var l="u">typeof window&&document.querySelector('meta[property="csp-nonce"]');this._nonce=l?l.getAttribute("content"):null}var t,r=e.prototype;return r.setOptimizeForSpeed=function(e){c("boolean"==typeof e,"`setOptimizeForSpeed` accepts a boolean"),c(0===this._rulesCount,"optimizeForSpeed cannot be when rules have already been inserted"),this.flush(),this._optimizeForSpeed=e,this.inject()},r.isOptimizeForSpeed=function(){return this._optimizeForSpeed},r.inject=function(){var e=this;if(c(!this._injected,"sheet already injected"),this._injected=!0,"u">typeof window&&this._optimizeForSpeed){this._tags[0]=this.makeStyleTag(this._name),this._optimizeForSpeed="insertRule"in this.getSheet(),this._optimizeForSpeed||(n||console.warn("StyleSheet: optimizeForSpeed mode not supported falling back to standard mode."),this.flush(),this._injected=!0);return}this._serverSheet={cssRules:[],insertRule:function(t,r){return"number"==typeof r?e._serverSheet.cssRules[r]={cssText:t}:e._serverSheet.cssRules.push({cssText:t}),r},deleteRule:function(t){e._serverSheet.cssRules[t]=null}}},r.getSheetForTag=function(e){if(e.sheet)return e.sheet;for(var t=0;t<document.styleSheets.length;t++)if(document.styleSheets[t].ownerNode===e)return document.styleSheets[t]},r.getSheet=function(){return this.getSheetForTag(this._tags[this._tags.length-1])},r.insertRule=function(e,t){if(c(s(e),"`insertRule` accepts only strings"),"u"<typeof window)return"number"!=typeof t&&(t=this._serverSheet.cssRules.length),this._serverSheet.insertRule(e,t),this._rulesCount++;if(this._optimizeForSpeed){var r=this.getSheet();"number"!=typeof t&&(t=r.cssRules.length);try{r.insertRule(e,t)}catch(t){return n||console.warn("StyleSheet: illegal rule: \n\n"+e+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),-1}}else{var o=this._tags[t];this._tags.push(this.makeStyleTag(this._name,e,o))}return this._rulesCount++},r.replaceRule=function(e,t){if(this._optimizeForSpeed||"u"<typeof window){var r="u">typeof window?this.getSheet():this._serverSheet;if(t.trim()||(t=this._deletedRulePlaceholder),!r.cssRules[e])return e;r.deleteRule(e);try{r.insertRule(t,e)}catch(o){n||console.warn("StyleSheet: illegal rule: \n\n"+t+"\n\nSee https://stackoverflow.com/q/20007992 for more info"),r.insertRule(this._deletedRulePlaceholder,e)}}else{var o=this._tags[e];c(o,"old rule at index `"+e+"` not found"),o.textContent=t}return e},r.deleteRule=function(e){if("u"<typeof window)return void this._serverSheet.deleteRule(e);if(this._optimizeForSpeed)this.replaceRule(e,"");else{var t=this._tags[e];c(t,"rule at index `"+e+"` not found"),t.parentNode.removeChild(t),this._tags[e]=null}},r.flush=function(){this._injected=!1,this._rulesCount=0,"u">typeof window?(this._tags.forEach(function(e){return e&&e.parentNode.removeChild(e)}),this._tags=[]):this._serverSheet.cssRules=[]},r.cssRules=function(){var e=this;return"u"<typeof window?this._serverSheet.cssRules:this._tags.reduce(function(t,r){return r?t=t.concat(Array.prototype.map.call(e.getSheetForTag(r).cssRules,function(t){return t.cssText===e._deletedRulePlaceholder?null:t})):t.push(null),t},[])},r.makeStyleTag=function(e,t,r){t&&c(s(t),"makeStyleTag accepts only strings as second parameter");var o=document.createElement("style");this._nonce&&o.setAttribute("nonce",this._nonce),o.type="text/css",o.setAttribute("data-"+e,""),t&&o.appendChild(document.createTextNode(t));var a=document.head||document.getElementsByTagName("head")[0];return r?a.insertBefore(o,r):a.appendChild(o),o},t=[{key:"length",get:function(){return this._rulesCount}}],function(e,t){for(var r=0;r<t.length;r++){var o=t[r];o.enumerable=o.enumerable||!1,o.configurable=!0,"value"in o&&(o.writable=!0),Object.defineProperty(e,o.key,o)}}(e.prototype,t),e}();function c(e,t){if(!e)throw Error("StyleSheet: "+t+".")}var d=function(e){for(var t=5381,r=e.length;r;)t=33*t^e.charCodeAt(--r);return t>>>0},u={};function p(e,t){if(!t)return"jsx-"+e;var r=String(t),o=e+r;return u[o]||(u[o]="jsx-"+d(e+"-"+r)),u[o]}function m(e,t){"u"<typeof window&&(t=t.replace(/\/style/gi,"\\/style"));var r=e+t;return u[r]||(u[r]=t.replace(/__jsx-style-dynamic-selector/g,e)),u[r]}var h=function(){function e(e){var t=void 0===e?{}:e,r=t.styleSheet,o=void 0===r?null:r,a=t.optimizeForSpeed,i=void 0!==a&&a;this._sheet=o||new l({name:"styled-jsx",optimizeForSpeed:i}),this._sheet.inject(),o&&"boolean"==typeof i&&(this._sheet.setOptimizeForSpeed(i),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),this._fromServer=void 0,this._indices={},this._instancesCounts={}}var t=e.prototype;return t.add=function(e){var t=this;void 0===this._optimizeForSpeed&&(this._optimizeForSpeed=Array.isArray(e.children),this._sheet.setOptimizeForSpeed(this._optimizeForSpeed),this._optimizeForSpeed=this._sheet.isOptimizeForSpeed()),"u">typeof window&&!this._fromServer&&(this._fromServer=this.selectFromServer(),this._instancesCounts=Object.keys(this._fromServer).reduce(function(e,t){return e[t]=0,e},{}));var r=this.getIdAndRules(e),o=r.styleId,a=r.rules;if(o in this._instancesCounts){this._instancesCounts[o]+=1;return}var i=a.map(function(e){return t._sheet.insertRule(e)}).filter(function(e){return -1!==e});this._indices[o]=i,this._instancesCounts[o]=1},t.remove=function(e){var t=this,r=this.getIdAndRules(e).styleId;if(function(e,t){if(!e)throw Error("StyleSheetRegistry: "+t+".")}(r in this._instancesCounts,"styleId: `"+r+"` not found"),this._instancesCounts[r]-=1,this._instancesCounts[r]<1){var o=this._fromServer&&this._fromServer[r];o?(o.parentNode.removeChild(o),delete this._fromServer[r]):(this._indices[r].forEach(function(e){return t._sheet.deleteRule(e)}),delete this._indices[r]),delete this._instancesCounts[r]}},t.update=function(e,t){this.add(t),this.remove(e)},t.flush=function(){this._sheet.flush(),this._sheet.inject(),this._fromServer=void 0,this._indices={},this._instancesCounts={}},t.cssRules=function(){var e=this,t=this._fromServer?Object.keys(this._fromServer).map(function(t){return[t,e._fromServer[t]]}):[],r=this._sheet.cssRules();return t.concat(Object.keys(this._indices).map(function(t){return[t,e._indices[t].map(function(e){return r[e].cssText}).join(e._optimizeForSpeed?"":"\n")]}).filter(function(e){return!!e[1]}))},t.styles=function(e){var t,r;return t=this.cssRules(),void 0===(r=e)&&(r={}),t.map(function(e){var t=e[0],o=e[1];return i.default.createElement("style",{id:"__"+t,key:"__"+t,nonce:r.nonce?r.nonce:void 0,dangerouslySetInnerHTML:{__html:o}})})},t.getIdAndRules=function(e){var t=e.children,r=e.dynamic,o=e.id;if(r){var a=p(o,r);return{styleId:a,rules:Array.isArray(t)?t.map(function(e){return m(a,e)}):[m(a,t)]}}return{styleId:p(o),rules:Array.isArray(t)?t:[t]}},t.selectFromServer=function(){return Array.prototype.slice.call(document.querySelectorAll('[id^="__jsx-"]')).reduce(function(e,t){return e[t.id.slice(2)]=t,e},{})},e}(),g=a.createContext(null);function f(){return new h}function b(){return a.useContext(g)}g.displayName="StyleSheetContext";var v=i.default.useInsertionEffect||i.default.useLayoutEffect,x="u">typeof window?f():void 0;function _(e){var t=x||b();return t&&("u"<typeof window?t.add(e):v(function(){return t.add(e),function(){t.remove(e)}},[e.id,String(e.dynamic)])),null}_.dynamic=function(e){return e.map(function(e){return p(e[0],e[1])}).join(" ")},r.StyleRegistry=function(e){var t=e.registry,r=e.children,o=a.useContext(g),n=a.useState(function(){return o||t||f()})[0];return i.default.createElement(g.Provider,{value:n},r)},r.createStyleRegistry=f,r.style=_,r.useStyleRegistry=b},437902,(e,t,r)=>{t.exports=e.r(898547).style},431343,569074,e=>{"use strict";var t=e.i(475254);let r=(0,t.default)("play",[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]]);e.s(["Play",0,r],431343);let o=(0,t.default)("upload",[["path",{d:"M12 3v12",key:"1x0j5s"}],["path",{d:"m17 8-5-5-5 5",key:"7q97r8"}],["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}]]);e.s(["Upload",0,o],569074)},727612,e=>{"use strict";let t=(0,e.i(475254).default)("trash-2",[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]]);e.s(["Trash2",0,t],727612)},98919,e=>{"use strict";let t=(0,e.i(475254).default)("shield",[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}]]);e.s(["Shield",0,t],98919)},84899,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645),o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M931.4 498.9L94.9 79.5c-3.4-1.7-7.3-2.1-11-1.2a15.99 15.99 0 00-11.7 19.3l86.2 352.2c1.3 5.3 5.2 9.6 10.4 11.3l147.7 50.7-147.6 50.7c-5.2 1.8-9.1 6-10.3 11.3L72.2 926.5c-.9 3.7-.5 7.6 1.2 10.9 3.9 7.9 13.5 11.1 21.5 7.2l836.5-417c3.1-1.5 5.6-4.1 7.2-7.1 3.9-8 .7-17.6-7.2-21.6zM170.8 826.3l50.3-205.6 295.2-101.3c2.3-.8 4.2-2.6 5-5 1.4-4.2-.8-8.7-5-10.2L221.1 403 171 198.2l628 314.9-628.2 313.2z"}}]},name:"send",theme:"outlined"},a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["SendOutlined",0,i],84899)},673709,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(678784);let a=(0,e.i(475254).default)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var i=e.i(650056);let n={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};e.s(["default",0,({code:e,language:s})=>{let[l,c]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-gray-200 overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),c(!0),setTimeout(()=>c(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 z-10","aria-label":"Copy code",children:l?(0,t.jsx)(o.CheckIcon,{size:16}):(0,t.jsx)(a,{size:16})}),(0,t.jsx)(i.Prism,{language:s,style:n,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",backgroundColor:"#fafafa"},showLineNumbers:!0,children:e})]})}],673709)},91500,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M531.3 574.4l.3-1.4c5.8-23.9 13.1-53.7 7.4-80.7-3.8-21.3-19.5-29.6-32.9-30.2-15.8-.7-29.9 8.3-33.4 21.4-6.6 24-.7 56.8 10.1 98.6-13.6 32.4-35.3 79.5-51.2 107.5-29.6 15.3-69.3 38.9-75.2 68.7-1.2 5.5.2 12.5 3.5 18.8 3.7 7 9.6 12.4 16.5 15 3 1.1 6.6 2 10.8 2 17.6 0 46.1-14.2 84.1-79.4 5.8-1.9 11.8-3.9 17.6-5.9 27.2-9.2 55.4-18.8 80.9-23.1 28.2 15.1 60.3 24.8 82.1 24.8 21.6 0 30.1-12.8 33.3-20.5 5.6-13.5 2.9-30.5-6.2-39.6-13.2-13-45.3-16.4-95.3-10.2-24.6-15-40.7-35.4-52.4-65.8zM421.6 726.3c-13.9 20.2-24.4 30.3-30.1 34.7 6.7-12.3 19.8-25.3 30.1-34.7zm87.6-235.5c5.2 8.9 4.5 35.8.5 49.4-4.9-19.9-5.6-48.1-2.7-51.4.8.1 1.5.7 2.2 2zm-1.6 120.5c10.7 18.5 24.2 34.4 39.1 46.2-21.6 4.9-41.3 13-58.9 20.2-4.2 1.7-8.3 3.4-12.3 5 13.3-24.1 24.4-51.4 32.1-71.4zm155.6 65.5c.1.2.2.5-.4.9h-.2l-.2.3c-.8.5-9 5.3-44.3-8.6 40.6-1.9 45 7.3 45.1 7.4zm191.4-388.2L639.4 73.4c-6-6-14.1-9.4-22.6-9.4H192c-17.7 0-32 14.3-32 32v832c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V311.3c0-8.5-3.4-16.7-9.4-22.7zM790.2 326H602V137.8L790.2 326zm1.8 562H232V136h302v216a42 42 0 0042 42h216v494z"}}]},name:"file-pdf",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["FilePdfOutlined",0,i],91500)},132104,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M868 545.5L536.1 163a31.96 31.96 0 00-48.3 0L156 545.5a7.97 7.97 0 006 13.2h81c4.6 0 9-2 12.1-5.5L474 300.9V864c0 4.4 3.6 8 8 8h60c4.4 0 8-3.6 8-8V300.9l218.9 252.3c3 3.5 7.4 5.5 12.1 5.5h81c6.8 0 10.5-8 6-13.2z"}}]},name:"arrow-up",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["ArrowUpOutlined",0,i],132104)},447593,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645),o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6a25.95 25.95 0 0025.6 30.4h723c1.5 0 3-.1 4.4-.4a25.88 25.88 0 0021.2-30zM204 390h272V182h72v208h272v104H204V390zm468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z"}}]},name:"clear",theme:"outlined"},a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["ClearOutlined",0,i],447593)},219470,e=>{"use strict";e.s(["coy",0,{'code[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",maxHeight:"inherit",height:"inherit",padding:"0 1em",display:"block",overflow:"auto"},'pre[class*="language-"]':{color:"black",background:"none",fontFamily:"Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace",fontSize:"1em",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",wordWrap:"normal",lineHeight:"1.5",MozTabSize:"4",OTabSize:"4",tabSize:"4",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",position:"relative",margin:".5em 0",overflow:"visible",padding:"1px",backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em"},'pre[class*="language-"] > code':{position:"relative",zIndex:"1",borderLeft:"10px solid #358ccb",boxShadow:"-1px 0px 0px 0px #358ccb, 0px 0px 0px 1px #dfdfdf",backgroundColor:"#fdfdfd",backgroundImage:"linear-gradient(transparent 50%, rgba(69, 142, 209, 0.04) 50%)",backgroundSize:"3em 3em",backgroundOrigin:"content-box",backgroundAttachment:"local"},':not(pre) > code[class*="language-"]':{backgroundColor:"#fdfdfd",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box",boxSizing:"border-box",marginBottom:"1em",position:"relative",padding:".2em",borderRadius:"0.3em",color:"#c92c2c",border:"1px solid rgba(0, 0, 0, 0.1)",display:"inline",whiteSpace:"normal"},'pre[class*="language-"]:before':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"0.18em",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(-2deg)",MozTransform:"rotate(-2deg)",msTransform:"rotate(-2deg)",OTransform:"rotate(-2deg)",transform:"rotate(-2deg)"},'pre[class*="language-"]:after':{content:"''",display:"block",position:"absolute",bottom:"0.75em",left:"auto",width:"40%",height:"20%",maxHeight:"13em",boxShadow:"0px 13px 8px #979797",WebkitTransform:"rotate(2deg)",MozTransform:"rotate(2deg)",msTransform:"rotate(2deg)",OTransform:"rotate(2deg)",transform:"rotate(2deg)",right:"0.75em"},comment:{color:"#7D8B99"},"block-comment":{color:"#7D8B99"},prolog:{color:"#7D8B99"},doctype:{color:"#7D8B99"},cdata:{color:"#7D8B99"},punctuation:{color:"#5F6364"},property:{color:"#c92c2c"},tag:{color:"#c92c2c"},boolean:{color:"#c92c2c"},number:{color:"#c92c2c"},"function-name":{color:"#c92c2c"},constant:{color:"#c92c2c"},symbol:{color:"#c92c2c"},deleted:{color:"#c92c2c"},selector:{color:"#2f9c0a"},"attr-name":{color:"#2f9c0a"},string:{color:"#2f9c0a"},char:{color:"#2f9c0a"},function:{color:"#2f9c0a"},builtin:{color:"#2f9c0a"},inserted:{color:"#2f9c0a"},operator:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},entity:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)",cursor:"help"},url:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},variable:{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},atrule:{color:"#1990b8"},"attr-value":{color:"#1990b8"},keyword:{color:"#1990b8"},"class-name":{color:"#1990b8"},regex:{color:"#e90"},important:{color:"#e90",fontWeight:"normal"},".language-css .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},".style .token.string":{color:"#a67f59",background:"rgba(255, 255, 255, 0.5)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:".7"},'pre[class*="language-"].line-numbers.line-numbers':{paddingLeft:"0"},'pre[class*="language-"].line-numbers.line-numbers code':{paddingLeft:"3.8em"},'pre[class*="language-"].line-numbers.line-numbers .line-numbers-rows':{left:"0"},'pre[class*="language-"][data-line]':{paddingTop:"0",paddingBottom:"0",paddingLeft:"0"},"pre[data-line] code":{position:"relative",paddingLeft:"4em"},"pre .line-highlight":{marginTop:"0"}}],219470)},812618,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M632 888H392c-4.4 0-8 3.6-8 8v32c0 17.7 14.3 32 32 32h192c17.7 0 32-14.3 32-32v-32c0-4.4-3.6-8-8-8zM512 64c-181.1 0-328 146.9-328 328 0 121.4 66 227.4 164 284.1V792c0 17.7 14.3 32 32 32h264c17.7 0 32-14.3 32-32V676.1c98-56.7 164-162.7 164-284.1 0-181.1-146.9-328-328-328zm127.9 549.8L604 634.6V752H420V634.6l-35.9-20.8C305.4 568.3 256 484.5 256 392c0-141.4 114.6-256 256-256s256 114.6 256 256c0 92.5-49.4 176.3-128.1 221.8z"}}]},name:"bulb",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["BulbOutlined",0,i],812618)},589362,464398,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 394c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H400V152c0-4.4-3.6-8-8-8h-64c-4.4 0-8 3.6-8 8v166H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v236H152c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h168v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h228v166c0 4.4 3.6 8 8 8h64c4.4 0 8-3.6 8-8V706h164c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8H708V394h164zM628 630H400V394h228v236z"}}]},name:"number",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["NumberOutlined",0,i],589362);let n={icon:{tag:"svg",attrs:{"fill-rule":"evenodd",viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M880 912H144c-17.7 0-32-14.3-32-32V144c0-17.7 14.3-32 32-32h360c4.4 0 8 3.6 8 8v56c0 4.4-3.6 8-8 8H184v656h656V520c0-4.4 3.6-8 8-8h56c4.4 0 8 3.6 8 8v360c0 17.7-14.3 32-32 32zM653.3 424.6l52.2 52.2a8.01 8.01 0 01-4.7 13.6l-179.4 21c-5.1.6-9.5-3.7-8.9-8.9l21-179.4c.8-6.6 8.9-9.4 13.6-4.7l52.4 52.4 256.2-256.2c3.1-3.1 8.2-3.1 11.3 0l42.4 42.4c3.1 3.1 3.1 8.2 0 11.3L653.3 424.6z"}}]},name:"import",theme:"outlined"};var s=r.forwardRef(function(e,o){return r.createElement(a.default,(0,t.default)({},e,{ref:o,icon:n}))});e.s(["ImportOutlined",0,s],464398)},285903,e=>{"use strict";var t=e.i(843476),r=e.i(592968),o=e.i(637235),a=e.i(589362),i=e.i(464398),n=e.i(872934),s=e.i(812618),l=e.i(366308),c=e.i(458505);e.s(["default",0,({timeToFirstToken:e,totalLatency:d,usage:u,toolName:p})=>e||d||u?(0,t.jsxs)("div",{className:"response-metrics mt-2 pt-2 border-t border-gray-100 text-xs text-gray-500 flex flex-wrap gap-3",children:[void 0!==e&&(0,t.jsx)(r.Tooltip,{title:"Time to first token",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(o.ClockCircleOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["TTFT: ",(e/1e3).toFixed(2),"s"]})]})}),void 0!==d&&(0,t.jsx)(r.Tooltip,{title:"Total latency",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(o.ClockCircleOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["Total Latency: ",(d/1e3).toFixed(2),"s"]})]})}),u?.promptTokens!==void 0&&(0,t.jsx)(r.Tooltip,{title:"Prompt tokens",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(i.ImportOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["In: ",u.promptTokens]})]})}),u?.completionTokens!==void 0&&(0,t.jsx)(r.Tooltip,{title:"Completion tokens",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(n.ExportOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["Out: ",u.completionTokens]})]})}),u?.reasoningTokens!==void 0&&(0,t.jsx)(r.Tooltip,{title:"Reasoning tokens",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(s.BulbOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["Reasoning: ",u.reasoningTokens]})]})}),u?.totalTokens!==void 0&&(0,t.jsx)(r.Tooltip,{title:"Total tokens",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(a.NumberOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["Total: ",u.totalTokens]})]})}),u?.cost!==void 0&&(0,t.jsx)(r.Tooltip,{title:"Cost",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(c.DollarOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["$",u.cost.toFixed(6)]})]})}),p&&(0,t.jsx)(r.Tooltip,{title:"Tool used",children:(0,t.jsxs)("div",{className:"flex items-center",children:[(0,t.jsx)(l.ToolOutlined,{className:"mr-1"}),(0,t.jsxs)("span",{children:["Tool: ",p]})]})})]}):null])},245094,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M516 673c0 4.4 3.4 8 7.5 8h185c4.1 0 7.5-3.6 7.5-8v-48c0-4.4-3.4-8-7.5-8h-185c-4.1 0-7.5 3.6-7.5 8v48zm-194.9 6.1l192-161c3.8-3.2 3.8-9.1 0-12.3l-192-160.9A7.95 7.95 0 00308 351v62.7c0 2.4 1 4.6 2.9 6.1L420.7 512l-109.8 92.2a8.1 8.1 0 00-2.9 6.1V673c0 6.8 7.9 10.5 13.1 6.1zM880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zm-40 728H184V184h656v656z"}}]},name:"code",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["CodeOutlined",0,i],245094)},458505,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm0 820c-205.4 0-372-166.6-372-372s166.6-372 372-372 372 166.6 372 372-166.6 372-372 372zm47.7-395.2l-25.4-5.9V348.6c38 5.2 61.5 29 65.5 58.2.5 4 3.9 6.9 7.9 6.9h44.9c4.7 0 8.4-4.1 8-8.8-6.1-62.3-57.4-102.3-125.9-109.2V263c0-4.4-3.6-8-8-8h-28.1c-4.4 0-8 3.6-8 8v33c-70.8 6.9-126.2 46-126.2 119 0 67.6 49.8 100.2 102.1 112.7l24.7 6.3v142.7c-44.2-5.9-69-29.5-74.1-61.3-.6-3.8-4-6.6-7.9-6.6H363c-4.7 0-8.4 4-8 8.7 4.5 55 46.2 105.6 135.2 112.1V761c0 4.4 3.6 8 8 8h28.4c4.4 0 8-3.6 8-8.1l-.2-31.7c78.3-6.9 134.3-48.8 134.3-124-.1-69.4-44.2-100.4-109-116.4zm-68.6-16.2c-5.6-1.6-10.3-3.1-15-5-33.8-12.2-49.5-31.9-49.5-57.3 0-36.3 27.5-57 64.5-61.7v124zM534.3 677V543.3c3.1.9 5.9 1.6 8.8 2.2 47.3 14.4 63.2 34.4 63.2 65.1 0 39.1-29.4 62.6-72 66.4z"}}]},name:"dollar",theme:"outlined"};var a=e.i(9583),i=r.forwardRef(function(e,i){return r.createElement(a.default,(0,t.default)({},e,{ref:i,icon:o}))});e.s(["DollarOutlined",0,i],458505)},903446,e=>{"use strict";let t=(0,e.i(475254).default)("settings",[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]]);e.s(["default",0,t])},434166,e=>{"use strict";e.s(["getSecureItem",0,function(e){try{let t=window.sessionStorage.getItem(e);if(null===t)return null;return decodeURIComponent(atob(t).split("").map(e=>"%"+e.charCodeAt(0).toString(16).padStart(2,"0")).join(""))}catch{return null}},"setSecureItem",0,function(e,t){window.sessionStorage.setItem(e,btoa(encodeURIComponent(t).replace(/%([0-9A-F]{2})/g,(e,t)=>String.fromCharCode(parseInt(t,16)))))}])},611052,2781,e=>{"use strict";var t=e.i(843476),r=e.i(271645),o=e.i(212931),a=e.i(311451),i=e.i(790848),n=e.i(888259),s=e.i(438957);e.i(247167);var l=e.i(931067);let c={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M832 464h-68V240c0-70.7-57.3-128-128-128H388c-70.7 0-128 57.3-128 128v224h-68c-17.7 0-32 14.3-32 32v384c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V496c0-17.7-14.3-32-32-32zM332 240c0-30.9 25.1-56 56-56h248c30.9 0 56 25.1 56 56v224H332V240zm460 600H232V536h560v304zM484 701v53c0 4.4 3.6 8 8 8h40c4.4 0 8-3.6 8-8v-53a48.01 48.01 0 10-56 0z"}}]},name:"lock",theme:"outlined"};var d=e.i(9583),u=r.forwardRef(function(e,t){return r.createElement(d.default,(0,l.default)({},e,{ref:t,icon:c}))});e.s(["LockOutlined",0,u],2781);var p=e.i(492030),m=e.i(266537),h=e.i(447566),g=e.i(149192),f=e.i(596239);e.s(["ByokCredentialModal",0,({server:e,open:l,onClose:c,onSuccess:d,accessToken:b})=>{let[v,x]=(0,r.useState)(1),[_,k]=(0,r.useState)(""),[y,w]=(0,r.useState)(!0),[S,C]=(0,r.useState)(!1),j=e.alias||e.server_name||"Service",z=j.charAt(0).toUpperCase(),N=()=>{x(1),k(""),w(!0),C(!1),c()},M=async()=>{if(!_.trim())return void n.default.error("Please enter your API key");C(!0);try{let t=await fetch(`/v1/mcp/server/${e.server_id}/user-credential`,{method:"POST",headers:{"Content-Type":"application/json",Authorization:`Bearer ${b}`},body:JSON.stringify({credential:_.trim(),save:y})});if(!t.ok){let e=await t.json();throw Error(e?.detail?.error||"Failed to save credential")}n.default.success(`Connected to ${j}`),d(e.server_id),N()}catch(e){n.default.error(e.message||"Failed to connect")}finally{C(!1)}};return(0,t.jsx)(o.Modal,{open:l,onCancel:N,footer:null,width:480,closeIcon:null,className:"byok-modal",children:(0,t.jsxs)("div",{className:"relative p-2",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between mb-6",children:[2===v?(0,t.jsxs)("button",{onClick:()=>x(1),className:"flex items-center gap-1 text-gray-500 hover:text-gray-800 text-sm",children:[(0,t.jsx)(h.ArrowLeftOutlined,{})," Back"]}):(0,t.jsx)("div",{}),(0,t.jsxs)("div",{className:"flex items-center gap-1.5",children:[(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${1===v?"bg-blue-500":"bg-gray-300"}`}),(0,t.jsx)("div",{className:`w-2 h-2 rounded-full ${2===v?"bg-blue-500":"bg-gray-300"}`})]}),(0,t.jsx)("button",{onClick:N,className:"text-gray-400 hover:text-gray-600",children:(0,t.jsx)(g.CloseOutlined,{})})]}),1===v?(0,t.jsxs)("div",{className:"text-center",children:[(0,t.jsxs)("div",{className:"flex items-center justify-center gap-3 mb-6",children:[(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-teal-400 to-cyan-600 flex items-center justify-center text-white font-bold text-xl shadow",children:"L"}),(0,t.jsx)(m.ArrowRightOutlined,{className:"text-gray-400 text-lg"}),(0,t.jsx)("div",{className:"w-14 h-14 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-800 flex items-center justify-center text-white font-bold text-xl shadow",children:z})]}),(0,t.jsxs)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:["Connect ",j]}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["LiteLLM needs access to ",j," to complete your request."]}),(0,t.jsx)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-4",children:(0,t.jsxs)("div",{className:"flex items-start gap-3",children:[(0,t.jsx)("div",{className:"mt-0.5",children:(0,t.jsxs)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:[(0,t.jsx)("rect",{x:"2",y:"4",width:"20",height:"16",rx:"2",stroke:"currentColor",strokeWidth:"2"}),(0,t.jsx)("path",{d:"M8 4v16M16 4v16",stroke:"currentColor",strokeWidth:"2"})]})}),(0,t.jsxs)("div",{children:[(0,t.jsx)("p",{className:"font-semibold text-gray-800 mb-1",children:"How it works"}),(0,t.jsxs)("p",{className:"text-gray-500 text-sm",children:["LiteLLM acts as a secure bridge. Your requests are routed through our MCP client directly to"," ",j,"'s API."]})]})]})}),e.byok_description&&e.byok_description.length>0&&(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 text-left mb-6",children:[(0,t.jsxs)("p",{className:"text-xs font-semibold text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2",children:[(0,t.jsxs)("svg",{width:"14",height:"14",viewBox:"0 0 24 24",fill:"none",className:"text-green-500",children:[(0,t.jsx)("path",{d:"M12 2L12 22M2 12L22 12",stroke:"currentColor",strokeWidth:"2",strokeLinecap:"round"}),(0,t.jsx)("circle",{cx:"12",cy:"12",r:"9",stroke:"currentColor",strokeWidth:"2"})]}),"Requested Access"]}),(0,t.jsx)("ul",{className:"space-y-2",children:e.byok_description.map((e,r)=>(0,t.jsxs)("li",{className:"flex items-center gap-2 text-sm text-gray-700",children:[(0,t.jsx)(p.CheckOutlined,{className:"text-green-500 flex-shrink-0"}),e]},r))})]}),(0,t.jsxs)("button",{onClick:()=>x(2),className:"w-full bg-gray-900 hover:bg-gray-700 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:["Continue to Authentication ",(0,t.jsx)(m.ArrowRightOutlined,{})]}),(0,t.jsx)("button",{onClick:N,className:"mt-3 w-full text-gray-400 hover:text-gray-600 text-sm py-2",children:"Cancel"})]}):(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center mb-4",children:(0,t.jsx)(s.KeyOutlined,{className:"text-blue-400 text-xl"})}),(0,t.jsx)("h2",{className:"text-2xl font-bold text-gray-900 mb-2",children:"Provide API Key"}),(0,t.jsxs)("p",{className:"text-gray-500 mb-6",children:["Enter your ",j," API key to authorize this connection."]}),(0,t.jsxs)("div",{className:"mb-4",children:[(0,t.jsxs)("label",{className:"block text-sm font-semibold text-gray-800 mb-2",children:[j," API Key"]}),(0,t.jsx)(a.Input.Password,{placeholder:"Enter your API key",value:_,onChange:e=>k(e.target.value),size:"large",className:"rounded-lg"}),e.byok_api_key_help_url&&(0,t.jsxs)("a",{href:e.byok_api_key_help_url,target:"_blank",rel:"noopener noreferrer",className:"text-blue-500 hover:text-blue-700 text-sm mt-2 flex items-center gap-1",children:["Where do I find my API key? ",(0,t.jsx)(f.LinkOutlined,{})]})]}),(0,t.jsxs)("div",{className:"bg-gray-50 rounded-xl p-4 flex items-center justify-between mb-4",children:[(0,t.jsxs)("div",{className:"flex items-center gap-3",children:[(0,t.jsx)("svg",{width:"20",height:"20",viewBox:"0 0 24 24",fill:"none",className:"text-gray-500",children:(0,t.jsx)("path",{d:"M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z",fill:"currentColor"})}),(0,t.jsx)("span",{className:"text-sm font-medium text-gray-800",children:"Save key for future use"})]}),(0,t.jsx)(i.Switch,{checked:y,onChange:w})]}),(0,t.jsxs)("div",{className:"bg-blue-50 rounded-xl p-4 flex items-start gap-3 mb-6",children:[(0,t.jsx)(u,{className:"text-blue-400 mt-0.5 flex-shrink-0"}),(0,t.jsx)("p",{className:"text-sm text-blue-700",children:"Your key is stored securely and transmitted over HTTPS. It is never shared with third parties."})]}),(0,t.jsxs)("button",{onClick:M,disabled:S,className:"w-full bg-blue-500 hover:bg-blue-600 disabled:opacity-60 text-white font-medium py-3 px-6 rounded-xl flex items-center justify-center gap-2 transition-colors",children:[(0,t.jsx)(u,{}),"Connect & Authorize"]})]})]})})}],611052)}]);