(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["LinkOutlined",0,o],596239)},571353,e=>{"use strict";e.i(602869);var t=e.i(221688);let r={"api-keys":"api-keys",models:"models-and-endpoints",api_ref:"api-reference","api-reference":"api-reference","llm-playground":"playground",projects:"projects",chat:"chat","access-groups":"access-groups",budgets:"budgets",workflows:"workflows","guardrails-monitor":"guardrails-monitor","mcp-servers":"mcp-servers","search-tools":"search-tools","tag-management":"tag-management","vector-stores":"vector-stores",memory:"memory",policies:"policies",guardrails:"guardrails",prompts:"prompts","tool-policies":"tool-policies",skills:"skills","claude-code-plugins":"skills",caching:"caching","cost-tracking":"cost-tracking","transform-request":"transform-request","ui-theme":"ui-theme",logs:"logs","admin-panel":"admin-panel","logging-and-alerts":"logging-and-alerts","model-hub-table":"model-hub-table",new_usage:"usage",usage:"old-usage",agents:"agents","router-settings":"router-settings",users:"users",teams:"teams",organizations:"organizations"};function a(){let e=t.serverRootPath&&"/"!==t.serverRootPath?`/${t.serverRootPath.replace(/^\/+|\/+$/g,"")}`:"";return`${e}/ui`}e.s(["MIGRATED_PAGES",0,r,"legacyKeyForPathname",0,function(e){let t=a(),i=(e.startsWith(t)?e.slice(t.length):e).replace(/^\/+|\/+$/g,"");for(let[e,t]of Object.entries(r))if(i===t)return e;return null},"legacyPageHref",0,function(e){return`${a()}/?page=${e}`},"migratedHref",0,function(e){return`${a()}/${e.replace(/^\/+/,"")}`}])},62478,e=>{"use strict";var t=e.i(602869);let r=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,r])},592392,e=>{"use strict";var t=e.i(62478),r=e.i(266027);let a=(0,e.i(243652).createQueryKeys)("proxySettings"),i={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};e.s(["default",0,function(e){let{data:o}=(0,r.useQuery)({queryKey:[...a.all,e],queryFn:()=>(0,t.fetchProxySettings)(e),enabled:!!e});return o??i}])},906579,e=>{"use strict";e.i(247167);var t=e.i(271645),r=e.i(343794),a=e.i(361275),i=e.i(702779),o=e.i(763731),n=e.i(242064);e.i(296059);var l=e.i(915654),s=e.i(694758),c=e.i(183293),d=e.i(403541),u=e.i(246422),m=e.i(838378);let p=new s.Keyframes("antStatusProcessing",{"0%":{transform:"scale(0.8)",opacity:.5},"100%":{transform:"scale(2.4)",opacity:0}}),g=new s.Keyframes("antZoomBadgeIn",{"0%":{transform:"scale(0) translate(50%, -50%)",opacity:0},"100%":{transform:"scale(1) translate(50%, -50%)"}}),f=new s.Keyframes("antZoomBadgeOut",{"0%":{transform:"scale(1) translate(50%, -50%)"},"100%":{transform:"scale(0) translate(50%, -50%)",opacity:0}}),h=new s.Keyframes("antNoWrapperZoomBadgeIn",{"0%":{transform:"scale(0)",opacity:0},"100%":{transform:"scale(1)"}}),b=new s.Keyframes("antNoWrapperZoomBadgeOut",{"0%":{transform:"scale(1)"},"100%":{transform:"scale(0)",opacity:0}}),y=new s.Keyframes("antBadgeLoadingCircle",{"0%":{transformOrigin:"50%"},"100%":{transform:"translate(50%, -50%) rotate(360deg)",transformOrigin:"50%"}}),x=e=>{let{fontHeight:t,lineWidth:r,marginXS:a,colorBorderBg:i}=e,o=e.colorTextLightSolid,n=e.colorError,l=e.colorErrorHover;return(0,m.mergeToken)(e,{badgeFontHeight:t,badgeShadowSize:r,badgeTextColor:o,badgeColor:n,badgeColorHover:l,badgeShadowColor:i,badgeProcessingDuration:"1.2s",badgeRibbonOffset:a,badgeRibbonCornerTransform:"scaleY(0.75)",badgeRibbonCornerFilter:"brightness(75%)"})},_=e=>{let{fontSize:t,lineHeight:r,fontSizeSM:a,lineWidth:i}=e;return{indicatorZIndex:"auto",indicatorHeight:Math.round(t*r)-2*i,indicatorHeightSM:t,dotSize:a/2,textFontSize:a,textFontSizeSM:a,textFontWeight:"normal",statusSize:a/2}},v=(0,u.genStyleHooks)("Badge",e=>(e=>{let{componentCls:t,iconCls:r,antCls:a,badgeShadowSize:i,textFontSize:o,textFontSizeSM:n,statusSize:s,dotSize:u,textFontWeight:m,indicatorHeight:x,indicatorHeightSM:_,marginXS:v,calc:w}=e,j=`${a}-scroll-number`,S=(0,d.genPresetColor)(e,(e,{darkColor:r})=>({[`&${t} ${t}-color-${e}`]:{background:r,[`&:not(${t}-count)`]:{color:r},"a:hover &":{background:r}}}));return{[t]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"relative",display:"inline-block",width:"fit-content",lineHeight:1,[`${t}-count`]:{display:"inline-flex",justifyContent:"center",zIndex:e.indicatorZIndex,minWidth:x,height:x,color:e.badgeTextColor,fontWeight:m,fontSize:o,lineHeight:(0,l.unit)(x),whiteSpace:"nowrap",textAlign:"center",background:e.badgeColor,borderRadius:w(x).div(2).equal(),boxShadow:`0 0 0 ${(0,l.unit)(i)} ${e.badgeShadowColor}`,transition:`background ${e.motionDurationMid}`,a:{color:e.badgeTextColor},"a:hover":{color:e.badgeTextColor},"a:hover &":{background:e.badgeColorHover}},[`${t}-count-sm`]:{minWidth:_,height:_,fontSize:n,lineHeight:(0,l.unit)(_),borderRadius:w(_).div(2).equal()},[`${t}-multiple-words`]:{padding:`0 ${(0,l.unit)(e.paddingXS)}`,bdi:{unicodeBidi:"plaintext"}},[`${t}-dot`]:{zIndex:e.indicatorZIndex,width:u,minWidth:u,height:u,background:e.badgeColor,borderRadius:"100%",boxShadow:`0 0 0 ${(0,l.unit)(i)} ${e.badgeShadowColor}`},[`${t}-count, ${t}-dot, ${j}-custom-component`]:{position:"absolute",top:0,insetInlineEnd:0,transform:"translate(50%, -50%)",transformOrigin:"100% 0%",[`&${r}-spin`]:{animationName:y,animationDuration:"1s",animationIterationCount:"infinite",animationTimingFunction:"linear"}},[`&${t}-status`]:{lineHeight:"inherit",verticalAlign:"baseline",[`${t}-status-dot`]:{position:"relative",top:-1,display:"inline-block",width:s,height:s,verticalAlign:"middle",borderRadius:"50%"},[`${t}-status-success`]:{backgroundColor:e.colorSuccess},[`${t}-status-processing`]:{overflow:"visible",color:e.colorInfo,backgroundColor:e.colorInfo,borderColor:"currentcolor","&::after":{position:"absolute",top:0,insetInlineStart:0,width:"100%",height:"100%",borderWidth:i,borderStyle:"solid",borderColor:"inherit",borderRadius:"50%",animationName:p,animationDuration:e.badgeProcessingDuration,animationIterationCount:"infinite",animationTimingFunction:"ease-in-out",content:'""'}},[`${t}-status-default`]:{backgroundColor:e.colorTextPlaceholder},[`${t}-status-error`]:{backgroundColor:e.colorError},[`${t}-status-warning`]:{backgroundColor:e.colorWarning},[`${t}-status-text`]:{marginInlineStart:v,color:e.colorText,fontSize:e.fontSize}}}),S),{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:g,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`${t}-zoom-leave`]:{animationName:f,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`&${t}-not-a-wrapper`]:{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:h,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`${t}-zoom-leave`]:{animationName:b,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`&:not(${t}-status)`]:{verticalAlign:"middle"},[`${j}-custom-component, ${t}-count`]:{transform:"none"},[`${j}-custom-component, ${j}`]:{position:"relative",top:"auto",display:"block",transformOrigin:"50% 50%"}},[j]:{overflow:"hidden",transition:`all ${e.motionDurationMid} ${e.motionEaseOutBack}`,[`${j}-only`]:{position:"relative",display:"inline-block",height:x,transition:`all ${e.motionDurationSlow} ${e.motionEaseOutBack}`,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden",[`> p${j}-only-unit`]:{height:x,margin:0,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden"}},[`${j}-symbol`]:{verticalAlign:"top"}},"&-rtl":{direction:"rtl",[`${t}-count, ${t}-dot, ${j}-custom-component`]:{transform:"translate(-50%, -50%)"}}})}})(x(e)),_),w=(0,u.genStyleHooks)(["Badge","Ribbon"],e=>(e=>{let{antCls:t,badgeFontHeight:r,marginXS:a,badgeRibbonOffset:i,calc:o}=e,n=`${t}-ribbon`,s=`${t}-ribbon-wrapper`,u=(0,d.genPresetColor)(e,(e,{darkColor:t})=>({[`&${n}-color-${e}`]:{background:t,color:t}}));return{[s]:{position:"relative"},[n]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"absolute",top:a,padding:`0 ${(0,l.unit)(e.paddingXS)}`,color:e.colorPrimary,lineHeight:(0,l.unit)(r),whiteSpace:"nowrap",backgroundColor:e.colorPrimary,borderRadius:e.borderRadiusSM,[`${n}-text`]:{color:e.badgeTextColor},[`${n}-corner`]:{position:"absolute",top:"100%",width:i,height:i,color:"currentcolor",border:`${(0,l.unit)(o(i).div(2).equal())} solid`,transform:e.badgeRibbonCornerTransform,transformOrigin:"top",filter:e.badgeRibbonCornerFilter}}),u),{[`&${n}-placement-end`]:{insetInlineEnd:o(i).mul(-1).equal(),borderEndEndRadius:0,[`${n}-corner`]:{insetInlineEnd:0,borderInlineEndColor:"transparent",borderBlockEndColor:"transparent"}},[`&${n}-placement-start`]:{insetInlineStart:o(i).mul(-1).equal(),borderEndStartRadius:0,[`${n}-corner`]:{insetInlineStart:0,borderBlockEndColor:"transparent",borderInlineStartColor:"transparent"}},"&-rtl":{direction:"rtl"}})}})(x(e)),_),j=e=>{let a,{prefixCls:i,value:o,current:n,offset:l=0}=e;return l&&(a={position:"absolute",top:`${l}00%`,left:0}),t.createElement("span",{style:a,className:(0,r.default)(`${i}-only-unit`,{current:n})},o)},S=e=>{let r,a,{prefixCls:i,count:o,value:n}=e,l=Number(n),s=Math.abs(o),[c,d]=t.useState(l),[u,m]=t.useState(s),p=()=>{d(l),m(s)};if(t.useEffect(()=>{let e=setTimeout(p,1e3);return()=>clearTimeout(e)},[l]),c===l||Number.isNaN(l)||Number.isNaN(c))r=[t.createElement(j,Object.assign({},e,{key:l,current:!0}))],a={transition:"none"};else{r=[];let i=l+10,o=[];for(let e=l;e<=i;e+=1)o.push(e);let n=u<s?1:-1,d=o.findIndex(e=>e%10===c);r=(n<0?o.slice(0,d+1):o.slice(d)).map((r,a)=>t.createElement(j,Object.assign({},e,{key:r,value:r%10,offset:n<0?a-d:a,current:a===d}))),a={transform:`translateY(${-function(e,t,r){let a=e,i=0;for(;(a+10)%10!==t;)a+=r,i+=r;return i}(c,l,n)}00%)`}}return t.createElement("span",{className:`${i}-only`,style:a,onTransitionEnd:p},r)};var C=function(e,t){var r={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(r[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(r[a[i]]=e[a[i]]);return r};let $=t.forwardRef((e,a)=>{let{prefixCls:i,count:l,className:s,motionClassName:c,style:d,title:u,show:m,component:p="sup",children:g}=e,f=C(e,["prefixCls","count","className","motionClassName","style","title","show","component","children"]),{getPrefixCls:h}=t.useContext(n.ConfigContext),b=h("scroll-number",i),y=Object.assign(Object.assign({},f),{"data-show":m,style:d,className:(0,r.default)(b,s,c),title:u}),x=l;if(l&&Number(l)%1==0){let e=String(l).split("");x=t.createElement("bdi",null,e.map((r,a)=>t.createElement(S,{prefixCls:b,count:Number(l),value:r,key:e.length-a})))}return((null==d?void 0:d.borderColor)&&(y.style=Object.assign(Object.assign({},d),{boxShadow:`0 0 0 1px ${d.borderColor} inset`})),g)?(0,o.cloneElement)(g,e=>({className:(0,r.default)(`${b}-custom-component`,null==e?void 0:e.className,c)})):t.createElement(p,Object.assign({},y,{ref:a}),x)});var k=function(e,t){var r={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(r[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(r[a[i]]=e[a[i]]);return r};let O=t.forwardRef((e,l)=>{var s,c,d,u,m;let{prefixCls:p,scrollNumberPrefixCls:g,children:f,status:h,text:b,color:y,count:x=null,overflowCount:_=99,dot:w=!1,size:j="default",title:S,offset:C,style:O,className:E,rootClassName:N,classNames:I,styles:T,showZero:z=!1}=e,R=k(e,["prefixCls","scrollNumberPrefixCls","children","status","text","color","count","overflowCount","dot","size","title","offset","style","className","rootClassName","classNames","styles","showZero"]),{getPrefixCls:P,direction:M,badge:A}=t.useContext(n.ConfigContext),L=P("badge",p),[B,D,H]=v(L),W=x>_?`${_}+`:x,F="0"===W||0===W||"0"===b||0===b,U=null===x||F&&!z,V=(null!=h||null!=y)&&U,G=null!=h||!F,K=w&&!F,Y=K?"":W,q=(0,t.useMemo)(()=>((null==Y||""===Y)&&(null==b||""===b)||F&&!z)&&!K,[Y,F,z,K,b]),Z=(0,t.useRef)(x);q||(Z.current=x);let X=Z.current,J=(0,t.useRef)(Y);q||(J.current=Y);let Q=J.current,ee=(0,t.useRef)(K);q||(ee.current=K);let et=(0,t.useMemo)(()=>{if(!C)return Object.assign(Object.assign({},null==A?void 0:A.style),O);let e={marginTop:C[1]};return"rtl"===M?e.left=Number.parseInt(C[0],10):e.right=-Number.parseInt(C[0],10),Object.assign(Object.assign(Object.assign({},e),null==A?void 0:A.style),O)},[M,C,O,null==A?void 0:A.style]),er=null!=S?S:"string"==typeof X||"number"==typeof X?X:void 0,ea=!q&&(0===b?z:!!b&&!0!==b),ei=ea?t.createElement("span",{className:`${L}-status-text`},b):null,eo=X&&"object"==typeof X?(0,o.cloneElement)(X,e=>({style:Object.assign(Object.assign({},et),e.style)})):void 0,en=(0,i.isPresetColor)(y,!1),el=(0,r.default)(null==I?void 0:I.indicator,null==(s=null==A?void 0:A.classNames)?void 0:s.indicator,{[`${L}-status-dot`]:V,[`${L}-status-${h}`]:!!h,[`${L}-color-${y}`]:en}),es={};y&&!en&&(es.color=y,es.background=y);let ec=(0,r.default)(L,{[`${L}-status`]:V,[`${L}-not-a-wrapper`]:!f,[`${L}-rtl`]:"rtl"===M},E,N,null==A?void 0:A.className,null==(c=null==A?void 0:A.classNames)?void 0:c.root,null==I?void 0:I.root,D,H);if(!f&&V&&(b||G||!U)){let e=et.color;return B(t.createElement("span",Object.assign({},R,{className:ec,style:Object.assign(Object.assign(Object.assign({},null==T?void 0:T.root),null==(d=null==A?void 0:A.styles)?void 0:d.root),et)}),t.createElement("span",{className:el,style:Object.assign(Object.assign(Object.assign({},null==T?void 0:T.indicator),null==(u=null==A?void 0:A.styles)?void 0:u.indicator),es)}),ea&&t.createElement("span",{style:{color:e},className:`${L}-status-text`},b)))}return B(t.createElement("span",Object.assign({ref:l},R,{className:ec,style:Object.assign(Object.assign({},null==(m=null==A?void 0:A.styles)?void 0:m.root),null==T?void 0:T.root)}),f,t.createElement(a.default,{visible:!q,motionName:`${L}-zoom`,motionAppear:!1,motionDeadline:1e3},({className:e})=>{var a,i;let o=P("scroll-number",g),n=ee.current,l=(0,r.default)(null==I?void 0:I.indicator,null==(a=null==A?void 0:A.classNames)?void 0:a.indicator,{[`${L}-dot`]:n,[`${L}-count`]:!n,[`${L}-count-sm`]:"small"===j,[`${L}-multiple-words`]:!n&&Q&&Q.toString().length>1,[`${L}-status-${h}`]:!!h,[`${L}-color-${y}`]:en}),s=Object.assign(Object.assign(Object.assign({},null==T?void 0:T.indicator),null==(i=null==A?void 0:A.styles)?void 0:i.indicator),et);return y&&!en&&((s=s||{}).background=y),t.createElement($,{prefixCls:o,show:!q,motionClassName:e,className:l,count:Q,title:er,style:s,key:"scrollNumber"},eo)}),ei))});O.Ribbon=e=>{let{className:a,prefixCls:o,style:l,color:s,children:c,text:d,placement:u="end",rootClassName:m}=e,{getPrefixCls:p,direction:g}=t.useContext(n.ConfigContext),f=p("ribbon",o),h=`${f}-wrapper`,[b,y,x]=w(f,h),_=(0,i.isPresetColor)(s,!1),v=(0,r.default)(f,`${f}-placement-${u}`,{[`${f}-rtl`]:"rtl"===g,[`${f}-color-${s}`]:_},a),j={},S={};return s&&!_&&(j.background=s,S.color=s),b(t.createElement("div",{className:(0,r.default)(h,m,y,x)},c,t.createElement("div",{className:(0,r.default)(v,y),style:Object.assign(Object.assign({},j),l)},t.createElement("span",{className:`${f}-text`},d),t.createElement("div",{className:`${f}-corner`,style:S}))))},e.s(["Badge",0,O],906579)},115571,e=>{"use strict";let t="local-storage-change";e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",0,function(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))},"getLocalStorageItem",0,function(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}},"removeLocalStorageItem",0,function(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}},"setLocalStorageItem",0,function(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}])},477189,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M464 144H160c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V160c0-8.8-7.2-16-16-16zm-52 268H212V212h200v200zm452-268H560c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V160c0-8.8-7.2-16-16-16zm-52 268H612V212h200v200zM464 544H160c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V560c0-8.8-7.2-16-16-16zm-52 268H212V612h200v200zm452-268H560c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V560c0-8.8-7.2-16-16-16zm-52 268H612V612h200v200z"}}]},name:"appstore",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["AppstoreOutlined",0,o],477189)},371401,e=>{"use strict";var t=e.i(115571),r=e.i(271645);function a(e){let r=t=>{"disableUsageIndicator"===t.key&&e()},a=t=>{let{key:r}=t.detail;"disableUsageIndicator"===r&&e()};return window.addEventListener("storage",r),window.addEventListener(t.LOCAL_STORAGE_EVENT,a),()=>{window.removeEventListener("storage",r),window.removeEventListener(t.LOCAL_STORAGE_EVENT,a)}}function i(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}e.s(["useDisableUsageIndicator",0,function(){return(0,r.useSyncExternalStore)(a,i)}])},283713,e=>{"use strict";var t=e.i(271645),r=e.i(602869),a=e.i(612256);let i="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,a.useUIConfig)(),o=e?.is_control_plane??!1,n=e?.workers??[],[l,s]=(0,t.useState)(()=>localStorage.getItem(i));(0,t.useEffect)(()=>{if(!l||0===n.length)return;let e=n.find(e=>e.worker_id===l);e&&(0,r.switchToWorkerUrl)(e.url)},[l,n]);let c=n.find(e=>e.worker_id===l)??null,d=(0,t.useCallback)(e=>{let t=n.find(t=>t.worker_id===e);t&&(s(e),localStorage.setItem(i,e),(0,r.switchToWorkerUrl)(t.url))},[n]);return{isControlPlane:o,workers:n,selectedWorkerId:l,selectedWorker:c,selectWorker:d,disconnectFromWorker:(0,t.useCallback)(()=>{s(null),localStorage.removeItem(i),(0,r.switchToWorkerUrl)(null)},[])}}])},295320,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M704 446H320c-4.4 0-8 3.6-8 8v402c0 4.4 3.6 8 8 8h384c4.4 0 8-3.6 8-8V454c0-4.4-3.6-8-8-8zm-328 64h272v117H376V510zm272 290H376V683h272v117z"}},{tag:"path",attrs:{d:"M424 748a32 32 0 1064 0 32 32 0 10-64 0zm0-178a32 32 0 1064 0 32 32 0 10-64 0z"}},{tag:"path",attrs:{d:"M811.4 368.9C765.6 248 648.9 162 512.2 162S258.8 247.9 213 368.8C126.9 391.5 63.5 470.2 64 563.6 64.6 668 145.6 752.9 247.6 762c4.7.4 8.7-3.3 8.7-8v-60.4c0-4-3-7.4-7-7.9-27-3.4-52.5-15.2-72.1-34.5-24-23.5-37.2-55.1-37.2-88.6 0-28 9.1-54.4 26.2-76.4 16.7-21.4 40.2-36.9 66.1-43.7l37.9-10 13.9-36.7c8.6-22.8 20.6-44.2 35.7-63.5 14.9-19.2 32.6-36 52.4-50 41.1-28.9 89.5-44.2 140-44.2s98.9 15.3 140 44.3c19.9 14 37.5 30.8 52.4 50 15.1 19.3 27.1 40.7 35.7 63.5l13.8 36.6 37.8 10c54.2 14.4 92.1 63.7 92.1 120 0 33.6-13.2 65.1-37.2 88.6-19.5 19.2-44.9 31.1-71.9 34.5-4 .5-6.9 3.9-6.9 7.9V754c0 4.7 4.1 8.4 8.8 8 101.7-9.2 182.5-94 183.2-198.2.6-93.4-62.7-172.1-148.6-194.9z"}}]},name:"cloud-server",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["CloudServerOutlined",0,o],295320)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["SafetyOutlined",0,o],602073)},818581,(e,t,r)=>{"use strict";Object.defineProperty(r,"__esModule",{value:!0}),Object.defineProperty(r,"useMergedRef",{enumerable:!0,get:function(){return i}});let a=e.r(271645);function i(e,t){let r=(0,a.useRef)(null),i=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=r.current;e&&(r.current=null,e());let t=i.current;t&&(i.current=null,t())}else e&&(r.current=o(e,a)),t&&(i.current=o(t,a))},[e,t])}function o(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let r=e(t);return"function"==typeof r?r:()=>e(null)}}("function"==typeof r.default||"object"==typeof r.default&&null!==r.default)&&void 0===r.default.__esModule&&(Object.defineProperty(r.default,"__esModule",{value:!0}),Object.assign(r.default,r),t.exports=r.default)},275144,e=>{"use strict";var t=e.i(843476),r=e.i(271645),a=e.i(602869);let i=(0,r.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:o})=>{let[n,l]=(0,r.useState)(null),[s,c]=(0,r.useState)(null);return(0,r.useEffect)(()=>{(async()=>{try{let e=(0,a.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",r=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(r.ok){let e=await r.json();e.values?.logo_url&&l(e.values.logo_url),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,r.useEffect)(()=>{if(s){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=s});else{let e=document.createElement("link");e.rel="icon",e.href=s,document.head.appendChild(e)}}},[s]),(0,t.jsx)(i.Provider,{value:{logoUrl:n,setLogoUrl:l,faviconUrl:s,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,r.useContext)(i);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["CrownOutlined",0,o],100486)},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",0,t])},174886,e=>{"use strict";var t=e.i(991124);e.s(["Copy",()=>t.default])},94629,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,r],94629)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},360820,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["ChevronUpIcon",0,r],360820)},871943,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["ChevronDownIcon",0,r],871943)},269200,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("Table"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement("div",{className:(0,a.tremorTwMerge)(i("root"),"overflow-auto",l)},r.default.createElement("table",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("table"),"w-full text-tremor-default","text-tremor-content","dark:text-dark-tremor-content")},s),n))});o.displayName="Table",e.s(["Table",0,o],269200)},427612,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("TableHead"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement(r.default.Fragment,null,r.default.createElement("thead",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("root"),"text-left","text-tremor-content","dark:text-dark-tremor-content",l)},s),n))});o.displayName="TableHead",e.s(["TableHead",0,o],427612)},64848,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("TableHeaderCell"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement(r.default.Fragment,null,r.default.createElement("th",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("root"),"whitespace-nowrap text-left font-semibold top-0 px-4 py-3.5","text-tremor-content-strong","dark:text-dark-tremor-content-strong",l)},s),n))});o.displayName="TableHeaderCell",e.s(["TableHeaderCell",0,o],64848)},942232,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("TableBody"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement(r.default.Fragment,null,r.default.createElement("tbody",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("root"),"align-top divide-y","divide-tremor-border","dark:divide-dark-tremor-border",l)},s),n))});o.displayName="TableBody",e.s(["TableBody",0,o],942232)},496020,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("TableRow"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement(r.default.Fragment,null,r.default.createElement("tr",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("row"),l)},s),n))});o.displayName="TableRow",e.s(["TableRow",0,o],496020)},977572,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(444755);let i=(0,e.i(673706).makeClassName)("TableCell"),o=r.default.forwardRef((e,o)=>{let{children:n,className:l}=e,s=(0,t.__rest)(e,["children","className"]);return r.default.createElement(r.default.Fragment,null,r.default.createElement("td",Object.assign({ref:o,className:(0,a.tremorTwMerge)(i("root"),"align-middle whitespace-nowrap text-left p-4",l)},s),n))});o.displayName="TableCell",e.s(["TableCell",0,o],977572)},389083,e=>{"use strict";var t=e.i(290571),r=e.i(271645),a=e.i(829087),i=e.i(480731),o=e.i(95779),n=e.i(444755),l=e.i(673706);let s={xs:{paddingX:"px-2",paddingY:"py-0.5",fontSize:"text-xs"},sm:{paddingX:"px-2.5",paddingY:"py-0.5",fontSize:"text-sm"},md:{paddingX:"px-3",paddingY:"py-0.5",fontSize:"text-md"},lg:{paddingX:"px-3.5",paddingY:"py-0.5",fontSize:"text-lg"},xl:{paddingX:"px-4",paddingY:"py-1",fontSize:"text-xl"}},c={xs:{height:"h-4",width:"w-4"},sm:{height:"h-4",width:"w-4"},md:{height:"h-4",width:"w-4"},lg:{height:"h-5",width:"w-5"},xl:{height:"h-6",width:"w-6"}},d=(0,l.makeClassName)("Badge"),u=r.default.forwardRef((e,u)=>{let{color:m,icon:p,size:g=i.Sizes.SM,tooltip:f,className:h,children:b}=e,y=(0,t.__rest)(e,["color","icon","size","tooltip","className","children"]),x=p||null,{tooltipProps:_,getReferenceProps:v}=(0,a.useTooltip)();return r.default.createElement("span",Object.assign({ref:(0,l.mergeRefs)([u,_.refs.setReference]),className:(0,n.tremorTwMerge)(d("root"),"w-max shrink-0 inline-flex justify-center items-center cursor-default rounded-tremor-small ring-1 ring-inset",m?(0,n.tremorTwMerge)((0,l.getColorClassNames)(m,o.colorPalette.background).bgColor,(0,l.getColorClassNames)(m,o.colorPalette.iconText).textColor,(0,l.getColorClassNames)(m,o.colorPalette.iconRing).ringColor,"bg-opacity-10 ring-opacity-20","dark:bg-opacity-5 dark:ring-opacity-60"):(0,n.tremorTwMerge)("bg-tremor-brand-faint text-tremor-brand-emphasis ring-tremor-brand/20","dark:bg-dark-tremor-brand-muted/50 dark:text-dark-tremor-brand dark:ring-dark-tremor-subtle/20"),s[g].paddingX,s[g].paddingY,s[g].fontSize,h)},v,y),r.default.createElement(a.default,Object.assign({text:f},_)),x?r.default.createElement(x,{className:(0,n.tremorTwMerge)(d("icon"),"shrink-0 -ml-1 mr-1.5",c[g].height,c[g].width)}):null,r.default.createElement("span",{className:(0,n.tremorTwMerge)(d("text"),"whitespace-nowrap")},b))});u.displayName="Badge",e.s(["Badge",0,u],389083)},555987,e=>{"use strict";var t=e.i(221688),r=e.i(950643);let a=/^(https?:|data:|blob:|\/\/)/i;e.s(["resolveLogoSrc",0,(e,i=t.serverRootPath)=>{if(e){let t;return a.test(e)?e:(t=(0,r.normalizeRootPath)(i),`${t}${e.startsWith("/")?e:`/${e}`}`)}}])},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["ArrowLeftOutlined",0,o],447566)},292639,e=>{"use strict";var t=e.i(602869),r=e.i(266027);let a=(0,e.i(243652).createQueryKeys)("uiSettings");e.s(["useUISettings",0,()=>(0,r.useQuery)({queryKey:a.list({}),queryFn:async()=>await (0,t.getUiSettings)(),staleTime:36e5,gcTime:36e5})])},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),r=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var i=e.i(9583),o=r.forwardRef(function(e,o){return r.createElement(i.default,(0,t.default)({},e,{ref:o,icon:a}))});e.s(["UserOutlined",0,o],771674)},829672,836938,310730,e=>{"use strict";e.i(247167);var t=e.i(271645),r=e.i(343794),a=e.i(914949),i=e.i(404948);let o=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,o],836938);var n=e.i(613541),l=e.i(763731),s=e.i(242064),c=e.i(491816);e.i(793154);var d=e.i(880476),u=e.i(183293),m=e.i(717356),p=e.i(320560),g=e.i(307358),f=e.i(246422),h=e.i(838378),b=e.i(617933);let y=(0,f.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:r}=e,a=(0,h.mergeToken)(e,{popoverBg:t,popoverColor:r});return[(e=>{let{componentCls:t,popoverColor:r,titleMinWidth:a,fontWeightStrong:i,innerPadding:o,boxShadowSecondary:n,colorTextHeading:l,borderRadiusLG:s,zIndexPopup:c,titleMarginBottom:d,colorBgElevated:m,popoverBg:g,titleBorderBottom:f,innerContentPadding:h,titlePadding:b}=e;return[{[t]:Object.assign(Object.assign({},(0,u.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:c,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":m,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:g,backgroundClip:"padding-box",borderRadius:s,boxShadow:n,padding:o},[`${t}-title`]:{minWidth:a,marginBottom:d,color:l,fontWeight:i,borderBottom:f,padding:b},[`${t}-inner-content`]:{color:r,padding:h}})},(0,p.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(a),(e=>{let{componentCls:t}=e;return{[t]:b.PresetColors.map(r=>{let a=e[`${r}6`];return{[`&${t}-${r}`]:{"--antd-arrow-background-color":a,[`${t}-inner`]:{backgroundColor:a},[`${t}-arrow`]:{background:"transparent"}}}})}})(a),(0,m.initZoomMotion)(a,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:r,fontHeight:a,padding:i,wireframe:o,zIndexPopupBase:n,borderRadiusLG:l,marginXS:s,lineType:c,colorSplit:d,paddingSM:u}=e,m=r-a;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:n+30},(0,g.getArrowToken)(e)),(0,p.getArrowOffsetToken)({contentRadius:l,limitVerticalRadius:!0})),{innerPadding:12*!o,titleMarginBottom:o?0:s,titlePadding:o?`${m/2}px ${i}px ${m/2-t}px`:0,titleBorderBottom:o?`${t}px ${c} ${d}`:"none",innerContentPadding:o?`${u}px ${i}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var x=function(e,t){var r={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(r[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(r[a[i]]=e[a[i]]);return r};let _=({title:e,content:r,prefixCls:a})=>e||r?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${a}-title`},e),r&&t.createElement("div",{className:`${a}-inner-content`},r)):null,v=e=>{let{hashId:a,prefixCls:i,className:n,style:l,placement:s="top",title:c,content:u,children:m}=e,p=o(c),g=o(u),f=(0,r.default)(a,i,`${i}-pure`,`${i}-placement-${s}`,n);return t.createElement("div",{className:f,style:l},t.createElement("div",{className:`${i}-arrow`}),t.createElement(d.Popup,Object.assign({},e,{className:a,prefixCls:i}),m||t.createElement(_,{prefixCls:i,title:p,content:g})))},w=e=>{let{prefixCls:a,className:i}=e,o=x(e,["prefixCls","className"]),{getPrefixCls:n}=t.useContext(s.ConfigContext),l=n("popover",a),[c,d,u]=y(l);return c(t.createElement(v,Object.assign({},o,{prefixCls:l,hashId:d,className:(0,r.default)(i,u)})))};e.s(["Overlay",0,_,"default",0,w],310730);var j=function(e,t){var r={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(r[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var i=0,a=Object.getOwnPropertySymbols(e);i<a.length;i++)0>t.indexOf(a[i])&&Object.prototype.propertyIsEnumerable.call(e,a[i])&&(r[a[i]]=e[a[i]]);return r};let S=t.forwardRef((e,d)=>{var u,m;let{prefixCls:p,title:g,content:f,overlayClassName:h,placement:b="top",trigger:x="hover",children:v,mouseEnterDelay:w=.1,mouseLeaveDelay:S=.1,onOpenChange:C,overlayStyle:$={},styles:k,classNames:O}=e,E=j(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:N,className:I,style:T,classNames:z,styles:R}=(0,s.useComponentConfig)("popover"),P=N("popover",p),[M,A,L]=y(P),B=N(),D=(0,r.default)(h,A,L,I,z.root,null==O?void 0:O.root),H=(0,r.default)(z.body,null==O?void 0:O.body),[W,F]=(0,a.default)(!1,{value:null!=(u=e.open)?u:e.visible,defaultValue:null!=(m=e.defaultOpen)?m:e.defaultVisible}),U=(e,t)=>{F(e,!0),null==C||C(e,t)},V=o(g),G=o(f);return M(t.createElement(c.default,Object.assign({placement:b,trigger:x,mouseEnterDelay:w,mouseLeaveDelay:S},E,{prefixCls:P,classNames:{root:D,body:H},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},R.root),T),$),null==k?void 0:k.root),body:Object.assign(Object.assign({},R.body),null==k?void 0:k.body)},ref:d,open:W,onOpenChange:e=>{U(e)},overlay:V||G?t.createElement(_,{prefixCls:P,title:V,content:G}):null,transitionName:(0,n.getTransitionName)(B,"zoom-big",E.transitionName),"data-popover-inject":!0}),(0,l.cloneElement)(v,{onKeyDown:e=>{var r,a;(0,t.isValidElement)(v)&&(null==(a=null==v?void 0:(r=v.props).onKeyDown)||a.call(r,e)),e.keyCode===i.default.ESC&&U(!1,e)}})))});S._InternalPanelDoNotUseOrYouWillBeFired=w,e.s(["default",0,S],829672)},282786,e=>{"use strict";var t=e.i(829672);e.s(["Popover",()=>t.default])},798496,e=>{"use strict";var t=e.i(843476),r=e.i(152990),a=e.i(682830),i=e.i(271645),o=e.i(269200),n=e.i(427612),l=e.i(64848),s=e.i(942232),c=e.i(496020),d=e.i(977572),u=e.i(94629),m=e.i(360820),p=e.i(871943);e.s(["ModelDataTable",0,function({data:e=[],columns:g,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:y,enablePagination:x=!1,onRowClick:_}){let[v,w]=i.default.useState(h),[j]=i.default.useState("onChange"),[S,C]=i.default.useState({}),[$,k]=i.default.useState({}),O=(0,r.useReactTable)({data:e,columns:g,state:{sorting:v,columnSizing:S,columnVisibility:$,...x&&b?{pagination:b}:{}},columnResizeMode:j,onSortingChange:w,onColumnSizingChange:C,onColumnVisibilityChange:k,...x&&y?{onPaginationChange:y}:{},getCoreRowModel:(0,a.getCoreRowModel)(),getSortedRowModel:(0,a.getSortedRowModel)(),...x?{getPaginationRowModel:(0,a.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(o.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:O.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:O.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,r.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(m.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(p.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):O.getRowModel().rows.length>0?O.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>_?.(e.original),className:_?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,r.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}])},652272,209261,e=>{"use strict";var t=e.i(843476),r=e.i(271645),a=e.i(447566),i=e.i(166406),o=e.i(492030),n=e.i(596239);let l=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,s=e=>e.trim().replace(/\/+$/,""),c=/\.(md|markdown|txt|json|ya?ml|toml)$/i,d=/^\d{1,3}(\.\d{1,3}){3}$/,u=/^[A-Za-z0-9-]+$/,m=/^[A-Za-z0-9._-]+$/,p=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},f=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),h=e=>{let{source:t}=e;return"github"===t.source&&t.repo?`/plugin marketplace add ${t.repo}`:("url"===t.source||"git-subdir"===t.source)&&t.url?`/plugin marketplace add ${t.url}`:`/plugin marketplace add ${e.name}`};e.s(["formatInstallCommand",0,h,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=s(e);return""!==t&&l.test(t)},"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let r=(e=>{let t,r=e.trim();if(""===r||r.startsWith("//"))return null;let a=/^[a-z][a-z0-9+.-]*:\/\//i.test(r)?r:`https://${r}`;try{t=new URL(a)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||d.test(t.hostname)?null:t})(e);if(!r)return null;if("github.com"===r.hostname.replace(/^www\./,""))return((e,t)=>{let r=p(e);if(r.length<2)return null;let a=r[0],i=r[1].replace(/\.git$/,"");if(!u.test(a)||!m.test(i))return null;let o=`${a}/${i}`,n=`https://github.com/${o}`,d={parsed:{source:"github",repo:o},label:`GitHub repo — ${o}`,suggestedName:f(i)};if(r.length>=4&&("tree"===r[2]||"blob"===r[2])){let e=r.slice(4),t=g(e.join("/")),a=c.test(t)?e.slice(0,-1):e;if(0===a.length)return d;let i=s(a.join("/"));return l.test(i)?{parsed:{source:"git-subdir",url:n,path:i},label:`GitHub subdir — ${o} @ ${i}`,suggestedName:f(g(i))}:null}if(2!==r.length)return null;let h=s(t??"");return""!==h?l.test(h)?{parsed:{source:"git-subdir",url:n,path:h},label:`GitHub subdir — ${o} @ ${h}`,suggestedName:f(g(h))}:null:d})(r,t);if(p(r).length<2)return null;let a=`${r.protocol}//${r.host}${r.pathname.replace(/\/+$/,"")}`,i=s(t??"");return""!==i?l.test(i)?{parsed:{source:"git-subdir",url:a,path:i},label:`Git subdir — ${a} @ ${i}`,suggestedName:f(g(i))}:null:{parsed:{source:"url",url:a},label:`Git repo — ${a}`,suggestedName:f(g(r.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:l})=>{let s,[c,d]=(0,r.useState)("overview"),[u,m]=(0,r.useState)(null),p=(e,t)=>{navigator.clipboard.writeText(e),m(t),setTimeout(()=>m(null),2e3)},g="github"===(s=e.source).source&&s.repo?`https://github.com/${s.repo}`:"git-subdir"===s.source&&s.url?s.path?`${s.url}/tree/main/${s.path}`:s.url:"url"===s.source&&s.url?s.url:null,f=h(e),b=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:l,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(a.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),style:{padding:"12px 20px",fontSize:14,color:c===e.key?"#1a73e8":"#5f6368",borderBottom:c===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:c===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===c&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:b.map((e,r)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),g&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:g,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[g.replace("https://",""),(0,t.jsx)(n.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>p(f,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===u?(0,t.jsx)(o.CheckOutlined,{}):(0,t.jsx)(i.CopyOutlined,{}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:f})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to"," ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," ","to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{p(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===u?(0,t.jsx)(o.CheckOutlined,{}):(0,t.jsx)(i.CopyOutlined,{}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)},339019,865361,e=>{"use strict";var t,r,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),i=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r.INTERACTIONS="interactions",r);let o={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>i,"getEndpointType",0,e=>Object.values(a).includes(e)?o[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:a,apiKey:o,inputMessage:n,chatHistory:l,selectedTags:s,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:m,mcpServers:p,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:y,proxySettings:x}=e,_="session"===r?a:o,v=window.location.origin,w=x?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:x?.PROXY_BASE_URL&&(v=x.PROXY_BASE_URL);let j=n||"Your prompt here",S=j.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),C=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),$={};s.length>0&&($.tags=s),c.length>0&&($.vector_stores=c),d.length>0&&($.guardrails=d),u.length>0&&($.policies=u);let k=b||"your-model-name",O="azure"===y?`import openai

client = openai.AzureOpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case i.CHAT:{let e=Object.keys($).length>0,r="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let a=C.length>0?C:[{role:"user",content:j}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(a,null,4)}${r}
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
`;break}case i.RESPONSES:{let e=Object.keys($).length>0,r="";if(e){let e=JSON.stringify({metadata:$},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let a=C.length>0?C:[{role:"user",content:j}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(a,null,4)}${r}
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
`;break}case i.IMAGE:t="azure"===y?`
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
prompt = "${S}"

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
`;break;case i.IMAGE_EDITS:t="azure"===y?`
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
prompt = "${S}"

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
`;break;case i.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${n||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case i.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case i.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${O}
${t}`}],339019)}]);