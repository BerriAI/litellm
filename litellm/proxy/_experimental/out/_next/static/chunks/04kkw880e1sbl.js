(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,571353,e=>{"use strict";e.i(602869);var t=e.i(221688);let i={"api-keys":"api-keys",models:"models-and-endpoints",api_ref:"api-reference","api-reference":"api-reference","llm-playground":"playground",projects:"projects",chat:"chat","access-groups":"access-groups",budgets:"budgets",workflows:"workflows","guardrails-monitor":"guardrails-monitor","mcp-servers":"mcp-servers","search-tools":"search-tools","tag-management":"tag-management","vector-stores":"vector-stores",memory:"memory",policies:"policies",guardrails:"guardrails",prompts:"prompts","tool-policies":"tool-policies",skills:"skills","claude-code-plugins":"skills",caching:"caching","cost-tracking":"cost-tracking","transform-request":"transform-request","ui-theme":"ui-theme",logs:"logs","admin-panel":"admin-panel","logging-and-alerts":"logging-and-alerts","model-hub-table":"model-hub-table",new_usage:"usage",usage:"old-usage",agents:"agents","router-settings":"router-settings",users:"users",teams:"teams",organizations:"organizations"};function o(){let e=t.serverRootPath&&"/"!==t.serverRootPath?`/${t.serverRootPath.replace(/^\/+|\/+$/g,"")}`:"";return`${e}/ui`}e.s(["MIGRATED_PAGES",0,i,"legacyKeyForPathname",0,function(e){let t=o(),n=(e.startsWith(t)?e.slice(t.length):e).replace(/^\/+|\/+$/g,"");for(let[e,t]of Object.entries(i))if(n===t)return e;return null},"legacyPageHref",0,function(e){return`${o()}/?page=${e}`},"migratedHref",0,function(e){return`${o()}/${e.replace(/^\/+/,"")}`}])},62478,e=>{"use strict";var t=e.i(602869);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},592392,e=>{"use strict";var t=e.i(62478),i=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("proxySettings"),n={PROXY_BASE_URL:"",PROXY_LOGOUT_URL:"",LITELLM_UI_API_DOC_BASE_URL:null};e.s(["default",0,function(e){let{data:r}=(0,i.useQuery)({queryKey:[...o.all,e],queryFn:()=>(0,t.fetchProxySettings)(e),enabled:!!e});return r??n}])},906579,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),o=e.i(361275),n=e.i(702779),r=e.i(763731),a=e.i(242064);e.i(296059);var s=e.i(915654),l=e.i(694758),c=e.i(183293),d=e.i(403541),u=e.i(246422),p=e.i(838378);let m=new l.Keyframes("antStatusProcessing",{"0%":{transform:"scale(0.8)",opacity:.5},"100%":{transform:"scale(2.4)",opacity:0}}),g=new l.Keyframes("antZoomBadgeIn",{"0%":{transform:"scale(0) translate(50%, -50%)",opacity:0},"100%":{transform:"scale(1) translate(50%, -50%)"}}),f=new l.Keyframes("antZoomBadgeOut",{"0%":{transform:"scale(1) translate(50%, -50%)"},"100%":{transform:"scale(0) translate(50%, -50%)",opacity:0}}),h=new l.Keyframes("antNoWrapperZoomBadgeIn",{"0%":{transform:"scale(0)",opacity:0},"100%":{transform:"scale(1)"}}),b=new l.Keyframes("antNoWrapperZoomBadgeOut",{"0%":{transform:"scale(1)"},"100%":{transform:"scale(0)",opacity:0}}),_=new l.Keyframes("antBadgeLoadingCircle",{"0%":{transformOrigin:"50%"},"100%":{transform:"translate(50%, -50%) rotate(360deg)",transformOrigin:"50%"}}),y=e=>{let{fontHeight:t,lineWidth:i,marginXS:o,colorBorderBg:n}=e,r=e.colorTextLightSolid,a=e.colorError,s=e.colorErrorHover;return(0,p.mergeToken)(e,{badgeFontHeight:t,badgeShadowSize:i,badgeTextColor:r,badgeColor:a,badgeColorHover:s,badgeShadowColor:n,badgeProcessingDuration:"1.2s",badgeRibbonOffset:o,badgeRibbonCornerTransform:"scaleY(0.75)",badgeRibbonCornerFilter:"brightness(75%)"})},x=e=>{let{fontSize:t,lineHeight:i,fontSizeSM:o,lineWidth:n}=e;return{indicatorZIndex:"auto",indicatorHeight:Math.round(t*i)-2*n,indicatorHeightSM:t,dotSize:o/2,textFontSize:o,textFontSizeSM:o,textFontWeight:"normal",statusSize:o/2}},v=(0,u.genStyleHooks)("Badge",e=>(e=>{let{componentCls:t,iconCls:i,antCls:o,badgeShadowSize:n,textFontSize:r,textFontSizeSM:a,statusSize:l,dotSize:u,textFontWeight:p,indicatorHeight:y,indicatorHeightSM:x,marginXS:v,calc:w}=e,S=`${o}-scroll-number`,j=(0,d.genPresetColor)(e,(e,{darkColor:i})=>({[`&${t} ${t}-color-${e}`]:{background:i,[`&:not(${t}-count)`]:{color:i},"a:hover &":{background:i}}}));return{[t]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"relative",display:"inline-block",width:"fit-content",lineHeight:1,[`${t}-count`]:{display:"inline-flex",justifyContent:"center",zIndex:e.indicatorZIndex,minWidth:y,height:y,color:e.badgeTextColor,fontWeight:p,fontSize:r,lineHeight:(0,s.unit)(y),whiteSpace:"nowrap",textAlign:"center",background:e.badgeColor,borderRadius:w(y).div(2).equal(),boxShadow:`0 0 0 ${(0,s.unit)(n)} ${e.badgeShadowColor}`,transition:`background ${e.motionDurationMid}`,a:{color:e.badgeTextColor},"a:hover":{color:e.badgeTextColor},"a:hover &":{background:e.badgeColorHover}},[`${t}-count-sm`]:{minWidth:x,height:x,fontSize:a,lineHeight:(0,s.unit)(x),borderRadius:w(x).div(2).equal()},[`${t}-multiple-words`]:{padding:`0 ${(0,s.unit)(e.paddingXS)}`,bdi:{unicodeBidi:"plaintext"}},[`${t}-dot`]:{zIndex:e.indicatorZIndex,width:u,minWidth:u,height:u,background:e.badgeColor,borderRadius:"100%",boxShadow:`0 0 0 ${(0,s.unit)(n)} ${e.badgeShadowColor}`},[`${t}-count, ${t}-dot, ${S}-custom-component`]:{position:"absolute",top:0,insetInlineEnd:0,transform:"translate(50%, -50%)",transformOrigin:"100% 0%",[`&${i}-spin`]:{animationName:_,animationDuration:"1s",animationIterationCount:"infinite",animationTimingFunction:"linear"}},[`&${t}-status`]:{lineHeight:"inherit",verticalAlign:"baseline",[`${t}-status-dot`]:{position:"relative",top:-1,display:"inline-block",width:l,height:l,verticalAlign:"middle",borderRadius:"50%"},[`${t}-status-success`]:{backgroundColor:e.colorSuccess},[`${t}-status-processing`]:{overflow:"visible",color:e.colorInfo,backgroundColor:e.colorInfo,borderColor:"currentcolor","&::after":{position:"absolute",top:0,insetInlineStart:0,width:"100%",height:"100%",borderWidth:n,borderStyle:"solid",borderColor:"inherit",borderRadius:"50%",animationName:m,animationDuration:e.badgeProcessingDuration,animationIterationCount:"infinite",animationTimingFunction:"ease-in-out",content:'""'}},[`${t}-status-default`]:{backgroundColor:e.colorTextPlaceholder},[`${t}-status-error`]:{backgroundColor:e.colorError},[`${t}-status-warning`]:{backgroundColor:e.colorWarning},[`${t}-status-text`]:{marginInlineStart:v,color:e.colorText,fontSize:e.fontSize}}}),j),{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:g,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`${t}-zoom-leave`]:{animationName:f,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack,animationFillMode:"both"},[`&${t}-not-a-wrapper`]:{[`${t}-zoom-appear, ${t}-zoom-enter`]:{animationName:h,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`${t}-zoom-leave`]:{animationName:b,animationDuration:e.motionDurationSlow,animationTimingFunction:e.motionEaseOutBack},[`&:not(${t}-status)`]:{verticalAlign:"middle"},[`${S}-custom-component, ${t}-count`]:{transform:"none"},[`${S}-custom-component, ${S}`]:{position:"relative",top:"auto",display:"block",transformOrigin:"50% 50%"}},[S]:{overflow:"hidden",transition:`all ${e.motionDurationMid} ${e.motionEaseOutBack}`,[`${S}-only`]:{position:"relative",display:"inline-block",height:y,transition:`all ${e.motionDurationSlow} ${e.motionEaseOutBack}`,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden",[`> p${S}-only-unit`]:{height:y,margin:0,WebkitTransformStyle:"preserve-3d",WebkitBackfaceVisibility:"hidden"}},[`${S}-symbol`]:{verticalAlign:"top"}},"&-rtl":{direction:"rtl",[`${t}-count, ${t}-dot, ${S}-custom-component`]:{transform:"translate(-50%, -50%)"}}})}})(y(e)),x),w=(0,u.genStyleHooks)(["Badge","Ribbon"],e=>(e=>{let{antCls:t,badgeFontHeight:i,marginXS:o,badgeRibbonOffset:n,calc:r}=e,a=`${t}-ribbon`,l=`${t}-ribbon-wrapper`,u=(0,d.genPresetColor)(e,(e,{darkColor:t})=>({[`&${a}-color-${e}`]:{background:t,color:t}}));return{[l]:{position:"relative"},[a]:Object.assign(Object.assign(Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"absolute",top:o,padding:`0 ${(0,s.unit)(e.paddingXS)}`,color:e.colorPrimary,lineHeight:(0,s.unit)(i),whiteSpace:"nowrap",backgroundColor:e.colorPrimary,borderRadius:e.borderRadiusSM,[`${a}-text`]:{color:e.badgeTextColor},[`${a}-corner`]:{position:"absolute",top:"100%",width:n,height:n,color:"currentcolor",border:`${(0,s.unit)(r(n).div(2).equal())} solid`,transform:e.badgeRibbonCornerTransform,transformOrigin:"top",filter:e.badgeRibbonCornerFilter}}),u),{[`&${a}-placement-end`]:{insetInlineEnd:r(n).mul(-1).equal(),borderEndEndRadius:0,[`${a}-corner`]:{insetInlineEnd:0,borderInlineEndColor:"transparent",borderBlockEndColor:"transparent"}},[`&${a}-placement-start`]:{insetInlineStart:r(n).mul(-1).equal(),borderEndStartRadius:0,[`${a}-corner`]:{insetInlineStart:0,borderBlockEndColor:"transparent",borderInlineStartColor:"transparent"}},"&-rtl":{direction:"rtl"}})}})(y(e)),x),S=e=>{let o,{prefixCls:n,value:r,current:a,offset:s=0}=e;return s&&(o={position:"absolute",top:`${s}00%`,left:0}),t.createElement("span",{style:o,className:(0,i.default)(`${n}-only-unit`,{current:a})},r)},j=e=>{let i,o,{prefixCls:n,count:r,value:a}=e,s=Number(a),l=Math.abs(r),[c,d]=t.useState(s),[u,p]=t.useState(l),m=()=>{d(s),p(l)};if(t.useEffect(()=>{let e=setTimeout(m,1e3);return()=>clearTimeout(e)},[s]),c===s||Number.isNaN(s)||Number.isNaN(c))i=[t.createElement(S,Object.assign({},e,{key:s,current:!0}))],o={transition:"none"};else{i=[];let n=s+10,r=[];for(let e=s;e<=n;e+=1)r.push(e);let a=u<l?1:-1,d=r.findIndex(e=>e%10===c);i=(a<0?r.slice(0,d+1):r.slice(d)).map((i,o)=>t.createElement(S,Object.assign({},e,{key:i,value:i%10,offset:a<0?o-d:o,current:o===d}))),o={transform:`translateY(${-function(e,t,i){let o=e,n=0;for(;(o+10)%10!==t;)o+=i,n+=i;return n}(c,s,a)}00%)`}}return t.createElement("span",{className:`${n}-only`,style:o,onTransitionEnd:m},i)};var $=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,o=Object.getOwnPropertySymbols(e);n<o.length;n++)0>t.indexOf(o[n])&&Object.prototype.propertyIsEnumerable.call(e,o[n])&&(i[o[n]]=e[o[n]]);return i};let C=t.forwardRef((e,o)=>{let{prefixCls:n,count:s,className:l,motionClassName:c,style:d,title:u,show:p,component:m="sup",children:g}=e,f=$(e,["prefixCls","count","className","motionClassName","style","title","show","component","children"]),{getPrefixCls:h}=t.useContext(a.ConfigContext),b=h("scroll-number",n),_=Object.assign(Object.assign({},f),{"data-show":p,style:d,className:(0,i.default)(b,l,c),title:u}),y=s;if(s&&Number(s)%1==0){let e=String(s).split("");y=t.createElement("bdi",null,e.map((i,o)=>t.createElement(j,{prefixCls:b,count:Number(s),value:i,key:e.length-o})))}return((null==d?void 0:d.borderColor)&&(_.style=Object.assign(Object.assign({},d),{boxShadow:`0 0 0 1px ${d.borderColor} inset`})),g)?(0,r.cloneElement)(g,e=>({className:(0,i.default)(`${b}-custom-component`,null==e?void 0:e.className,c)})):t.createElement(m,Object.assign({},_,{ref:o}),y)});var k=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,o=Object.getOwnPropertySymbols(e);n<o.length;n++)0>t.indexOf(o[n])&&Object.prototype.propertyIsEnumerable.call(e,o[n])&&(i[o[n]]=e[o[n]]);return i};let E=t.forwardRef((e,s)=>{var l,c,d,u,p;let{prefixCls:m,scrollNumberPrefixCls:g,children:f,status:h,text:b,color:_,count:y=null,overflowCount:x=99,dot:w=!1,size:S="default",title:j,offset:$,style:E,className:O,rootClassName:I,classNames:N,styles:z,showZero:R=!1}=e,T=k(e,["prefixCls","scrollNumberPrefixCls","children","status","text","color","count","overflowCount","dot","size","title","offset","style","className","rootClassName","classNames","styles","showZero"]),{getPrefixCls:A,direction:P,badge:M}=t.useContext(a.ConfigContext),L=A("badge",m),[D,B,H]=v(L),W=y>x?`${x}+`:y,U="0"===W||0===W||"0"===b||0===b,F=null===y||U&&!R,V=(null!=h||null!=_)&&F,G=null!=h||!U,q=w&&!U,K=q?"":W,Z=(0,t.useMemo)(()=>((null==K||""===K)&&(null==b||""===b)||U&&!R)&&!q,[K,U,R,q,b]),Y=(0,t.useRef)(y);Z||(Y.current=y);let J=Y.current,X=(0,t.useRef)(K);Z||(X.current=K);let Q=X.current,ee=(0,t.useRef)(q);Z||(ee.current=q);let et=(0,t.useMemo)(()=>{if(!$)return Object.assign(Object.assign({},null==M?void 0:M.style),E);let e={marginTop:$[1]};return"rtl"===P?e.left=Number.parseInt($[0],10):e.right=-Number.parseInt($[0],10),Object.assign(Object.assign(Object.assign({},e),null==M?void 0:M.style),E)},[P,$,E,null==M?void 0:M.style]),ei=null!=j?j:"string"==typeof J||"number"==typeof J?J:void 0,eo=!Z&&(0===b?R:!!b&&!0!==b),en=eo?t.createElement("span",{className:`${L}-status-text`},b):null,er=J&&"object"==typeof J?(0,r.cloneElement)(J,e=>({style:Object.assign(Object.assign({},et),e.style)})):void 0,ea=(0,n.isPresetColor)(_,!1),es=(0,i.default)(null==N?void 0:N.indicator,null==(l=null==M?void 0:M.classNames)?void 0:l.indicator,{[`${L}-status-dot`]:V,[`${L}-status-${h}`]:!!h,[`${L}-color-${_}`]:ea}),el={};_&&!ea&&(el.color=_,el.background=_);let ec=(0,i.default)(L,{[`${L}-status`]:V,[`${L}-not-a-wrapper`]:!f,[`${L}-rtl`]:"rtl"===P},O,I,null==M?void 0:M.className,null==(c=null==M?void 0:M.classNames)?void 0:c.root,null==N?void 0:N.root,B,H);if(!f&&V&&(b||G||!F)){let e=et.color;return D(t.createElement("span",Object.assign({},T,{className:ec,style:Object.assign(Object.assign(Object.assign({},null==z?void 0:z.root),null==(d=null==M?void 0:M.styles)?void 0:d.root),et)}),t.createElement("span",{className:es,style:Object.assign(Object.assign(Object.assign({},null==z?void 0:z.indicator),null==(u=null==M?void 0:M.styles)?void 0:u.indicator),el)}),eo&&t.createElement("span",{style:{color:e},className:`${L}-status-text`},b)))}return D(t.createElement("span",Object.assign({ref:s},T,{className:ec,style:Object.assign(Object.assign({},null==(p=null==M?void 0:M.styles)?void 0:p.root),null==z?void 0:z.root)}),f,t.createElement(o.default,{visible:!Z,motionName:`${L}-zoom`,motionAppear:!1,motionDeadline:1e3},({className:e})=>{var o,n;let r=A("scroll-number",g),a=ee.current,s=(0,i.default)(null==N?void 0:N.indicator,null==(o=null==M?void 0:M.classNames)?void 0:o.indicator,{[`${L}-dot`]:a,[`${L}-count`]:!a,[`${L}-count-sm`]:"small"===S,[`${L}-multiple-words`]:!a&&Q&&Q.toString().length>1,[`${L}-status-${h}`]:!!h,[`${L}-color-${_}`]:ea}),l=Object.assign(Object.assign(Object.assign({},null==z?void 0:z.indicator),null==(n=null==M?void 0:M.styles)?void 0:n.indicator),et);return _&&!ea&&((l=l||{}).background=_),t.createElement(C,{prefixCls:r,show:!Z,motionClassName:e,className:s,count:Q,title:ei,style:l,key:"scrollNumber"},er)}),en))});E.Ribbon=e=>{let{className:o,prefixCls:r,style:s,color:l,children:c,text:d,placement:u="end",rootClassName:p}=e,{getPrefixCls:m,direction:g}=t.useContext(a.ConfigContext),f=m("ribbon",r),h=`${f}-wrapper`,[b,_,y]=w(f,h),x=(0,n.isPresetColor)(l,!1),v=(0,i.default)(f,`${f}-placement-${u}`,{[`${f}-rtl`]:"rtl"===g,[`${f}-color-${l}`]:x},o),S={},j={};return l&&!x&&(S.background=l,j.color=l),b(t.createElement("div",{className:(0,i.default)(h,p,_,y)},c,t.createElement("div",{className:(0,i.default)(v,_),style:Object.assign(Object.assign({},S),s)},t.createElement("span",{className:`${f}-text`},d),t.createElement("div",{className:`${f}-corner`,style:j}))))},e.s(["Badge",0,E],906579)},115571,e=>{"use strict";let t="local-storage-change";e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",0,function(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))},"getLocalStorageItem",0,function(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}},"removeLocalStorageItem",0,function(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}},"setLocalStorageItem",0,function(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}])},477189,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M464 144H160c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V160c0-8.8-7.2-16-16-16zm-52 268H212V212h200v200zm452-268H560c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V160c0-8.8-7.2-16-16-16zm-52 268H612V212h200v200zM464 544H160c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V560c0-8.8-7.2-16-16-16zm-52 268H212V612h200v200zm452-268H560c-8.8 0-16 7.2-16 16v304c0 8.8 7.2 16 16 16h304c8.8 0 16-7.2 16-16V560c0-8.8-7.2-16-16-16zm-52 268H612V612h200v200z"}}]},name:"appstore",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:o}))});e.s(["AppstoreOutlined",0,r],477189)},371401,e=>{"use strict";var t=e.i(115571),i=e.i(271645);function o(e){let i=t=>{"disableUsageIndicator"===t.key&&e()},o=t=>{let{key:i}=t.detail;"disableUsageIndicator"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t.LOCAL_STORAGE_EVENT,o),()=>{window.removeEventListener("storage",i),window.removeEventListener(t.LOCAL_STORAGE_EVENT,o)}}function n(){return"true"===(0,t.getLocalStorageItem)("disableUsageIndicator")}e.s(["useDisableUsageIndicator",0,function(){return(0,i.useSyncExternalStore)(o,n)}])},283713,e=>{"use strict";var t=e.i(271645),i=e.i(602869),o=e.i(612256);let n="litellm_selected_worker_id";e.s(["useWorker",0,()=>{let{data:e}=(0,o.useUIConfig)(),r=e?.is_control_plane??!1,a=e?.workers??[],[s,l]=(0,t.useState)(()=>localStorage.getItem(n));(0,t.useEffect)(()=>{if(!s||0===a.length)return;let e=a.find(e=>e.worker_id===s);e&&(0,i.switchToWorkerUrl)(e.url)},[s,a]);let c=a.find(e=>e.worker_id===s)??null,d=(0,t.useCallback)(e=>{let t=a.find(t=>t.worker_id===e);t&&(l(e),localStorage.setItem(n,e),(0,i.switchToWorkerUrl)(t.url))},[a]);return{isControlPlane:r,workers:a,selectedWorkerId:s,selectedWorker:c,selectWorker:d,disconnectFromWorker:(0,t.useCallback)(()=>{l(null),localStorage.removeItem(n),(0,i.switchToWorkerUrl)(null)},[])}}])},295320,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M704 446H320c-4.4 0-8 3.6-8 8v402c0 4.4 3.6 8 8 8h384c4.4 0 8-3.6 8-8V454c0-4.4-3.6-8-8-8zm-328 64h272v117H376V510zm272 290H376V683h272v117z"}},{tag:"path",attrs:{d:"M424 748a32 32 0 1064 0 32 32 0 10-64 0zm0-178a32 32 0 1064 0 32 32 0 10-64 0z"}},{tag:"path",attrs:{d:"M811.4 368.9C765.6 248 648.9 162 512.2 162S258.8 247.9 213 368.8C126.9 391.5 63.5 470.2 64 563.6 64.6 668 145.6 752.9 247.6 762c4.7.4 8.7-3.3 8.7-8v-60.4c0-4-3-7.4-7-7.9-27-3.4-52.5-15.2-72.1-34.5-24-23.5-37.2-55.1-37.2-88.6 0-28 9.1-54.4 26.2-76.4 16.7-21.4 40.2-36.9 66.1-43.7l37.9-10 13.9-36.7c8.6-22.8 20.6-44.2 35.7-63.5 14.9-19.2 32.6-36 52.4-50 41.1-28.9 89.5-44.2 140-44.2s98.9 15.3 140 44.3c19.9 14 37.5 30.8 52.4 50 15.1 19.3 27.1 40.7 35.7 63.5l13.8 36.6 37.8 10c54.2 14.4 92.1 63.7 92.1 120 0 33.6-13.2 65.1-37.2 88.6-19.5 19.2-44.9 31.1-71.9 34.5-4 .5-6.9 3.9-6.9 7.9V754c0 4.7 4.1 8.4 8.8 8 101.7-9.2 182.5-94 183.2-198.2.6-93.4-62.7-172.1-148.6-194.9z"}}]},name:"cloud-server",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:o}))});e.s(["CloudServerOutlined",0,r],295320)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:o}))});e.s(["SafetyOutlined",0,r],602073)},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return n}});let o=e.r(271645);function n(e,t){let i=(0,o.useRef)(null),n=(0,o.useRef)(null);return(0,o.useCallback)(o=>{if(null===o){let e=i.current;e&&(i.current=null,e());let t=n.current;t&&(n.current=null,t())}else e&&(i.current=r(e,o)),t&&(n.current=r(t,o))},[e,t])}function r(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(602869);let n=(0,i.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:r})=>{let[a,s]=(0,i.useState)(null),[l,c]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,o.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let e=await i.json();e.values?.logo_url&&s(e.values.logo_url),e.values?.favicon_url&&c(e.values.favicon_url)}}catch(e){console.warn("Failed to load theme settings from backend:",e)}})()},[]),(0,i.useEffect)(()=>{if(l){let e=document.querySelectorAll("link[rel*='icon']");if(e.length>0)e.forEach(e=>{e.href=l});else{let e=document.createElement("link");e.rel="icon",e.href=l,document.head.appendChild(e)}}},[l]),(0,t.jsx)(n.Provider,{value:{logoUrl:a,setLogoUrl:s,faviconUrl:l,setFaviconUrl:c},children:e})},"useTheme",0,()=>{let e=(0,i.useContext)(n);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:o}))});e.s(["CrownOutlined",0,r],100486)},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",0,t])},174886,e=>{"use strict";var t=e.i(991124);e.s(["Copy",()=>t.default])},339019,865361,e=>{"use strict";var t,i,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i.INTERACTIONS="interactions",i);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>Object.values(o).includes(e)?r[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:o,apiKey:r,inputMessage:a,chatHistory:s,selectedTags:l,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:p,mcpServers:m,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:_,proxySettings:y}=e,x="session"===i?o:r,v=window.location.origin,w=y?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?v=w:y?.PROXY_BASE_URL&&(v=y.PROXY_BASE_URL);let S=a||"Your prompt here",j=S.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),$=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),c.length>0&&(C.vector_stores=c),d.length>0&&(C.guardrails=d),u.length>0&&(C.policies=u);let k=b||"your-model-name",E="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(h){case n.CHAT:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=$.length>0?$:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${k}",
    messages=${JSON.stringify(o,null,4)}${i}
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
#     ]${i}
# )
# print(response_with_file)
`;break}case n.RESPONSES:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=$.length>0?$:[{role:"user",content:S}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${k}",
    input=${JSON.stringify(o,null,4)}${i}
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
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case n.IMAGE:t="azure"===_?`
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
	prompt="${a}",
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
`;break;case n.IMAGE_EDITS:t="azure"===_?`
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
`;break;case n.EMBEDDINGS:t=`
response = client.embeddings.create(
	input="${a||"Your string here"}",
	model="${k}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case n.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${k}",
	file=audio_file${a?`,
	prompt="${a.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${k}",
	input="${a||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${k}",
#     input="${a||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${E}
${t}`}],339019)},798496,e=>{"use strict";var t=e.i(843476),i=e.i(152990),o=e.i(682830),n=e.i(271645),r=e.i(269200),a=e.i(427612),s=e.i(64848),l=e.i(942232),c=e.i(496020),d=e.i(977572),u=e.i(94629),p=e.i(360820),m=e.i(871943);e.s(["ModelDataTable",0,function({data:e=[],columns:g,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:_,enablePagination:y=!1,onRowClick:x}){let[v,w]=n.default.useState(h),[S]=n.default.useState("onChange"),[j,$]=n.default.useState({}),[C,k]=n.default.useState({}),E=(0,i.useReactTable)({data:e,columns:g,state:{sorting:v,columnSizing:j,columnVisibility:C,...y&&b?{pagination:b}:{}},columnResizeMode:S,onSortingChange:w,onColumnSizingChange:$,onColumnVisibilityChange:k,...y&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...y?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(r.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:E.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(a.TableHead,{children:E.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(s.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,i.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(p.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(m.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(l.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):E.getRowModel().rows.length>0?E.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{onClick:()=>x?.(e.original),className:x?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,i.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}])},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(447566),n=e.i(166406),r=e.i(492030),a=e.i(596239);let s=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,l=e=>e.trim().replace(/\/+$/,""),c=/\.(md|markdown|txt|json|ya?ml|toml)$/i,d=/^\d{1,3}(\.\d{1,3}){3}$/,u=/^[A-Za-z0-9-]+$/,p=/^[A-Za-z0-9._-]+$/,m=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},f=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),h=e=>{let{source:t}=e;return"github"===t.source&&t.repo?`/plugin marketplace add ${t.repo}`:("url"===t.source||"git-subdir"===t.source)&&t.url?`/plugin marketplace add ${t.url}`:`/plugin marketplace add ${e.name}`};e.s(["formatInstallCommand",0,h,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=l(e);return""!==t&&s.test(t)},"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let i=(e=>{let t,i=e.trim();if(""===i||i.startsWith("//"))return null;let o=/^[a-z][a-z0-9+.-]*:\/\//i.test(i)?i:`https://${i}`;try{t=new URL(o)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||d.test(t.hostname)?null:t})(e);if(!i)return null;if("github.com"===i.hostname.replace(/^www\./,""))return((e,t)=>{let i=m(e);if(i.length<2)return null;let o=i[0],n=i[1].replace(/\.git$/,"");if(!u.test(o)||!p.test(n))return null;let r=`${o}/${n}`,a=`https://github.com/${r}`,d={parsed:{source:"github",repo:r},label:`GitHub repo — ${r}`,suggestedName:f(n)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=g(e.join("/")),o=c.test(t)?e.slice(0,-1):e;if(0===o.length)return d;let n=l(o.join("/"));return s.test(n)?{parsed:{source:"git-subdir",url:a,path:n},label:`GitHub subdir — ${r} @ ${n}`,suggestedName:f(g(n))}:null}if(2!==i.length)return null;let h=l(t??"");return""!==h?s.test(h)?{parsed:{source:"git-subdir",url:a,path:h},label:`GitHub subdir — ${r} @ ${h}`,suggestedName:f(g(h))}:null:d})(i,t);if(m(i).length<2)return null;let o=`${i.protocol}//${i.host}${i.pathname.replace(/\/+$/,"")}`,n=l(t??"");return""!==n?s.test(n)?{parsed:{source:"git-subdir",url:o,path:n},label:`Git subdir — ${o} @ ${n}`,suggestedName:f(g(n))}:null:{parsed:{source:"url",url:o},label:`Git repo — ${o}`,suggestedName:f(g(i.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:s})=>{let l,[c,d]=(0,i.useState)("overview"),[u,p]=(0,i.useState)(null),m=(e,t)=>{navigator.clipboard.writeText(e),p(t),setTimeout(()=>p(null),2e3)},g="github"===(l=e.source).source&&l.repo?`https://github.com/${l.repo}`:"git-subdir"===l.source&&l.url?l.path?`${l.url}/tree/main/${l.path}`:l.url:"url"===l.source&&l.url?l.url:null,f=h(e),b=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:s,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(o.ArrowLeftOutlined,{style:{fontSize:11}}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),style:{padding:"12px 20px",fontSize:14,color:c===e.key?"#1a73e8":"#5f6368",borderBottom:c===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:c===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===c&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:b.map((e,i)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),g&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:g,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[g.replace("https://",""),(0,t.jsx)(a.LinkOutlined,{style:{fontSize:11,flexShrink:0}})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>m(f,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===u?(0,t.jsx)(r.CheckOutlined,{}):(0,t.jsx)(n.CopyOutlined,{}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:f})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===c&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to"," ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," ","to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>{m(JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2),"settings")},style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===u?(0,t.jsx)(r.CheckOutlined,{}):(0,t.jsx)(n.CopyOutlined,{}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:JSON.stringify({extraKnownMarketplaces:{"my-org":{source:"url",url:`${window.location.origin}/claude-code/marketplace.json`}}},null,2)})]})]})]})}],652272)}]);