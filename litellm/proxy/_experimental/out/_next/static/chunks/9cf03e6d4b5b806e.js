(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,209261,e=>{"use strict";e.s(["extractCategories",0,e=>{let t=new Set;return e.forEach(e=>{e.category&&""!==e.category.trim()&&t.add(e.category)}),["All",...Array.from(t).sort(),"Other"]},"filterPluginsByCategory",0,(e,t)=>"All"===t?e:"Other"===t?e.filter(e=>!e.category||""===e.category.trim()):e.filter(e=>e.category===t),"filterPluginsBySearch",0,(e,t)=>{if(!t||""===t.trim())return e;let i=t.toLowerCase().trim();return e.filter(e=>{let t=e.name.toLowerCase().includes(i),o=e.description?.toLowerCase().includes(i)||!1,r=e.keywords?.some(e=>e.toLowerCase().includes(i))||!1;return t||o||r})},"formatDateString",0,e=>{if(!e)return"N/A";try{return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"})}catch(e){return"Invalid date"}},"formatInstallCommand",0,e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"url"===e.source&&e.url?e.url:"Unknown source","getSourceLink",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:"url"===e.source&&e.url?e.url:null,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},928685,e=>{"use strict";var t=e.i(38953);e.s(["SearchOutlined",()=>t.default])},280898,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(121229),o=e.i(864517),r=e.i(343794),n=e.i(931067),a=e.i(209428),l=e.i(211577),s=e.i(703923),c=e.i(404948),d=["className","prefixCls","style","active","status","iconPrefix","icon","wrapperStyle","stepNumber","disabled","description","title","subTitle","progressDot","stepIcon","tailContent","icons","stepIndex","onStepClick","onClick","render"];function u(e){return"string"==typeof e}let m=function(e){var i,o,m,g,p,f=e.className,h=e.prefixCls,b=e.style,v=e.active,$=e.status,_=e.iconPrefix,y=e.icon,w=(e.wrapperStyle,e.stepNumber),x=e.disabled,C=e.description,I=e.title,S=e.subTitle,k=e.progressDot,A=e.stepIcon,E=e.tailContent,O=e.icons,j=e.stepIndex,T=e.onStepClick,N=e.onClick,M=e.render,z=(0,s.default)(e,d),P={};T&&!x&&(P.role="button",P.tabIndex=0,P.onClick=function(e){null==N||N(e),T(j)},P.onKeyDown=function(e){var t=e.which;(t===c.default.ENTER||t===c.default.SPACE)&&T(j)});var L=$||"wait",R=(0,r.default)("".concat(h,"-item"),"".concat(h,"-item-").concat(L),f,(p={},(0,l.default)(p,"".concat(h,"-item-custom"),y),(0,l.default)(p,"".concat(h,"-item-active"),v),(0,l.default)(p,"".concat(h,"-item-disabled"),!0===x),p)),D=(0,a.default)({},b),H=t.createElement("div",(0,n.default)({},z,{className:R,style:D}),t.createElement("div",(0,n.default)({onClick:N},P,{className:"".concat(h,"-item-container")}),t.createElement("div",{className:"".concat(h,"-item-tail")},E),t.createElement("div",{className:"".concat(h,"-item-icon")},(m=(0,r.default)("".concat(h,"-icon"),"".concat(_,"icon"),(i={},(0,l.default)(i,"".concat(_,"icon-").concat(y),y&&u(y)),(0,l.default)(i,"".concat(_,"icon-check"),!y&&"finish"===$&&(O&&!O.finish||!O)),(0,l.default)(i,"".concat(_,"icon-cross"),!y&&"error"===$&&(O&&!O.error||!O)),i)),g=t.createElement("span",{className:"".concat(h,"-icon-dot")}),o=k?"function"==typeof k?t.createElement("span",{className:"".concat(h,"-icon")},k(g,{index:w-1,status:$,title:I,description:C})):t.createElement("span",{className:"".concat(h,"-icon")},g):y&&!u(y)?t.createElement("span",{className:"".concat(h,"-icon")},y):O&&O.finish&&"finish"===$?t.createElement("span",{className:"".concat(h,"-icon")},O.finish):O&&O.error&&"error"===$?t.createElement("span",{className:"".concat(h,"-icon")},O.error):y||"finish"===$||"error"===$?t.createElement("span",{className:m}):t.createElement("span",{className:"".concat(h,"-icon")},w),A&&(o=A({index:w-1,status:$,title:I,description:C,node:o})),o)),t.createElement("div",{className:"".concat(h,"-item-content")},t.createElement("div",{className:"".concat(h,"-item-title")},I,S&&t.createElement("div",{title:"string"==typeof S?S:void 0,className:"".concat(h,"-item-subtitle")},S)),C&&t.createElement("div",{className:"".concat(h,"-item-description")},C))));return M&&(H=M(H)||null),H};var g=["prefixCls","style","className","children","direction","type","labelPlacement","iconPrefix","status","size","current","progressDot","stepIcon","initial","icons","onChange","itemRender","items"];function p(e){var i,o=e.prefixCls,c=void 0===o?"rc-steps":o,d=e.style,u=void 0===d?{}:d,p=e.className,f=(e.children,e.direction),h=e.type,b=void 0===h?"default":h,v=e.labelPlacement,$=e.iconPrefix,_=void 0===$?"rc":$,y=e.status,w=void 0===y?"process":y,x=e.size,C=e.current,I=void 0===C?0:C,S=e.progressDot,k=e.stepIcon,A=e.initial,E=void 0===A?0:A,O=e.icons,j=e.onChange,T=e.itemRender,N=e.items,M=(0,s.default)(e,g),z="inline"===b,P=z||void 0!==S&&S,L=z||void 0===f?"horizontal":f,R=z?void 0:x,D=(0,r.default)(c,"".concat(c,"-").concat(L),p,(i={},(0,l.default)(i,"".concat(c,"-").concat(R),R),(0,l.default)(i,"".concat(c,"-label-").concat(P?"vertical":void 0===v?"horizontal":v),"horizontal"===L),(0,l.default)(i,"".concat(c,"-dot"),!!P),(0,l.default)(i,"".concat(c,"-navigation"),"navigation"===b),(0,l.default)(i,"".concat(c,"-inline"),z),i)),H=function(e){j&&I!==e&&j(e)};return t.default.createElement("div",(0,n.default)({className:D,style:u},M),(void 0===N?[]:N).filter(function(e){return e}).map(function(e,i){var o=(0,a.default)({},e),r=E+i;return"error"===w&&i===I-1&&(o.className="".concat(c,"-next-error")),o.status||(r===I?o.status=w:r<I?o.status="finish":o.status="wait"),z&&(o.icon=void 0,o.subTitle=void 0),!o.render&&T&&(o.render=function(e){return T(o,e)}),t.default.createElement(m,(0,n.default)({},o,{active:r===I,stepNumber:r+1,stepIndex:r,key:r,prefixCls:c,iconPrefix:_,wrapperStyle:u,progressDot:P,stepIcon:k,icons:O,onStepClick:j&&H}))}))}p.Step=m;var f=e.i(242064),h=e.i(517455),b=e.i(150073),v=e.i(309821),$=e.i(491816);e.i(296059);var _=e.i(915654),y=e.i(183293),w=e.i(246422),x=e.i(838378);let C=(e,t)=>{let i=`${t.componentCls}-item`,o=`${e}IconColor`,r=`${e}TitleColor`,n=`${e}DescriptionColor`,a=`${e}TailColor`,l=`${e}IconBgColor`,s=`${e}IconBorderColor`,c=`${e}DotColor`;return{[`${i}-${e} ${i}-icon`]:{backgroundColor:t[l],borderColor:t[s],[`> ${t.componentCls}-icon`]:{color:t[o],[`${t.componentCls}-icon-dot`]:{background:t[c]}}},[`${i}-${e}${i}-custom ${i}-icon`]:{[`> ${t.componentCls}-icon`]:{color:t[c]}},[`${i}-${e} > ${i}-container > ${i}-content > ${i}-title`]:{color:t[r],"&::after":{backgroundColor:t[a]}},[`${i}-${e} > ${i}-container > ${i}-content > ${i}-description`]:{color:t[n]},[`${i}-${e} > ${i}-container > ${i}-tail::after`]:{backgroundColor:t[a]}}},I=(0,w.genStyleHooks)("Steps",e=>{let{colorTextDisabled:t,controlHeightLG:i,colorTextLightSolid:o,colorText:r,colorPrimary:n,colorTextDescription:a,colorTextQuaternary:l,colorError:s,colorBorderSecondary:c,colorSplit:d}=e;return(e=>{let{componentCls:t}=e;return{[t]:Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({},(0,y.resetComponent)(e)),{display:"flex",width:"100%",fontSize:0,textAlign:"initial"}),(e=>{let{componentCls:t,motionDurationSlow:i}=e,o=`${t}-item`,r=`${o}-icon`;return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({[o]:{position:"relative",display:"inline-block",flex:1,overflow:"hidden",verticalAlign:"top","&:last-child":{flex:"none",[`> ${o}-container > ${o}-tail, > ${o}-container >  ${o}-content > ${o}-title::after`]:{display:"none"}}},[`${o}-container`]:{outline:"none",[`&:focus-visible ${r}`]:(0,y.genFocusOutline)(e)},[`${r}, ${o}-content`]:{display:"inline-block",verticalAlign:"top"},[r]:{width:e.iconSize,height:e.iconSize,marginTop:0,marginBottom:0,marginInlineStart:0,marginInlineEnd:e.marginXS,fontSize:e.iconFontSize,fontFamily:e.fontFamily,lineHeight:(0,_.unit)(e.iconSize),textAlign:"center",borderRadius:e.iconSize,border:`${(0,_.unit)(e.lineWidth)} ${e.lineType} transparent`,transition:`background-color ${i}, border-color ${i}`,[`${t}-icon`]:{position:"relative",top:e.iconTop,color:e.colorPrimary,lineHeight:1}},[`${o}-tail`]:{position:"absolute",top:e.calc(e.iconSize).div(2).equal(),insetInlineStart:0,width:"100%","&::after":{display:"inline-block",width:"100%",height:e.lineWidth,background:e.colorSplit,borderRadius:e.lineWidth,transition:`background ${i}`,content:'""'}},[`${o}-title`]:{position:"relative",display:"inline-block",paddingInlineEnd:e.padding,color:e.colorText,fontSize:e.fontSizeLG,lineHeight:(0,_.unit)(e.titleLineHeight),"&::after":{position:"absolute",top:e.calc(e.titleLineHeight).div(2).equal(),insetInlineStart:"100%",display:"block",width:9999,height:e.lineWidth,background:e.processTailColor,content:'""'}},[`${o}-subtitle`]:{display:"inline",marginInlineStart:e.marginXS,color:e.colorTextDescription,fontWeight:"normal",fontSize:e.fontSize},[`${o}-description`]:{color:e.colorTextDescription,fontSize:e.fontSize}},C("wait",e)),C("process",e)),{[`${o}-process > ${o}-container > ${o}-title`]:{fontWeight:e.fontWeightStrong}}),C("finish",e)),C("error",e)),{[`${o}${t}-next-error > ${t}-item-title::after`]:{background:e.colorError},[`${o}-disabled`]:{cursor:"not-allowed"}})})(e)),(e=>{let{componentCls:t,motionDurationSlow:i}=e;return{[`& ${t}-item`]:{[`&:not(${t}-item-active)`]:{[`& > ${t}-item-container[role='button']`]:{cursor:"pointer",[`${t}-item`]:{[`&-title, &-subtitle, &-description, &-icon ${t}-icon`]:{transition:`color ${i}`}},"&:hover":{[`${t}-item`]:{"&-title, &-subtitle, &-description":{color:e.colorPrimary}}}},[`&:not(${t}-item-process)`]:{[`& > ${t}-item-container[role='button']:hover`]:{[`${t}-item`]:{"&-icon":{borderColor:e.colorPrimary,[`${t}-icon`]:{color:e.colorPrimary}}}}}}},[`&${t}-horizontal:not(${t}-label-vertical)`]:{[`${t}-item`]:{paddingInlineStart:e.padding,whiteSpace:"nowrap","&:first-child":{paddingInlineStart:0},[`&:last-child ${t}-item-title`]:{paddingInlineEnd:0},"&-tail":{display:"none"},"&-description":{maxWidth:e.descriptionMaxWidth,whiteSpace:"normal"}}}}})(e)),(e=>{let{componentCls:t,customIconTop:i,customIconSize:o,customIconFontSize:r}=e;return{[`${t}-item-custom`]:{[`> ${t}-item-container > ${t}-item-icon`]:{height:"auto",background:"none",border:0,[`> ${t}-icon`]:{top:i,width:o,height:o,fontSize:r,lineHeight:(0,_.unit)(o)}}},[`&:not(${t}-vertical)`]:{[`${t}-item-custom`]:{[`${t}-item-icon`]:{width:"auto",background:"none"}}}}})(e)),(e=>{let{componentCls:t,iconSizeSM:i,fontSizeSM:o,fontSize:r,colorTextDescription:n}=e;return{[`&${t}-small`]:{[`&${t}-horizontal:not(${t}-label-vertical) ${t}-item`]:{paddingInlineStart:e.paddingSM,"&:first-child":{paddingInlineStart:0}},[`${t}-item-icon`]:{width:i,height:i,marginTop:0,marginBottom:0,marginInline:`0 ${(0,_.unit)(e.marginXS)}`,fontSize:o,lineHeight:(0,_.unit)(i),textAlign:"center",borderRadius:i},[`${t}-item-title`]:{paddingInlineEnd:e.paddingSM,fontSize:r,lineHeight:(0,_.unit)(i),"&::after":{top:e.calc(i).div(2).equal()}},[`${t}-item-description`]:{color:n,fontSize:r},[`${t}-item-tail`]:{top:e.calc(i).div(2).sub(e.paddingXXS).equal()},[`${t}-item-custom ${t}-item-icon`]:{width:"inherit",height:"inherit",lineHeight:"inherit",background:"none",border:0,borderRadius:0,[`> ${t}-icon`]:{fontSize:i,lineHeight:(0,_.unit)(i),transform:"none"}}}}})(e)),(e=>{let{componentCls:t,iconSizeSM:i,iconSize:o}=e;return{[`&${t}-vertical`]:{display:"flex",flexDirection:"column",[`> ${t}-item`]:{display:"block",flex:"1 0 auto",paddingInlineStart:0,overflow:"visible",[`${t}-item-icon`]:{float:"left",marginInlineEnd:e.margin},[`${t}-item-content`]:{display:"block",minHeight:e.calc(e.controlHeight).mul(1.5).equal(),overflow:"hidden"},[`${t}-item-title`]:{lineHeight:(0,_.unit)(o)},[`${t}-item-description`]:{paddingBottom:e.paddingSM}},[`> ${t}-item > ${t}-item-container > ${t}-item-tail`]:{position:"absolute",top:0,insetInlineStart:e.calc(o).div(2).sub(e.lineWidth).equal(),width:e.lineWidth,height:"100%",padding:`${(0,_.unit)(e.calc(e.marginXXS).mul(1.5).add(o).equal())} 0 ${(0,_.unit)(e.calc(e.marginXXS).mul(1.5).equal())}`,"&::after":{width:e.lineWidth,height:"100%"}},[`> ${t}-item:not(:last-child) > ${t}-item-container > ${t}-item-tail`]:{display:"block"},[` > ${t}-item > ${t}-item-container > ${t}-item-content > ${t}-item-title`]:{"&::after":{display:"none"}},[`&${t}-small ${t}-item-container`]:{[`${t}-item-tail`]:{position:"absolute",top:0,insetInlineStart:e.calc(i).div(2).sub(e.lineWidth).equal(),padding:`${(0,_.unit)(e.calc(e.marginXXS).mul(1.5).add(i).equal())} 0 ${(0,_.unit)(e.calc(e.marginXXS).mul(1.5).equal())}`},[`${t}-item-title`]:{lineHeight:(0,_.unit)(i)}}}}})(e)),(e=>{let{componentCls:t}=e,i=`${t}-item`;return{[`${t}-horizontal`]:{[`${i}-tail`]:{transform:"translateY(-50%)"}}}})(e)),(e=>{let{componentCls:t,iconSize:i,lineHeight:o,iconSizeSM:r}=e;return{[`&${t}-label-vertical`]:{[`${t}-item`]:{overflow:"visible","&-tail":{marginInlineStart:e.calc(i).div(2).add(e.controlHeightLG).equal(),padding:`0 ${(0,_.unit)(e.paddingLG)}`},"&-content":{display:"block",width:e.calc(i).div(2).add(e.controlHeightLG).mul(2).equal(),marginTop:e.marginSM,textAlign:"center"},"&-icon":{display:"inline-block",marginInlineStart:e.controlHeightLG},"&-title":{paddingInlineEnd:0,paddingInlineStart:0,"&::after":{display:"none"}},"&-subtitle":{display:"block",marginBottom:e.marginXXS,marginInlineStart:0,lineHeight:o}},[`&${t}-small:not(${t}-dot)`]:{[`${t}-item`]:{"&-icon":{marginInlineStart:e.calc(i).sub(r).div(2).add(e.controlHeightLG).equal()}}}}}})(e)),(e=>{let{componentCls:t,descriptionMaxWidth:i,lineHeight:o,dotCurrentSize:r,dotSize:n,motionDurationSlow:a}=e;return{[`&${t}-dot, &${t}-dot${t}-small`]:{[`${t}-item`]:{"&-title":{lineHeight:o},"&-tail":{top:e.calc(e.dotSize).sub(e.calc(e.lineWidth).mul(3).equal()).div(2).equal(),width:"100%",marginTop:0,marginBottom:0,marginInline:`${(0,_.unit)(e.calc(i).div(2).equal())} 0`,padding:0,"&::after":{width:`calc(100% - ${(0,_.unit)(e.calc(e.marginSM).mul(2).equal())})`,height:e.calc(e.lineWidth).mul(3).equal(),marginInlineStart:e.marginSM}},"&-icon":{width:n,height:n,marginInlineStart:e.calc(e.descriptionMaxWidth).sub(n).div(2).equal(),paddingInlineEnd:0,lineHeight:(0,_.unit)(n),background:"transparent",border:0,[`${t}-icon-dot`]:{position:"relative",float:"left",width:"100%",height:"100%",borderRadius:100,transition:`all ${a}`,"&::after":{position:"absolute",top:e.calc(e.marginSM).mul(-1).equal(),insetInlineStart:e.calc(n).sub(e.calc(e.controlHeightLG).mul(1.5).equal()).div(2).equal(),width:e.calc(e.controlHeightLG).mul(1.5).equal(),height:e.controlHeight,background:"transparent",content:'""'}}},"&-content":{width:i},[`&-process ${t}-item-icon`]:{position:"relative",top:e.calc(n).sub(r).div(2).equal(),width:r,height:r,lineHeight:(0,_.unit)(r),background:"none",marginInlineStart:e.calc(e.descriptionMaxWidth).sub(r).div(2).equal()},[`&-process ${t}-icon`]:{[`&:first-child ${t}-icon-dot`]:{insetInlineStart:0}}}},[`&${t}-vertical${t}-dot`]:{[`${t}-item-icon`]:{marginTop:e.calc(e.controlHeight).sub(n).div(2).equal(),marginInlineStart:0,background:"none"},[`${t}-item-process ${t}-item-icon`]:{marginTop:e.calc(e.controlHeight).sub(r).div(2).equal(),top:0,insetInlineStart:e.calc(n).sub(r).div(2).equal(),marginInlineStart:0},[`${t}-item > ${t}-item-container > ${t}-item-tail`]:{top:e.calc(e.controlHeight).sub(n).div(2).equal(),insetInlineStart:0,margin:0,padding:`${(0,_.unit)(e.calc(n).add(e.paddingXS).equal())} 0 ${(0,_.unit)(e.paddingXS)}`,"&::after":{marginInlineStart:e.calc(n).sub(e.lineWidth).div(2).equal()}},[`&${t}-small`]:{[`${t}-item-icon`]:{marginTop:e.calc(e.controlHeightSM).sub(n).div(2).equal()},[`${t}-item-process ${t}-item-icon`]:{marginTop:e.calc(e.controlHeightSM).sub(r).div(2).equal()},[`${t}-item > ${t}-item-container > ${t}-item-tail`]:{top:e.calc(e.controlHeightSM).sub(n).div(2).equal()}},[`${t}-item:first-child ${t}-icon-dot`]:{insetInlineStart:0},[`${t}-item-content`]:{width:"inherit"}}}})(e)),(e=>{let{componentCls:t,navContentMaxWidth:i,navArrowColor:o,stepsNavActiveColor:r,motionDurationSlow:n}=e;return{[`&${t}-navigation`]:{paddingTop:e.paddingSM,[`&${t}-small`]:{[`${t}-item`]:{"&-container":{marginInlineStart:e.calc(e.marginSM).mul(-1).equal()}}},[`${t}-item`]:{overflow:"visible",textAlign:"center","&-container":{display:"inline-block",height:"100%",marginInlineStart:e.calc(e.margin).mul(-1).equal(),paddingBottom:e.paddingSM,textAlign:"start",transition:`opacity ${n}`,[`${t}-item-content`]:{maxWidth:i},[`${t}-item-title`]:Object.assign(Object.assign({maxWidth:"100%",paddingInlineEnd:0},y.textEllipsis),{"&::after":{display:"none"}})},[`&:not(${t}-item-active)`]:{[`${t}-item-container[role='button']`]:{cursor:"pointer","&:hover":{opacity:.85}}},"&:last-child":{flex:1,"&::after":{display:"none"}},"&::after":{position:"absolute",top:`calc(50% - ${(0,_.unit)(e.calc(e.paddingSM).div(2).equal())})`,insetInlineStart:"100%",display:"inline-block",width:e.fontSizeIcon,height:e.fontSizeIcon,borderTop:`${(0,_.unit)(e.lineWidth)} ${e.lineType} ${o}`,borderBottom:"none",borderInlineStart:"none",borderInlineEnd:`${(0,_.unit)(e.lineWidth)} ${e.lineType} ${o}`,transform:"translateY(-50%) translateX(-50%) rotate(45deg)",content:'""'},"&::before":{position:"absolute",bottom:0,insetInlineStart:"50%",display:"inline-block",width:0,height:e.lineWidthBold,backgroundColor:r,transition:`width ${n}, inset-inline-start ${n}`,transitionTimingFunction:"ease-out",content:'""'}},[`${t}-item${t}-item-active::before`]:{insetInlineStart:0,width:"100%"}},[`&${t}-navigation${t}-vertical`]:{[`> ${t}-item`]:{marginInlineEnd:0,"&::before":{display:"none"},[`&${t}-item-active::before`]:{top:0,insetInlineEnd:0,insetInlineStart:"unset",display:"block",width:e.calc(e.lineWidth).mul(3).equal(),height:`calc(100% - ${(0,_.unit)(e.marginLG)})`},"&::after":{position:"relative",insetInlineStart:"50%",display:"block",width:e.calc(e.controlHeight).mul(.25).equal(),height:e.calc(e.controlHeight).mul(.25).equal(),marginBottom:e.marginXS,textAlign:"center",transform:"translateY(-50%) translateX(-50%) rotate(135deg)"},"&:last-child":{"&::after":{display:"none"}},[`> ${t}-item-container > ${t}-item-tail`]:{visibility:"hidden"}}},[`&${t}-navigation${t}-horizontal`]:{[`> ${t}-item > ${t}-item-container > ${t}-item-tail`]:{visibility:"hidden"}}}})(e)),(e=>{let{componentCls:t}=e;return{[`&${t}-rtl`]:{direction:"rtl",[`${t}-item`]:{"&-subtitle":{float:"left"}},[`&${t}-navigation`]:{[`${t}-item::after`]:{transform:"rotate(-45deg)"}},[`&${t}-vertical`]:{[`> ${t}-item`]:{"&::after":{transform:"rotate(225deg)"},[`${t}-item-icon`]:{float:"right"}}},[`&${t}-dot`]:{[`${t}-item-icon ${t}-icon-dot, &${t}-small ${t}-item-icon ${t}-icon-dot`]:{float:"right"}}}}})(e)),(e=>{let{antCls:t,componentCls:i,iconSize:o,iconSizeSM:r,processIconColor:n,marginXXS:a,lineWidthBold:l,lineWidth:s,paddingXXS:c}=e,d=e.calc(o).add(e.calc(l).mul(4).equal()).equal(),u=e.calc(r).add(e.calc(e.lineWidth).mul(4).equal()).equal();return{[`&${i}-with-progress`]:{[`${i}-item`]:{paddingTop:c,[`&-process ${i}-item-container ${i}-item-icon ${i}-icon`]:{color:n}},[`&${i}-vertical > ${i}-item `]:{paddingInlineStart:c,[`> ${i}-item-container > ${i}-item-tail`]:{top:a,insetInlineStart:e.calc(o).div(2).sub(s).add(c).equal()}},[`&, &${i}-small`]:{[`&${i}-horizontal ${i}-item:first-child`]:{paddingBottom:c,paddingInlineStart:c}},[`&${i}-small${i}-vertical > ${i}-item > ${i}-item-container > ${i}-item-tail`]:{insetInlineStart:e.calc(r).div(2).sub(s).add(c).equal()},[`&${i}-label-vertical ${i}-item ${i}-item-tail`]:{top:e.calc(o).div(2).add(c).equal()},[`${i}-item-icon`]:{position:"relative",[`${t}-progress`]:{position:"absolute",insetInlineStart:"50%",top:"50%",transform:"translate(-50%, -50%)","&-inner":{width:`${(0,_.unit)(d)} !important`,height:`${(0,_.unit)(d)} !important`}}},[`&${i}-small`]:{[`&${i}-label-vertical ${i}-item ${i}-item-tail`]:{top:e.calc(r).div(2).add(c).equal()},[`${i}-item-icon ${t}-progress-inner`]:{width:`${(0,_.unit)(u)} !important`,height:`${(0,_.unit)(u)} !important`}}}}})(e)),(e=>{let{componentCls:t,inlineDotSize:i,inlineTitleColor:o,inlineTailColor:r}=e,n=e.calc(e.paddingXS).add(e.lineWidth).equal(),a={[`${t}-item-container ${t}-item-content ${t}-item-title`]:{color:o}};return{[`&${t}-inline`]:{width:"auto",display:"inline-flex",[`${t}-item`]:{flex:"none","&-container":{padding:`${(0,_.unit)(n)} ${(0,_.unit)(e.paddingXXS)} 0`,margin:`0 ${(0,_.unit)(e.calc(e.marginXXS).div(2).equal())}`,borderRadius:e.borderRadiusSM,cursor:"pointer",transition:`background-color ${e.motionDurationMid}`,"&:hover":{background:e.controlItemBgHover},"&[role='button']:hover":{opacity:1}},"&-icon":{width:i,height:i,marginInlineStart:`calc(50% - ${(0,_.unit)(e.calc(i).div(2).equal())})`,[`> ${t}-icon`]:{top:0},[`${t}-icon-dot`]:{borderRadius:e.calc(e.fontSizeSM).div(4).equal(),"&::after":{display:"none"}}},"&-content":{width:"auto",marginTop:e.calc(e.marginXS).sub(e.lineWidth).equal()},"&-title":{color:o,fontSize:e.fontSizeSM,lineHeight:e.lineHeightSM,fontWeight:"normal",marginBottom:e.calc(e.marginXXS).div(2).equal()},"&-description":{display:"none"},"&-tail":{marginInlineStart:0,top:e.calc(i).div(2).add(n).equal(),transform:"translateY(-50%)","&:after":{width:"100%",height:e.lineWidth,borderRadius:0,marginInlineStart:0,background:r}},[`&:first-child ${t}-item-tail`]:{width:"50%",marginInlineStart:"50%"},[`&:last-child ${t}-item-tail`]:{display:"block",width:"50%"},"&-wait":Object.assign({[`${t}-item-icon ${t}-icon ${t}-icon-dot`]:{backgroundColor:e.colorBorderBg,border:`${(0,_.unit)(e.lineWidth)} ${e.lineType} ${r}`}},a),"&-finish":Object.assign({[`${t}-item-tail::after`]:{backgroundColor:r},[`${t}-item-icon ${t}-icon ${t}-icon-dot`]:{backgroundColor:r,border:`${(0,_.unit)(e.lineWidth)} ${e.lineType} ${r}`}},a),"&-error":a,"&-active, &-process":Object.assign({[`${t}-item-icon`]:{width:i,height:i,marginInlineStart:`calc(50% - ${(0,_.unit)(e.calc(i).div(2).equal())})`,top:0}},a),[`&:not(${t}-item-active) > ${t}-item-container[role='button']:hover`]:{[`${t}-item-title`]:{color:o}}}}}})(e))}})((0,x.mergeToken)(e,{processIconColor:o,processTitleColor:r,processDescriptionColor:r,processIconBgColor:n,processIconBorderColor:n,processDotColor:n,processTailColor:d,waitTitleColor:a,waitDescriptionColor:a,waitTailColor:d,waitDotColor:t,finishIconColor:n,finishTitleColor:r,finishDescriptionColor:a,finishTailColor:n,finishDotColor:n,errorIconColor:o,errorTitleColor:s,errorDescriptionColor:s,errorTailColor:d,errorIconBgColor:s,errorIconBorderColor:s,errorDotColor:s,stepsNavActiveColor:n,stepsProgressSize:i,inlineDotSize:6,inlineTitleColor:l,inlineTailColor:c}))},e=>({titleLineHeight:e.controlHeight,customIconSize:e.controlHeight,customIconTop:0,customIconFontSize:e.controlHeightSM,iconSize:e.controlHeight,iconTop:-.5,iconFontSize:e.fontSize,iconSizeSM:e.fontSizeHeading3,dotSize:e.controlHeight/4,dotCurrentSize:e.controlHeightLG/4,navArrowColor:e.colorTextDisabled,navContentMaxWidth:"unset",descriptionMaxWidth:140,waitIconColor:e.wireframe?e.colorTextDisabled:e.colorTextLabel,waitIconBgColor:e.wireframe?e.colorBgContainer:e.colorFillContent,waitIconBorderColor:e.wireframe?e.colorTextDisabled:"transparent",finishIconBgColor:e.wireframe?e.colorBgContainer:e.controlItemBgActive,finishIconBorderColor:e.wireframe?e.colorPrimary:e.controlItemBgActive}));var S=e.i(876556),k=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,o=Object.getOwnPropertySymbols(e);r<o.length;r++)0>t.indexOf(o[r])&&Object.prototype.propertyIsEnumerable.call(e,o[r])&&(i[o[r]]=e[o[r]]);return i};let A=e=>{var n,a;let{percent:l,size:s,className:c,rootClassName:d,direction:u,items:m,responsive:g=!0,current:_=0,children:y,style:w}=e,x=k(e,["percent","size","className","rootClassName","direction","items","responsive","current","children","style"]),{xs:C}=(0,b.default)(g),{getPrefixCls:A,direction:E,className:O,style:j}=(0,f.useComponentConfig)("steps"),T=t.useMemo(()=>g&&C?"vertical":u,[g,C,u]),N=(0,h.default)(s),M=A("steps",e.prefixCls),[z,P,L]=I(M),R="inline"===e.type,D=A("",e.iconPrefix),H=(n=m,a=y,n?n:(0,S.default)(a).map(e=>{if(t.isValidElement(e)){let{props:t}=e;return Object.assign({},t)}return null}).filter(e=>e)),q=R?void 0:l,B=Object.assign(Object.assign({},j),w),W=(0,r.default)(O,{[`${M}-rtl`]:"rtl"===E,[`${M}-with-progress`]:void 0!==q},c,d,P,L),G={finish:t.createElement(i.default,{className:`${M}-finish-icon`}),error:t.createElement(o.default,{className:`${M}-error-icon`})};return z(t.createElement(p,Object.assign({icons:G},x,{style:B,current:_,size:N,items:H,itemRender:R?(e,i)=>e.description?t.createElement($.default,{title:e.description},i):i:void 0,stepIcon:({node:e,status:i})=>"process"===i&&void 0!==q?t.createElement("div",{className:`${M}-progress-icon`},t.createElement(v.default,{type:"circle",percent:q,size:"small"===N?32:40,strokeWidth:4,format:()=>null}),e):e,direction:T,prefixCls:M,iconPrefix:D,className:W})))};A.Step=p.Step,e.s(["Steps",0,A],280898)},771674,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M858.5 763.6a374 374 0 00-80.6-119.5 375.63 375.63 0 00-119.5-80.6c-.4-.2-.8-.3-1.2-.5C719.5 518 760 444.7 760 362c0-137-111-248-248-248S264 225 264 362c0 82.7 40.5 156 102.8 201.1-.4.2-.8.3-1.2.5-44.8 18.9-85 46-119.5 80.6a375.63 375.63 0 00-80.6 119.5A371.7 371.7 0 00136 901.8a8 8 0 008 8.2h60c4.4 0 7.9-3.5 8-7.8 2-77.2 33-149.5 87.8-204.3 56.7-56.7 132-87.9 212.2-87.9s155.5 31.2 212.2 87.9C779 752.7 810 825 812 902.2c.1 4.4 3.6 7.8 8 7.8h60a8 8 0 008-8.2c-1-47.8-10.9-94.3-29.5-138.2zM512 534c-45.9 0-89.1-17.9-121.6-50.4S340 407.9 340 362c0-45.9 17.9-89.1 50.4-121.6S466.1 190 512 190s89.1 17.9 121.6 50.4S684 316.1 684 362c0 45.9-17.9 89.1-50.4 121.6S557.9 534 512 534z"}}]},name:"user",theme:"outlined"};var r=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(r.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["UserOutlined",0,n],771674)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var r=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(r.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["SafetyOutlined",0,n],602073)},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return r}});let o=e.r(271645);function r(e,t){let i=(0,o.useRef)(null),r=(0,o.useRef)(null);return(0,o.useCallback)(o=>{if(null===o){let e=i.current;e&&(i.current=null,e());let t=r.current;t&&(r.current=null,t())}else e&&(i.current=n(e,o)),t&&(r.current=n(t,o))},[e,t])}function n(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},62478,e=>{"use strict";var t=e.i(764205);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},190272,785913,e=>{"use strict";var t,i,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),r=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i);let n={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>r,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=n[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:o,apiKey:n,inputMessage:a,chatHistory:l,selectedTags:s,selectedVectorStores:c,selectedGuardrails:d,selectedPolicies:u,selectedMCPServers:m,mcpServers:g,mcpServerToolRestrictions:p,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:v,proxySettings:$}=e,_="session"===i?o:n,y=window.location.origin,w=$?.LITELLM_UI_API_DOC_BASE_URL;w&&w.trim()?y=w:$?.PROXY_BASE_URL&&(y=$.PROXY_BASE_URL);let x=a||"Your prompt here",C=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),I=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),S={};s.length>0&&(S.tags=s),c.length>0&&(S.vector_stores=c),d.length>0&&(S.guardrails=d),u.length>0&&(S.policies=u);let k=b||"your-model-name",A="azure"===v?`import openai

client = openai.AzureOpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${y}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${_||"YOUR_LITELLM_API_KEY"}",
	base_url="${y}"
)`;switch(h){case r.CHAT:{let e=Object.keys(S).length>0,i="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=I.length>0?I:[{role:"user",content:x}];t=`
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
#                     "text": "${C}"
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
`;break}case r.RESPONSES:{let e=Object.keys(S).length>0,i="";if(e){let e=JSON.stringify({metadata:S},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=I.length>0?I:[{role:"user",content:x}];t=`
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
#                 {"type": "input_text", "text": "${C}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case r.IMAGE:t="azure"===v?`
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
prompt = "${C}"

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
`;break;case r.IMAGE_EDITS:t="azure"===v?`
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
prompt = "${C}"

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
prompt = "${C}"

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
	input="${a||"Your string here"}",
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
	file=audio_file${a?`,
	prompt="${a.replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case r.SPEECH:t=`
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
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${A}
${t}`}],190272)},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},326373,e=>{"use strict";var t=e.i(21539);e.s(["Dropdown",()=>t.default])},115571,371401,e=>{"use strict";let t="local-storage-change";function i(e){window.dispatchEvent(new CustomEvent(t,{detail:{key:e}}))}function o(e){try{return window.localStorage.getItem(e)}catch(t){return console.warn(`Error reading localStorage key "${e}":`,t),null}}function r(e,t){try{window.localStorage.setItem(e,t)}catch(t){console.warn(`Error setting localStorage key "${e}":`,t)}}function n(e){try{window.localStorage.removeItem(e)}catch(t){console.warn(`Error removing localStorage key "${e}":`,t)}}e.s(["LOCAL_STORAGE_EVENT",0,t,"emitLocalStorageChange",()=>i,"getLocalStorageItem",()=>o,"removeLocalStorageItem",()=>n,"setLocalStorageItem",()=>r],115571);var a=e.i(271645);function l(e){let i=t=>{"disableUsageIndicator"===t.key&&e()},o=t=>{let{key:i}=t.detail;"disableUsageIndicator"===i&&e()};return window.addEventListener("storage",i),window.addEventListener(t,o),()=>{window.removeEventListener("storage",i),window.removeEventListener(t,o)}}function s(){return"true"===o("disableUsageIndicator")}function c(){return(0,a.useSyncExternalStore)(l,s)}e.s(["useDisableUsageIndicator",()=>c],371401)},275144,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(764205);let r=(0,i.createContext)(void 0);e.s(["ThemeProvider",0,({children:e,accessToken:n})=>{let[a,l]=(0,i.useState)(null);return(0,i.useEffect)(()=>{(async()=>{try{let e=(0,o.getProxyBaseUrl)(),t=e?`${e}/get/ui_theme_settings`:"/get/ui_theme_settings",i=await fetch(t,{method:"GET",headers:{"Content-Type":"application/json"}});if(i.ok){let e=await i.json();e.values?.logo_url&&l(e.values.logo_url)}}catch(e){console.warn("Failed to load logo settings from backend:",e)}})()},[]),(0,t.jsx)(r.Provider,{value:{logoUrl:a,setLogoUrl:l},children:e})},"useTheme",0,()=>{let e=(0,i.useContext)(r);if(!e)throw Error("useTheme must be used within a ThemeProvider");return e}])},798496,e=>{"use strict";var t=e.i(843476),i=e.i(152990),o=e.i(682830),r=e.i(271645),n=e.i(269200),a=e.i(427612),l=e.i(64848),s=e.i(942232),c=e.i(496020),d=e.i(977572),u=e.i(94629),m=e.i(360820),g=e.i(871943);function p({data:e=[],columns:p,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:v,enablePagination:$=!1}){let[_,y]=r.default.useState(h),[w]=r.default.useState("onChange"),[x,C]=r.default.useState({}),[I,S]=r.default.useState({}),k=(0,i.useReactTable)({data:e,columns:p,state:{sorting:_,columnSizing:x,columnVisibility:I,...$&&b?{pagination:b}:{}},columnResizeMode:w,onSortingChange:y,onColumnSizingChange:C,onColumnVisibilityChange:S,...$&&v?{onPaginationChange:v}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...$?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(n.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:k.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(a.TableHead,{children:k.getHeaderGroups().map(e=>(0,t.jsx)(c.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,i.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(m.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(g.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:p.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):k.getRowModel().rows.length>0?k.getRowModel().rows.map(e=>(0,t.jsx)(c.TableRow,{children:e.getVisibleCells().map(e=>(0,t.jsx)(d.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,i.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(c.TableRow,{children:(0,t.jsx)(d.TableCell,{colSpan:p.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>p])},916925,e=>{"use strict";var t,i=((t={}).A2A_Agent="A2A Agent",t.AIML="AI/ML API",t.Bedrock="Amazon Bedrock",t.Anthropic="Anthropic",t.AssemblyAI="AssemblyAI",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.Cerebras="Cerebras",t.Cohere="Cohere",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.ElevenLabs="ElevenLabs",t.FalAI="Fal AI",t.FireworksAI="Fireworks AI",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.Hosted_Vllm="vllm",t.Infinity="Infinity",t.JinaAI="Jina AI",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.Ollama="Ollama",t.OpenAI="OpenAI",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.Perplexity="Perplexity",t.RunwayML="RunwayML",t.Sambanova="Sambanova",t.Snowflake="Snowflake",t.TogetherAI="TogetherAI",t.Triton="Triton",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.xAI="xAI",t.SAP="SAP Generative AI Hub",t.Watsonx="Watsonx",t);let o={A2A_Agent:"a2a_agent",AIML:"aiml",OpenAI:"openai",OpenAI_Text:"text-completion-openai",Azure:"azure",Azure_AI_Studio:"azure_ai",Anthropic:"anthropic",Google_AI_Studio:"gemini",Bedrock:"bedrock",Groq:"groq",MiniMax:"minimax",MistralAI:"mistral",Cohere:"cohere",OpenAI_Compatible:"openai",OpenAI_Text_Compatible:"text-completion-openai",Vertex_AI:"vertex_ai",Databricks:"databricks",Dashscope:"dashscope",xAI:"xai",Deepseek:"deepseek",Ollama:"ollama",AssemblyAI:"assemblyai",Cerebras:"cerebras",Sambanova:"sambanova",Perplexity:"perplexity",RunwayML:"runwayml",TogetherAI:"together_ai",Openrouter:"openrouter",Oracle:"oci",Snowflake:"snowflake",FireworksAI:"fireworks_ai",GradientAI:"gradient_ai",Triton:"triton",Deepgram:"deepgram",ElevenLabs:"elevenlabs",FalAI:"fal_ai",SageMaker:"sagemaker_chat",Voyage:"voyage",JinaAI:"jina_ai",VolcEngine:"volcengine",DeepInfra:"deepinfra",Hosted_Vllm:"hosted_vllm",Infinity:"infinity",SAP:"sap",Watsonx:"watsonx"},r="../ui/assets/logos/",n={"A2A Agent":`${r}a2a_agent.png`,"AI/ML API":`${r}aiml_api.svg`,Anthropic:`${r}anthropic.svg`,AssemblyAI:`${r}assemblyai_small.png`,Azure:`${r}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${r}microsoft_azure.svg`,"Amazon Bedrock":`${r}bedrock.svg`,"AWS SageMaker":`${r}bedrock.svg`,Cerebras:`${r}cerebras.svg`,Cohere:`${r}cohere.svg`,"Databricks (Qwen API)":`${r}databricks.svg`,Dashscope:`${r}dashscope.svg`,Deepseek:`${r}deepseek.svg`,"Fireworks AI":`${r}fireworks.svg`,Groq:`${r}groq.svg`,"Google AI Studio":`${r}google.svg`,vllm:`${r}vllm.png`,Infinity:`${r}infinity.png`,MiniMax:`${r}minimax.svg`,"Mistral AI":`${r}mistral.svg`,Ollama:`${r}ollama.svg`,OpenAI:`${r}openai_small.svg`,"OpenAI Text Completion":`${r}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${r}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${r}openai_small.svg`,Openrouter:`${r}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${r}oracle.svg`,Perplexity:`${r}perplexity-ai.svg`,RunwayML:`${r}runwayml.png`,Sambanova:`${r}sambanova.svg`,Snowflake:`${r}snowflake.svg`,TogetherAI:`${r}togetherai.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${r}google.svg`,xAI:`${r}xai.svg`,GradientAI:`${r}gradientai.svg`,Triton:`${r}nvidia_triton.png`,Deepgram:`${r}deepgram.png`,ElevenLabs:`${r}elevenlabs.png`,"Fal AI":`${r}fal_ai.jpg`,"Voyage AI":`${r}voyage.webp`,"Jina AI":`${r}jina.png`,VolcEngine:`${r}volcengine.png`,DeepInfra:`${r}deepinfra.png`,"SAP Generative AI Hub":`${r}sap.png`};e.s(["Providers",()=>i,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:n[e],displayName:e}}let t=Object.keys(o).find(t=>o[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let r=i[t];return{logo:n[r],displayName:r}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let i=o[e];console.log(`Provider mapped to: ${i}`);let r=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let o=t.litellm_provider;(o===i||"string"==typeof o&&o.includes(i))&&r.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&r.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&r.push(e)}))),r},"providerLogoMap",0,n,"provider_map",0,o])},94629,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,i],94629)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},262218,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),o=e.i(529681),r=e.i(702779),n=e.i(563113),a=e.i(763731),l=e.i(121872),s=e.i(242064);e.i(296059);var c=e.i(915654);e.i(262370);var d=e.i(135551),u=e.i(183293),m=e.i(246422),g=e.i(838378);let p=e=>{let{lineWidth:t,fontSizeIcon:i,calc:o}=e,r=e.fontSizeSM;return(0,g.mergeToken)(e,{tagFontSize:r,tagLineHeight:(0,c.unit)(o(e.lineHeightSM).mul(r).equal()),tagIconSize:o(i).sub(o(t).mul(2)).equal(),tagPaddingHorizontal:8,tagBorderlessBg:e.defaultBg})},f=e=>({defaultBg:new d.FastColor(e.colorFillQuaternary).onBackground(e.colorBgContainer).toHexString(),defaultColor:e.colorText}),h=(0,m.genStyleHooks)("Tag",e=>(e=>{let{paddingXXS:t,lineWidth:i,tagPaddingHorizontal:o,componentCls:r,calc:n}=e,a=n(o).sub(i).equal(),l=n(t).sub(i).equal();return{[r]:Object.assign(Object.assign({},(0,u.resetComponent)(e)),{display:"inline-block",height:"auto",marginInlineEnd:e.marginXS,paddingInline:a,fontSize:e.tagFontSize,lineHeight:e.tagLineHeight,whiteSpace:"nowrap",background:e.defaultBg,border:`${(0,c.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,opacity:1,transition:`all ${e.motionDurationMid}`,textAlign:"start",position:"relative",[`&${r}-rtl`]:{direction:"rtl"},"&, a, a:hover":{color:e.defaultColor},[`${r}-close-icon`]:{marginInlineStart:l,fontSize:e.tagIconSize,color:e.colorIcon,cursor:"pointer",transition:`all ${e.motionDurationMid}`,"&:hover":{color:e.colorTextHeading}},[`&${r}-has-color`]:{borderColor:"transparent",[`&, a, a:hover, ${e.iconCls}-close, ${e.iconCls}-close:hover`]:{color:e.colorTextLightSolid}},"&-checkable":{backgroundColor:"transparent",borderColor:"transparent",cursor:"pointer",[`&:not(${r}-checkable-checked):hover`]:{color:e.colorPrimary,backgroundColor:e.colorFillSecondary},"&:active, &-checked":{color:e.colorTextLightSolid},"&-checked":{backgroundColor:e.colorPrimary,"&:hover":{backgroundColor:e.colorPrimaryHover}},"&:active":{backgroundColor:e.colorPrimaryActive}},"&-hidden":{display:"none"},[`> ${e.iconCls} + span, > span + ${e.iconCls}`]:{marginInlineStart:a}}),[`${r}-borderless`]:{borderColor:"transparent",background:e.tagBorderlessBg}}})(p(e)),f);var b=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,o=Object.getOwnPropertySymbols(e);r<o.length;r++)0>t.indexOf(o[r])&&Object.prototype.propertyIsEnumerable.call(e,o[r])&&(i[o[r]]=e[o[r]]);return i};let v=t.forwardRef((e,o)=>{let{prefixCls:r,style:n,className:a,checked:l,children:c,icon:d,onChange:u,onClick:m}=e,g=b(e,["prefixCls","style","className","checked","children","icon","onChange","onClick"]),{getPrefixCls:p,tag:f}=t.useContext(s.ConfigContext),v=p("tag",r),[$,_,y]=h(v),w=(0,i.default)(v,`${v}-checkable`,{[`${v}-checkable-checked`]:l},null==f?void 0:f.className,a,_,y);return $(t.createElement("span",Object.assign({},g,{ref:o,style:Object.assign(Object.assign({},n),null==f?void 0:f.style),className:w,onClick:e=>{null==u||u(!l),null==m||m(e)}}),d,t.createElement("span",null,c)))});var $=e.i(403541);let _=(0,m.genSubStyleComponent)(["Tag","preset"],e=>{let t;return t=p(e),(0,$.genPresetColor)(t,(e,{textColor:i,lightBorderColor:o,lightColor:r,darkColor:n})=>({[`${t.componentCls}${t.componentCls}-${e}`]:{color:i,background:r,borderColor:o,"&-inverse":{color:t.colorTextLightSolid,background:n,borderColor:n},[`&${t.componentCls}-borderless`]:{borderColor:"transparent"}}}))},f),y=(e,t,i)=>{let o="string"!=typeof i?i:i.charAt(0).toUpperCase()+i.slice(1);return{[`${e.componentCls}${e.componentCls}-${t}`]:{color:e[`color${i}`],background:e[`color${o}Bg`],borderColor:e[`color${o}Border`],[`&${e.componentCls}-borderless`]:{borderColor:"transparent"}}}},w=(0,m.genSubStyleComponent)(["Tag","status"],e=>{let t=p(e);return[y(t,"success","Success"),y(t,"processing","Info"),y(t,"error","Error"),y(t,"warning","Warning")]},f);var x=function(e,t){var i={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(i[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var r=0,o=Object.getOwnPropertySymbols(e);r<o.length;r++)0>t.indexOf(o[r])&&Object.prototype.propertyIsEnumerable.call(e,o[r])&&(i[o[r]]=e[o[r]]);return i};let C=t.forwardRef((e,c)=>{let{prefixCls:d,className:u,rootClassName:m,style:g,children:p,icon:f,color:b,onClose:v,bordered:$=!0,visible:y}=e,C=x(e,["prefixCls","className","rootClassName","style","children","icon","color","onClose","bordered","visible"]),{getPrefixCls:I,direction:S,tag:k}=t.useContext(s.ConfigContext),[A,E]=t.useState(!0),O=(0,o.default)(C,["closeIcon","closable"]);t.useEffect(()=>{void 0!==y&&E(y)},[y]);let j=(0,r.isPresetColor)(b),T=(0,r.isPresetStatusColor)(b),N=j||T,M=Object.assign(Object.assign({backgroundColor:b&&!N?b:void 0},null==k?void 0:k.style),g),z=I("tag",d),[P,L,R]=h(z),D=(0,i.default)(z,null==k?void 0:k.className,{[`${z}-${b}`]:N,[`${z}-has-color`]:b&&!N,[`${z}-hidden`]:!A,[`${z}-rtl`]:"rtl"===S,[`${z}-borderless`]:!$},u,m,L,R),H=e=>{e.stopPropagation(),null==v||v(e),e.defaultPrevented||E(!1)},[,q]=(0,n.useClosable)((0,n.pickClosable)(e),(0,n.pickClosable)(k),{closable:!1,closeIconRender:e=>{let o=t.createElement("span",{className:`${z}-close-icon`,onClick:H},e);return(0,a.replaceElement)(e,o,e=>({onClick:t=>{var i;null==(i=null==e?void 0:e.onClick)||i.call(e,t),H(t)},className:(0,i.default)(null==e?void 0:e.className,`${z}-close-icon`)}))}}),B="function"==typeof C.onClick||p&&"a"===p.type,W=f||null,G=W?t.createElement(t.Fragment,null,W,p&&t.createElement("span",null,p)):p,X=t.createElement("span",Object.assign({},O,{ref:c,className:D,style:M}),G,q,j&&t.createElement(_,{key:"preset",prefixCls:z}),T&&t.createElement(w,{key:"status",prefixCls:z}));return P(B?t.createElement(l.default,{component:"Tag"},X):X)});C.CheckableTag=v,e.s(["Tag",0,C],262218)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},250980,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,i],250980)},502547,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,i],502547)},434626,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,i],434626)},902555,e=>{"use strict";var t=e.i(843476),i=e.i(591935),o=e.i(122577),r=e.i(278587),n=e.i(68155),a=e.i(360820),l=e.i(871943),s=e.i(434626),c=e.i(592968),d=e.i(115504),u=e.i(752978);function m({icon:e,onClick:i,className:o,disabled:r,dataTestId:n}){return r?(0,t.jsx)(u.Icon,{icon:e,size:"sm",className:"opacity-50 cursor-not-allowed","data-testid":n}):(0,t.jsx)(u.Icon,{icon:e,size:"sm",onClick:i,className:(0,d.cx)("cursor-pointer",o),"data-testid":n})}let g={Edit:{icon:i.PencilAltIcon,className:"hover:text-blue-600"},Delete:{icon:n.TrashIcon,className:"hover:text-red-600"},Test:{icon:o.PlayIcon,className:"hover:text-blue-600"},Regenerate:{icon:r.RefreshIcon,className:"hover:text-green-600"},Up:{icon:a.ChevronUpIcon,className:"hover:text-blue-600"},Down:{icon:l.ChevronDownIcon,className:"hover:text-blue-600"},Open:{icon:s.ExternalLinkIcon,className:"hover:text-green-600"}};function p({onClick:e,tooltipText:i,disabled:o=!1,disabledTooltipText:r,dataTestId:n,variant:a}){let{icon:l,className:s}=g[a];return(0,t.jsx)(c.Tooltip,{title:o?r:i,children:(0,t.jsx)("span",{children:(0,t.jsx)(m,{icon:l,onClick:e,className:s,disabled:o,dataTestId:n})})})}e.s(["default",()=>p],902555)},122577,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlayIcon",0,i],122577)},278587,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["RefreshIcon",0,i],278587)},207670,e=>{"use strict";function t(){for(var e,t,i=0,o="",r=arguments.length;i<r;i++)(e=arguments[i])&&(t=function e(t){var i,o,r="";if("string"==typeof t||"number"==typeof t)r+=t;else if("object"==typeof t)if(Array.isArray(t)){var n=t.length;for(i=0;i<n;i++)t[i]&&(o=e(t[i]))&&(r&&(r+=" "),r+=o)}else for(o in t)t[o]&&(r&&(r+=" "),r+=o);return r}(e))&&(o&&(o+=" "),o+=t);return o}e.s(["clsx",()=>t,"default",0,t])},728889,e=>{"use strict";var t=e.i(290571),i=e.i(271645),o=e.i(829087),r=e.i(480731),n=e.i(444755),a=e.i(673706),l=e.i(95779);let s={xs:{paddingX:"px-1.5",paddingY:"py-1.5"},sm:{paddingX:"px-1.5",paddingY:"py-1.5"},md:{paddingX:"px-2",paddingY:"py-2"},lg:{paddingX:"px-2",paddingY:"py-2"},xl:{paddingX:"px-2.5",paddingY:"py-2.5"}},c={xs:{height:"h-3",width:"w-3"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-7",width:"w-7"},xl:{height:"h-9",width:"w-9"}},d={simple:{rounded:"",border:"",ring:"",shadow:""},light:{rounded:"rounded-tremor-default",border:"",ring:"",shadow:""},shadow:{rounded:"rounded-tremor-default",border:"border",ring:"",shadow:"shadow-tremor-card dark:shadow-dark-tremor-card"},solid:{rounded:"rounded-tremor-default",border:"border-2",ring:"ring-1",shadow:""},outlined:{rounded:"rounded-tremor-default",border:"border",ring:"ring-2",shadow:""}},u=(0,a.makeClassName)("Icon"),m=i.default.forwardRef((e,m)=>{let{icon:g,variant:p="simple",tooltip:f,size:h=r.Sizes.SM,color:b,className:v}=e,$=(0,t.__rest)(e,["icon","variant","tooltip","size","color","className"]),_=((e,t)=>{switch(e){case"simple":return{textColor:t?(0,a.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:"",borderColor:"",ringColor:""};case"light":return{textColor:t?(0,a.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,n.tremorTwMerge)((0,a.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand-muted dark:bg-dark-tremor-brand-muted",borderColor:"",ringColor:""};case"shadow":return{textColor:t?(0,a.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,n.tremorTwMerge)((0,a.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:"border-tremor-border dark:border-dark-tremor-border",ringColor:""};case"solid":return{textColor:t?(0,a.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,n.tremorTwMerge)((0,a.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand dark:bg-dark-tremor-brand",borderColor:"border-tremor-brand-inverted dark:border-dark-tremor-brand-inverted",ringColor:"ring-tremor-ring dark:ring-dark-tremor-ring"};case"outlined":return{textColor:t?(0,a.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,n.tremorTwMerge)((0,a.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:t?(0,a.getColorClassNames)(t,l.colorPalette.ring).borderColor:"border-tremor-brand-subtle dark:border-dark-tremor-brand-subtle",ringColor:t?(0,n.tremorTwMerge)((0,a.getColorClassNames)(t,l.colorPalette.ring).ringColor,"ring-opacity-40"):"ring-tremor-brand-muted dark:ring-dark-tremor-brand-muted"}}})(p,b),{tooltipProps:y,getReferenceProps:w}=(0,o.useTooltip)();return i.default.createElement("span",Object.assign({ref:(0,a.mergeRefs)([m,y.refs.setReference]),className:(0,n.tremorTwMerge)(u("root"),"inline-flex shrink-0 items-center justify-center",_.bgColor,_.textColor,_.borderColor,_.ringColor,d[p].rounded,d[p].border,d[p].shadow,d[p].ring,s[h].paddingX,s[h].paddingY,v)},w,$),i.default.createElement(o.default,Object.assign({text:f},y)),i.default.createElement(g,{className:(0,n.tremorTwMerge)(u("icon"),"shrink-0",c[h].height,c[h].width)}))});m.displayName="Icon",e.s(["default",()=>m],728889)},752978,e=>{"use strict";var t=e.i(728889);e.s(["Icon",()=>t.default])},591935,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))});e.s(["PencilAltIcon",0,i],591935)},100486,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M899.6 276.5L705 396.4 518.4 147.5a8.06 8.06 0 00-12.9 0L319 396.4 124.3 276.5c-5.7-3.5-13.1 1.2-12.2 7.9L188.5 865c1.1 7.9 7.9 14 16 14h615.1c8 0 14.9-6 15.9-14l76.4-580.6c.8-6.7-6.5-11.4-12.3-7.9zm-126 534.1H250.3l-53.8-409.4 139.8 86.1L512 252.9l175.7 234.4 139.8-86.1-53.9 409.4zM512 509c-62.1 0-112.6 50.5-112.6 112.6S449.9 734.2 512 734.2s112.6-50.5 112.6-112.6S574.1 509 512 509zm0 160.9c-26.6 0-48.2-21.6-48.2-48.3 0-26.6 21.6-48.3 48.2-48.3s48.2 21.6 48.2 48.3c0 26.6-21.6 48.3-48.2 48.3z"}}]},name:"crown",theme:"outlined"};var r=e.i(9583),n=i.forwardRef(function(e,n){return i.createElement(r.default,(0,t.default)({},e,{ref:n,icon:o}))});e.s(["CrownOutlined",0,n],100486)},292639,e=>{"use strict";var t=e.i(764205),i=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("uiSettings");e.s(["useUISettings",0,()=>(0,i.useQuery)({queryKey:o.list({}),queryFn:async()=>await (0,t.getUiSettings)(),staleTime:36e5,gcTime:36e5})])}]);