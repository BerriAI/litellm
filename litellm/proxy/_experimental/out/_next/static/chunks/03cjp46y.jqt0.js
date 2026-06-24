(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,185793,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),a=e.i(242064),n=e.i(529681);let r=e=>{let{prefixCls:a,className:n,style:r,size:o,shape:s}=e,l=(0,i.default)({[`${a}-lg`]:"large"===o,[`${a}-sm`]:"small"===o}),p=(0,i.default)({[`${a}-circle`]:"circle"===s,[`${a}-square`]:"square"===s,[`${a}-round`]:"round"===s}),d=t.useMemo(()=>"number"==typeof o?{width:o,height:o,lineHeight:`${o}px`}:{},[o]);return t.createElement("span",{className:(0,i.default)(a,l,p,n),style:Object.assign(Object.assign({},d),r)})};e.i(296059);var o=e.i(694758),s=e.i(915654),l=e.i(246422),p=e.i(838378);let d=new o.Keyframes("ant-skeleton-loading",{"0%":{backgroundPosition:"100% 50%"},"100%":{backgroundPosition:"0 50%"}}),c=e=>({height:e,lineHeight:(0,s.unit)(e)}),u=e=>Object.assign({width:e},c(e)),g=(e,t)=>Object.assign({width:t(e).mul(5).equal(),minWidth:t(e).mul(5).equal()},c(e)),m=e=>Object.assign({width:e},c(e)),f=(e,t,i)=>{let{skeletonButtonCls:a}=e;return{[`${i}${a}-circle`]:{width:t,minWidth:t,borderRadius:"50%"},[`${i}${a}-round`]:{borderRadius:t}}},_=(e,t)=>Object.assign({width:t(e).mul(2).equal(),minWidth:t(e).mul(2).equal()},c(e)),b=(0,l.genStyleHooks)("Skeleton",e=>{let{componentCls:t,calc:i}=e;return(e=>{let{componentCls:t,skeletonAvatarCls:i,skeletonTitleCls:a,skeletonParagraphCls:n,skeletonButtonCls:r,skeletonInputCls:o,skeletonImageCls:s,controlHeight:l,controlHeightLG:p,controlHeightSM:c,gradientFromColor:b,padding:h,marginSM:y,borderRadius:$,titleHeight:v,blockRadius:O,paragraphLiHeight:x,controlHeightXS:j,paragraphMarginTop:E}=e;return{[t]:{display:"table",width:"100%",[`${t}-header`]:{display:"table-cell",paddingInlineEnd:h,verticalAlign:"top",[i]:Object.assign({display:"inline-block",verticalAlign:"top",background:b},u(l)),[`${i}-circle`]:{borderRadius:"50%"},[`${i}-lg`]:Object.assign({},u(p)),[`${i}-sm`]:Object.assign({},u(c))},[`${t}-content`]:{display:"table-cell",width:"100%",verticalAlign:"top",[a]:{width:"100%",height:v,background:b,borderRadius:O,[`+ ${n}`]:{marginBlockStart:c}},[n]:{padding:0,"> li":{width:"100%",height:x,listStyle:"none",background:b,borderRadius:O,"+ li":{marginBlockStart:j}}},[`${n}> li:last-child:not(:first-child):not(:nth-child(2))`]:{width:"61%"}},[`&-round ${t}-content`]:{[`${a}, ${n} > li`]:{borderRadius:$}}},[`${t}-with-avatar ${t}-content`]:{[a]:{marginBlockStart:y,[`+ ${n}`]:{marginBlockStart:E}}},[`${t}${t}-element`]:Object.assign(Object.assign(Object.assign(Object.assign({display:"inline-block",width:"auto"},(e=>{let{borderRadiusSM:t,skeletonButtonCls:i,controlHeight:a,controlHeightLG:n,controlHeightSM:r,gradientFromColor:o,calc:s}=e;return Object.assign(Object.assign(Object.assign(Object.assign(Object.assign({[i]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:t,width:s(a).mul(2).equal(),minWidth:s(a).mul(2).equal()},_(a,s))},f(e,a,i)),{[`${i}-lg`]:Object.assign({},_(n,s))}),f(e,n,`${i}-lg`)),{[`${i}-sm`]:Object.assign({},_(r,s))}),f(e,r,`${i}-sm`))})(e)),(e=>{let{skeletonAvatarCls:t,gradientFromColor:i,controlHeight:a,controlHeightLG:n,controlHeightSM:r}=e;return{[t]:Object.assign({display:"inline-block",verticalAlign:"top",background:i},u(a)),[`${t}${t}-circle`]:{borderRadius:"50%"},[`${t}${t}-lg`]:Object.assign({},u(n)),[`${t}${t}-sm`]:Object.assign({},u(r))}})(e)),(e=>{let{controlHeight:t,borderRadiusSM:i,skeletonInputCls:a,controlHeightLG:n,controlHeightSM:r,gradientFromColor:o,calc:s}=e;return{[a]:Object.assign({display:"inline-block",verticalAlign:"top",background:o,borderRadius:i},g(t,s)),[`${a}-lg`]:Object.assign({},g(n,s)),[`${a}-sm`]:Object.assign({},g(r,s))}})(e)),(e=>{let{skeletonImageCls:t,imageSizeBase:i,gradientFromColor:a,borderRadiusSM:n,calc:r}=e;return{[t]:Object.assign(Object.assign({display:"inline-flex",alignItems:"center",justifyContent:"center",verticalAlign:"middle",background:a,borderRadius:n},m(r(i).mul(2).equal())),{[`${t}-path`]:{fill:"#bfbfbf"},[`${t}-svg`]:Object.assign(Object.assign({},m(i)),{maxWidth:r(i).mul(4).equal(),maxHeight:r(i).mul(4).equal()}),[`${t}-svg${t}-svg-circle`]:{borderRadius:"50%"}}),[`${t}${t}-circle`]:{borderRadius:"50%"}}})(e)),[`${t}${t}-block`]:{width:"100%",[r]:{width:"100%"},[o]:{width:"100%"}},[`${t}${t}-active`]:{[`
        ${a},
        ${n} > li,
        ${i},
        ${r},
        ${o},
        ${s}
      `]:Object.assign({},{background:e.skeletonLoadingBackground,backgroundSize:"400% 100%",animationName:d,animationDuration:e.skeletonLoadingMotionDuration,animationTimingFunction:"ease",animationIterationCount:"infinite"})}}})((0,p.mergeToken)(e,{skeletonAvatarCls:`${t}-avatar`,skeletonTitleCls:`${t}-title`,skeletonParagraphCls:`${t}-paragraph`,skeletonButtonCls:`${t}-button`,skeletonInputCls:`${t}-input`,skeletonImageCls:`${t}-image`,imageSizeBase:i(e.controlHeight).mul(1.5).equal(),borderRadius:100,skeletonLoadingBackground:`linear-gradient(90deg, ${e.gradientFromColor} 25%, ${e.gradientToColor} 37%, ${e.gradientFromColor} 63%)`,skeletonLoadingMotionDuration:"1.4s"}))},e=>{let{colorFillContent:t,colorFill:i}=e;return{color:t,colorGradientEnd:i,gradientFromColor:t,gradientToColor:i,titleHeight:e.controlHeight/2,blockRadius:e.borderRadiusSM,paragraphMarginTop:e.marginLG+e.marginXXS,paragraphLiHeight:e.controlHeight/2}},{deprecatedTokens:[["color","gradientFromColor"],["colorGradientEnd","gradientToColor"]]}),h=e=>{let{prefixCls:a,className:n,style:r,rows:o=0}=e,s=Array.from({length:o}).map((i,a)=>t.createElement("li",{key:a,style:{width:((e,t)=>{let{width:i,rows:a=2}=t;return Array.isArray(i)?i[e]:a-1===e?i:void 0})(a,e)}}));return t.createElement("ul",{className:(0,i.default)(a,n),style:r},s)},y=({prefixCls:e,className:a,width:n,style:r})=>t.createElement("h3",{className:(0,i.default)(e,a),style:Object.assign({width:n},r)});function $(e){return e&&"object"==typeof e?e:{}}let v=e=>{let{prefixCls:n,loading:o,className:s,rootClassName:l,style:p,children:d,avatar:c=!1,title:u=!0,paragraph:g=!0,active:m,round:f}=e,{getPrefixCls:_,direction:v,className:O,style:x}=(0,a.useComponentConfig)("skeleton"),j=_("skeleton",n),[E,w,C]=b(j);if(o||!("loading"in e)){let e,a,n=!!c,o=!!u,d=!!g;if(n){let i=Object.assign(Object.assign({prefixCls:`${j}-avatar`},o&&!d?{size:"large",shape:"square"}:{size:"large",shape:"circle"}),$(c));e=t.createElement("div",{className:`${j}-header`},t.createElement(r,Object.assign({},i)))}if(o||d){let e,i;if(o){let i=Object.assign(Object.assign({prefixCls:`${j}-title`},!n&&d?{width:"38%"}:n&&d?{width:"50%"}:{}),$(u));e=t.createElement(y,Object.assign({},i))}if(d){let e,a=Object.assign(Object.assign({prefixCls:`${j}-paragraph`},(e={},n&&o||(e.width="61%"),!n&&o?e.rows=3:e.rows=2,e)),$(g));i=t.createElement(h,Object.assign({},a))}a=t.createElement("div",{className:`${j}-content`},e,i)}let _=(0,i.default)(j,{[`${j}-with-avatar`]:n,[`${j}-active`]:m,[`${j}-rtl`]:"rtl"===v,[`${j}-round`]:f},O,s,l,w,C);return E(t.createElement("div",{className:_,style:Object.assign(Object.assign({},x),p)},e,a))}return null!=d?d:null};v.Button=e=>{let{prefixCls:o,className:s,rootClassName:l,active:p,block:d=!1,size:c="default"}=e,{getPrefixCls:u}=t.useContext(a.ConfigContext),g=u("skeleton",o),[m,f,_]=b(g),h=(0,n.default)(e,["prefixCls"]),y=(0,i.default)(g,`${g}-element`,{[`${g}-active`]:p,[`${g}-block`]:d},s,l,f,_);return m(t.createElement("div",{className:y},t.createElement(r,Object.assign({prefixCls:`${g}-button`,size:c},h))))},v.Avatar=e=>{let{prefixCls:o,className:s,rootClassName:l,active:p,shape:d="circle",size:c="default"}=e,{getPrefixCls:u}=t.useContext(a.ConfigContext),g=u("skeleton",o),[m,f,_]=b(g),h=(0,n.default)(e,["prefixCls","className"]),y=(0,i.default)(g,`${g}-element`,{[`${g}-active`]:p},s,l,f,_);return m(t.createElement("div",{className:y},t.createElement(r,Object.assign({prefixCls:`${g}-avatar`,shape:d,size:c},h))))},v.Input=e=>{let{prefixCls:o,className:s,rootClassName:l,active:p,block:d,size:c="default"}=e,{getPrefixCls:u}=t.useContext(a.ConfigContext),g=u("skeleton",o),[m,f,_]=b(g),h=(0,n.default)(e,["prefixCls"]),y=(0,i.default)(g,`${g}-element`,{[`${g}-active`]:p,[`${g}-block`]:d},s,l,f,_);return m(t.createElement("div",{className:y},t.createElement(r,Object.assign({prefixCls:`${g}-input`,size:c},h))))},v.Image=e=>{let{prefixCls:n,className:r,rootClassName:o,style:s,active:l}=e,{getPrefixCls:p}=t.useContext(a.ConfigContext),d=p("skeleton",n),[c,u,g]=b(d),m=(0,i.default)(d,`${d}-element`,{[`${d}-active`]:l},r,o,u,g);return c(t.createElement("div",{className:m},t.createElement("div",{className:(0,i.default)(`${d}-image`,r),style:s},t.createElement("svg",{viewBox:"0 0 1098 1024",xmlns:"http://www.w3.org/2000/svg",className:`${d}-image-svg`},t.createElement("title",null,"Image placeholder"),t.createElement("path",{d:"M365.714286 329.142857q0 45.714286-32.036571 77.677714t-77.677714 32.036571-77.677714-32.036571-32.036571-77.677714 32.036571-77.677714 77.677714-32.036571 77.677714 32.036571 32.036571 77.677714zM950.857143 548.571429l0 256-804.571429 0 0-109.714286 182.857143-182.857143 91.428571 91.428571 292.571429-292.571429zM1005.714286 146.285714l-914.285714 0q-7.460571 0-12.873143 5.412571t-5.412571 12.873143l0 694.857143q0 7.460571 5.412571 12.873143t12.873143 5.412571l914.285714 0q7.460571 0 12.873143-5.412571t5.412571-12.873143l0-694.857143q0-7.460571-5.412571-12.873143t-12.873143-5.412571zM1097.142857 164.571429l0 694.857143q0 37.741714-26.843429 64.585143t-64.585143 26.843429l-914.285714 0q-37.741714 0-64.585143-26.843429t-26.843429-64.585143l0-694.857143q0-37.741714 26.843429-64.585143t64.585143-26.843429l914.285714 0q37.741714 0 64.585143 26.843429t26.843429 64.585143z",className:`${d}-image-path`})))))},v.Node=e=>{let{prefixCls:n,className:r,rootClassName:o,style:s,active:l,children:p}=e,{getPrefixCls:d}=t.useContext(a.ConfigContext),c=d("skeleton",n),[u,g,m]=b(c),f=(0,i.default)(c,`${c}-element`,{[`${c}-active`]:l},g,r,o,m);return u(t.createElement("div",{className:f},t.createElement("div",{className:(0,i.default)(`${c}-image`,r),style:s},p)))},e.s(["default",0,v],185793)},959013,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M482 152h60q8 0 8 8v704q0 8-8 8h-60q-8 0-8-8V160q0-8 8-8z"}},{tag:"path",attrs:{d:"M192 474h672q8 0 8 8v60q0 8-8 8H160q-8 0-8-8v-60q0-8 8-8z"}}]},name:"plus",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["default",0,r],959013)},282786,836938,310730,829672,e=>{"use strict";e.i(247167);var t=e.i(271645),i=e.i(343794),a=e.i(914949),n=e.i(404948);let r=e=>e?"function"==typeof e?e():e:null;e.s(["getRenderPropValue",0,r],836938);var o=e.i(613541),s=e.i(763731),l=e.i(242064),p=e.i(491816);e.i(793154);var d=e.i(880476),c=e.i(183293),u=e.i(717356),g=e.i(320560),m=e.i(307358),f=e.i(246422),_=e.i(838378),b=e.i(617933);let h=(0,f.genStyleHooks)("Popover",e=>{let{colorBgElevated:t,colorText:i}=e,a=(0,_.mergeToken)(e,{popoverBg:t,popoverColor:i});return[(e=>{let{componentCls:t,popoverColor:i,titleMinWidth:a,fontWeightStrong:n,innerPadding:r,boxShadowSecondary:o,colorTextHeading:s,borderRadiusLG:l,zIndexPopup:p,titleMarginBottom:d,colorBgElevated:u,popoverBg:m,titleBorderBottom:f,innerContentPadding:_,titlePadding:b}=e;return[{[t]:Object.assign(Object.assign({},(0,c.resetComponent)(e)),{position:"absolute",top:0,left:{_skip_check_:!0,value:0},zIndex:p,fontWeight:"normal",whiteSpace:"normal",textAlign:"start",cursor:"auto",userSelect:"text","--valid-offset-x":"var(--arrow-offset-horizontal, var(--arrow-x))",transformOrigin:"var(--valid-offset-x, 50%) var(--arrow-y, 50%)","--antd-arrow-background-color":u,width:"max-content",maxWidth:"100vw","&-rtl":{direction:"rtl"},"&-hidden":{display:"none"},[`${t}-content`]:{position:"relative"},[`${t}-inner`]:{backgroundColor:m,backgroundClip:"padding-box",borderRadius:l,boxShadow:o,padding:r},[`${t}-title`]:{minWidth:a,marginBottom:d,color:s,fontWeight:n,borderBottom:f,padding:b},[`${t}-inner-content`]:{color:i,padding:_}})},(0,g.default)(e,"var(--antd-arrow-background-color)"),{[`${t}-pure`]:{position:"relative",maxWidth:"none",margin:e.sizePopupArrow,display:"inline-block",[`${t}-content`]:{display:"inline-block"}}}]})(a),(e=>{let{componentCls:t}=e;return{[t]:b.PresetColors.map(i=>{let a=e[`${i}6`];return{[`&${t}-${i}`]:{"--antd-arrow-background-color":a,[`${t}-inner`]:{backgroundColor:a},[`${t}-arrow`]:{background:"transparent"}}}})}})(a),(0,u.initZoomMotion)(a,"zoom-big")]},e=>{let{lineWidth:t,controlHeight:i,fontHeight:a,padding:n,wireframe:r,zIndexPopupBase:o,borderRadiusLG:s,marginXS:l,lineType:p,colorSplit:d,paddingSM:c}=e,u=i-a;return Object.assign(Object.assign(Object.assign({titleMinWidth:177,zIndexPopup:o+30},(0,m.getArrowToken)(e)),(0,g.getArrowOffsetToken)({contentRadius:s,limitVerticalRadius:!0})),{innerPadding:12*!r,titleMarginBottom:r?0:l,titlePadding:r?`${u/2}px ${n}px ${u/2-t}px`:0,titleBorderBottom:r?`${t}px ${p} ${d}`:"none",innerContentPadding:r?`${c}px ${n}px`:0})},{resetStyle:!1,deprecatedTokens:[["width","titleMinWidth"],["minWidth","titleMinWidth"]]});var y=function(e,t){var i={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(i[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,a=Object.getOwnPropertySymbols(e);n<a.length;n++)0>t.indexOf(a[n])&&Object.prototype.propertyIsEnumerable.call(e,a[n])&&(i[a[n]]=e[a[n]]);return i};let $=({title:e,content:i,prefixCls:a})=>e||i?t.createElement(t.Fragment,null,e&&t.createElement("div",{className:`${a}-title`},e),i&&t.createElement("div",{className:`${a}-inner-content`},i)):null,v=e=>{let{hashId:a,prefixCls:n,className:o,style:s,placement:l="top",title:p,content:c,children:u}=e,g=r(p),m=r(c),f=(0,i.default)(a,n,`${n}-pure`,`${n}-placement-${l}`,o);return t.createElement("div",{className:f,style:s},t.createElement("div",{className:`${n}-arrow`}),t.createElement(d.Popup,Object.assign({},e,{className:a,prefixCls:n}),u||t.createElement($,{prefixCls:n,title:g,content:m})))},O=e=>{let{prefixCls:a,className:n}=e,r=y(e,["prefixCls","className"]),{getPrefixCls:o}=t.useContext(l.ConfigContext),s=o("popover",a),[p,d,c]=h(s);return p(t.createElement(v,Object.assign({},r,{prefixCls:s,hashId:d,className:(0,i.default)(n,c)})))};e.s(["Overlay",0,$,"default",0,O],310730);var x=function(e,t){var i={};for(var a in e)Object.prototype.hasOwnProperty.call(e,a)&&0>t.indexOf(a)&&(i[a]=e[a]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var n=0,a=Object.getOwnPropertySymbols(e);n<a.length;n++)0>t.indexOf(a[n])&&Object.prototype.propertyIsEnumerable.call(e,a[n])&&(i[a[n]]=e[a[n]]);return i};let j=t.forwardRef((e,d)=>{var c,u;let{prefixCls:g,title:m,content:f,overlayClassName:_,placement:b="top",trigger:y="hover",children:v,mouseEnterDelay:O=.1,mouseLeaveDelay:j=.1,onOpenChange:E,overlayStyle:w={},styles:C,classNames:k}=e,I=x(e,["prefixCls","title","content","overlayClassName","placement","trigger","children","mouseEnterDelay","mouseLeaveDelay","onOpenChange","overlayStyle","styles","classNames"]),{getPrefixCls:N,className:S,style:A,classNames:R,styles:P}=(0,l.useComponentConfig)("popover"),T=N("popover",g),[M,L,q]=h(T),z=N(),D=(0,i.default)(_,L,q,S,R.root,null==k?void 0:k.root),B=(0,i.default)(R.body,null==k?void 0:k.body),[H,G]=(0,a.default)(!1,{value:null!=(c=e.open)?c:e.visible,defaultValue:null!=(u=e.defaultOpen)?u:e.defaultVisible}),F=(e,t)=>{G(e,!0),null==E||E(e,t)},W=r(m),U=r(f);return M(t.createElement(p.default,Object.assign({placement:b,trigger:y,mouseEnterDelay:O,mouseLeaveDelay:j},I,{prefixCls:T,classNames:{root:D,body:B},styles:{root:Object.assign(Object.assign(Object.assign(Object.assign({},P.root),A),w),null==C?void 0:C.root),body:Object.assign(Object.assign({},P.body),null==C?void 0:C.body)},ref:d,open:H,onOpenChange:e=>{F(e)},overlay:W||U?t.createElement($,{prefixCls:T,title:W,content:U}):null,transitionName:(0,o.getTransitionName)(z,"zoom-big",I.transitionName),"data-popover-inject":!0}),(0,s.cloneElement)(v,{onKeyDown:e=>{var i,a;(0,t.isValidElement)(v)&&(null==(a=null==v?void 0:(i=v.props).onKeyDown)||a.call(i,e)),e.keyCode===n.default.ESC&&F(!1,e)}})))});j._InternalPanelDoNotUseOrYouWillBeFired=O,e.s(["default",0,j],829672),e.s(["Popover",0,j],282786)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},447566,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M872 474H286.9l350.2-304c5.6-4.9 2.2-14-5.2-14h-88.5c-3.9 0-7.6 1.4-10.5 3.9L155 487.8a31.96 31.96 0 000 48.3L535.1 866c1.5 1.3 3.3 2 5.2 2h91.5c7.4 0 10.8-9.2 5.2-14L286.9 550H872c4.4 0 8-3.6 8-8v-60c0-4.4-3.6-8-8-8z"}}]},name:"arrow-left",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["ArrowLeftOutlined",0,r],447566)},492030,e=>{"use strict";var t=e.i(121229);e.s(["CheckOutlined",()=>t.default])},755151,e=>{"use strict";var t=e.i(247153);e.s(["DownOutlined",()=>t.default])},596239,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"path",attrs:{d:"M574 665.4a8.03 8.03 0 00-11.3 0L446.5 781.6c-53.8 53.8-144.6 59.5-204 0-59.5-59.5-53.8-150.2 0-204l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3l-39.8-39.8a8.03 8.03 0 00-11.3 0L191.4 526.5c-84.6 84.6-84.6 221.5 0 306s221.5 84.6 306 0l116.2-116.2c3.1-3.1 3.1-8.2 0-11.3L574 665.4zm258.6-474c-84.6-84.6-221.5-84.6-306 0L410.3 307.6a8.03 8.03 0 000 11.3l39.7 39.7c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c53.8-53.8 144.6-59.5 204 0 59.5 59.5 53.8 150.2 0 204L665.3 562.6a8.03 8.03 0 000 11.3l39.8 39.8c3.1 3.1 8.2 3.1 11.3 0l116.2-116.2c84.5-84.6 84.5-221.5 0-306.1zM610.1 372.3a8.03 8.03 0 00-11.3 0L372.3 598.7a8.03 8.03 0 000 11.3l39.6 39.6c3.1 3.1 8.2 3.1 11.3 0l226.4-226.4c3.1-3.1 3.1-8.2 0-11.3l-39.5-39.6z"}}]},name:"link",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["LinkOutlined",0,r],596239)},62478,e=>{"use strict";var t=e.i(764205);let i=async e=>{if(!e)return null;try{return await (0,t.getProxyUISettings)(e)}catch(e){return console.error("Error fetching proxy settings:",e),null}};e.s(["fetchProxySettings",0,i])},818581,(e,t,i)=>{"use strict";Object.defineProperty(i,"__esModule",{value:!0}),Object.defineProperty(i,"useMergedRef",{enumerable:!0,get:function(){return n}});let a=e.r(271645);function n(e,t){let i=(0,a.useRef)(null),n=(0,a.useRef)(null);return(0,a.useCallback)(a=>{if(null===a){let e=i.current;e&&(i.current=null,e());let t=n.current;t&&(n.current=null,t())}else e&&(i.current=r(e,a)),t&&(n.current=r(t,a))},[e,t])}function r(e,t){if("function"!=typeof e)return e.current=t,()=>{e.current=null};{let i=e(t);return"function"==typeof i?i:()=>e(null)}}("function"==typeof i.default||"object"==typeof i.default&&null!==i.default)&&void 0===i.default.__esModule&&(Object.defineProperty(i.default,"__esModule",{value:!0}),Object.assign(i.default,i),t.exports=i.default)},602073,e=>{"use strict";e.i(247167);var t=e.i(931067),i=e.i(271645);let a={icon:{tag:"svg",attrs:{viewBox:"0 0 1024 1024",focusable:"false"},children:[{tag:"path",attrs:{d:"M512 64L128 192v384c0 212.1 171.9 384 384 384s384-171.9 384-384V192L512 64zm312 512c0 172.3-139.7 312-312 312S200 748.3 200 576V246l312-110 312 110v330z"}},{tag:"path",attrs:{d:"M378.4 475.1a35.91 35.91 0 00-50.9 0 35.91 35.91 0 000 50.9l129.4 129.4 2.1 2.1a33.98 33.98 0 0048.1 0L730.6 434a33.98 33.98 0 000-48.1l-2.8-2.8a33.98 33.98 0 00-48.1 0L483 579.7 378.4 475.1z"}}]},name:"safety",theme:"outlined"};var n=e.i(9583),r=i.forwardRef(function(e,r){return i.createElement(n.default,(0,t.default)({},e,{ref:r,icon:a}))});e.s(["SafetyOutlined",0,r],602073)},190272,785913,e=>{"use strict";var t,i,a=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),n=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i.INTERACTIONS="interactions",i);let r={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>n,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(a).includes(e)){let t=r[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:a,apiKey:r,inputMessage:o,chatHistory:s,selectedTags:l,selectedVectorStores:p,selectedGuardrails:d,selectedPolicies:c,selectedMCPServers:u,mcpServers:g,mcpServerToolRestrictions:m,selectedVoice:f,endpointType:_,selectedModel:b,selectedSdk:h,proxySettings:y}=e,$="session"===i?a:r,v=window.location.origin,O=y?.LITELLM_UI_API_DOC_BASE_URL;O&&O.trim()?v=O:y?.PROXY_BASE_URL&&(v=y.PROXY_BASE_URL);let x=o||"Your prompt here",j=x.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),E=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),w={};l.length>0&&(w.tags=l),p.length>0&&(w.vector_stores=p),d.length>0&&(w.guardrails=d),c.length>0&&(w.policies=c);let C=b||"your-model-name",k="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${$||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${v}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${$||"YOUR_LITELLM_API_KEY"}",
	base_url="${v}"
)`;switch(_){case n.CHAT:{let e=Object.keys(w).length>0,i="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=E.length>0?E:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(a,null,4)}${i}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${C}",
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
`;break}case n.RESPONSES:{let e=Object.keys(w).length>0,i="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let a=E.length>0?E:[{role:"user",content:x}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(a,null,4)}${i}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${C}",
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
`;break}case n.IMAGE:t="azure"===h?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${C}",
	prompt="${o}",
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
	model="${C}",
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
`;break;case n.IMAGE_EDITS:t="azure"===h?`
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
	model="${C}",
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
	model="${C}",
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
	input="${o||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case n.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${o?`,
	prompt="${o.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case n.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${o||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${o||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${k}
${t}`}],190272)}]);