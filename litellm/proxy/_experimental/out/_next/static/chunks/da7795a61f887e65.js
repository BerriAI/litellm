(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,349356,e=>{e.v({AElig:"Æ",AMP:"&",Aacute:"Á",Acirc:"Â",Agrave:"À",Aring:"Å",Atilde:"Ã",Auml:"Ä",COPY:"©",Ccedil:"Ç",ETH:"Ð",Eacute:"É",Ecirc:"Ê",Egrave:"È",Euml:"Ë",GT:">",Iacute:"Í",Icirc:"Î",Igrave:"Ì",Iuml:"Ï",LT:"<",Ntilde:"Ñ",Oacute:"Ó",Ocirc:"Ô",Ograve:"Ò",Oslash:"Ø",Otilde:"Õ",Ouml:"Ö",QUOT:'"',REG:"®",THORN:"Þ",Uacute:"Ú",Ucirc:"Û",Ugrave:"Ù",Uuml:"Ü",Yacute:"Ý",aacute:"á",acirc:"â",acute:"´",aelig:"æ",agrave:"à",amp:"&",aring:"å",atilde:"ã",auml:"ä",brvbar:"¦",ccedil:"ç",cedil:"¸",cent:"¢",copy:"©",curren:"¤",deg:"°",divide:"÷",eacute:"é",ecirc:"ê",egrave:"è",eth:"ð",euml:"ë",frac12:"½",frac14:"¼",frac34:"¾",gt:">",iacute:"í",icirc:"î",iexcl:"¡",igrave:"ì",iquest:"¿",iuml:"ï",laquo:"«",lt:"<",macr:"¯",micro:"µ",middot:"·",nbsp:" ",not:"¬",ntilde:"ñ",oacute:"ó",ocirc:"ô",ograve:"ò",ordf:"ª",ordm:"º",oslash:"ø",otilde:"õ",ouml:"ö",para:"¶",plusmn:"±",pound:"£",quot:'"',raquo:"»",reg:"®",sect:"§",shy:"­",sup1:"¹",sup2:"²",sup3:"³",szlig:"ß",thorn:"þ",times:"×",uacute:"ú",ucirc:"û",ugrave:"ù",uml:"¨",uuml:"ü",yacute:"ý",yen:"¥",yuml:"ÿ"})},137429,e=>{e.v({0:"�",128:"€",130:"‚",131:"ƒ",132:"„",133:"…",134:"†",135:"‡",136:"ˆ",137:"‰",138:"Š",139:"‹",140:"Œ",142:"Ž",145:"‘",146:"’",147:"“",148:"”",149:"•",150:"–",151:"—",152:"˜",153:"™",154:"š",155:"›",156:"œ",158:"ž",159:"Ÿ"})},91874,e=>{"use strict";var t=e.i(931067),r=e.i(209428),o=e.i(211577),a=e.i(392221),i=e.i(703923),n=e.i(343794),l=e.i(914949),s=e.i(271645),d=["prefixCls","className","style","checked","disabled","defaultChecked","type","title","onChange"],c=(0,s.forwardRef)(function(e,c){var u=e.prefixCls,m=void 0===u?"rc-checkbox":u,p=e.className,g=e.style,f=e.checked,h=e.disabled,b=e.defaultChecked,_=e.type,v=void 0===_?"checkbox":_,C=e.title,x=e.onChange,A=(0,i.default)(e,d),y=(0,s.useRef)(null),I=(0,s.useRef)(null),k=(0,l.default)(void 0!==b&&b,{value:f}),w=(0,a.default)(k,2),E=w[0],T=w[1];(0,s.useImperativeHandle)(c,function(){return{focus:function(e){var t;null==(t=y.current)||t.focus(e)},blur:function(){var e;null==(e=y.current)||e.blur()},input:y.current,nativeElement:I.current}});var O=(0,n.default)(m,p,(0,o.default)((0,o.default)({},"".concat(m,"-checked"),E),"".concat(m,"-disabled"),h));return s.createElement("span",{className:O,title:C,style:g,ref:I},s.createElement("input",(0,t.default)({},A,{className:"".concat(m,"-input"),ref:y,onChange:function(t){h||("checked"in e||T(t.target.checked),null==x||x({target:(0,r.default)((0,r.default)({},e),{},{type:v,checked:t.target.checked}),stopPropagation:function(){t.stopPropagation()},preventDefault:function(){t.preventDefault()},nativeEvent:t.nativeEvent}))},disabled:h,checked:!!E,type:v})),s.createElement("span",{className:"".concat(m,"-inner")}))});e.s(["default",0,c])},421512,236836,e=>{"use strict";let t=e.i(271645).default.createContext(null);e.s(["default",0,t],421512),e.i(296059);var r=e.i(915654),o=e.i(183293),a=e.i(246422),i=e.i(838378);function n(e,t){return(e=>{let{checkboxCls:t}=e,a=`${t}-wrapper`;return[{[`${t}-group`]:Object.assign(Object.assign({},(0,o.resetComponent)(e)),{display:"inline-flex",flexWrap:"wrap",columnGap:e.marginXS,[`> ${e.antCls}-row`]:{flex:1}}),[a]:Object.assign(Object.assign({},(0,o.resetComponent)(e)),{display:"inline-flex",alignItems:"baseline",cursor:"pointer","&:after":{display:"inline-block",width:0,overflow:"hidden",content:"'\\a0'"},[`& + ${a}`]:{marginInlineStart:0},[`&${a}-in-form-item`]:{'input[type="checkbox"]':{width:14,height:14}}}),[t]:Object.assign(Object.assign({},(0,o.resetComponent)(e)),{position:"relative",whiteSpace:"nowrap",lineHeight:1,cursor:"pointer",borderRadius:e.borderRadiusSM,alignSelf:"center",[`${t}-input`]:{position:"absolute",inset:0,zIndex:1,cursor:"pointer",opacity:0,margin:0,[`&:focus-visible + ${t}-inner`]:(0,o.genFocusOutline)(e)},[`${t}-inner`]:{boxSizing:"border-box",display:"block",width:e.checkboxSize,height:e.checkboxSize,direction:"ltr",backgroundColor:e.colorBgContainer,border:`${(0,r.unit)(e.lineWidth)} ${e.lineType} ${e.colorBorder}`,borderRadius:e.borderRadiusSM,borderCollapse:"separate",transition:`all ${e.motionDurationSlow}`,"&:after":{boxSizing:"border-box",position:"absolute",top:"50%",insetInlineStart:"25%",display:"table",width:e.calc(e.checkboxSize).div(14).mul(5).equal(),height:e.calc(e.checkboxSize).div(14).mul(8).equal(),border:`${(0,r.unit)(e.lineWidthBold)} solid ${e.colorWhite}`,borderTop:0,borderInlineStart:0,transform:"rotate(45deg) scale(0) translate(-50%,-50%)",opacity:0,content:'""',transition:`all ${e.motionDurationFast} ${e.motionEaseInBack}, opacity ${e.motionDurationFast}`}},"& + span":{paddingInlineStart:e.paddingXS,paddingInlineEnd:e.paddingXS}})},{[`
        ${a}:not(${a}-disabled),
        ${t}:not(${t}-disabled)
      `]:{[`&:hover ${t}-inner`]:{borderColor:e.colorPrimary}},[`${a}:not(${a}-disabled)`]:{[`&:hover ${t}-checked:not(${t}-disabled) ${t}-inner`]:{backgroundColor:e.colorPrimaryHover,borderColor:"transparent"},[`&:hover ${t}-checked:not(${t}-disabled):after`]:{borderColor:e.colorPrimaryHover}}},{[`${t}-checked`]:{[`${t}-inner`]:{backgroundColor:e.colorPrimary,borderColor:e.colorPrimary,"&:after":{opacity:1,transform:"rotate(45deg) scale(1) translate(-50%,-50%)",transition:`all ${e.motionDurationMid} ${e.motionEaseOutBack} ${e.motionDurationFast}`}}},[`
        ${a}-checked:not(${a}-disabled),
        ${t}-checked:not(${t}-disabled)
      `]:{[`&:hover ${t}-inner`]:{backgroundColor:e.colorPrimaryHover,borderColor:"transparent"}}},{[t]:{"&-indeterminate":{"&":{[`${t}-inner`]:{backgroundColor:`${e.colorBgContainer}`,borderColor:`${e.colorBorder}`,"&:after":{top:"50%",insetInlineStart:"50%",width:e.calc(e.fontSizeLG).div(2).equal(),height:e.calc(e.fontSizeLG).div(2).equal(),backgroundColor:e.colorPrimary,border:0,transform:"translate(-50%, -50%) scale(1)",opacity:1,content:'""'}},[`&:hover ${t}-inner`]:{backgroundColor:`${e.colorBgContainer}`,borderColor:`${e.colorPrimary}`}}}}},{[`${a}-disabled`]:{cursor:"not-allowed"},[`${t}-disabled`]:{[`&, ${t}-input`]:{cursor:"not-allowed",pointerEvents:"none"},[`${t}-inner`]:{background:e.colorBgContainerDisabled,borderColor:e.colorBorder,"&:after":{borderColor:e.colorTextDisabled}},"&:after":{display:"none"},"& + span":{color:e.colorTextDisabled},[`&${t}-indeterminate ${t}-inner::after`]:{background:e.colorTextDisabled}}}]})((0,i.mergeToken)(t,{checkboxCls:`.${e}`,checkboxSize:t.controlInteractiveSize}))}let l=(0,a.genStyleHooks)("Checkbox",(e,{prefixCls:t})=>[n(t,e)]);e.s(["default",0,l,"getStyle",()=>n],236836)},681216,e=>{"use strict";var t=e.i(271645),r=e.i(963188);function o(e){let o=t.default.useRef(null),a=()=>{r.default.cancel(o.current),o.current=null};return[()=>{a(),o.current=(0,r.default)(()=>{o.current=null})},t=>{o.current&&(t.stopPropagation(),a()),null==e||e(t)}]}e.s(["default",()=>o])},374276,e=>{"use strict";e.i(247167);var t=e.i(271645),r=e.i(343794),o=e.i(91874),a=e.i(611935),i=e.i(121872),n=e.i(26905),l=e.i(242064),s=e.i(937328),d=e.i(321883),c=e.i(62139),u=e.i(421512),m=e.i(236836),p=e.i(681216),g=function(e,t){var r={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(r[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,o=Object.getOwnPropertySymbols(e);a<o.length;a++)0>t.indexOf(o[a])&&Object.prototype.propertyIsEnumerable.call(e,o[a])&&(r[o[a]]=e[o[a]]);return r};let f=t.forwardRef((e,f)=>{var h;let{prefixCls:b,className:_,rootClassName:v,children:C,indeterminate:x=!1,style:A,onMouseEnter:y,onMouseLeave:I,skipGroup:k=!1,disabled:w}=e,E=g(e,["prefixCls","className","rootClassName","children","indeterminate","style","onMouseEnter","onMouseLeave","skipGroup","disabled"]),{getPrefixCls:T,direction:O,checkbox:$}=t.useContext(l.ConfigContext),N=t.useContext(u.default),{isFormItemInput:S}=t.useContext(c.FormItemInputContext),M=t.useContext(s.default),R=null!=(h=(null==N?void 0:N.disabled)||w)?h:M,P=t.useRef(E.value),L=t.useRef(null),j=(0,a.composeRef)(f,L);t.useEffect(()=>{null==N||N.registerValue(E.value)},[]),t.useEffect(()=>{if(!k)return E.value!==P.current&&(null==N||N.cancelValue(P.current),null==N||N.registerValue(E.value),P.current=E.value),()=>null==N?void 0:N.cancelValue(E.value)},[E.value]),t.useEffect(()=>{var e;(null==(e=L.current)?void 0:e.input)&&(L.current.input.indeterminate=x)},[x]);let D=T("checkbox",b),z=(0,d.default)(D),[B,H,G]=(0,m.default)(D,z),V=Object.assign({},E);N&&!k&&(V.onChange=(...e)=>{E.onChange&&E.onChange.apply(E,e),N.toggleOption&&N.toggleOption({label:C,value:E.value})},V.name=N.name,V.checked=N.value.includes(E.value));let U=(0,r.default)(`${D}-wrapper`,{[`${D}-rtl`]:"rtl"===O,[`${D}-wrapper-checked`]:V.checked,[`${D}-wrapper-disabled`]:R,[`${D}-wrapper-in-form-item`]:S},null==$?void 0:$.className,_,v,G,z,H),F=(0,r.default)({[`${D}-indeterminate`]:x},n.TARGET_CLS,H),[X,Y]=(0,p.default)(V.onClick);return B(t.createElement(i.default,{component:"Checkbox",disabled:R},t.createElement("label",{className:U,style:Object.assign(Object.assign({},null==$?void 0:$.style),A),onMouseEnter:y,onMouseLeave:I,onClick:X},t.createElement(o.default,Object.assign({},V,{onClick:Y,prefixCls:D,className:F,disabled:R,ref:j})),null!=C&&t.createElement("span",{className:`${D}-label`},C))))});var h=e.i(8211),b=e.i(529681),_=function(e,t){var r={};for(var o in e)Object.prototype.hasOwnProperty.call(e,o)&&0>t.indexOf(o)&&(r[o]=e[o]);if(null!=e&&"function"==typeof Object.getOwnPropertySymbols)for(var a=0,o=Object.getOwnPropertySymbols(e);a<o.length;a++)0>t.indexOf(o[a])&&Object.prototype.propertyIsEnumerable.call(e,o[a])&&(r[o[a]]=e[o[a]]);return r};let v=t.forwardRef((e,o)=>{let{defaultValue:a,children:i,options:n=[],prefixCls:s,className:c,rootClassName:p,style:g,onChange:v}=e,C=_(e,["defaultValue","children","options","prefixCls","className","rootClassName","style","onChange"]),{getPrefixCls:x,direction:A}=t.useContext(l.ConfigContext),[y,I]=t.useState(C.value||a||[]),[k,w]=t.useState([]);t.useEffect(()=>{"value"in C&&I(C.value||[])},[C.value]);let E=t.useMemo(()=>n.map(e=>"string"==typeof e||"number"==typeof e?{label:e,value:e}:e),[n]),T=e=>{w(t=>t.filter(t=>t!==e))},O=e=>{w(t=>[].concat((0,h.default)(t),[e]))},$=e=>{let t=y.indexOf(e.value),r=(0,h.default)(y);-1===t?r.push(e.value):r.splice(t,1),"value"in C||I(r),null==v||v(r.filter(e=>k.includes(e)).sort((e,t)=>E.findIndex(t=>t.value===e)-E.findIndex(e=>e.value===t)))},N=x("checkbox",s),S=`${N}-group`,M=(0,d.default)(N),[R,P,L]=(0,m.default)(N,M),j=(0,b.default)(C,["value","disabled"]),D=n.length?E.map(e=>t.createElement(f,{prefixCls:N,key:e.value.toString(),disabled:"disabled"in e?e.disabled:C.disabled,value:e.value,checked:y.includes(e.value),onChange:e.onChange,className:(0,r.default)(`${S}-item`,e.className),style:e.style,title:e.title,id:e.id,required:e.required},e.label)):i,z=t.useMemo(()=>({toggleOption:$,value:y,disabled:C.disabled,name:C.name,registerValue:O,cancelValue:T}),[$,y,C.disabled,C.name,O,T]),B=(0,r.default)(S,{[`${S}-rtl`]:"rtl"===A},c,p,L,M,P);return R(t.createElement("div",Object.assign({className:B,style:g},j,{ref:o}),t.createElement(u.default.Provider,{value:z},D)))});f.Group=v,f.__ANT_CHECKBOX=!0,e.s(["default",0,f],374276)},536916,e=>{"use strict";var t=e.i(374276);e.s(["Checkbox",()=>t.default])},599724,936325,e=>{"use strict";var t=e.i(95779),r=e.i(444755),o=e.i(673706),a=e.i(271645);let i=a.default.forwardRef((e,i)=>{let{color:n,className:l,children:s}=e;return a.default.createElement("p",{ref:i,className:(0,r.tremorTwMerge)("text-tremor-default",n?(0,o.getColorClassNames)(n,t.colorPalette.text).textColor:(0,r.tremorTwMerge)("text-tremor-content","dark:text-dark-tremor-content"),l)},s)});i.displayName="Text",e.s(["default",()=>i],936325),e.s(["Text",()=>i],599724)},994388,e=>{"use strict";var t=e.i(290571),r=e.i(829087),o=e.i(271645);let a=["preEnter","entering","entered","preExit","exiting","exited","unmounted"],i=e=>({_s:e,status:a[e],isEnter:e<3,isMounted:6!==e,isResolved:2===e||e>4}),n=e=>e?6:5,l=(e,t,r,o,a)=>{clearTimeout(o.current);let n=i(e);t(n),r.current=n,a&&a({current:n})};var s=e.i(480731),d=e.i(444755),c=e.i(673706);let u=e=>{var r=(0,t.__rest)(e,[]);return o.default.createElement("svg",Object.assign({},r,{xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 24 24",fill:"currentColor"}),o.default.createElement("path",{fill:"none",d:"M0 0h24v24H0z"}),o.default.createElement("path",{d:"M18.364 5.636L16.95 7.05A7 7 0 1 0 19 12h2a9 9 0 1 1-2.636-6.364z"}))};var m=e.i(95779);let p={xs:{height:"h-4",width:"w-4"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-6",width:"w-6"},xl:{height:"h-6",width:"w-6"}},g=(e,t)=>{switch(e){case"primary":return{textColor:t?(0,c.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",hoverTextColor:t?(0,c.getColorClassNames)("white").textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,c.getColorClassNames)(t,m.colorPalette.background).bgColor:"bg-tremor-brand dark:bg-dark-tremor-brand",hoverBgColor:t?(0,c.getColorClassNames)(t,m.colorPalette.darkBackground).hoverBgColor:"hover:bg-tremor-brand-emphasis dark:hover:bg-dark-tremor-brand-emphasis",borderColor:t?(0,c.getColorClassNames)(t,m.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",hoverBorderColor:t?(0,c.getColorClassNames)(t,m.colorPalette.darkBorder).hoverBorderColor:"hover:border-tremor-brand-emphasis dark:hover:border-dark-tremor-brand-emphasis"};case"secondary":return{textColor:t?(0,c.getColorClassNames)(t,m.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,c.getColorClassNames)(t,m.colorPalette.text).textColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,c.getColorClassNames)("transparent").bgColor,hoverBgColor:t?(0,d.tremorTwMerge)((0,c.getColorClassNames)(t,m.colorPalette.background).hoverBgColor,"hover:bg-opacity-20 dark:hover:bg-opacity-20"):"hover:bg-tremor-brand-faint dark:hover:bg-dark-tremor-brand-faint",borderColor:t?(0,c.getColorClassNames)(t,m.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand"};case"light":return{textColor:t?(0,c.getColorClassNames)(t,m.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",hoverTextColor:t?(0,c.getColorClassNames)(t,m.colorPalette.darkText).hoverTextColor:"hover:text-tremor-brand-emphasis dark:hover:text-dark-tremor-brand-emphasis",bgColor:(0,c.getColorClassNames)("transparent").bgColor,borderColor:"",hoverBorderColor:""}}},f=(0,c.makeClassName)("Button"),h=({loading:e,iconSize:t,iconPosition:r,Icon:a,needMargin:i,transitionStatus:n})=>{let l=i?r===s.HorizontalPositions.Left?(0,d.tremorTwMerge)("-ml-1","mr-1.5"):(0,d.tremorTwMerge)("-mr-1","ml-1.5"):"",c=(0,d.tremorTwMerge)("w-0 h-0"),m={default:c,entering:c,entered:t,exiting:t,exited:c};return e?o.default.createElement(u,{className:(0,d.tremorTwMerge)(f("icon"),"animate-spin shrink-0",l,m.default,m[n]),style:{transition:"width 150ms"}}):o.default.createElement(a,{className:(0,d.tremorTwMerge)(f("icon"),"shrink-0",t,l)})},b=o.default.forwardRef((e,a)=>{let{icon:u,iconPosition:m=s.HorizontalPositions.Left,size:b=s.Sizes.SM,color:_,variant:v="primary",disabled:C,loading:x=!1,loadingText:A,children:y,tooltip:I,className:k}=e,w=(0,t.__rest)(e,["icon","iconPosition","size","color","variant","disabled","loading","loadingText","children","tooltip","className"]),E=x||C,T=void 0!==u||x,O=x&&A,$=!(!y&&!O),N=(0,d.tremorTwMerge)(p[b].height,p[b].width),S="light"!==v?(0,d.tremorTwMerge)("rounded-tremor-default border","shadow-tremor-input","dark:shadow-dark-tremor-input"):"",M=g(v,_),R=("light"!==v?{xs:{paddingX:"px-2.5",paddingY:"py-1.5",fontSize:"text-xs"},sm:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-sm"},md:{paddingX:"px-4",paddingY:"py-2",fontSize:"text-md"},lg:{paddingX:"px-4",paddingY:"py-2.5",fontSize:"text-lg"},xl:{paddingX:"px-4",paddingY:"py-3",fontSize:"text-xl"}}:{xs:{paddingX:"",paddingY:"",fontSize:"text-xs"},sm:{paddingX:"",paddingY:"",fontSize:"text-sm"},md:{paddingX:"",paddingY:"",fontSize:"text-md"},lg:{paddingX:"",paddingY:"",fontSize:"text-lg"},xl:{paddingX:"",paddingY:"",fontSize:"text-xl"}})[b],{tooltipProps:P,getReferenceProps:L}=(0,r.useTooltip)(300),[j,D]=(({enter:e=!0,exit:t=!0,preEnter:r,preExit:a,timeout:s,initialEntered:d,mountOnEnter:c,unmountOnExit:u,onStateChange:m}={})=>{let[p,g]=(0,o.useState)(()=>i(d?2:n(c))),f=(0,o.useRef)(p),h=(0,o.useRef)(0),[b,_]="object"==typeof s?[s.enter,s.exit]:[s,s],v=(0,o.useCallback)(()=>{let e=((e,t)=>{switch(e){case 1:case 0:return 2;case 4:case 3:return n(t)}})(f.current._s,u);e&&l(e,g,f,h,m)},[m,u]);return[p,(0,o.useCallback)(o=>{let i=e=>{switch(l(e,g,f,h,m),e){case 1:b>=0&&(h.current=((...e)=>setTimeout(...e))(v,b));break;case 4:_>=0&&(h.current=((...e)=>setTimeout(...e))(v,_));break;case 0:case 3:h.current=((...e)=>setTimeout(...e))(()=>{isNaN(document.body.offsetTop)||i(e+1)},0)}},s=f.current.isEnter;"boolean"!=typeof o&&(o=!s),o?s||i(e?+!r:2):s&&i(t?a?3:4:n(u))},[v,m,e,t,r,a,b,_,u]),v]})({timeout:50});return(0,o.useEffect)(()=>{D(x)},[x]),o.default.createElement("button",Object.assign({ref:(0,c.mergeRefs)([a,P.refs.setReference]),className:(0,d.tremorTwMerge)(f("root"),"shrink-0 inline-flex justify-center items-center group font-medium outline-none",S,R.paddingX,R.paddingY,R.fontSize,M.textColor,M.bgColor,M.borderColor,M.hoverBorderColor,E?"opacity-50 cursor-not-allowed":(0,d.tremorTwMerge)(g(v,_).hoverTextColor,g(v,_).hoverBgColor,g(v,_).hoverBorderColor),k),disabled:E},L,w),o.default.createElement(r.default,Object.assign({text:I},P)),T&&m!==s.HorizontalPositions.Right?o.default.createElement(h,{loading:x,iconSize:N,iconPosition:m,Icon:u,transitionStatus:j.status,needMargin:$}):null,O||y?o.default.createElement("span",{className:(0,d.tremorTwMerge)(f("text"),"text-tremor-default whitespace-nowrap")},O?A:y):null,T&&m===s.HorizontalPositions.Right?o.default.createElement(h,{loading:x,iconSize:N,iconPosition:m,Icon:u,transitionStatus:j.status,needMargin:$}):null)});b.displayName="Button",e.s(["Button",()=>b],994388)},304967,e=>{"use strict";var t=e.i(290571),r=e.i(271645),o=e.i(480731),a=e.i(95779),i=e.i(444755),n=e.i(673706);let l=(0,n.makeClassName)("Card"),s=r.default.forwardRef((e,s)=>{let{decoration:d="",decorationColor:c,children:u,className:m}=e,p=(0,t.__rest)(e,["decoration","decorationColor","children","className"]);return r.default.createElement("div",Object.assign({ref:s,className:(0,i.tremorTwMerge)(l("root"),"relative w-full text-left ring-1 rounded-tremor-default p-6","bg-tremor-background ring-tremor-ring shadow-tremor-card","dark:bg-dark-tremor-background dark:ring-dark-tremor-ring dark:shadow-dark-tremor-card",c?(0,n.getColorClassNames)(c,a.colorPalette.border).borderColor:"border-tremor-brand dark:border-dark-tremor-brand",(e=>{if(!e)return"";switch(e){case o.HorizontalPositions.Left:return"border-l-4";case o.VerticalPositions.Top:return"border-t-4";case o.HorizontalPositions.Right:return"border-r-4";case o.VerticalPositions.Bottom:return"border-b-4";default:return""}})(d),m)},p),u)});s.displayName="Card",e.s(["Card",()=>s],304967)},629569,e=>{"use strict";var t=e.i(290571),r=e.i(95779),o=e.i(444755),a=e.i(673706),i=e.i(271645);let n=i.default.forwardRef((e,n)=>{let{color:l,children:s,className:d}=e,c=(0,t.__rest)(e,["color","children","className"]);return i.default.createElement("p",Object.assign({ref:n,className:(0,o.tremorTwMerge)("font-medium text-tremor-title",l?(0,a.getColorClassNames)(l,r.colorPalette.darkText).textColor:"text-tremor-content-strong dark:text-dark-tremor-content-strong",d)},c),s)});n.displayName="Title",e.s(["Title",()=>n],629569)},653496,e=>{"use strict";var t=e.i(721369);e.s(["Tabs",()=>t.default])},292639,e=>{"use strict";var t=e.i(764205),r=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("uiSettings");e.s(["useUISettings",0,()=>(0,r.useQuery)({queryKey:o.list({}),queryFn:async()=>await (0,t.getUiSettings)(),staleTime:36e5,gcTime:36e5})])},250980,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlusCircleIcon",0,r],250980)},502547,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["ChevronRightIcon",0,r],502547)},122577,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["PlayIcon",0,r],122577)},551332,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});e.s(["ClipboardCopyIcon",0,r],551332)},902555,e=>{"use strict";var t=e.i(843476),r=e.i(591935),o=e.i(122577),a=e.i(278587),i=e.i(68155),n=e.i(360820),l=e.i(871943),s=e.i(434626),d=e.i(551332),c=e.i(592968),u=e.i(115504),m=e.i(752978);function p({icon:e,onClick:r,className:o,disabled:a,dataTestId:i}){return a?(0,t.jsx)(m.Icon,{icon:e,size:"sm",className:"opacity-50 cursor-not-allowed","data-testid":i}):(0,t.jsx)(m.Icon,{icon:e,size:"sm",onClick:r,className:(0,u.cx)("cursor-pointer",o),"data-testid":i})}let g={Edit:{icon:r.PencilAltIcon,className:"hover:text-blue-600"},Delete:{icon:i.TrashIcon,className:"hover:text-red-600"},Test:{icon:o.PlayIcon,className:"hover:text-blue-600"},Regenerate:{icon:a.RefreshIcon,className:"hover:text-green-600"},Up:{icon:n.ChevronUpIcon,className:"hover:text-blue-600"},Down:{icon:l.ChevronDownIcon,className:"hover:text-blue-600"},Open:{icon:s.ExternalLinkIcon,className:"hover:text-green-600"},Copy:{icon:d.ClipboardCopyIcon,className:"hover:text-blue-600"}};function f({onClick:e,tooltipText:r,disabled:o=!1,disabledTooltipText:a,dataTestId:i,variant:n}){let{icon:l,className:s}=g[n];return(0,t.jsx)(c.Tooltip,{title:o?a:r,children:(0,t.jsx)("span",{children:(0,t.jsx)(p,{icon:l,onClick:e,className:s,disabled:o,dataTestId:i})})})}e.s(["default",()=>f],902555)},434626,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,r],434626)},207670,e=>{"use strict";function t(){for(var e,t,r=0,o="",a=arguments.length;r<a;r++)(e=arguments[r])&&(t=function e(t){var r,o,a="";if("string"==typeof t||"number"==typeof t)a+=t;else if("object"==typeof t)if(Array.isArray(t)){var i=t.length;for(r=0;r<i;r++)t[r]&&(o=e(t[r]))&&(a&&(a+=" "),a+=o)}else for(o in t)t[o]&&(a&&(a+=" "),a+=o);return a}(e))&&(o&&(o+=" "),o+=t);return o}e.s(["clsx",()=>t,"default",0,t])},728889,e=>{"use strict";var t=e.i(290571),r=e.i(271645),o=e.i(829087),a=e.i(480731),i=e.i(444755),n=e.i(673706),l=e.i(95779);let s={xs:{paddingX:"px-1.5",paddingY:"py-1.5"},sm:{paddingX:"px-1.5",paddingY:"py-1.5"},md:{paddingX:"px-2",paddingY:"py-2"},lg:{paddingX:"px-2",paddingY:"py-2"},xl:{paddingX:"px-2.5",paddingY:"py-2.5"}},d={xs:{height:"h-3",width:"w-3"},sm:{height:"h-5",width:"w-5"},md:{height:"h-5",width:"w-5"},lg:{height:"h-7",width:"w-7"},xl:{height:"h-9",width:"w-9"}},c={simple:{rounded:"",border:"",ring:"",shadow:""},light:{rounded:"rounded-tremor-default",border:"",ring:"",shadow:""},shadow:{rounded:"rounded-tremor-default",border:"border",ring:"",shadow:"shadow-tremor-card dark:shadow-dark-tremor-card"},solid:{rounded:"rounded-tremor-default",border:"border-2",ring:"ring-1",shadow:""},outlined:{rounded:"rounded-tremor-default",border:"border",ring:"ring-2",shadow:""}},u=(0,n.makeClassName)("Icon"),m=r.default.forwardRef((e,m)=>{let{icon:p,variant:g="simple",tooltip:f,size:h=a.Sizes.SM,color:b,className:_}=e,v=(0,t.__rest)(e,["icon","variant","tooltip","size","color","className"]),C=((e,t)=>{switch(e){case"simple":return{textColor:t?(0,n.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:"",borderColor:"",ringColor:""};case"light":return{textColor:t?(0,n.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand-muted dark:bg-dark-tremor-brand-muted",borderColor:"",ringColor:""};case"shadow":return{textColor:t?(0,n.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:"border-tremor-border dark:border-dark-tremor-border",ringColor:""};case"solid":return{textColor:t?(0,n.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand-inverted dark:text-dark-tremor-brand-inverted",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-brand dark:bg-dark-tremor-brand",borderColor:"border-tremor-brand-inverted dark:border-dark-tremor-brand-inverted",ringColor:"ring-tremor-ring dark:ring-dark-tremor-ring"};case"outlined":return{textColor:t?(0,n.getColorClassNames)(t,l.colorPalette.text).textColor:"text-tremor-brand dark:text-dark-tremor-brand",bgColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,l.colorPalette.background).bgColor,"bg-opacity-20"):"bg-tremor-background dark:bg-dark-tremor-background",borderColor:t?(0,n.getColorClassNames)(t,l.colorPalette.ring).borderColor:"border-tremor-brand-subtle dark:border-dark-tremor-brand-subtle",ringColor:t?(0,i.tremorTwMerge)((0,n.getColorClassNames)(t,l.colorPalette.ring).ringColor,"ring-opacity-40"):"ring-tremor-brand-muted dark:ring-dark-tremor-brand-muted"}}})(g,b),{tooltipProps:x,getReferenceProps:A}=(0,o.useTooltip)();return r.default.createElement("span",Object.assign({ref:(0,n.mergeRefs)([m,x.refs.setReference]),className:(0,i.tremorTwMerge)(u("root"),"inline-flex shrink-0 items-center justify-center",C.bgColor,C.textColor,C.borderColor,C.ringColor,c[g].rounded,c[g].border,c[g].shadow,c[g].ring,s[h].paddingX,s[h].paddingY,_)},A,v),r.default.createElement(o.default,Object.assign({text:f},x)),r.default.createElement(p,{className:(0,i.tremorTwMerge)(u("icon"),"shrink-0",d[h].height,d[h].width)}))});m.displayName="Icon",e.s(["default",()=>m],728889)},752978,e=>{"use strict";var t=e.i(728889);e.s(["Icon",()=>t.default])},591935,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))});e.s(["PencilAltIcon",0,r],591935)},166406,e=>{"use strict";var t=e.i(190144);e.s(["CopyOutlined",()=>t.default])},916925,e=>{"use strict";var t,r=((t={}).A2A_Agent="A2A Agent",t.AI21="Ai21",t.AI21_CHAT="Ai21 Chat",t.AIML="AI/ML API",t.AIOHTTP_OPENAI="Aiohttp Openai",t.Anthropic="Anthropic",t.ANTHROPIC_TEXT="Anthropic Text",t.AssemblyAI="AssemblyAI",t.AUTO_ROUTER="Auto Router",t.Bedrock="Amazon Bedrock",t.BedrockMantle="Amazon Bedrock Mantle",t.SageMaker="AWS SageMaker",t.Azure="Azure",t.Azure_AI_Studio="Azure AI Foundry (Studio)",t.AZURE_TEXT="Azure Text",t.BASETEN="Baseten",t.BYTEZ="Bytez",t.Cerebras="Cerebras",t.CLARIFAI="Clarifai",t.CLOUDFLARE="Cloudflare",t.CODESTRAL="Codestral",t.Cohere="Cohere",t.COHERE_CHAT="Cohere Chat",t.COMETAPI="Cometapi",t.COMPACTIFAI="Compactifai",t.Cursor="Cursor",t.Dashscope="Dashscope",t.Databricks="Databricks (Qwen API)",t.DATAROBOT="Datarobot",t.DeepInfra="DeepInfra",t.Deepgram="Deepgram",t.Deepseek="Deepseek",t.DOCKER_MODEL_RUNNER="Docker Model Runner",t.DOTPROMPT="Dotprompt",t.ElevenLabs="ElevenLabs",t.EMPOWER="Empower",t.FalAI="Fal AI",t.FEATHERLESS_AI="Featherless Ai",t.FireworksAI="Fireworks AI",t.FRIENDLIAI="Friendliai",t.GALADRIEL="Galadriel",t.GITHUB_COPILOT="Github Copilot",t.Google_AI_Studio="Google AI Studio",t.GradientAI="GradientAI",t.Groq="Groq",t.HEROKU="Heroku",t.Hosted_Vllm="vllm",t.HUGGINGFACE="Huggingface",t.HYPERBOLIC="Hyperbolic",t.Infinity="Infinity",t.JinaAI="Jina AI",t.LAMBDA_AI="Lambda Ai",t.LEMONADE="Lemonade",t.LLAMAFILE="Llamafile",t.LM_STUDIO="Lm Studio",t.LLAMA="Meta Llama",t.MARITALK="Maritalk",t.MiniMax="MiniMax",t.MistralAI="Mistral AI",t.MOONSHOT="Moonshot",t.MORPH="Morph",t.NEBIUS="Nebius",t.NLP_CLOUD="Nlp Cloud",t.NOVITA="Novita",t.NSCALE="Nscale",t.NVIDIA_NIM="Nvidia Nim",t.Ollama="Ollama",t.OLLAMA_CHAT="Ollama Chat",t.OOBABOOGA="Oobabooga",t.OpenAI="OpenAI",t.OPENAI_LIKE="Openai Like",t.OpenAI_Compatible="OpenAI-Compatible Endpoints (Together AI, etc.)",t.OpenAI_Text="OpenAI Text Completion",t.OpenAI_Text_Compatible="OpenAI-Compatible Text Completion Models (Together AI, etc.)",t.Openrouter="Openrouter",t.Oracle="Oracle Cloud Infrastructure (OCI)",t.OVHCLOUD="Ovhcloud",t.Perplexity="Perplexity",t.PETALS="Petals",t.PG_VECTOR="Pg Vector",t.PREDIBASE="Predibase",t.RECRAFT="Recraft",t.REPLICATE="Replicate",t.RunwayML="RunwayML",t.SAGEMAKER_LEGACY="Sagemaker",t.Sambanova="Sambanova",t.SAP="SAP Generative AI Hub",t.Snowflake="Snowflake",t.TEXT_COMPLETION_CODESTRAL="Text-Completion-Codestral",t.TogetherAI="TogetherAI",t.TOPAZ="Topaz",t.Triton="Triton",t.V0="V0",t.VERCEL_AI_GATEWAY="Vercel Ai Gateway",t.Vertex_AI="Vertex AI (Anthropic, Gemini, etc.)",t.VERTEX_AI_BETA="Vertex Ai Beta",t.VLLM="Vllm",t.VolcEngine="VolcEngine",t.Voyage="Voyage AI",t.WANDB="Wandb",t.WATSONX="Watsonx",t.WATSONX_TEXT="Watsonx Text",t.xAI="xAI",t.XINFERENCE="Xinference",t);let o={A2A_Agent:"a2a_agent",AI21:"ai21",AI21_CHAT:"ai21_chat",AIML:"aiml",AIOHTTP_OPENAI:"aiohttp_openai",Anthropic:"anthropic",ANTHROPIC_TEXT:"anthropic_text",AssemblyAI:"assemblyai",AUTO_ROUTER:"auto_router",Azure:"azure",Azure_AI_Studio:"azure_ai",AZURE_TEXT:"azure_text",BASETEN:"baseten",Bedrock:"bedrock",BedrockMantle:"bedrock_mantle",BYTEZ:"bytez",Cerebras:"cerebras",CLARIFAI:"clarifai",CLOUDFLARE:"cloudflare",CODESTRAL:"codestral",Cohere:"cohere",COHERE_CHAT:"cohere_chat",COMETAPI:"cometapi",COMPACTIFAI:"compactifai",Cursor:"cursor",Dashscope:"dashscope",Databricks:"databricks",DATAROBOT:"datarobot",DeepInfra:"deepinfra",Deepgram:"deepgram",Deepseek:"deepseek",DOCKER_MODEL_RUNNER:"docker_model_runner",DOTPROMPT:"dotprompt",ElevenLabs:"elevenlabs",EMPOWER:"empower",FalAI:"fal_ai",FEATHERLESS_AI:"featherless_ai",FireworksAI:"fireworks_ai",FRIENDLIAI:"friendliai",GALADRIEL:"galadriel",GITHUB_COPILOT:"github_copilot",Google_AI_Studio:"gemini",GradientAI:"gradient_ai",Groq:"groq",HEROKU:"heroku",Hosted_Vllm:"hosted_vllm",HUGGINGFACE:"huggingface",HYPERBOLIC:"hyperbolic",Infinity:"infinity",JinaAI:"jina_ai",LAMBDA_AI:"lambda_ai",LEMONADE:"lemonade",LLAMAFILE:"llamafile",LLAMA:"meta_llama",LM_STUDIO:"lm_studio",MARITALK:"maritalk",MiniMax:"minimax",MistralAI:"mistral",MOONSHOT:"moonshot",MORPH:"morph",NEBIUS:"nebius",NLP_CLOUD:"nlp_cloud",NOVITA:"novita",NSCALE:"nscale",NVIDIA_NIM:"nvidia_nim",Ollama:"ollama",OLLAMA_CHAT:"ollama_chat",OOBABOOGA:"oobabooga",OpenAI:"openai",OPENAI_LIKE:"openai_like",OpenAI_Compatible:"openai",OpenAI_Text:"text-completion-openai",OpenAI_Text_Compatible:"text-completion-openai",Openrouter:"openrouter",Oracle:"oci",OVHCLOUD:"ovhcloud",Perplexity:"perplexity",PETALS:"petals",PG_VECTOR:"pg_vector",PREDIBASE:"predibase",RECRAFT:"recraft",REPLICATE:"replicate",RunwayML:"runwayml",SAGEMAKER_LEGACY:"sagemaker",SageMaker:"sagemaker_chat",Sambanova:"sambanova",SAP:"sap",Snowflake:"snowflake",TEXT_COMPLETION_CODESTRAL:"text-completion-codestral",TogetherAI:"together_ai",TOPAZ:"topaz",Triton:"triton",V0:"v0",VERCEL_AI_GATEWAY:"vercel_ai_gateway",Vertex_AI:"vertex_ai",VERTEX_AI_BETA:"vertex_ai_beta",VLLM:"vllm",VolcEngine:"volcengine",Voyage:"voyage",WANDB:"wandb",WATSONX:"watsonx",WATSONX_TEXT:"watsonx_text",xAI:"xai",XINFERENCE:"xinference"},a="../ui/assets/logos/",i={"A2A Agent":`${a}a2a_agent.png`,Ai21:`${a}ai21.svg`,"Ai21 Chat":`${a}ai21.svg`,"AI/ML API":`${a}aiml_api.svg`,"Aiohttp Openai":`${a}openai_small.svg`,Anthropic:`${a}anthropic.svg`,"Anthropic Text":`${a}anthropic.svg`,AssemblyAI:`${a}assemblyai_small.png`,Azure:`${a}microsoft_azure.svg`,"Azure AI Foundry (Studio)":`${a}microsoft_azure.svg`,"Azure Text":`${a}microsoft_azure.svg`,Baseten:`${a}baseten.svg`,"Amazon Bedrock":`${a}bedrock.svg`,"Amazon Bedrock Mantle":`${a}bedrock.svg`,"AWS SageMaker":`${a}bedrock.svg`,Cerebras:`${a}cerebras.svg`,Cloudflare:`${a}cloudflare.svg`,Codestral:`${a}mistral.svg`,Cohere:`${a}cohere.svg`,"Cohere Chat":`${a}cohere.svg`,Cometapi:`${a}cometapi.svg`,Cursor:`${a}cursor.svg`,"Databricks (Qwen API)":`${a}databricks.svg`,Dashscope:`${a}dashscope.svg`,Deepseek:`${a}deepseek.svg`,Deepgram:`${a}deepgram.png`,DeepInfra:`${a}deepinfra.png`,ElevenLabs:`${a}elevenlabs.png`,"Fal AI":`${a}fal_ai.jpg`,"Featherless Ai":`${a}featherless.svg`,"Fireworks AI":`${a}fireworks.svg`,Friendliai:`${a}friendli.svg`,"Github Copilot":`${a}github_copilot.svg`,"Google AI Studio":`${a}google.svg`,GradientAI:`${a}gradientai.svg`,Groq:`${a}groq.svg`,vllm:`${a}vllm.png`,Huggingface:`${a}huggingface.svg`,Hyperbolic:`${a}hyperbolic.svg`,Infinity:`${a}infinity.png`,"Jina AI":`${a}jina.png`,"Lambda Ai":`${a}lambda.svg`,"Lm Studio":`${a}lmstudio.svg`,"Meta Llama":`${a}meta_llama.svg`,MiniMax:`${a}minimax.svg`,"Mistral AI":`${a}mistral.svg`,Moonshot:`${a}moonshot.svg`,Morph:`${a}morph.svg`,Nebius:`${a}nebius.svg`,Novita:`${a}novita.svg`,"Nvidia Nim":`${a}nvidia_nim.svg`,Ollama:`${a}ollama.svg`,"Ollama Chat":`${a}ollama.svg`,Oobabooga:`${a}openai_small.svg`,OpenAI:`${a}openai_small.svg`,"Openai Like":`${a}openai_small.svg`,"OpenAI Text Completion":`${a}openai_small.svg`,"OpenAI-Compatible Text Completion Models (Together AI, etc.)":`${a}openai_small.svg`,"OpenAI-Compatible Endpoints (Together AI, etc.)":`${a}openai_small.svg`,Openrouter:`${a}openrouter.svg`,"Oracle Cloud Infrastructure (OCI)":`${a}oracle.svg`,Perplexity:`${a}perplexity-ai.svg`,Recraft:`${a}recraft.svg`,Replicate:`${a}replicate.svg`,RunwayML:`${a}runwayml.png`,Sagemaker:`${a}bedrock.svg`,Sambanova:`${a}sambanova.svg`,"SAP Generative AI Hub":`${a}sap.png`,Snowflake:`${a}snowflake.svg`,"Text-Completion-Codestral":`${a}mistral.svg`,TogetherAI:`${a}togetherai.svg`,Topaz:`${a}topaz.svg`,Triton:`${a}nvidia_triton.png`,V0:`${a}v0.svg`,"Vercel Ai Gateway":`${a}vercel.svg`,"Vertex AI (Anthropic, Gemini, etc.)":`${a}google.svg`,"Vertex Ai Beta":`${a}google.svg`,Vllm:`${a}vllm.png`,VolcEngine:`${a}volcengine.png`,"Voyage AI":`${a}voyage.webp`,Watsonx:`${a}watsonx.svg`,"Watsonx Text":`${a}watsonx.svg`,xAI:`${a}xai.svg`,Xinference:`${a}xinference.svg`};e.s(["Providers",()=>r,"getPlaceholder",0,e=>{if("AI/ML API"===e)return"aiml/flux-pro/v1.1";if("Vertex AI (Anthropic, Gemini, etc.)"===e)return"gemini-pro";if("Anthropic"==e)return"claude-3-opus";if("Amazon Bedrock"==e)return"claude-3-opus";if("AWS SageMaker"==e)return"sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b";else if("Google AI Studio"==e)return"gemini-pro";else if("Azure AI Foundry (Studio)"==e)return"azure_ai/command-r-plus";else if("Azure"==e)return"my-deployment";else if("Oracle Cloud Infrastructure (OCI)"==e)return"oci/xai.grok-4";else if("Snowflake"==e)return"snowflake/mistral-7b";else if("Voyage AI"==e)return"voyage/";else if("Jina AI"==e)return"jina_ai/";else if("VolcEngine"==e)return"volcengine/<any-model-on-volcengine>";else if("DeepInfra"==e)return"deepinfra/<any-model-on-deepinfra>";else if("Fal AI"==e)return"fal_ai/fal-ai/flux-pro/v1.1-ultra";else if("RunwayML"==e)return"runwayml/gen4_turbo";else if("Watsonx"===e)return"watsonx/ibm/granite-3-3-8b-instruct";else if("Cursor"===e)return"cursor/claude-4-sonnet";else return"gpt-3.5-turbo"},"getProviderLogoAndName",0,e=>{if(!e)return{logo:"",displayName:"-"};if("gemini"===e.toLowerCase()){let e="Google AI Studio";return{logo:i[e],displayName:e}}let t=Object.keys(o).find(t=>o[t].toLowerCase()===e.toLowerCase());if(!t)return{logo:"",displayName:e};let a=r[t];return{logo:i[a],displayName:a}},"getProviderModels",0,(e,t)=>{console.log(`Provider key: ${e}`);let r=o[e];console.log(`Provider mapped to: ${r}`);let a=[];return e&&"object"==typeof t&&(Object.entries(t).forEach(([e,t])=>{if(null!==t&&"object"==typeof t&&"litellm_provider"in t){let o=t.litellm_provider;(o===r||"string"==typeof o&&o.includes(r))&&a.push(e)}}),"Cohere"==e&&(console.log("Adding cohere chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"cohere_chat"===t.litellm_provider&&a.push(e)})),"AWS SageMaker"==e&&(console.log("Adding sagemaker chat models"),Object.entries(t).forEach(([e,t])=>{null!==t&&"object"==typeof t&&"litellm_provider"in t&&"sagemaker_chat"===t.litellm_provider&&a.push(e)}))),a},"providerLogoMap",0,i,"provider_map",0,o])},209261,e=>{"use strict";e.s(["extractCategories",0,e=>{let t=new Set;return e.forEach(e=>{e.category&&""!==e.category.trim()&&t.add(e.category)}),["All",...Array.from(t).sort(),"Other"]},"filterPluginsByCategory",0,(e,t)=>"All"===t?e:"Other"===t?e.filter(e=>!e.category||""===e.category.trim()):e.filter(e=>e.category===t),"filterPluginsBySearch",0,(e,t)=>{if(!t||""===t.trim())return e;let r=t.toLowerCase().trim();return e.filter(e=>{let t=e.name.toLowerCase().includes(r),o=e.description?.toLowerCase().includes(r)||!1,a=e.keywords?.some(e=>e.toLowerCase().includes(r))||!1;return t||o||a})},"formatDateString",0,e=>{if(!e)return"N/A";try{return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"})}catch(e){return"Invalid date"}},"formatInstallCommand",0,e=>"github"===e.source.source&&e.source.repo?`/plugin marketplace add ${e.source.repo}`:"url"===e.source.source&&e.source.url?`/plugin marketplace add ${e.source.url}`:`/plugin marketplace add ${e.name}`,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"getSourceDisplayText",0,e=>"github"===e.source&&e.repo?`GitHub: ${e.repo}`:"url"===e.source&&e.url?e.url:"Unknown source","getSourceLink",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:"url"===e.source&&e.url?e.url:null,"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidUrl",0,e=>{if(!e)return!0;try{return new URL(e),!0}catch{return!1}},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},190272,785913,e=>{"use strict";var t,r,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.RESPONSES="responses",t.IMAGE_EDITS="image_edits",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t),a=((r={}).IMAGE="image",r.VIDEO="video",r.CHAT="chat",r.RESPONSES="responses",r.IMAGE_EDITS="image_edits",r.ANTHROPIC_MESSAGES="anthropic_messages",r.EMBEDDINGS="embeddings",r.SPEECH="speech",r.TRANSCRIPTION="transcription",r.A2A_AGENTS="a2a_agents",r.MCP="mcp",r.REALTIME="realtime",r);let i={image_generation:"image",video_generation:"video",chat:"chat",responses:"responses",image_edits:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings"};e.s(["EndpointType",()=>a,"getEndpointType",0,e=>{if(console.log("getEndpointType:",e),Object.values(o).includes(e)){let t=i[e];return console.log("endpointType:",t),t}return"chat"}],785913),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:r,accessToken:o,apiKey:i,inputMessage:n,chatHistory:l,selectedTags:s,selectedVectorStores:d,selectedGuardrails:c,selectedPolicies:u,selectedMCPServers:m,mcpServers:p,mcpServerToolRestrictions:g,selectedVoice:f,endpointType:h,selectedModel:b,selectedSdk:_,proxySettings:v}=e,C="session"===r?o:i,x=window.location.origin,A=v?.LITELLM_UI_API_DOC_BASE_URL;A&&A.trim()?x=A:v?.PROXY_BASE_URL&&(x=v.PROXY_BASE_URL);let y=n||"Your prompt here",I=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),k=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),w={};s.length>0&&(w.tags=s),d.length>0&&(w.vector_stores=d),c.length>0&&(w.guardrails=c),u.length>0&&(w.policies=u);let E=b||"your-model-name",T="azure"===_?`import openai

client = openai.AzureOpenAI(
	api_key="${C||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${C||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"
)`;switch(h){case a.CHAT:{let e=Object.keys(w).length>0,r="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${E}",
    messages=${JSON.stringify(o,null,4)}${r}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${E}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${I}"
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
`;break}case a.RESPONSES:{let e=Object.keys(w).length>0,r="";if(e){let e=JSON.stringify({metadata:w},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();r=`,
    extra_body=${e}`}let o=k.length>0?k:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${E}",
    input=${JSON.stringify(o,null,4)}${r}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${E}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${I}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${r}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===_?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${E}",
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${E}",
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
`;break;case a.IMAGE_EDITS:t="azure"===_?`
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${E}",
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
prompt = "${I}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${E}",
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
	model="${E}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${E}",
	file=audio_file${n?`,
	prompt="${n.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${E}",
	input="${n||"Your text to convert to speech here"}",
	voice="${f}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${E}",
#     input="${n||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${T}
${t}`}],190272)},94629,e=>{"use strict";var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"}))});e.s(["SwitchVerticalIcon",0,r],94629)},991124,e=>{"use strict";let t=(0,e.i(475254).default)("copy",[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]]);e.s(["default",()=>t])},879664,e=>{"use strict";let t=(0,e.i(475254).default)("info",[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M12 16v-4",key:"1dtifu"}],["path",{d:"M12 8h.01",key:"e9boi3"}]]);e.s(["default",()=>t])},798496,e=>{"use strict";var t=e.i(843476),r=e.i(152990),o=e.i(682830),a=e.i(271645),i=e.i(269200),n=e.i(427612),l=e.i(64848),s=e.i(942232),d=e.i(496020),c=e.i(977572),u=e.i(94629),m=e.i(360820),p=e.i(871943);function g({data:e=[],columns:g,isLoading:f=!1,defaultSorting:h=[],pagination:b,onPaginationChange:_,enablePagination:v=!1,onRowClick:C}){let[x,A]=a.default.useState(h),[y]=a.default.useState("onChange"),[I,k]=a.default.useState({}),[w,E]=a.default.useState({}),T=(0,r.useReactTable)({data:e,columns:g,state:{sorting:x,columnSizing:I,columnVisibility:w,...v&&b?{pagination:b}:{}},columnResizeMode:y,onSortingChange:A,onColumnSizingChange:k,onColumnVisibilityChange:E,...v&&_?{onPaginationChange:_}:{},getCoreRowModel:(0,o.getCoreRowModel)(),getSortedRowModel:(0,o.getSortedRowModel)(),...v?{getPaginationRowModel:(0,o.getPaginationRowModel)()}:{},enableSorting:!0,enableColumnResizing:!0,defaultColumn:{minSize:40,maxSize:500}});return(0,t.jsx)("div",{className:"rounded-lg custom-border relative",children:(0,t.jsx)("div",{className:"overflow-x-auto",children:(0,t.jsx)("div",{className:"relative min-w-full",children:(0,t.jsxs)(i.Table,{className:"[&_td]:py-2 [&_th]:py-2",style:{width:T.getTotalSize(),minWidth:"100%",tableLayout:"fixed"},children:[(0,t.jsx)(n.TableHead,{children:T.getHeaderGroups().map(e=>(0,t.jsx)(d.TableRow,{children:e.headers.map(e=>(0,t.jsxs)(l.TableHeaderCell,{className:`py-1 h-8 relative ${"actions"===e.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.id?120:e.getSize(),position:"actions"===e.id?"sticky":"relative",right:"actions"===e.id?0:"auto"},onClick:e.column.getCanSort()?e.column.getToggleSortingHandler():void 0,children:[(0,t.jsxs)("div",{className:"flex items-center justify-between gap-2",children:[(0,t.jsx)("div",{className:"flex items-center",children:e.isPlaceholder?null:(0,r.flexRender)(e.column.columnDef.header,e.getContext())}),"actions"!==e.id&&e.column.getCanSort()&&(0,t.jsx)("div",{className:"w-4",children:e.column.getIsSorted()?({asc:(0,t.jsx)(m.ChevronUpIcon,{className:"h-4 w-4 text-blue-500"}),desc:(0,t.jsx)(p.ChevronDownIcon,{className:"h-4 w-4 text-blue-500"})})[e.column.getIsSorted()]:(0,t.jsx)(u.SwitchVerticalIcon,{className:"h-4 w-4 text-gray-400"})})]}),e.column.getCanResize()&&(0,t.jsx)("div",{onMouseDown:e.getResizeHandler(),onTouchStart:e.getResizeHandler(),className:`absolute right-0 top-0 h-full w-2 cursor-col-resize select-none touch-none ${e.column.getIsResizing()?"bg-blue-500":"hover:bg-blue-200"}`})]},e.id))},e.id))}),(0,t.jsx)(s.TableBody,{children:f?(0,t.jsx)(d.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"🚅 Loading models..."})})})}):T.getRowModel().rows.length>0?T.getRowModel().rows.map(e=>(0,t.jsx)(d.TableRow,{onClick:()=>C?.(e.original),className:C?"cursor-pointer hover:bg-gray-50":"",children:e.getVisibleCells().map(e=>(0,t.jsx)(c.TableCell,{className:`py-0.5 overflow-hidden ${"actions"===e.column.id?"sticky right-0 bg-white shadow-[-4px_0_8px_-6px_rgba(0,0,0,0.1)] w-[120px] ml-8":""} ${e.column.columnDef.meta?.className||""}`,style:{width:"actions"===e.column.id?120:e.column.getSize(),position:"actions"===e.column.id?"sticky":"relative",right:"actions"===e.column.id?0:"auto"},children:(0,r.flexRender)(e.column.columnDef.cell,e.getContext())},e.id))},e.id)):(0,t.jsx)(d.TableRow,{children:(0,t.jsx)(c.TableCell,{colSpan:g.length,className:"h-8 text-center",children:(0,t.jsx)("div",{className:"text-center text-gray-500",children:(0,t.jsx)("p",{children:"No models found"})})})})})]})})})})}e.s(["ModelDataTable",()=>g])},195529,e=>{"use strict";var t=e.i(843476),r=e.i(934879),o=e.i(135214);e.s(["default",0,()=>{let{accessToken:e,premiumUser:a,userRole:i}=(0,o.default)();return(0,t.jsx)(r.default,{accessToken:e,publicPage:!1,premiumUser:a,userRole:i})}])}]);