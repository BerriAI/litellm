(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,205133,408085,794494,806727,569389,227690,189793,539277,e=>{var t=e.i(160948);e.s(["f",0,t],205133),e.s(["f",0,t],408085),e.s(["f",0,t],794494),e.s(["f",0,t],806727),e.s(["f",0,t],569389),e.s(["f",0,t],227690),e.s(["f",0,t],189793),e.s(["f",0,t],539277)},434626,e=>{var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["f",0,i],434626)},180127,e=>{let t=(0,e.i(475254).f)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["f",0,t])},871689,e=>{var t=e.i(180127);e.s(["f",()=>t.f])},233565,e=>{var t=e.i(246349);e.s(["f",()=>t.f])},373488,e=>{let t=(0,e.i(475254).f)("ellipsis",[["circle",{cx:"12",cy:"12",r:"1",key:"41hilf"}],["circle",{cx:"19",cy:"12",r:"1",key:"1wjl8i"}],["circle",{cx:"5",cy:"12",r:"1",key:"1pcz8c"}]]);e.s(["f",0,t])},541071,e=>{var t=e.i(373488);e.s(["f",()=>t.f])},546467,e=>{let t=(0,e.i(475254).f)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["f",0,t])},778917,e=>{var t=e.i(546467);e.s(["f",()=>t.f])},332102,e=>{let t=(0,e.i(475254).f)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["f",0,t],332102)},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},345771,e=>{var t=e.i(271645),i=e.i(645141);let r={area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0,"!doctype":!0,"!DOCTYPE":!0},n=/\s([^'"/\s><]+?)[\s/>]|([^\s=]+)=\s?("[^"]*"|'[^']*')/g;function a(e){let t={type:"tag",name:"",voidElement:!1,attrs:{},children:[]},i=e.match(/<\/?([^\s]+?)[/\s>]/);if(i&&(t.name=i[1],(r[i[1]]||"/"===e.charAt(e.length-2))&&(t.voidElement=!0),t.name.startsWith("!--"))){let t=e.indexOf("-->");return{type:"comment",comment:-1!==t?e.slice(4,t):""}}let a=new RegExp(n),s=null;for(;null!==(s=a.exec(e));)if(s[0].trim())if(s[1]){let e=s[1].trim(),i=[e,null],r=e.indexOf("=");r>-1&&(i=[e.slice(0,r),e.slice(r+1)]),t.attrs[i[0]]=i[1],a.lastIndex--}else s[2]&&(t.attrs[s[2]]=s[3].trim().substring(1,s[3].length-1));return t}let s=/<!--[\s\S]*?-->|<[a-zA-Z0-9\-!/](?:"[^"]*"|'[^']*'|[^'">])*>/g,l=/<\/?([^\s]+?)[/\s>]/,o=/^\s*$/,d=/^(script|style)$/i,u=Object.create(null);var p=function(e,t){let i,r,n,p=t&&t.components||u,c=t&&t.allowedTags,m=!1;if(c){let t,i="function"==typeof c?c:function(e){return c.indexOf(e)>-1},r="",n=0;for(s.lastIndex=0;t=s.exec(e);){let a=t[0];r+=e.slice(n,t.index);let o=a.match(l);a.startsWith("<!--")||o&&i(o[1])?(r+=a,n=t.index+a.length):(m=!0,r+="\0",s.lastIndex=n=t.index+1)}e=r+e.slice(n)}let g=[],f=[],h=-1,x=!1,b=0;if(0!==e.indexOf("<")){let t=e.indexOf("<");g.push({type:"text",content:-1===t?e:e.substring(0,t)})}let _=[];for(;n=s.exec(e);)_.push(n);return _.forEach(function(e,t){let i=e[0];if(!i||i.startsWith("<!--"))return;let r=0,n=0,a=-1,s=null;for(let e=0;e<i.length;e++){let t=i.charAt(e);s?t===s&&(s=null):'"'===t||"'"===t?s=t:"<"===t?2==++r&&(a=e):">"===t&&n++}let l=a>-1&&/[a-zA-Z0-9\-!/]/.test(i.charAt(a+1));if(r>n&&l){let e=i.substring(0,a),r=i.substring(e.length);_[t][0]=r,_[t].index+=e.length}}),_.forEach(function(t,n){let s,l,u=t[0];if(!u)return;let c=t.index;if(c<b)return;if(x)if(u!=="</"+i.name+">")return;else x=!1;let m="/"!==u.charAt(1),y=u.startsWith("<!--"),v=c+u.length,j=e.charAt(v),w=_[n+1];if("<"===j&&w){let t=e.substring(v,w.index);s=t.split("<").length>t.split(">").length}if(y){let t=a(u);if(h<0)return g.push(t),g;(l=f[h]).children.push(t);let i=e.slice(v,w?w.index:void 0);return i.length>0&&l.children.push({type:"text",content:i}),g}if(m){h++,"tag"===(i=a(u)).type&&p[i.name]&&(i.type="component",x=!0);let t=!1;if(!x&&!i.voidElement&&d.test(i.name)){t=!0,r||(r=e.toLowerCase());let n=r.indexOf("</"+i.name.toLowerCase()+">",v),a=-1===n?e.length:n,s=e.slice(v,a);s&&i.children.push({type:"text",content:s}),b=a}i.voidElement||x||t||!j||"<"===j||i.children.push({type:"text",content:e.slice(v,w?w.index:void 0)}),0===h&&g.push(i),(l=f[h-1])&&l.children.push(i),f[h]=i}if((!m||i.voidElement)&&(h>-1&&(i.voidElement||i.name===u.slice(2,-1))&&(i=-1==--h?g:f[h]),!x&&("<"!==j||s)&&j)){l=-1===h?g:f[h].children;let t=w?w.index:-1,i=e.slice(v,-1===t?void 0:t);o.test(i)&&(i=" "),(t>-1&&h+l.length>=0||" "!==i)&&l.push({type:"text",content:i})}}),m&&function e(t){t.forEach(function(t){if("text"===t.type){t.content=t.content.split("\0").join("<");return}if("comment"===t.type){t.comment=t.comment.split("\0").join("<");return}for(let e in t.attrs){let i=t.attrs[e];"string"==typeof i&&i.indexOf("\0")>-1&&(t.attrs[e]=i.split("\0").join("<"))}t.children.length&&e(t.children)})}(g),g},c=e.i(529894),m=e.i(305706),g=e.i(596043),f=e.i(247464);let h=(e,t)=>{if(!e)return!1;let i=e.props?.children??e.children;return t?i.length>0:!!i},x=e=>{if(!e)return[];let t=e.props?.children??e.children;return e.props?.i18nIsDynamicList?b(t):t},b=e=>Array.isArray(e)?e:[e],_=(e,i,r,n)=>{if(!e)return"";let a="",s=b(e),l=i?.transSupportBasicHtmlNodes?i.transKeepBasicHtmlNodesFor??[]:[];return s.forEach((e,s)=>{if((0,c.Z)(e)){a+=`${e}`;return}if((0,t.isValidElement)(e)){let{props:t,type:o}=e,d=Object.keys(t).length,u=l.indexOf(o)>-1,p=t.children;if(!p&&u&&!d){a+=`<${o}/>`;return}if(!p&&(!u||d)||t.i18nIsDynamicList){a+=`<${s}></${s}>`;return}if(u&&d<=1){let e=(0,c.Z)(p)?p:_(p,i,r,n);a+=`<${o}>${e}</${o}>`;return}let m=_(p,i,r,n);a+=`<${s}>${m}</${s}>`;return}if(null===e)return void(0,c.S)(r,"TRANS_NULL_VALUE","Passed in a null value as child",{i18nKey:n});if((0,c.C)(e)){let{format:t,...i}=e,s=Object.keys(i);if(1===s.length){let e=t?`${s[0]}, ${t}`:s[0];a+=`{{${e}}}`;return}if(!s.length&&1===Object.keys(e).length&&"format"in e){a+="{{format}}";return}(0,c.S)(r,"TRANS_INVALID_OBJ","Invalid child - Object should only have keys {{ value, format }} (format is optional).",{i18nKey:n,child:e});return}(0,c.S)(r,"TRANS_INVALID_VAR","Passed in a variable like {number} - pass variables for interpolation as full objects like {{number}}.",{i18nKey:n,child:e})}),a},y=(e,i,r)=>{let n=e.key||i,a=(0,t.cloneElement)(e,{key:n});return!a.props||!a.props.children||0>r.indexOf(`${i}/>`)&&0>r.indexOf(`${i} />`)?a:(0,t.createElement)(function(){return(0,t.createElement)(t.Fragment,null,a)},{key:n})};var v=e.i(806593);e.s(["f",0,function({children:e,count:r,parent:n,i18nKey:a,context:s,tOptions:l={},values:o,defaults:d,components:u,ns:j,i18n:w,t:k,shouldUnescape:N,...$}){let{i18n:C,defaultNS:O}=(0,t.useContext)(v.k)||{},P=w||C||(0,g.e)(),E=k||P?.t.bind(P);return function({children:e,count:r,parent:n,i18nKey:a,context:s,tOptions:l={},values:o,defaults:d,components:u,ns:v,i18n:j,t:w,shouldUnescape:k,...N}){let $=j||(0,g.e)();if(!$)return(0,c.R)($,"NO_I18NEXT_INSTANCE","Trans: You need to pass in an i18next instance using initReactI18next or by passing it via props or context. In monorepo setups, make sure there is only one instance of react-i18next.",{i18nKey:a}),e;let C=w||$.t.bind($)||(e=>e),O={...(0,m.n)(),...$.options?.react},P=v||(C.ns?void 0:$.options?.defaultNS||"translation");(0,c.Z)(P)&&(P=[P]);let{transDefaultProps:E}=O,I=E?.tOptions?{...E.tOptions,...l}:l,A=k??E?.shouldUnescape,D=E?.values?{...E.values,...o}:o,S=E?.components?{...E.components,...u}:u,z=_(e,O,$,a),L=Array.isArray(a)?a[a.length-1]:a,T=d||I?.defaultValue||z||O.transEmptyNodeValue||("function"==typeof L?(0,i.e)(L):L),{hashTransKey:R}=O,M=a||(R?R(z||T):z||T);o=$.options?.interpolation?.defaultVariables?{...$.options.interpolation.defaultVariables,...D}:D;let F=(e=>{let i={};if(!e)return i;let r=e=>{b(e).forEach(e=>{(0,c.Z)(e)||(h(e)?r(x(e)):(0,c.C)(e)&&!(0,t.isValidElement)(e)&&Object.assign(i,e))})};return r(e),i})(e);F&&"number"==typeof F.count&&void 0===r&&(r=F.count),void 0===r&&(r=I?.count);let H=!o&&(void 0===r||$.options?.interpolation?.alwaysFormat)&&e?{interpolation:{...I.interpolation,prefix:"#$?",suffix:"?$#"}}:I.interpolation,Z={...I,context:s||I.context,count:r,...o,...H,defaultValue:T,ns:P},U=M?C(M,Z):T;U===M&&T&&(U=T);let V=((e,t,i,r)=>{if(!e)return null;if(Array.isArray(e))return e.map((e,i)=>y(e,i,t));if((0,c.C)(e)){let i;return i={},Object.keys(e).forEach(r=>{Object.assign(i,{[r]:y(e[r],r,t)})}),i}return(0,c.R)(i,"TRANS_INVALID_COMPONENTS",'<Trans /> "components" prop expects an object or array',{i18nKey:r}),null})(S,U,$,a),B=V||e,q=null;!(!(0,c.C)(V)||Array.isArray(V))&&Object.keys(V).reduce((e,t)=>e&&Number.isNaN(Number.parseFloat(t)),!0)?(q=V,B=e):V&&!Array.isArray(V)&&(q=V,B=void 0);let Y=((e,i,r,n,a,s,l)=>{if(""===r)return[];let o=a.transKeepBasicHtmlNodesFor||[],d=r&&new RegExp(o.map(e=>`<${e}`).join("|")).test(r);if(!e&&!i&&!d&&!l)return[r];let u=i??{},g=e=>{b(e).forEach(e=>{(0,c.Z)(e)||(h(e)?g(x(e)):(0,c.C)(e)&&!(0,t.isValidElement)(e)&&Object.assign(u,e))})};g(e);let _=Object.keys(u),y=p(`<0>${r}</0>`,{allowedTags:e=>/^\d+$/.test(e)||o.indexOf(e)>-1||_.indexOf(e)>-1}),v={...u,...s},j=(e,i,r)=>{let n=x(e),a=k(n,i.children,r);return b(n).every(t.isValidElement)&&0===a.length||e.props?.i18nIsDynamicList?n:a},w=(e,i,r,n,a)=>{e.dummy?(e.children=i,r.push((0,t.cloneElement)(e,{key:n},a?void 0:i))):r.push(...t.Children.map([e],e=>{if(e.type===t.Fragment||e.props?.i18nIsDynamicList!==void 0){let r={key:n};return e&&e.props&&Object.keys(e.props).forEach(t=>{"children"!==t&&"i18nIsDynamicList"!==t&&(r[t]=e.props[t])}),(0,t.createElement)(e.type,r,a?null:i)}let r={key:n};return e&&e.props&&Object.keys(e.props).forEach(t=>{"ref"!==t&&"children"!==t&&(r[t]=e.props[t])}),(0,t.cloneElement)(e,r,a?null:i)}))},k=(e,r,s)=>{let u=b(e),p=b(r),g={};return p.reduce((e,r,p)=>{let _=r.children?.[0]?.content&&n.services.interpolator.interpolate(r.children[0].content,v,n.language);if("tag"===r.type){var y,N;let m,$=u[parseInt(r.name,10)];!$&&i&&($=i[r.name]),1===s.length&&!$&&(0,c.C)(s[0])&&($=s[0][r.name]),$||($={});let C={...r.attrs};l&&Object.keys(C).forEach(e=>{let t=C[e];(0,c.Z)(t)&&(C[e]=(0,f.f)(t))});let O=0!==Object.keys(C).length?(y={props:C},(m={...N=$}).props={...N.props,...y.props},m):$,P=(0,t.isValidElement)(O),E=P&&h(r,!0)&&!r.voidElement,I=d&&(0,c.C)(O)&&O.dummy&&!P,A=(0,c.C)(i)&&Object.hasOwnProperty.call(i,r.name);if((0,c.Z)(O)){let t=n.services.interpolator.interpolate(O,v,n.language);e.push(t)}else if(h(O)||E){let t=j(O,r,s);w(O,t,e,p)}else if(I)w(O,k(u,r.children,s),e,p);else if(Number.isNaN(parseFloat(r.name)))if(A){let t=j(O,r,s);w(O,t,e,p,r.voidElement)}else if(a.transSupportBasicHtmlNodes&&o.indexOf(r.name)>-1)if(r.voidElement)e.push((0,t.createElement)(r.name,{key:`${r.name}-${p}`}));else{let i,n=g[r.name]||0;g[r.name]=n+1;let a=0;for(let e=0;e<u.length;e+=1){let s=u[e];if((0,t.isValidElement)(s)&&s.type===r.name){if(a===n){i=s;break}a+=1}}let l=k(i?b(x(i)):u,r.children,s);e.push((0,t.createElement)(r.name,{key:`${r.name}-${p}`},l))}else if(r.voidElement)e.push(`<${r.name} />`);else{let t=k(u,r.children,s);e.push(`<${r.name}>${t}</${r.name}>`)}else if((0,c.C)(O)&&!P){let t=r.children[0]?_:null;t&&e.push(t)}else w(O,_,e,p,1!==r.children.length||!_)}else if("text"===r.type){let i=a.transWrapTextNodes,s="function"==typeof a.unescape?a.unescape:(0,m.n)().unescape,o=l?s(n.services.interpolator.interpolate(r.content,v,n.language)):n.services.interpolator.interpolate(r.content,v,n.language);i?e.push((0,t.createElement)(i,{key:`${r.name}-${p}`},o)):e.push(o)}return e},[])};return x(k([{dummy:!0,children:e||[]}],y,b(e||[]))[0])})(B,q,U,$,O,Z,A),W=n??O.defaultTransParent;return W?(0,t.createElement)(W,N,Y):Y}({children:e,count:r,parent:n,i18nKey:a,context:s,tOptions:l,values:o,defaults:d,components:u,ns:j||(E?.ns?void 0:O||P?.options?.defaultNS),i18n:P,t:k,shouldUnescape:N,...$})}],345771)},322831,529894,e=>{var t=e.i(271645),i=e.i(802239),r=e.i(645141),n=e.i(596043),a=e.i(305706),s=e.i(806593);let l=(e,t,i,r)=>{let n=[i,{code:t,...r||{}}];if(e?.services?.logger?.forward)return e.services.logger.forward(n,"warn","react-i18next::",!0);g(n[0])&&(n[0]=`react-i18next:: ${n[0]}`),e?.services?.logger?.warn?e.services.logger.warn(...n):console?.warn&&console.warn(...n)},o={},d=(e,t,i,r)=>{g(i)&&o[i]||(g(i)&&(o[i]=new Date),l(e,t,i,r))},u=(e,t)=>()=>{if(e.isInitialized)t();else{let i=()=>{setTimeout(()=>{e.off("initialized",i)},0),t()};e.on("initialized",i)}},p=(e,t,i)=>{e.loadNamespaces(t,u(e,i))},c=(e,t,i,r)=>{if(g(i)&&(i=[i]),e.options.preload&&e.options.preload.indexOf(t)>-1)return p(e,i,r);i.forEach(t=>{0>e.options.ns.indexOf(t)&&e.options.ns.push(t)}),e.loadLanguages(t,u(e,r))},m=(e,t,i={})=>t.languages&&t.languages.length?t.hasLoadedNamespace(e,{lng:i.lng,precheck:(t,r)=>{if(i.bindI18n&&i.bindI18n.indexOf("languageChanging")>-1&&t.services.backendConnector.backend&&t.isLanguageChangingTo&&!r(t.isLanguageChangingTo,e))return!1}}):(d(t,"NO_LANGUAGES","i18n.languages were undefined or empty",{languages:t.languages}),!0),g=e=>"string"==typeof e,f=e=>"object"==typeof e&&null!==e;e.s(["T",0,m,"C",0,f,"Z",0,g,"_",0,c,"D",0,p,"S",0,l,"R",0,d],529894);let h={t:(e,t)=>{if(g(t))return t;if(f(t)&&g(t.defaultValue))return t.defaultValue;if("function"==typeof e)return"";if(Array.isArray(e)){let t=e[e.length-1];return"function"==typeof t?"":t}return e},ready:!1},x=()=>()=>{};e.s(["f",0,(e,l={})=>{let o,{i18n:u}=l,{i18n:f,defaultNS:b}=(0,t.useContext)(s.k)||{},_=u||f||(0,n.e)();_&&!_.reportNamespaces&&(_.reportNamespaces=new s.v),_||d(_,"NO_I18NEXT_INSTANCE","useTranslation: You will need to pass in an i18next instance by using initReactI18next or by passing it via props or context. In monorepo setups, make sure there is only one instance of react-i18next.");let y=(0,t.useMemo)(()=>({...(0,a.n)(),..._?.options?.react,...l}),[_,l]),{useSuspense:v,keyPrefix:j,bindI18n:w,bindI18nStore:k}=y,N=e||b||_?.options?.defaultNS,$=g(N)?[N]:N||["translation"],C=JSON.stringify($),O=(0,t.useMemo)(()=>$,[C]);_?.reportNamespaces?.addUsedNamespaces?.(O);let P=(0,t.useRef)(0),E=(0,t.useCallback)(e=>{if(!_)return x;let t=()=>{P.current+=1,e()};return w&&_.on(w,t),k&&_.store.on(k,t),()=>{w&&w.split(" ").forEach(e=>_.off(e,t)),k&&k.split(" ").forEach(e=>_.store.off(e,t))}},[_,w,k]),I=(0,t.useRef)(),A=(0,t.useCallback)(()=>{if(!_)return h;let e=!!(_.isInitialized||_.initializedStoreOnce)&&O.every(e=>m(e,_,y)),t=l.lng||_.language,i=P.current,n="function"==typeof j?(0,r.e)(j,{..._.options,ns:O}):j,a=I.current;if(a&&a.i18n===_&&a.ready===e&&a.lng===t&&a.namespaces===O&&a.keyPrefix===n&&a.revision===i)return a;let s={t:_.getFixedT(t,"fallback"===y.nsMode?O:O[0],j,{scopeNs:O}),i18n:_,ready:e,lng:t,namespaces:O,keyPrefix:n,revision:i};return I.current=s,s},[_,O,j,y,l.lng]),[D,S]=(0,t.useState)(0),{t:z,ready:L}=(0,i.useSyncExternalStore)(E,A,A);_&&v&&!L&&(o=new Promise(e=>{let t=()=>e();l.lng?c(_,l.lng,O,t):p(_,O,t)}),{t:z,ready:L}=A()),(0,t.useEffect)(()=>{if(_&&!L&&!v){let e=()=>S(e=>e+1);l.lng?c(_,l.lng,O,e):p(_,O,e)}},[_,l.lng,O,L,v,D]);let T=_||{},R=(0,t.useRef)(null),M=(0,t.useRef)(),F=e=>{let t=Object.getOwnPropertyDescriptors(e);t.__original&&delete t.__original;let i=Object.create(Object.getPrototypeOf(e),t);if(!Object.prototype.hasOwnProperty.call(i,"__original"))try{Object.defineProperty(i,"__original",{value:e,writable:!1,enumerable:!1,configurable:!1})}catch(e){}return i},H=(0,t.useMemo)(()=>{let e=T&&`${T.language}|${T.resolvedLanguage}|${T.languages?.join(",")}`,t=T;R.current&&R.current.__original===T?M.current!==e?(R.current=t=F(T),M.current=e):t=R.current:(R.current=t=F(T),M.current=e);let i=L||v?z:(...e)=>(d(_,"USE_T_BEFORE_READY","useTranslation: t was called before ready. When using useSuspense: false, make sure to check the ready flag before using t."),z(...e)),r=[i,t,L];return r.t=i,r.i18n=t,r.ready=L,r},[z,T,L,T.resolvedLanguage,T.language,T.languages]);if(_&&v&&!L)throw o;return H}],322831)},842354,e=>{var t=e.i(768371);e.s(["f",0,()=>t.q.useQuery("get","/.well-known/litellm-ui-config").data?.claude_code_marketplace_auth_required!==!1])},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let i,{apiKeySource:r,accessToken:n,apiKey:a,inputMessage:s,chatHistory:l,selectedTags:o,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:p,selectedVoice:c,endpointType:m,selectedModel:g,selectedSdk:f,proxySettings:h,customHeaders:x}=e,b="session"===r?n:a,_=window.location.origin,y=h?.LITELLM_UI_API_DOC_BASE_URL;y&&y.trim()?_=y:h?.PROXY_BASE_URL&&(_=h.PROXY_BASE_URL);let v=s||"Your prompt here",j=v.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=l.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),k={};o.length>0&&(k.tags=o),d.length>0&&(k.vector_stores=d),u.length>0&&(k.guardrails=u),p.length>0&&(k.policies=p);let N=g||"your-model-name",$=x&&Object.keys(x).length>0?`,
	default_headers=${JSON.stringify(x,null,2).replace(/\n/g,"\n	")}`:"",C="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"${$}
)`:`import openai

client = openai.OpenAI(
	api_key="${b||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"${$}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=w.length>0?w:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${N}",
    messages=${JSON.stringify(r,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.Q.RESPONSES:{let e=Object.keys(k).length>0,t="";if(e){let e=JSON.stringify({metadata:k},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let r=w.length>0?w:[{role:"user",content:v}];i=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${N}",
    input=${JSON.stringify(r,null,4)}${t}
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
#                 {"type": "input_text", "text": "${j}"},
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
`;break}case t.Q.IMAGE:i="azure"===f?`
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
	prompt="${s}",
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
	model="${N}",
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
`;break;case t.Q.IMAGE_EDITS:i="azure"===f?`
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
	model="${N}",
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
prompt = "${j}"

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
`;break;case t.Q.EMBEDDINGS:i=`
response = client.embeddings.create(
	input="${s||"Your string here"}",
	model="${N}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:i=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${N}",
	file=audio_file${s?`,
	prompt="${s.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:i=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${N}",
	input="${s||"Your text to convert to speech here"}",
	voice="${c}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${N}",
#     input="${s||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:i="\n# Code generation for this endpoint is not implemented yet."}return`${C}
${i}`}])},209261,e=>{var t=e.i(201787);let i=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,r=e=>e.trim().replace(/\/+$/,""),n=/\.(md|markdown|txt|json|ya?ml|toml)$/i,a=/\.zip$/i,s=/^[0-9a-fA-F]{64}$/,l=/^\d{1,3}(\.\d{1,3}){3}$/,o=/^[A-Za-z0-9-]+$/,d=/^[A-Za-z0-9._-]+$/,u=/^https?:\/\//i,p="ssh://",c=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,m=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{try{return new URL(e)}catch{return null}},f=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!l.test(e.hostname),h=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},x=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),b=(e,n,a,s)=>{let l=r(s??"");return""!==l?i.test(l)?{parsed:{source:"git-subdir",url:n,path:l},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:n,path:l}),suggestedName:x(h(l))}:null:{parsed:{source:"url",url:n},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:n}),suggestedName:x(a)}},_="<your-litellm-key>";e.s(["i",0,_,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${_}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&u.test(e.url)?e.url:null,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||s.test(e.trim()),"P",0,e=>{let t=r(e);return""!==t&&i.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,s)=>{let l=((e,t)=>{let i=e.trim(),r=c.exec(i),n=r?`${p}${r[1]}@${r[2]}/${r[3]}`:i;if(!n.toLowerCase().startsWith(p))return null;let a=g(n);if(!a||""===a.username||""!==a.password||!f(a))return null;let s=n.indexOf("/",p.length);return -1===s||a.pathname!==n.slice(s)||m(a).length<2?null:b("SSH",i,h(a.pathname).replace(/\.git$/i,""),t)})(e,s);if(l)return l;let u=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let i=g(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return i&&"https:"===i.protocol&&""===i.username&&""===i.password&&f(i)?i:null})(e);if(!u)return null;if(a.test(u.pathname))return{parsed:{source:"archive",url:u.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${u.host}${u.pathname}`}),suggestedName:x(h(u.pathname).replace(a,""))};if("github.com"===u.hostname.replace(/^www\./,""))return((e,a)=>{let s=m(e);if(s.length<2)return null;let l=s[0],u=s[1].replace(/\.git$/,"");if(!o.test(l)||!d.test(u))return null;let p=`${l}/${u}`,c=`https://github.com/${p}`,g={parsed:{source:"github",repo:p},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:p}),suggestedName:x(u)};if(s.length>=4&&("tree"===s[2]||"blob"===s[2])){let e=s.slice(4),a=h(e.join("/")),l=n.test(a)?e.slice(0,-1):e;if(0===l.length)return g;let o=r(l.join("/"));return i.test(o)?{parsed:{source:"git-subdir",url:c,path:o},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:p,path:o}),suggestedName:x(h(o))}:null}if(2!==s.length)return null;let f=r(a??"");return""!==f?i.test(f)?{parsed:{source:"git-subdir",url:c,path:f},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:p,path:f}),suggestedName:x(h(f))}:null:g})(u,s);if(m(u).length<2)return null;let _=h(u.pathname).replace(/\.git$/,"");return b("Git",`${u.protocol}//${u.host}${u.pathname.replace(/\/+$/,"")}`,_,s)},"e",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},652272,e=>{var t=e.i(843476),i=e.i(271645),r=e.i(871689),n=e.i(643531),a=e.i(174886),s=e.i(306228),l=e.i(345771),o=e.i(322831),d=e.i(842354),u=e.i(196631),p=e.i(209261);let c=({source:e})=>{let{t:i}=(0,o.f)(),r=(0,p.E)(e),n=r&&"git-subdir"===e.source&&e.path?`${r}/tree/main/${e.path}`:r;return n?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:i("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsxs)("a",{href:n,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[n.replace("https://",""),(0,t.jsx)(s.f,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:i("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:(0,p._)(e)})]}):null};e.s(["f",0,({skill:e,onBack:s})=>{let{t:m}=(0,o.f)(),[g,f]=(0,i.useState)("overview"),[h,x]=(0,i.useState)(null),b=(0,d.f)(),_=(e,t)=>{navigator.clipboard.writeText(e),x(t),setTimeout(()=>x(null),2e3)},y=(0,p.a)(e),v=(0,p.U)(window.location.origin,b),j=[...e.category?[{property:m("claudeCodePlugins.skillDetail.properties.category"),value:e.category}]:[],...e.domain?[{property:m("claudeCodePlugins.skillDetail.properties.domain"),value:e.domain}]:[],...e.namespace?[{property:m("claudeCodePlugins.skillDetail.properties.namespace"),value:e.namespace}]:[],...e.version?[{property:m("claudeCodePlugins.skillDetail.properties.version"),value:e.version}]:[],...e.author?.name?[{property:m("claudeCodePlugins.skillDetail.properties.author"),value:e.author.name}]:[],...e.created_at?[{property:m("claudeCodePlugins.skillDetail.properties.added"),value:new Date(e.created_at).toLocaleDateString()}]:[]],w=[{key:"overview",label:m("claudeCodePlugins.skillDetail.tabs.overview")},{key:"usage",label:m("claudeCodePlugins.skillDetail.tabs.usage")}];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:s,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(r.f,{className:"size-3"}),(0,t.jsx)("span",{children:m("claudeCodePlugins.skillDetail.back")})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:w.map(e=>(0,t.jsx)("div",{onClick:()=>f(e.key),className:(0,u.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",g===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===g&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.overview.title")}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.subtitle")}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.propertyHeader")}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:j.map((e,i)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.status")}),(0,t.jsx)("span",{className:(0,u.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:m(e.enabled?"claudeCodePlugins.skillDetail.overview.public":"claudeCodePlugins.skillDetail.overview.draft")})]}),(0,t.jsx)(c,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.tags")}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.skillId")}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.usage.title")}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.usage.intro")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:m("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>_(y,"install"),className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===h?"text-success":"text-info"),children:["install"===h?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(a.f,{className:"size-3"}),m("install"===h?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:y})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.usage.staleCatalog",{name:e.name})}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:[m("claudeCodePlugins.skillDetail.usage.notConfigured")," ",(0,t.jsx)("span",{onClick:()=>f("setup"),className:"cursor-pointer text-info",children:m("claudeCodePlugins.skillDetail.usage.seeSetup")})]})]}),"setup"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.setup.title")}),b?(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.setup.removeFirst")}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace remove litellm"})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(l.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.requiresKey",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"}),placeholder:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:p.i})}})})]}):(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.setup.runCommand")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:m("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;_(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===h?"text-success":"text-info"),children:["marketplace-cmd"===h?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(a.f,{className:"size-3"}),m("marketplace-cmd"===h?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(l.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.orSettings",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})}})})]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>_(v,"settings"),className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===h?"text-success":"text-info"),children:["settings"===h?(0,t.jsx)(n.f,{className:"size-3"}):(0,t.jsx)(a.f,{className:"size-3"}),m("settings"===h?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:v})]})]})]})}])},755146,e=>{var t=e.i(843476),i=e.i(65426),r=e.i(606078),n=e.i(984756),a=e.i(256737),s=e.i(383110),l=e.i(205133),o=e.i(408085),d=e.i(777222);e.i(794494),e.i(806727),e.i(218141),e.i(873969);var u=e.i(569389),p=e.i(227690),c=e.i(189793),m=e.i(539277),g=e.i(196631);e.i(233565);var f=e.i(678784);e.s(["G",0,function({...e}){return(0,t.jsx)(i.f.Root,{"data-slot":"dropdown-menu",...e})},"L",0,function({align:e="start",alignOffset:i=0,side:n="bottom",sideOffset:l=4,className:o,...d}){return(0,t.jsx)(r.f.Portal,{children:(0,t.jsx)(a.f.Positioner,{className:"isolate z-popup outline-none",align:e,alignOffset:i,side:n,sideOffset:l,children:(0,t.jsx)(s.f.Popup,{"data-slot":"dropdown-menu-content",className:(0,g.c)("z-popup max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-md bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 duration-100 outline-none data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:overflow-hidden data-closed:fade-out-0 data-closed:zoom-out-95",o),...d})})})},"P",0,function({...e}){return(0,t.jsx)(l.f.Group,{"data-slot":"dropdown-menu-group",...e})},"r",0,function({className:e,inset:i,variant:r="default",...n}){return(0,t.jsx)(d.f.Item,{"data-slot":"dropdown-menu-item","data-inset":i,"data-variant":r,className:(0,g.c)("group/dropdown-menu-item relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground not-data-[variant=destructive]:focus:**:text-accent-foreground data-inset:pl-8 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[variant=destructive]:*:[svg]:text-destructive",e),...n})},"B",0,function({className:e,inset:i,...r}){return(0,t.jsx)(o.f.GroupLabel,{"data-slot":"dropdown-menu-label","data-inset":i,className:(0,g.c)("px-2 py-1.5 text-xs font-medium text-muted-foreground data-inset:pl-8",e),...r})},"O",0,function({...e}){return(0,t.jsx)(u.f.RadioGroup,{"data-slot":"dropdown-menu-radio-group",...e})},"T",0,function({className:e,children:i,inset:r,...n}){return(0,t.jsxs)(p.f.RadioItem,{"data-slot":"dropdown-menu-radio-item","data-inset":r,className:(0,g.c)("relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground focus:**:text-accent-foreground data-inset:pl-8 data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",e),...n,children:[(0,t.jsx)("span",{className:"pointer-events-none absolute right-2 flex items-center justify-center","data-slot":"dropdown-menu-radio-item-indicator",children:(0,t.jsx)(c.f.RadioItemIndicator,{children:(0,t.jsx)(f.f,{})})}),i]})},"U",0,function({className:e,...i}){return(0,t.jsx)(m.f.Separator,{"data-slot":"dropdown-menu-separator",className:(0,g.c)("-mx-1 my-1 h-px bg-border",e),...i})},"X",0,function({...e}){return(0,t.jsx)(n.f.Trigger,{"data-slot":"dropdown-menu-trigger",...e})}])},735450,e=>{var t=e.i(411382),i=e.i(772325),r=e.i(54105),n=e.i(240192),a=e.i(138620);let s="https://docs.litellm.ai/docs/decisions",l=`${s}#supported-providers`,o=`${s}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),u=t.z.object({litellm_provider:r.z.string(),mode:r.z.string().nullish(),supported_endpoints:i.z.array(r.z.string()).nullish()}),p="decisions",c=["evaluation",p],m=e=>null!=e&&c.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),f=e=>{let t=n.z.record(r.z.string(),a.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let i=u.safeParse(t);return i.success?[[e,i.data]]:[]}):[]},h={decisions:0,"self-hosted":1,"needs-upgrade":2},x=t.z.object({decision_model_providers:i.z.array(r.z.string())}),b=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),_=(e,t,i)=>({row:e,routeOrder:h[e.route],rankInProvider:i.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,s,"d",0,p,"Z",0,l,"L",0,o,"B",0,"playground?tab=system-one","F",0,function(e){let t=f(e),i=t.filter(([,e])=>g(e)),r=new Map([...new Set(i.map(([,e])=>e.litellm_provider))].map(e=>{let i=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:i.every(([,e])=>g(e))}]}));return{models:new Set(i.map(([e])=>e)),providers:r}},"W",0,(e,t)=>f(e).filter(([,e])=>g(e)).map(([e,i])=>({model:e,provider:i.litellm_provider,route:t?.has(i.litellm_provider)?"decisions":(i.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(b).map(_).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?p:e??void 0,"q",0,e=>e.flatMap(e=>e===p?c:[e]),"m",0,g,"_",0,m,"r",0,(e,t,i)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||i.some(t=>e.models.has(t)),"P",0,e=>{let t=x.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function i(e,i){let r=t(e);if(""===r)return!0;let n=i.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!n.some(e=>e.includes(r))||r.split(/\s+/).every(e=>n.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,r){return e.filter(e=>i(t,r(e)))},"h",0,i,"e",0,function(e,i,r){let n=t(i);if(""===n)return[...e];let a=e=>{let t=r(e).toLowerCase();return 1e3*(t===n)+100*!!t.startsWith(n)+(1e3-t.length)};return[...e].sort((e,t)=>a(t)-a(e))}])}])})();