(()=>{"use strict";(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,871943,502547,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 9l-7 7-7-7"}))});e.s(["f",0,r],871943);let i=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M9 5l7 7-7 7"}))});e.s(["f",0,i],502547)},360820,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M5 15l7-7 7 7"}))});e.s(["f",0,r],360820)},434626,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["f",0,r],434626)},250980,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"}))});e.s(["f",0,r],250980)},278587,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"}))});e.s(["f",0,r],278587)},68155,e=>{var t=e.i(271645);let r=t.forwardRef(function(e,r){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:r},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"}))});e.s(["f",0,r],68155)},586448,e=>{var t=e.i(271645),r=e.i(540143),i=e.i(869230),o=e.i(915823),n=e.i(619273);function l(e,t){let r=new Set(t);return e.filter(e=>!r.has(e))}var s=class extends o.f{#e;#t;#r;#i;#o;#n;#l;#s;#a;#d=[];constructor(e,t,r){super(),this.#e=e,this.#i=r,this.#r=[],this.#o=[],this.#t=[],this.setQueries(t)}onSubscribe(){1===this.listeners.size&&this.#o.forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})}onUnsubscribe(){this.listeners.size||this.destroy()}destroy(){this.listeners=new Set,this.#o.forEach(e=>{e.destroy()})}setQueries(e,t){this.#r=e,this.#i=t,r.f.batch(()=>{let e=this.#o,t=this.#c(this.#r);t.forEach(e=>e.observer.setOptions(e.defaultedQueryOptions));let r=t.map(e=>e.observer),i=r.map(e=>e.getCurrentResult()),o=e.length!==r.length,s=r.some((t,r)=>t!==e[r]),a=o||s,d=!!a||i.some((e,t)=>{let r=this.#t[t];return!r||!(0,n.U)(e,r)});(a||d)&&(a&&(this.#d=t,this.#o=r),this.#t=i,this.hasListeners()&&(a&&(l(e,r).forEach(e=>{e.destroy()}),l(r,e).forEach(e=>{e.subscribe(t=>{this.#u(e,t)})})),this.#p()))})}getCurrentResult(){return this.#t}getQueries(){return this.#o.map(e=>e.getCurrentQuery())}getObservers(){return this.#o}getOptimisticResult(e,t){let r=this.#c(e),i=r.map(e=>e.observer.getOptimisticResult(e.defaultedQueryOptions)),o=r.map(e=>e.defaultedQueryOptions.queryHash);return[i,e=>this.#m(e??i,t,o),()=>this.#g(i,r)]}#g(e,t){return t.map((r,i)=>{let o=e[i];return r.defaultedQueryOptions.notifyOnChangeProps?o:r.observer.trackResult(o,e=>{t.forEach(t=>{t.observer.trackProp(e)})})})}#m(e,t,r){if(t){let i=this.#a,o=void 0!==r&&void 0!==i&&(i.length!==r.length||r.some((e,t)=>e!==i[t]));return(!this.#n||this.#t!==this.#s||o||t!==this.#l)&&(this.#l=t,this.#s=this.#t,void 0!==r&&(this.#a=r),this.#n=(0,n.B)(this.#n,t(e))),this.#n}return e}#h(){return this.#i?.combine!==void 0&&this.#o.some((e,t)=>e.options.suspense&&this.#t[t]?.data===void 0)}#c(e){let t=new Map;this.#o.forEach(e=>{let r=e.options.queryHash;if(!r)return;let i=t.get(r);i?i.push(e):t.set(r,[e])});let r=[];return e.forEach(e=>{let o=this.#e.defaultQueryOptions(e),n=t.get(o.queryHash)?.shift()??new i.f(this.#e,o);r.push({defaultedQueryOptions:o,observer:n})}),r}#u(e,t){let r=this.#o.indexOf(e);if(-1!==r){var i;let e;this.#t=(i=this.#t,(e=i.slice(0))[r]=t,e),this.#p()}}#p(){if(this.hasListeners()){let e=this.#g(this.#t,this.#d),t=this.#h(),i=this.#n,o=t?i:this.#m(e,this.#i?.combine);(t||i!==o)&&r.f.batch(()=>{this.listeners.forEach(e=>{e(this.#t)})})}}},a=e.i(912598),d=e.i(381384),u=e.i(673664),c=e.i(427001),p=e.i(254440);e.s(["f",0,function({queries:e,...o},l){let m=(0,a.d)(l),g=(0,d.f)(),h=(0,u.p)(),f=t.useMemo(()=>e.map(e=>{let t=m.defaultQueryOptions(e);return t._optimisticResults=g?"isRestoring":"optimistic",t}),[e,m,g]);f.forEach(e=>{(0,p.p)(e);let t=m.getQueryCache().get(e.queryHash);(0,c.l)(e,h,t)}),(0,c.E)(h);let[b]=t.useState(()=>new s(m,f,o)),[v,x,k]=b.getOptimisticResult(f,o.combine),_=!g&&!1!==o.subscribed;t.useSyncExternalStore(t.useCallback(e=>_?b.subscribe(r.f.batchCalls(e)):n.z,[b,_]),()=>b.getCurrentResult(),()=>b.getCurrentResult()),t.useEffect(()=>{b.setQueries(f,o)},[f,o,b]);let y=v.some((e,t)=>(0,p.Q)(f[t],e))?v.flatMap((e,t)=>{let r=f[t];if(r&&(0,p.Q)(r,e)){let e=new i.f(m,r);return(0,p.X)(r,e,h)}return[]}):[];if(y.length>0)throw Promise.all(y);let w=v.find((e,t)=>{let r=f[t];return r&&(0,c.P)({result:e,errorResetBoundary:h,throwOnError:r.throwOnError,query:m.getQueryCache().get(r.queryHash),suspense:r.suspense})});if(w?.error)throw w.error;return x(k())}],586448)},180127,e=>{let t=(0,e.i(475254).f)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["f",0,t])},871689,e=>{var t=e.i(180127);e.s(["f",()=>t.f])},541071,e=>{var t=e.i(373488);e.s(["f",()=>t.f])},332102,e=>{let t=(0,e.i(475254).f)("inbox",[["polyline",{points:"22 12 16 12 14 15 10 15 8 12 2 12",key:"o97t9d"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}]]);e.s(["f",0,t],332102)},306228,e=>{let t=(0,e.i(475254).f)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["f",0,t],306228)},283086,576144,e=>{let t=(0,e.i(475254).f)("sparkles",[["path",{d:"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",key:"4pj2yx"}],["path",{d:"M20 3v4",key:"1olli1"}],["path",{d:"M22 5h-4",key:"1gvqau"}],["path",{d:"M4 17v2",key:"vumght"}],["path",{d:"M5 18H3",key:"zchphs"}]]);e.s(["f",0,t],576144),e.s(["f",0,t],283086)},345771,e=>{var t=e.i(271645),r=e.i(645141);let i={area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0,"!doctype":!0,"!DOCTYPE":!0},o=/\s([^'"/\s><]+?)[\s/>]|([^\s=]+)=\s?("[^"]*"|'[^']*')/g;function n(e){let t={type:"tag",name:"",voidElement:!1,attrs:{},children:[]},r=e.match(/<\/?([^\s]+?)[/\s>]/);if(r&&(t.name=r[1],(i[r[1]]||"/"===e.charAt(e.length-2))&&(t.voidElement=!0),t.name.startsWith("!--"))){let t=e.indexOf("-->");return{type:"comment",comment:-1!==t?e.slice(4,t):""}}let n=new RegExp(o),l=null;for(;null!==(l=n.exec(e));)if(l[0].trim())if(l[1]){let e=l[1].trim(),r=[e,null],i=e.indexOf("=");i>-1&&(r=[e.slice(0,i),e.slice(i+1)]),t.attrs[r[0]]=r[1],n.lastIndex--}else l[2]&&(t.attrs[l[2]]=l[3].trim().substring(1,l[3].length-1));return t}let l=/<!--[\s\S]*?-->|<[a-zA-Z0-9\-!/](?:"[^"]*"|'[^']*'|[^'">])*>/g,s=/<\/?([^\s]+?)[/\s>]/,a=/^\s*$/,d=/^(script|style)$/i,u=Object.create(null);var c=function(e,t){let r,i,o,c=t&&t.components||u,p=t&&t.allowedTags,m=!1;if(p){let t,r="function"==typeof p?p:function(e){return p.indexOf(e)>-1},i="",o=0;for(l.lastIndex=0;t=l.exec(e);){let n=t[0];i+=e.slice(o,t.index);let a=n.match(s);n.startsWith("<!--")||a&&r(a[1])?(i+=n,o=t.index+n.length):(m=!0,i+="\0",l.lastIndex=o=t.index+1)}e=i+e.slice(o)}let g=[],h=[],f=-1,b=!1,v=0;if(0!==e.indexOf("<")){let t=e.indexOf("<");g.push({type:"text",content:-1===t?e:e.substring(0,t)})}let x=[];for(;o=l.exec(e);)x.push(o);return x.forEach(function(e,t){let r=e[0];if(!r||r.startsWith("<!--"))return;let i=0,o=0,n=-1,l=null;for(let e=0;e<r.length;e++){let t=r.charAt(e);l?t===l&&(l=null):'"'===t||"'"===t?l=t:"<"===t?2==++i&&(n=e):">"===t&&o++}let s=n>-1&&/[a-zA-Z0-9\-!/]/.test(r.charAt(n+1));if(i>o&&s){let e=r.substring(0,n),i=r.substring(e.length);x[t][0]=i,x[t].index+=e.length}}),x.forEach(function(t,o){let l,s,u=t[0];if(!u)return;let p=t.index;if(p<v)return;if(b)if(u!=="</"+r.name+">")return;else b=!1;let m="/"!==u.charAt(1),k=u.startsWith("<!--"),_=p+u.length,y=e.charAt(_),w=x[o+1];if("<"===y&&w){let t=e.substring(_,w.index);l=t.split("<").length>t.split(">").length}if(k){let t=n(u);if(f<0)return g.push(t),g;(s=h[f]).children.push(t);let r=e.slice(_,w?w.index:void 0);return r.length>0&&s.children.push({type:"text",content:r}),g}if(m){f++,"tag"===(r=n(u)).type&&c[r.name]&&(r.type="component",b=!0);let t=!1;if(!b&&!r.voidElement&&d.test(r.name)){t=!0,i||(i=e.toLowerCase());let o=i.indexOf("</"+r.name.toLowerCase()+">",_),n=-1===o?e.length:o,l=e.slice(_,n);l&&r.children.push({type:"text",content:l}),v=n}r.voidElement||b||t||!y||"<"===y||r.children.push({type:"text",content:e.slice(_,w?w.index:void 0)}),0===f&&g.push(r),(s=h[f-1])&&s.children.push(r),h[f]=r}if((!m||r.voidElement)&&(f>-1&&(r.voidElement||r.name===u.slice(2,-1))&&(r=-1==--f?g:h[f]),!b&&("<"!==y||l)&&y)){s=-1===f?g:h[f].children;let t=w?w.index:-1,r=e.slice(_,-1===t?void 0:t);a.test(r)&&(r=" "),(t>-1&&f+s.length>=0||" "!==r)&&s.push({type:"text",content:r})}}),m&&function e(t){t.forEach(function(t){if("text"===t.type){t.content=t.content.split("\0").join("<");return}if("comment"===t.type){t.comment=t.comment.split("\0").join("<");return}for(let e in t.attrs){let r=t.attrs[e];"string"==typeof r&&r.indexOf("\0")>-1&&(t.attrs[e]=r.split("\0").join("<"))}t.children.length&&e(t.children)})}(g),g},p=e.i(529894),m=e.i(305706),g=e.i(596043),h=e.i(247464);let f=(e,t)=>{if(!e)return!1;let r=e.props?.children??e.children;return t?r.length>0:!!r},b=e=>{if(!e)return[];let t=e.props?.children??e.children;return e.props?.i18nIsDynamicList?v(t):t},v=e=>Array.isArray(e)?e:[e],x=(e,r,i,o)=>{if(!e)return"";let n="",l=v(e),s=r?.transSupportBasicHtmlNodes?r.transKeepBasicHtmlNodesFor??[]:[];return l.forEach((e,l)=>{if((0,p.Z)(e)){n+=`${e}`;return}if((0,t.isValidElement)(e)){let{props:t,type:a}=e,d=Object.keys(t).length,u=s.indexOf(a)>-1,c=t.children;if(!c&&u&&!d){n+=`<${a}/>`;return}if(!c&&(!u||d)||t.i18nIsDynamicList){n+=`<${l}></${l}>`;return}if(u&&d<=1){let e=(0,p.Z)(c)?c:x(c,r,i,o);n+=`<${a}>${e}</${a}>`;return}let m=x(c,r,i,o);n+=`<${l}>${m}</${l}>`;return}if(null===e)return void(0,p.S)(i,"TRANS_NULL_VALUE","Passed in a null value as child",{i18nKey:o});if((0,p.C)(e)){let{format:t,...r}=e,l=Object.keys(r);if(1===l.length){let e=t?`${l[0]}, ${t}`:l[0];n+=`{{${e}}}`;return}if(!l.length&&1===Object.keys(e).length&&"format"in e){n+="{{format}}";return}(0,p.S)(i,"TRANS_INVALID_OBJ","Invalid child - Object should only have keys {{ value, format }} (format is optional).",{i18nKey:o,child:e});return}(0,p.S)(i,"TRANS_INVALID_VAR","Passed in a variable like {number} - pass variables for interpolation as full objects like {{number}}.",{i18nKey:o,child:e})}),n},k=(e,r,i)=>{let o=e.key||r,n=(0,t.cloneElement)(e,{key:o});return!n.props||!n.props.children||0>i.indexOf(`${r}/>`)&&0>i.indexOf(`${r} />`)?n:(0,t.createElement)(function(){return(0,t.createElement)(t.Fragment,null,n)},{key:o})};var _=e.i(806593);e.s(["f",0,function({children:e,count:i,parent:o,i18nKey:n,context:l,tOptions:s={},values:a,defaults:d,components:u,ns:y,i18n:w,t:j,shouldUnescape:C,...N}){let{i18n:E,defaultNS:O}=(0,t.useContext)(_.k)||{},A=w||E||(0,g.e)(),S=j||A?.t.bind(A);return function({children:e,count:i,parent:o,i18nKey:n,context:l,tOptions:s={},values:a,defaults:d,components:u,ns:_,i18n:y,t:w,shouldUnescape:j,...C}){let N=y||(0,g.e)();if(!N)return(0,p.R)(N,"NO_I18NEXT_INSTANCE","Trans: You need to pass in an i18next instance using initReactI18next or by passing it via props or context. In monorepo setups, make sure there is only one instance of react-i18next.",{i18nKey:n}),e;let E=w||N.t.bind(N)||(e=>e),O={...(0,m.n)(),...N.options?.react},A=_||(E.ns?void 0:N.options?.defaultNS||"translation");(0,p.Z)(A)&&(A=[A]);let{transDefaultProps:S}=O,$=S?.tOptions?{...S.tOptions,...s}:s,T=j??S?.shouldUnescape,P=S?.values?{...S.values,...a}:a,I=S?.components?{...S.components,...u}:u,R=x(e,O,N,n),z=Array.isArray(n)?n[n.length-1]:n,L=d||$?.defaultValue||R||O.transEmptyNodeValue||("function"==typeof z?(0,r.e)(z):z),{hashTransKey:D}=O,H=n||(D?D(R||L):R||L);a=N.options?.interpolation?.defaultVariables?{...N.options.interpolation.defaultVariables,...P}:P;let M=(e=>{let r={};if(!e)return r;let i=e=>{v(e).forEach(e=>{(0,p.Z)(e)||(f(e)?i(b(e)):(0,p.C)(e)&&!(0,t.isValidElement)(e)&&Object.assign(r,e))})};return i(e),r})(e);M&&"number"==typeof M.count&&void 0===i&&(i=M.count),void 0===i&&(i=$?.count);let U=!a&&(void 0===i||N.options?.interpolation?.alwaysFormat)&&e?{interpolation:{...$.interpolation,prefix:"#$?",suffix:"?$#"}}:$.interpolation,B={...$,context:l||$.context,count:i,...a,...U,defaultValue:L,ns:A},q=H?E(H,B):L;q===H&&L&&(q=L);let F=((e,t,r,i)=>{if(!e)return null;if(Array.isArray(e))return e.map((e,r)=>k(e,r,t));if((0,p.C)(e)){let r;return r={},Object.keys(e).forEach(i=>{Object.assign(r,{[i]:k(e[i],i,t)})}),r}return(0,p.R)(r,"TRANS_INVALID_COMPONENTS",'<Trans /> "components" prop expects an object or array',{i18nKey:i}),null})(I,q,N,n),V=F||e,Q=null;!(!(0,p.C)(F)||Array.isArray(F))&&Object.keys(F).reduce((e,t)=>e&&Number.isNaN(Number.parseFloat(t)),!0)?(Q=F,V=e):F&&!Array.isArray(F)&&(Q=F,V=void 0);let W=((e,r,i,o,n,l,s)=>{if(""===i)return[];let a=n.transKeepBasicHtmlNodesFor||[],d=i&&new RegExp(a.map(e=>`<${e}`).join("|")).test(i);if(!e&&!r&&!d&&!s)return[i];let u=r??{},g=e=>{v(e).forEach(e=>{(0,p.Z)(e)||(f(e)?g(b(e)):(0,p.C)(e)&&!(0,t.isValidElement)(e)&&Object.assign(u,e))})};g(e);let x=Object.keys(u),k=c(`<0>${i}</0>`,{allowedTags:e=>/^\d+$/.test(e)||a.indexOf(e)>-1||x.indexOf(e)>-1}),_={...u,...l},y=(e,r,i)=>{let o=b(e),n=j(o,r.children,i);return v(o).every(t.isValidElement)&&0===n.length||e.props?.i18nIsDynamicList?o:n},w=(e,r,i,o,n)=>{e.dummy?(e.children=r,i.push((0,t.cloneElement)(e,{key:o},n?void 0:r))):i.push(...t.Children.map([e],e=>{if(e.type===t.Fragment||e.props?.i18nIsDynamicList!==void 0){let i={key:o};return e&&e.props&&Object.keys(e.props).forEach(t=>{"children"!==t&&"i18nIsDynamicList"!==t&&(i[t]=e.props[t])}),(0,t.createElement)(e.type,i,n?null:r)}let i={key:o};return e&&e.props&&Object.keys(e.props).forEach(t=>{"ref"!==t&&"children"!==t&&(i[t]=e.props[t])}),(0,t.cloneElement)(e,i,n?null:r)}))},j=(e,i,l)=>{let u=v(e),c=v(i),g={};return c.reduce((e,i,c)=>{let x=i.children?.[0]?.content&&o.services.interpolator.interpolate(i.children[0].content,_,o.language);if("tag"===i.type){var k,C;let m,N=u[parseInt(i.name,10)];!N&&r&&(N=r[i.name]),1===l.length&&!N&&(0,p.C)(l[0])&&(N=l[0][i.name]),N||(N={});let E={...i.attrs};s&&Object.keys(E).forEach(e=>{let t=E[e];(0,p.Z)(t)&&(E[e]=(0,h.f)(t))});let O=0!==Object.keys(E).length?(k={props:E},(m={...C=N}).props={...C.props,...k.props},m):N,A=(0,t.isValidElement)(O),S=A&&f(i,!0)&&!i.voidElement,$=d&&(0,p.C)(O)&&O.dummy&&!A,T=(0,p.C)(r)&&Object.hasOwnProperty.call(r,i.name);if((0,p.Z)(O)){let t=o.services.interpolator.interpolate(O,_,o.language);e.push(t)}else if(f(O)||S){let t=y(O,i,l);w(O,t,e,c)}else if($)w(O,j(u,i.children,l),e,c);else if(Number.isNaN(parseFloat(i.name)))if(T){let t=y(O,i,l);w(O,t,e,c,i.voidElement)}else if(n.transSupportBasicHtmlNodes&&a.indexOf(i.name)>-1)if(i.voidElement)e.push((0,t.createElement)(i.name,{key:`${i.name}-${c}`}));else{let r,o=g[i.name]||0;g[i.name]=o+1;let n=0;for(let e=0;e<u.length;e+=1){let l=u[e];if((0,t.isValidElement)(l)&&l.type===i.name){if(n===o){r=l;break}n+=1}}let s=j(r?v(b(r)):u,i.children,l);e.push((0,t.createElement)(i.name,{key:`${i.name}-${c}`},s))}else if(i.voidElement)e.push(`<${i.name} />`);else{let t=j(u,i.children,l);e.push(`<${i.name}>${t}</${i.name}>`)}else if((0,p.C)(O)&&!A){let t=i.children[0]?x:null;t&&e.push(t)}else w(O,x,e,c,1!==i.children.length||!x)}else if("text"===i.type){let r=n.transWrapTextNodes,l="function"==typeof n.unescape?n.unescape:(0,m.n)().unescape,a=s?l(o.services.interpolator.interpolate(i.content,_,o.language)):o.services.interpolator.interpolate(i.content,_,o.language);r?e.push((0,t.createElement)(r,{key:`${i.name}-${c}`},a)):e.push(a)}return e},[])};return b(j([{dummy:!0,children:e||[]}],k,v(e||[]))[0])})(V,Q,q,N,O,B,T),K=o??O.defaultTransParent;return K?(0,t.createElement)(K,C,W):W}({children:e,count:i,parent:o,i18nKey:n,context:l,tOptions:s,values:a,defaults:d,components:u,ns:y||(S?.ns?void 0:O||A?.options?.defaultNS),i18n:A,t:j,shouldUnescape:C,...N})}],345771)},455037,e=>{var t=e.i(494144);e.s(["f",()=>t.f])},842354,e=>{var t=e.i(768371);e.s(["f",0,()=>t.q.useQuery("get","/.well-known/litellm-ui-config").data?.claude_code_marketplace_auth_required!==!1])},198458,e=>{var t=e.i(655063),r=e.i(266027),i=e.i(271645),o=e.i(741466);e.s(["f",0,function(e){let{queryKey:n,fetchPage:l,serializeFilters:s,defaultSorting:a,defaultPageSize:d,enabled:u}=e,[c,p]=(0,i.useState)(a),[m,g]=(0,i.useState)({pageIndex:0,pageSize:d}),[h,f]=(0,i.useState)([]),[b,v]=(0,i.useState)(""),[x]=(0,t.f)(b,{wait:o.f}),k=(0,i.useMemo)(()=>{let e=c.map(e=>e.desc?`-${e.id}`:e.id).join(","),t=x.trim();return{page:m.pageIndex+1,page_size:m.pageSize,...""===e?{}:{sort:e},...""===t?{}:{q:t},...s(h)}},[c,m.pageIndex,m.pageSize,x,h,s]),_={queryKey:[...n,k],queryFn:({signal:e})=>l(k,e),enabled:u,placeholderData:e=>e},{data:y,isLoading:w,isPlaceholderData:j,isFetching:C,error:N,refetch:E}=(0,r.f)(_),O=(0,i.useCallback)(()=>g(e=>({...e,pageIndex:0})),[]),A=(0,i.useCallback)(e=>{p(e),O()},[O]),S=(0,i.useCallback)(e=>{f(e),O()},[O]),$=(0,i.useCallback)(e=>{v(e),O()},[O]),T=(0,i.useCallback)(()=>{E()},[E]);return{rows:(0,i.useMemo)(()=>y?.data??[],[y]),rowCount:y?.meta.total_count??0,isLoading:w||j,isFetching:C,error:N,refetch:T,sorting:c,onSortingChange:A,pagination:m,onPaginationChange:g,columnFilters:h,onColumnFiltersChange:S,searchValue:b,onSearchChange:$}}])},368670,e=>{var t=e.i(602869),r=e.i(266027);let i=(0,e.i(243652).f)("modelCostMap");e.s(["Q",0,i,"X",0,(e=!0,o=!1)=>(0,r.f)({enabled:e,queryKey:i.list(o?{filters:{catalog_only:"true"}}:{}),queryFn:async()=>await (0,t.T)(o),staleTime:6e4,gcTime:6e4})])},803099,e=>{var t=e.i(602869),r=e.i(266027);let i=(0,e.i(243652).f)("providerFields");e.s(["f",0,()=>(0,r.f)({queryKey:i.list({}),queryFn:async()=>await (0,t.Bg)(),staleTime:864e5,gcTime:864e5})])},466828,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(678784);let o=(0,e.i(475254).f)("clipboard",[["rect",{width:"8",height:"4",x:"8",y:"2",rx:"1",ry:"1",key:"tgr4d6"}],["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2",key:"116196"}]]);var n=e.i(650056);let l={'code[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none"},'pre[class*="language-"]':{background:"hsl(230, 1%, 98%)",color:"hsl(230, 8%, 24%)",fontFamily:'"Fira Code", "Fira Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace',direction:"ltr",textAlign:"left",whiteSpace:"pre",wordSpacing:"normal",wordBreak:"normal",lineHeight:"1.5",MozTabSize:"2",OTabSize:"2",tabSize:"2",WebkitHyphens:"none",MozHyphens:"none",msHyphens:"none",hyphens:"none",padding:"1em",margin:"0.5em 0",overflow:"auto",borderRadius:"0.3em"},'code[class*="language-"]::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::-moz-selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"]::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'code[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},'pre[class*="language-"] *::selection':{background:"hsl(230, 1%, 90%)",color:"inherit"},':not(pre) > code[class*="language-"]':{padding:"0.2em 0.3em",borderRadius:"0.3em",whiteSpace:"normal"},comment:{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},prolog:{color:"hsl(230, 4%, 64%)"},cdata:{color:"hsl(230, 4%, 64%)"},doctype:{color:"hsl(230, 8%, 24%)"},punctuation:{color:"hsl(230, 8%, 24%)"},entity:{color:"hsl(230, 8%, 24%)",cursor:"help"},"attr-name":{color:"hsl(35, 99%, 36%)"},"class-name":{color:"hsl(35, 99%, 36%)"},boolean:{color:"hsl(35, 99%, 36%)"},constant:{color:"hsl(35, 99%, 36%)"},number:{color:"hsl(35, 99%, 36%)"},atrule:{color:"hsl(35, 99%, 36%)"},keyword:{color:"hsl(301, 63%, 40%)"},property:{color:"hsl(5, 74%, 59%)"},tag:{color:"hsl(5, 74%, 59%)"},symbol:{color:"hsl(5, 74%, 59%)"},deleted:{color:"hsl(5, 74%, 59%)"},important:{color:"hsl(5, 74%, 59%)"},selector:{color:"hsl(119, 34%, 47%)"},string:{color:"hsl(119, 34%, 47%)"},char:{color:"hsl(119, 34%, 47%)"},builtin:{color:"hsl(119, 34%, 47%)"},inserted:{color:"hsl(119, 34%, 47%)"},regex:{color:"hsl(119, 34%, 47%)"},"attr-value":{color:"hsl(119, 34%, 47%)"},"attr-value > .token.punctuation":{color:"hsl(119, 34%, 47%)"},variable:{color:"hsl(221, 87%, 60%)"},operator:{color:"hsl(221, 87%, 60%)"},function:{color:"hsl(221, 87%, 60%)"},url:{color:"hsl(198, 99%, 37%)"},"attr-value > .token.punctuation.attr-equals":{color:"hsl(230, 8%, 24%)"},"special-attr > .token.attr-value > .token.value.css":{color:"hsl(230, 8%, 24%)"},".language-css .token.selector":{color:"hsl(5, 74%, 59%)"},".language-css .token.property":{color:"hsl(230, 8%, 24%)"},".language-css .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.function":{color:"hsl(198, 99%, 37%)"},".language-css .token.url > .token.string.url":{color:"hsl(119, 34%, 47%)"},".language-css .token.important":{color:"hsl(301, 63%, 40%)"},".language-css .token.atrule .token.rule":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.operator":{color:"hsl(301, 63%, 40%)"},".language-javascript .token.template-string > .token.interpolation > .token.interpolation-punctuation.punctuation":{color:"hsl(344, 84%, 43%)"},".language-json .token.operator":{color:"hsl(230, 8%, 24%)"},".language-json .token.null.keyword":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.url":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.operator":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url-reference.url > .token.string":{color:"hsl(230, 8%, 24%)"},".language-markdown .token.url > .token.content":{color:"hsl(221, 87%, 60%)"},".language-markdown .token.url > .token.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.url-reference.url":{color:"hsl(198, 99%, 37%)"},".language-markdown .token.blockquote.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.hr.punctuation":{color:"hsl(230, 4%, 64%)",fontStyle:"italic"},".language-markdown .token.code-snippet":{color:"hsl(119, 34%, 47%)"},".language-markdown .token.bold .token.content":{color:"hsl(35, 99%, 36%)"},".language-markdown .token.italic .token.content":{color:"hsl(301, 63%, 40%)"},".language-markdown .token.strike .token.content":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.strike .token.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.list.punctuation":{color:"hsl(5, 74%, 59%)"},".language-markdown .token.title.important > .token.punctuation":{color:"hsl(5, 74%, 59%)"},bold:{fontWeight:"bold"},italic:{fontStyle:"italic"},namespace:{Opacity:"0.8"},"token.tab:not(:empty):before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.cr:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.lf:before":{color:"hsla(230, 8%, 24%, 0.2)"},"token.space:before":{color:"hsla(230, 8%, 24%, 0.2)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item":{marginRight:"0.4em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 6%, 44%)",padding:"0.1em 0.4em",borderRadius:"0.3em"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > button:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > a:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:hover":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},"div.code-toolbar > .toolbar.toolbar > .toolbar-item > span:focus":{background:"hsl(230, 1%, 78%)",color:"hsl(230, 8%, 24%)"},".line-highlight.line-highlight":{background:"hsla(230, 8%, 24%, 0.05)"},".line-highlight.line-highlight:before":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},".line-highlight.line-highlight[data-end]:after":{background:"hsl(230, 1%, 90%)",color:"hsl(230, 8%, 24%)",padding:"0.1em 0.6em",borderRadius:"0.3em",boxShadow:"0 2px 0 0 rgba(0, 0, 0, 0.2)"},"pre[id].linkable-line-numbers.linkable-line-numbers span.line-numbers-rows > span:hover:before":{backgroundColor:"hsla(230, 8%, 24%, 0.05)"},".line-numbers.line-numbers .line-numbers-rows":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".command-line .command-line-prompt":{borderRightColor:"hsla(230, 8%, 24%, 0.2)"},".line-numbers .line-numbers-rows > span:before":{color:"hsl(230, 1%, 62%)"},".command-line .command-line-prompt > span:before":{color:"hsl(230, 1%, 62%)"},".rainbow-braces .token.token.punctuation.brace-level-1":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-5":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-9":{color:"hsl(5, 74%, 59%)"},".rainbow-braces .token.token.punctuation.brace-level-2":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-6":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-10":{color:"hsl(119, 34%, 47%)"},".rainbow-braces .token.token.punctuation.brace-level-3":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-7":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-11":{color:"hsl(221, 87%, 60%)"},".rainbow-braces .token.token.punctuation.brace-level-4":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-8":{color:"hsl(301, 63%, 40%)"},".rainbow-braces .token.token.punctuation.brace-level-12":{color:"hsl(301, 63%, 40%)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)":{backgroundColor:"hsla(353, 100%, 66%, 0.15)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix)::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre > code.diff-highlight .token.token.deleted:not(.prefix) *::selection":{backgroundColor:"hsla(353, 95%, 66%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)":{backgroundColor:"hsla(137, 100%, 55%, 0.15)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::-moz-selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre.diff-highlight > code .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix)::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},"pre > code.diff-highlight .token.token.inserted:not(.prefix) *::selection":{backgroundColor:"hsla(135, 73%, 55%, 0.25)"},".prism-previewer.prism-previewer:before":{borderColor:"hsl(0, 0, 95%)"},".prism-previewer-gradient.prism-previewer-gradient div":{borderColor:"hsl(0, 0, 95%)",borderRadius:"0.3em"},".prism-previewer-color.prism-previewer-color:before":{borderRadius:"0.3em"},".prism-previewer-easing.prism-previewer-easing:before":{borderRadius:"0.3em"},".prism-previewer.prism-previewer:after":{borderTopColor:"hsl(0, 0, 95%)"},".prism-previewer-flipped.prism-previewer-flipped.after":{borderBottomColor:"hsl(0, 0, 95%)"},".prism-previewer-angle.prism-previewer-angle:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-time.prism-previewer-time:before":{background:"hsl(0, 0%, 100%)"},".prism-previewer-easing.prism-previewer-easing":{background:"hsl(0, 0%, 100%)"},".prism-previewer-angle.prism-previewer-angle circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-time.prism-previewer-time circle":{stroke:"hsl(230, 8%, 24%)",strokeOpacity:"1"},".prism-previewer-easing.prism-previewer-easing circle":{stroke:"hsl(230, 8%, 24%)",fill:"transparent"},".prism-previewer-easing.prism-previewer-easing path":{stroke:"hsl(230, 8%, 24%)"},".prism-previewer-easing.prism-previewer-easing line":{stroke:"hsl(230, 8%, 24%)"}};var s=e.i(488012);e.s(["f",0,({code:e,language:a})=>{let d=(0,s.f)(l),[u,c]=(0,r.useState)(!1);return(0,t.jsxs)("div",{className:"relative rounded-lg border border-border bg-muted overflow-hidden",children:[(0,t.jsx)("button",{onClick:()=>{navigator.clipboard.writeText(e),c(!0),setTimeout(()=>c(!1),2e3)},className:"absolute top-3 right-3 p-2 rounded-md border border-border bg-background text-muted-foreground hover:bg-accent hover:text-foreground z-raised","aria-label":"Copy code",children:u?(0,t.jsx)(i.f,{size:16}):(0,t.jsx)(o,{size:16})}),(0,t.jsx)(n.f,{language:a,style:d,customStyle:{margin:0,padding:"1.5rem",borderRadius:"0.5rem",fontSize:"0.9rem",background:"transparent"},codeTagProps:{style:{background:"transparent"}},showLineNumbers:!0,children:e})]})}],466828)},909947,e=>{var t=e.i(865361);e.s(["f",0,e=>{let r,{apiKeySource:i,accessToken:o,apiKey:n,inputMessage:l,chatHistory:s,selectedTags:a,selectedVectorStores:d,selectedGuardrails:u,selectedPolicies:c,selectedVoice:p,endpointType:m,selectedModel:g,selectedSdk:h,proxySettings:f,customHeaders:b}=e,v="session"===i?o:n,x=window.location.origin,k=f?.LITELLM_UI_API_DOC_BASE_URL;k&&k.trim()?x=k:f?.PROXY_BASE_URL&&(x=f.PROXY_BASE_URL);let _=l||"Your prompt here",y=_.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),w=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),j={};a.length>0&&(j.tags=a),d.length>0&&(j.vector_stores=d),u.length>0&&(j.guardrails=u),c.length>0&&(j.policies=c);let C=g||"your-model-name",N=b&&Object.keys(b).length>0?`,
	default_headers=${JSON.stringify(b,null,2).replace(/\n/g,"\n	")}`:"",E="azure"===h?`import openai

client = openai.AzureOpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${x}",
	api_version="2024-02-01"${N}
)`:`import openai

client = openai.OpenAI(
	api_key="${v||"YOUR_LITELLM_API_KEY"}",
	base_url="${x}"${N}
)`;switch(m){case t.Q.CHAT:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:_}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${C}",
    messages=${JSON.stringify(i,null,4)}${t}
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
#     ]${t}
# )
# print(response_with_file)
`;break}case t.Q.RESPONSES:{let e=Object.keys(j).length>0,t="";if(e){let e=JSON.stringify({metadata:j},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();t=`,
    extra_body=${e}`}let i=w.length>0?w:[{role:"user",content:_}];r=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${C}",
    input=${JSON.stringify(i,null,4)}${t}
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
#                 {"type": "input_text", "text": "${y}"},
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
`;break}case t.Q.IMAGE:r="azure"===h?`
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
	prompt="${l}",
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
	model="${C}",
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
`;break;case t.Q.IMAGE_EDITS:r="azure"===h?`
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
	model="${C}",
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
prompt = "${y}"

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
`;break;case t.Q.EMBEDDINGS:r=`
response = client.embeddings.create(
	input="${l||"Your string here"}",
	model="${C}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case t.Q.TRANSCRIPTION:r=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${C}",
	file=audio_file${l?`,
	prompt="${l.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case t.Q.SPEECH:r=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${C}",
	input="${l||"Your text to convert to speech here"}",
	voice="${p}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${C}",
#     input="${l||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:r="\n# Code generation for this endpoint is not implemented yet."}return`${E}
${r}`}])},209261,e=>{var t=e.i(201787);let r=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,i=e=>e.trim().replace(/\/+$/,""),o=/\.(md|markdown|txt|json|ya?ml|toml)$/i,n=/\.zip$/i,l=/^[0-9a-fA-F]{64}$/,s=/^\d{1,3}(\.\d{1,3}){3}$/,a=/^[A-Za-z0-9-]+$/,d=/^[A-Za-z0-9._-]+$/,u=/^https?:\/\//i,c="ssh://",p=/^([a-z0-9._-]+)@([^:/@]+):(?!\/)(.+)$/i,m=e=>e.pathname.split("/").filter(e=>""!==e),g=e=>{try{return new URL(e)}catch{return null}},h=e=>e.hostname.includes(".")&&!e.hostname.startsWith("[")&&!s.test(e.hostname),f=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},b=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),v=(e,o,n,l)=>{let s=i(l??"");return""!==s?r.test(s)?{parsed:{source:"git-subdir",url:o,path:s},label:t.f.t("claudeCodePlugins.sources.gitSubdir",{kind:e,url:o,path:s}),suggestedName:b(f(s))}:null:{parsed:{source:"url",url:o},label:t.f.t("claudeCodePlugins.sources.gitRepo",{kind:e,url:o}),suggestedName:b(n)}},x="<your-litellm-key>";e.s(["i",0,x,"U",0,(e,t)=>JSON.stringify({extraKnownMarketplaces:{litellm:{source:{source:"url",url:`${e}/claude-code/marketplace.json`,...t?{headers:{Authorization:`Bearer ${x}`}}:{}}}}},null,2),"a",0,e=>`/plugin install ${e.name}@litellm`,"u",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"_",0,e=>"github"===e.source&&e.repo?t.f.t("claudeCodePlugins.sources.githubDisplay",{repo:e.repo}):"git-subdir"===e.source&&e.url&&e.path?`${e.url} @ ${e.path}`:("url"===e.source||"archive"===e.source)&&e.url?e.url:t.f.t("claudeCodePlugins.sources.unknown"),"E",0,e=>"github"===e.source&&e.repo?`https://github.com/${e.repo}`:("url"===e.source||"git-subdir"===e.source||"archive"===e.source)&&e.url&&u.test(e.url)?e.url:null,"r",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"R",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"b",0,e=>""===e.trim()||l.test(e.trim()),"P",0,e=>{let t=i(e);return""!==t&&r.test(t)},"D",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"Z",0,(e,l)=>{let s=((e,t)=>{let r=e.trim(),i=p.exec(r),o=i?`${c}${i[1]}@${i[2]}/${i[3]}`:r;if(!o.toLowerCase().startsWith(c))return null;let n=g(o);if(!n||""===n.username||""!==n.password||!h(n))return null;let l=o.indexOf("/",c.length);return -1===l||n.pathname!==o.slice(l)||m(n).length<2?null:v("SSH",r,f(n.pathname).replace(/\.git$/i,""),t)})(e,l);if(s)return s;let u=(e=>{let t=e.trim();if(""===t||t.startsWith("//"))return null;let r=g(/^[a-z][a-z0-9+.-]*:\/\//i.test(t)?t:`https://${t}`);return r&&"https:"===r.protocol&&""===r.username&&""===r.password&&h(r)?r:null})(e);if(!u)return null;if(n.test(u.pathname))return{parsed:{source:"archive",url:u.href},label:t.f.t("claudeCodePlugins.sources.zipArchive",{location:`${u.host}${u.pathname}`}),suggestedName:b(f(u.pathname).replace(n,""))};if("github.com"===u.hostname.replace(/^www\./,""))return((e,n)=>{let l=m(e);if(l.length<2)return null;let s=l[0],u=l[1].replace(/\.git$/,"");if(!a.test(s)||!d.test(u))return null;let c=`${s}/${u}`,p=`https://github.com/${c}`,g={parsed:{source:"github",repo:c},label:t.f.t("claudeCodePlugins.sources.githubRepo",{repo:c}),suggestedName:b(u)};if(l.length>=4&&("tree"===l[2]||"blob"===l[2])){let e=l.slice(4),n=f(e.join("/")),s=o.test(n)?e.slice(0,-1):e;if(0===s.length)return g;let a=i(s.join("/"));return r.test(a)?{parsed:{source:"git-subdir",url:p,path:a},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:a}),suggestedName:b(f(a))}:null}if(2!==l.length)return null;let h=i(n??"");return""!==h?r.test(h)?{parsed:{source:"git-subdir",url:p,path:h},label:t.f.t("claudeCodePlugins.sources.githubSubdir",{repo:c,path:h}),suggestedName:b(f(h))}:null:g})(u,l);if(m(u).length<2)return null;let x=f(u.pathname).replace(/\.git$/,"");return v("Git",`${u.protocol}//${u.host}${u.pathname.replace(/\/+$/,"")}`,x,l)},"e",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)])},652272,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(871689),o=e.i(643531),n=e.i(174886),l=e.i(306228),s=e.i(345771),a=e.i(322831),d=e.i(842354),u=e.i(196631),c=e.i(209261);let p=({source:e})=>{let{t:r}=(0,a.f)(),i=(0,c.E)(e),o=i&&"git-subdir"===e.source&&e.path?`${i}/tree/main/${e.path}`:i;return o?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:r("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsxs)("a",{href:o,target:"_blank",rel:"noopener noreferrer",className:"flex items-center gap-1 break-all text-[13px] text-info",children:[o.replace("https://",""),(0,t.jsx)(l.f,{className:"size-3 shrink-0"})]})]}):e.url?(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:r("claudeCodePlugins.skillDetail.overview.source")}),(0,t.jsx)("div",{className:"break-all text-[13px] text-foreground",children:(0,c._)(e)})]}):null};e.s(["f",0,({skill:e,onBack:l})=>{let{t:m}=(0,a.f)(),[g,h]=(0,r.useState)("overview"),[f,b]=(0,r.useState)(null),v=(0,d.f)(),x=(e,t)=>{navigator.clipboard.writeText(e),b(t),setTimeout(()=>b(null),2e3)},k=(0,c.a)(e),_=(0,c.U)(window.location.origin,v),y=[...e.category?[{property:m("claudeCodePlugins.skillDetail.properties.category"),value:e.category}]:[],...e.domain?[{property:m("claudeCodePlugins.skillDetail.properties.domain"),value:e.domain}]:[],...e.namespace?[{property:m("claudeCodePlugins.skillDetail.properties.namespace"),value:e.namespace}]:[],...e.version?[{property:m("claudeCodePlugins.skillDetail.properties.version"),value:e.version}]:[],...e.author?.name?[{property:m("claudeCodePlugins.skillDetail.properties.author"),value:e.author.name}]:[],...e.created_at?[{property:m("claudeCodePlugins.skillDetail.properties.added"),value:new Date(e.created_at).toLocaleDateString()}]:[]],w=[{key:"overview",label:m("claudeCodePlugins.skillDetail.tabs.overview")},{key:"usage",label:m("claudeCodePlugins.skillDetail.tabs.usage")}];return(0,t.jsxs)("div",{className:"py-6 pl-0 pr-8",children:[(0,t.jsxs)("div",{onClick:l,className:"mb-6 inline-flex cursor-pointer items-center gap-1.5 text-sm text-muted-foreground",children:[(0,t.jsx)(i.f,{className:"size-3"}),(0,t.jsx)("span",{children:m("claudeCodePlugins.skillDetail.back")})]}),(0,t.jsxs)("div",{className:"mb-2",children:[(0,t.jsx)("h1",{className:"m-0 text-[28px] font-normal leading-tight text-foreground",children:e.name}),e.description&&(0,t.jsx)("p",{className:"mb-0 ml-0 mr-0 mt-2 text-sm leading-relaxed text-muted-foreground",children:e.description})]}),(0,t.jsx)("div",{className:"mb-7 mt-6 border-b border-border",children:(0,t.jsx)("div",{className:"flex",children:w.map(e=>(0,t.jsx)("div",{onClick:()=>h(e.key),className:(0,u.c)("-mb-px cursor-pointer border-b-[3px] px-5 py-3 text-sm",g===e.key?"border-info font-medium text-info":"border-transparent font-normal text-muted-foreground"),children:e.label},e.key))})}),"overview"===g&&(0,t.jsxs)("div",{className:"flex gap-16",children:[(0,t.jsxs)("div",{className:"min-w-0 flex-1",children:[(0,t.jsx)("h2",{className:"m-0 mb-1 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.overview.title")}),(0,t.jsx)("p",{className:"m-0 mb-4 text-[13px] text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.subtitle")}),(0,t.jsxs)("table",{className:"w-full border-collapse text-sm",children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("th",{className:"w-40 py-3 text-left font-medium text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.propertyHeader")}),(0,t.jsx)("th",{className:"py-3 text-left font-medium text-muted-foreground",children:e.name})]})}),(0,t.jsx)("tbody",{children:y.map((e,r)=>(0,t.jsxs)("tr",{className:"border-b border-border",children:[(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.property}),(0,t.jsx)("td",{className:"py-3 text-foreground",children:e.value})]},r))})]})]}),(0,t.jsxs)("div",{className:"w-60 shrink-0",children:[(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.status")}),(0,t.jsx)("span",{className:(0,u.c)("rounded-xl px-2.5 py-[3px] text-xs font-medium",e.enabled?"bg-success/10 text-success":"bg-muted text-muted-foreground"),children:m(e.enabled?"claudeCodePlugins.skillDetail.overview.public":"claudeCodePlugins.skillDetail.overview.draft")})]}),(0,t.jsx)(p,{source:e.source}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{className:"mb-6",children:[(0,t.jsx)("div",{className:"mb-2 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.tags")}),(0,t.jsx)("div",{className:"flex flex-wrap gap-1.5",children:e.keywords.map(e=>(0,t.jsx)("span",{className:"rounded-2xl border border-border bg-card px-3 py-1 text-xs text-foreground",children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{className:"mb-1 text-xs text-muted-foreground",children:m("claudeCodePlugins.skillDetail.overview.skillId")}),(0,t.jsx)("div",{className:"break-all font-mono text-xs text-foreground",children:e.id})]})]})]}),"usage"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.usage.title")}),(0,t.jsx)("p",{className:"m-0 mb-6 text-sm leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.usage.intro")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:m("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>x(k,"install"),className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","install"===f?"text-success":"text-info"),children:["install"===f?(0,t.jsx)(o.f,{className:"size-3"}):(0,t.jsx)(n.f,{className:"size-3"}),m("install"===f?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-sm text-foreground",children:k})]}),(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.usage.staleCatalog",{name:e.name})}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace update litellm"})]}),(0,t.jsxs)("p",{className:"m-0 text-[13px] leading-relaxed text-muted-foreground",children:[m("claudeCodePlugins.skillDetail.usage.notConfigured")," ",(0,t.jsx)("span",{onClick:()=>h("setup"),className:"cursor-pointer text-info",children:m("claudeCodePlugins.skillDetail.usage.seeSetup")})]})]}),"setup"===g&&(0,t.jsxs)("div",{className:"max-w-[640px]",children:[(0,t.jsx)("h2",{className:"m-0 mb-2 text-lg font-normal text-foreground",children:m("claudeCodePlugins.skillDetail.setup.title")}),v?(0,t.jsxs)(t.Fragment,{children:[(0,t.jsxs)("div",{className:"mb-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3",children:[(0,t.jsx)("p",{className:"m-0 mb-2 text-[13px] leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.setup.removeFirst")}),(0,t.jsx)("pre",{className:"m-0 bg-transparent font-mono text-[13px] text-foreground",children:"/plugin marketplace remove litellm"})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(s.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.requiresKey",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"}),placeholder:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:c.i})}})})]}):(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:m("claudeCodePlugins.skillDetail.setup.runCommand")}),(0,t.jsxs)("div",{className:"mb-6 overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:m("claudeCodePlugins.skillDetail.runInClaudeCode")}),(0,t.jsxs)("button",{onClick:()=>{let e=window.location.origin;x(`/plugin marketplace add ${e}/claude-code/marketplace.json`,"marketplace-cmd")},className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","marketplace-cmd"===f?"text-success":"text-info"),children:["marketplace-cmd"===f?(0,t.jsx)(o.f,{className:"size-3"}):(0,t.jsx)(n.f,{className:"size-3"}),m("marketplace-cmd"===f?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:`/plugin marketplace add ${window.location.origin}/claude-code/marketplace.json`})]}),(0,t.jsx)("p",{className:"m-0 mb-3 text-sm leading-relaxed text-muted-foreground",children:(0,t.jsx)(s.f,{i18nKey:"claudeCodePlugins.skillDetail.setup.orSettings",components:{settings:(0,t.jsx)("code",{className:"rounded bg-muted px-1.5 py-px text-[13px]",children:"~/.claude/settings.json"})}})})]}),(0,t.jsxs)("div",{className:"overflow-hidden rounded-lg border border-border",children:[(0,t.jsxs)("div",{className:"flex items-center justify-between border-b border-border bg-muted px-4 py-2.5",children:[(0,t.jsx)("span",{className:"text-[13px] font-medium text-foreground",children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>x(_,"settings"),className:(0,u.c)("flex cursor-pointer items-center gap-1 border-none bg-transparent p-0 text-xs","settings"===f?"text-success":"text-info"),children:["settings"===f?(0,t.jsx)(o.f,{className:"size-3"}):(0,t.jsx)(n.f,{className:"size-3"}),m("settings"===f?"claudeCodePlugins.skillDetail.copied":"claudeCodePlugins.skillDetail.copy")]})]}),(0,t.jsx)("pre",{className:"m-0 bg-card px-4 py-3.5 font-mono text-[13px] text-foreground",children:_})]})]})]})}])},902555,e=>{var t=e.i(843476),r=e.i(746798),i=e.i(271645);let o=i.forwardRef(function(e,t){return i.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),i.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"}))}),n=i.forwardRef(function(e,t){return i.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),i.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"}),i.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M21 12a9 9 0 11-18 0 9 9 0 0118 0z"}))});var l=e.i(278587),s=e.i(68155),a=e.i(360820),d=e.i(871943),u=e.i(434626);let c=i.forwardRef(function(e,t){return i.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:t},e),i.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"}))});var p=e.i(196631);function m({icon:e,onClick:r,className:i,disabled:o,dataTestId:n}){return o?(0,t.jsx)("span",{className:"inline-flex shrink-0 cursor-not-allowed items-center justify-center p-1.5 opacity-50","data-testid":n,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})}):(0,t.jsx)("span",{className:(0,p.P)("inline-flex shrink-0 cursor-pointer items-center justify-center p-1.5",i),onClick:r,"data-testid":n,children:(0,t.jsx)(e,{className:"size-5 shrink-0"})})}let g={Edit:{icon:o,className:"hover:text-info"},Delete:{icon:s.f,className:"hover:text-destructive"},Test:{icon:n,className:"hover:text-info"},Regenerate:{icon:l.f,className:"hover:text-success"},Reset:{icon:l.f,className:"hover:text-info"},Up:{icon:a.f,className:"hover:text-info"},Down:{icon:d.f,className:"hover:text-info"},Open:{icon:u.f,className:"hover:text-success"},Copy:{icon:c,className:"hover:text-info"}};e.s(["f",0,function({onClick:e,tooltipText:i,disabled:o=!1,disabledTooltipText:n,dataTestId:l,variant:s}){let{icon:a,className:d}=g[s],u=o?n:i,c=(0,t.jsx)(m,{icon:a,onClick:e,className:d,disabled:o,dataTestId:l});return u?(0,t.jsx)(r.g,{children:(0,t.jsxs)(r.W,{children:[(0,t.jsx)(r.y,{render:(0,t.jsx)("span",{tabIndex:o?0:void 0}),children:c}),(0,t.jsx)(r.Y,{children:u})]})}):(0,t.jsx)("span",{children:c})}],902555)},292335,e=>{let t={NONE:"none",API_KEY:"api_key",BEARER_TOKEN:"bearer_token",TOKEN:"token",BASIC:"basic",AUTHORIZATION:"authorization",OAUTH2:"oauth2",OAUTH2_TOKEN_EXCHANGE:"oauth2_token_exchange",OAUTH2_ID_JAG:"oauth2_id_jag",AWS_SIGV4:"aws_sigv4",TRUE_PASSTHROUGH:"true_passthrough",OAUTH_DELEGATE:"oauth_delegate"},r=[{value:t.NONE,label:"None"},{value:t.API_KEY,label:"API Key"},{value:t.BEARER_TOKEN,label:"Bearer Token"},{value:t.TOKEN,label:"Token"},{value:t.BASIC,label:"Basic Auth"},{value:t.AUTHORIZATION,label:"Authorization (raw header)"},{value:t.OAUTH2,label:"OAuth"},{value:t.OAUTH2_TOKEN_EXCHANGE,label:"OAuth Token Exchange (OBO)"},{value:t.OAUTH2_ID_JAG,label:"ID-JAG (Okta Cross App Access)"},{value:t.AWS_SIGV4,label:"AWS SigV4 (Bedrock AgentCore MCPs)"},{value:t.TRUE_PASSTHROUGH,label:"True Passthrough (no LiteLLM auth)"},{value:t.OAUTH_DELEGATE,label:"OAuth Delegate (client-supplied upstream token)"}],i=e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE,o={INTERACTIVE:"interactive",M2M:"m2m"},n=e=>{let t=e.credentials??{};return JSON.stringify({url:"string"==typeof e.url?e.url:null,spec_path:"string"==typeof e.spec_path?e.spec_path:null,auth_type:e.auth_type??null,oauth_flow_type:e.oauth_flow_type??null,client_id:t.client_id??null,client_secret:t.client_secret??null,scopes:t.scopes??null,upstream_resource:t.upstream_resource??null,issuer:e.issuer??null,authorization_url:e.authorization_url??null,token_url:e.token_url??null,registration_url:e.registration_url??null})},l=["client_id","client_secret"],s=["upstream_resource","upstream_token_header"],a=["access_token","refresh_token","expires_in","scope"],d=(e,t)=>{if(!e)return;let r=Object.fromEntries(t.filter(t=>"string"==typeof e[t]&&""!==e[t]).map(t=>[t,e[t]]));return Object.keys(r).length>0?r:void 0},u="client_credentials",c={SSE:"sse",HTTP:"http",STDIO:"stdio",OPENAPI:"openapi"},p=[{value:c.HTTP,label:"Streamable HTTP (Recommended)"},{value:c.SSE,label:"Server-Sent Events (SSE)"},{value:c.STDIO,label:"Standard Input/Output (stdio)"},{value:c.OPENAPI,label:"OpenAPI Spec"}];e.s(["_",0,s,"W",0,t,"B",0,r,"L",0,["credentials"],"V",0,"authorization_code","P",0,u,"S",0,"Server responded. Authentication and tools were not checked","n",0,o,"v",0,c,"m",0,p,"J",0,e=>e===t.TRUE_PASSTHROUGH||e===t.OAUTH_DELEGATE?"client_forwarded":e??null,"A",0,e=>e.auth_type===t.TRUE_PASSTHROUGH||e.auth_type===t.OAUTH_DELEGATE&&!e.dcr_bridge,"j",0,function(e){return e.auth_type===t.OAUTH2_TOKEN_EXCHANGE?"token_exchange":e.auth_type!==t.OAUTH2?null:e.oauth2_flow===u?"m2m":e.delegate_auth_to_upstream?"passthrough":"authorization_code"},"Z",0,n,"d",0,e=>null==e?t.NONE:e,"K",0,(e,t)=>null==e?c.SSE:t&&e!==c.STDIO?c.OPENAPI:e,"C",0,i,"i",0,(e,t)=>void 0!==t&&n(e)!==t,"z",0,e=>i(e)||e===t.OAUTH2_TOKEN_EXCHANGE,"Y",0,function(e){return e===u?o.M2M:e?o.INTERACTIVE:void 0},"o",0,e=>{let t=d(e,[...l,...s]),r=(e=>{if(!Array.isArray(e))return;let t=e.filter(e=>"string"==typeof e&&""!==e);return t.length>0?t:void 0})(e?.scopes);return void 0===r?t:{...t??{},scopes:r}},"T",0,e=>d(e,l),"f",0,e=>{if(!e)return;let t=Object.fromEntries(Object.entries(e).filter(([e])=>!a.includes(e)));return Object.keys(t).length>0?t:void 0}])},174553,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(916925),o=e.i(555987),n=e.i(196631);let l=/(?:\/assets\/logos\/|\/_next\/static\/media\/)/,s={"baseten.svg":"invert","cursor.svg":"invert","edenai.svg":"invert","enkrypt_ai.avif":"invert","friendli.svg":"invert","github.svg":"invert","github_copilot.svg":"invert","lago.svg":"invert","lambda.svg":"invert","langflow.svg":"invert","litellm_monogram.svg":"invert","lmstudio.svg":"invert","moonshot.svg":"invert","nebius.svg":"invert","notion.svg":"invert","ollama.svg":"invert","openrouter.svg":"invert","promptguard.svg":"invert","recraft.svg":"invert","replicate.svg":"invert","runway.png":"invert","scx_ai.svg":"invert","secret_detect.png":"invert","topaz.svg":"invert","v0.svg":"invert","vercel.svg":"invert","watsonx.svg":"invert","aiand.svg":"plate","aiml_api.svg":"plate","akto.svg":"plate","aws.svg":"plate","deepkeep.svg":"plate","fireworks.svg":"plate","llm_guard.png":"plate","pangea.png":"plate","repelloai.png":"plate","sambanova.svg":"plate","sentry.svg":"plate","valkey.svg":"plate"},a={invert:"dark:[filter:brightness(0)_invert(1)]",plate:"dark:bg-logo-surface dark:object-contain dark:p-0.5"};e.s(["f",0,({provider:e,src:d,label:u,className:c="w-4 h-4"})=>{let[p,m]=(0,r.useState)(null),g=void 0!==e?(0,i.h)(e).logo:(0,o.u)(d)??"",h=u??e??"";if(p===g||!g)return(0,t.jsx)("div",{className:`${c} rounded-full bg-border flex items-center justify-center text-xs`,children:h.charAt(0)||"-"});let f=(e=>{let t;if(!e||(0,o.i)(e)||!l.test(e))return;let r=e.split(/[?#]/)[0].split("/").pop()||void 0,i=void 0===r||(t=r.split(".")).length<2?void 0:`${t[0]}.${t[t.length-1]}`;return void 0===i?void 0:s[i]})(g);return(0,t.jsx)("img",{src:g,alt:`${h||"-"} logo`,className:void 0===f?c:(0,n.c)(c,a[f]),onError:()=>{console.warn(`Logo failed to load: ${g}`),m(g)}})}],174553)},845150,e=>{var t=e.i(843476),r=e.i(271645),i=e.i(131792);let o=(e,t)=>{let r=t.trim().toLowerCase();return!r||e.label.toLowerCase().includes(r)||e.value.toLowerCase().includes(r)||(e.description?.toLowerCase().includes(r)??!1)};e.s(["f",0,function({id:e,options:n,value:l=[],onValueChange:s,placeholder:a="Select options",emptyText:d="No options found",disabled:u=!1,loading:c=!1,allowCustomValues:p=!1,className:m,contentClassName:g}){let h=(0,i.d)(),[f,b]=(0,r.useState)(""),v=n.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),x=l.filter(e=>"string"==typeof e&&e.length>0).map(e=>v.find(t=>t.value===e)??{label:e,value:e}),k=f.trim(),_=v.some(e=>e.value.toLowerCase()===k.toLowerCase()),y=p&&k&&!_?[...v,{label:`Create "${k}"`,value:k}]:v;return(0,t.jsxs)(i.x,{multiple:!0,items:y,value:x,onValueChange:e=>{s(Array.from(new Set(p?e.flatMap(e=>l.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),b("")},inputValue:f,onInputValueChange:b,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:o,disabled:u||c,children:[(0,t.jsx)(i.V,{render:(0,t.jsx)("div",{ref:h}),className:`min-h-8 py-1 text-sm ${m??""}`,children:(0,t.jsx)(i.l,{children:r=>(0,t.jsxs)(t.Fragment,{children:[r.map(e=>(0,t.jsx)(i.Q,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(i.z,{id:e,placeholder:c?"Loading...":a,className:"min-w-24","aria-label":a||void 0}),r.length>0&&!u&&!c&&(0,t.jsx)(i.y,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(i.R,{anchor:h,className:g,children:[(0,t.jsx)(i.S,{children:d}),(0,t.jsx)(i.k,{children:e=>(0,t.jsx)(i.T,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},735450,e=>{var t=e.i(411382),r=e.i(772325),i=e.i(54105),o=e.i(240192),n=e.i(138620);let l="https://docs.litellm.ai/docs/decisions",s=`${l}#supported-providers`,a=`${l}#self-hosted-laya-and-nimble`,d=new Set(["/v1/decisions","/v1/systemone"]),u=t.z.object({litellm_provider:i.z.string(),mode:i.z.string().nullish(),supported_endpoints:r.z.array(i.z.string()).nullish()}),c="decisions",p=["evaluation",c],m=e=>null!=e&&p.includes(e),g=e=>m(e.mode)||(e.supported_endpoints??[]).some(e=>d.has(e)),h=e=>{let t=o.z.record(i.z.string(),n.z.unknown()).safeParse(e);return t.success?Object.entries(t.data).flatMap(([e,t])=>{let r=u.safeParse(t);return r.success?[[e,r.data]]:[]}):[]},f={decisions:0,"self-hosted":1,"needs-upgrade":2},b=t.z.object({decision_model_providers:r.z.array(i.z.string())}),v=(e,t)=>e.model.length-t.model.length||e.model.localeCompare(t.model),x=(e,t,r)=>({row:e,routeOrder:f[e.route],rankInProvider:r.slice(0,t).filter(t=>t.provider===e.provider).length});e.s(["O",0,l,"d",0,c,"Z",0,s,"L",0,a,"B",0,"playground?tab=system-one","F",0,function(e){let t=h(e),r=t.filter(([,e])=>g(e)),i=new Map([...new Set(r.map(([,e])=>e.litellm_provider))].map(e=>{let r=t.filter(([,t])=>t.litellm_provider===e);return[e,{decisionOnly:r.every(([,e])=>g(e))}]}));return{models:new Set(r.map(([e])=>e)),providers:i}},"W",0,(e,t)=>h(e).filter(([,e])=>g(e)).map(([e,r])=>({model:e,provider:r.litellm_provider,route:t?.has(r.litellm_provider)?"decisions":(r.supported_endpoints??["/v1/decisions"]).includes("/v1/decisions")?null===t?"decisions":"needs-upgrade":"self-hosted"})).toSorted(v).map(x).toSorted((e,t)=>e.routeOrder-t.routeOrder||e.rankInProvider-t.rankInProvider||e.row.provider.localeCompare(t.row.provider)).map(({row:e})=>e),"i",0,e=>m(e)?c:e??void 0,"q",0,e=>e.flatMap(e=>e===c?p:[e]),"m",0,g,"_",0,m,"r",0,(e,t,r)=>void 0!==t&&e.providers.get(t)?.decisionOnly===!0||r.some(t=>e.models.has(t)),"P",0,e=>{let t=b.safeParse(e);return t.success?new Set(t.data.decision_model_providers):null}])},899426,e=>{let t=e=>e.trim().toLowerCase();function r(e,r){let i=t(e);if(""===i)return!0;let o=r.filter(e=>"string"==typeof e).map(e=>e.toLowerCase());return!!o.some(e=>e.includes(i))||i.split(/\s+/).every(e=>o.some(t=>t.includes(e)))}e.s(["$",0,function(e,t,i){return e.filter(e=>r(t,i(e)))},"h",0,r,"e",0,function(e,r,i){let o=t(r);if(""===o)return[...e];let n=e=>{let t=i(e).toLowerCase();return 1e3*(t===o)+100*!!t.startsWith(o)+(1e3-t.length)};return[...e].sort((e,t)=>n(t)-n(e))}])}])})();