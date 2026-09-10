(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,546467,e=>{"use strict";let t=(0,e.i(475254).default)("external-link",[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]]);e.s(["default",0,t])},778917,e=>{"use strict";var t=e.i(546467);e.s(["ExternalLink",()=>t.default])},180127,e=>{"use strict";let t=(0,e.i(475254).default)("arrow-left",[["path",{d:"m12 19-7-7 7-7",key:"1l729n"}],["path",{d:"M19 12H5",key:"x3x0zl"}]]);e.s(["default",0,t])},871689,e=>{"use strict";var t=e.i(180127);e.s(["ArrowLeft",()=>t.default])},845150,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(131792);let a=(e,t)=>{let i=t.trim().toLowerCase();return!i||e.label.toLowerCase().includes(i)||e.value.toLowerCase().includes(i)||(e.description?.toLowerCase().includes(i)??!1)};e.s(["MultiSelect",0,function({id:e,options:n,value:r=[],onValueChange:s,placeholder:l="Select options",emptyText:p="No options found",disabled:d=!1,loading:u=!1,allowCustomValues:c=!1,className:g}){let m=(0,o.useComboboxAnchor)(),[f,h]=(0,i.useState)(""),x=n.filter(e=>null!=e&&"string"==typeof e.value&&e.value.length>0),_=r.filter(e=>"string"==typeof e&&e.length>0).map(e=>x.find(t=>t.value===e)??{label:e,value:e}),b=f.trim(),y=x.some(e=>e.value.toLowerCase()===b.toLowerCase()),v=c&&b&&!y?[...x,{label:`Create "${b}"`,value:b}]:x;return(0,t.jsxs)(o.Combobox,{multiple:!0,items:v,value:_,onValueChange:e=>{s(Array.from(new Set(c?e.flatMap(e=>r.includes(e.value)?[e.value]:e.value.split(",").map(e=>e.trim()).filter(e=>e.length>0)):e.map(e=>e.value)))),h("")},inputValue:f,onInputValueChange:h,isItemEqualToValue:(e,t)=>e.value===t.value,itemToStringLabel:e=>e.label,filter:a,disabled:d||u,children:[(0,t.jsx)(o.ComboboxChips,{render:(0,t.jsx)("div",{ref:m}),className:`min-h-8 py-1 text-sm ${g??""}`,children:(0,t.jsx)(o.ComboboxValue,{children:i=>(0,t.jsxs)(t.Fragment,{children:[i.map(e=>(0,t.jsx)(o.ComboboxChip,{"aria-label":e.label,children:e.label},e.value)),(0,t.jsx)(o.ComboboxChipsInput,{id:e,placeholder:u?"Loading...":l,className:"min-w-24","aria-label":l||void 0}),i.length>0&&!d&&!u&&(0,t.jsx)(o.ComboboxClear,{className:"ml-auto self-center","aria-label":"Clear all"})]})})}),(0,t.jsxs)(o.ComboboxContent,{anchor:m,children:[(0,t.jsx)(o.ComboboxEmpty,{children:p}),(0,t.jsx)(o.ComboboxList,{children:e=>(0,t.jsx)(o.ComboboxItem,{value:e,disabled:e.disabled,children:(0,t.jsxs)("span",{className:"min-w-0",children:[(0,t.jsx)("span",{className:"block truncate",children:e.label}),e.description&&(0,t.jsx)("span",{className:"block truncate text-xs text-muted-foreground",children:e.description})]})},e.value)})]})]})}])},337822,e=>{"use strict";var t,i=e.i(843476);e.s([],158421),e.i(158421);var o=e.i(271645),a=e.i(956789),n=e.i(17989),r=e.i(46420);e.i(247167);var s=e.i(733332);let l=o.createContext(void 0);function p(e){let t=o.useContext(l);if(void 0===t&&!e)throw Error((0,s.default)(47));return t}var d=e.i(174080),u=e.i(301252),c=e.i(616269),g=e.i(439957),m=e.i(56434),f=e.i(264111),h=e.i(116786),x=e.i(990627),_=e.i(638396);let b={...h.popupStoreSelectors,disabled:(0,c.createSelector)(e=>e.disabled),instantType:(0,c.createSelector)(e=>e.instantType),openMethod:(0,c.createSelector)(e=>e.openMethod),openChangeReason:(0,c.createSelector)(e=>e.openChangeReason),modal:(0,c.createSelector)(e=>e.modal),focusManagerModal:(0,c.createSelector)(e=>e.focusManagerModal),stickIfOpen:(0,c.createSelector)(e=>e.stickIfOpen),titleElementId:(0,c.createSelector)(e=>e.titleElementId),descriptionElementId:(0,c.createSelector)(e=>e.descriptionElementId),openOnHover:(0,c.createSelector)(e=>e.openOnHover),closeDelay:(0,c.createSelector)(e=>e.closeDelay),hasViewport:(0,c.createSelector)(e=>e.hasViewport)};class y extends u.ReactStore{constructor(e,t,i=!1){const a={...(0,h.createInitialPopupStoreState)(),disabled:!1,modal:!1,focusManagerModal:!1,instantType:void 0,openMethod:null,openChangeReason:null,titleElementId:void 0,descriptionElementId:void 0,stickIfOpen:!0,nested:!1,openOnHover:!1,closeDelay:0,hasViewport:!1,...e},n=new x.PopupTriggerMap;a.open&&e?.mounted===void 0&&(a.mounted=!0),a.floatingRootContext=(0,h.createPopupFloatingRootContext)(n,t,i),super(a,{popupRef:o.createRef(),backdropRef:o.createRef(),internalBackdropRef:o.createRef(),onOpenChange:void 0,onOpenChangeComplete:void 0,triggerFocusTargetRef:o.createRef(),beforeContentFocusGuardRef:o.createRef(),stickIfOpenTimeout:new g.Timeout,triggerElements:n},b)}setOpen=(e,t)=>{let i=t.reason===m.REASONS.triggerHover,o=t.reason===m.REASONS.triggerPress&&0===t.event.detail,a=!e&&(t.reason===m.REASONS.escapeKey||null==t.reason),n=(0,f.attachPreventUnmountOnClose)(t),r=this.select("activeTriggerId");if(e||t.reason!==m.REASONS.closePress||null!=t.trigger||null==r||(t.trigger=this.context.triggerElements.getById(r)??this.select("activeTriggerElement")??void 0),this.context.onOpenChange?.(e,t),t.isCanceled)return;this.state.floatingRootContext.dispatchOpenChange(e,t);let s=()=>{let i={open:e,openChangeReason:t.reason};(0,f.setPopupOpenState)(i,e,t.trigger,n()),this.update(i)};i?(this.set("stickIfOpen",!0),this.context.stickIfOpenTimeout.start(_.PATIENT_CLICK_THRESHOLD,()=>{this.set("stickIfOpen",!1)}),d.flushSync(s)):s(),o||a?this.set("instantType",o?"click":"dismiss"):t.reason===m.REASONS.focusOut?this.set("instantType","focus"):this.set("instantType",void 0)};static useStore(e,t){let{store:i,internalStore:a}=(0,f.usePopupStore)(e,(e,i)=>new y(t,e,i));return o.useEffect(()=>a?.disposeEffect(),[a]),i}disposeEffect=()=>this.context.stickIfOpenTimeout.disposeEffect()}var v=e.i(675606),S=e.i(176782);function C({props:e}){let{children:t,open:a,defaultOpen:n=!1,onOpenChange:s,onOpenChangeComplete:p,modal:d=!1,handle:u,triggerId:c,defaultTriggerId:g=null}=e,h=y.useStore(u?.store,{modal:d,open:n,openProp:a,activeTriggerId:g,triggerIdProp:c});(0,f.useInitialOpenSync)(h,a,n,g),h.useControlledProp("openProp",a),h.useControlledProp("triggerIdProp",c);let x=h.useState("open"),_=h.useState("mounted"),b=h.useState("payload"),S=null!=(0,r.useFloatingParentNodeId)();h.useContextCallback("onOpenChange",s),h.useContextCallback("onOpenChangeComplete",p),(0,f.usePopupRootSync)(h,x),(0,f.useImplicitActiveTrigger)(h);let{forceUnmount:E}=(0,f.useOpenStateTransitions)(x,h,()=>{h.update({stickIfOpen:!0,openChangeReason:null})});h.useSyncedValues({modal:d,nested:S}),o.useEffect(()=>{x||h.context.stickIfOpenTimeout.clear()},[h,x]);let I=o.useCallback(()=>{h.setOpen(!1,(0,v.createChangeEventDetails)(m.REASONS.imperativeAction))},[h]);o.useImperativeHandle(e.actionsRef,()=>({unmount:E,close:I}),[E,I]);let k=x||_,w=o.useMemo(()=>({store:h}),[h]);return(0,i.jsxs)(l.Provider,{value:w,children:[k&&(0,i.jsx)(j,{store:h,modal:d}),"function"==typeof t?t({payload:b}):t]})}function j({store:e,modal:t}){let i=e.useState("floatingRootContext"),r=(0,n.useDismiss)(i,{outsidePressEvent:{mouse:"trap-focus"===t?"sloppy":"intentional",touch:"sloppy"}}),s=r.reference??a.EMPTY_OBJECT,l=r.trigger??a.EMPTY_OBJECT,p=o.useMemo(()=>(0,S.mergeProps)(f.FOCUSABLE_POPUP_PROPS,r.floating),[r.floating]);return(0,f.usePopupInteractionProps)(e,{activeTriggerProps:s,inactiveTriggerProps:l,popupProps:p}),null}var E=e.i(540886),I=e.i(405005),k=e.i(552245),w=e.i(650316),R=e.i(385689),O=e.i(872135),T=e.i(788015),P=e.i(152535),A=e.i(346570),N=e.i(32199);let $=o.forwardRef(function(e,t){let{render:a,className:n,style:r,disabled:l=!1,nativeButton:d=!0,handle:u,payload:c,openOnHover:g=!1,delay:h=300,closeDelay:x=0,id:b,...y}=e,v=p(!0),S=u?.store??v?.store;if(!S)throw Error((0,s.default)(74));let C=(0,T.useBaseUiId)(b),j=S.useState("isTriggerActive",C),$=S.useState("floatingRootContext"),M=S.useState("isOpenedByTrigger",C),z=S.useState("triggerPopupId",C),D=o.useRef(null),{registerTrigger:L,isMountedByThisTrigger:H}=(0,f.useTriggerDataForwarding)(C,D,S,{payload:c,disabled:l,openOnHover:g,closeDelay:x}),F=S.useState("openChangeReason"),B=S.useState("stickIfOpen"),G=S.useState("openMethod"),U=S.useState("focusManagerModal"),V=(0,O.useHoverReferenceInteraction)($,{enabled:!l&&null!=$&&g&&("touch"!==G||F!==m.REASONS.triggerPress),mouseOnly:!0,move:!1,handleClose:(0,w.safePolygon)(),restMs:h,delay:{close:x},triggerElementRef:D,isActiveTrigger:j,isClosing:()=>"ending"===S.select("transitionStatus")}),W=(0,R.useClick)($,{enabled:null!=$,stickIfOpen:B}),q=(0,N.useOpenMethodTriggerProps)(()=>S.select("open"),e=>{S.set("openMethod",e)}),K=S.useState("triggerProps",H),{getButtonProps:Y,buttonRef:Z}=(0,E.useButton)({disabled:l,native:d}),{preFocusGuardRef:J,handlePreFocusGuardFocus:Q,handleFocusTargetFocus:X}=(0,A.useTriggerFocusGuards)(S,D),ee=(0,k.useRenderElement)("button",e,{state:{disabled:l,open:M},ref:[Z,t,L,D],props:[W.reference,V,K,q,{[_.CLICK_TRIGGER_IDENTIFIER]:"",id:C,"aria-haspopup":"dialog","aria-expanded":M,"aria-controls":z},y,Y],stateAttributesMapping:{open:e=>e&&F===m.REASONS.triggerPress?I.pressableTriggerOpenStateMapping.open(e):I.triggerOpenStateMapping.open(e)}});return H&&!U?(0,i.jsxs)(o.Fragment,{children:[(0,i.jsx)(P.FocusGuard,{ref:J,onFocus:Q}),(0,i.jsx)(o.Fragment,{children:ee},C),(0,i.jsx)(P.FocusGuard,{ref:S.context.triggerFocusTargetRef,onFocus:X})]}):(0,i.jsx)(o.Fragment,{children:ee},C)});var M=e.i(726674);let z=o.createContext(void 0),D=o.forwardRef(function(e,t){let{keepMounted:o=!1,...a}=e,{store:n}=p();return n.useState("mounted")||o?(0,i.jsx)(z.Provider,{value:o,children:(0,i.jsx)(M.FloatingPortal,{ref:t,...a})}):null});var L=e.i(144394),H=e.i(146376);let F=o.createContext(void 0);function B(){let e=o.useContext(F);if(!e)throw Error((0,s.default)(46));return e}var G=e.i(329365),U=e.i(426),V=e.i(222640),W=e.i(360495),q=e.i(789579),K=e.i(33383);let Y=o.forwardRef(function(e,t){let{render:a,className:n,style:l,anchor:d,positionMethod:u="absolute",side:c="bottom",align:g="center",sideOffset:f=0,alignOffset:h=0,collisionBoundary:x="clipping-ancestors",collisionPadding:b=5,arrowPadding:y=5,sticky:v=!1,disableAnchorTracking:S=!1,collisionAvoidance:C=_.POPUP_COLLISION_AVOIDANCE,...j}=e,{store:E}=p(),I=function(){let e=o.useContext(z);if(void 0===e)throw Error((0,s.default)(45));return e}(),k=(0,r.useFloatingNodeId)(),w=E.useState("floatingRootContext"),R=E.useState("mounted"),O=E.useState("open"),T=E.useState("openChangeReason"),P=E.useState("activeTriggerElement"),A=E.useState("modal"),N=E.useState("openMethod"),$=E.useState("positionerElement"),M=E.useState("instantType"),D=E.useState("transitionStatus"),B=E.useState("hasViewport"),Y=o.useRef(null),Z=(0,V.useAnimationsFinished)($,!1,!1),J=(0,G.useAnchorPositioning)({anchor:d,floatingRootContext:w,positionMethod:u,mounted:R,side:c,sideOffset:f,align:g,alignOffset:h,arrowPadding:y,collisionBoundary:x,collisionPadding:b,sticky:v,disableAnchorTracking:S,keepMounted:I,nodeId:k,collisionAvoidance:C,adaptiveOrigin:B?W.adaptiveOrigin:void 0}),Q=w.useState("domReferenceElement");(0,H.useIsoLayoutEffect)(()=>{let e=Y.current;if(Q&&(Y.current=Q),e&&Q&&Q!==e){E.set("instantType",void 0);let e=new AbortController;return Z(()=>{E.set("instantType","trigger-change")},e.signal),()=>{e.abort()}}},[Q,Z,E]),(0,K.useAnchoredPopupScrollLock)(O&&!0===A&&T!==m.REASONS.triggerHover,"touch"===N,$,P);let X=o.useCallback(e=>{E.set("positionerElement",e)},[E]),ee={open:O,side:J.side,align:J.align,anchorHidden:J.anchorHidden,instant:M},et=(0,q.usePositioner)(e,ee,{styles:J.positionerStyles,transitionStatus:D,props:j,refs:[t,X],hidden:!R,inert:!O});return(0,i.jsxs)(F.Provider,{value:J,children:[R&&!0===A&&T!==m.REASONS.triggerHover&&(0,i.jsx)(U.InternalBackdrop,{ref:E.context.internalBackdropRef,inert:(0,L.inertValue)(!O),cutout:P}),(0,i.jsx)(r.FloatingNode,{id:k,children:et})]})});var Z=e.i(229315),J=e.i(61487),Q=e.i(431157),X=e.i(209407),ee=e.i(137584),et=e.i(673327),ei=e.i(96533),eo=e.i(815982),ea=e.i(667865);let en=o.createContext(void 0);function er(e){let{value:t,children:o}=e;return(0,i.jsx)(en.Provider,{value:t,children:o})}let es={...I.popupStateMapping,...X.transitionStatusMapping},el=o.forwardRef(function(e,t){let{render:a,className:n,style:r,initialFocus:s,finalFocus:l,...d}=e,{store:u}=p(),c=B(),g=null!=(0,ei.useToolbarRootContext)(!0),{context:h,hasClosePart:x}=function(){let[e,t]=o.useState(0),i=(0,ea.useStableCallback)(()=>(t(e=>e+1),()=>{t(e=>Math.max(0,e-1))}));return{context:o.useMemo(()=>({register:i}),[i]),hasClosePart:e>0}}(),_=u.useState("open"),b=u.useState("openMethod"),y=u.useState("instantType"),v=u.useState("transitionStatus"),S=u.useState("popupProps"),C=u.useState("titleElementId"),j=u.useState("descriptionElementId"),E=u.useState("modal"),I=u.useState("mounted"),w=u.useState("openChangeReason"),R=u.useState("activeTriggerElement"),O=u.useState("floatingRootContext"),T=O.useState("floatingId"),P=u.useState("disabled"),A=u.useState("openOnHover"),N=u.useState("closeDelay"),$=d.id??T;(0,ee.useOpenChangeComplete)({open:_,ref:u.context.popupRef,onComplete(){_&&u.context.onOpenChangeComplete?.(!0)}}),(0,Q.useHoverFloatingInteraction)(O,{enabled:A&&!P,closeDelay:N});let M=void 0===s?(0,f.createDefaultInitialFocus)(u.context.popupRef):s,z=!1!==E&&x;u.useSyncedValue("focusManagerModal",z);let D=o.useCallback(e=>{u.set("popupElement",e)},[u]),L={open:_,side:c.side,align:c.align,instant:y,transitionStatus:v},H=(0,k.useRenderElement)("div",e,{state:L,ref:[t,u.context.popupRef,D],props:[S,{id:$,role:"dialog",...f.FOCUSABLE_POPUP_PROPS,"aria-labelledby":C,"aria-describedby":j,onKeyDown(e){g&&et.COMPOSITE_KEYS.has(e.key)&&e.stopPropagation()}},(0,eo.getDisabledMountTransitionStyles)(v),d],stateAttributesMapping:es});return(0,i.jsx)(J.FloatingFocusManager,{context:O,openInteractionType:b,modal:z,disabled:!I||w===m.REASONS.triggerHover,initialFocus:M,returnFocus:l,restoreFocus:"popup",previousFocusableElement:(0,Z.isHTMLElement)(R)?R:void 0,nextFocusableElement:u.context.triggerFocusTargetRef,beforeContentFocusGuardRef:u.context.beforeContentFocusGuardRef,children:(0,i.jsx)(er,{value:h,children:H})})}),ep=o.forwardRef(function(e,t){let{render:i,className:o,style:a,...n}=e,{store:r}=p(),s=r.useState("open"),{arrowRef:l,side:d,align:u,arrowUncentered:c,arrowStyles:g}=B();return(0,k.useRenderElement)("div",e,{state:{open:s,side:d,align:u,uncentered:c},ref:[t,l],props:[{style:g,"aria-hidden":!0},n],stateAttributesMapping:I.popupStateMapping})}),ed={...I.popupStateMapping,...X.transitionStatusMapping},eu=o.forwardRef(function(e,t){let{render:i,className:o,style:a,...n}=e,{store:r}=p(),s=r.useState("open"),l=r.useState("mounted"),d=r.useState("transitionStatus"),u=r.useState("openChangeReason");return(0,k.useRenderElement)("div",e,{state:{open:s,transitionStatus:d},ref:[r.context.backdropRef,t],props:[{role:"presentation",hidden:!l,style:{pointerEvents:u===m.REASONS.triggerHover?"none":void 0,userSelect:"none",WebkitUserSelect:"none"}},n],stateAttributesMapping:ed})}),ec=o.forwardRef(function(e,t){let{render:i,className:o,style:a,...n}=e,{store:r}=p(),s=(0,T.useBaseUiId)(n.id);return r.useSyncedValueWithCleanup("titleElementId",s),(0,k.useRenderElement)("h2",e,{ref:t,props:[{id:s},n]})}),eg=o.forwardRef(function(e,t){let{render:i,className:o,style:a,...n}=e,{store:r}=p(),s=(0,T.useBaseUiId)(n.id);return r.useSyncedValueWithCleanup("descriptionElementId",s),(0,k.useRenderElement)("p",e,{ref:t,props:[{id:s},n]})}),em=o.forwardRef(function(e,t){let i,{render:a,className:n,style:r,disabled:s=!1,nativeButton:l=!0,...d}=e,{buttonRef:u,getButtonProps:c}=(0,E.useButton)({disabled:s,focusableWhenDisabled:!1,native:l}),{store:g}=p();return i=o.useContext(en),(0,H.useIsoLayoutEffect)(()=>i?.register(),[i]),(0,k.useRenderElement)("button",e,{ref:[t,u],props:[{onClick(e){g.setOpen(!1,(0,v.createChangeEventDetails)(m.REASONS.closePress,e.nativeEvent))}},d,c]})}),ef=((t={}).popupWidth="--popup-width",t.popupHeight="--popup-height",t);var eh=e.i(818390);let ex={activationDirection:e=>e?{"data-activation-direction":e}:null},e_=o.forwardRef(function(e,t){let{render:i,className:o,style:a,children:n,...r}=e,{store:s}=p(),{side:l}=B(),d=s.useState("instantType"),{children:u,state:c}=(0,eh.usePopupViewport)({store:s,side:l,cssVars:ef,children:n}),g={activationDirection:c.activationDirection,transitioning:c.transitioning,instant:d};return(0,k.useRenderElement)("div",e,{state:g,ref:t,props:[r,{children:u}],stateAttributesMapping:ex})});class eb{constructor(){this.store=new y}open(e){let t=e?this.store.context.triggerElements.getById(e)??void 0:void 0;if(e&&!t)throw Error((0,s.default)(80,e));this.store.setOpen(!0,(0,v.createChangeEventDetails)(m.REASONS.imperativeAction,void 0,t))}close(){this.store.setOpen(!1,(0,v.createChangeEventDetails)(m.REASONS.imperativeAction,void 0,void 0))}get isOpen(){return this.store.select("open")}}e.s(["Arrow",0,ep,"Backdrop",0,eu,"Close",0,em,"Description",0,eg,"Handle",0,eb,"Popup",0,el,"Portal",0,D,"Positioner",0,Y,"Root",0,function(e){return p(!0)?(0,i.jsx)(C,{props:e}):(0,i.jsx)(r.FloatingTree,{children:(0,i.jsx)(C,{props:e})})},"Title",0,ec,"Trigger",0,$,"Viewport",0,e_,"createHandle",0,function(){return new eb}],466914);var ey=e.i(466914),ey=ey,ev=e.i(115504);e.s(["Popover",0,function({...e}){return(0,i.jsx)(ey.Root,{"data-slot":"popover",...e})},"PopoverContent",0,function({className:e,align:t="center",alignOffset:o=0,side:a="bottom",sideOffset:n=4,...r}){return(0,i.jsx)(ey.Portal,{children:(0,i.jsx)(ey.Positioner,{align:t,alignOffset:o,side:a,sideOffset:n,className:"isolate z-50",children:(0,i.jsx)(ey.Popup,{"data-slot":"popover-content",className:(0,ev.cn)("z-50 flex w-72 origin-(--transform-origin) flex-col gap-4 rounded-md bg-popover p-4 text-sm text-popover-foreground shadow-md ring-1 ring-foreground/10 outline-hidden duration-100 data-[side=bottom]:slide-in-from-top-2 data-[side=inline-end]:slide-in-from-left-2 data-[side=inline-start]:slide-in-from-right-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95",e),...r})})})},"PopoverDescription",0,function({className:e,...t}){return(0,i.jsx)(ey.Description,{"data-slot":"popover-description",className:(0,ev.cn)("text-muted-foreground",e),...t})},"PopoverTitle",0,function({className:e,...t}){return(0,i.jsx)(ey.Title,{"data-slot":"popover-title",className:(0,ev.cn)("font-medium",e),...t})},"PopoverTrigger",0,function({...e}){return(0,i.jsx)(ey.Trigger,{"data-slot":"popover-trigger",...e})}],337822)},284614,e=>{"use strict";let t=(0,e.i(475254).default)("user",[["path",{d:"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2",key:"975kel"}],["circle",{cx:"12",cy:"7",r:"4",key:"17ys0d"}]]);e.s(["User",0,t],284614)},581418,e=>{"use strict";let t=(0,e.i(475254).default)("shield-check",[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]]);e.s(["ShieldCheck",0,t],581418)},292639,e=>{"use strict";var t=e.i(602869),i=e.i(266027);let o=(0,e.i(243652).createQueryKeys)("uiSettings");e.s(["useUISettings",0,e=>(0,i.useQuery)({queryKey:o.list({}),queryFn:async()=>await (0,t.getUiSettings)(),staleTime:e?.staleTime??36e5,gcTime:36e5,refetchInterval:e?.refetchInterval})])},922407,e=>{"use strict";var t=e.i(843476),i=e.i(519455),o=e.i(115504),a=e.i(643531),n=e.i(174886),r=e.i(271645);e.s(["default",0,({value:e,label:s,className:l,iconClassName:p="size-[15px]"})=>{let[d,u]=(0,r.useState)(!1);if((0,r.useEffect)(()=>{if(!d)return;let e=setTimeout(()=>u(!1),1200);return()=>clearTimeout(e)},[d]),!e)return null;let c=async()=>{if(navigator.clipboard)try{await navigator.clipboard.writeText(e),u(!0)}catch{u(!1)}};return(0,t.jsx)(i.Button,{type:"button",variant:"ghost",size:"icon-xs",onClick:c,"aria-label":s,title:s,className:(0,o.cn)("text-muted-foreground hover:text-primary",l),children:d?(0,t.jsx)(a.Check,{className:p}):(0,t.jsx)(n.Copy,{className:p})})}])},571353,e=>{"use strict";e.i(602869);var t=e.i(221688);let i={"api-keys":"api-keys",models:"models-and-endpoints",api_ref:"api-reference","api-reference":"api-reference","llm-playground":"playground",projects:"projects",chat:"chat","access-groups":"access-groups",budgets:"budgets",workflows:"workflows","guardrails-monitor":"guardrails-monitor","mcp-servers":"mcp-servers","search-tools":"search-tools","tag-management":"tag-management","vector-stores":"vector-stores",memory:"memory",policies:"policies",guardrails:"guardrails",prompts:"prompts","tool-policies":"tool-policies",skills:"skills","claude-code-plugins":"skills",caching:"caching","cost-tracking":"cost-tracking","transform-request":"transform-request","ui-theme":"ui-theme",logs:"logs","admin-panel":"admin-panel","logging-and-alerts":"logging-and-alerts","model-hub-table":"model-hub-table",new_usage:"usage",usage:"old-usage","cost-optimization":"cost-optimization",agents:"agents","router-settings":"router-settings",users:"users",teams:"teams",organizations:"organizations"};function o(){let e=t.serverRootPath&&"/"!==t.serverRootPath?`/${t.serverRootPath.replace(/^\/+|\/+$/g,"")}`:"";return`${e}/ui`}e.s(["MIGRATED_PAGES",0,i,"legacyKeyForPathname",0,function(e){let t=o(),a=(e.startsWith(t)?e.slice(t.length):e).replace(/^\/+|\/+$/g,"");for(let[e,t]of Object.entries(i))if(a===t)return e;return null},"legacyPageHref",0,function(e){return`${o()}/?page=${e}`},"migratedHref",0,function(e){return`${o()}/${e.replace(/^\/+/,"")}`}])},434626,e=>{"use strict";var t=e.i(271645);let i=t.forwardRef(function(e,i){return t.createElement("svg",Object.assign({xmlns:"http://www.w3.org/2000/svg",fill:"none",viewBox:"0 0 24 24",strokeWidth:2,stroke:"currentColor","aria-hidden":"true",ref:i},e),t.createElement("path",{strokeLinecap:"round",strokeLinejoin:"round",d:"M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"}))});e.s(["ExternalLinkIcon",0,i],434626)},909947,865361,e=>{"use strict";var t,i,o=((t={}).AUDIO_SPEECH="audio_speech",t.AUDIO_TRANSCRIPTION="audio_transcription",t.IMAGE_GENERATION="image_generation",t.VIDEO_GENERATION="video_generation",t.CHAT="chat",t.COMPLETION="completion",t.RESPONSES="responses",t.IMAGE_EDITS="image_edit",t.ANTHROPIC_MESSAGES="anthropic_messages",t.EMBEDDING="embedding",t.REALTIME="realtime",t),a=((i={}).IMAGE="image",i.VIDEO="video",i.CHAT="chat",i.RESPONSES="responses",i.IMAGE_EDITS="image_edits",i.ANTHROPIC_MESSAGES="anthropic_messages",i.EMBEDDINGS="embeddings",i.SPEECH="speech",i.TRANSCRIPTION="transcription",i.A2A_AGENTS="a2a_agents",i.MCP="mcp",i.REALTIME="realtime",i.INTERACTIONS="interactions",i);let n={image_generation:"image",video_generation:"video",chat:"chat",completion:"chat",responses:"responses",image_edit:"image_edits",anthropic_messages:"anthropic_messages",audio_speech:"speech",audio_transcription:"transcription",embedding:"embeddings",realtime:"realtime"};e.s(["EndpointType",()=>a,"ModelMode",()=>o,"getEndpointType",0,e=>Object.values(o).includes(e)?n[e]:"chat"],865361),e.s(["generateCodeSnippet",0,e=>{let t,{apiKeySource:i,accessToken:o,apiKey:n,inputMessage:r,chatHistory:s,selectedTags:l,selectedVectorStores:p,selectedGuardrails:d,selectedPolicies:u,selectedVoice:c,endpointType:g,selectedModel:m,selectedSdk:f,proxySettings:h}=e,x="session"===i?o:n,_=window.location.origin,b=h?.LITELLM_UI_API_DOC_BASE_URL;b&&b.trim()?_=b:h?.PROXY_BASE_URL&&(_=h.PROXY_BASE_URL);let y=r||"Your prompt here",v=y.replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,"\\n"),S=s.filter(e=>!e.isImage).map(({role:e,content:t})=>({role:e,content:t})),C={};l.length>0&&(C.tags=l),p.length>0&&(C.vector_stores=p),d.length>0&&(C.guardrails=d),u.length>0&&(C.policies=u);let j=m||"your-model-name",E="azure"===f?`import openai

client = openai.AzureOpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	azure_endpoint="${_}",
	api_version="2024-02-01"
)`:`import openai

client = openai.OpenAI(
	api_key="${x||"YOUR_LITELLM_API_KEY"}",
	base_url="${_}"
)`;switch(g){case a.CHAT:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=S.length>0?S:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.chat.completions.create(
    model="${j}",
    messages=${JSON.stringify(o,null,4)}${i}
)

print(response)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.chat.completions.create(
#     model="${j}",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "${v}"
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
`;break}case a.RESPONSES:{let e=Object.keys(C).length>0,i="";if(e){let e=JSON.stringify({metadata:C},null,2).split("\n").map(e=>" ".repeat(4)+e).join("\n").trim();i=`,
    extra_body=${e}`}let o=S.length>0?S:[{role:"user",content:y}];t=`
import base64

# Helper function to encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example with text only
response = client.responses.create(
    model="${j}",
    input=${JSON.stringify(o,null,4)}${i}
)

print(response.output_text)

# Example with image or PDF (uncomment and provide file path to use)
# base64_file = encode_image("path/to/your/file.jpg")  # or .pdf
# response_with_file = client.responses.create(
#     model="${j}",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "input_text", "text": "${v}"},
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_file}",  # or data:application/pdf;base64,{base64_file}
#                 },
#             ],
#         }
#     ]${i}
# )
# print(response_with_file.output_text)
`;break}case a.IMAGE:t="azure"===f?`
# NOTE: The Azure SDK does not have a direct equivalent to the multi-modal 'responses.create' method shown for OpenAI.
# This snippet uses 'client.images.generate' and will create a new image based on your prompt.
# It does not use the uploaded image, as 'client.images.generate' does not support image inputs in this context.
import os
import requests
import json
import time
from PIL import Image

result = client.images.generate(
	model="${j}",
	prompt="${r}",
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${j}",
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
`;break;case a.IMAGE_EDITS:t="azure"===f?`
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${j}",
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
prompt = "${v}"

# Encode images to base64
base64_image1 = encode_image("body-lotion.png")
base64_image2 = encode_image("soap.png")

# Create file IDs
file_id1 = create_file("body-lotion.png")
file_id2 = create_file("incense-kit.png")

response = client.responses.create(
	model="${j}",
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
	input="${r||"Your string here"}",
	model="${j}",
	encoding_format="base64" # or "float"
)

print(response.data[0].embedding)
`;break;case a.TRANSCRIPTION:t=`
# Open the audio file
audio_file = open("path/to/your/audio/file.mp3", "rb")

# Make the transcription request
response = client.audio.transcriptions.create(
	model="${j}",
	file=audio_file${r?`,
	prompt="${r.replace(/\\/g,"\\\\").replace(/"/g,'\\"')}"`:""}
)

print(response.text)
`;break;case a.SPEECH:t=`
# Make the text-to-speech request
response = client.audio.speech.create(
	model="${j}",
	input="${r||"Your text to convert to speech here"}",
	voice="${c}"  # Options: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer
)

# Save the audio to a file
output_filename = "output_speech.mp3"
response.stream_to_file(output_filename)
print(f"Audio saved to {output_filename}")

# Optional: Customize response format and speed
# response = client.audio.speech.create(
#     model="${j}",
#     input="${r||"Your text to convert to speech here"}",
#     voice="alloy",
#     response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
#     speed=1.0  # Range: 0.25 to 4.0
# )
# response.stream_to_file("output_speech.mp3")
`;break;default:t="\n# Code generation for this endpoint is not implemented yet."}return`${E}
${t}`}],909947)},306228,e=>{"use strict";let t=(0,e.i(475254).default)("link-2",[["path",{d:"M9 17H7A5 5 0 0 1 7 7h2",key:"8i5ue5"}],["path",{d:"M15 7h2a5 5 0 1 1 0 10h-2",key:"1b9ql8"}],["line",{x1:"8",x2:"16",y1:"12",y2:"12",key:"1jonct"}]]);e.s(["Link2",0,t],306228)},652272,209261,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(871689),a=e.i(643531),n=e.i(174886),r=e.i(306228);let s=/^[a-zA-Z0-9][a-zA-Z0-9._-]*(\/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$/,l=e=>e.trim().replace(/\/+$/,""),p=/\.(md|markdown|txt|json|ya?ml|toml)$/i,d=/^\d{1,3}(\.\d{1,3}){3}$/,u=/^[A-Za-z0-9-]+$/,c=/^[A-Za-z0-9._-]+$/,g=e=>e.pathname.split("/").filter(e=>""!==e),m=e=>{let t=e.split("/").filter(e=>""!==e);return t[t.length-1]??""},f=e=>e.toLowerCase().replace(/[^a-z0-9-]+/g,"-").replace(/-+/g,"-").replace(/^-+|-+$/g,""),h=e=>JSON.stringify({extraKnownMarketplaces:{"my-org":{source:{source:"url",url:`${e}/claude-code/marketplace.json`}}}},null,2),x=e=>{let{source:t}=e;return"github"===t.source&&t.repo?`/plugin marketplace add ${t.repo}`:("url"===t.source||"git-subdir"===t.source)&&t.url?`/plugin marketplace add ${t.url}`:`/plugin marketplace add ${e.name}`};e.s(["buildMarketplaceSettingsSnippet",0,h,"formatInstallCommand",0,x,"getCategoryBadgeColor",0,e=>{if(!e)return"gray";let t=e.toLowerCase();if(t.includes("development")||t.includes("dev"))return"blue";if(t.includes("productivity")||t.includes("workflow"))return"green";if(t.includes("learning")||t.includes("education"))return"purple";if(t.includes("security")||t.includes("safety"))return"red";if(t.includes("data")||t.includes("analytics"))return"orange";else if(t.includes("integration")||t.includes("api"))return"yellow";return"gray"},"isValidEmail",0,e=>!e||/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e),"isValidSemanticVersion",0,e=>!e||/^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$/.test(e),"isValidSubPath",0,e=>{let t=l(e);return""!==t&&s.test(t)},"parseKeywords",0,e=>e&&""!==e.trim()?e.split(",").map(e=>e.trim()).filter(e=>""!==e):[],"parseSkillSource",0,(e,t)=>{let i=(e=>{let t,i=e.trim();if(""===i||i.startsWith("//"))return null;let o=/^[a-z][a-z0-9+.-]*:\/\//i.test(i)?i:`https://${i}`;try{t=new URL(o)}catch{return null}return"https:"!==t.protocol||""!==t.username||""!==t.password||!t.hostname.includes(".")||t.hostname.startsWith("[")||d.test(t.hostname)?null:t})(e);if(!i)return null;if("github.com"===i.hostname.replace(/^www\./,""))return((e,t)=>{let i=g(e);if(i.length<2)return null;let o=i[0],a=i[1].replace(/\.git$/,"");if(!u.test(o)||!c.test(a))return null;let n=`${o}/${a}`,r=`https://github.com/${n}`,d={parsed:{source:"github",repo:n},label:`GitHub repo — ${n}`,suggestedName:f(a)};if(i.length>=4&&("tree"===i[2]||"blob"===i[2])){let e=i.slice(4),t=m(e.join("/")),o=p.test(t)?e.slice(0,-1):e;if(0===o.length)return d;let a=l(o.join("/"));return s.test(a)?{parsed:{source:"git-subdir",url:r,path:a},label:`GitHub subdir — ${n} @ ${a}`,suggestedName:f(m(a))}:null}if(2!==i.length)return null;let h=l(t??"");return""!==h?s.test(h)?{parsed:{source:"git-subdir",url:r,path:h},label:`GitHub subdir — ${n} @ ${h}`,suggestedName:f(m(h))}:null:d})(i,t);if(g(i).length<2)return null;let o=`${i.protocol}//${i.host}${i.pathname.replace(/\/+$/,"")}`,a=l(t??"");return""!==a?s.test(a)?{parsed:{source:"git-subdir",url:o,path:a},label:`Git subdir — ${o} @ ${a}`,suggestedName:f(m(a))}:null:{parsed:{source:"url",url:o},label:`Git repo — ${o}`,suggestedName:f(m(i.pathname).replace(/\.git$/,""))}},"validatePluginName",0,e=>!!e&&""!==e.trim()&&/^[a-z0-9-]+$/.test(e)],209261),e.s(["default",0,({skill:e,onBack:s})=>{let l,[p,d]=(0,i.useState)("overview"),[u,c]=(0,i.useState)(null),g=(e,t)=>{navigator.clipboard.writeText(e),c(t),setTimeout(()=>c(null),2e3)},m="github"===(l=e.source).source&&l.repo?`https://github.com/${l.repo}`:"git-subdir"===l.source&&l.url?l.path?`${l.url}/tree/main/${l.path}`:l.url:"url"===l.source&&l.url?l.url:null,f=x(e),_=h(window.location.origin),b=[...e.category?[{property:"Category",value:e.category}]:[],...e.domain?[{property:"Domain",value:e.domain}]:[],...e.namespace?[{property:"Namespace",value:e.namespace}]:[],...e.version?[{property:"Version",value:e.version}]:[],...e.author?.name?[{property:"Author",value:e.author.name}]:[],...e.created_at?[{property:"Added",value:new Date(e.created_at).toLocaleDateString()}]:[]];return(0,t.jsxs)("div",{style:{padding:"24px 32px 24px 0"},children:[(0,t.jsxs)("div",{onClick:s,style:{display:"inline-flex",alignItems:"center",gap:6,color:"#5f6368",cursor:"pointer",fontSize:14,marginBottom:24},children:[(0,t.jsx)(o.ArrowLeft,{className:"size-3"}),(0,t.jsx)("span",{children:"Skills"})]}),(0,t.jsxs)("div",{style:{marginBottom:8},children:[(0,t.jsx)("h1",{style:{fontSize:28,fontWeight:400,color:"#202124",margin:0,lineHeight:1.2},children:e.name}),e.description&&(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"8px 0 0 0",lineHeight:1.6},children:e.description})]}),(0,t.jsx)("div",{style:{borderBottom:"1px solid #dadce0",marginBottom:28,marginTop:24},children:(0,t.jsx)("div",{style:{display:"flex",gap:0},children:[{key:"overview",label:"Overview"},{key:"usage",label:"How to Use"}].map(e=>(0,t.jsx)("div",{onClick:()=>d(e.key),style:{padding:"12px 20px",fontSize:14,color:p===e.key?"#1a73e8":"#5f6368",borderBottom:p===e.key?"3px solid #1a73e8":"3px solid transparent",cursor:"pointer",fontWeight:p===e.key?500:400,marginBottom:-1},children:e.label},e.key))})}),"overview"===p&&(0,t.jsxs)("div",{style:{display:"flex",gap:64},children:[(0,t.jsxs)("div",{style:{flex:1,minWidth:0},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 4px 0"},children:"Skill Details"}),(0,t.jsx)("p",{style:{fontSize:13,color:"#5f6368",margin:"0 0 16px 0"},children:"Metadata registered with this skill"}),(0,t.jsxs)("table",{style:{width:"100%",borderCollapse:"collapse",fontSize:14},children:[(0,t.jsx)("thead",{children:(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500,width:160},children:"Property"}),(0,t.jsx)("th",{style:{textAlign:"left",padding:"12px 0",color:"#5f6368",fontWeight:500},children:e.name})]})}),(0,t.jsx)("tbody",{children:b.map((e,i)=>(0,t.jsxs)("tr",{style:{borderBottom:"1px solid #f1f3f4"},children:[(0,t.jsx)("td",{style:{padding:"12px 0",color:"#3c4043"},children:e.property}),(0,t.jsx)("td",{style:{padding:"12px 0",color:"#202124"},children:e.value})]},i))})]})]}),(0,t.jsxs)("div",{style:{width:240,flexShrink:0},children:[(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Status"}),(0,t.jsx)("span",{style:{fontSize:12,padding:"3px 10px",borderRadius:12,backgroundColor:e.enabled?"#e6f4ea":"#f1f3f4",color:e.enabled?"#137333":"#5f6368",fontWeight:500},children:e.enabled?"Public":"Draft"})]}),m&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Source"}),(0,t.jsxs)("a",{href:m,target:"_blank",rel:"noopener noreferrer",style:{fontSize:13,color:"#1a73e8",wordBreak:"break-all",display:"flex",alignItems:"center",gap:4},children:[m.replace("https://",""),(0,t.jsx)(r.Link2,{className:"size-3 shrink-0"})]})]}),e.keywords&&e.keywords.length>0&&(0,t.jsxs)("div",{style:{marginBottom:24},children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:8},children:"Tags"}),(0,t.jsx)("div",{style:{display:"flex",flexWrap:"wrap",gap:6},children:e.keywords.map(e=>(0,t.jsx)("span",{style:{fontSize:12,padding:"4px 12px",borderRadius:16,border:"1px solid #dadce0",color:"#3c4043",backgroundColor:"#fff"},children:e},e))})]}),(0,t.jsxs)("div",{children:[(0,t.jsx)("div",{style:{fontSize:12,color:"#5f6368",marginBottom:4},children:"Skill ID"}),(0,t.jsx)("div",{style:{fontSize:12,fontFamily:"monospace",color:"#3c4043",wordBreak:"break-all"},children:e.id})]})]})]}),"usage"===p&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"Using this skill"}),(0,t.jsx)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:"Once your proxy is set as a marketplace, enable this skill in Claude Code with one command:"}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden",marginBottom:24},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"Run in Claude Code"}),(0,t.jsxs)("button",{onClick:()=>g(f,"install"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"install"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["install"===u?(0,t.jsx)(a.Check,{className:"size-3"}):(0,t.jsx)(n.Copy,{className:"size-3"}),"install"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:14,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:f})]}),(0,t.jsxs)("p",{style:{fontSize:13,color:"#5f6368",lineHeight:1.6,margin:0},children:["Don't have the marketplace configured yet?"," ",(0,t.jsx)("span",{onClick:()=>d("setup"),style:{color:"#1a73e8",cursor:"pointer"},children:"See one-time setup →"})]})]}),"setup"===p&&(0,t.jsxs)("div",{style:{maxWidth:640},children:[(0,t.jsx)("h2",{style:{fontSize:18,fontWeight:400,color:"#202124",margin:"0 0 8px 0"},children:"One-time marketplace setup"}),(0,t.jsxs)("p",{style:{fontSize:14,color:"#5f6368",margin:"0 0 24px 0",lineHeight:1.6},children:["Add this to"," ",(0,t.jsx)("code",{style:{fontSize:13,backgroundColor:"#f1f3f4",padding:"1px 6px",borderRadius:4},children:"~/.claude/settings.json"})," ","to point Claude Code at your proxy:"]}),(0,t.jsxs)("div",{style:{border:"1px solid #dadce0",borderRadius:8,overflow:"hidden"},children:[(0,t.jsxs)("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"10px 16px",backgroundColor:"#f8f9fa",borderBottom:"1px solid #dadce0"},children:[(0,t.jsx)("span",{style:{fontSize:13,color:"#3c4043",fontWeight:500},children:"~/.claude/settings.json"}),(0,t.jsxs)("button",{onClick:()=>g(_,"settings"),style:{display:"flex",alignItems:"center",gap:4,fontSize:12,color:"settings"===u?"#137333":"#1a73e8",background:"none",border:"none",cursor:"pointer",padding:0},children:["settings"===u?(0,t.jsx)(a.Check,{className:"size-3"}):(0,t.jsx)(n.Copy,{className:"size-3"}),"settings"===u?"Copied":"Copy"]})]}),(0,t.jsx)("pre",{style:{margin:0,padding:"14px 16px",fontSize:13,fontFamily:"monospace",color:"#202124",backgroundColor:"#fff"},children:_})]})]})]})}],652272)},560280,e=>{"use strict";var t=e.i(843476),i=e.i(271645),o=e.i(618566),a=e.i(976883);function n(){let e=(0,o.useSearchParams)().get("key"),[n,r]=(0,i.useState)(null);return(0,i.useEffect)(()=>{e&&r(e)},[e]),(0,t.jsx)(a.default,{accessToken:n})}e.s(["default",0,function(){return(0,t.jsx)(i.Suspense,{fallback:(0,t.jsx)("div",{className:"flex items-center justify-center min-h-screen",children:"Loading..."}),children:(0,t.jsx)(n,{})})}])}]);