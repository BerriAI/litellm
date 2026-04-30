import React from 'react';

const s = {
  fig: {margin: '2.5rem 0', fontFamily: 'inherit'},
  box: {borderRadius: 12, border: '1px solid #e5e7eb', background: '#fff', padding: '2rem 2.5rem'},
  label: {fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.12em', color: '#9ca3af', textAlign: 'center', marginBottom: '1.5rem'},
  caption: {textAlign: 'center', fontSize: 12, color: '#9ca3af', marginTop: 12},
  node: (border='#d1d5db', bg='#f9fafb') => ({
    border: `1px solid ${border}`, borderRadius: 6, padding: '8px 20px',
    fontSize: 13, background: bg, display: 'inline-block',
  }),
  arrow: {display: 'flex', flexDirection: 'column', alignItems: 'center'},
};

const SmallArrow = ({color='#9ca3af'}) => (
  <svg width="2" height="28" style={{display:'block'}}>
    <line x1="1" y1="0" x2="1" y2="22" stroke={color} strokeWidth="1.5"/>
    <polygon points="1,28 -2,21 4,21" fill={color}/>
  </svg>
);

export function CascadeFailure() {
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Without circuit breaker — cascade failure</p>
        <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:0}}>
          <div style={s.node()}>LiteLLM Pod (×100)</div>
          <SmallArrow />
          <div style={s.node()}>Rate limit / cache check</div>
          <div style={{position:'relative', display:'flex', flexDirection:'column', alignItems:'center'}}>
            <SmallArrow color="#f87171"/>
            <span style={{position:'absolute', left:8, top:4, fontSize:11, color:'#f87171', whiteSpace:'nowrap'}}>hangs 30s per request</span>
          </div>
          <div style={s.node('#fca5a5','#fef2f2')}><span style={{color:'#b91c1c', fontWeight:600}}>Redis — degraded, timing out</span></div>
          <SmallArrow color="#fb923c"/>
          <div style={s.node('#fdba74','#fff7ed')}><span style={{color:'#c2410c', fontWeight:600}}>Postgres — 100× normal read load</span></div>
          <SmallArrow />
          <div style={{...s.node('#111827','#111827'), color:'#fff', fontWeight:600}}>Total outage — gateway down</div>
        </div>
      </div>
      <figcaption style={s.caption}>Slow Redis → every auth check times out → database overwhelmed → full cascade</figcaption>
    </figure>
  );
}

export function CircuitBreakerStates() {
  const circle = (border, color, label, sub) => (
    <div style={{display:'flex', flexDirection:'column', alignItems:'center', width: 140}}>
      <div style={{width:88, height:88, borderRadius:'50%', border:`2px solid ${border}`, background:'#fff', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center'}}>
        <span style={{fontSize:11, fontWeight:700, color, letterSpacing:'0.06em'}}>{label}</span>
        <span style={{fontSize:10, color:'#9ca3af', marginTop:2}}>{sub}</span>
      </div>
      <p style={{fontSize:11, color:'#6b7280', textAlign:'center', marginTop:10, lineHeight:1.5}}>{'\u00a0'}</p>
    </div>
  );
  const arrow = (label) => (
    <div style={{display:'flex', flexDirection:'column', alignItems:'center', marginTop:36, marginLeft:4, marginRight:4}}>
      <span style={{fontSize:10, color:'#6b7280', marginBottom:4}}>{label}</span>
      <div style={{display:'flex', alignItems:'center'}}>
        <div style={{height:1, width:48, background:'#9ca3af'}}/>
        <svg width="8" height="8" style={{marginLeft:-1}}><polygon points="0,0 8,4 0,8" fill="#6b7280"/></svg>
      </div>
    </div>
  );
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Circuit breaker state machine</p>
        <div style={{display:'flex', justifyContent:'center', alignItems:'flex-start'}}>
          {circle('#1f2937','#111827','CLOSED','normal')}
          {arrow('5 failures')}
          {circle('#f87171','#dc2626','OPEN','fast-fail')}
          {arrow('60s timeout')}
          {circle('#fbbf24','#b45309','HALF-OPEN','probing')}
        </div>
        <div style={{display:'flex', justifyContent:'center', gap:32, marginTop:24}}>
          <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:4}}>
            <div style={{display:'flex', alignItems:'center', gap:4}}>
              <svg width="8" height="8"><polygon points="8,0 0,4 8,8" fill="#16a34a"/></svg>
              <div style={{height:1, width:100, background:'#16a34a'}}/>
            </div>
            <span style={{fontSize:10, color:'#16a34a'}}>probe success → CLOSED</span>
          </div>
          <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:4}}>
            <div style={{display:'flex', alignItems:'center', gap:4}}>
              <svg width="8" height="8"><polygon points="8,0 0,4 8,8" fill="#ef4444"/></svg>
              <div style={{height:1, width:100, borderTop:'2px dashed #f87171'}}/>
            </div>
            <span style={{fontSize:10, color:'#ef4444'}}>probe failure → OPEN again</span>
          </div>
        </div>
      </div>
    </figure>
  );
}

export function CircuitBreakerFlow() {
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>With circuit breaker — graceful degradation</p>
        <div style={{display:'flex', flexDirection:'column', alignItems:'center'}}>
          <div style={s.node()}>Incoming request</div>
          <SmallArrow />
          <div style={{...s.node('#111827'), border:'2px solid #111827', fontWeight:600}}>Circuit Breaker</div>
          <div style={{display:'flex', gap:80, marginTop:20, alignItems:'flex-start'}}>
            <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:8}}>
              <SmallArrow />
              <span style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.06em', color:'#6b7280', border:'1px solid #e5e7eb', borderRadius:4, padding:'2px 8px'}}>Closed</span>
              <div style={{...s.node(), textAlign:'center', fontSize:13}}>Redis call<br/><span style={{fontSize:11, color:'#9ca3af'}}>normal latency</span></div>
            </div>
            <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:8}}>
              <SmallArrow />
              <span style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.06em', color:'#ef4444', border:'1px solid #fca5a5', borderRadius:4, padding:'2px 8px'}}>Open</span>
              <div style={{...s.node('#fca5a5'), textAlign:'center', fontSize:13}}>Fast-fail — 0ms<br/><span style={{fontSize:11, color:'#9ca3af'}}>no network call</span></div>
              <SmallArrow />
              <div style={{...s.node(), textAlign:'center', fontSize:13}}>DB fallback<br/><span style={{fontSize:11, color:'#9ca3af'}}>bounded load</span></div>
            </div>
          </div>
          <div style={{...s.node('#111827','#111827'), color:'#fff', fontWeight:600, marginTop:24}}>Request completes — gateway stays up</div>
        </div>
      </div>
      <figcaption style={s.caption}>Redis down → circuit opens → 0ms rejection → DB absorbs bounded fallback traffic</figcaption>
    </figure>
  );
}

export function IncidentTimeline() {
  const row = (color, text) => (
    <div style={{display:'flex', alignItems:'flex-start', gap:10, marginBottom:12}}>
      <div style={{marginTop:5, width:6, height:6, borderRadius:'50%', background:color, flexShrink:0}}/>
      <p style={{fontSize:13, color:'#4b5563', margin:0, lineHeight:1.5}}>{text}</p>
    </div>
  );
  return (
    <figure style={s.fig}>
      <div style={s.box}>
        <p style={s.label}>Redis degrades — before vs. after</p>
        <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:20}}>
          <div style={{border:'1px solid #e5e7eb', borderRadius:8, padding:20}}>
            <p style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.1em', color:'#9ca3af', marginBottom:16}}>Without circuit breaker</p>
            {row('#f87171','All 100 pods hang for 30s on each auth check')}
            {row('#f87171','Threadpools fill up, requests queue')}
            {row('#f87171','100× simultaneous DB fallbacks overwhelm Postgres')}
            {row('#f87171','Requires manual intervention to recover')}
          </div>
          <div style={{border:'1px solid #111827', borderRadius:8, padding:20}}>
            <p style={{fontSize:10, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.1em', color:'#9ca3af', marginBottom:16}}>With circuit breaker</p>
            {row('#111827','Circuit opens after 5 failures — 0ms fast-fail')}
            {row('#111827','Auth falls back to DB — bounded, not 100× load')}
            {row('#111827','Cache miss rate temporarily elevated — gateway stays up')}
            {row('#111827','Auto-recovers when Redis comes back — no intervention needed')}
          </div>
        </div>
      </div>
    </figure>
  );
}
