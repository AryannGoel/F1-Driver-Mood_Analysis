import React, { useState, useEffect, useMemo } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine,
} from "recharts";

/* ------------------------------------------------------------------ *
 *  PIT WALL — The Silent Co-Driver
 *  Reads driver stress from team radio and lines it up against lap time.
 *
 *  CONNECTS TO THE BACKEND: on load it fetches API_BASE/api/stint. If the
 *  backend is running it uses that data (demo or live); if not, it falls
 *  back to the embedded stint so the UI always renders.
 *
 *  >>> point this at your backend if it's not on localhost:8000 <<<
 * ------------------------------------------------------------------ */
const API_BASE = "http://localhost:8000";

// fallback stint — used only if the backend can't be reached
const DEMO_STINT = [
  { lap: 28, t: 92.4, stress: 12, mood: "calm",     radio: "Box confirmed, box this lap. Hards fitted." },
  { lap: 29, t: 91.9, stress: 10, mood: "calm",     radio: null },
  { lap: 30, t: 91.2, stress: 11, mood: "calm",     radio: "Tyres switched on. Pace is there." },
  { lap: 31, t: 91.3, stress: 14, mood: "calm",     radio: null },
  { lap: 32, t: 91.5, stress: 24, mood: "focused",  radio: "Car behind is closing. Keep me posted." },
  { lap: 33, t: 91.4, stress: 21, mood: "focused",  radio: null },
  { lap: 34, t: 91.9, stress: 64, mood: "stressed", radio: "I'm losing the rear on entry - it's getting worse." },
  { lap: 35, t: 92.7, stress: 72, mood: "stressed", radio: "No grip at the rear. This is really tough now!" },
  { lap: 36, t: 92.4, stress: 68, mood: "stressed", radio: null },
  { lap: 37, t: 93.1, stress: 76, mood: "tired",    radio: "Front-left's gone. I'm struggling to hold it." },
  { lap: 38, t: 92.9, stress: 79, mood: "tired",    radio: "How many laps left? I can't keep this pace." },
  { lap: 39, t: 93.4, stress: 81, mood: "tired",    radio: null },
  { lap: 40, t: 93.2, stress: 82, mood: "tired",    radio: "Just get me to the end." },
  { lap: 41, t: 93.6, stress: 78, mood: "tired",    radio: null },
  { lap: 42, t: 93.9, stress: 38, mood: "calm",     radio: "Okay - box, box. That's it. Good job, everyone." },
];

const MOOD = {
  calm:     { c: "#37d67a", label: "CALM" },
  focused:  { c: "#35c5e8", label: "FOCUSED" },
  stressed: { c: "#ff9f1c", label: "STRESSED" },
  tired:    { c: "#ff4d4d", label: "TIRED" },
};

const RED = "#e10600";
const INK = "#e8eaed";
const MUTE = "#7b828d";
const PANEL = "#14161b";
const LINE = "#23262e";

const avg = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);

function bars(seed, n = 56) {
  const out = [];
  let x = seed * 9301 + 49297;
  for (let k = 0; k < n; k++) {
    x = (x * 9301 + 49297) % 233280;
    const base = 0.25 + (x / 233280) * 0.75;
    const env = Math.sin((k / n) * Math.PI);
    out.push(Math.max(0.12, base * (0.55 + env * 0.6)));
  }
  return out;
}

export default function PitWall() {
  const [stint, setStint] = useState(DEMO_STINT);
  const [meta, setMeta] = useState(null);
  const [source, setSource] = useState("demo");   // live | demo | offline
  const [sel, setSel] = useState(6);
  const [playing, setPlaying] = useState(false);
  const [head, setHead] = useState(0);
  const [autostint, setAutostint] = useState(false);

  // --- connect to backend on load ---
  useEffect(() => {
    let ok = true;
    fetch(`${API_BASE}/api/stint`)
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d) => { if (!ok) return; setStint(d.laps); setMeta(d.meta); setSource(d.meta?.source || "live"); })
      .catch(() => { if (ok) setSource("offline"); });
    return () => { ok = false; };
  }, []);

  // when the data changes, jump to the first stressed radio call
  useEffect(() => {
    const rl = stint.map((d, i) => ({ ...d, i })).filter((d) => d.radio);
    const hot = rl.find((d) => d.mood === "stressed" || d.mood === "tired") || rl[0];
    if (hot) setSel(hot.i);
  }, [stint]);

  const RADIO = useMemo(() => stint.map((d, i) => ({ ...d, i })).filter((d) => d.radio), [stint]);
  const clamped = Math.min(sel, stint.length - 1);
  const cur = stint[clamped] || stint[0];
  const mood = MOOD[cur.mood] || MOOD.calm;
  const wf = useMemo(() => bars(cur.lap), [cur.lap]);

  const moment = useMemo(() => {
    const i = stint.findIndex((d) => d.mood === "stressed" || d.mood === "tired");
    if (i < 0 || !stint[i + 1]) return null;
    return { lap: stint[i].lap, jump: (stint[i + 1].t - stint[i].t).toFixed(1) };
  }, [stint]);

  // waveform playback
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setHead((h) => { if (h >= wf.length - 1) { setPlaying(false); return wf.length - 1; } return h + 1; });
    }, 34);
    return () => clearInterval(id);
  }, [playing, wf.length]);

  // auto-demo: walk the whole stint radio-by-radio
  useEffect(() => {
    if (!autostint) return;
    const order = RADIO.map((r) => r.i);
    let p = order.indexOf(clamped);
    const id = setInterval(() => {
      p = (p + 1) % order.length;
      setSel(order[p]); setHead(0); setPlaying(true);
      if (p === order.length - 1) setAutostint(false);
    }, 2600);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autostint]);

  const pick = (i) => { setSel(i); setHead(0); setPlaying(true); setAutostint(false); };

  const calmAvg = avg(stint.filter((d) => d.mood === "calm" || d.mood === "focused").map((d) => d.t));
  const hotAvg = avg(stint.filter((d) => d.mood === "stressed" || d.mood === "tired").map((d) => d.t));
  const delta = (hotAvg - calmAvg).toFixed(2);
  const trigger = stint.find((d) => d.mood === "stressed" || d.mood === "tired");

  const tMin = Math.min(...stint.map((d) => d.t)) - 0.2;
  const tMax = Math.max(...stint.map((d) => d.t)) + 0.2;
  const hotLap = trigger ? trigger.lap : null;
  const lastLap = stint[stint.length - 1].lap;

  const badge = { live: [INK, "LIVE API"], demo: ["#35c5e8", "DEMO API"], offline: [RED, "OFFLINE · FALLBACK"] }[source] || ["#35c5e8", "DEMO API"];

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      <header style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={S.bolt} />
          <div>
            <div style={S.eyebrow}>AI RACE MONTH · PROBLEM 01</div>
            <div style={S.title}>THE SILENT CO&#8209;DRIVER</div>
          </div>
        </div>
        <div style={S.session}>
          <Chip k="DRIVER" v={meta?.driver || "#7 K. RENNER"} />
          <Chip k="TEAM" v={meta?.team || "APEX RACING"} />
          <TyreChip compound={meta?.compound} />
          <Chip k="STINT" v={meta?.stint || "L28–L42"} />
          <div style={S.badge}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: badge[0], boxShadow: `0 0 8px ${badge[0]}` }} />
            {badge[1]}
          </div>
        </div>
      </header>

      <div className="pw-grid" style={S.grid}>
        <aside style={S.radioCol}>
          <div style={S.colHead}>
            <span>TEAM RADIO</span>
            <span style={{ color: MUTE }}>{RADIO.length} CALLS</span>
          </div>
          <div style={S.radioList}>
            {RADIO.map((r) => {
              const m = MOOD[r.mood] || MOOD.calm;
              const on = r.i === clamped;
              return (
                <button key={r.i} onClick={() => pick(r.i)} className="radiobtn"
                  style={{ ...S.radioItem, borderColor: on ? m.c : LINE, background: on ? "rgba(255,255,255,0.03)" : "transparent" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={S.lapTag}>LAP {r.lap}</span>
                    <span style={{ ...S.moodDot, background: m.c, boxShadow: `0 0 8px ${m.c}` }} />
                  </div>
                  <div style={{ ...S.radioText, color: on ? INK : MUTE }}>{r.radio}</div>
                  <div style={{ ...S.moodMini, color: m.c }}>{m.label}</div>
                </button>
              );
            })}
          </div>
        </aside>

        <main style={S.main}>
          <section style={S.player}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={S.eyebrow}>RADIO CLIP · LAP {cur.lap}</div>
                <div style={S.transcript}>{cur.radio ? `“${cur.radio}”` : "— no radio this lap —"}</div>
              </div>
              <div style={{ textAlign: "right", flexShrink: 0, marginLeft: 16 }}>
                <div style={S.eyebrow}>TONE</div>
                <div style={{ ...S.moodBig, color: mood.c }}>{mood.label}</div>
                <div style={S.moodSub}>confidence {50 + Math.round(cur.stress / 2.2)}%</div>
              </div>
            </div>

            <div style={S.waveWrap}>
              <button onClick={() => { if (head >= wf.length - 1) setHead(0); setPlaying((p) => !p); }}
                className="playbtn" style={S.play} aria-label={playing ? "pause" : "play"}>
                {playing ? "❚❚" : "▶"}
              </button>
              <div style={S.wave}>
                {wf.map((h, k) => (
                  <span key={k} style={{ height: `${h * 100}%`, background: k <= head ? mood.c : "#2c3038", opacity: k <= head ? 1 : 0.7 }} className="wbar" />
                ))}
              </div>
              <span style={S.dur}>0:0{Math.min(9, Math.round((head / wf.length) * 6))} / 0:06</span>
            </div>
          </section>

          <section style={S.chartCard}>
            <div style={S.colHead}>
              <span>STRESS vs LAP TIME</span>
              <span style={{ color: MUTE, fontSize: 11 }}>higher line = slower lap ▲</span>
            </div>

            <div style={{ position: "relative" }}>
              {moment && (
                <div style={S.finding}>
                  <div style={{ color: RED, fontSize: 10, letterSpacing: 1.5, fontFamily: "'Chakra Petch',sans-serif" }}>▲ THE MOMENT</div>
                  <div style={{ fontSize: 13, color: INK, marginTop: 2, lineHeight: 1.3 }}>
                    First stress call, <b>Lap {moment.lap}</b>. Next lap: <b style={{ color: RED }}>{moment.jump >= 0 ? "+" : ""}{moment.jump}s</b>. The tyre never recovered.
                  </div>
                </div>
              )}

              <ResponsiveContainer width="100%" height={252}>
                <ComposedChart data={stint} margin={{ top: 24, right: 8, left: -6, bottom: 4 }}
                  onClick={(e) => { if (e && e.activeTooltipIndex != null) pick(e.activeTooltipIndex); }}>
                  <defs>
                    <linearGradient id="st" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={RED} stopOpacity={0.34} />
                      <stop offset="100%" stopColor={RED} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={LINE} vertical={false} />
                  {hotLap != null && (
                    <ReferenceArea x1={hotLap} x2={lastLap} fill={RED} fillOpacity={0.05}
                      label={{ value: "TYRE CLIFF", position: "insideTopRight", fill: MUTE, fontSize: 10, fontFamily: "'Chakra Petch',sans-serif" }} />
                  )}
                  <XAxis dataKey="lap" stroke={LINE} tick={{ fill: MUTE, fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} tickLine={false} />
                  <YAxis yAxisId="s" domain={[0, 100]} hide />
                  <YAxis yAxisId="t" orientation="right" domain={[tMin, tMax]} width={40}
                    tick={{ fill: MUTE, fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(1)} />
                  <Tooltip content={<TT />} cursor={{ stroke: "#3a3f49" }} />
                  <Area yAxisId="s" dataKey="stress" stroke={RED} strokeWidth={1.5} fill="url(#st)" type="monotone" name="Stress" dot={false} />
                  <Line yAxisId="t" dataKey="t" stroke={INK} strokeWidth={2.5} type="monotone" name="Lap time" dot={{ r: 2, fill: INK }} activeDot={{ r: 4, fill: RED, stroke: "#000" }} />
                  <ReferenceLine yAxisId="s" x={cur.lap} stroke={mood.c} strokeWidth={1.5} strokeDasharray="3 3" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div style={S.stats}>
              <Stat k="CALM AVG" v={calmAvg ? `${calmAvg.toFixed(2)}s` : "—"} c="#37d67a" />
              <Stat k="STRESSED AVG" v={hotAvg ? `${hotAvg.toFixed(2)}s` : "—"} c="#ff9f1c" />
              <Stat k="PACE LOST" v={hotAvg && calmAvg ? `+${delta}s / lap` : "—"} c={RED} />
              <Stat k="TRIGGER" v={trigger ? `LAP ${trigger.lap} RADIO` : "—"} c={INK} />
            </div>
          </section>

          <div style={S.footer}>
            <div style={{ color: MUTE, fontSize: 12 }}>
              Whisper → transcript · wav2vec2 → tone · synced to lap delta.
              <span style={{ color: "#4a4f59" }}> {source === "offline" ? "Backend offline — showing demo data." : "Live from backend."}</span>
            </div>
            <button className="stintbtn" style={{ ...S.stintBtn, borderColor: autostint ? RED : LINE }}
              onClick={() => { setAutostint((a) => !a); if (!autostint) { setHead(0); setPlaying(true); } }}>
              {autostint ? "■ STOP STINT" : "▶ PLAY FULL STINT"}
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}

function Chip({ k, v }) {
  return (
    <div style={S.chip}>
      <span style={{ color: MUTE, fontSize: 9, letterSpacing: 1 }}>{k}</span>
      <span style={{ color: INK, fontSize: 12, fontFamily: "'JetBrains Mono',monospace" }}>{v}</span>
    </div>
  );
}
function TyreChip({ compound }) {
  return (
    <div style={S.chip}>
      <span style={{ color: MUTE, fontSize: 9, letterSpacing: 1 }}>TYRE</span>
      <span style={{ display: "flex", alignItems: "center", gap: 5, color: INK, fontSize: 12 }}>
        <span style={{ width: 11, height: 11, borderRadius: "50%", border: "2px solid #eaeaea", display: "inline-block" }} />
        {compound || "HARD"}
      </span>
    </div>
  );
}
function Stat({ k, v, c }) {
  return (
    <div style={S.stat}>
      <div style={{ color: MUTE, fontSize: 9, letterSpacing: 1 }}>{k}</div>
      <div style={{ color: c, fontSize: 16, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700, marginTop: 3 }}>{v}</div>
    </div>
  );
}
function TT({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  const m = d ? MOOD[d.mood] : null;
  return (
    <div style={{ background: "#0c0d10", border: `1px solid ${LINE}`, padding: "8px 11px", fontFamily: "'JetBrains Mono',monospace" }}>
      <div style={{ color: INK, fontSize: 12 }}>LAP {label}</div>
      <div style={{ color: MUTE, fontSize: 11, marginTop: 3 }}>{d && d.t.toFixed(1)}s · stress {d && d.stress}</div>
      {m && <div style={{ color: m.c, fontSize: 10, marginTop: 2, fontFamily: "'Chakra Petch',sans-serif" }}>{m.label}</div>}
    </div>
  );
}

const S = {
  root: { background: "#0a0b0d", color: INK, fontFamily: "'Titillium Web',sans-serif", minHeight: "100%", padding: 18, boxSizing: "border-box" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 14, borderBottom: `1px solid ${LINE}`, paddingBottom: 16, marginBottom: 16 },
  bolt: { width: 6, height: 40, background: RED, boxShadow: `0 0 14px ${RED}`, transform: "skewX(-12deg)" },
  eyebrow: { color: RED, fontSize: 10, letterSpacing: 2.5, fontFamily: "'Chakra Petch',sans-serif", fontWeight: 600 },
  title: { fontSize: 24, fontWeight: 700, letterSpacing: 0.5, fontFamily: "'Chakra Petch',sans-serif", fontStyle: "italic", lineHeight: 1.05, marginTop: 2 },
  session: { display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" },
  chip: { display: "flex", flexDirection: "column", gap: 2, padding: "6px 11px", border: `1px solid ${LINE}`, background: PANEL },
  badge: { display: "flex", alignItems: "center", gap: 6, padding: "6px 11px", border: `1px solid ${LINE}`, background: PANEL, fontSize: 11, letterSpacing: 1, fontFamily: "'Chakra Petch',sans-serif" },
  grid: { display: "grid", gridTemplateColumns: "minmax(220px, 300px) 1fr", gap: 16, alignItems: "start" },
  radioCol: { background: PANEL, border: `1px solid ${LINE}` },
  colHead: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "11px 14px", borderBottom: `1px solid ${LINE}`, fontFamily: "'Chakra Petch',sans-serif", fontSize: 12, letterSpacing: 1.5, color: INK },
  radioList: { display: "flex", flexDirection: "column", maxHeight: 560, overflowY: "auto" },
  radioItem: { textAlign: "left", padding: "11px 14px", borderLeft: "2px solid", borderTop: "none", borderRight: "none", borderBottom: `1px solid ${LINE}`, cursor: "pointer", color: INK, transition: "background .15s" },
  lapTag: { fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: INK, letterSpacing: 0.5 },
  moodDot: { width: 8, height: 8, borderRadius: "50%" },
  radioText: { fontSize: 13, marginTop: 6, lineHeight: 1.35 },
  moodMini: { fontSize: 9, letterSpacing: 1.5, marginTop: 6, fontFamily: "'Chakra Petch',sans-serif" },
  main: { display: "flex", flexDirection: "column", gap: 16, minWidth: 0 },
  player: { background: PANEL, border: `1px solid ${LINE}`, padding: 18 },
  transcript: { fontSize: 19, lineHeight: 1.4, marginTop: 8, maxWidth: 560, fontStyle: "italic" },
  moodBig: { fontSize: 22, fontWeight: 700, fontFamily: "'Chakra Petch',sans-serif", letterSpacing: 1, marginTop: 2 },
  moodSub: { color: MUTE, fontSize: 11, fontFamily: "'JetBrains Mono',monospace", marginTop: 2 },
  waveWrap: { display: "flex", alignItems: "center", gap: 12, marginTop: 18 },
  play: { width: 38, height: 38, flexShrink: 0, borderRadius: "50%", border: `1px solid ${LINE}`, background: "#0c0d10", color: INK, cursor: "pointer", fontSize: 13 },
  wave: { flex: 1, height: 46, display: "flex", alignItems: "center", gap: 2, overflow: "hidden" },
  dur: { color: MUTE, fontSize: 11, fontFamily: "'JetBrains Mono',monospace", flexShrink: 0 },
  chartCard: { background: PANEL, border: `1px solid ${LINE}`, paddingBottom: 4 },
  finding: { position: "absolute", top: 6, left: 14, zIndex: 5, maxWidth: 260, background: "rgba(10,11,13,0.82)", border: `1px solid ${LINE}`, borderLeft: `2px solid ${RED}`, padding: "8px 11px", backdropFilter: "blur(2px)" },
  stats: { display: "grid", gridTemplateColumns: "repeat(4,1fr)", borderTop: `1px solid ${LINE}` },
  stat: { padding: "12px 14px", borderRight: `1px solid ${LINE}` },
  footer: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 },
  stintBtn: { background: "#0c0d10", color: INK, border: "1px solid", padding: "9px 16px", cursor: "pointer", fontFamily: "'Chakra Petch',sans-serif", letterSpacing: 1.5, fontSize: 12 },
};

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&family=Chakra+Petch:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
.wbar{flex:1;min-width:1px;border-radius:1px;transition:background .05s,opacity .05s;align-self:center}
.radiobtn:hover{background:rgba(255,255,255,0.02)!important}
.playbtn:hover{border-color:${RED}!important}
.stintbtn:hover{border-color:${RED}!important}
::-webkit-scrollbar{width:8px}
::-webkit-scrollbar-thumb{background:#23262e}
::-webkit-scrollbar-track{background:transparent}
@media (max-width:760px){ .pw-grid{grid-template-columns:1fr!important} }
`;
