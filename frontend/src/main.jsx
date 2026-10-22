import { StrictMode, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = 'http://127.0.0.1:8000'

function App() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [state, setState] = useState('idle')
  const [error, setError] = useState('')
  const input = useRef()
  const chooseFile = (selected) => {
    if (!selected) return
    setFile(selected); setResult(null); setError(''); setState('idle')
  }
  const classify = async () => {
    if (!file) return
    setState('loading'); setError(''); setResult(null)
    const data = new FormData(); data.append('video', file)
    try {
      const response = await fetch(`${API}/api/predict`, { method: 'POST', body: data })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Analysis failed.')
      setResult(body); setState('done')
    } catch (err) { setError(err.message); setState('idle') }
  }
  return <main>
    <nav><div className="brand"><span>🤗</span> AI Project <i>/</i> Shot Classification</div><a href="#about">About</a></nav>
    <section className="hero"><p className="eyebrow">COMPUTER VISION DEMO</p><h1>Cricket shot<br/><em>classification.</em></h1><p className="subtitle">Upload a short broadcast clip. The published model detects the striker and bat, applies segmentation, then identifies the shot.</p><div className="chips"><span>EfficientNetV2-S</span><span>GRU</span><span>15 shot classes</span><span>Runs locally</span></div></section>
    <section className="workspace">
      <div className="panel upload-panel"><div className="panel-title"><b>Input video</b><small>MP4 · AVI · MOV · MKV</small></div>
        <button className={`dropzone ${file ? 'has-file' : ''}`} onClick={() => input.current.click()} onDragOver={e => e.preventDefault()} onDrop={e => {e.preventDefault(); chooseFile(e.dataTransfer.files[0])}}>
          <input ref={input} type="file" accept="video/*" onChange={e => chooseFile(e.target.files[0])}/>
          <span className="upload-icon">⇧</span><strong>{file ? file.name : 'Drop a cricket video here'}</strong><small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ready to analyse` : 'or click to browse your computer'}</small>
        </button>
        <button className="primary" disabled={!file || state === 'loading'} onClick={classify}>{state === 'loading' ? 'Analysing 15 frames…' : 'Classify shot  →'}</button>
        {error && <p className="error">{error}</p>}
      </div>
      <div className="panel result-panel"><div className="panel-title"><b>Model output</b><small>{state === 'done' ? 'Complete' : 'Waiting for video'}</small></div>
        {!result ? <div className="empty"><span>◌</span><p>Your prediction will appear here.</p></div> : <><div className="winner"><p>TOP PREDICTION</p><h2>{result.prediction.label}</h2><strong>{(result.prediction.confidence * 100).toFixed(1)}%</strong><span>confidence</span></div><div className="ranking">{result.ranking.map((item, i) => <div className="rank" key={item.label}><span>{i + 1}</span><label>{item.label}<i><b style={{width: `${item.confidence * 100}%`}}/></i></label><strong>{(item.confidence * 100).toFixed(1)}%</strong></div>)}</div></>}
      </div>
    </section>
    {result && <section className="preview"><div><p className="eyebrow">MODEL PREPROCESSING</p><h2>What the model sees</h2><p>The original frames after striker-and-bat cropping plus the segmentation overlay.</p></div><video controls src={result.previewUrl}/></section>}
    <section id="about" className="about"><b>How it works</b><span>1. Crop batter + bat</span><span>2. Segment the player and bat</span><span>3. Classify the 15-frame sequence</span></section>
  </main>
}
createRoot(document.getElementById('root')).render(<StrictMode><App /></StrictMode>)
