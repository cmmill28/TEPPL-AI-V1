window.__TEPPL_RENDERER_VERSION__ = 'pro-1.2.0';
console.log('Renderer version', window.__TEPPL_RENDERER_VERSION__);
class ProfessionalContentRenderer {
  createProfessionalMessage(data){
    const { answer = '', confidence = 0, sources = [] } = data || {};
    const wrap = document.createElement('div');
    wrap.className = 'message assistant-message animate-in';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    const content = document.createElement('div');
    content.className = 'message-content';
    let html = this.renderMarkdownContent(answer);
    html = this.injectFootnotes(html, sources);
    const body = document.createElement('div');
    body.className = 'markdown-content professional-content md-body';
    body.innerHTML = html;
    const confRow = document.createElement('div');
    const pct = Math.round(Math.max(0, Math.min(1, confidence)) * 100);
    confRow.className = 'confidence-row';
    confRow.innerHTML = `🎯 <strong>${pct}% Confidence</strong>`;
    content.appendChild(body);
    content.appendChild(confRow);
    if (sources && sources.length) content.appendChild(this.createCollapsibleSources(sources));
    wrap.appendChild(avatar); wrap.appendChild(content);
    return wrap;
  }
  renderMarkdownContent(md){
    if(!md) return ''; md = md.replace(/\r\n?/g,'\n');
    md = md.replace(/^#\s+(.+)$/gm,'<h1>$1</h1>')
           .replace(/^##\s+(.+)$/gm,'<h2>$1</h2>')
           .replace(/^###\s+(.+)$/gm,'<h3>$1</h3>');
    md = md.replace(/```([\s\S]*?)```/g,(m,c)=>`<pre><code>${this._esc(c.trim())}</code></pre>`);
    md = md.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
           .replace(/\*(.+?)\*/g,'<em>$1</em>')
           .replace(/`([^`]+?)`/g,'<code>$1</code>');
    // bullets -> list detection
    md = md.split(/\n{2,}/).map(block=>{
      if(/^<h\d|^<pre/.test(block)) return block;
      if(/^(?:- |• ).+/m.test(block)){
        const items = block.split(/\n/).filter(l=>/^(?:- |• )/.test(l)).map(l=>`<li>${this._esc(l.replace(/^(?:- |• )/,'').trim())}</li>`).join('');
        return `<ul>${items}</ul>`;
      }
      return `<p>${block}</p>`;
    }).join('\n');
    return md;
  }
  injectFootnotes(html,sources){
    html = html.replace(/\[\^(\d+)\]/g,(_,n)=>`<sup id="cite-${n}"><a class="md-cite" href="#src-${n}">[${n}]</a></sup>`);
    if(!sources || !sources.length) return html;
    const list = sources.map((s,i)=>{ const n=i+1; const t=this._esc(s.title||'NCDOT TEPPL Document'); const p=s.page_number||s.pages||'N/A'; const link=s.internal_link?`${s.internal_link}#page=${p}`:''; return `<li id="src-${n}">${link?`<a href="${link}" target="_blank">${t}</a>`:t}${p?` — p. ${p}`:''}</li>`; }).join('');
    return `${html}\n<h3>Sources</h3><ol class="md-sources">${list}</ol>`;
  }
  createCollapsibleSources(sources){
    const det = document.createElement('details'); det.className='sources-collapsible';
    const sum = document.createElement('summary'); sum.innerHTML=`📚 Sources (${sources.length})`; det.appendChild(sum);
    const body = document.createElement('div'); body.className='sources-body';
    sources.forEach((s,i)=>{ const card=document.createElement('div'); card.className='source-card'; const t=this._esc(s.title||'NCDOT TEPPL Document'); const p=s.page_number||s.pages||'N/A'; card.innerHTML=`<div class="source-meta"><div class="source-index">${i+1}</div><div class="source-title">${t}</div><div class="source-sub">Page ${p}</div></div>`; if(s.internal_link){ const btn=document.createElement('button'); btn.className='source-btn primary'; btn.textContent='📄 View PDF'; btn.onclick=()=>window.searchBar?.viewPDF(s.internal_link,p||1); card.appendChild(btn);} body.appendChild(card); });
    det.appendChild(body); return det;
  }
  _esc(s){return String(s).replace(/[&<>]/g,c=>({ '&':'&amp;','<':'&lt;','>':'&gt;' }[c]));}
}
window.ProfessionalContentRenderer = ProfessionalContentRenderer;