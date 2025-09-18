window.__TEPPL_RENDERER_VERSION__ = 'pro-1.3.0';
console.log('Renderer version', window.__TEPPL_RENDERER_VERSION__);
class ProfessionalContentRenderer {
  createProfessionalMessage(data) {
    const wrap = document.createElement('div');
    wrap.className = 'message assistant-message animate-in';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    // Header + confidence
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `<span class="message-author">TEPPL AI</span>
      <span class="message-timestamp">${new Date().toLocaleTimeString()}</span>`;
    const conf = Math.round((data.confidence || 0) * 100);
    const confRow = document.createElement('div');
    confRow.className = 'confidence-row';
    confRow.innerHTML = `🎯 <strong>${conf}% Confidence</strong>`;

    // Body: render markdown (your headings/bullets) with HTML superscripts already present
    const body = document.createElement('div');
    body.className = 'markdown-content professional-content md-body';
    body.innerHTML = this.renderMarkdownContent(data.answer || '');

    content.appendChild(header);
    content.appendChild(confRow);
    content.appendChild(body);

    // Collapsible Sources (from JSON)
    if (Array.isArray(data.sources) && data.sources.length) {
      content.appendChild(this.createCollapsibleSources(data.sources));
    }

    wrap.appendChild(avatar);
    wrap.appendChild(content);
    return wrap;
  }

  // Lightweight markdown: keep yours if you have one; this is safe enough for headers/lists/strong/em
  renderMarkdownContent(md) {
    if (!md) return '';
    // headings
    md = md.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>')
           .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
           .replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    // bold/italic/code
    md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
           .replace(/\*(.+?)\*/g, '<em>$1</em>')
           .replace(/`([^`]+?)`/g, '<code>$1</code>');
    // bullet lists
    md = md.replace(/(^|\n)-\s+(.+)(?=\n|$)/g, '$1• $2');
    // paragraphs
    return md.split(/\n{2,}/).map(block => {
      if (/^<h\d/.test(block) || block.startsWith('• ') ) {
        if (block.startsWith('• ')) {
          const items = block.split('\n').map(li => li.trim().startsWith('• ') ? `<li>${li.slice(2).trim()}</li>` : '').join('');
          return `<ul>${items}</ul>`;
        }
        return block;
      }
      return `<p>${block}</p>`;
    }).join('\n');
  }

  createCollapsibleSources(sources) {
    const details = document.createElement('details');
    details.className = 'sources-collapsible';
    // collapsed by default
    const summary = document.createElement('summary');
    summary.innerHTML = `📚 Sources (${sources.length})`;
    details.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'sources-body';

    sources.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'source-card';
      const title = s.title || 'NCDOT TEPPL Document';
      const page  = s.page_number || s.pages || 'N/A';
      const rel   = s.relevance ? ` • ${s.relevance}` : '';
      row.innerHTML = `
        <div class="source-meta">
          <div class="source-index">${i + 1}</div>
          <div class="source-title" title="${this._esc(title)}">${this._esc(title)}</div>
          <div class="source-sub">Page ${page}${rel}</div>
        </div>
      `;
      if (s.internal_link) {
        const btn = document.createElement('button');
        btn.className = 'source-btn primary';
        btn.textContent = '📄 View PDF';
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          window.searchBar?.viewPDF(s.internal_link, page || 1);
        });
        row.appendChild(btn);
      }
      body.appendChild(row);
    });

    details.appendChild(body);
    return details;
  }

  _esc(s){return String(s ?? '').replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
}

window.ProfessionalContentRenderer = ProfessionalContentRenderer;