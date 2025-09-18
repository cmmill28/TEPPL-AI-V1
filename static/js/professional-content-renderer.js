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
    const container = document.createElement('div');
    container.className = 'sources-collapsible-list';
    const header = document.createElement("h3");
    header.textContent = `📚 Sources (${sources.length} found)`;
    container.appendChild(header);

    sources.forEach((s, i) => {
      const details = document.createElement("details");
      details.className = "source-item";
      details.open = false; // collapsed by default

      const summary = document.createElement("summary");
      const rel = s.relevance ? `<span class="source-rel">${s.relevance}</span>` : "";
      summary.innerHTML = `<strong>${i + 1}. ${s.title}</strong> ${rel}`;
      details.appendChild(summary);

      const body = document.createElement("div");
      body.className = "source-body";

      // Open PDF button
      if (s.internal_link) {
        const url = s.page_number
          ? `${s.internal_link}?page=${encodeURIComponent(s.page_number)}`
          : s.internal_link;
        const btn = document.createElement("a");
        btn.href = url;
        btn.target = "_blank";
        btn.rel = "noopener";
        btn.className = "btn btn-primary";
        btn.textContent = "Open PDF";
        body.appendChild(btn);
      } else {
        const note = document.createElement("div");
        note.className = "source-note";
        note.textContent = "No link available for this source.";
        body.appendChild(note);
      }

      // Optional: brief snippet if present
      if (s.content) {
        const p = document.createElement("p");
        p.className = "source-snippet";
        p.textContent = s.content.slice(0, 220);
        body.appendChild(p);
      }

      // Optional: page display
      if (s.page_number) {
        const meta = document.createElement("div");
        meta.className = "source-meta";
        meta.textContent = `Page ${s.page_number}`;
        body.appendChild(meta);
      }

      details.appendChild(body);
      container.appendChild(details);
    });

    return container;
  }

  _esc(s){return String(s ?? '').replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
}

window.ProfessionalContentRenderer = ProfessionalContentRenderer;