// Professional Content Renderer for TEPPL AI - REPLACED VERSION
(function () {
  class ProfessionalContentRenderer {
    createProfessionalMessage(data) {
      const { answer = "", confidence = 0, sources = [] } = data || {};
      const wrap = document.createElement("div");
      wrap.className = "message assistant-message animate-in";

      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      avatar.textContent = "🤖";

      const content = document.createElement("div");
      content.className = "message-content";

      // Header (author + timestamp)
      const header = document.createElement("div");
      header.className = "message-header";
      header.innerHTML = `<span class="message-author">TEPPL AI</span><span class="message-timestamp">${new Date().toLocaleTimeString()}</span>`;

      // Confidence row
      const confRow = document.createElement("div");
      confRow.className = "confidence-row";
      const conf = Math.max(0, Math.min(1, confidence || 0));
      const pct = Math.round(conf * 100);
      confRow.innerHTML = `🎯 <strong>${pct}% Confidence</strong>`;

      // Render markdown then inject footnotes (needs sources)
      let html = this.renderMarkdownContent(answer);
      html = this.injectFootnotes(html, sources);
      const body = document.createElement("div");
      body.className = "markdown-content professional-content md-body";
      body.innerHTML = html;

      content.appendChild(header);
      content.appendChild(confRow);
      content.appendChild(body);

      // Collapsible sources block (professional path expects this)
      if (sources && sources.length) {
        const collapsible = this.createCollapsibleSources(sources);
        content.appendChild(collapsible);
      }

      wrap.appendChild(avatar);
      wrap.appendChild(content);
      return wrap;
    }

    renderMarkdownContent(text) {
      if (!text) return "";
      text = this.ensureSpacing(text);
      // Existing lightweight markdown conversions (retain prior behavior)
      const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const fences = [];
      text = text.replace(/```([\s\S]*?)```/g, (m, code) => {
        const token = `__FENCE_${fences.length}__`;
        fences.push(`<pre><code>${esc(code.trim())}</code></pre>`);
        return token;
      });
      text = text.replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n");
      text = text.replace(/^>\s?(.*)$/gm, `<blockquote>$1</blockquote>`);
      text = text.replace(/^#\s+(.+)$/gm, `<h1 class="md-h1">$1</h1>`);
      text = text.replace(/^##\s+(.+)$/gm, `<h2 class="md-h2">$1</h2>`);
      text = text.replace(/^###\s+(.+)$/gm, `<h3 class="md-h3">$1</h3>`);
      text = this._groupListBlocks(text, true);
      text = this._groupListBlocks(text, false);
      text = text.replace(/\*\*(.+?)\*\*/g, `<strong>$1</strong>`);
      text = text.replace(/\*(.+?)\*/g, `<em>$1</em>`);
      text = text.replace(/`([^`]+?)`/g, `<code>$1</code>`);
      text = text
        .split(/\n{2,}/)
        .map((block) => {
          const trimmed = block.trim();
          if (!trimmed) return "";
          if (/^<h\d|^<ul|^<ol|^<pre|^<blockquote|^<table|^<p|^__FENCE_/.test(trimmed)) return trimmed;
          if (trimmed.includes("\n")) return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
          return `<p>${trimmed}</p>`;
        })
        .join("\n");
      fences.forEach((html, i) => { text = text.replace(`__FENCE_${i}__`, html); });
      return text;
    }

    injectFootnotes(markdown, sources) {
      if (!markdown) return markdown;
      // Superscripts for inline markers
      const withSuperscripts = markdown.replace(/\[\^(\d+)\]/g, (m, num) => `<sup id="cite-${num}"><a href="#src-${num}" class="md-cite">[${num}]</a></sup>`);
      const hasCites = /\[\^(\d+)\]/.test(markdown);
      if (!hasCites && (!sources || !sources.length)) return withSuperscripts;
      const indexed = (sources || []).map((s, i) => ({
        n: i + 1,
        title: s.title || 'NCDOT TEPPL Document',
        page: s.page_number || s.pages || 'N/A',
        link: s.internal_link || s.file_path || s.filename || '',
        relevance: s.relevance || ''
      }));
      const list = indexed.map(({ n, title, page, link, relevance }) => {
        const titleSafe = this.escapeHtml(title);
        const rel = relevance ? `<span class="src-rel">${relevance}</span>` : '';
        const pageTxt = page ? ` • p. ${page}` : '';
        const viewBtn = link ? `<button class="source-btn primary" data-link="${link}" data-page="${page || 1}">📄 View PDF</button>` : '';
        return `<li id="src-${n}" class="src-item"><span class="src-index">[${n}]</span><span class="src-title">${titleSafe}${pageTxt}</span>${rel}${viewBtn}</li>`;
      }).join("");
      const block = `<div class="md-sources-block"><h3 class="md-h3">Sources</h3><ul class="src-list">${list}</ul></div>`;
      return `${withSuperscripts}\n\n${block}`;
    }

    createCollapsibleSources(sources) {
      const count = (sources || []).length;
      const safe = (t) => this.escapeHtml(t || 'NCDOT TEPPL Document');
      const items = (sources || []).map((s, i) => {
        const link = s.internal_link || s.file_path || s.filename || '';
        const page = s.page_number || s.pages || 'N/A';
        const rel = s.relevance || '';
        return `<div class="source-card"><div class="source-meta"><div class="source-index">${i + 1}</div><div class="source-title" title="${safe(s.title)}">${safe(s.title)}</div><div class="source-sub">Page ${page}${rel ? ` • ${rel}` : ''}</div></div><div class="source-actions">${link ? `<button class="source-btn primary" data-link="${link}" data-page="${page || 1}">📄 View PDF</button>` : ''}</div></div>`;
      }).join("");
      const root = document.createElement('details');
      root.className = 'sources-collapsible';
      root.open = false;
      root.innerHTML = `<summary><span class="sources-pill">📚 Sources (${count})</span></summary><div class="sources-body">${items || '<div class="source-empty">No sources</div>'}</div>`;
      root.addEventListener('click', (e) => {
        const btn = e.target.closest('button.source-btn');
        if (!btn) return;
        const link = btn.getAttribute('data-link');
        const page = parseInt(btn.getAttribute('data-page') || '1', 10);
        if (window.searchBar && typeof window.searchBar.viewPDF === 'function') {
          e.preventDefault();
          window.searchBar.viewPDF(link, page);
        }
      });
      return root;
    }

    ensureSpacing(md) {
      return md.replace(/(^|\n)(#{1,3}\s[^\n]+)\n(?!\n)/g, '$1$2\n\n').replace(/\n{3,}/g, '\n\n');
    }

    _groupListBlocks(text, ordered) {
      const rx = ordered ? /^(\s*\d+\.\s+.*)$/gm : /^(\s*(?:-|\*)\s+.*)$/gm;
      const tag = ordered ? 'ol' : 'ul';
      const itemRx = ordered ? /^\s*\d+\.\s+(.*)$/ : /^\s*(?:-|\*)\s+(.*)$/;
      const lines = text.split('\n');
      const out = []; let buf = [];
      const flush = () => { if (!buf.length) return; const items = buf.map(l => { const m = l.match(itemRx); return m ? `<li>${m[1]}</li>` : ''; }).join(''); out.push(`<${tag}>${items}</${tag}>`); buf = []; };
      for (const line of lines) { if (rx.test(line)) { buf.push(line); rx.lastIndex = 0; continue; } flush(); out.push(line); }
      flush();
      return out.join('\n');
    }

    escapeHtml(str) {
      const d = document.createElement('div');
      d.textContent = String(str ?? '');
      return d.innerHTML;
    }
  }
  window.ProfessionalContentRenderer = ProfessionalContentRenderer;
})();