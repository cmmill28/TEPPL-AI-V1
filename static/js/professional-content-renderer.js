// Professional Content Renderer for TEPPL AI - REPLACED VERSION
(function () {
  class ProfessionalContentRenderer {
    createProfessionalMessage(data) {
      const { answer = "", confidence = 0 } = data || {};
      const wrap = document.createElement("div");
      wrap.className = "message assistant-message animate-in";

      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      avatar.textContent = "🤖";

      const content = document.createElement("div");
      content.className = "message-content";

      const html = this.renderMarkdownContent(answer);
      const body = document.createElement("div");
      body.className = "markdown-content professional-content";
      body.innerHTML = html;

      const conf = Math.max(0, Math.min(1, confidence || 0));
      const pct = Math.round(conf * 100);
      const confEl = document.createElement("div");
      confEl.className = "confidence-indicator";
      confEl.innerHTML = `🎯 <strong>${pct}% Confidence</strong>`;

      content.appendChild(body);
      content.appendChild(confEl);
      wrap.appendChild(avatar);
      wrap.appendChild(content);
      return wrap;
    }

    renderMarkdownContent(text) {
      if (!text) return "";
      text = this.ensureSpacing(text);
      const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

      const fences = [];
      text = text.replace(/```([\s\S]*?)```/g, (m, code) => {
        const token = `__FENCE_${fences.length}__`;
        fences.push(`<pre><code>${esc(code.trim())}</code></pre>`);
        return token;
      });

      text = text.replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n");
      text = text.replace(/^\s*>\s?(.*)$/gm, `<blockquote>$1</blockquote>`);
      text = text.replace(/^#\s+(.+)$/gm, `<h1>$1</h1>`);
      text = text.replace(/^##\s+(.+)$/gm, `<h2>$1</h2>`);
      text = text.replace(/^###\s+(.+)$/gm, `<h3>$1</h3>`);

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
            if (/^<h\d|^<ul|^<ol|^<pre|^<blockquote|^<table|^<p|^__FENCE_/.test(trimmed)) {
              return trimmed;
            }
            if (trimmed.includes("\n")) {
              return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
            }
            return `<p>${trimmed}</p>`;
          })
        .join("\n");

      fences.forEach((html, i) => {
        text = text.replace(`__FENCE_${i}__`, html);
      });
      return text;
    }

    ensureSpacing(md) {
      return md
        .replace(/(^|\n)(#{1,3}\s[^\n]+)\n(?!\n)/g, '$1$2\n\n')
        .replace(/\n{3,}/g, '\n\n');
    }

    _groupListBlocks(text, ordered) {
      const rx = ordered ? /^(\s*\d+\.\s+.*)$/gm : /^(\s*(?:-|\*)\s+.*)$/gm;
      const tag = ordered ? "ol" : "ul";
      const itemRx = ordered ? /^\s*\d+\.\s+(.*)$/ : /^\s*(?:-|\*)\s+(.*)$/;

      const lines = text.split("\n");
      const out = [];
      let buf = [];
      const flush = () => {
        if (!buf.length) return;
        const items = buf.map(l => {
          const m = l.match(itemRx);
          return m ? `<li>${m[1]}</li>` : "";
        }).join("");
        out.push(`<${tag}>${items}</${tag}>`);
        buf = [];
      };

      for (const line of lines) {
        if (rx.test(line)) {
          buf.push(line); rx.lastIndex = 0; continue;
        }
        flush();
        out.push(line);
      }
      flush();
      return out.join("\n");
    }

    _wireCitations(root) {
      // Convert inline [^n] markers to superscript links
      root.innerHTML = root.innerHTML.replace(/\[\^(\d+)\]/g, (m, num) => {
        const n = Number(num);
        return `<sup class="teppl-cite"><a href="#src-${n}" aria-label="Source ${n}">[${n}]</a></sup>`;
      });

      const sourcesH2 = Array.from(root.querySelectorAll('h2')).find(h => /sources/i.test(h.textContent.trim()));
      if (!sourcesH2) return;

      // Gather consecutive footnote paragraphs after Sources
      let node = sourcesH2.nextElementSibling;
      const items = [];
      while (node && !/^H[12]$/.test(node.tagName)) {
        const text = (node.textContent || '').trim();
        const match = text.match(/^\[\^(\d+)\]:\s*(.+)$/);
        if (match) {
          items.push({ n: Number(match[1]), label: match[2] });
        }
        node = node.nextElementSibling;
      }
      if (!items.length) return;

      const ul = document.createElement('ul');
      ul.className = 'md-ol sources-list';
      items.forEach(({ n, label }) => {
        const li = document.createElement('li');
        li.id = `src-${n}`;
        li.innerHTML = `<span class="src-num">[${n}]</span> ${this._escape(label)} <a class="backref" href="#top" aria-label="Back to content">↩</a>`;
        ul.appendChild(li);
      });

      // Remove original footnote lines
      node = sourcesH2.nextElementSibling;
      const toRemove = [];
      while (node && !/^H[12]$/.test(node.tagName)) {
        const t = (node.textContent || '').trim();
        if (/^\[\^\d+\]:/.test(t)) toRemove.push(node);
        node = node.nextElementSibling;
      }
      toRemove.forEach(el => el.remove());
      sourcesH2.insertAdjacentElement('afterend', ul);
    }
  }
  window.ProfessionalContentRenderer = ProfessionalContentRenderer;
})();