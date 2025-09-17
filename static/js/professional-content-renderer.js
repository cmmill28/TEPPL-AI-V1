// Professional Content Renderer for TEPPL AI - REPLACED VERSION
(function (global) {
  class ProfessionalContentRenderer {
    renderMarkdownContent(mdText) {
      return `<div class="markdown-content professional-content">
        ${this._processMarkdown(mdText || "")}
      </div>`;
    }

    _escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    _processMarkdown(src) {
      let text = (src || "").replace(/\r\n/g, "\n");

      // Handle fenced code blocks first (``` … ```
      const codeBlocks = [];
      text = text.replace(/```([\s\S]*?)```/g, (_, code) => {
        const i = codeBlocks.push(this._escapeHtml(code)) - 1;
        return `[[[CODE_BLOCK_${i}]]]`;
      });

      // Inline formatting
      // links [text](url) – conservative allowlist
      text = text.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
      // bold, italic, inline code
      text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");
      text = text.replace(/`([^`]+?)`/g, "<code>$1</code>");

      // Headings
      text = text.replace(/^####\s+(.*)$/gm, '<h4 class="md-h4">$1</h4>');
      text = text.replace(/^###\s+(.*)$/gm, '<h3 class="md-h3">$1</h3>');
      text = text.replace(/^##\s+(.*)$/gm, '<h2 class="md-h2">$1</h2>');
      text = text.replace(/^#\s+(.*)$/gm, '<h1 class="md-h1">$1</h1>');

      // Blockquotes
      text = text.replace(/^\s*>\s+(.*)$/gm, "<blockquote>$1</blockquote>");

      // List grouping (unordered and ordered)
      const lines = text.split("\n");
      const out = [];
      let inUL = false,
        inOL = false;

      const flushUL = () => {
        if (inUL) {
          out.push("</ul>");
          inUL = false;
        }
      };
      const flushOL = () => {
        if (inOL) {
          out.push("</ol>");
          inOL = false;
        }
      };

      const ulItem = /^\s*(?:-|\*|\+)\s+(.+)\s*$/;
      const olItem = /^\s*(\d+)\.\s+(.+)\s*$/;

      for (const raw of lines) {
        const line = raw;
        let m;

        if ((m = line.match(ulItem))) {
          flushOL();
          if (!inUL) {
            out.push('<ul class="md-ul">');
            inUL = true;
          }
          out.push(`<li>${m[1]}</li>`);
          continue;
        }
        if ((m = line.match(olItem))) {
          flushUL();
          if (!inOL) {
            out.push('<ol class="md-ol">');
            inOL = true;
          }
          out.push(`<li>${m[2]}</li>`);
          continue;
        }

        // not a list item
        flushUL();
        flushOL();

        if (!line.trim()) {
          out.push(""); // preserve blank line boundary
        } else if (/^<h\d|^<blockquote|^<\/?(?:ul|ol|li)/.test(line)) {
          out.push(line); // already-converted line
        } else {
          out.push(`<p>${line}</p>`);
        }
      }
      flushUL();
      flushOL();

      let html = out.join("\n");

      // Restore fenced code blocks
      html = html.replace(/\[\[\[CODE_BLOCK_(\d+)\]\]\]/g, (_, idx) => {
        const c = codeBlocks[Number(idx)] || "";
        return `<pre class="md-pre"><code>${c}</code></pre>`;
      });

      return html;
    }

    // Optional: build the collapsible sources etc. (you already have this)
  }

  global.ProfessionalContentRenderer = ProfessionalContentRenderer;
})(window);