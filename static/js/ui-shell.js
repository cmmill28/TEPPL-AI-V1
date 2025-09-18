/**
 * ui-shell.js
 * Restores removed inline functionality:
 *  - Theme toggle
 *  - System info modal loading
 *  - PDF modal (basic PDF.js viewer)
 *  - Image modal helper (enhanced)
 *  - Quick actions & history search minimal behavior
 */

(function() {
  if (window.__UI_SHELL_INIT__) return;
  window.__UI_SHELL_INIT__ = true;

  const qs = sel => document.querySelector(sel);
  const qsa = sel => Array.from(document.querySelectorAll(sel));

  /* ---------------- Theme Manager ---------------- */
  const ThemeManager = {
    key: 'theme',
    init() {
      const btn = qs('#theme-toggle');
      const current = localStorage.getItem(this.key) || 'dark';
      this.apply(current);
      if (btn) {
        btn.addEventListener('click', () => {
          const next = document.documentElement.getAttribute('data-color-scheme') === 'dark' ? 'light' : 'dark';
          this.apply(next);
        });
      }
    },
    apply(theme) {
      document.documentElement.setAttribute('data-color-scheme', theme);
      localStorage.setItem(this.key, theme);
      const btn = qs('#theme-toggle');
      if (btn) btn.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
    }
  };

  /* ---------------- System Info Modal ---------------- */
  const SystemInfo = {
    openBtn: null,
    closeBtn: null,
    modal: null,
    content: null,
    init() {
      this.modal = qs('#system-info-modal');
      this.content = qs('#system-info-content');
      this.openBtn = qs('#system-info-btn');
      this.closeBtn = qs('#close-system-info');
      if (this.openBtn) this.openBtn.addEventListener('click', () => this.show());
      if (this.closeBtn) this.closeBtn.addEventListener('click', () => this.hide());
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && this.modal?.classList.contains('active')) this.hide();
      });
    },
    async show() {
      if (!this.modal) return;
      this.modal.classList.add('active');
      if (this.content) {
        this.content.innerHTML = '<p>Loading system information...</p>';
        try {
          const r = await fetch('/api/system-info');
          const data = await r.json();
            this.content.innerHTML = `
              <div class="system-stats">
                <h4>System Status</h4>
                <p><strong>Type:</strong> ${data.system_type || 'Unknown'}</p>
                <p><strong>Multimodal:</strong> ${data.multimodal_available ? 'Yes' : 'No'}</p>
                <h4>Documents</h4>
                <p><strong>Total Files:</strong> ${data.file_stats?.total_files ?? 'N/A'}</p>
                <p><strong>Total Size:</strong> ${data.file_stats?.total_size_mb ?? 'N/A'} MB</p>
                <h4>Images</h4>
                <p><strong>Kept:</strong> ${data.image_counts?.kept ?? 0}</p>
                <p><strong>Total:</strong> ${data.image_counts?.total ?? 0}</p>
                <h4>Categories</h4>
                <pre style="font-size:.75rem;background:var(--color-surface-alt,#111);padding:.5rem;border-radius:6px;max-height:140px;overflow:auto;">${JSON.stringify(data.categories || {}, null, 2)}</pre>
                <p style="opacity:.7;font-size:.75rem;">Fetched: ${new Date().toLocaleString()}</p>
              </div>
            `;
        } catch(e) {
          this.content.innerHTML = '<p>Error loading system info.</p>';
        }
      }
    },
    hide() {
      if (this.modal) this.modal.classList.remove('active');
    }
  };

  /* ---------------- PDF Viewer (basic) ---------------- */
  const PDFViewer = {
    pdfDoc: null,
    currentPage: 1,
    scale: 1.0,
    loading: false,
    init() {
      this.modal = qs('#document-modal');
      this.canvas = qs('#pdf-canvas');
      this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
      this.prevBtn = qs('#pdf-prev');
      this.nextBtn = qs('#pdf-next');
      this.zoomIn = qs('#pdf-zoom-in');
      this.zoomOut = qs('#pdf-zoom-out');
      this.pageInfo = qs('#page-info');
      this.viewer = qs('#pdf-viewer');
      this.container = qs('#pdf-container');
      this.closeBtn = qs('#close-document-modal');
      if (!window.pdfjsLib) return;

      [this.prevBtn, this.nextBtn, this.zoomIn, this.zoomOut].forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', () => {
          if (btn === this.prevBtn) this.changePage(-1);
          if (btn === this.nextBtn) this.changePage(1);
          if (btn === this.zoomIn) this.changeZoom(0.1);
          if (btn === this.zoomOut) this.changeZoom(-0.1);
        });
      });
      if (this.closeBtn) this.closeBtn.addEventListener('click', () => this.hide());
      window.openPDFModal = (path, page=1) => this.open(path, page);
    },
    async open(path, page=1) {
      if (!window.pdfjsLib || !this.modal) {
        // fallback
        window.open(`${path}#page=${page}`, '_blank','noopener');
        return;
      }
      this.modal.classList.add('active');
      this.viewer.style.display = 'block';
      this.container.style.display = 'none';
      this.viewer.querySelector('.loading-indicator')?.classList.remove('hidden');
      try {
        this.pdfDoc = await window.pdfjsLib.getDocument(path).promise;
        this.currentPage = Math.min(Math.max(1, page), this.pdfDoc.numPages);
        await this.renderPage();
        this.viewer.style.display = 'none';
        this.container.style.display = 'block';
      } catch(e) {
        console.error('PDF load failed', e);
        const err = qs('#pdf-error');
        if (err) err.style.display = 'block';
      }
    },
    async renderPage() {
      if (!this.pdfDoc || !this.ctx) return;
      this.loading = true;
      const pg = await this.pdfDoc.getPage(this.currentPage);
      const vp = pg.getViewport({ scale: this.scale });
      this.canvas.width = vp.width;
      this.canvas.height = vp.height;
      await pg.render({ canvasContext: this.ctx, viewport: vp }).promise;
      if (this.pageInfo) this.pageInfo.textContent = `${this.currentPage} / ${this.pdfDoc.numPages}`;
      this.loading = false;
    },
    async changePage(delta) {
      if (this.loading) return;
      const target = this.currentPage + delta;
      if (target < 1 || target > (this.pdfDoc?.numPages || 1)) return;
      this.currentPage = target;
      await this.renderPage();
    },
    async changeZoom(delta) {
      if (this.loading) return;
      this.scale = Math.min(3, Math.max(0.5, this.scale + delta));
      await this.renderPage();
    },
    hide() {
      this.modal?.classList.remove('active');
    }
  };

  /* ---------------- Image Modal Helper ---------------- */
  function initImageModal() {
    const closeBtn = qs('#close-image-modal');
    if (closeBtn) closeBtn.addEventListener('click', () => {
      const m = qs('#image-modal');
      if (m) m.classList.remove('active');
    });
    window.openImageModal = (src, title='Image') => {
      const modal = qs('#image-modal');
      const content = qs('#image-modal-content');
      const heading = qs('#image-modal-title');
      if (!modal || !content) return;
      if (heading) heading.textContent = title;
      content.innerHTML = `<img src="${src}" alt="${title}" style="max-width:100%;max-height:80vh;object-fit:contain;">`;
      modal.classList.add('active');
    };
    // Enhance existing cards if they have data-image-src
    qsa('[data-image-src]').forEach(el => {
      el.addEventListener('click', () => openImageModal(el.getAttribute('data-image-src'), el.getAttribute('data-image-title') || 'Image'));
    });
  }

  /* ---------------- Quick Actions & History (minimal) ---------------- */
  function initQuickActions() {
    const list = qs('#custom-actions-list');
    const addBtn = qs('#add-custom-action');
    const searchInput = qs('#quick-actions-search');
    if (!list || !addBtn) return;

    function load() {
      try {
        return JSON.parse(localStorage.getItem('teppl_custom_actions') || '[]');
      } catch { return []; }
    }
    function save(actions) {
      localStorage.setItem('teppl_custom_actions', JSON.stringify(actions.slice(0, 50)));
    }
    function render(filter='') {
      const actions = load().filter(a => a.name.toLowerCase().includes(filter.toLowerCase()));
      list.innerHTML = actions.map((a,i)=>`
        <div class="action-item" data-idx="${i}">
          <span>${a.name}</span>
          <div>
            <button class="edit-btn" data-edit="${i}" title="Edit">✎</button>
            <button class="delete-btn" data-del="${i}" title="Delete">🗑</button>
          </div>
        </div>`).join('') || '<div style="opacity:.6;font-size:.8rem;">No custom actions</div>';
    }
    list.addEventListener('click', e => {
      const idxEdit = e.target.getAttribute('data-edit');
      const idxDel = e.target.getAttribute('data-del');
      let actions = load();
      if (idxEdit !== null) {
        const item = actions[+idxEdit];
        const newName = prompt('Rename action', item?.name);
        if (newName) {
          item.name = newName.trim();
          save(actions);
          render(searchInput?.value || '');
        }
      }
      if (idxDel !== null) {
        actions.splice(+idxDel,1);
        save(actions);
        render(searchInput?.value || '');
      }
    });
    addBtn.addEventListener('click', () => {
      const name = prompt('Action name:');
      const query = name && prompt('Query for this action:');
      if (name && query) {
        const actions = load();
        actions.unshift({ name: name.trim(), query: query.trim() });
        save(actions);
        render(searchInput?.value || '');
      }
    });
    list.addEventListener('dblclick', e => {
      const root = e.target.closest('.action-item');
      if (!root) return;
      const idx = +root.getAttribute('data-idx');
      const actions = load();
      const a = actions[idx];
      if (a?.query && window.searchBar) {
        window.searchBar.submitQuery(a.query);
      }
    });
    if (searchInput) {
      searchInput.addEventListener('input', () => render(searchInput.value));
    }
    render();
  }

  function initHistorySearch() {
    const input = qs('#history-search');
    const results = qs('#history-search-results');
    if (!input || !results) return;
    function loadHistory() {
      try { return JSON.parse(localStorage.getItem('teppl_search_history') || '[]'); }
      catch { return []; }
    }
    function render(filter='') {
      const data = loadHistory().filter(q => q.toLowerCase().includes(filter.toLowerCase()));
      results.innerHTML = data.map(q=>`<div class="history-item" tabindex="0" data-q="${q}">${q}</div>`).join('') ||
        '<div style="opacity:.6;font-size:.8rem;padding:.5rem;">No matches</div>';
    }
    results.addEventListener('click', e => {
      const q = e.target.getAttribute('data-q');
      if (q && window.searchBar) window.searchBar.submitQuery(q);
    });
    results.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        const q = e.target.getAttribute('data-q');
        if (q && window.searchBar) window.searchBar.submitQuery(q);
      }
    });
    input.addEventListener('input', () => render(input.value));
    render();
  }

  /* ---------------- Boot ---------------- */
  document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    SystemInfo.init();
    PDFViewer.init();
    initImageModal();
    initQuickActions();
    initHistorySearch();
    console.log('🧩 ui-shell initialized');
  });
})();
