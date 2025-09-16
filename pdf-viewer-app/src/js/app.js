Sure, here's the contents for the file: /pdf-viewer-app/pdf-viewer-app/src/js/app.js

import { openSidebar, closeSidebar } from './sidebar.js';
import { loadPDF, renderPage, goToPage } from './viewer.js';
import { formatMessage } from './utils.js';

const app = {
    currentPDF: null,
    currentPage: 1,
    totalPages: 0,

    init() {
        this.bindEvents();
    },

    bindEvents() {
        document.getElementById('toggle-sidebar').addEventListener('click', this.toggleSidebar.bind(this));
        document.getElementById('load-pdf').addEventListener('change', this.handlePDFLoad.bind(this));
        document.getElementById('prev-page').addEventListener('click', this.goToPreviousPage.bind(this));
        document.getElementById('next-page').addEventListener('click', this.goToNextPage.bind(this));
        document.getElementById('include-figures').addEventListener('change', this.toggleFigures.bind(this));
    },

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        if (sidebar.classList.contains('open')) {
            closeSidebar();
        } else {
            openSidebar();
        }
    },

    handlePDFLoad(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadPDF(file);
        }
    },

    loadPDF(file) {
        loadPDF(file).then(pdf => {
            this.currentPDF = pdf;
            this.totalPages = pdf.numPages;
            this.renderPage(this.currentPage);
        });
    },

    renderPage(pageNum) {
        if (this.currentPDF) {
            renderPage(this.currentPDF, pageNum);
        }
    },

    goToPreviousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.renderPage(this.currentPage);
        }
    },

    goToNextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            this.renderPage(this.currentPage);
        }
    },

    toggleFigures(event) {
        const includeFigures = event.target.checked;
        // Logic to include/exclude figures in the viewer
    }
};

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});