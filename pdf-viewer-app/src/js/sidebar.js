Sure, here's the contents for the file: /pdf-viewer-app/pdf-viewer-app/src/js/sidebar.js

import { toggleClass } from './utils';

const sidebar = document.getElementById('image-sidebar');
const toggleButton = document.getElementById('toggle-sidebar');

export function openSidebar() {
    sidebar.classList.add('open');
}

export function closeSidebar() {
    sidebar.classList.remove('open');
}

export function toggleSidebar() {
    toggleClass(sidebar, 'open');
}

export function loadImages(imageList) {
    const imageContainer = document.getElementById('image-container');
    imageContainer.innerHTML = '';

    imageList.forEach(image => {
        const imgElement = document.createElement('img');
        imgElement.src = image.src;
        imgElement.alt = image.alt;
        imgElement.classList.add('sidebar-image');
        imageContainer.appendChild(imgElement);
    });
}

toggleButton.addEventListener('click', toggleSidebar);