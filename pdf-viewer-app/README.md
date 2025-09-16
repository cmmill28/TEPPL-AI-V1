# PDF Viewer App

## Overview
The PDF Viewer App is a web application that allows users to view PDF documents with a toggleable image sidebar. The application supports deep-linking for documents, includes an "Include Figures" toggle, and provides parallel text and image retrieval. The layout is designed to be user-friendly with a revamped three-column structure.

## Features
- Toggleable image sidebar for easy access to related images.
- Deep-linking support for direct access to specific documents.
- "Include Figures" toggle to enhance document viewing experience.
- Parallel retrieval of text and images for comprehensive information display.
- Responsive three-column layout for optimal viewing on various devices.

## Project Structure
```
pdf-viewer-app
├── src
│   ├── js
│   │   ├── app.js
│   │   ├── sidebar.js
│   │   ├── viewer.js
│   │   └── utils.js
│   ├── css
│   │   ├── styles.css
│   │   ├── sidebar.css
│   │   └── viewer.css
│   └── components
│       ├── Sidebar.js
│       └── Viewer.js
├── public
│   ├── index.html
│   └── assets
│       └── icons
├── package.json
├── webpack.config.js
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd pdf-viewer-app
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Usage
1. Start the development server:
   ```
   npm start
   ```
2. Open your browser and navigate to `http://localhost:3000` to view the application.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.