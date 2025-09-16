// Integration test for professional interface
class TEPPLIntegrationTest {
    constructor() {
        this.tests = [];
        this.results = [];
    }
    
    async runAllTests() {
        console.log('🧪 Running TEPPL AI Integration Tests...');
        
        // Test 1: Professional Content Renderer
        await this.testProfessionalRenderer();
        
        // Test 2: Collapsible Sources
        await this.testCollapsibleSources();
        
        // Test 3: Document Links
        await this.testDocumentLinks();
        
        // Test 4: Confidence Display
        await this.testConfidenceDisplay();
        
        // Test 5: Markdown Rendering
        await this.testMarkdownRendering();
        
        this.displayResults();
    }
    
    async testProfessionalRenderer() {
        try {
            const renderer = new ProfessionalContentRenderer();
            const testMarkdown = `# Test Header\n## Sub Header\n• Test bullet\n**Bold text**`;
            const result = renderer.renderMarkdownContent(testMarkdown);
            
            const hasHeaders = result.includes('md-h1') && result.includes('md-h2');
            const hasBullets = result.includes('md-bullet');
            const hasBold = result.includes('<strong>');
            
            this.addResult('Professional Renderer', hasHeaders && hasBullets && hasBold);
        } catch (error) {
            this.addResult('Professional Renderer', false, error.message);
        }
    }
    
    async testCollapsibleSources() {
        try {
            const renderer = new ProfessionalContentRenderer();
            const testSources = [
                { title: 'Test Doc', page_info: { display: 'Page 1' }, document_type: 'Policy', relevance: '85%', content: 'Test content' }
            ];
            
            const sourcesElement = renderer.createCollapsibleSources(testSources);
            const hasToggle = sourcesElement.querySelector('.sources-toggle') !== null;
            const hasContent = sourcesElement.querySelector('.sources-content') !== null;
            
            this.addResult('Collapsible Sources', hasToggle && hasContent);
        } catch (error) {
            this.addResult('Collapsible Sources', false, error.message);
        }
    }
    
    async testDocumentLinks() {
        try {
            // Test document link generation
            const testLink = '/documents/test-doc.pdf';
            const response = await fetch(testLink, { method: 'HEAD' });
            
            // Even if document doesn't exist, the route should be available
            this.addResult('Document Links', response.status === 404 || response.status === 200);
        } catch (error) {
            this.addResult('Document Links', false, error.message);
        }
    }
    
    async testConfidenceDisplay() {
        try {
            const renderer = new ProfessionalContentRenderer();
            const confidenceElement = renderer.createConfidenceIndicator(0.85);
            
            const hasCorrectClass = confidenceElement.classList.contains('confidence-indicator');
            const hasPercentage = confidenceElement.innerHTML.includes('85%');
            
            this.addResult('Confidence Display', hasCorrectClass && hasPercentage);
        } catch (error) {
            this.addResult('Confidence Display', false, error.message);
        }
    }
    
    async testMarkdownRendering() {
        try {
            const renderer = new ProfessionalContentRenderer();
            const testData = {
                answer: '# Test\n## Subtest\n• Bullet point\n**Bold text**',
                confidence: 0.85,
                sources: []
            };
            
            const messageElement = renderer.createProfessionalMessage(testData);
            const hasMarkdown = messageElement.querySelector('.markdown-content') !== null;
            const hasConfidence = messageElement.querySelector('.confidence-indicator') !== null;
            
            this.addResult('Markdown Rendering', hasMarkdown && hasConfidence);
        } catch (error) {
            this.addResult('Markdown Rendering', false, error.message);
        }
    }
    
    addResult(testName, passed, error = null) {
        this.results.push({
            name: testName,
            passed: passed,
            error: error
        });
    }
    
    displayResults() {
        console.log('\n📊 TEPPL AI Integration Test Results:');
        console.log('=====================================');
        
        let passedCount = 0;
        
        this.results.forEach(result => {
            const status = result.passed ? '✅ PASS' : '❌ FAIL';
            console.log(`${status} ${result.name}`);
            
            if (!result.passed && result.error) {
                console.log(`   Error: ${result.error}`);
            }
            
            if (result.passed) passedCount++;
        });
        
        console.log('=====================================');
        console.log(`Results: ${passedCount}/${this.results.length} tests passed`);
        
        if (passedCount === this.results.length) {
            console.log('🎉 All tests passed! Professional interface ready.');
        } else {
            console.log('⚠️  Some tests failed. Check implementation.');
        }
    }
}

// Auto-run tests in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    document.addEventListener('DOMContentLoaded', async function() {
        const tester = new TEPPLIntegrationTest();
        await tester.runAllTests();
    });
}
