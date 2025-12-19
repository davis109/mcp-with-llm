// Premium Workflow Viewer - JavaScript
// =====================================

class PremiumWorkflowApp {
    constructor() {
        this.savedWorkflows = this.loadSavedWorkflows();
        this.currentWorkflow = null;
        this.darkMode = localStorage.getItem('darkMode') === 'true';
        
        this.workflows = [
            {
                name: 'employee_onboarding',
                title: 'Employee Onboarding',
                icon: '👥',
                domain: 'Human Resources',
                description: 'Streamlined onboarding process with document verification, IT setup, manager assignment, and comprehensive training schedules for new employees.'
            },
            {
                name: 'purchase_approval',
                title: 'Purchase Approval',
                icon: '💰',
                domain: 'Finance',
                description: 'Multi-tier approval workflow with intelligent routing based on purchase amounts, finance review processes, and procurement handling.'
            },
            {
                name: 'customer_support',
                title: 'Customer Support',
                icon: '🎫',
                domain: 'Customer Service',
                description: 'Automated support ticket system with priority-based routing, 24-hour escalation protocols, and customer satisfaction tracking.'
            },
            {
                name: 'code_review',
                title: 'Code Review',
                icon: '💻',
                domain: 'Engineering',
                description: 'Comprehensive code review pipeline with automated testing, quality checks, senior developer review, and merge approval workflows.'
            }
        ];
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Premium Workflow Viewer...');
        
        // Apply dark mode if enabled
        if (this.darkMode) {
            document.body.classList.add('dark-mode');
        }
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Add dark mode toggle button
        this.addDarkModeToggle();
        
        // Hide loading after short delay
        setTimeout(() => {
            document.getElementById('loading').classList.add('hidden');
        }, 1000);
        
        // Render workflows
        await this.renderWorkflows();
        
        // Update stats
        this.updateStats();
        
        console.log('✅ Application ready');
    }
    
    async renderWorkflows() {
        const grid = document.getElementById('workflows-grid');
        
        for (const workflow of this.workflows) {
            const data = await this.loadWorkflowData(workflow.name);
            workflow.data = data;
            
            const card = this.createWorkflowCard(workflow);
            grid.appendChild(card);
            
            // Animate card entrance
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        }
    }
    
    async loadWorkflowData(name) {
        try {
            const response = await fetch(`${name}.json`);
            if (!response.ok) throw new Error('Failed to load');
            return await response.json();
        } catch (error) {
            console.error(`Failed to load ${name}:`, error);
            return { steps: [], connections: [], metadata: {} };
        }
    }
    
    createWorkflowCard(workflow) {
        const card = document.createElement('div');
        card.className = 'workflow-card';
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        
        const steps = workflow.data?.steps?.length || 0;
        const connections = workflow.data?.connections?.length || 0;
        
        card.innerHTML = `
            <div class="workflow-header">
                <div class="workflow-icon">${workflow.icon}</div>
                <div class="workflow-info">
                    <h3>${workflow.title}</h3>
                    <div class="workflow-domain">${workflow.domain}</div>
                </div>
            </div>
            
            <p class="workflow-description">${workflow.description}</p>
            
            <div class="workflow-stats">
                <div class="workflow-stat">
                    <span class="workflow-stat-value">${steps}</span>
                    <span class="workflow-stat-label">Steps</span>
                </div>
                <div class="workflow-stat">
                    <span class="workflow-stat-value">${connections}</span>
                    <span class="workflow-stat-label">Connections</span>
                </div>
            </div>
            
            <div class="workflow-actions">
                <button class="btn-view" onclick="app.viewDetails('${workflow.name}')">
                    <span>👁️</span> View Details
                </button>
                <button class="btn-visualize" onclick="app.visualize('${workflow.name}')">
                    <span>📊</span> Visualize
                </button>
            </div>
            
            <div class="workflow-downloads">
                <button class="download-link" onclick="app.viewWorkflowCode('${workflow.name}')">
                    <span>👁️</span> View Code
                </button>
                <button class="download-link" onclick="app.downloadWorkflow('${workflow.name}', 'bpmn')">
                    <span>🔄</span> BPMN
                </button>
            </div>
        `;
        
        return card;
    }
    
    updateStats() {
        const totalSteps = this.workflows.reduce((sum, w) => {
            return sum + (w.data?.steps?.length || 0);
        }, 0);
        
        document.getElementById('total-steps').textContent = totalSteps;
    }
    
    viewDetails(workflowName) {
        const workflow = this.workflows.find(w => w.name === workflowName);
        if (!workflow || !workflow.data) return;
        
        const modal = document.getElementById('modal');
        const content = document.getElementById('modal-content');
        
        content.innerHTML = `
            <div style="margin-bottom: 2rem;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <span style="font-size: 3rem;">${workflow.icon}</span>
                    <div>
                        <h2 style="font-size: 2rem; margin-bottom: 0.25rem;">${workflow.title}</h2>
                        <p style="color: rgba(255,255,255,0.6);">${workflow.domain}</p>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #6366f1; margin-bottom: 0.5rem;">
                        ${workflow.data.steps?.length || 0}
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">Total Steps</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6; margin-bottom: 0.5rem;">
                        ${workflow.data.connections?.length || 0}
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">Connections</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #10b981; margin-bottom: 0.5rem;">
                        ${workflow.data.workflow_id}
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">Workflow ID</div>
                </div>
            </div>
            
            <h3 style="font-size: 1.5rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>📋</span> Workflow Steps
            </h3>
            
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                ${workflow.data.steps?.map((step, index) => `
                    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1;">
                        <div style="display: flex; align-items: start; gap: 1rem;">
                            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;">
                                ${index + 1}
                            </div>
                            <div style="flex: 1;">
                                <h4 style="font-size: 1.1rem; margin-bottom: 0.5rem;">${step.name}</h4>
                                <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                                    <span style="background: rgba(99,102,241,0.2); color: #a5b4fc; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.85rem;">
                                        ${step.type}
                                    </span>
                                    <span style="background: rgba(139,92,246,0.2); color: #c4b5fd; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.85rem;">
                                        ${step.assigned_role || 'N/A'}
                                    </span>
                                </div>
                                ${step.description ? `<p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; line-height: 1.6;">${step.description}</p>` : ''}
                            </div>
                        </div>
                    </div>
                `).join('') || '<p style="color: rgba(255,255,255,0.6);">No steps defined</p>'}
            </div>
            
            <h3 style="font-size: 1.5rem; margin: 2rem 0 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <span>🔗</span> Connections
            </h3>
            
            <div style="display: grid; gap: 0.75rem;">
                ${workflow.data.connections?.map((conn, index) => `
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem 1.5rem; border-radius: 12px; display: flex; align-items: center; gap: 1rem;">
                        <span style="background: #8b5cf6; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 700;">
                            ${index + 1}
                        </span>
                        <code style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 8px; font-family: monospace;">${conn.from_step}</code>
                        <span style="color: #6366f1; font-size: 1.5rem;">→</span>
                        <code style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 8px; font-family: monospace;">${conn.to_step}</code>
                    </div>
                `).join('') || '<p style="color: rgba(255,255,255,0.6);">No connections defined</p>'}
            </div>
        `;
        
        modal.classList.add('active');
    }
    
    visualize(workflowName) {
        const workflow = this.workflows.find(w => w.name === workflowName);
        if (!workflow || !workflow.data) return;
        
        const modal = document.getElementById('modal');
        const content = document.getElementById('modal-content');
        
        const svg = this.generateFlowchart(workflow.data);
        
        content.innerHTML = `
            <div style="margin-bottom: 2rem;">
                <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">
                    ${workflow.icon} ${workflow.title} - Flowchart
                </h2>
                <p style="color: rgba(255,255,255,0.6);">Visual representation of the workflow structure</p>
            </div>
            
            <div style="background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 12px; overflow-x: auto;">
                ${svg}
            </div>
        `;
        
        modal.classList.add('active');
    }
    
    generateFlowchart(data, animated = false) {
        const steps = data.steps || [];
        const connections = data.connections || [];
        const boxWidth = 240;
        const boxHeight = 90;
        const padding = 80;
        const verticalSpacing = 160;
        const horizontalSpacing = 140;
        
        // Create position map for steps (vertical flow layout)
        const stepPositions = {};
        steps.forEach((step, index) => {
            const x = padding + (index % 2) * (boxWidth + horizontalSpacing);
            const y = padding + Math.floor(index / 2) * verticalSpacing;
            stepPositions[step.id] = { x, y, index };
        });
        
        let svgContent = '';
        let maxY = 0;
        
        // Draw connection arrows first (so they appear behind boxes)
        connections.forEach(conn => {
            const from = stepPositions[conn.from_step];
            const to = stepPositions[conn.to_step];
            
            if (from && to) {
                const fromX = from.x + boxWidth / 2;
                const fromY = from.y + boxHeight;
                const toX = to.x + boxWidth / 2;
                const toY = to.y;
                
                // Draw smooth curved arrow
                const controlOffset = Math.abs(toY - fromY) * 0.4;
                const midY = (fromY + toY) / 2;
                
                svgContent += `
                    <path d="M ${fromX} ${fromY} C ${fromX} ${fromY + controlOffset}, ${toX} ${toY - controlOffset}, ${toX} ${toY}" 
                          stroke="#6366f1" stroke-width="3" fill="none" opacity="0.7" stroke-linecap="round"/>
                    <polygon points="${toX},${toY} ${toX-7},${toY-12} ${toX+7},${toY-12}" 
                             fill="#6366f1" opacity="0.8"/>
                `;
                
                // Add label if exists (positioned to the side to avoid overlap)
                if (conn.label) {
                    const labelX = fromX > toX ? (fromX + toX) / 2 + 40 : (fromX + toX) / 2 - 40;
                    const labelY = midY - 5;
                    svgContent += `
                        <rect x="${labelX - 50}" y="${labelY - 14}" width="100" height="20" 
                              rx="4" fill="rgba(30,41,59,0.8)"/>
                        <text x="${labelX}" y="${labelY}" text-anchor="middle" 
                              fill="#d1d5db" font-size="11" font-weight="500">
                            ${conn.label.length > 15 ? conn.label.substring(0, 15) + '...' : conn.label}
                        </text>
                    `;
                }
            }
        });
        
        // Draw step boxes with shadows
        steps.forEach((step, index) => {
            const pos = stepPositions[step.id];
            maxY = Math.max(maxY, pos.y + boxHeight);
            
            const color = this.getStepColor(step.type);
            
            // Drop shadow
            svgContent += `
                <rect x="${pos.x + 4}" y="${pos.y + 4}" width="${boxWidth}" height="${boxHeight}" 
                      rx="16" fill="rgba(0,0,0,0.3)"/>
            `;
            
            // Main box with hover effects
            const stepId = `step-${index}`;
            const animationDelay = animated ? index * 0.3 : 0;
            
            svgContent += `
                <g class="workflow-step" data-step-id="${stepId}" 
                   style="${animated ? `animation: stepFadeIn 0.6s ease-out ${animationDelay}s both;` : ''}">
                    <rect x="${pos.x}" y="${pos.y}" width="${boxWidth}" height="${boxHeight}" 
                          rx="16" fill="${color}" stroke="rgba(255,255,255,0.2)" stroke-width="2" 
                          class="step-box" style="transition: all 0.3s; cursor: pointer;"/>
                    <text x="${pos.x + boxWidth/2}" y="${pos.y + 38}" 
                          text-anchor="middle" fill="white" font-weight="700" font-size="15" pointer-events="none">
                        ${step.name.length > 22 ? step.name.substring(0, 22) + '...' : step.name}
                    </text>
                    <text x="${pos.x + boxWidth/2}" y="${pos.y + 65}" 
                          text-anchor="middle" fill="rgba(255,255,255,0.7)" font-size="12" font-weight="500" pointer-events="none">
                        ${step.type.replace('_', ' ')}
                    </text>
                    <title>${step.name}\nType: ${step.type}\nRole: ${step.assigned_role || 'N/A'}${step.description ? '\n' + step.description : ''}</title>
                </g>
            `;
        });
        
        const width = (boxWidth + horizontalSpacing) * 2 + padding * 2;
        const height = maxY + padding + 40;
        
        return `
            <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#6366f1;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#8b5cf6;stop-opacity:1" />
                    </linearGradient>
                </defs>
                ${svgContent}
            </svg>
        `;
    }
    
    getStepColor(type) {
        const colors = {
            'manual_task': '#6366f1',
            'automated_task': '#10b981',
            'decision_gate': '#f59e0b',
            'approval': '#3b82f6',
            'notification': '#8b5cf6',
            'timer': '#ef4444',
            'integration': '#06b6d4'
        };
        return colors[type] || '#64748b';
    }
    
    getStepIcon(type) {
        const icons = {
            'manual_task': '👤',
            'automated_task': '⚙️',
            'decision_gate': '🔀',
            'approval': '✅',
            'notification': '📧',
            'timer': '⏱️',
            'integration': '🔗',
            'review': '🔍',
            'submission': '📤',
            'processing': '⚡',
            'validation': '✔️',
            'escalation': '⬆️'
        };
        return icons[type] || '📋';
    }
    
    // LocalStorage Management
    loadSavedWorkflows() {
        try {
            const saved = localStorage.getItem('customWorkflows');
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.error('Failed to load saved workflows:', error);
            return [];
        }
    }
    
    saveWorkflow(workflow, name) {
        const workflowData = {
            id: Date.now(),
            name: name || workflow.metadata?.name || 'Untitled Workflow',
            workflow: workflow,
            createdAt: new Date().toISOString()
        };
        
        this.savedWorkflows.push(workflowData);
        localStorage.setItem('customWorkflows', JSON.stringify(this.savedWorkflows));
        this.showNotification('✅ Workflow saved successfully!', 'success');
        return workflowData;
    }
    
    deleteSavedWorkflow(id) {
        this.savedWorkflows = this.savedWorkflows.filter(w => w.id !== id);
        localStorage.setItem('customWorkflows', JSON.stringify(this.savedWorkflows));
        this.showNotification('🗑️ Workflow deleted', 'info');
    }
    
    // Keyboard Shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + S: Save current workflow
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                if (this.currentWorkflow) {
                    const name = prompt('Save workflow as:', this.currentWorkflow.metadata?.name || 'My Workflow');
                    if (name) this.saveWorkflow(this.currentWorkflow, name);
                }
            }
            
            // Ctrl/Cmd + D: Toggle dark mode
            if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
                e.preventDefault();
                this.toggleDarkMode();
            }
            
            // Escape: Close modal
            if (e.key === 'Escape') {
                this.closeModal();
            }
            
            // Ctrl/Cmd + E: Export as image
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                this.exportAsImage();
            }
        });
    }
    
    // Dark Mode
    addDarkModeToggle() {
        const nav = document.querySelector('.nav-menu');
        if (!nav) return;
        
        const toggle = document.createElement('button');
        toggle.innerHTML = this.darkMode ? '☀️' : '🌙';
        toggle.className = 'dark-mode-toggle';
        toggle.style.cssText = `
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 1.2rem;
            transition: all 0.3s;
        `;
        toggle.onclick = () => this.toggleDarkMode();
        nav.appendChild(toggle);
    }
    
    toggleDarkMode() {
        this.darkMode = !this.darkMode;
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', this.darkMode);
        
        const toggle = document.querySelector('.dark-mode-toggle');
        if (toggle) toggle.innerHTML = this.darkMode ? '☀️' : '🌙';
        
        this.showNotification(this.darkMode ? '🌙 Dark mode enabled' : '☀️ Light mode enabled', 'info');
    }
    
    // Export as Image
    async exportAsImage() {
        const timeline = document.getElementById('timeline-wrapper');
        if (!timeline) {
            this.showNotification('⚠️ Open a visualization first', 'info');
            return;
        }
        
        try {
            // Use html2canvas if available, otherwise fallback
            if (typeof html2canvas !== 'undefined') {
                const canvas = await html2canvas(timeline);
                const link = document.createElement('a');
                link.download = `workflow_${Date.now()}.png`;
                link.href = canvas.toDataURL();
                link.click();
                this.showNotification('📸 Image exported!', 'success');
            } else {
                // Fallback: SVG export
                this.exportAsSVG();
            }
        } catch (error) {
            console.error('Export failed:', error);
            this.showNotification('❌ Export failed. Try SVG export.', 'info');
        }
    }
    
    exportAsSVG() {
        const svg = document.querySelector('#svg-layer-wf');
        if (!svg) return;
        
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svg);
        const blob = new Blob([svgString], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `workflow_${Date.now()}.svg`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showNotification('📸 SVG exported!', 'success');
    }
    
    closeModal() {
        document.getElementById('modal').classList.remove('active');
    }
    
    useExample(text) {
        document.getElementById('workflow-prompt').value = text;
        document.getElementById('workflow-prompt').focus();
        // Scroll to the input
        document.getElementById('generate').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    async generateWorkflow() {
        const prompt = document.getElementById('workflow-prompt').value.trim();
        const domain = document.getElementById('workflow-domain').value;
        const resultDiv = document.getElementById('generation-result');
        const generateBtn = document.querySelector('.btn-generate');
        
        if (!prompt) {
            alert('Please enter a workflow description');
            return;
        }
        
        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="spinner-ring"></span><span>Generating...</span>';
        
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `
            <div class="generating-spinner">
                <div class="spinner-ring"></div>
                <p>🤖 Gemini AI is analyzing your request...</p>
            </div>
        `;
        
        try {
            // Call the API server to generate workflow
            const response = await fetch('http://localhost:5000/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    description: prompt,
                    domain: domain
                })
            });
            
            if (!response.ok) {
                throw new Error('Generation failed');
            }
            
            const result = await response.json();
            
            if (result.success) {
                this.displayGeneratedWorkflow(result.workflow, result.workflow_id);
            } else {
                throw new Error(result.error || 'Unknown error');
            }
            
        } catch (error) {
            console.error('Generation error:', error);
            resultDiv.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                    <h3 style="margin-bottom: 1rem;">Generation Failed</h3>
                    <p style="color: rgba(255,255,255,0.6);">
                        The MCP server is not running or couldn't process your request.
                    </p>
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-top: 1rem;">
                        Tip: Run 'python main_demo.py' to start the MCP server
                    </p>
                </div>
            `;
        } finally {
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<span class="btn-icon">⚡</span><span>Generate Workflow</span>';
        }
    }
    
    displayGeneratedWorkflow(workflow, workflowId) {
        const resultDiv = document.getElementById('generation-result');
        this.currentWorkflow = workflow; // Store for keyboard shortcuts
        
        const steps = workflow.steps?.length || 0;
        const connections = workflow.connections?.length || 0;
        
        resultDiv.innerHTML = `
            <div class="result-header">
                <span class="result-icon">✅</span>
                <div>
                    <div class="result-title">${workflow.metadata?.name || 'Generated Workflow'}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">ID: ${workflowId}</div>
                </div>
            </div>
            
            <div class="result-stats">
                <div class="result-stat">
                    <span class="result-stat-value">${steps}</span>
                    <span class="result-stat-label">Steps Created</span>
                </div>
                <div class="result-stat">
                    <span class="result-stat-value">${connections}</span>
                    <span class="result-stat-label">Connections</span>
                </div>
                <div class="result-stat">
                    <span class="result-stat-value">✅</span>
                    <span class="result-stat-label">Validated</span>
                </div>
            </div>
            
            <div style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
                <h4 style="margin-bottom: 1rem;">📋 Generated Steps:</h4>
                <ol style="padding-left: 1.5rem; line-height: 2;">
                    ${workflow.steps?.map(step => `
                        <li style="color: rgba(255,255,255,0.8);">
                            <strong>${step.name}</strong>
                            <span style="color: rgba(255,255,255,0.5); font-size: 0.9rem;"> - ${step.type}</span>
                        </li>
                    `).join('') || '<li>No steps</li>'}
                </ol>
            </div>
            
            <div class="result-actions">
                <button class="btn-view" onclick="app.viewGeneratedDetails(${JSON.stringify(workflow).replace(/"/g, '&quot;')})">
                    <span>👁️</span> View Full Details
                </button>
                <button class="btn-visualize" onclick="app.visualizeGenerated(${JSON.stringify(workflow).replace(/"/g, '&quot;')})" 
                        style="background: linear-gradient(135deg, #f093fb, #f5576c); color: white; border: none;">
                    <span>✨</span> Visualize (Animated)
                </button>
            </div>
            
            <button onclick="app.saveWorkflow(${JSON.stringify(workflow).replace(/"/g, '&quot;')})" 
                    style="width: 100%; margin-top: 1rem; padding: 0.75rem; background: linear-gradient(135deg, #10b981, #059669); border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                💾 Save Workflow (Ctrl+S)
            </button>
            
            <div id="code-viewer-${Date.now()}" style="margin-top: 1.5rem;">
                <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <button onclick="app.toggleCodeFormat('code-viewer-${Date.now()}', 'json')" 
                            class="format-toggle" data-format="json"
                            style="flex: 1; padding: 0.75rem; background: rgba(16,185,129,0.3); border: 2px solid #10b981; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                        📄 JSON
                    </button>
                    <button onclick="app.toggleCodeFormat('code-viewer-${Date.now()}', 'yaml')" 
                            class="format-toggle" data-format="yaml"
                            style="flex: 1; padding: 0.75rem; background: rgba(59,130,246,0.2); border: 1px solid #3b82f6; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                        📋 YAML
                    </button>
                    <button onclick="app.toggleCodeFormat('code-viewer-${Date.now()}', 'bpmn')" 
                            class="format-toggle" data-format="bpmn"
                            style="flex: 1; padding: 0.75rem; background: rgba(236,72,153,0.2); border: 1px solid #ec4899; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                        🔄 BPMN
                    </button>
                </div>
                <div style="position: relative;">
                    <button onclick="app.copyCode('code-content-${Date.now()}')" 
                            style="position: absolute; top: 8px; right: 8px; padding: 0.5rem 1rem; background: rgba(99,102,241,0.3); border: 1px solid #6366f1; border-radius: 6px; color: white; cursor: pointer; font-size: 0.875rem; z-index: 1;">
                        📋 Copy
                    </button>
                    <pre id="code-content-${Date.now()}" style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 8px; overflow-x: auto; max-height: 400px; margin: 0; font-family: 'Courier New', monospace; font-size: 0.875rem; line-height: 1.5; color: #e5e7eb;" data-workflow='${JSON.stringify(workflow).replace(/'/g, "&#39;")}'>${JSON.stringify(workflow, null, 2)}</pre>
                </div>
            </div>
        `;
    }
    
    viewGeneratedDetails(workflow) {
        const modal = document.getElementById('modal');
        const content = document.getElementById('modal-content');
        
        content.innerHTML = `
            <h2 style="font-size: 2rem; margin-bottom: 2rem;">Generated Workflow Details</h2>
            
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                ${workflow.steps?.map((step, index) => `
                    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1;">
                        <div style="display: flex; gap: 1rem;">
                            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700;">
                                ${index + 1}
                            </div>
                            <div>
                                <h4 style="margin-bottom: 0.5rem;">${step.name}</h4>
                                <div style="display: flex; gap: 0.5rem;">
                                    <span style="background: rgba(99,102,241,0.2); color: #a5b4fc; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.85rem;">
                                        ${step.type}
                                    </span>
                                    <span style="background: rgba(139,92,246,0.2); color: #c4b5fd; padding: 0.25rem 0.75rem; border-radius: 50px; font-size: 0.85rem;">
                                        ${step.assigned_role || 'N/A'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        modal.classList.add('active');
    }
    
    visualizeGenerated(workflow) {
        const modal = document.getElementById('modal');
        const content = document.getElementById('modal-content');
        
        // Show loading state first
        content.innerHTML = `
            <h2 style="font-size: 2rem; margin-bottom: 2rem;">🎨 Building Visualization...</h2>
            <div style="background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 12px; text-align: center;">
                <div class="spinner-ring" style="margin: 3rem auto;"></div>
                <p style="color: rgba(255,255,255,0.6);">Constructing workflow components...</p>
            </div>
        `;
        
        modal.classList.add('active');
        
        // Generate timeline visualization with animation
        setTimeout(() => {
            this.renderTimelineAnimation(workflow, content);
        }, 800);
    }
    
    renderTimelineAnimation(workflow, container) {
        const steps = workflow.steps || [];
        
        container.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem; margin: 0;">✨ Workflow Timeline</h2>
                <div style="display: flex; gap: 1rem;">
                    <button onclick="app.exportAsImage()" style="padding: 0.5rem 1rem; background: linear-gradient(135deg, #ec4899, #be185d); border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        📸 Export Image (Ctrl+E)
                    </button>
                    <button onclick="app.saveWorkflow(app.currentWorkflow)" style="padding: 0.5rem 1rem; background: linear-gradient(135deg, #10b981, #059669); border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        💾 Save (Ctrl+S)
                    </button>
                </div>
            </div>
            <div id="timeline-wrapper" style="background: rgba(255,255,255,0.05); padding: 3rem; border-radius: 12px; overflow-x: auto; position: relative;">
                <style>
                    .timeline-container-wf {
                        position: relative;
                        display: flex;
                        gap: 120px;
                        padding: 50px 20px;
                        min-width: max-content;
                    }
                    
                    #svg-layer-wf {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        z-index: 0;
                        pointer-events: none;
                    }
                    
                    .wf-node {
                        position: relative;
                        width: 80px;
                        height: 80px;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        opacity: 0;
                        transform: scale(0);
                        transition: all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                        z-index: 2;
                    }
                    
                    .wf-node.active {
                        opacity: 1;
                        transform: scale(1);
                    }
                    
                    .wf-circle {
                        width: 80px;
                        height: 80px;
                        border-radius: 50%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        font-weight: 700;
                        font-size: 1.8rem;
                        color: white;
                        position: relative;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                        cursor: pointer;
                        transition: all 0.3s;
                    }
                    
                    .wf-circle:hover {
                        transform: scale(1.15);
                        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
                    }
                    
                    .wf-label {
                        position: absolute;
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        color: white;
                        padding: 8px 16px;
                        font-size: 13px;
                        border-radius: 8px;
                        white-space: nowrap;
                        opacity: 0;
                        transition: opacity 0.5s ease 0.2s;
                        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
                        max-width: 200px;
                        text-overflow: ellipsis;
                        overflow: hidden;
                    }
                    
                    .wf-node.active .wf-label {
                        opacity: 1;
                    }
                    
                    .wf-stem {
                        position: absolute;
                        width: 2px;
                        height: 30px;
                        background: linear-gradient(180deg, rgba(255,255,255,0.6), transparent);
                        z-index: 1;
                    }
                    
                    .wf-stem::after {
                        content: '';
                        position: absolute;
                        width: 6px;
                        height: 6px;
                        background-color: rgba(255,255,255,0.8);
                        border-radius: 50%;
                        top: -3px;
                        left: 50%;
                        transform: translateX(-50%);
                    }
                    
                    .wf-node:nth-child(odd) {
                        align-self: flex-start;
                    }
                    
                    .wf-node:nth-child(odd) .wf-label {
                        bottom: 100px;
                        left: 50%;
                        transform: translateX(-50%);
                    }
                    
                    .wf-node:nth-child(odd) .wf-stem {
                        bottom: 80px;
                        left: 50%;
                        transform: translateX(-50%);
                    }
                    
                    .wf-node:nth-child(even) {
                        align-self: flex-end;
                        margin-top: 120px;
                    }
                    
                    .wf-node:nth-child(even) .wf-label {
                        top: 100px;
                        left: 50%;
                        transform: translateX(-50%);
                    }
                    
                    .wf-node:nth-child(even) .wf-stem {
                        top: 80px;
                        left: 50%;
                        transform: translateX(-50%) rotate(180deg);
                    }
                    
                    .wf-type-badge {
                        position: absolute;
                        bottom: -8px;
                        left: 50%;
                        transform: translateX(-50%);
                        background: rgba(0,0,0,0.8);
                        color: white;
                        padding: 2px 8px;
                        border-radius: 10px;
                        font-size: 10px;
                        white-space: nowrap;
                        opacity: 0;
                        transition: opacity 0.5s ease 0.3s;
                    }
                    
                    .wf-node.active .wf-type-badge {
                        opacity: 1;
                    }
                    
                    #svg-layer-wf path {
                        fill: none;
                        stroke: rgba(99,102,241,0.6);
                        stroke-width: 3;
                        stroke-linecap: round;
                    }
                </style>
                
                <div class="timeline-container-wf" id="timeline-container-wf">
                    <svg id="svg-layer-wf"></svg>
                    ${steps.map((step, i) => `
                        <div class="wf-node" data-index="${i}">
                            <div class="wf-label" title="${step.description || step.name}">${step.name}</div>
                            <div class="wf-stem"></div>
                            <div class="wf-circle" style="background: ${this.getStepColor(step.type)};" 
                                 title="${step.name}\nType: ${step.type}\nRole: ${step.assigned_role || 'N/A'}">
                                ${this.getStepIcon(step.type)}
                            </div>
                            <div class="wf-type-badge">${step.type.replace('_', ' ')}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        // Run animation after render
        setTimeout(() => this.animateTimeline(), 100);
    }
    
    async animateTimeline() {
        const nodes = document.querySelectorAll('.wf-node');
        const svgLayer = document.getElementById('svg-layer-wf');
        const container = document.getElementById('timeline-container-wf');
        
        if (!nodes.length || !svgLayer || !container) return;
        
        // Set SVG dimensions
        svgLayer.setAttribute('width', container.offsetWidth);
        svgLayer.setAttribute('height', container.offsetHeight);
        
        const delay = ms => new Promise(res => setTimeout(res, ms));
        
        // Helper to draw animated line between nodes
        const createLine = (el1, el2) => {
            const rect1 = el1.querySelector('.wf-circle').getBoundingClientRect();
            const rect2 = el2.querySelector('.wf-circle').getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            const x1 = rect1.left + rect1.width / 2 - containerRect.left;
            const y1 = rect1.top + rect1.height / 2 - containerRect.top;
            const x2 = rect2.left + rect2.width / 2 - containerRect.left;
            const y2 = rect2.top + rect2.height / 2 - containerRect.top;
            
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            
            // Create curved path
            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;
            const d = `M ${x1} ${y1} Q ${midX} ${midY + 20} ${x2} ${y2}`;
            path.setAttribute("d", d);
            
            // Setup animation
            const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)) * 1.2;
            path.style.strokeDasharray = length;
            path.style.strokeDashoffset = length;
            path.style.transition = "stroke-dashoffset 0.8s ease-in-out";
            
            svgLayer.appendChild(path);
            return path;
        };
        
        // Animate sequence: node → line → node → line...
        nodes[0].classList.add('active');
        await delay(600);
        
        for (let i = 0; i < nodes.length - 1; i++) {
            const line = createLine(nodes[i], nodes[i + 1]);
            line.getBoundingClientRect(); // Force reflow
            line.style.strokeDashoffset = "0";
            
            await delay(700);
            
            nodes[i + 1].classList.add('active');
            await delay(400);
        }
    }
    
    async downloadWorkflow(workflowName, format) {
        console.log('Download requested:', workflowName, format);
        try {
            // Load the workflow data
            const response = await fetch(`${workflowName}.json`);
            if (!response.ok) {
                throw new Error(`Failed to fetch workflow: ${response.status}`);
            }
            const workflow = await response.json();
            console.log('Workflow loaded:', workflow);
            
            // Download using the same function as generated workflows
            this.downloadGenerated(workflow, format);
        } catch (error) {
            console.error('Download failed:', error);
            alert(`Failed to download workflow: ${error.message}\n\nTip: Try right-clicking the workflow card and selecting "View Details" then copy the data.`);
        }
    }
    
    async toggleCodeFormat(viewerId, format) {
        const viewer = document.getElementById(viewerId);
        if (!viewer) return;
        
        const codeElement = viewer.querySelector('pre');
        const workflow = JSON.parse(codeElement.getAttribute('data-workflow').replace(/&#39;/g, "'"));
        
        // Update button styles
        const buttons = viewer.querySelectorAll('.format-toggle');
        buttons.forEach(btn => {
            const btnFormat = btn.getAttribute('data-format');
            if (btnFormat === format) {
                // Active state
                if (format === 'json') {
                    btn.style.background = 'rgba(16,185,129,0.3)';
                    btn.style.border = '2px solid #10b981';
                } else if (format === 'yaml') {
                    btn.style.background = 'rgba(59,130,246,0.3)';
                    btn.style.border = '2px solid #3b82f6';
                } else if (format === 'bpmn') {
                    btn.style.background = 'rgba(236,72,153,0.3)';
                    btn.style.border = '2px solid #ec4899';
                }
            } else {
                // Inactive state
                if (btnFormat === 'json') {
                    btn.style.background = 'rgba(16,185,129,0.2)';
                    btn.style.border = '1px solid #10b981';
                } else if (btnFormat === 'yaml') {
                    btn.style.background = 'rgba(59,130,246,0.2)';
                    btn.style.border = '1px solid #3b82f6';
                } else if (btnFormat === 'bpmn') {
                    btn.style.background = 'rgba(236,72,153,0.2)';
                    btn.style.border = '1px solid #ec4899';
                }
            }
        });
        
        // Update content
        if (format === 'json') {
            codeElement.textContent = JSON.stringify(workflow, null, 2);
        } else if (format === 'yaml') {
            codeElement.textContent = this.jsonToYaml(workflow);
        } else if (format === 'bpmn') {
            // Show loading
            codeElement.textContent = 'Loading BPMN...';
            try {
                const response = await fetch('http://localhost:5000/api/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        workflow: workflow,
                        format: 'bpmn'
                    })
                });
                const bpmnContent = await response.text();
                codeElement.textContent = bpmnContent;
            } catch (error) {
                codeElement.textContent = `Error loading BPMN: ${error.message}`;
            }
        }
    }
    
    copyCode(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        navigator.clipboard.writeText(element.textContent).then(() => {
            this.showNotification('✅ Copied to clipboard!', 'success');
        }).catch(() => {
            this.showNotification('❌ Failed to copy', 'error');
        });
    }
    
    async viewWorkflowCode(workflowName) {
        try {
            const response = await fetch(`${workflowName}.json`);
            if (!response.ok) throw new Error('Failed to load workflow');
            const workflow = await response.json();
            
            const modal = document.getElementById('modal');
            const content = document.getElementById('modal-content');
            const viewerId = `code-viewer-modal-${Date.now()}`;
            
            content.innerHTML = `
                <h2 style="font-size: 2rem; margin-bottom: 2rem;">${workflow.metadata?.title || workflowName} - Code</h2>
                
                <div id="${viewerId}" style="margin-top: 1.5rem;">
                    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <button onclick="app.toggleCodeFormat('${viewerId}', 'json')" 
                                class="format-toggle" data-format="json"
                                style="flex: 1; padding: 0.75rem; background: rgba(16,185,129,0.3); border: 2px solid #10b981; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                            📄 JSON
                        </button>
                        <button onclick="app.toggleCodeFormat('${viewerId}', 'yaml')" 
                                class="format-toggle" data-format="yaml"
                                style="flex: 1; padding: 0.75rem; background: rgba(59,130,246,0.2); border: 1px solid #3b82f6; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                            📋 YAML
                        </button>
                        <button onclick="app.toggleCodeFormat('${viewerId}', 'bpmn')" 
                                class="format-toggle" data-format="bpmn"
                                style="flex: 1; padding: 0.75rem; background: rgba(236,72,153,0.2); border: 1px solid #ec4899; border-radius: 8px; color: white; cursor: pointer; font-weight: 600;">
                            🔄 BPMN
                        </button>
                    </div>
                    <div style="position: relative;">
                        <button onclick="app.copyCode('code-content-modal-${Date.now()}')" 
                                style="position: absolute; top: 8px; right: 8px; padding: 0.5rem 1rem; background: rgba(99,102,241,0.3); border: 1px solid #6366f1; border-radius: 6px; color: white; cursor: pointer; font-size: 0.875rem; z-index: 1;">
                            📋 Copy
                        </button>
                        <pre id="code-content-modal-${Date.now()}" style="background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 8px; overflow-x: auto; max-height: 500px; margin: 0; font-family: 'Courier New', monospace; font-size: 0.875rem; line-height: 1.5; color: #e5e7eb;" data-workflow='${JSON.stringify(workflow).replace(/'/g, "&#39;")}'>${JSON.stringify(workflow, null, 2)}</pre>
                    </div>
                </div>
            `;
            
            modal.classList.add('active');
        } catch (error) {
            console.error('Failed to view code:', error);
            this.showNotification('❌ Failed to load workflow code', 'error');
        }
    }
    
    downloadGenerated(workflow, format) {
        console.log('Generating download:', format);
        let content, filename, mimeType;
        
        const timestamp = Date.now();
        const workflowName = workflow.metadata?.name?.replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'workflow';
        
        if (format === 'json') {
            content = JSON.stringify(workflow, null, 2);
            filename = `output/${workflowName}_${timestamp}.json`;
            mimeType = 'application/json';
        } else if (format === 'yaml') {
            // Simple YAML conversion
            content = this.jsonToYaml(workflow);
            filename = `output/${workflowName}_${timestamp}.yaml`;
            mimeType = 'text/yaml';
        } else if (format === 'bpmn') {
            // For BPMN, call the export API
            this.downloadBPMN(workflow);
            return;
        }
        
        try {
            console.log('Creating file with', content.length, 'bytes');
            
            // Create blob and download immediately
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            
            // Create temporary anchor and trigger download
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.style.display = 'none';
            
            // Add to DOM, click, and remove immediately
            document.body.appendChild(a);
            
            // Force immediate download
            setTimeout(() => {
                a.click();
                console.log('Download triggered:', filename);
                
                // Cleanup
                setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }, 150);
                
                // Show success notification with folder path
                this.showNotification(`✅ Saved to: ${filename}`, 'success');
            }, 10);
            
        } catch (error) {
            console.error('Download error:', error);
            // Show content in modal as fallback
            this.showContentModal(content, filename);
        }
    }
    
    showContentModal(content, filename) {
        const modal = document.getElementById('modal');
        const modalContent = document.getElementById('modal-content');
        
        modalContent.innerHTML = `
            <div style="margin-bottom: 2rem;">
                <h2 style="font-size: 1.8rem; margin-bottom: 1rem;">📄 ${filename}</h2>
                <p style="color: rgba(255,255,255,0.6); margin-bottom: 1.5rem;">
                    Copy the content below:
                </p>
                <textarea 
                    readonly
                    style="
                        width: 100%;
                        height: 400px;
                        background: rgba(0,0,0,0.4);
                        border: 1px solid rgba(255,255,255,0.2);
                        border-radius: 12px;
                        padding: 1rem;
                        color: white;
                        font-family: 'Courier New', monospace;
                        font-size: 0.9rem;
                        resize: vertical;
                    "
                    onclick="this.select()"
                >${content}</textarea>
                <button 
                    onclick="navigator.clipboard.writeText(\`${content.replace(/`/g, '\\`')}\`).then(() => app.showNotification('✅ Copied to clipboard!', 'success'))"
                    style="
                        margin-top: 1rem;
                        padding: 0.75rem 1.5rem;
                        background: linear-gradient(135deg, #6366f1, #8b5cf6);
                        border: none;
                        border-radius: 12px;
                        color: white;
                        font-weight: 600;
                        cursor: pointer;
                    "
                >
                    📋 Copy to Clipboard
                </button>
            </div>
        `;
        
        modal.classList.add('active');
    }
    
    showNotification(message, type = 'info') {
        // Simple notification
        const notification = document.createElement('div');
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: ${type === 'success' ? '#10b981' : '#6366f1'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            z-index: 10000;
            animation: slideIn 0.3s ease-out;
        `;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => document.body.removeChild(notification), 300);
        }, 3000);
    }
    
    async downloadBPMN(workflow) {
        try {
            const response = await fetch('http://localhost:5000/api/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    workflow: workflow,
                    format: 'bpmn'
                })
            });
            
            const content = await response.text();
            const blob = new Blob([content], { type: 'application/xml' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `workflow_${Date.now()}.bpmn`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error('BPMN download failed:', error);
            alert('Failed to download BPMN. Please try again.');
        }
    }
    
    jsonToYaml(obj, indent = 0) {
        let yaml = '';
        const spacing = '  '.repeat(indent);
        
        for (const [key, value] of Object.entries(obj)) {
            if (Array.isArray(value)) {
                yaml += `${spacing}${key}:\n`;
                value.forEach(item => {
                    if (typeof item === 'object') {
                        yaml += `${spacing}- \n`;
                        yaml += this.jsonToYaml(item, indent + 2);
                    } else {
                        yaml += `${spacing}- ${item}\n`;
                    }
                });
            } else if (typeof value === 'object' && value !== null) {
                yaml += `${spacing}${key}:\n`;
                yaml += this.jsonToYaml(value, indent + 1);
            } else {
                yaml += `${spacing}${key}: ${value}\n`;
            }
        }
        
        return yaml;
    }
}

// Initialize app
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new PremiumWorkflowApp();
});