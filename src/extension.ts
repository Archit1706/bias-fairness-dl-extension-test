// src/extension.ts
import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';
import * as path from 'path';
import * as fs from 'fs';

let serverProcess: ChildProcess | null = null;
const SERVER_PORT = 8765;

export async function activate(context: vscode.ExtensionContext) {
    console.log('Fairness DL Extension activating...');

    // Start Python backend
    await startBackend(context);

    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('fairness-dl.analyzeDataset', analyzeDataset));

    // Register file context menu
    context.subscriptions.push(
        vscode.commands.registerCommand('fairness-dl.analyzeFromMenu', async (uri: vscode.Uri) => {
            await analyzeDataset(uri);
        }),
    );
}

async function startBackend(context: vscode.ExtensionContext) {
    const pythonPath = vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath') || 'python3';
    const backendPath = context.asAbsolutePath('python_backend');

    console.log(`Starting backend at: ${backendPath}`);

    serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'bias_server:app', '--port', String(SERVER_PORT)], {
        cwd: backendPath,
        shell: true,
    });

    serverProcess.stdout?.on('data', (data) => {
        console.log(`Backend: ${data}`);
    });

    serverProcess.stderr?.on('data', (data) => {
        console.error(`Backend Error: ${data}`);
    });

    // Wait for server to start
    await new Promise((resolve) => setTimeout(resolve, 5000));

    // Verify server is running
    try {
        const response = await axios.get(`http://localhost:${SERVER_PORT}/`);
        vscode.window.showInformationMessage('✅ Fairness backend started successfully');
    } catch (error) {
        vscode.window.showErrorMessage('❌ Failed to start backend. Check Python environment.');
    }
}

async function analyzeDataset(uri: vscode.Uri) {
    const filePath = uri.fsPath;

    // Verify it's a CSV
    if (!filePath.endsWith('.csv')) {
        vscode.window.showErrorMessage('Please select a CSV file');
        return;
    }

    // Ask user for label column
    const labelColumn = await vscode.window.showInputBox({
        prompt: 'Enter the name of the label/target column',
        placeHolder: 'e.g., income, label, target',
    });

    if (!labelColumn) {
        return;
    }

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Analyzing dataset for fairness...',
            cancellable: false,
        },
        async (progress) => {
            try {
                // Step 1: Train model
                progress.report({ increment: 0, message: 'Training DNN (this may take 2-3 minutes)...' });

                const trainResponse = await axios.post(
                    `http://localhost:${SERVER_PORT}/train`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        num_epochs: 30,
                    },
                    {
                        timeout: 300000, // 5 minute timeout
                    },
                );

                vscode.window.showInformationMessage(
                    `✅ Model trained: ${trainResponse.data.accuracy.toFixed(2)}% accuracy`,
                );

                // Step 2: Analyze
                progress.report({ increment: 40, message: 'Computing QID metrics...' });

                const protectedFeatures = trainResponse.data.protected_features;

                // Create protected values dict (assume binary for now)
                const protectedValues: any = {};
                protectedFeatures.forEach((_feature: string, idx: number) => {
                    protectedValues[idx.toString()] = [0.0, 1.0];
                });

                const analyzeResponse = await axios.post(
                    `http://localhost:${SERVER_PORT}/analyze`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        sensitive_features: protectedFeatures,
                        protected_values: protectedValues,
                        max_samples: 500,
                    },
                    {
                        timeout: 120000, // 2 minute timeout
                    },
                );

                // Step 3: Search for discriminatory instances
                progress.report({ increment: 70, message: 'Searching for discriminatory instances...' });

                const searchResponse = await axios.post(
                    `http://localhost:${SERVER_PORT}/search`,
                    {
                        protected_values: protectedValues,
                        num_iterations: 50,
                        num_neighbors: 30,
                    },
                    {
                        timeout: 120000,
                    },
                );

                progress.report({ increment: 100, message: 'Complete!' });

                // Show results
                showResults({
                    training: trainResponse.data,
                    analysis: analyzeResponse.data,
                    search: searchResponse.data,
                });
            } catch (error: any) {
                console.error('Analysis error:', error);
                vscode.window.showErrorMessage(`Error: ${error.message}`);

                if (error.response) {
                    vscode.window.showErrorMessage(`Server error: ${JSON.stringify(error.response.data)}`);
                }
            }
        },
    );
}

function showResults(results: any) {
    const panel = vscode.window.createWebviewPanel(
        'fairnessResults',
        'Fairness Analysis Results',
        vscode.ViewColumn.One,
        { enableScripts: true },
    );

    panel.webview.html = getWebviewHtml(results);
}

function getWebviewHtml(results: any): string {
    const qidMetrics = results.analysis.qid_metrics;
    const searchResults = results.search.search_results;

    return `
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            padding: 20px; 
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        h1 { color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 10px; }
        h2 { color: #81c784; margin-top: 30px; }
        .metric { 
            margin: 15px 0; 
            padding: 10px;
            background-color: #252526;
            border-left: 3px solid #4fc3f7;
            border-radius: 4px;
        }
        .metric-label { font-weight: bold; color: #4fc3f7; }
        .metric-value { font-size: 1.2em; margin-left: 10px; }
        .warning { background-color: #332b00; border-left-color: #ffa726; }
        .error { background-color: #330000; border-left-color: #ef5350; }
        .success { background-color: #003300; border-left-color: #66bb6a; }
        #qid-chart { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Fairness Analysis Results</h1>
    
    <h2>📊 Model Training</h2>
    <div class="metric success">
        <span class="metric-label">Accuracy:</span>
        <span class="metric-value">${results.training.accuracy.toFixed(2)}%</span>
    </div>
    <div class="metric">
        <span class="metric-label">Protected Features:</span>
        <span class="metric-value">${results.training.protected_features.join(', ')}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Model Parameters:</span>
        <span class="metric-value">${results.training.num_parameters.toLocaleString()}</span>
    </div>
    
    <h2>⚖️ QID Metrics (Information-Theoretic Fairness)</h2>
    <div class="metric ${qidMetrics.mean_qid > 1.0 ? 'warning' : ''}">
        <span class="metric-label">Mean QID:</span>
        <span class="metric-value">${qidMetrics.mean_qid.toFixed(4)} bits</span>
        <br><small>Average protected information used in decisions</small>
    </div>
    <div class="metric ${qidMetrics.max_qid > 2.0 ? 'error' : ''}">
        <span class="metric-label">Max QID:</span>
        <span class="metric-value">${qidMetrics.max_qid.toFixed(4)} bits</span>
        <br><small>Worst-case protected information leakage</small>
    </div>
    <div class="metric ${qidMetrics.pct_discriminatory > 10 ? 'warning' : 'success'}">
        <span class="metric-label">Discriminatory Instances:</span>
        <span class="metric-value">${qidMetrics.num_discriminatory} (${qidMetrics.pct_discriminatory.toFixed(
            1,
        )}%)</span>
        <br><small>Instances showing significant bias (QID > 0.1 bits)</small>
    </div>
    <div class="metric ${qidMetrics.mean_disparate_impact < 0.8 ? 'error' : 'success'}">
        <span class="metric-label">Mean Disparate Impact Ratio:</span>
        <span class="metric-value">${qidMetrics.mean_disparate_impact.toFixed(3)}</span>
        ${
            qidMetrics.mean_disparate_impact < 0.8
                ? '<br><strong style="color: #ef5350;">⚠️ Violates 80% Rule (Legal Threshold)</strong>'
                : '<br><span style="color: #66bb6a;">✅ Passes 80% Rule</span>'
        }
    </div>
    
    <h2>🔎 Discriminatory Instance Search</h2>
    <div class="metric">
        <span class="metric-label">Best QID Found:</span>
        <span class="metric-value">${searchResults.best_qid.toFixed(4)} bits</span>
        <br><small>Maximum discrimination discovered via gradient search</small>
    </div>
    <div class="metric">
        <span class="metric-label">Instances Generated:</span>
        <span class="metric-value">${searchResults.num_found}</span>
        <br><small>Discriminatory test cases found in local search</small>
    </div>
    
    <div id="qid-chart"></div>
    
    <script>
        const instances = ${JSON.stringify(searchResults.discriminatory_instances)};
        
        if (instances.length > 0) {
            const trace = {
                x: instances.map((_, i) => i + 1),
                y: instances.map(inst => inst.qid),
                type: 'scatter',
                mode: 'markers',
                marker: { 
                    size: 10, 
                    color: instances.map(inst => inst.qid),
                    colorscale: 'Reds',
                    showscale: true,
                    colorbar: {
                        title: 'QID (bits)'
                    }
                },
                text: instances.map(inst => \`QID: \${inst.qid.toFixed(4)} bits<br>Variance: \${inst.variance.toFixed(4)}\`),
                hoverinfo: 'text'
            };
            
            const layout = {
                title: 'QID Distribution of Discriminatory Instances',
                xaxis: { title: 'Instance Index' },
                yaxis: { title: 'QID (bits)' },
                plot_bgcolor: '#1e1e1e',
                paper_bgcolor: '#1e1e1e',
                font: { color: '#d4d4d4' }
            };
            
            Plotly.newPlot('qid-chart', [trace], layout);
        }
    </script>
    
    <h2>💡 Interpretation</h2>
    <div class="metric">
        <strong>What is QID?</strong><br>
        Quantitative Individual Discrimination measures how many bits of protected information 
        (e.g., gender, race) the model uses to make decisions. 0 bits = perfectly fair, higher = more bias.
    </div>
    <div class="metric">
        <strong>80% Rule:</strong><br>
        The disparate impact ratio should be ≥ 0.8 (80%) to comply with legal standards in hiring/lending.
        Values below 0.8 may indicate legally actionable discrimination.
    </div>
</body>
</html>
    `;
}

export function deactivate() {
    if (serverProcess) {
        serverProcess.kill();
        console.log('Backend server stopped');
    }
}
