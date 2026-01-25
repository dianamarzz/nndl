// app.js
/**
 * Titanic EDA Dashboard
 * Client-Side Exploratory Data Analysis Application
 * Uses: PapaParse (CSV parsing), Chart.js (visualizations), Bootstrap 5 (UI)
 */

// Global variables to store data and chart instances
let titanicData = [];
let charts = {};
let columnStats = {};

// DOM Content Loaded Event Listener
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    // Initialize event listeners
    document.getElementById('runEDA').addEventListener('click', loadAndAnalyzeData);
    document.getElementById('csvFile').addEventListener('change', handleFileSelect);
});

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('runButtonText').textContent = `Analyze ${file.name}`;
    }
}

// Main function to load and analyze data
function loadAndAnalyzeData() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first. Use Titanic-Dataset.csv from Kaggle.');
        return;
    }
    
    // Show loading state
    const runButton = document.getElementById('runEDA');
    const buttonText = document.getElementById('runButtonText');
    const spinner = document.getElementById('loadingSpinner');
    
    buttonText.textContent = 'Analyzing...';
    spinner.style.display = 'inline-block';
    runButton.disabled = true;
    
    // Parse CSV file using PapaParse
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
                alert('Error parsing CSV: ' + results.errors[0].message);
                resetButtonState();
                return;
            }
            
            titanicData = results.data;
            console.log(`Loaded ${titanicData.length} records`);
            
            // Perform complete EDA pipeline
            try {
                performEDA();
            } catch (error) {
                console.error('Error in EDA pipeline:', error);
                alert('Error during analysis: ' + error.message);
            }
            
            // Reset button state
            resetButtonState();
        },
        error: function(error) {
            alert('Error loading file: ' + error.message);
            resetButtonState();
        }
    });
    
    function resetButtonState() {
        buttonText.textContent = '🔄 Re-run EDA Analysis';
        spinner.style.display = 'none';
        runButton.disabled = false;
    }
}

// Main EDA pipeline
function performEDA() {
    console.log('Starting EDA pipeline...');
    
    // Show all sections
    document.querySelectorAll('.card').forEach(card => {
        card.style.display = 'block';
    });
    
    // 1. Data Overview
    displayDataOverview();
    
    // 2. Data Types & Missing Values
    analyzeDataTypesAndMissing();
    
    // 3. Distribution Analysis
    createDistributionCharts();
    
    // 4. Outlier Detection (with SIMPLIFIED visualization)
    detectOutliers();
    
    // 5. Correlation Analysis
    calculateCorrelations();
    
    // 6. Survival Analysis
    analyzeSurvivalFactors();
    
    // 7. Key Insights & Conclusion
    generateInsightsAndConclusion();
    
    console.log('EDA pipeline completed');
}

// 1. Display Data Overview
function displayDataOverview() {
    try {
        const rowCount = titanicData.length;
        const colCount = Object.keys(titanicData[0] || {}).length;
        
        console.log(`Data overview: ${rowCount} rows, ${colCount} columns`);
        
        // Update overview stats
        document.getElementById('rowCount').textContent = rowCount.toLocaleString();
        document.getElementById('colCount').textContent = colCount;
        
        // Calculate survival rate
        const survivedCount = titanicData.filter(row => row.Survived === 1).length;
        const survivalRate = ((survivedCount / rowCount) * 100).toFixed(1);
        document.getElementById('survivalRate').textContent = `${survivalRate}%`;
        
        // Calculate complete cases
        const completeCases = titanicData.filter(row => {
            return Object.values(row).every(value => value !== null && value !== undefined && value !== '');
        }).length;
        const completeCasesPct = ((completeCases / rowCount) * 100).toFixed(1);
        document.getElementById('completeCases').textContent = `${completeCasesPct}%`;
        
        // Display data preview
        displayDataPreview();
        
        // Show overview section
        document.getElementById('dataOverview').style.display = 'block';
    } catch (error) {
        console.error('Error in displayDataOverview:', error);
    }
}

// Display data preview table
function displayDataPreview() {
    try {
        const previewHeader = document.getElementById('previewHeader');
        const previewBody = document.getElementById('previewBody');
        
        // Clear existing content
        previewHeader.innerHTML = '';
        previewBody.innerHTML = '';
        
        // Get column names from first row
        const columns = Object.keys(titanicData[0] || {});
        
        // Create header row
        const headerRow = document.createElement('tr');
        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            th.scope = 'col';
            headerRow.appendChild(th);
        });
        previewHeader.appendChild(headerRow);
        
        // Create data rows (first 10)
        const previewRows = titanicData.slice(0, 10);
        previewRows.forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                let value = row[col];
                
                // Handle null/undefined values
                if (value === null || value === undefined || value === '') {
                    value = '<span class="text-muted">-</span>';
                    td.innerHTML = value;
                } else if (typeof value === 'number') {
                    td.textContent = Number.isInteger(value) ? value : value.toFixed(2);
                } else {
                    td.textContent = value;
                }
                
                tr.appendChild(td);
            });
            previewBody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error in displayDataPreview:', error);
    }
}

// 2. Analyze Data Types and Missing Values
function analyzeDataTypesAndMissing() {
    try {
        const columnInfoBody = document.getElementById('columnInfo');
        columnInfoBody.innerHTML = '';
        
        const columns = Object.keys(titanicData[0] || {});
        
        columns.forEach(col => {
            const nonMissingCount = titanicData.filter(row => {
                const value = row[col];
                return value !== null && value !== undefined && value !== '';
            }).length;
            
            const missingCount = titanicData.length - nonMissingCount;
            const missingPct = ((missingCount / titanicData.length) * 100).toFixed(1);
            
            // Determine data type
            const sampleRow = titanicData.find(row => row[col] !== null && row[col] !== undefined && row[col] !== '');
            let dataType = 'unknown';
            if (sampleRow) {
                dataType = typeof sampleRow[col];
                if (dataType === 'number' && ['Pclass', 'SibSp', 'Parch', 'Survived'].includes(col)) {
                    dataType = 'integer (categorical)';
                } else if (dataType === 'number') {
                    dataType = 'float';
                } else if (dataType === 'string') {
                    dataType = 'text';
                }
            }
            
            // Store column stats for later use
            columnStats[col] = {
                missingCount,
                missingPct,
                dataType,
                nonMissingCount
            };
            
            // Add to table
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${col}</strong></td>
                <td><span class="badge bg-secondary">${dataType}</span></td>
                <td>${missingCount}</td>
                <td>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar bg-warning" role="progressbar" 
                             style="width: ${missingPct}%" 
                             aria-valuenow="${missingPct}" 
                             aria-valuemin="0" 
                             aria-valuemax="100"></div>
                    </div>
                    <span class="small">${missingPct}%</span>
                </td>
            `;
            columnInfoBody.appendChild(tr);
        });
        
        // Update missing summary
        const missingSummary = document.getElementById('missingSummary');
        missingSummary.textContent = `Age (${columnStats['Age']?.missingPct || 0}%) and Cabin (${columnStats['Cabin']?.missingPct || 0}%) have the most missing values.`;
        
        // Create missing values chart
        createMissingValuesChart();
    } catch (error) {
        console.error('Error in analyzeDataTypesAndMissing:', error);
    }
}

// Create missing values bar chart
function createMissingValuesChart() {
    try {
        const ctx = document.getElementById('missingValuesChart');
        if (!ctx) {
            console.error('Missing chart canvas');
            return;
        }
        
        // Destroy previous chart if exists
        if (charts.missingValues) {
            charts.missingValues.destroy();
        }
        
        // Sort columns by missing percentage
        const columns = Object.keys(columnStats).sort((a, b) => 
            parseFloat(columnStats[b].missingPct) - parseFloat(columnStats[a].missingPct)
        );
        
        const labels = columns;
        const missingData = columns.map(col => parseFloat(columnStats[col].missingPct) || 0);
        
        charts.missingValues = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Missing Values %',
                    data: missingData,
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgb(52, 152, 219)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Percentage Missing' }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in createMissingValuesChart:', error);
    }
}

// 3. Create Distribution Charts
function createDistributionCharts() {
    createNumericalDistributionCharts();
    createCategoricalDistributionCharts();
}

// Create numerical distribution charts
function createNumericalDistributionCharts() {
    try {
        // Age Distribution
        const ageData = titanicData
            .filter(row => row.Age !== null && !isNaN(row.Age))
            .map(row => row.Age);
        
        if (ageData.length > 0) {
            createHistogram('ageDistChart', ageData, 'Age', 'Passenger Count', 'Age Distribution');
        }
        
        // Fare Distribution
        const fareData = titanicData
            .filter(row => row.Fare !== null && !isNaN(row.Fare))
            .map(row => row.Fare);
        
        if (fareData.length > 0) {
            createHistogram('fareDistChart', fareData, 'Fare', 'Passenger Count', 'Fare Distribution', 20);
        }
        
        // SibSp Distribution
        const sibspData = titanicData
            .filter(row => row.SibSp !== null && !isNaN(row.SibSp))
            .map(row => row.SibSp);
        
        if (sibspData.length > 0) {
            createHistogram('sibspDistChart', sibspData, 'Siblings/Spouses', 'Passenger Count', 'SibSp Distribution', 7);
        }
        
        // Parch Distribution
        const parchData = titanicData
            .filter(row => row.Parch !== null && !isNaN(row.Parch))
            .map(row => row.Parch);
        
        if (parchData.length > 0) {
            createHistogram('parchDistChart', parchData, 'Parents/Children', 'Passenger Count', 'Parch Distribution', 7);
        }
    } catch (error) {
        console.error('Error in createNumericalDistributionCharts:', error);
    }
}

// Create categorical distribution charts
function createCategoricalDistributionCharts() {
    try {
        // Gender Distribution
        const sexCounts = countCategories('Sex');
        if (Object.keys(sexCounts).length > 0) {
            createBarChart('sexDistChart', 
                Object.keys(sexCounts), 
                Object.values(sexCounts), 
                'Gender', 
                'Count', 
                'Gender Distribution',
                ['rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
            );
        }
        
        // Passenger Class Distribution
        const pclassCounts = countCategories('Pclass');
        if (Object.keys(pclassCounts).length > 0) {
            createBarChart('pclassDistChart', 
                Object.keys(pclassCounts).map(key => `Class ${key}`), 
                Object.values(pclassCounts), 
                'Passenger Class', 
                'Count', 
                'Passenger Class Distribution',
                ['rgba(46, 204, 113, 0.7)', 'rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
            );
        }
        
        // Embarked Port Distribution
        const embarkedCounts = countCategories('Embarked');
        if (Object.keys(embarkedCounts).length > 0) {
            createBarChart('embarkedDistChart', 
                Object.keys(embarkedCounts).map(key => {
                    if (key === 'C') return 'Cherbourg';
                    if (key === 'Q') return 'Queenstown';
                    if (key === 'S') return 'Southampton';
                    return key;
                }), 
                Object.values(embarkedCounts), 
                'Embarkation Port', 
                'Count', 
                'Embarkation Port Distribution',
                ['rgba(52, 152, 219, 0.7)', 'rgba(46, 204, 113, 0.7)', 'rgba(241, 196, 15, 0.7)']
            );
        }
        
        // Survival Distribution
        const survivedCounts = countCategories('Survived');
        if (Object.keys(survivedCounts).length > 0) {
            createBarChart('survivedDistChart', 
                ['Perished', 'Survived'], 
                [survivedCounts[0] || 0, survivedCounts[1] || 0], 
                'Outcome', 
                'Count', 
                'Survival Distribution',
                ['rgba(231, 76, 60, 0.7)', 'rgba(46, 204, 113, 0.7)']
            );
        }
    } catch (error) {
        console.error('Error in createCategoricalDistributionCharts:', error);
    }
}

// Helper function to count categories
function countCategories(columnName) {
    const counts = {};
    titanicData.forEach(row => {
        const value = row[columnName];
        if (value !== null && value !== undefined && value !== '') {
            counts[value] = (counts[value] || 0) + 1;
        }
    });
    return counts;
}

// Helper function to create histogram
function createHistogram(canvasId, data, xLabel, yLabel, title, bins = 15) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas ${canvasId} not found`);
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy previous chart if exists
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        
        // Calculate bin ranges
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binWidth = (max - min) / bins;
        
        // Create bins
        const binCounts = new Array(bins).fill(0);
        data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
            binCounts[binIndex]++;
        });
        
        // Create bin labels
        const binLabels = [];
        for (let i = 0; i < bins; i++) {
            const binStart = min + (i * binWidth);
            const binEnd = min + ((i + 1) * binWidth);
            binLabels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
        }
        
        charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: yLabel,
                    data: binCounts,
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgb(52, 152, 219)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: xLabel },
                        ticks: { maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: yLabel },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error(`Error in createHistogram for ${canvasId}:`, error);
    }
}

// Helper function to create bar chart
function createBarChart(canvasId, labels, data, xLabel, yLabel, title, colors = null) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas ${canvasId} not found`);
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy previous chart if exists
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        
        // Generate colors if not provided
        const backgroundColors = colors || labels.map((_, i) => 
            `hsl(${(i * 360) / labels.length}, 70%, 65%)`
        );
        
        charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: yLabel,
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: xLabel }
                    },
                    y: {
                        title: { display: true, text: yLabel },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error(`Error in createBarChart for ${canvasId}:`, error);
    }
}

// 4. Detect Outliers - SIMPLIFIED VERSION
function detectOutliers() {
    try {
        // Detect outliers for Age
        const ageData = titanicData
            .filter(row => row.Age !== null && !isNaN(row.Age))
            .map(row => row.Age);
        
        if (ageData.length > 0) {
            const ageOutliers = detectOutliersIQR(ageData);
            const ageOutliersCount = ageOutliers.length;
            const ageOutliersPct = ((ageOutliersCount / ageData.length) * 100).toFixed(1);
            
            document.getElementById('ageOutliers').textContent = ageOutliersCount;
            document.getElementById('ageOutliersPct').textContent = `${ageOutliersPct}%`;
        }
        
        // Detect outliers for Fare
        const fareData = titanicData
            .filter(row => row.Fare !== null && !isNaN(row.Fare))
            .map(row => row.Fare);
        
        if (fareData.length > 0) {
            const fareOutliers = detectOutliersIQR(fareData);
            const fareOutliersCount = fareOutliers.length;
            const fareOutliersPct = ((fareOutliersCount / fareData.length) * 100).toFixed(1);
            
            document.getElementById('fareOutliers').textContent = fareOutliersCount;
            document.getElementById('fareOutliersPct').textContent = `${fareOutliersPct}%`;
            
            // Create SIMPLE bar chart for Fare distribution instead of complex visualization
            createSimpleOutlierChart(fareData, fareOutliers);
        }
    } catch (error) {
        console.error('Error in detectOutliers:', error);
    }
}

// Detect outliers using IQR method
function detectOutliersIQR(data) {
    try {
        const sorted = [...data].sort((a, b) => a - b);
        const q1Index = Math.floor(sorted.length * 0.25);
        const q3Index = Math.floor(sorted.length * 0.75);
        const q1 = sorted[q1Index];
        const q3 = sorted[q3Index];
        const iqr = q3 - q1;
        const lowerBound = q1 - (1.5 * iqr);
        const upperBound = q3 + (1.5 * iqr);
        
        return data.filter(value => value < lowerBound || value > upperBound);
    } catch (error) {
        console.error('Error in detectOutliersIQR:', error);
        return [];
    }
}

// Create SIMPLE bar chart for outliers
function createSimpleOutlierChart(data, outliers) {
    try {
        const canvas = document.getElementById('outlierChart');
        if (!canvas) {
            console.error('outlierChart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy previous chart if exists
        if (charts.outlierChart) {
            charts.outlierChart.destroy();
        }
        
        // Create bins for histogram
        const maxFare = Math.max(...data);
        const bins = 20;
        const binSize = maxFare / bins;
        
        const binCounts = new Array(bins).fill(0);
        const outlierBinCounts = new Array(bins).fill(0);
        
        data.forEach(value => {
            const binIndex = Math.min(Math.floor(value / binSize), bins - 1);
            if (outliers.includes(value)) {
                outlierBinCounts[binIndex]++;
            } else {
                binCounts[binIndex]++;
            }
        });
        
        // Create bin labels
        const binLabels = [];
        for (let i = 0; i < bins; i++) {
            binLabels.push(`${(i * binSize).toFixed(0)}-${((i + 1) * binSize).toFixed(0)}`);
        }
        
        charts.outlierChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [
                    {
                        label: 'Normal Fares',
                        data: binCounts,
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgb(52, 152, 219)',
                        borderWidth: 1
                    },
                    {
                        label: 'Outlier Fares',
                        data: outlierBinCounts,
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgb(231, 76, 60)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Fare Distribution with Outliers Highlighted'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Fare Range'
                        },
                        ticks: {
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Number of Passengers'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in createSimpleOutlierChart:', error);
    }
}

// 5. Calculate Correlations - SIMPLIFIED VERSION
function calculateCorrelations() {
    try {
        const numericalColumns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived'];
        
        // Filter data to only include rows with all numerical values
        const completeData = titanicData.filter(row => 
            numericalColumns.every(col => 
                row[col] !== null && !isNaN(row[col])
            )
        );
        
        console.log(`Complete data for correlation: ${completeData.length} rows`);
        
        if (completeData.length < 10) {
            console.warn('Not enough data for correlation analysis');
            document.getElementById('correlationInsights').innerHTML = 
                '<li>Not enough complete data for correlation analysis</li>';
            return;
        }
        
        // Calculate correlation matrix
        const correlationMatrix = [];
        for (let i = 0; i < numericalColumns.length; i++) {
            correlationMatrix[i] = [];
            for (let j = 0; j < numericalColumns.length; j++) {
                correlationMatrix[i][j] = calculateCorrelation(
                    completeData.map(row => row[numericalColumns[i]]),
                    completeData.map(row => row[numericalColumns[j]])
                );
            }
        }
        
        // Create correlation chart
        createCorrelationChart(numericalColumns, correlationMatrix);
        
        // Generate correlation insights
        generateCorrelationInsights(numericalColumns, correlationMatrix);
    } catch (error) {
        console.error('Error in calculateCorrelations:', error);
        document.getElementById('correlationInsights').innerHTML = 
            '<li>Error calculating correlations</li>';
    }
}

// Calculate Pearson correlation coefficient
function calculateCorrelation(x, y) {
    try {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
        const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
        const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
        
        const numerator = (n * sumXY) - (sumX * sumY);
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    } catch (error) {
        console.error('Error in calculateCorrelation:', error);
        return 0;
    }
}

// Create correlation chart
function createCorrelationChart(labels, matrix) {
    try {
        const canvas = document.getElementById('correlationChart');
        if (!canvas) {
            console.error('correlationChart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Destroy previous chart if exists
        if (charts.correlationChart) {
            charts.correlationChart.destroy();
        }
        
        // Extract unique correlations (excluding self-correlations and duplicates)
        const correlations = [];
        for (let i = 0; i < labels.length; i++) {
            for (let j = i + 1; j < labels.length; j++) {
                correlations.push({
                    pair: `${labels[i]} - ${labels[j]}`,
                    value: matrix[i][j]
                });
            }
        }
        
        // Sort by absolute value
        correlations.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
        
        // Take top 8 correlations
        const topCorrelations = correlations.slice(0, 8);
        
        charts.correlationChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: topCorrelations.map(c => c.pair),
                datasets: [{
                    label: 'Correlation Coefficient',
                    data: topCorrelations.map(c => c.value),
                    backgroundColor: topCorrelations.map(c => 
                        c.value > 0 ? 'rgba(46, 204, 113, 0.7)' : 'rgba(231, 76, 60, 0.7)'
                    ),
                    borderColor: topCorrelations.map(c => 
                        c.value > 0 ? 'rgb(46, 204, 113)' : 'rgb(231, 76, 60)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    title: {
                        display: true,
                        text: 'Top Feature Correlations'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Correlation: ${context.raw.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: -1,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Correlation Coefficient'
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in createCorrelationChart:', error);
    }
}

// Generate correlation insights
function generateCorrelationInsights(labels, matrix) {
    try {
        const insightsList = document.getElementById('correlationInsights');
        insightsList.innerHTML = '';
        
        // Find strongest correlations
        const insights = [];
        
        for (let i = 0; i < labels.length; i++) {
            for (let j = i + 1; j < labels.length; j++) {
                const corr = matrix[i][j];
                if (Math.abs(corr) > 0.3) {
                    const strength = Math.abs(corr) > 0.7 ? 'Strong' : 
                                    Math.abs(corr) > 0.5 ? 'Moderate' : 'Weak';
                    const direction = corr > 0 ? 'positive' : 'negative';
                    
                    insights.push({
                        feature1: labels[i],
                        feature2: labels[j],
                        corr: corr,
                        text: `${strength} ${direction} correlation (${corr.toFixed(3)})`
                    });
                }
            }
        }
        
        // Sort by absolute correlation strength
        insights.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
        
        // Add insights to list
        if (insights.length > 0) {
            insights.slice(0, 4).forEach(insight => {
                const li = document.createElement('li');
                li.className = 'mb-1';
                li.innerHTML = `<strong>${insight.feature1} & ${insight.feature2}:</strong> ${insight.text}`;
                insightsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No strong correlations found (all |r| < 0.3)';
            insightsList.appendChild(li);
        }
    } catch (error) {
        console.error('Error in generateCorrelationInsights:', error);
    }
}

// 6. Analyze Survival Factors
function analyzeSurvivalFactors() {
    try {
        // Survival by Gender
        analyzeSurvivalByGender();
        
        // Survival by Passenger Class
        analyzeSurvivalByPclass();
        
        // Survival by Age Group
        analyzeSurvivalByAge();
        
        // Calculate and display survival metrics
        calculateSurvivalMetrics();
    } catch (error) {
        console.error('Error in analyzeSurvivalFactors:', error);
    }
}

// Analyze survival by gender
function analyzeSurvivalByGender() {
    try {
        const genderGroups = {
            'male': { total: 0, survived: 0 },
            'female': { total: 0, survived: 0 }
        };
        
        titanicData.forEach(row => {
            if (row.Sex && (row.Sex === 'male' || row.Sex === 'female')) {
                genderGroups[row.Sex].total++;
                if (row.Survived === 1) {
                    genderGroups[row.Sex].survived++;
                }
            }
        });
        
        const maleSurvivalRate = genderGroups.male.total > 0 ? 
            (genderGroups.male.survived / genderGroups.male.total * 100).toFixed(1) : '0.0';
        const femaleSurvivalRate = genderGroups.female.total > 0 ? 
            (genderGroups.female.survived / genderGroups.female.total * 100).toFixed(1) : '0.0';
        
        // Create grouped bar chart
        const canvas = document.getElementById('survivalSexChart');
        if (!canvas) {
            console.error('survivalSexChart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        if (charts.survivalSexChart) {
            charts.survivalSexChart.destroy();
        }
        
        charts.survivalSexChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Male', 'Female'],
                datasets: [
                    {
                        label: 'Survived',
                        data: [genderGroups.male.survived, genderGroups.female.survived],
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgb(46, 204, 113)',
                        borderWidth: 1
                    },
                    {
                        label: 'Perished',
                        data: [
                            genderGroups.male.total - genderGroups.male.survived,
                            genderGroups.female.total - genderGroups.female.survived
                        ],
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgb(231, 76, 60)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Survival by Gender (Female: ${femaleSurvivalRate}%, Male: ${maleSurvivalRate}%)`
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Gender' }
                    },
                    y: {
                        title: { display: true, text: 'Passenger Count' },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in analyzeSurvivalByGender:', error);
    }
}

// Analyze survival by passenger class
function analyzeSurvivalByPclass() {
    try {
        const classGroups = {
            1: { total: 0, survived: 0 },
            2: { total: 0, survived: 0 },
            3: { total: 0, survived: 0 }
        };
        
        titanicData.forEach(row => {
            if (row.Pclass && (row.Pclass === 1 || row.Pclass === 2 || row.Pclass === 3)) {
                classGroups[row.Pclass].total++;
                if (row.Survived === 1) {
                    classGroups[row.Pclass].survived++;
                }
            }
        });
        
        const survivalRates = [
            classGroups[1].total > 0 ? (classGroups[1].survived / classGroups[1].total * 100).toFixed(1) : '0.0',
            classGroups[2].total > 0 ? (classGroups[2].survived / classGroups[2].total * 100).toFixed(1) : '0.0',
            classGroups[3].total > 0 ? (classGroups[3].survived / classGroups[3].total * 100).toFixed(1) : '0.0'
        ];
        
        // Create grouped bar chart
        const canvas = document.getElementById('survivalClassChart');
        if (!canvas) {
            console.error('survivalClassChart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        if (charts.survivalClassChart) {
            charts.survivalClassChart.destroy();
        }
        
        charts.survivalClassChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Class 1', 'Class 2', 'Class 3'],
                datasets: [
                    {
                        label: 'Survived',
                        data: [classGroups[1].survived, classGroups[2].survived, classGroups[3].survived],
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgb(46, 204, 113)',
                        borderWidth: 1
                    },
                    {
                        label: 'Perished',
                        data: [
                            classGroups[1].total - classGroups[1].survived,
                            classGroups[2].total - classGroups[2].survived,
                            classGroups[3].total - classGroups[3].survived
                        ],
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgb(231, 76, 60)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Survival by Class (1st: ${survivalRates[0]}%, 2nd: ${survivalRates[1]}%, 3rd: ${survivalRates[2]}%)`
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Passenger Class' }
                    },
                    y: {
                        title: { display: true, text: 'Passenger Count' },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in analyzeSurvivalByPclass:', error);
    }
}

// Analyze survival by age group
function analyzeSurvivalByAge() {
    try {
        // Create age groups
        const ageGroups = {
            'Child (0-12)': { total: 0, survived: 0 },
            'Teen (13-19)': { total: 0, survived: 0 },
            'Adult (20-39)': { total: 0, survived: 0 },
            'Middle (40-59)': { total: 0, survived: 0 },
            'Senior (60+)': { total: 0, survived: 0 }
        };
        
        // Count passengers in each age group
        titanicData.forEach(row => {
            if (row.Age !== null && !isNaN(row.Age)) {
                const age = row.Age;
                let group;
                
                if (age <= 12) group = 'Child (0-12)';
                else if (age <= 19) group = 'Teen (13-19)';
                else if (age <= 39) group = 'Adult (20-39)';
                else if (age <= 59) group = 'Middle (40-59)';
                else group = 'Senior (60+)';
                
                ageGroups[group].total++;
                if (row.Survived === 1) {
                    ageGroups[group].survived++;
                }
            }
        });
        
        // Create grouped bar chart
        const canvas = document.getElementById('survivalAgeChart');
        if (!canvas) {
            console.error('survivalAgeChart canvas not found');
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        if (charts.survivalAgeChart) {
            charts.survivalAgeChart.destroy();
        }
        
        const labels = Object.keys(ageGroups);
        
        charts.survivalAgeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Survived',
                        data: labels.map(group => ageGroups[group].survived),
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgb(46, 204, 113)',
                        borderWidth: 1
                    },
                    {
                        label: 'Perished',
                        data: labels.map(group => ageGroups[group].total - ageGroups[group].survived),
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgb(231, 76, 60)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Survival by Age Group'
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Age Group' }
                    },
                    y: {
                        title: { display: true, text: 'Passenger Count' },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in analyzeSurvivalByAge:', error);
    }
}

// Calculate and display survival metrics
function calculateSurvivalMetrics() {
    try {
        const metricsContainer = document.getElementById('survivalMetrics');
        metricsContainer.innerHTML = '';
        
        // Overall survival
        const totalPassengers = titanicData.length;
        const totalSurvived = titanicData.filter(row => row.Survived === 1).length;
        const overallSurvivalRate = totalPassengers > 0 ? 
            (totalSurvived / totalPassengers * 100).toFixed(1) : '0.0';
        
        // Gender survival rates
        const malePassengers = titanicData.filter(row => row.Sex === 'male');
        const femalePassengers = titanicData.filter(row => row.Sex === 'female');
        
        const maleSurvived = malePassengers.filter(row => row.Survived === 1).length;
        const femaleSurvived = femalePassengers.filter(row => row.Survived === 1).length;
        
        const maleSurvivalRate = malePassengers.length > 0 ? 
            (maleSurvived / malePassengers.length * 100).toFixed(1) : '0.0';
        const femaleSurvivalRate = femalePassengers.length > 0 ? 
            (femaleSurvived / femalePassengers.length * 100).toFixed(1) : '0.0';
        const genderDifferential = (femaleSurvivalRate - maleSurvivalRate).toFixed(1);
        
        // Class survival rates
        const class1Passengers = titanicData.filter(row => row.Pclass === 1);
        const class3Passengers = titanicData.filter(row => row.Pclass === 3);
        
        const class1Survived = class1Passengers.filter(row => row.Survived === 1).length;
        const class3Survived = class3Passengers.filter(row => row.Survived === 1).length;
        
        const class1SurvivalRate = class1Passengers.length > 0 ? 
            (class1Survived / class1Passengers.length * 100).toFixed(1) : '0.0';
        const class3SurvivalRate = class3Passengers.length > 0 ? 
            (class3Survived / class3Passengers.length * 100).toFixed(1) : '0.0';
        const classDifferential = (class1SurvivalRate - class3SurvivalRate).toFixed(1);
        
        // Create metric cards
        const metrics = [
            {
                title: 'Overall Survival',
                value: `${overallSurvivalRate}%`,
                description: `${totalSurvived} of ${totalPassengers} passengers`,
                color: 'info'
            },
            {
                title: 'Gender Gap',
                value: `${genderDifferential}%`,
                description: `Female ${femaleSurvivalRate}% vs Male ${maleSurvivalRate}%`,
                color: 'primary'
            },
            {
                title: 'Class Gap',
                value: `${classDifferential}%`,
                description: `1st Class ${class1SurvivalRate}% vs 3rd Class ${class3SurvivalRate}%`,
                color: 'success'
            }
        ];
        
        metrics.forEach(metric => {
            const col = document.createElement('div');
            col.className = 'col-md-4';
            col.innerHTML = `
                <div class="card text-center h-100">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">${metric.title}</h6>
                        <h3 class="card-title text-${metric.color}">${metric.value}</h3>
                        <p class="card-text small">${metric.description}</p>
                    </div>
                </div>
            `;
            metricsContainer.appendChild(col);
        });
    } catch (error) {
        console.error('Error in calculateSurvivalMetrics:', error);
    }
}

// 7. Generate Insights and Conclusion
function generateInsightsAndConclusion() {
    try {
        // Determine the most important survival factor
        const topFactor = determineTopSurvivalFactor();
        document.getElementById('topFactor').textContent = topFactor.factor;
        document.getElementById('factorExplanation').innerHTML = topFactor.explanation;
        
        // Generate key insights
        const insightsList = document.getElementById('keyInsights');
        insightsList.innerHTML = '';
        
        // Overall statistics
        const totalPassengers = titanicData.length;
        const totalSurvived = titanicData.filter(row => row.Survived === 1).length;
        const survivalRate = totalPassengers > 0 ? 
            (totalSurvived / totalPassengers * 100).toFixed(1) : '0.0';
        
        // Gender statistics
        const malePassengers = titanicData.filter(row => row.Sex === 'male');
        const femalePassengers = titanicData.filter(row => row.Sex === 'female');
        const maleSurvivalRate = malePassengers.length > 0 ? 
            (malePassengers.filter(row => row.Survived === 1).length / malePassengers.length * 100).toFixed(1) : '0.0';
        const femaleSurvivalRate = femalePassengers.length > 0 ? 
            (femalePassengers.filter(row => row.Survived === 1).length / femalePassengers.length * 100).toFixed(1) : '0.0';
        
        // Class statistics
        const class1Passengers = titanicData.filter(row => row.Pclass === 1);
        const class3Passengers = titanicData.filter(row => row.Pclass === 3);
        const class1SurvivalRate = class1Passengers.length > 0 ? 
            (class1Passengers.filter(row => row.Survived === 1).length / class1Passengers.length * 100).toFixed(1) : '0.0';
        const class3SurvivalRate = class3Passengers.length > 0 ? 
            (class3Passengers.filter(row => row.Survived === 1).length / class3Passengers.length * 100).toFixed(1) : '0.0';
        
        // Missing data
        const ageMissingPct = columnStats['Age']?.missingPct || '0.0';
        const cabinMissingPct = columnStats['Cabin']?.missingPct || '0.0';
        
        const insights = [
            `Overall survival rate was ${survivalRate}% (${totalSurvived} of ${totalPassengers} passengers).`,
            `Female passengers had a ${femaleSurvivalRate}% survival rate vs ${maleSurvivalRate}% for males.`,
            `First class passengers had a ${class1SurvivalRate}% survival rate vs ${class3SurvivalRate}% for third class.`,
            `Age data is ${ageMissingPct}% missing, requiring careful handling in analysis.`,
            `Fare outliers represent ${document.getElementById('fareOutliersPct').textContent} of passengers.`
        ];
        
        insights.forEach((insight, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.innerHTML = `<strong>${index + 1}.</strong> ${insight}`;
            insightsList.appendChild(li);
        });
        
        // Update conclusion text
        const conclusionText = document.getElementById('conclusionText');
        conclusionText.innerHTML = `
            The Exploratory Data Analysis reveals that <strong>${topFactor.factor}</strong> was the most significant factor in determining survival on the Titanic. 
            ${topFactor.explanation} This dashboard demonstrates a complete client-side EDA pipeline using modern web technologies.
        `;
        
    } catch (error) {
        console.error('Error in generateInsightsAndConclusion:', error);
    }
}

// Determine the most important survival factor
function determineTopSurvivalFactor() {
    try {
        // Calculate gender survival differential
        const malePassengers = titanicData.filter(row => row.Sex === 'male');
        const femalePassengers = titanicData.filter(row => row.Sex === 'female');
        
        const maleSurvivalRate = malePassengers.length > 0 ? 
            malePassengers.filter(row => row.Survived === 1).length / malePassengers.length : 0;
        const femaleSurvivalRate = femalePassengers.length > 0 ? 
            femalePassengers.filter(row => row.Survived === 1).length / femalePassengers.length : 0;
        
        const genderDifferential = Math.abs(femaleSurvivalRate - maleSurvivalRate);
        
        // Calculate class survival differential
        const class1Passengers = titanicData.filter(row => row.Pclass === 1);
        const class3Passengers = titanicData.filter(row => row.Pclass === 3);
        
        const class1SurvivalRate = class1Passengers.length > 0 ? 
            class1Passengers.filter(row => row.Survived === 1).length / class1Passengers.length : 0;
        const class3SurvivalRate = class3Passengers.length > 0 ? 
            class3Passengers.filter(row => row.Survived === 1).length / class3Passengers.length : 0;
        
        const classDifferential = Math.abs(class1SurvivalRate - class3SurvivalRate);
        
        // Determine which factor has the largest differential
        if (genderDifferential > classDifferential) {
            return {
                factor: "Gender (Sex)",
                explanation: `Female passengers had a ${(genderDifferential * 100).toFixed(1)} percentage point higher survival rate than male passengers. The "women and children first" protocol was clearly followed.`
            };
        } else {
            return {
                factor: "Passenger Class (Pclass)",
                explanation: `First class passengers had a ${(classDifferential * 100).toFixed(1)} percentage point higher survival rate than third class passengers. Socio-economic status played a crucial role in survival chances.`
            };
        }
    } catch (error) {
        console.error('Error in determineTopSurvivalFactor:', error);
        return {
            factor: "Unknown",
            explanation: "Unable to determine the most important factor due to data issues."
        };
    }
}
