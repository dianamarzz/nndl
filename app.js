/**
 * Titanic EDA Dashboard - Main Application Script
 * This file contains all data processing, visualization, and interaction logic.
 * 
 * Dataset Configuration:
 * - Update the DEFAULT_CSV_URL to point to your dataset
 * - Update column name constants if using a different dataset
 */

// ==============================================
// CONFIGURATION & CONSTANTS
// ==============================================

// Default dataset URL (for demonstration)
const DEFAULT_CSV_URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv';

// Titanic dataset column names (Update these if using a different dataset)
const COLUMNS = {
    PASSENGER_ID: 'PassengerId',
    SURVIVED: 'Survived',
    PCLASS: 'Pclass',
    NAME: 'Name',
    SEX: 'Sex',
    AGE: 'Age',
    SIBSP: 'SibSp',
    PARCH: 'Parch',
    TICKET: 'Ticket',
    FARE: 'Fare',
    CABIN: 'Cabin',
    EMBARKED: 'Embarked'
};

// Chart color scheme
const CHART_COLORS = {
    primary: 'rgba(52, 152, 219, 0.7)',
    secondary: 'rgba(155, 89, 182, 0.7)',
    success: 'rgba(46, 204, 113, 0.7)',
    danger: 'rgba(231, 76, 60, 0.7)',
    warning: 'rgba(241, 196, 15, 0.7)',
    info: 'rgba(52, 152, 219, 0.7)',
    light: 'rgba(236, 240, 241, 0.7)',
    dark: 'rgba(52, 73, 94, 0.7)',
    female: 'rgba(255, 99, 132, 0.7)',
    male: 'rgba(54, 162, 235, 0.7)',
    survived: 'rgba(46, 204, 113, 0.7)',
    deceased: 'rgba(231, 76, 60, 0.7)'
};

// Global variables
let titanicData = [];
let charts = {}; // Store chart instances for potential updates
let analysisResults = {};

// ==============================================
// DOM ELEMENT REFERENCES
// ==============================================

const domElements = {
    fileInput: document.getElementById('csvFile'),
    fileDropArea: document.getElementById('fileDropArea'),
    runEDAButton: document.getElementById('runEDA'),
    loadingSpinner: document.getElementById('loadingSpinner'),
    dataPreview: document.getElementById('dataPreview'),
    previewTable: document.getElementById('previewTable'),
    rowCount: document.getElementById('rowCount'),
    colCount: document.getElementById('colCount'),
    
    // Sections
    dataTypesSection: document.getElementById('dataTypesSection'),
    distributionSection: document.getElementById('distributionSection'),
    outlierSection: document.getElementById('outlierSection'),
    correlationSection: document.getElementById('correlationSection'),
    survivalSection: document.getElementById('survivalSection'),
    insightsSection: document.getElementById('insightsSection'),
    
    // Missing values
    columnInfoTable: document.getElementById('columnInfoTable'),
    missingSummary: document.getElementById('missingSummary'),
    
    // Outliers
    ageOutlierCount: document.getElementById('ageOutlierCount'),
    fareOutlierCount: document.getElementById('fareOutlierCount'),
    outlierImpactText: document.getElementById('outlierImpactText'),
    
    // Correlation
    topCorrelations: document.getElementById('topCorrelations'),
    correlationInsight: document.getElementById('correlationInsight'),
    
    // Survival rates
    overallSurvivalRate: document.getElementById('overallSurvivalRate'),
    femaleSurvivalRate: document.getElementById('femaleSurvivalRate'),
    firstClassSurvivalRate: document.getElementById('firstClassSurvivalRate'),
    
    // Insights
    topFactor: document.getElementById('topFactor'),
    factorExplanation: document.getElementById('factorExplanation'),
    supportingEvidence: document.getElementById('supportingEvidence'),
    keyInsights: document.getElementById('keyInsights')
};

// ==============================================
// INITIALIZATION & EVENT LISTENERS
// ==============================================

/**
 * Initialize the dashboard and set up event listeners
 */
function initializeDashboard() {
    // Set up file input event listeners
    domElements.fileInput.addEventListener('change', handleFileSelect);
    
    // Set up drag and drop
    setupDragAndDrop();
    
    // Set up EDA button
    domElements.runEDAButton.addEventListener('click', runFullEDA);
    
    // Load default dataset for demonstration
    loadDefaultDataset();
}

/**
 * Set up drag and drop functionality for file upload
 */
function setupDragAndDrop() {
    const dropArea = domElements.fileDropArea;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlightDropArea, false);
    });
    
    // Remove highlight when dragging leaves
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlightDropArea, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlightDropArea() {
    domElements.fileDropArea.classList.add('dragover');
}

function unhighlightDropArea() {
    domElements.fileDropArea.classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        domElements.fileInput.files = files;
        handleFileSelect({ target: domElements.fileInput });
    }
}

/**
 * Load a default dataset for demonstration purposes
 */
function loadDefaultDataset() {
    console.log('Loading default dataset from:', DEFAULT_CSV_URL);
    
    Papa.parse(DEFAULT_CSV_URL, {
        download: true,
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
                console.warn('Parsing errors:', results.errors);
                alert('Some errors occurred while parsing the CSV. Analysis may be incomplete.');
            }
            
            titanicData = results.data;
            console.log(`Loaded ${titanicData.length} rows with ${Object.keys(titanicData[0]).length} columns`);
            
            // Enable the EDA button and update overview
            domElements.runEDAButton.disabled = false;
            updateDataOverview();
            showDataPreview();
        },
        error: function(error) {
            console.error('Error loading default dataset:', error);
            alert('Error loading default dataset. Please upload a CSV file manually.');
        }
    });
}

/**
 * Handle file selection from input element
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // Check if it's a CSV file
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please select a CSV file.');
        return;
    }
    
    console.log(`Loading file: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    
    // Parse the CSV file
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
                console.warn('Parsing errors:', results.errors);
                alert('Some errors occurred while parsing the CSV. Analysis may be incomplete.');
            }
            
            titanicData = results.data;
            console.log(`Loaded ${titanicData.length} rows with ${Object.keys(titanicData[0]).length} columns`);
            
            // Enable the EDA button and update overview
            domElements.runEDAButton.disabled = false;
            updateDataOverview();
            showDataPreview();
        },
        error: function(error) {
            console.error('Error parsing CSV:', error);
            alert('Error parsing CSV file. Please check the file format and try again.');
        }
    });
}

// ==============================================
// DATA PROCESSING FUNCTIONS
// ==============================================

/**
 * Update the data overview section with basic dataset stats
 */
function updateDataOverview() {
    if (!titanicData || titanicData.length === 0) {
        return;
    }
    
    const rowCount = titanicData.length;
    const colCount = Object.keys(titanicData[0]).length;
    
    domElements.rowCount.textContent = rowCount;
    domElements.colCount.textContent = colCount;
}

/**
 * Display a preview of the data in a table
 */
function showDataPreview() {
    if (!titanicData || titanicData.length === 0) {
        return;
    }
    
    // Show the preview section
    domElements.dataPreview.style.display = 'block';
    
    // Clear existing table content
    domElements.previewTable.innerHTML = '';
    
    // Get column names
    const columns = Object.keys(titanicData[0]);
    
    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    domElements.previewTable.appendChild(thead);
    
    // Create table body with first 10 rows
    const tbody = document.createElement('tbody');
    
    for (let i = 0; i < Math.min(10, titanicData.length); i++) {
        const row = document.createElement('tr');
        
        columns.forEach(col => {
            const td = document.createElement('td');
            let value = titanicData[i][col];
            
            // Handle undefined/null values
            if (value === undefined || value === null) {
                value = '';
                td.classList.add('text-muted');
            }
            
            // Truncate long values
            if (typeof value === 'string' && value.length > 30) {
                value = value.substring(0, 30) + '...';
            }
            
            td.textContent = value;
            row.appendChild(td);
        });
        
        tbody.appendChild(row);
    }
    
    domElements.previewTable.appendChild(tbody);
}

/**
 * Analyze data types and missing values
 */
function analyzeDataTypesAndMissingValues() {
    if (!titanicData || titanicData.length === 0) {
        return;
    }
    
    const columns = Object.keys(titanicData[0]);
    const totalRows = titanicData.length;
    let missingValues
