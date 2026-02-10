## **Summary**

### **1. Data Processing Pipeline**
- **CSV Parsing**: Custom parser with quote handling, NaN conversion, and null detection
- **Missing Value Imputation**:
  - `Age`: Median imputation (training data median)
  - `Fare`: Median imputation (training data median)  
  - `Embarked`: Mode imputation (most frequent value)
  - `SibSp/Parch`: Zero imputation for missing
- **Feature Encoding**:
  - Numerical: Standardization (mean=0, std=1)
  - Categorical: One-hot encoding
  - No feature engineering (removed family features)

### **2. Model Architecture**
- **Type**: Sequential Neural Network (TensorFlow.js)
- **Layers**:
  1. Input: [n_features] → 32 units (ReLU activation)
  2. Hidden: 32 → 16 units (ReLU activation) 
  3. Output: 16 → 1 unit (Sigmoid activation)
- **Parameters**: ~1,000-2,000 total (varies by feature count)
- **Input Features**: 11 total (4 numeric + 7 one-hot encoded)

### **3. Training Configuration**
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Split**: 80/20 train/validation
- **Callbacks**: Epoch-end logging, progress tracking

### **4. Feature Importance Calculation**
- **Method**: Permutation Importance
- **Process**:
  1. Calculate baseline validation accuracy
  2. Shuffle each feature column individually
  3. Measure accuracy drop vs baseline
  4. Normalize scores to 100% total
- **Output**: Relative importance percentages (sum to 100%)

### **5. Prediction System**
- **Threshold**: 0.5 (configurable via slider 0.0-1.0)
- **Probability Display**: Sigmoid output (0.0-1.0)
- **Confidence Categories**:
  - High: ≥0.7 or ≤0.3 (certain prediction)
  - Medium: 0.4-0.6 or 0.3-0.7 (moderate certainty)
  - Low: Near 0.5 (uncertain prediction)
- **Output Format**: Passenger ID, Binary Prediction, Probability, Confidence Level

### **6. Performance Metrics**
- **Confusion Matrix**: TP, TN, FP, FN counts
- **Calculated Metrics**:
  - Accuracy: (TP+TN)/Total
  - Precision: TP/(TP+FP)
  - Recall: TP/(TP+FN)  
  - F1-Score: 2×(Precision×Recall)/(Precision+Recall)
- **Real-time Updates**: Metrics update with threshold changes

### **7. UI/UX Features**
- **Progress Visualization**: Epoch progress bar with live metrics
- **Table Display**: Scrollable containers for wide data tables
- **Status System**: Color-coded messages (info/success/warning/error)
- **Interactive Elements**: Threshold slider, disabled state management
- **Responsive Design**: Mobile-friendly grid layout

### **8. Data Flow**
```
CSV Upload → Parse → Inspect → Preprocess → Build Model → 
Train → Evaluate → Predict → Display Results
```

### **9. Key Technical Decisions**
- **In-browser processing**: No server calls, all client-side
- **TensorFlow.js**: WebGL acceleration for training
- **Progressive enhancement**: Buttons enable sequentially
- **Memory management**: Tensor disposal to prevent leaks
- **Error handling**: Try-catch with user-friendly messages

### **10. Browser Requirements**
- **JavaScript**: ES6+ compatible browser
- **TensorFlow.js**: WebGL support recommended
- **Memory**: 100MB+ recommended for large datasets
- **Modern browsers**: Chrome 70+, Firefox 65+, Safari 12+
