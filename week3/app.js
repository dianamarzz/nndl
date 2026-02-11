/**
 * Neural Network Design: The Gradient Puzzle
 * ------------------------------------------------------------
 * Mission: Transform random noise into a smooth, directional gradient
 * WITHOUT using target labels. Only rearrange existing pixels.
 *
 * Constraint: Input histogram ≈ Output histogram (no new colors).
 * Analogy: Sliding puzzle – move tiles, don't paint over them.
 *
 * ============= STUDENT TODOs =============
 * [TODO-A] Architecture: Implement 'transformation' and 'expansion' in createStudentModel()
 * [TODO-B] Custom Loss: Add smoothness + direction terms to studentLoss()
 * [TODO-C] Compare baseline vs student (visual + loss)
 * ==========================================
 */

// ------------------------------------------------------------
// 1. GLOBAL STATE & CONFIGURATION
// ------------------------------------------------------------
const CONFIG = {
  inputShapeModel: [16, 16, 1],   // no batch dim (for layer definition)
  inputShapeData: [1, 16, 16, 1], // includes batch dim (for tensors)
  learningRate: 0.03,
  autoTrainSpeed: 60,            // ms per step
  smoothnessWeight: 0.15,        // [TODO-B] suggested starting weight
  directionWeight: 0.25,        // [TODO-B] suggested starting weight
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null,                // fixed noise tensor
  baselineModel: null,
  studentModel: null,
  optimizer: null,
  currentArch: 'compression',
};

// ------------------------------------------------------------
// 2. LOSS COMPONENTS (fully implemented)
// ------------------------------------------------------------

// Standard Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

/**
 * Smoothness penalty (Total Variation)
 * Encourages local coherence: neighbor pixels should be similar.
 */
function smoothness(yPred) {
  return tf.tidy(() => {
    // diffX: pixel[i,j] - pixel[i,j+1]  (horizontal)
    const left = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1]);
    const right = yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]);
    const diffX = left.sub(right);

    // diffY: pixel[i,j] - pixel[i+1,j]  (vertical)
    const top = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1]);
    const bottom = yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]);
    const diffY = top.sub(bottom);

    // Return average squared difference
    return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
  });
}

/**
 * Directionality penalty (gradient from left to right)
 * Reward pixels being brighter on the right side.
 * Returns negative correlation so minimizing = stronger gradient.
 */
function directionX(yPred) {
  return tf.tidy(() => {
    const width = 16;
    // Linear ramp from -1 (left) to +1 (right)
    const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
    // Negative mean: we want high correlation (yPred * mask)
    return tf.mean(yPred.mul(mask)).mul(-1);
  });
}

// ------------------------------------------------------------
// 3. MODEL ARCHITECTURE
// ------------------------------------------------------------

// Baseline: fixed compression autoencoder (MSE only, not modified)
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// -----------------------------------------------------------------
// [TODO-A]: STUDENT ARCHITECTURE DESIGN
// Modify this function to implement 'transformation' and 'expansion'.
// -----------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === 'compression') {
    // Bottleneck: information is compressed
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  } else if (archType === 'transformation') {
    // ========== [TODO-A] ==========
    // Transformation (1:1 mapping): hidden dimension ≈ input size (256)
    // Goal: preserve capacity, rearrange pixels, no compression.
    // Remove the throw statement and uncomment the lines below.
    // throw new Error("Transformation architecture NOT implemented yet! (TODO-A)");
    
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  } else if (archType === 'expansion') {
    // ========== [TODO-A] ==========
    // Expansion (overcomplete): hidden dimension > 256
    // Goal: increased capacity, can learn more complex remapping.
    // Remove the throw statement and uncomment the lines below.
    // throw new Error("Expansion architecture NOT implemented yet! (TODO-A)");
    
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));

  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------
// 4. CUSTOM LOSS FUNCTION – [TODO-B]
// ------------------------------------------------------------
// -----------------------------------------------------------------
// [TODO-B]: STUDENT LOSS DESIGN
// Modify this function to create a smooth, directional gradient.
// Start with MSE, then add smoothness and direction penalties.
// -----------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Reconstruction – be faithful to input (pixel values)
    const lossMSE = mse(yTrue, yPred);

    // 2. [TODO-B] Smoothness – encourage local similarity
    //    Increase weight to make output smoother
    const lossSmooth = smoothness(yPred).mul(CONFIG.smoothnessWeight);

    // 3. [TODO-B] Direction – encourage left-dark, right-bright
    //    Increase weight to make gradient stronger
    const lossDir = directionX(yPred).mul(CONFIG.directionWeight);

    // ============================================
    // Total loss: combine all terms.
    // Experiment with weights in CONFIG object above.
    // ============================================
    return lossMSE.add(lossSmooth).add(lossDir);
    
    // ---------- Default (MSE only) ----------
    // return lossMSE; 
  });
}

// ------------------------------------------------------------
// 5. TRAINING LOOP (gradient tape)
// ------------------------------------------------------------
async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.baselineModel) {
    log('❌ Models not initialized. Resetting...', true);
    resetModels();
    return;
  }

  // ----- Train Baseline (MSE only) -----
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // ----- Train Student (Custom Loss) -----
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });

    log(`✅ Step ${state.step} | Baseline: ${baselineLossVal.toFixed(5)} | Student: ${studentLossVal.toFixed(5)}`);
  } catch (e) {
    log(`❌ Student training error: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize every 3 steps or on manual step
  if (state.step % 3 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ------------------------------------------------------------
// 6. UI RENDERING & INITIALIZATION
// ------------------------------------------------------------
async function render() {
  if (!state.baselineModel || !state.studentModel) return;
  
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById('canvas-baseline'));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById('canvas-student'));

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(baseLoss, studLoss) {
  document.getElementById('loss-baseline').innerText = `Loss: ${baseLoss.toFixed(5)}`;
  document.getElementById('loss-student').innerText = `Loss: ${studLoss.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById('log-area');
  const entry = document.createElement('div');
  entry.innerText = `> ${msg}`;
  if (isError) entry.classList.add('error');
  el.prepend(entry);
  if (el.children.length > 8) el.removeChild(el.lastChild);
}

// ------------------------------------------------------------
// 7. RESET & INIT
// ------------------------------------------------------------
function resetModels(archType = null) {
  // Handle event object edge-case
  if (typeof archType !== 'string') {
    archType = document.querySelector('input[name="arch"]:checked')?.value || 'compression';
  }
  state.currentArch = archType;

  // Stop auto-train during reset
  if (state.isAutoTraining) stopAutoTrain();

  // Dispose old resources
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  // Create fresh models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
    log(`🔄 Student architecture set to: ${archType}`);
  } catch (e) {
    log(`⚠️ ${e.message}`, true);
    // Fallback: use compression so app doesn't break
    state.studentModel = createStudentModel('compression');
    log('⚡ Falling back to Compression for stability.', true);
  }

  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  document.getElementById('student-arch-label').innerText = 
    archType.charAt(0).toUpperCase() + archType.slice(1);
  
  render().then(() => {
    updateLossDisplay(0, 0);
  });
}

function init() {
  // Fixed random seed for reproducibility
  tf.setBackend('cpu');
  
  // Generate fixed noise (same every session)
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1, 'float32');
  
  // Draw input canvas
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById('canvas-input'));

  // Bind UI events
  document.getElementById('btn-train').addEventListener('click', () => trainStep());
  document.getElementById('btn-auto').addEventListener('click', toggleAutoTrain);
  document.getElementById('btn-reset').addEventListener('click', () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      resetModels(e.target.value);
    });
  });

  // Initial models
  resetModels('compression');
  
  log('🚀 Ready. Start training — then edit [TODO-A] and [TODO-B] in app.js');
}

// ------------------------------------------------------------
// 8. AUTO TRAIN LOOP
// ------------------------------------------------------------
function toggleAutoTrain() {
  const btn = document.getElementById('btn-auto');
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = '⏸️ Auto Train (Stop)';
    btn.classList.add('btn-stop');
    autoTrainLoop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById('btn-auto');
  btn.innerText = '▶️ Auto Train (Start)';
  btn.classList.remove('btn-stop');
}

function autoTrainLoop() {
  if (state.isAutoTraining) {
    trainStep().then(() => {
      setTimeout(autoTrainLoop, CONFIG.autoTrainSpeed);
    });
  }
}

// Start the app
init();

// ------------------------------------------------------------
// [TODO-C] Compare baseline vs student
// Visualization and loss comparison are already shown live.
// Observe: Baseline memorizes noise; Student (with custom loss)
//          forms a gradient while preserving histogram.
// ------------------------------------------------------------