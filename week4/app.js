const studentLossFn = (yTrue, yPred) => {
    // MSE loss - base reconstruction error
    const mseLoss = tf.losses.meanSquaredError(yTrue, yPred);
    
    // TV loss - now using mean() instead of sum() for better scaling
    const rightDiff = yPred.slice([0,0,0,0], [1,16,15,1]).sub(yPred.slice([0,0,1,0], [1,16,15,1]));
    const downDiff = yPred.slice([0,0,0,0], [1,15,16,1]).sub(yPred.slice([0,1,0,0], [1,15,16,1]));
    const tvLoss = tf.square(rightDiff).mean().add(tf.square(downDiff).mean());
    
    // Direction loss - stronger influence to encourage ramp pattern
    const correlation = yPred.mul(targetRamp).mean();
    const dirLoss = correlation.neg();
    
    // New weights: TV=0.05 (less smoothing), Direction=0.5 (much stronger direction)
    return mseLoss.add(tf.scalar(0.05).mul(tvLoss)).add(tf.scalar(0.5).mul(dirLoss));
};