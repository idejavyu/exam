class RPSDataset {
  constructor() {
    this.labels = [];
    this.exampleArrays = []; // store Float32Arrays (on CPU) to avoid GPU texture leak
    this.exampleShape = null; // shape WITHOUT batch dim, e.g. [7,7,256]
    this.xs = null; // will hold tf.Tensor4D created just before training
    this.ys = null;
  }

  // example: Tensor with shape [1, ...]
  addExample(example, label) {
    // move data to CPU immediately and free tensor to avoid accumulating GPU textures
    const data = example.dataSync(); // synchronous read to CPU
    const arr = new Float32Array(data); // copy
    if (this.exampleShape == null) {
      // store shape excluding batch dim
      this.exampleShape = example.shape.slice(1);
    }
    example.dispose();
    this.exampleArrays.push(arr);
    this.labels.push(label);
  }

  // Build xs and ys tensors from stored CPU arrays. Call before training.
  encodeLabels(numClasses) {
    const numExamples = this.exampleArrays.length;
    if (numExamples === 0) {
      return;
    }

    // dispose old tensors if present
    if (this.xs) {
      this.xs.dispose();
      this.xs = null;
    }
    if (this.ys) {
      this.ys.dispose();
      this.ys = null;
    }

    const exampleSize = this.exampleArrays[0].length; // length of flattened activation
    // concat all example arrays into one big Float32Array
    const allData = new Float32Array(numExamples * exampleSize);
    for (let i = 0; i < numExamples; i++) {
      allData.set(this.exampleArrays[i], i * exampleSize);
    }

    // create tf.Tensor4D with shape [numExamples, ...exampleShape]
    const shape = [numExamples].concat(this.exampleShape);
    this.xs = tf.tensor4d(allData, shape);

    // create one-hot labels tensor
    const labelsTensor = tf.tensor1d(this.labels, 'int32');
    this.ys = tf.oneHot(labelsTensor, numClasses);
    labelsTensor.dispose();

    // Optionally free CPU arrays to save memory (keep if you want incremental adds)
    // this.exampleArrays = [];
    // this.labels = [];
  }
}
