let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var aSamples=0, bSamples=0, cSamples=0, dSamples=0, eSamples=0, fSamples=0, gSamples=0, hSamples=0, iSamples=0, jSamples=0, kSamples=0, lSamples=0, mSamples=0, nSamples=0, oSamples=0, pSamples=0, qSamples=0, rSamples=0, sSamples=0, tSamples=0, uSamples=0, vSamples=0, wSamples=0, xSamples=0, ySamples=0, zSamples=0;
var names = ["A", "B", "C", "D", "E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  if (!dataset.labels || dataset.labels.length === 0) {
    alert('Немає прикладів для навчання.');
    return;
  }
  const numClasses = Math.max(...dataset.labels) + 1;

  dataset.ys = null;
  dataset.encodeLabels(numClasses);

  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({units: 100, activation: 'relu'}),
      tf.layers.dense({units: 50, activation: 'relu'}),
      tf.layers.dense({units: numClasses, activation: 'softmax'})
    ]
  });

  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']});

  await model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('LOSS: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      }
    }
  });
}


function handleButton(elem){
	switch(elem.id){
    case "0":
      aSamples++;
      document.getElementById("asamples").innerText = "A samples:" + aSamples;
      break;
    case "1":
      bSamples++;
      document.getElementById("bsamples").innerText = "B samples:" + bSamples;
      break;
    case "2":
      cSamples++;
      document.getElementById("csamples").innerText = "C samples:" + cSamples;
      break;
    case "3":
      dSamples++;
      document.getElementById("dsamples").innerText = "D samples:" + dSamples;
      break;
    case "4":
      eSamples++;
      document.getElementById("esamples").innerText = "E samples:" + eSamples;
      break;
    case "5":
      fSamples++;
      document.getElementById("fsamples").innerText = "F samples:" + fSamples;
      break;
    case "6":
      gSamples++;
      document.getElementById("gsamples").innerText = "G samples:" + gSamples;
      break;
    case "7":
      hSamples++;
      document.getElementById("hsamples").innerText = "H samples:" + hSamples;
      break;
    case "8":
      iSamples++;
      document.getElementById("isamples").innerText = "I samples:" + iSamples;
      break;
    case "9":
      jSamples++;
      document.getElementById("jsamples").innerText = "J samples:" + jSamples;
      break;
    case "10":
      kSamples++;
      document.getElementById("ksamples").innerText = "K samples:" + kSamples;
      break;
    case "11":
      lSamples++;
      document.getElementById("lsamples").innerText = "L samples:" + lSamples;
      break;
    case "12":
      mSamples++;
      document.getElementById("msamples").innerText = "M samples:" + mSamples;
      break;
    case "13":
      nSamples++;
      document.getElementById("nsamples").innerText = "N samples:" + nSamples;
      break;
    case "14":
      oSamples++;
      document.getElementById("osamples").innerText = "O samples:" + oSamples;
      break;
    case "15":
      pSamples++;
      document.getElementById("psamples").innerText = "P samples:" + pSamples;
      break;
    case "16":
      qSamples++;
      document.getElementById("qsamples").innerText = "Q samples:" + qSamples;
      break;
    case "17":
      rSamples++;
      document.getElementById("rsamples").innerText = "R samples:" + rSamples;
      break;
    case "18":
      sSamples++;
      document.getElementById("ssamples").innerText = "S samples:" + sSamples;
      break;
    case "19":
      tSamples++;
      document.getElementById("tsamples").innerText = "T samples:" + tSamples;
      break;
    case "20":
      uSamples++;
      document.getElementById("usamples").innerText = "U samples:" + uSamples;
      break;
    case "21":
      vSamples++;
      document.getElementById("vsamples").innerText = "V samples:" + vSamples;
      break;
    case "22":
      wSamples++;
      document.getElementById("wsamples").innerText = "W samples:" + wSamples;
      break;
    case "23":
      xSamples++;
      document.getElementById("xsamples").innerText = "X samples:" + xSamples;
      break;
    case "24":
      ySamples++;
      document.getElementById("ysamples").innerText = "Y samples:" + ySamples;
      break;
    case "25":
      zSamples++;
      document.getElementById("zsamples").innerText = "Z samples:" + zSamples;
      break;

	}
	label = parseInt(elem.id);
  const img = webcam.capture();
  const activation = tf.tidy(() => mobilenet.predict(img).clone());
  img.dispose();
  dataset.addExample(activation, label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    predictionText = "I see " + names[classId];
            
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

async function uploadModel(event){
    const inputFiles = event && event.target && event.target.files ? event.target.files : null;
    if (!inputFiles || inputFiles.length === 0) {
        alert("Файли не обрані.");
        return;
    }

    const files = Array.from(inputFiles);

    if (files.length !== 2) {
        alert("Ви повинні обрати 2 файли.");
        return;
    }

    const hasModelJson = files.some(f => f.name.endsWith('model.json'));
    const hasBin = files.some(f => f.name.endsWith('.bin'));

    if (!hasModelJson || !hasBin) {
        alert("Выберите model.json и соответствующий .bin файл.");
        return;
    }

    try {
        model = await tf.loadLayersModel(tf.io.browserFiles(files));

        // Warm-up
        if (mobilenet && webcam) {
            tf.tidy(() => {
                const act = mobilenet.predict(webcam.capture());
                model.predict(act);
            });
        }
        alert("Model loaded successfully.");
    } catch (err) {
        console.error(err);
        alert("Failed to load model: " + (err && err.message ? err.message : err));
    }
}

function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


function saveModel(){
    model.save('downloads://my_model');
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();