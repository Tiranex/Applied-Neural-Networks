const dropArea = document.getElementById('drop-area');
const inputFile = document.getElementById('input-file');
const imageView = document.getElementById('img-view');
const textarea = document.getElementById('input-text');
const canvas = document.getElementById('canvas');
const coder = document.getElementById('go_button');
const image_wrapper = document.getElementById('img-wrapper');
let tensor = null;
let predictions
let current_mode = 0; // 0 for encode, 1 for decode

/* Parameters */
const IMAGE_SIZE = 256;
const TEXT_SIZE = 128;
const DICT_LEN = 255;

/* Handle classes change */
const encode_button = document.getElementById('encode');
const decode_button = document.getElementById('decode');

encode_button.addEventListener('click', function(){
    encode_button.classList.add('active');
    decode_button.classList.remove('active');
    textarea.readOnly = false;
    coder.textContent = 'ENCODE'; 
    current_mode = 0;
    reset();
});

decode_button.addEventListener('click', function(){
    encode_button.classList.remove('active');
    decode_button.classList.add('active');
    textarea.readOnly = true;
    coder.textContent = 'DECODE';
    current_mode = 1;
    reset();
});

function reset(){
    tensor = null;
    predictions = null;
    imageView.style.backgroundImage = 'none';
    textarea.value = '';
    image_wrapper.style.visibility = 'visible';
    imageView.style.border = '2px dashed #1C6758';
}

inputFile.addEventListener('change', uploadImage);

function uploadImage(e){
    let imgLink = URL.createObjectURL(inputFile.files[0]);
    imageView.style.backgroundImage = `url(${imgLink})`;

    handleFileUpload(e)
    

    image_wrapper.style.visibility = 'hidden'
    imageView.style.border = 'none';
}

function handleFileUpload(event) {
    const file = inputFile.files[0];
    const reader = new FileReader();
    

    reader.onload = function(event) {
      const img = new Image();
      img.onload = function() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        const pixelData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
        tensor = tf.browser.fromPixels(pixelData, 3);

      };
      img.src = event.target.result;
    };
    
    reader.readAsDataURL(file);
  }

  // Helper function to convert data URL to Blob
function daaURLtoBlob(dataURL) {
    const parts = dataURL.split(';base64,');
    const contentType = parts[0].split(':')[1];
    const raw = window.atob(parts[1]);
    const rawLength = raw.length;
    const uInt8Array = new Uint8Array(rawLength);

    for (let i = 0; i < rawLength; ++i) {
      uInt8Array[i] = raw.charCodeAt(i);
}

    return new Blob([uInt8Array], { type: contentType });
}

dropArea.addEventListener('dragover', function(e){
    e.preventDefault();
});

dropArea.addEventListener('drop', function(e){
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadImage(e);
});

coder.addEventListener('click', function(){
    if(tensor == null)
        return
    if(current_mode == 0)
        encode(tensor);
    else if(current_mode == 1)
        decode(tensor);
});
function get_text(){
    const text = textarea.value;
    const encodedText = [new Array(TEXT_SIZE).fill(32)];

    for (let i = 0; i < text.length; i++) {
        const charCode = text.charCodeAt(i);
        encodedText[0][i] = charCode;
    }

    return encodedText;
}


/* Tensorflow */
function encode(tensor){
    const encodedText = tf.tensor(get_text(), [1, TEXT_SIZE]);
    tensor = tensor.toFloat().div(tf.scalar(255.0))
    
    tensor= tensor.expandDims(0)
    
    predictions = encoder.predict([tensor, encodedText])
    
    
    predictions=predictions.dataSync();
    for (let i = 0; i < predictions.length; i++) {
        predictions[i] = parseInt(predictions[i]*255)
    }
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width=IMAGE_SIZE;
    canvas.height=IMAGE_SIZE;
    const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    const data = imageData.data;
    let j =0;
    for (let i = 0; i < data.length; i=i+4) {
        data[i] = predictions[j]; // Red channel
        data[i + 1] = predictions[j+1]; // Green channel
        data[i + 2] = predictions[j+2]; // Blue channel
        data[i + 3] = 255; // Alpha channel
        j+=3;
    }
    ctx.putImageData(imageData, 0, 0);

    const link = document.createElement('a');
    link.href = canvas.toDataURL();
    link.download = 'image.jpg';
    link.click();
    
    
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function decode(tensor){
    tensor = tensor.toFloat().div(tf.scalar(255.0))
    tensor= tensor.expandDims(0)
    predictions = decoder.predict(tensor)
    predictions=predictions.dataSync();
    const text = new Array(TEXT_SIZE);
    for (let i = 0; i < TEXT_SIZE; i++) {
        text[i] = String.fromCharCode(tf.argMax(predictions.slice(i*DICT_LEN, (i+1)*DICT_LEN), axis=-1).dataSync()[0] + 1);
    }
    const joinedText = text.join('');
    textarea.value = joinedText;
}
/* Model loading */
let encoder;
let decoder;

async function loadModel() {
  encoder = await tf.loadLayersModel('model/encoder/model.json');
  decoder = await tf.loadLayersModel('model/decoder/model.json');
}

loadModel();
