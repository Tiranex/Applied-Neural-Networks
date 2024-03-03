// Ahora se puede añadir una tabla interactiva, que venga de un icono interrogación y te diga lo que puedes pintar
// Por último se puede añadir función pa borrar Y cambiar el color de la barra de arriba para facilitar.
// Obtén el canvas y el botón por su id
const canvas = document.getElementById("input-canvas");
const ctx = canvas.getContext("2d");
const limpiarBoton = document.getElementById("clear-canvas");
const toolbar=document.getElementById("text-wrapper");
const predictionsList = document.getElementById('predict');
const preview_canvas = document.getElementById('preview');

// Establecer Canvas
canvas.width=canvas.clientWidth;
canvas.height=canvas.clientHeight;
ctx.lineWidth = 12;
ctx.imageSmoothingEnabled = true;
// Set the stroke color to white
ctx.strokeStyle = 'white';

// Preview canvas
preview_canvas.width = toolbar.clientHeight;
preview_canvas.height = toolbar.clientHeight;

// Variable para rastrear si el mouse está presionado
var mousePresionado = false;

// Trackeo ratón
var xmax=0;
var xmin=0;
var ymin=0;
var ymax=0;

// Registra el evento de clic del mouse en el canvas
canvas.addEventListener("mousedown", function(e) {
    mousePresionado = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX , e.clientY-toolbar.clientHeight);
    // Wrap around a rectangle
    if(xmax == 0 && xmin == 0 &&	ymin == 0 && ymax == 0){
		xmax=e.clientX;
		xmin=e.clientX;
		ymin=e.clientY-toolbar.clientHeight;
		ymax=e.clientY-toolbar.clientHeight;
	}
});

// Registra el evento de movimiento del mouse en el canvas
canvas.addEventListener("mousemove", function(e) {
    if (mousePresionado) {
        ctx.lineTo(e.clientX ,  e.clientY-toolbar.clientHeight);
        ctx.stroke();
        update_pos(e);
    }
});

// Registra el evento de liberación del mouse en el canvas
canvas.addEventListener("mouseup", function() {
    mousePresionado = false;
    xmin-=6;
    ymin-=6;
    xmax+=6;
    ymax+=6;
    //draw_rect();
    getTopPredictions();
});

// Registra el evento de salir del área del canvas para detener el dibujo
canvas.addEventListener("mouseleave", function() {
    mousePresionado = false;
    ctx.closePath();
});

// Registra resize del canvas
// Registra el evento de clic del mouse en el canvas
window.addEventListener("resize", function(e) {
    mousePresionado = false;
    canvas.width=canvas.clientWidth;
    canvas.height=canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    background_fill();
});

// Asocia la función de limpiar al botón "Limpiar"
limpiarBoton.addEventListener("click", limpiarCanvas)

// app.js
// Function to get the top 10 predictions for the canvas drawing
function getTopPredictions() {
  var pixelData = ctx.getImageData(xmin, ymin, xmax-xmin, ymax-ymin);

  const targetDim = 28,
			edgeSize = 2,
			resizeDim = targetDim-edgeSize*2,
			padVertically = pixelData.width > pixelData.height,
			padSize = Math.round((Math.max(pixelData.width, pixelData.height) - Math.min(pixelData.width, pixelData.height))/2),
			padSquare = padVertically ? [[padSize,padSize], [0,0], [0,0]] : [[0,0], [padSize,padSize], [0,0]];

    // convert the pixel data into a tensor with 1 data channel per pixel
		// i.e. from [h, w, 4] to [h, w, 1]
		let tensor = tf.browser.fromPixels(pixelData, 1)
			// pad it such that w = h = max(w, h)
			.pad(padSquare, 0)

		// scale it down
		tensor = tf.image.resizeBilinear(tensor, [resizeDim, resizeDim])
			// pad it with blank pixels along the edges (to better match the training data)
			.pad([[edgeSize,edgeSize], [edgeSize,edgeSize], [0,0]], 0)

    // preview canvas
    preview_canvas.width = toolbar.clientHeight;
    preview_canvas.height = toolbar.clientHeight;
    tf.browser.toPixels(tensor, preview_canvas);

		// invert and normalize to match training data
		tensor = tensor.toFloat().div(tf.scalar(255.0))

    tensor= tensor.expandDims(0)
    var predictions = model.predict(tensor).as1D();

    predictions=predictions.dataSync();
    console.log(predictions);

    var topPredictions = getTopKClasses(predictions, 6);
    displayPredictions(topPredictions);
}

const index_list={
  0: "Star",
  1: "Sheep",
  2: "Ambulance",
  3: "Banana",
  4: "Headphones",
  5: "Grass",
  6: "Clock",
  7: "Airplane",
  8: "Rainbow",
  9: "Triangle",
  10: "Pants",
  11: "Anvil",
  12: "Apple",
  13: "Crown",
  14: "Bucket",
  15: "Sun",
  16: "Ant",
  17: "Octopus",
  18: "Circle",
  19: "Table",
  20: "House",
  21: "Donut",
  22: "Alarm clock",
  23: "Parachute",
  24: "The Eiffel Tower",
  25: "Angel",
  26: "Arm",
  27: "Nose",
  28: "Sword",
  29: "Television",
  30: "Telephone",
  31: "Microphone"
}
// Function to display the top predictions in the HTML
function displayPredictions(predictions) {
  predictionsList.innerHTML = '';
  for (const prediction of predictions) {
    const listItem = document.createElement('li');
    listItem.textContent = `${index_list[prediction.className]} (Trust: ${prediction.value.toFixed(4)})`;
    predictionsList.appendChild(listItem);
  }
}

// Utility function to get the top K classes with the highest values
function getTopKClasses(predictions, k) {
  const predictionArray = Array.from(predictions);
  return predictionArray
    .map((value, index) => ({ value, className: index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}

// Set the background color to black
function background_fill(){
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
background_fill();

// limpiar canvas
function limpiarCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  	xmax=0;
  	xmin=0;
  	ymin=0;
  	ymax=0;
    background_fill();
}

// actualizar coordenadas
function update_pos(e){
  if(e.clientX > xmax)
    xmax=e.clientX;
  if(e.clientX < xmin)
    xmin=e.clientX;
  if(e.clientY-toolbar.clientHeight > ymax)
    ymax=e.clientY-toolbar.clientHeight;
  if(e.clientY-toolbar.clientHeight < ymin)
    ymin=e.clientY-toolbar.clientHeight;
}

// cargar modelo
// your_model.js

let model;

async function loadModel() {
  model = await tf.loadLayersModel('model-doodle2/model.json');
  document.getElementById('output').style.display = 'none';
  predictionsList.style.display = 'block'
}

loadModel();

// funcion auxiliar
function draw_rect (){
  ctx.beginPath();
  ctx.moveTo(xmin,ymax);
  ctx.lineTo(xmax, ymax);
  ctx.lineTo(xmax,ymin);
  ctx.lineTo(xmin,ymin);
  ctx.lineTo(xmin,ymax);
  ctx.stroke();
  ctx.closePath();
}
