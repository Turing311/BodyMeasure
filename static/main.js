//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var inputHeight = document.getElementById("height");

//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button
  console.log("submit");

  if (!imagePreview.src || !imagePreview.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

//  loader.classList.remove("hidden");
  // call the predict function of the backend
  var height = inputHeight.value
  predictImage(imagePreview.src, height);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";
  predResult.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}

function previewFile(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = reader.result//URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
//    imageDisplay.classList.remove("loading");

//    displayImage(reader.result, "image-display");
  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image, height) {
  let val = {
	  image: image,
	  height: height
  }
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(val)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  predResult.innerHTML = data.result;
  show(predResult);
}

function hide(el) {
  // hide an element
//  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}

function startWebcam(videoSelector, errorSelector) {
	var videoElement = $(videoSelector)[0];

	var constraints = {
		audio: false,
		video: true
	};

	navigator.mediaDevices.getUserMedia(constraints)
	.then(function(stream) {

		var videoTracks = stream.getVideoTracks();
		console.log('Got stream with constraints:', constraints);
		console.log('Using video device: ' + videoTracks[0].label);
		stream.onended = function() {
			console.log('Stream ended');
		};
		videoElement.srcObject = stream;
	})
	.catch(function(error) {
		if (error.name === 'ConstraintNotSatisfiedError') {
			errorMsg('The resolution ' + constraints.video.width.exact + 'x' +
					constraints.video.width.exact + ' px is not supported by your device.');
		} else if (error.name === 'PermissionDeniedError') {
			errorMsg('Permissions have not been granted to use your camera and ' +
				'microphone, you need to allow the page access to your devices in ' +
				'order for the demo to work.');
		}
		errorMsg('getUserMedia error: ' + error.name, error);
	});

	function errorMsg(msg, error) {
		$(errorSelector).append('<p>' + msg + '</p>');
		if (typeof error !== 'undefined') {
			console.error(error);
		}
	}
}



function captureImage(videoSelector, errorSelector, zoomFactor, stillSelector, inputSelector ) {
	// videoSelector: The VIDEO element with the stream
	// errorSelector: Error messages are dropped here
	// zoomFactor: 1 = original, 0.5 = half size
	// stillSelector: IMG element for captured image
	// inputSelector: INPUT element for form

	var videoElement = $(videoSelector)[0];

	var canvas = $('<canvas></canvas')[0];
	var ctx = canvas.getContext('2d');

	// Hier Größe einstellen
	canvas.width = videoElement.videoWidth * zoomFactor;
	canvas.height = videoElement.videoHeight * zoomFactor;
	// canvas.width = videoElement.videoWidth;
	// canvas.height = videoElement.videoHeight;

	ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
	//save canvas image as data url
	var dataURL = canvas.toDataURL('image/jpeg');
	console.log('Data length:', dataURL.length);
	//set preview image src to dataURL

///////////////////////////////////////////////////  
  imagePreview.src = dataURL;

  show(imagePreview);
  hide(uploadCaption);

  // reset
  predResult.innerHTML = "";
//  imageDisplay.classList.remove("loading");
//  displayImage(dataURL, "image-display");
///////////////////////////////////////////////////

//	$(stillSelector).attr('src', dataURL);
//	$(inputSelector).val(dataURL);
}

$(function() {
	startWebcam('#capturevideo', '#errorMsg');

	//Bind a click to a button to capture an image from the video stream
	$('#shutterbutton').click(function(){
		console.log("Shutter pressed.");
    setTimeout(function() {
      captureImage('#capturevideo', '#errorMsg', 0.5, '#capture_result', '#capture_input' );
    }, 10000)		
	});
});
