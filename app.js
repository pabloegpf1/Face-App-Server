const express = require('express');
const http = require("http");
const { DEFAULT_PORT, CLIENT_URL, CLIENT_URL_DEV } = require('./constants.js');
const app = express();
const PORT = process.env.PORT || DEFAULT_PORT;

const faceApi = require('./faceApi.js');

faceApi.loadTensorFlow().then(() => faceApi.loadFaceApi());

app.get('/', async (req, res) => {
  res.send("Welcome to Face App Rest API");
})

//Socket.io
const server = http.createServer(app);
const io = require("socket.io")(server, {
  cors: {
    origin: process.env.NODE_ENV == 'production' ? CLIENT_URL : CLIENT_URL_DEV,
    methods: ["GET", "POST"]
  }
});

io.on("connection", (socket) => {

  console.log("New client connected");

  let labeledFaceDescriptors;

  socket.on("sendDescriptors", async data => {
    const descriptors = await faceApi.loadLabeledFaceDescriptors(data);
    labeledFaceDescriptors = descriptors;
    console.log("descriptors received!; ", labeledFaceDescriptors);
  });
  
  socket.on("recognize", async (data, respond) => {
    let detections, recognitions, canvas;
    detections = await faceApi.getAllDetectionsForImage(data.base64image);
    if(detections.length == 0) respond({success: false});
    try {
      recognitions = await faceApi.recognizeInImage(labeledFaceDescriptors, detections);
      canvas = await faceApi.drawRecognitionsInCanvas(recognitions, detections);
    } catch (error) {
      canvas = await faceApi.drawDetectionsInCanvas(detections);
    }
    respond({
      success: true, 
      base64canvas: canvas.toDataURL('image/png'), 
      initialTime: data.initialTime
    });
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected");
  });

});

server.listen(PORT, () => {
  console.log(`App listening at http://localhost:${PORT}`)
})