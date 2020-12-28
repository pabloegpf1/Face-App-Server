const express = require('express');
const http = require("http");
const app = express();
const PORT = process.env.PORT || 4000;

const faceApi = require('./faceApi.js');

faceApi.loadTensorFlow()
.then(() => faceApi.loadFaceApi());

app.get('/', async (req, res) => {
  res.send("Welcome to Face App Rest API");
})

//Socket.io

const server = http.createServer(app);
const io = require("socket.io")(server, {
  cors: {
    origin: "https://www.pabloescriva.com/Face-App/",
    methods: ["GET", "POST"]
  }
});

io.on("connection", (socket) => {
  let labeledFaceDescriptors;
  console.log("New client connected");

  io.emit("getLabeledDescriptors");

  socket.on("sendDescriptors", async data => {
    const descriptors = await faceApi.loadLabeledFaceDescriptors(data);
    labeledFaceDescriptors = descriptors;
    console.log("descriptors received!; ", labeledFaceDescriptors);
  });
  
  socket.on("sendImage", async (data, respond) => {
    const detections = await faceApi.getAllDetectionsForImage(data);
    if(detections.length == 0) return;
    const recognitions = await faceApi.recognizeInImage(labeledFaceDescriptors, detections);
    const canvas = await faceApi.createCanvasFromRecognitions(recognitions, detections);
    respond({base64: canvas.toDataURL('image/png')});
  });

  socket.on("disconnect", () => {
    console.log("Client disconnected");
  });

});

server.listen(PORT, () => {
  console.log(`App listening at http://localhost:${PORT}`)
})