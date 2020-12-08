const express = require('express');
const app = express();
const multer = require('multer');

const faceApi = require('./faceApi.js');
const constants = require('./constants.js');

const upload = multer();

faceApi.loadTensorFlow()
.then(() => faceApi.loadFaceApi());

app.get('/', async (req, res) => {
  res.send("Welcome to Face App Rest API");
})

app.post('/loadDescriptors', upload.none(constants.DESCRIPTORS_KEY), async (req, res) => {
  faceApi.loadLabeledFaceDescriptors(req.body[constants.DESCRIPTORS_KEY]);
  res.sendStatus(200);
})

app.post('/detection', upload.single(constants.IMAGE_KEY), async (req, res) => {
  const image = faceApi.image(req.file.buffer);
  const detection = await faceApi.getDetectionForImage(image);
  res.send(detection);
})

app.post('/detections', upload.single(constants.IMAGE_KEY), async (req, res) => {
  const image = faceApi.image(req.file.buffer);
  const detections = await faceApi.getAllDetectionsForImage(image);
  res.send(detections);
})

app.post('/recognize', upload.single(constants.IMAGE_KEY), async (req, res) => {
  const image = faceApi.image(req.file.buffer);
  const recognitions = await faceApi.recognizeInImage(image);
  res.send(recognitions);
})

app.listen(constants.PORT, () => {
  console.log(`App listening at http://localhost:${constants.PORT}`)
})