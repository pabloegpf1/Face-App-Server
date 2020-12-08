const tf = require('@tensorflow/tfjs-node')
const faceapi = require('@vladmandic/face-api');
const constants = require('./constants');
const path = require('path');

const faceDetectorOptions = {inputSize: constants.FACE_DETECTOR_INPUT_SIZE};
const useTinyModel = true;
let labeledFaceDescriptors = [];

const loadTensorFlow = async () => {
    await faceapi.tf.ready();
    return faceapi.tf;
}

const loadFaceApi = async () => {
    return Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromDisk(path.join(__dirname, constants.MODEL_PATH)),
        faceapi.nets.faceLandmark68TinyNet.loadFromDisk(path.join(__dirname, constants.MODEL_PATH)),
        faceapi.nets.faceRecognitionNet.loadFromDisk(path.join(__dirname, constants.MODEL_PATH)),
    ])
    .catch((error)=> console.error(constants.FACEAPI_ERROR_TEXT, error));
}

const getLabeledDescriptors = async (label, images) => {
    for(let i = 0; i<images.length; i++){
        await faceapi.awaitMediaLoaded(images[i]);
        const detection = await getDetectionForImage(images[i]);
        descriptorsForSubject.push(detection.descriptor);
    }
    const newDescriptors = await new faceapi.LabeledFaceDescriptors(label, descriptorsForSubject);
    labeledFaceDescriptors.push(newDescriptors);
}

async function getDetectionForImage(image) {
    return await faceapi
        .detectSingleFace(image, new faceapi.TinyFaceDetectorOptions(faceDetectorOptions))
        .withFaceLandmarks(useTinyModel)
        .withFaceDescriptor();
}

async function getAllDetectionsForImage(image) {
    return await faceapi
        .detectAllFaces(image, new faceapi.TinyFaceDetectorOptions(faceDetectorOptions))
        .withFaceLandmarks(useTinyModel)
        .withFaceDescriptors();
}

async function recognizeInImage(image) {
    const detections = await getAllDetectionsForImage(image);
    console.log(labeledFaceDescriptors);
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, constants.MAX_DESCRIPTOR_DISTANCE);
    return detections.map(detection =>
        faceMatcher.findBestMatch(detection.descriptor)
    );
}

async function loadLabeledFaceDescriptors(descriptors) {
    const loadedDescriptors = await JSON.parse(descriptors);
    if(!loadedDescriptors) return;
    await loadedDescriptors.map(async (subject) => {
        const newSubject = new faceapi.LabeledFaceDescriptors(
            subject.label,
            await subject.descriptors.map((descriptor) => Float32Array.from(descriptor))
        );
        labeledFaceDescriptors.push(newSubject);
    });
}

const image = (image) => {
    const decoded = tf.node.decodeImage(image);
    const casted = decoded.toFloat();
    const result = casted.expandDims(0);
    decoded.dispose();
    casted.dispose();
    return result;
}

module.exports = {
    loadTensorFlow,
    loadFaceApi,
    getDetectionForImage,
    getAllDetectionsForImage,
    getLabeledDescriptors,
    recognizeInImage,
    loadLabeledFaceDescriptors,
    image
 }