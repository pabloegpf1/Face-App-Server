const faceapi = require('@vladmandic/face-api');
const constants = require('./constants');
const path = require('path');
const canvas = require('canvas');
const { Canvas, Image } = canvas;

const faceDetectorOptions = {inputSize: constants.FACE_DETECTOR_INPUT_SIZE};
const useTinyModel = true;

const loadTensorFlow = async () => {    
    await faceapi.tf.ready();
    faceapi.env.monkeyPatch({ Canvas, Image });
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

const getDetectionForImage = async(image) => {
    return await faceapi
        .detectSingleFace(image, new faceapi.TinyFaceDetectorOptions(faceDetectorOptions))
        .withFaceLandmarks(useTinyModel)
        .withFaceDescriptor();
}

const getAllDetectionsForImage = async (image) => {
    return await faceapi
        .detectAllFaces(await canvas.loadImage(image), new faceapi.TinyFaceDetectorOptions(faceDetectorOptions))
        .withFaceLandmarks(useTinyModel)
        .withFaceDescriptors();
}

const recognizeInImage = async (labeledFaceDescriptors, detections) => {
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, constants.MAX_DESCRIPTOR_DISTANCE);
    return detections.map(detection =>
        faceMatcher.findBestMatch(detection.descriptor)
    );
}

const loadLabeledFaceDescriptors = async (descriptors) => {
    const labeledFaceDescriptors = [];
    const loadedDescriptors = await JSON.parse(descriptors);
    if(!loadedDescriptors) return;
    await loadedDescriptors.map(async (subject) => {
        const newSubject = new faceapi.LabeledFaceDescriptors(
            subject.label,
            await subject.descriptors.map((descriptor) => Float32Array.from(descriptor))
        );
        labeledFaceDescriptors.push(newSubject);
    });
    return labeledFaceDescriptors;
}

const createCanvasFromRecognitions = async (recognitions, detections) => {
    const canvas = new Canvas();
    canvas.height = 300;
    canvas.width = 300;
    await recognitions.forEach((detection, i) => {      
        const text = detection.toString();
        const drawBox = new faceapi.draw.DrawBox(detections[i].detection._box, { label: text });
        drawBox.draw(canvas);
    });
    return canvas;
}

module.exports = {
    loadTensorFlow,
    loadFaceApi,
    getDetectionForImage,
    getAllDetectionsForImage,
    getLabeledDescriptors,
    recognizeInImage,
    loadLabeledFaceDescriptors,
    createCanvasFromRecognitions
 }