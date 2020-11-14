package com.example.flowersclassifier.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Detector extends TFliteModel {
    /**
     * Float MobileNet requires additional normalization of the used input.
     */
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    protected static final int MAX_RESULTS = 10;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    public Detector(Activity activity, String modelPath, String labelsMap, Device device, int numThreads) {
        super(activity, modelPath, labelsMap, device, numThreads);
    }

    @Override
    public List<Recognition> runInference(Bitmap inputImage) {
        // create input tensor
        TensorImage inputTensor = this.loadImage(inputImage);
//        Log.d(TAG, String.format("crop size : (%d, %d)", inputTensor.getWidth(), inputTensor.getHeight()));
//        for (int i = 0; i < this.tflite.getOutputTensorCount(); i++) {
//            int[] outputTensorShape = tflite.getOutputTensor(i).shape();
//            DataType outputTensorDataType = tflite.getOutputTensor(i).dataType();
//            Log.d(TAG, String.format("%d, %d", outputTensorShape.length, outputTensorDataType.byteSize()));
//        }

        // Copy the input data into TensorFlow.
        //Trace.beginSection("feed");
        outputLocations = new float[1][MAX_RESULTS][4];
        outputClasses = new float[1][MAX_RESULTS];
        outputScores = new float[1][MAX_RESULTS];
        numDetections = new float[1];

        Object[] inputArray = {inputTensor.getBuffer()};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        // Run the inference call.
        //Trace.beginSection("run");
        this.tflite.runForMultipleInputsOutputs(inputArray, outputMap);
        //Trace.endSection();
        // cast from float to integer, use min for safety
        int numDetectionsOutput = Math.min(MAX_RESULTS, (int) numDetections[0]);
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        for (int i = 0; i < numDetectionsOutput; ++i) {
            final RectF detection =
                    new RectF(
                            Math.max(0, Math.min(outputLocations[0][i][1],1) * this.imageSizeX),
                            Math.max(0, Math.min(outputLocations[0][i][0],1) * this.imageSizeX),
                            Math.max(0, Math.min(outputLocations[0][i][3],1) * this.imageSizeX),
                            Math.max(0, Math.min(outputLocations[0][i][2],1) * this.imageSizeX)
                            );

            recognitions.add(
                    new Recognition(
                            "" + i, labels.get((int) outputClasses[0][i]), outputScores[0][i], detection));
        }
        //Trace.endSection();
        for (Recognition box : recognitions) {
            Log.d(TAG, box.toString());
        }
        return recognitions;
    }

    /**
     * Loads input image, and applies preprocessing.
     */
    protected TensorImage loadImage(final Bitmap bitmap) {
        int imageTensorIndex = 0;
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        TensorImage inputImageBuffer = new TensorImage(imageDataType);
        inputImageBuffer.load(bitmap);
        //TensorImage inputImage = TensorImage.fromBitmap(bitmap);
        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        //Log.d(TAG, "crop size " + cropSize);
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(this.imageSizeX, this.imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }
}
