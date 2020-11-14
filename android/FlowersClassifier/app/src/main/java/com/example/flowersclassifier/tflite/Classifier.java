package com.example.flowersclassifier.tflite;


import android.app.Activity;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class Classifier extends TFliteModel {

    /**
     * Float MobileNet requires additional normalization of the used input.
     */
    private static final float IMAGE_MEAN = 0f;
    private static final float IMAGE_STD = 255f;
    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;
    protected static final int MAX_RESULTS = 3;

    public Classifier(Activity activity, String modelPath, String labelsMap, Device device, int numThreads) {
        super(activity, modelPath, labelsMap, device, numThreads);
    }

    @Override
    public List<Recognition> runInference(Bitmap inputImage) {
        // create input tensor
        TensorImage inputTensor = this.process(inputImage);
        Log.d(TAG, String.format("crop size : (%d, %d)", inputTensor.getWidth(), inputTensor.getHeight()));

        // create output tensor
        int outputTensorIndex = 0;
        int[] outputTensorShape = tflite.getOutputTensor(outputTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType outputTensorDataType = tflite.getOutputTensor(outputTensorIndex).dataType();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputTensorShape, outputTensorDataType);


        // run inference
        Trace.beginSection("runInference");
        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputTensor.getBuffer(), outputTensorBuffer.getBuffer().rewind());
        long endTimeForReference = SystemClock.uptimeMillis();
        Trace.endSection();
        Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

        // process outputs
        NormalizeOp normalizationProbOpt = new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
        TensorProcessor outputProcessor = new TensorProcessor.Builder().add(normalizationProbOpt).build();
        Map<String, Float> predictions = new TensorLabel(labels, outputProcessor.process(outputTensorBuffer)).getMapWithFloatValue();
        List<Recognition> results = getTopKProbability(predictions);
        for (Recognition rec : results) {
            Log.d(TAG, String.format("%s : %f", rec.getTitle(), rec.getConfidence()));
        }
        return results;
    }

    /**
     * Gets the top-k results.
     */
    private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
        }
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }


    /**
     * Loads input image, and applies preprocessing.
     */
    protected TensorImage process(final Bitmap bitmap) {
        int imageTensorIndex = 0;
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        TensorImage inputImageBuffer = new TensorImage(imageDataType);
        inputImageBuffer.load(bitmap);
        //TensorImage inputImage = TensorImage.fromBitmap(bitmap);
        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        Log.d(TAG, "crop size " + cropSize);
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(this.imageSizeX, this.imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }
}
