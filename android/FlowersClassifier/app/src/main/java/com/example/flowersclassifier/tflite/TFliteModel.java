package com.example.flowersclassifier.tflite;

import android.app.Activity;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;

public abstract class TFliteModel {
    public static final String TAG = "TfliteModelApi";

    protected int imageSizeX = 0;
    protected int imageSizeY = 0;
    //The loaded TensorFlow Lite model.
    private MappedByteBuffer tfliteModel;
    //An instance of the driver class to run model inference with Tensorflow Lite.
    protected Interpreter tflite;
    protected final Interpreter.Options tfliteOptions = new Interpreter.Options();
    protected List<String> labels;
    protected GpuDelegate gpuDelegate = null;

    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    public TFliteModel(Activity activity,
                       String modelPath,
                       String labelsMap,
                       Device device,
                       int numThreads) {
        try {
            // Create the ImageClassifier instance.
            switch (device) {
                case GPU:
                    // TODO: Create a GPU delegate instance and add it to the interpreter options
                    this.gpuDelegate = new GpuDelegate();
                    this.tfliteOptions.addDelegate(gpuDelegate);
                    break;
                case CPU:
                    // Sets whether to use NN API (if available) for op execution.
                    break;
                case NNAPI:
                    this.tfliteOptions.setUseNNAPI(true);
            }
            this.tfliteOptions.setNumThreads(numThreads);
            // load model
            this.tfliteModel = FileUtil.loadMappedFile(activity, modelPath);
            //load labels
            this.labels = FileUtil.loadLabels(activity, labelsMap);
            // initialize interpreter
            this.tflite = new Interpreter(tfliteModel, this.tfliteOptions);
            // Get the input image size information of the underlying tflite model.
            MetadataExtractor metadataExtractor = new MetadataExtractor(tfliteModel);
            // Image shape is in the format of {1, height, width, 3}.
            int[] imageShape = metadataExtractor.getInputTensorShape(/*inputIndex=*/ 0);
            this.imageSizeY = imageShape[1];
            this.imageSizeX = imageShape[2];
            // Creates the input tensor.
            int imageTensorIndex = 0;
            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
            TensorImage inputImageBuffer = new TensorImage(imageDataType);

//            Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
//            Log.d(TAG, String.format("%s, %s", this.imageSizeX, this.imageSizeY));
//            Log.d(TAG, String.valueOf(this.labels));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Runs inference and returns the classification results.
     */
    protected abstract List<Recognition> runInference(final Bitmap inputImage);

    /**
     * Closes the interpreter and model to release resources.
     */
    public void close() {
        if (tflite != null) {
            // TODO: Close the interpreter
            tflite.close();
            tflite = null;
        }
        // TODO: Close the GPU delegate
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }

        tfliteModel = null;
    }

}
