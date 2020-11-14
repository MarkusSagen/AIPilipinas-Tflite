package com.example.flowersclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.flowersclassifier.tflite.Classifier;
import com.example.flowersclassifier.tflite.Recognition;
import com.example.flowersclassifier.tflite.TFliteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.model.Model;

import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private Intent requestFileIntent;
    private ParcelFileDescriptor inputPFD;
    private ImageView imageView;
    private Classifier model;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.model = new Classifier(this, "model.tflite", "labels.txt", TFliteModel.Device.CPU, 4);
        this.imageView = this.findViewById(R.id.imageView);

    }

    public void btnLoadImageClick(View view) {
        requestFileIntent = new Intent(Intent.ACTION_PICK);
        requestFileIntent.setType("image/jpg");
        this.startActivityForResult(requestFileIntent, 0);
    }

    @SuppressLint("DefaultLocale")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent resultIntent) {
        if(resultCode != RESULT_OK){
            return;
        }
        else{
            // Get the file's content URI from the incoming Intent
            try {
                Uri returnUri = resultIntent.getData();
                inputPFD = getContentResolver().openFileDescriptor(returnUri, "r");
                FileDescriptor fd = inputPFD.getFileDescriptor();
                Bitmap image = BitmapFactory.decodeFileDescriptor(fd);
                List<Recognition> recognitions = model.runInference(image);
                for(Recognition rec: recognitions){
                    Toast.makeText(this, String.format("%s - %f",rec.getTitle(), rec.getConfidence()), Toast.LENGTH_LONG).show();
                }
                Log.d("DEBUG", String.valueOf(image.getHeight()));
                this.imageView.setImageBitmap(image);
                inputPFD.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        super.onActivityResult(requestCode, resultCode, resultIntent);
    }

}

