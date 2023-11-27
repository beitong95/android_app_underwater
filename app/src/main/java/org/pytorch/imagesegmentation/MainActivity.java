package org.pytorch.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import android.graphics.Color;


import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.Device;

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

import android.content.res.AssetManager;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import android.widget.EditText;

import android.widget.Spinner;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.TextView;

import android.graphics.Canvas;
import android.graphics.Matrix;


public class MainActivity extends AppCompatActivity implements Runnable {
    private ImageView mImageView;
    private Button mButtonSegment;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private Module mEncoder1 = null;
    private Module mEncoder2 = null;
    private Module mEncoder3 = null;

    private int currentIndex = 0;
    private String mImagename = "test1.jpg";
    private String defaultModelName = "lite_optimized_seg_240p.ptl";
    private int defaultPosition = 0;

    private String[] allFiles;
    private String[] imageFiles;
    private String[] ptlFiles;

    private String currentModelName = "";

    private Button buttonRestart;
    private Button resizeButton;
    private EditText widthEditText;
    private EditText heightEditText;
    private Integer currentWidth;
    private Integer currentHeight;
    // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
    private Spinner modelSpinner;

    private TextView fishCountTextView;
    private TextView inferenceTimeTextView;
    private TextView fishExistTextView;
    private static final int CLASSNUM = 2;
    private int compressImageSize = 128;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public static int findIndex(String[] array, String target) {
        for (int i = 0; i < array.length; i++) {
            if (array[i].equals(target)) {
                return i;
            }
        }
        return 0; // Element not found
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // get all image files and models in the assets folder
        AssetManager assetManager = getAssets();
        try {
            allFiles = assetManager.list("");
            imageFiles = filterJpgFiles(allFiles);
            ptlFiles = filterPtlFiles(allFiles);

            Log.d("ImageSegmentation", "Image Files: " + Arrays.toString(imageFiles));
            Log.d("ImageSegmentation", "Ptl Files: " + Arrays.toString(ptlFiles));

        } catch (IOException e) {
            e.printStackTrace();
        }

        // Image View
        // Create image view
        mImageView = findViewById(R.id.imageView);

        // Resize:
        // Create widgets for image resize feature (width edit text box, height edit text box and resize button)
        widthEditText = findViewById(R.id.editWidth);
        heightEditText = findViewById(R.id.editHeight);
        resizeButton = findViewById(R.id.buttonResize);
        // Set click listener for the resize button
        resizeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Resize the image based on user input
                resizeImage();
            }
        });

        // Run Model:
        // set listener of the segment button (do model inference with the current displayed image).
        mButtonSegment = findViewById(R.id.segmentButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        // Go to Next Image:
        // set listener of the next image button
        buttonRestart = findViewById(R.id.restartButton);
        buttonRestart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Show the next image
                currentIndex = (currentIndex + 1) % imageFiles.length;
                if (currentModelName.equals("VQGANEncode")) {
                    displayImageCenterCropWithSize(currentIndex, compressImageSize);
                } else {
                    displayImage(currentIndex);
                }
            }
        });

        // Count Fish Text View
        fishCountTextView = findViewById(R.id.fishCount);
        inferenceTimeTextView = findViewById(R.id.inferenceTime);
        fishExistTextView = findViewById(R.id.fishExist);

        // Load the initial model
        if (ptlFiles != null && ptlFiles.length > 0) {
            // Model Selector:
            modelSpinner = findViewById(R.id.modelSpinner);
            ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, ptlFiles);
            spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            modelSpinner.setAdapter(spinnerAdapter);

            defaultPosition = findIndex(ptlFiles, defaultModelName);
            Log.d("ImageSegmentation", "default position of" + defaultModelName + " is " + defaultPosition);

            modelSpinner.setSelection(defaultPosition);
            modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> adapterView, View view, int position, long id) {
                    // Load the selected model
                    currentModelName = ptlFiles[position];
                    try {
                        Log.d("ImageSegmentation", currentModelName);

                        // special handle counting fish
                        if (currentModelName.equals("lite_optimized_count_fish_224_224.ptl") || currentModelName.equals("lite_optimized_clf.ptl")) {
                            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), currentModelName));
                            currentHeight = 224;
                            currentWidth = 224;
                            // do we need it?
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    widthEditText.setText("224");
                                    heightEditText.setText("224");
                                }
                            });

                            resizeImage();
                        } else if (currentModelName.equals("lite_optimized_seg_240p.ptl")) {
                            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), currentModelName));
                            currentHeight = 240;
                            currentWidth = 426;
                            // do we need it?
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    widthEditText.setText("426");
                                    heightEditText.setText("240");
                                }
                            });

                            resizeImage();
                        } else if (currentModelName.equals("VQGANEncode")) {
                            mEncoder1 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "encoder_scripted_optimized.ptl"));
                            mEncoder2 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "quant_conv_scripted_optimized.ptl"));
                            mEncoder3 = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "quantize_scripted_optimized.ptl"));
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    widthEditText.setText(String.valueOf(compressImageSize));
                                    heightEditText.setText(String.valueOf(compressImageSize));
                                }
                            });

                            displayImageCenterCropWithSize(currentIndex, compressImageSize);
                        } else {
                            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), currentModelName));
                        }
                    } catch (IOException e) {
                        Log.e("ImageSegmentation", "Error reading assets", e);
                        finish();
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> adapterView) {
                    // Do nothing
                }
            });

        }

        // Load the initial image
        if (imageFiles != null && imageFiles.length > 0) {
            // Display the first image
            resizeImage();
        } else {
            // Handle the case when no images are found
            buttonRestart.setEnabled(false);
            resizeButton.setEnabled(false);
        }
    }

    private void resizeImage() {
        // Get user input for new width and height
        String widthStr = widthEditText.getText().toString();
        String heightStr = heightEditText.getText().toString();

        if (!widthStr.isEmpty() && !heightStr.isEmpty()) {
            int newWidth = Integer.parseInt(widthStr);
            int newHeight = Integer.parseInt(heightStr);
            currentWidth = newWidth;
            currentHeight = newHeight;

            displayImage(currentIndex);
        }
    }


    private void displayImageCenterCropWithSize(int index, int targetSize) {

        try {
            String imagePath = imageFiles[index];
            mBitmap = BitmapFactory.decodeStream(getAssets().open(imagePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        int sourceWidth = mBitmap.getWidth();
        int sourceHeight = mBitmap.getHeight();
        int s = Math.min(sourceWidth, sourceHeight);
        float r = (float)targetSize / s;
        int widthAfterScaled = Math.round(r * sourceWidth);
        Log.d("tbt", ""+widthAfterScaled);
        int heightAfterScaled = Math.round(r * sourceHeight);
        Log.d("tbt", ""+heightAfterScaled);
        mBitmap = Bitmap.createScaledBitmap(mBitmap, widthAfterScaled, heightAfterScaled, true);

        // Calculate the coordinates to center crop the scaled bitmap
        int x = (int)Math.round((widthAfterScaled - targetSize) / 2);
        int y = (int)Math.round((heightAfterScaled - targetSize) / 2);
        Log.d("tbt", x+" "+y);


        // Create a new bitmap and draw the center-cropped region
        Bitmap resultBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(resultBitmap);
        canvas.drawBitmap(mBitmap, -x, -y, null);
        mBitmap = resultBitmap;
        mImageView.setImageBitmap(mBitmap);
    }

    private void displayImage(int index) {
        try {
            // Load the image from the assets folder
            String imagePath = imageFiles[index];
            mBitmap = BitmapFactory.decodeStream(getAssets().open(imagePath));
            mBitmap = Bitmap.createScaledBitmap(mBitmap, currentWidth, currentHeight, true);
            // Display the image in the ImageView
            mImageView.setImageBitmap(mBitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String[] filterJpgFiles(String[] files) {
        List<String> jpgFiles = new ArrayList<>();
        for (String file : files) {
            if (file.toLowerCase().endsWith(".jpg")) {
                jpgFiles.add(file);
            }
        }
        return jpgFiles.toArray(new String[0]);
    }

    private String[] filterPtlFiles(String[] files) {
        List<String> ptlFiles = new ArrayList<>();
        for (String file : files) {
            if (file.toLowerCase().endsWith(".ptl")) {
                ptlFiles.add(file);
            }
        }
        ptlFiles.add("VQGANEncode");
        return ptlFiles.toArray(new String[0]);
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static String clf_res_postprocess(float x) {
        if (sigmoid(x) > 0.5)
            return "true";
        return "false";
    }

    @Override
    public void run() {
        if (currentModelName.equals("VQGANEncode")) {
            float[] mu = {0.0f, 0.0f, 0.0f};
            float[] std = {1.0f, 1.0f, 1.0f};
            final Tensor tempInputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                    mu, std);


            // Convert the PyTorch tensor to a float array
            float[] tempArray = tempInputTensor.getDataAsFloatArray();

            // Apply the operation x = 2 * x - 1 to the float array
            for (int i = 0; i < tempArray.length; i++) {
                tempArray[i] = 2.0f * tempArray[i] - 1.0f;
            }

            // Convert the float array back to a PyTorch tensor
            Tensor inputTensor = Tensor.fromBlob(tempArray, tempInputTensor.shape()); // Adjust the shape as needed
            final long startTime = SystemClock.elapsedRealtime();
            // run model
            Tensor outTensors = mEncoder1.forward(IValue.from(inputTensor)).toTensor();
//            final float[] intResult = outTensors.getDataAsFloatArray();
//            Log.d("tbt", Arrays.toString(intResult));
            outTensors = mEncoder2.forward(IValue.from(outTensors)).toTensor();
            outTensors = mEncoder3.forward(IValue.from(outTensors)).toTensor();
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            Log.d("tbt",  "inference time (ms): " + inferenceTime);
            final long[] results = outTensors.getDataAsLongArray();

            Log.d("tbt", "result: " + Arrays.toString(results));
            Log.d("tbt", "result length: " + results.length );
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
//                    fishCountTextView.setText(String.valueOf(Math.round(results[0])));
                    mButtonSegment.setEnabled(true);
                    mButtonSegment.setText(getString(R.string.segment));
                    mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                    inferenceTimeTextView.setText(inferenceTime+" ms");
                }
            });
        }
        else if (currentModelName.equals("lite_optimized_count_fish_224_224.ptl")) {

            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            final float[] inputs = inputTensor.getDataAsFloatArray();

            final long startTime = SystemClock.elapsedRealtime();
//        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
            final Tensor outTensors = mModule.forward(IValue.from(inputTensor)).toTensor();
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            Log.d("ImageSegmentation",  "inference time (ms): " + inferenceTime);


//        final Tensor outputTensor = outTensors.get("out").toTensor();
            final Tensor outputTensor = outTensors;

            final float[] results = outputTensor.getDataAsFloatArray();

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    fishCountTextView.setText(String.valueOf(Math.round(results[0])));
                    mButtonSegment.setEnabled(true);
                    mButtonSegment.setText(getString(R.string.segment));
                    mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                    inferenceTimeTextView.setText(inferenceTime+" ms");
                }
            });
        } else if (currentModelName.equals("lite_optimized_clf.ptl")) {
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            final float[] inputs = inputTensor.getDataAsFloatArray();

            final long startTime = SystemClock.elapsedRealtime();
//        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
            final Tensor outTensors = mModule.forward(IValue.from(inputTensor)).toTensor();
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            Log.d("ImageSegmentation",  "inference time (ms): " + inferenceTime);


//        final Tensor outputTensor = outTensors.get("out").toTensor();
            final Tensor outputTensor = outTensors;

            final float[] results = outputTensor.getDataAsFloatArray();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    fishExistTextView.setText(clf_res_postprocess(results[0]));
                    mButtonSegment.setEnabled(true);
                    mButtonSegment.setText(getString(R.string.segment));
                    mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                    inferenceTimeTextView.setText(inferenceTime+" ms");
                }
            });
        }
        else {
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            final float[] inputs = inputTensor.getDataAsFloatArray();

            final long startTime = SystemClock.elapsedRealtime();
//        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
            final Tensor outTensors = mModule.forward(IValue.from(inputTensor)).toTensor();
            final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
            Log.d("ImageSegmentation",  "inference time (ms): " + inferenceTime);
            final Tensor outputTensor = outTensors;

            final float[] scores = outputTensor.getDataAsFloatArray();

            int width = mBitmap.getWidth();
            int height = mBitmap.getHeight();
//        for (int i = 0; i < width * height; i++) {
//            Log.d("ImageSegmentation",  "class 0 " + scores[i] + " class 1" + scores[i+width*height]);
//        }
            Log.d("ImageSegmentation",  "outputTensor len: " + scores.length);
            Log.d("ImageSegmentation",  "bit map width: " + width);
            Log.d("ImageSegmentation",  "bit map height: " + height);
            int[] intValues = new int[width * height];
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    int maxi = 0, maxj = 0, maxk = 0;
                    double maxnum = -Double.MAX_VALUE;
                    for (int i = 0; i < CLASSNUM; i++) {
                        float score = scores[i * (width * height) + j * width + k];
                        if (score > maxnum) {
                            maxnum = score;
                            maxi = i; maxj = j; maxk = k;
                        }
                    }
                    if (maxi == 1) {
//                    mBitmap.setPixel(10, 10, Color.GREEN);
                        intValues[maxj * width + maxk] = 0xFF00FF00;
                    }

                    else
//                    continue;
                        intValues[maxj * width + maxk] = 0xFF000000;
//                    intValues[maxj * width + maxk] = int(mBitmap.getColor(maxj, maxk));

                }
            }

            Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
            Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mImageView.setImageBitmap(transferredBitmap);
                    mButtonSegment.setEnabled(true);
                    mButtonSegment.setText(getString(R.string.segment));
                    mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                    inferenceTimeTextView.setText(inferenceTime+" ms");

                }
            });
        }



    }
}
