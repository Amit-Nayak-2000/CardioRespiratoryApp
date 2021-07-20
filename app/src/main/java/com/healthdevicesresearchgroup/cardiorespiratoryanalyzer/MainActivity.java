package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.DialogFragment;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.primitives.Doubles;
import com.google.firebase.firestore.DocumentReference;
import com.google.firebase.firestore.FirebaseFirestore;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

import java.util.Calendar;
import java.util.List;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity implements SensorEventListener, SharePrompt.ShareDialogListener, DataPrompt.dataPromptListener {
    //GyroX for breathing
    //gFZ for heart beat
    SensorManager Eugene;
    Sensor Gyroscope;
    Sensor Gforce;
    Sensor Magnetometer;

    //Record and plot buttons
    ToggleButton record;
    Button plot;

    //timer and time variable
    Timer timer;
    Double time;

    //recording boolean and filewriter for logging signal values
    boolean isRecording = false;
    FileWriter Squidward;

    //lists for recording time and sensor signals
    List<Double> timestamp;
    ArrayList<Double> gFZ;
    ArrayList<Double> gyroX;
    ArrayList<Double> gyroY;
    ArrayList<Double> gyroZ;
    ArrayList<Double> gFY;
    ArrayList<Double> gFX;
    ArrayList<Double> magX;
    ArrayList<Double> magY;
    ArrayList<Double> magZ;


    //filtered Signals
    double[] filtered_gyroX;
    double[] filtered_gFZ;

    //dataset conversion for JNI
    double[] gyroXDataset;
    double[] gFZDataset;

    // Breathing Rate 0, Heart Rate 1
    double[] rates;

    //List of previously recorded data
    public ArrayList<String> fileList;

    //Template for .csv file
    StringBuilder dataString;

    String filename;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setTitle("Cardiorespiratory Analyzer");

        isRecording = false;

        //Initialize Sensors
        Eugene = (SensorManager)getSystemService(Context.SENSOR_SERVICE);
        Gyroscope = Eugene.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Gforce = Eugene.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Magnetometer = Eugene.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        //Initialize Buttons and fileList
        record = findViewById(R.id.toggleButton);
        plot = findViewById(R.id.plot_button);
        fileList = new ArrayList<String>();

        record.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @SuppressLint("DefaultLocale")
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    // The toggle is enabled
                    initializeLists();
                    Eugene.registerListener(MainActivity.this, Gforce, 20000);
                    Eugene.registerListener(MainActivity.this, Gyroscope, 20000);
                    Eugene.registerListener(MainActivity.this, Magnetometer, 20000);
                    timer = new Timer();
                    time = 0.00D;
                    timestamp = new ArrayList<>();
                    timestamp.add(time);
                    timer.scheduleAtFixedRate(new timeStampAdder(), 20, 20);
                    isRecording = true;
                }
                else {
                    // The toggle is disabled
                    timer.cancel();
                    time = 0.00D;
                    isRecording = false;
                    Eugene.flush(MainActivity.this);
                    Eugene.unregisterListener(MainActivity.this);
                    getFilteredData();
                    confirmShareData();
                    try {
                        //create log file and log data
                        filename = "Sensordata_" + Calendar.getInstance().getTime().toString() + ".csv";
                        Squidward = new FileWriter(new File(getStorage(), filename));
                        logDataToFile();
                        Squidward.write(String.valueOf(dataString));
                        Squidward.close();
                        filename = getStorage() + "/" + filename;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        //Plotting
        plot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                File root = new File(getStorage());
                listDirectory(root);
                Intent intent = new Intent(MainActivity.this, FileViewer.class);
                Bundle args = new Bundle();
                args.putSerializable("ARRAYLIST", (Serializable)fileList);
                intent.putExtra("BUNDLE", args);
                startActivity(intent);
            }
        });

    }

    //get directory of files
    void listDirectory(File root){
        File[] files = root.listFiles();
        fileList.clear();
        for(File file : files) {
            fileList.add(file.getPath());
        }
    }

    //get storage location
    public String getStorage() {
        return Objects.requireNonNull(this.getExternalFilesDir(null)).getAbsolutePath();
    }

    public void onSensorChanged(SensorEvent event){
        if(isRecording){
            switch(event.sensor.getType()){
                case Sensor.TYPE_ACCELEROMETER:
                    gFZ.add((double) event.values[2]);
                    gFY.add((double) event.values[1]);
                    gFX.add((double) event.values[0]);
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    gyroX.add((double) event.values[0]);
                    gyroY.add((double) event.values[1]);
                    gyroZ.add((double) event.values[2]);
                    break;
                case Sensor.TYPE_MAGNETIC_FIELD:
                    magX.add((double) event.values[0]);
                    magY.add((double) event.values[1]);
                    magZ.add((double) event.values[2]);
            }
        }
    }

    //initialize data collection lists
    private void initializeLists(){
        timestamp = new ArrayList<>();
        gFZ = new ArrayList<>();
        gyroX = new ArrayList<>();
        gyroY = new ArrayList<>();
        gyroZ = new ArrayList<>();
        gFY = new ArrayList<>();
        gFX = new ArrayList<>();
        magX = new ArrayList<>();
        magY = new ArrayList<>();
        magZ = new ArrayList<>();


        double[] rates;
        double[] gFZDataset;
        double[] gyroXDataset;
        double[] filtered_gyroX;
        double[] filtered_gFZ;
    }

    //clear data collection lists
    private void clearEntries(){
        timestamp.clear();
        gFZ.clear();
        gFY.clear();
        gFX.clear();
        gyroX.clear();
        gyroY.clear();
        gyroZ.clear();
        magX.clear();
        magY.clear();
        magZ.clear();
        rates[0] = 0;
        rates[1] = 0;
        for(int i = 0; i < shortestList(); i++){
            gFZDataset[i] = 0;
            gyroXDataset[i] = 0;
            filtered_gyroX[i] = 0;
            filtered_gFZ[i] = 0;
        }
        dataString.setLength(0);
        StringBuilder dataString = new StringBuilder("time (s), gyroX (rad/s), gFZ (m/s^2), filtered gyroX (rad/s), filtered gFZ (m/s^2), gyroY (rad/s), gyroZ (rad/s), gFX (m/s^2), gFY (m/s^2), magX (µT), magY (µT), magZ (µT), breathing rate (breaths/min), heart rate (beats/min)\n");
    }

    //shortest list length
    private int shortestList(){
        int[] lengths = new int[4];
        int result;

        lengths[0] = gFZ.size();
        lengths[1] = gyroX.size();
        lengths[2] = timestamp.size();
        lengths[3] = magY.size();

        result = lengths[0];

        for(int i = 1; i < 4; i++){
            if(lengths[i-1] >= lengths[i]){
                result = lengths[i];
            }
        }

        return result;
    }

    //record all the data to large string. Eventually written to file.
    @SuppressLint("DefaultLocale")
    public void logDataToFile(){
        dataString = new StringBuilder("time (s), gyroX (rad/s), gFZ (m/s^2), filtered gyroX (rad/s), filtered gFZ (m/s^2), gyroY (rad/s), gyroZ (rad/s), gFX (m/s^2), gFY (m/s^2), magX (µT), magY (µT), magZ (µT), breathing rate (breaths/min), heart rate (beats/min)\n");

        for(int i = 0; i < shortestList(); i++){
            if(i == 0){
                dataString.append(String.format("%.2f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
                        timestamp.get(i), gyroX.get(i), gFZ.get(i), filtered_gyroX[i], filtered_gFZ[i], gyroY.get(i), gyroZ.get(i), gFX.get(i), gFY.get(i), magX.get(i), magY.get(i), magZ.get(i), rates[0], rates[1]));
            }
            else{
                dataString.append(String.format("%.2f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
                        timestamp.get(i), gyroX.get(i), gFZ.get(i), filtered_gyroX[i], filtered_gFZ[i], gyroY.get(i), gyroZ.get(i), gFX.get(i), gFY.get(i), magX.get(i), magY.get(i), magZ.get(i)));
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    @Override
    public void passBool(boolean shareData) {
        if(shareData){
            Toast.makeText(getApplicationContext(),"Sharing Data",Toast.LENGTH_SHORT).show();
            DialogFragment shareFrag = new DataPrompt();
            shareFrag.show(getSupportFragmentManager(), "share data");
        }
        else{
            Toast.makeText(getApplicationContext(),"Not Sharing Data",Toast.LENGTH_SHORT).show();
            displayResults(filename);
        }
    }

    public List<Double> convertToList(double[] input){
        return Doubles.asList(input);
    }

    public List<Double> truncateList(List<Double> input){
        return input.subList(0,shortestList());
    }

    //time stamp incrementer
    class timeStampAdder extends TimerTask {
        @Override
        public void run() {
            timestamp.add(time+=0.02);
        }
    }

    //convert arraylist to regular array
    public double[] convertArray(ArrayList<Double> arr, int size){
        double[] result = new double[size];

        for(int i = 0; i < result.length; i++){
            result[i] = arr.get(i);
        }

        return result;
    }

    //run signal processing algorithms
    public void getFilteredData(){
        int length = shortestList();
        filtered_gyroX = new double[length];
        filtered_gFZ = new double[length];
        rates = new double[2];
        DataStruct dataPackage = new DataStruct(filtered_gyroX, filtered_gFZ, rates);

        gyroXDataset = convertArray(gyroX, length);
        gFZDataset = convertArray(gFZ, length);

        FilterBrHr(dataPackage, gyroXDataset, gFZDataset);
    }

    //Ask user if they want to share their data
    public void confirmShareData(){
        DialogFragment shareFrag = new SharePrompt();
        shareFrag.show(getSupportFragmentManager(), "ask to share data");

    }

    //get Data From Prompt
    public void sendData(int age, int bpm, int brpm, String gender, boolean covid19){
        DataDoc document = new DataDoc(age, rates[1], rates[0], bpm, brpm, gender, covid19, false, convertToList(gFZDataset), convertToList(filtered_gFZ), convertToList(gyroXDataset), convertToList(filtered_gyroX), truncateList(timestamp));

        FirebaseFirestore db = FirebaseFirestore.getInstance();

        db.collection("recordings")
                .add(document)
                .addOnSuccessListener(new OnSuccessListener<DocumentReference>() {
                    @Override
                    public void onSuccess(DocumentReference documentReference) {
                        Toast.makeText(getApplicationContext(), "Data successfully shared",Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        Toast.makeText(getApplicationContext(), " Failed to upload data",Toast.LENGTH_SHORT).show();
                    }
                });

        displayResults(filename);
    }

    public void displayResults(String filename){
        Intent intent = new Intent(MainActivity.this, DataPlot.class);
        intent.putExtra("Filename", filename);
        startActivity(intent);
    }

    //Signal Processing Algorithms (JNI)
    public native void FilterBrHr(DataStruct dataPackage, double[] breathing, double[] heart);

    //load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("Algorithm");
    }
}
