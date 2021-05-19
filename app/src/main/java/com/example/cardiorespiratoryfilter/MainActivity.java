package com.example.cardiorespiratoryfilter;

import androidx.appcompat.app.AppCompatActivity;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.hardware.SensorEvent;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ToggleButton;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity implements SensorEventListener{

    //GyroX for breathing
    //gFZ for heart beat
    SensorManager Eugene;
    Sensor Gyroscope;
    Sensor Gforce;
//    Sensor Magnetometer;

    ToggleButton record;
    Button plot;
    Timer timer;
    float time;

    boolean isRecording = false;
    FileWriter Squidward;
    ArrayList<Float> timestamp;
    ArrayList<Double> gFZ;
    ArrayList<Double> gyroX;

    //first dimension is either breathing rate or heart rate
    //0 is breathing rate, 1 is heart rate
    double[] filtered_gyroX;
    double[] filtered_gFZ;
    double[] gyroXDataset;
    double[] gFZDataset;
    public ArrayList<String> fileList;
    double estimatedBR;
    double estimatedHR;
    double[] rates;

    StringBuilder dataString = new StringBuilder("time (s), gyroX (rad/s), gFZ (m/s^2), filtered gyroX (rad/s), filtered gFZ (m/s^2), breathing rate (breaths/min), heart rate (beats/min)\n");


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        isRecording = false;

        Eugene = (SensorManager)getSystemService(Context.SENSOR_SERVICE);
        Gyroscope = Eugene.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Gforce = Eugene.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
//        Magnetometer = Eugene.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);


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
//                    Eugene.registerListener(MainActivity.this, Magnetometer, 20000);
                    timer = new Timer();
                    time = 0;
                    timestamp.add(time);
                    timer.scheduleAtFixedRate(new timeStampAdder(), 20, 20);
                    isRecording = true;
                }
                else {
                    // The toggle is disabled
                    timer.cancel();
                    time = 0;
                    isRecording = false;
                    Eugene.flush(MainActivity.this);
                    Eugene.unregisterListener(MainActivity.this);
                    getFilteredData();
//                    heartBeatDataset = convertArray(gFY);
//                    estimatedHR = FinalMetric(heartBeatDataset, false);
//                    estimatedBR = FinalMetric(filtered_gyroX[0], true);
                    try {
                        Squidward = new FileWriter(new File(getStorage(), "Sensordata_" + System.currentTimeMillis() + ".csv"));
                        logDataToFile();
                        Squidward.write(String.valueOf(dataString));
//                        Squidward.write("\n");
//                        Squidward.write("Estimated Breaths per minute:," +  String.format("%.2f", estimatedBR)  + "\n");
//                        Squidward.write("\n");
//                        Squidward.write("Estimated Beats per minute:," +  String.format("%.2f", estimatedHR)  + "\n");
                        Squidward.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    clearEntries();
                }
            }
        });

        plot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                File root = new File(getStorage());
                listDirectory(root);
                Intent intent = new Intent(MainActivity.this, Fileviewer.class);
                Bundle args = new Bundle();
                args.putSerializable("ARRAYLIST", (Serializable)fileList);
                intent.putExtra("BUNDLE", args);
                startActivity(intent);
            }
        });

    }

    void listDirectory(File root){
        File[] files = root.listFiles();
        fileList.clear();
        for(File file : files) {
            fileList.add(file.getPath());
        }
    }

    public String getStorage(){
        return Objects.requireNonNull(this.getExternalFilesDir(null)).getAbsolutePath();
    }

    public void onSensorChanged(SensorEvent event){
        if(isRecording){
            switch(event.sensor.getType()){
                case Sensor.TYPE_ACCELEROMETER:
//                    gFX.add((double) event.values[0]);
//                    gFY.add((double) event.values[1]);
                    gFZ.add((double) event.values[2]);
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    gyroX.add((double) event.values[0]);
//                    gyroY.add((double) event.values[1]);
//                    gyroZ.add((double) event.values[2]);
                    break;
//                case Sensor.TYPE_MAGNETIC_FIELD:
//                    magX.add((double) event.values[0]);
//                    magY.add((double) event.values[1]);
//                    magZ.add((double) event.values[2]);
//                    break;
            }
        }
    }

    private void initializeLists(){
        timestamp = new ArrayList<>();
//        gFX = new ArrayList<>();
//        gFY = new ArrayList<>();
        gFZ = new ArrayList<>();
        gyroX = new ArrayList<>();
//        gyroY = new ArrayList<>();
//        gyroZ = new ArrayList<>();
//        magX = new ArrayList<>();
//        magY = new ArrayList<>();
//        magZ = new ArrayList<>();
    }

    private void clearEntries(){
        timestamp.clear();
        gFZ.clear();
        gyroX.clear();


        StringBuilder dataString = new StringBuilder("time (s), gyroX (rad/s), gFZ (m/s^2), filtered gyroX (rad/s), filtered gFZ (m/s^2), breathing rate (breaths/min), heart rate (beats/min)\n");
    }

    private int shortestList(){
        int[] lengths = new int[3];
        int result;

        lengths[0] = gFZ.size();
        lengths[1] = gyroX.size();
        lengths[2] = timestamp.size();

        result = lengths[0];

        for(int i = 1; i < 3; i++){
            if(lengths[i-1] >= lengths[i]){
                result = lengths[i];
            }
        }

        return result;
    }

    @SuppressLint("DefaultLocale")
    public void logDataToFile(){
        for(int i = 0; i < shortestList(); i++){
            if(i == 0){
                dataString.append(String.format("%.2f, %f, %f, %f, %f, %f, %f\n",
                        timestamp.get(i), gyroX.get(i), gFZ.get(i), filtered_gyroX[i], filtered_gFZ[i], estimatedBR, estimatedHR));
            }
            else{
                dataString.append(String.format("%.2f, %f, %f, %f, %f\n",
                        timestamp.get(i), gyroX.get(i), gFZ.get(i), filtered_gyroX[i], filtered_gFZ[i]));
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    class timeStampAdder extends TimerTask {
        @Override
        public void run() {
            timestamp.add(time+=0.02);
        }
    }

    public double[] convertArray(ArrayList<Double> arr, int size){
        double[] result = new double[arr.size()];

        for(int i = 0; i < result.length; i++){
            result[i] = arr.get(i);
        }

        return result;
    }

    public void getFilteredData(){
        int length = shortestList();
        filtered_gyroX = new double[length];
        filtered_gFZ = new double[length];
        rates = new double[2];
        DataStruct dataPackage = new DataStruct(filtered_gyroX, filtered_gFZ, rates);

        gyroXDataset = convertArray(gyroX, length);
        gFZDataset = convertArray(gFZ, length);

        FilterBrHr(dataPackage, gyroXDataset, gFZDataset);

        estimatedBR = rates[0];
        estimatedHR = rates[1];
    }

    public native void FilterBrHr(DataStruct dataPackage, double[] breathing, double[] heart);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("Algorithm");
    }
}
