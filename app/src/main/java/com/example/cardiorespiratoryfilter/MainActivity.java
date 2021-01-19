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
import android.widget.ArrayAdapter;
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

    SensorManager Eugene;
    Sensor Gyroscope;
    Sensor Gforce;
    Sensor Magnetometer;

    ToggleButton record;
    Button plot;
    Timer timer;
    float time;

    boolean isRecording = false;
    FileWriter Squidward;
    //Arraylists to retrieve data
    ArrayList<Float> timestamp;
    ArrayList<Double> gFX;
    ArrayList<Double> gFY;
    ArrayList<Double> gFZ;
    ArrayList<Double> gyroX;
    ArrayList<Double> gyroY;
    ArrayList<Double> gyroZ;
    ArrayList<Double> magX;
    ArrayList<Double> magY;
    ArrayList<Double> magZ;
    //first dimension is either breathing rate or heart rate
    //0 is breathing rate, 1 is heart rate
    double[][] filtered_gFX = new double[2][];
    double[][] filtered_gFY = new double[2][];
    double[][] filtered_gFZ = new double[2][];
    double[][] filtered_gyroX = new double[2][];
    double[][] filtered_gyroY = new double[2][];
    double[][] filtered_gyroZ = new double[2][];
    double[][] filtered_magX = new double[2][];
    double[][] filtered_magY = new double[2][];
    double[][] filtered_magZ = new double[2][];
    public ArrayList<String> fileList;
    double estimatedBR;

//    listView listview;


    StringBuilder dataString = new StringBuilder("time (s), gFX (m/s^2), gFY (m/s^2), gFZ (m/s^2), gyroX (rad/s), gyroY (rad/s), gyroZ (rad/s), magX (µT), magY (µT), magZ (µT)," +
            " BR_Filtered gFX (m/s^2), BR_Filtered gFY (m/s^2), BR_Filtered gFZ (m/s^2), BR_Filtered gyroX (rad/s), BR_Filtered gyroY (rad/s), BR_Filtered gyroZ (rad/s)," +
            " BR_Filtered magX (µT), BR_Filtered magY (µT), BR_Filtered magZ (µT), HR_Filtered gFX (m/s^2), HR_Filtered gFY (m/s^2), HR_Filtered gFZ (m/s^2)," +
            " HR_Filtered gyroX (rad/s), HR_Filtered gyroY (rad/s), HR_Filtered gyroZ (rad/s), HR_Filtered magX (µT), HR_Filtered magY (µT), HR_Filtered magZ (µT)\n");


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        isRecording = false;

        Eugene = (SensorManager)getSystemService(Context.SENSOR_SERVICE);
        Gyroscope = Eugene.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Gforce = Eugene.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Magnetometer = Eugene.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);


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
                    time = 0;
                    timestamp.add(time);
                    timer.scheduleAtFixedRate(new timeStampAdder(), 20, 20);
                    isRecording = true;
                }
                else {
                    // The toggle is disabled
                    timer.cancel();
                    isRecording = false;
                    Eugene.flush(MainActivity.this);
                    Eugene.unregisterListener(MainActivity.this);
                    getFilteredData();
                    estimatedBR = BreathingRateFFT(filtered_gyroX[0], 50);
                    try {
                        Squidward = new FileWriter(new File(getStorage(), "Sensordata_" + System.currentTimeMillis() + ".csv"));
                        logDataToFile();
                        Squidward.write(String.valueOf(dataString));
                        Squidward.write("\n");
                        Squidward.write("Estimated Breaths per minute:," +  String.format("%.2f", estimatedBR)  + "\n");
                        Squidward.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
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
                    gFX.add((double) event.values[0]);
                    gFY.add((double) event.values[1]);
                    gFZ.add((double) event.values[2]);
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
                    break;
            }
        }
    }

    private void initializeLists(){
        timestamp = new ArrayList<>();
        gFX = new ArrayList<>();
        gFY = new ArrayList<>();
        gFZ = new ArrayList<>();
        gyroX = new ArrayList<>();
        gyroY = new ArrayList<>();
        gyroZ = new ArrayList<>();
        magX = new ArrayList<>();
        magY = new ArrayList<>();
        magZ = new ArrayList<>();
    }

    private int shortestList(){
        int[] lengths = new int[10];
        int result;

        lengths[0] = gFX.size();
        lengths[1] = gFY.size();
        lengths[2] = gFZ.size();
        lengths[3] = gyroX.size();
        lengths[4] = gyroY.size();
        lengths[5] = gyroZ.size();
        lengths[6] = magX.size();
        lengths[7] = magY.size();
        lengths[8] = magZ.size();
        lengths[9] = timestamp.size();

        result = lengths[0];

        for(int i = 1; i < 10; i++){
            if(lengths[i-1] >= lengths[i]){
                result = lengths[i];
            }
        }

        return result;
    }

    @SuppressLint("DefaultLocale")
    public void logDataToFile(){
        for(int i = 0; i < shortestList(); i++){
            dataString.append(String.format("%.2f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
                    timestamp.get(i), gFX.get(i), gFY.get(i), gFZ.get(i), gyroX.get(i), gyroY.get(i), gyroZ.get(i), magX.get(i), magY.get(i), magZ.get(i),
                    filtered_gFX[0][i], filtered_gFY[0][i], filtered_gFZ[0][i], filtered_gyroX[0][i], filtered_gyroY[0][i], filtered_gyroZ[0][i],
                    filtered_magX[0][i], filtered_magY[0][i], filtered_magZ[0][i], filtered_gFX[1][i], filtered_gFY[1][i], filtered_gFZ[1][i],
                    filtered_gyroX[1][i], filtered_gyroY[1][i], filtered_gyroZ[1][i], filtered_magX[0][i], filtered_magY[0][i], filtered_magZ[0][i]));
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

    public double[] convertArray(ArrayList<Double> arr){
        double[] result = new double[arr.size()];

        for(int i = 0; i < result.length; i++){
            result[i] = arr.get(i);
        }

        return result;
    }

    public void getFilteredData(){
        filtered_gFX[0] = FilterBrHr(convertArray(gFX), true);
        filtered_gFY[0] = FilterBrHr(convertArray(gFY), true);
        filtered_gFZ[0] = FilterBrHr(convertArray(gFZ), true);
        filtered_gFX[1] = FilterBrHr(convertArray(gFX), false);
        filtered_gFY[1] = FilterBrHr(convertArray(gFY), false);
        filtered_gFZ[1] = FilterBrHr(convertArray(gFZ), false);

        filtered_gyroX[0] = FilterBrHr(convertArray(gyroX), true);
        filtered_gyroY[0] = FilterBrHr(convertArray(gyroY), true);
        filtered_gyroZ[0] = FilterBrHr(convertArray(gyroZ), true);
        filtered_gyroX[1] = FilterBrHr(convertArray(gyroX), false);
        filtered_gyroY[1] = FilterBrHr(convertArray(gyroY), false);
        filtered_gyroZ[1] = FilterBrHr(convertArray(gyroZ), false);

        filtered_magX[0] = FilterBrHr(convertArray(magX), true);
        filtered_magY[0] = FilterBrHr(convertArray(magY), true);
        filtered_magZ[0] = FilterBrHr(convertArray(magZ), true);
        filtered_magX[1] = FilterBrHr(convertArray(magX), false);
        filtered_magY[1] = FilterBrHr(convertArray(magY), false);
        filtered_magZ[1] = FilterBrHr(convertArray(magZ), false);
    }

    public native double[] FilterBrHr(double[] data, boolean BR);

    public native double BreathingRateFFT(double[] data, int samplingFreq);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("Algorithm");
    }
}
