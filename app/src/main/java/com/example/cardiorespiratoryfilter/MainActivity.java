package com.example.cardiorespiratoryfilter;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.hardware.SensorEvent;
import android.widget.CompoundButton;
import android.widget.Filter;
import android.widget.ToggleButton;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity implements SensorEventListener{

    SensorManager Eugene;
    Sensor Gyroscope;
    Sensor Gforce;
    Sensor Magnetometer;

    ToggleButton record;
    Timer timer;
    float time;

    boolean isRecording = false;
    FileWriter Squidward;
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
    //ArrayList<Double> Filtered_Data;
    double convertedData[];
    double Filtered_Data[];


    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        isRecording = false;

        Eugene = (SensorManager)getSystemService(Context.SENSOR_SERVICE);
        Gyroscope = Eugene.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Gforce = Eugene.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Magnetometer = Eugene.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);


        record = (ToggleButton) findViewById(R.id.toggleButton);
        record.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
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
                    convertedData = convertArray(gFZ);
                    Filtered_Data = FilterBrHr(convertedData, true);
                    try {
                        Squidward = new FileWriter(new File(getStorage(), "Sensordata_" + System.currentTimeMillis() + ".csv"));
                        Squidward.write("time (s), gFX (m/s^2), gFY (m/s^2), gFZ (m/s^2), gyroX (rad/s), gyroY (rad/s), gyroZ (rad/s), magX (uT), magY (uT), magZ (uT), Filtered gFZ\n");
                        int shortestLength = shortestList();
                        for(int i = 0; i < shortestLength; i++){
                            Squidward.write(String.format("%.2f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", timestamp.get(i), gFX.get(i), gFY.get(i), gFZ.get(i), gyroX.get(i), gyroY.get(i), gyroZ.get(i), magX.get(i), magY.get(i), magZ.get(i), Filtered_Data[i]));
                        }
                        Squidward.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

    }

    private String getStorage(){
        return this.getExternalFilesDir(null).getAbsolutePath();
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
        timestamp = new ArrayList<Float>();
        gFX = new ArrayList<Double>();
        gFY = new ArrayList<Double>();
        gFZ = new ArrayList<Double>();
        gyroX = new ArrayList<Double>();
        gyroY = new ArrayList<Double>();
        gyroZ = new ArrayList<Double>();
        magX = new ArrayList<Double>();
        magY = new ArrayList<Double>();
        magZ = new ArrayList<Double>();
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

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native double[] FilterBrHr(double[] data, boolean BR);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("Algorithm");
    }
}
