package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.utils.ColorTemplate;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

public class dataplot extends AppCompatActivity {

    boolean titleSet;
    DecimalFormat df = new DecimalFormat("##.##");


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dataplot);

        titleSet = false;

        ArrayList<Entry> lineEntries = new ArrayList<Entry>();

        Bundle fileintent = getIntent().getExtras();
        String filename = fileintent.getString("Filename");
        String data;
        File file = new File(filename);
        try {
            Scanner sc = new Scanner(file);
            if(sc.hasNextLine()){
                data = sc.nextLine();
            }
            while(sc.hasNextLine()){
                data = sc.nextLine();
                if(data.isEmpty()){
                    break;
                }
                else{
                    //use 3 for breathing or 4 for heartbeat
                    String values[] = data.split(",");
                    if(!titleSet){
                        String title = "BRPM: " + df.format(Float.parseFloat(values[5])) + "; BPM: " + df.format(Float.parseFloat(values[6]));
                        setTitle(title);
                        titleSet = true;
                    }
                    lineEntries.add(new Entry(Float.parseFloat(values[0]), Float.parseFloat(values[4])));
                }
            }
            sc.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        LineChart lineChart = findViewById(R.id.lineChart);
        LineDataSet lineDataSet = new LineDataSet(lineEntries, "Data");
        lineDataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        lineDataSet.setValueTextColor(Color.BLACK);
        lineDataSet.setValueTextSize(16f);

        LineData lineData = new LineData(lineDataSet);

        lineChart.setData(lineData);

    }
}