package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;

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

public class DataPlot extends AppCompatActivity {

    boolean titleSet;
    boolean breathingDisp;
    DecimalFormat df = new DecimalFormat("##.##");
    LineChart lineChart;
    LineData lineData;
    LineData blineData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dataplot);

        titleSet = false;
        breathingDisp = false;
        lineChart = findViewById(R.id.lineChart);

        ArrayList<Entry> lineEntries = new ArrayList<Entry>();
        ArrayList<Entry> B_lineEntries = new ArrayList<Entry>();

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
                        String title = "BRPM: " + df.format(Float.parseFloat(values[12])) + " | BPM: " + df.format(Float.parseFloat(values[13]));
                        setTitle(title);
                        titleSet = true;
                    }
                    lineEntries.add(new Entry(Float.parseFloat(values[0]), Float.parseFloat(values[4])));
                    B_lineEntries.add(new Entry(Float.parseFloat(values[0]), Float.parseFloat(values[3])));
                }
            }
            sc.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        LineDataSet lineDataSet = new LineDataSet(lineEntries, "Heart Beat Data");
        LineDataSet blineDataSet = new LineDataSet(B_lineEntries, "Breathing Data");

        lineDataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        lineDataSet.setValueTextColor(Color.BLACK);
        lineDataSet.setValueTextSize(16f);

        blineDataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        blineDataSet.setValueTextColor(Color.BLUE);
        blineDataSet.setValueTextSize(16f);


        lineData = new LineData(lineDataSet);
        blineData = new LineData(blineDataSet);

        lineChart.setData(lineData);

    }

    public boolean onCreateOptionsMenu(Menu menu){
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.togglemenu, menu);
        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item){
        switch(item.getItemId()){
            case R.id.item1:
            toggleData();
            return true;
        }
        return false;
    }

    protected void toggleData(){
        if(!breathingDisp){
            breathingDisp = true;
            lineChart.setData(blineData);
            lineChart.invalidate();
        }
        else{
            breathingDisp = false;
            lineChart.setData(lineData);
            lineChart.invalidate();
        }
    }
}