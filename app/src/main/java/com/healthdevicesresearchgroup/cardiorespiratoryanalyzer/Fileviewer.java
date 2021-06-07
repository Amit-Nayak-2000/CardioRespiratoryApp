package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import android.app.ListActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import java.util.ArrayList;
import java.lang.String;


public class Fileviewer extends ListActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_fileviewer);

        ListView listView;

        Intent intent = getIntent();
        Bundle args = intent.getBundleExtra("BUNDLE");
        final ArrayList<String> filenames = (ArrayList<String>) args.getSerializable("ARRAYLIST");
        ArrayList<String> filenamestrings = new ArrayList<String>();

        for(int i = 0; i < filenames.size(); i++){
            filenamestrings.add( filenames.get(i).substring(filenames.get(i).length() - 32, filenames.get(i).length() - 4));
        }

        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, filenamestrings);

        listView = (ListView) findViewById(android.R.id.list);
        listView.setAdapter(adapter);

        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Intent intent = new Intent(Fileviewer.this, dataplot.class);
                intent.putExtra("Filename", filenames.get(position));
                startActivity(intent);
            }
        });


    }

}