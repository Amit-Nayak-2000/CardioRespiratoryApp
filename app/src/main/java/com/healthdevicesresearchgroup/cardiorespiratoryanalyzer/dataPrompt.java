package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;

import androidx.appcompat.app.AppCompatDialogFragment;

public class dataPrompt extends AppCompatDialogFragment {
    private dataPromptListener listener;
    private EditText age;
    private EditText BRPM;
    private EditText BPM;
    private RadioGroup radioGroup;
    private RadioButton radioButton;
    private CheckBox covid;

    private int old;
    private int brpm;
    private int bpm;
    private String gender = "";
    private boolean covid19;


    public Dialog onCreateDialog(Bundle savedInstanceState){
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        LayoutInflater inflater = requireActivity().getLayoutInflater();
        View view = inflater.inflate(R.layout.activity_data_collect, null);

        radioGroup = (RadioGroup) view.findViewById(R.id.radiogroup);
        RadioGroup.LayoutParams layoutParams = new RadioGroup.LayoutParams(
                RadioGroup.LayoutParams.WRAP_CONTENT,
                RadioGroup.LayoutParams.WRAP_CONTENT
        );

        for(int i = 0; i < 3; i++){
            radioButton = new RadioButton(getContext());
            if(i == 0){
                radioButton.setText("Male");
            }
            else if (i == 1){
                radioButton.setText("Female");
            }
            else{
                radioButton.setText("Other");
            }
            radioButton.setId(i);
            radioGroup.addView(radioButton, layoutParams);
        }


        builder.setView(view)
                .setPositiveButton("Share Data", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                    }
                })
                .setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                });

        age = view.findViewById(R.id.age);
        BRPM = view.findViewById(R.id.estimated_breaths);
        BPM = view.findViewById(R.id.estimated_beats);
        covid = view.findViewById(R.id.checkBox);



        final AlertDialog dialog =  builder.create();
        dialog.show();
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getData();
                listener.sendData(old, bpm, brpm, gender, covid19);
                dialog.dismiss();
            }
        });

        return dialog;
    }

    public interface dataPromptListener{
        void sendData(int age, int bpm, int brpm, String gender, boolean covid19);
    }

    @SuppressLint("ResourceType")
    public void getData(){
        if(age.getText().toString().isEmpty()){
            old = 0;
        }
        else{
            old = Integer.parseInt(age.getText().toString());
        }

        if(BRPM.getText().toString().isEmpty()){
            brpm = 0;
        }
        else{
            brpm = Integer.parseInt(BRPM.getText().toString());
        }

        if(BPM.getText().toString().isEmpty()){
            bpm = 0;
        }
        else{
            bpm = Integer.parseInt(BPM.getText().toString());
        }

        if(radioGroup.getCheckedRadioButtonId() == -1){
            gender = "no entry";
        }
        else if(radioGroup.getCheckedRadioButtonId() == 0){
            gender = "male";
        }
        else if(radioGroup.getCheckedRadioButtonId() == 1){
            gender = "female";
        }
        else{
            gender = "other";
        }

        if(covid.isChecked()){
            covid19 = true;
        }
        else{
            covid19 = false;
        }

    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);

        try {
            listener = (dataPrompt.dataPromptListener) context;
        } catch (ClassCastException e) {
            throw new ClassCastException(context.toString());
        }
    }



}
