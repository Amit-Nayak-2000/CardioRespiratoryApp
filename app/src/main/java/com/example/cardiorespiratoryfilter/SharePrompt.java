package com.example.cardiorespiratoryfilter;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatDialogFragment;

public class SharePrompt extends AppCompatDialogFragment {

    private ShareDialogListener listener;

    public Dialog onCreateDialog(Bundle savedInstanceState){
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setMessage("If you wish to share your data, you must consent to the following conditions:\n\n" +
                "Any information that could be used to identify you will not be collected. \n\n" +
                "Your name will not be associated with the data in any way. \n\n" +
                "Sharing your data is completely voluntary and optional. \n\n" +
                "All of the data will be kept strictly confidential and will be used for academic purposes. \n\n" +
                "By clicking \"Share Data\", you agree to the above terms. \n")
                .setPositiveButton("SHARE DATA", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        listener.passBool(true);
                    }
                })
                .setNegativeButton("Do Not Share", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        listener.passBool(false);
                    }
                });
        return builder.create();
    }

    public void onCancel(DialogInterface dialog){
        listener.passBool(false);
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);

        try {
            listener = (ShareDialogListener) context;
        } catch (ClassCastException e) {
            throw new ClassCastException(context.toString());
        }
    }

    public interface ShareDialogListener{
        void passBool(boolean shareData);
    }
}
