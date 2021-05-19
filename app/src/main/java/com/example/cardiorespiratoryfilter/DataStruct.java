package com.example.cardiorespiratoryfilter;

public class DataStruct {
    private double[] breathingFilter;
    private double[] heartFilter;
    private double[] rates;

    public DataStruct(double[] breathingFilter, double[] heartFilter, double[] rates){
        this.breathingFilter = breathingFilter;
        this.heartFilter = heartFilter;
        this.rates = rates;
    }

    public double[] getBreathingFilter() {
        return breathingFilter;
    }

    public double[] getHeartFilter() {
        return heartFilter;
    }

    public double[] getRates() {
        return rates;
    }

}
