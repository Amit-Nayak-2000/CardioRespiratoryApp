package com.healthdevicesresearchgroup.cardiorespiratoryanalyzer;

import java.util.List;

public class DataDoc {
    private int age;
    private double bpm;
    private double brpm;
    private double estimated_bpm;
    private double estimated_brpm;
    private String gender;
    private boolean covid19;
    private boolean healthCondition;
    private List<Double> rawHeart;
    private List<Double> filteredHeart;
    private List<Double> rawBreathing;
    private List<Double> filteredBreathing;
    private List<Double> time;


    public DataDoc(int age, double bpm, double brpm, double estimated_bpm, double estimated_brpm, String gender, boolean covid19, boolean healthCondition, List<Double> rawHeart, List<Double> filteredHeart, List<Double> rawBreathing, List<Double> filteredBreathing, List<Double> time) {
        this.age = age;
        this.bpm = bpm;
        this.brpm = brpm;
        this.estimated_bpm = estimated_bpm;
        this.estimated_brpm = estimated_brpm;
        this.gender = gender;
        this.covid19 = covid19;
        this.healthCondition = healthCondition;
        this.rawHeart = rawHeart;
        this.filteredHeart = filteredHeart;
        this.rawBreathing = rawBreathing;
        this.filteredBreathing = filteredBreathing;
        this.time = time;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public double getBpm() {
        return bpm;
    }

    public void setBpm(double bpm) {
        this.bpm = bpm;
    }

    public double getBrpm() {
        return brpm;
    }

    public void setBrpm(double brpm) {
        this.brpm = brpm;
    }

    public double getEstimated_bpm() {
        return estimated_bpm;
    }

    public void setEstimated_bpm(double estimated_bpm) {
        this.estimated_bpm = estimated_bpm;
    }

    public double getEstimated_brpm() {
        return estimated_brpm;
    }

    public void setEstimated_brpm(double estimated_brpm) {
        this.estimated_brpm = estimated_brpm;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public boolean isCovid19() {
        return covid19;
    }

    public void setCovid19(boolean covid19) {
        this.covid19 = covid19;
    }

    public boolean isHealthCondition() {
        return healthCondition;
    }

    public void setHealthCondition(boolean healthCondition) {
        this.healthCondition = healthCondition;
    }

    public List<Double> getRawHeart() {
        return rawHeart;
    }

    public void setRawHeart(List<Double> rawHeart) {
        this.rawHeart = rawHeart;
    }

    public List<Double> getFilteredHeart() {
        return filteredHeart;
    }

    public void setFilteredHeart(List<Double> filteredHeart) {
        this.filteredHeart = filteredHeart;
    }

    public List<Double> getRawBreathing() {
        return rawBreathing;
    }

    public void setRawBreathing(List<Double> rawBreathing) {
        this.rawBreathing = rawBreathing;
    }

    public List<Double> getFilteredBreathing() {
        return filteredBreathing;
    }

    public void setFilteredBreathing(List<Double> filteredBreathing) {
        this.filteredBreathing = filteredBreathing;
    }

    public List<Double> getTime() {
        return time;
    }

    public void setTime(List<Double> time) {
        this.time = time;
    }
}
