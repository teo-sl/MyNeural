package v1.layer;

import v1.activationFunction.ActivationFunction;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Layer {
    private double[][] weights;
    private ActivationFunction activationFunction;
    private int neuronNumber,inputDim;

    private double[] z;
    private double[] a;

    private double[] z_prime;

    private double[] delta;


    public Layer(int neuronNumber,int inputDim,ActivationFunction activationFunction) {
        this.neuronNumber=neuronNumber;
        this.inputDim=inputDim;
        this.activationFunction=activationFunction;

        weights=new double[neuronNumber][inputDim];

        for(int i=0;i<neuronNumber;++i)
            weights[i]= ThreadLocalRandom.current().doubles(inputDim, -1, 1).toArray();

        z = new double[neuronNumber];
        a = new double[neuronNumber];
        z_prime=new double[neuronNumber];

        delta=new double[neuronNumber];
    }

    public int getOutputDim() {
        return neuronNumber;
    }

    public double[] getOutput(double[] input) {
        if(input.length!=inputDim)
            throw new IllegalArgumentException("Dimensione errata dell'input per il neurone");

        double[] ret=new double[neuronNumber];

        for(int i=0;i<neuronNumber;++i) {
            double sum=0;
            for(int j=0;j<inputDim;++j)
                sum+=weights[i][j]*input[j];
            z[i]=sum;
            ret[i]=activationFunction.evaluate(sum);
            z_prime[i]=activationFunction.derive(sum);

        }
        a=Arrays.copyOf(ret,ret.length);
        return ret;

    }

    private void printAll() {
        System.out.println(Arrays.toString(weights[0]));
    }

    public String printWeights() {
        StringBuilder str = new StringBuilder(100);
        for(int i=0;i<weights.length;++i)
            str.append(Arrays.toString(weights[i])+" | ");
        str.append(" ||||| ");

        return str.toString();
    }

    public int getNeuronNumber() {
        return neuronNumber;
    }

    public int getInputDim() {
        return inputDim;
    }

    public double[] getActivation() {
        return a;
    }

    public double[] evaluateDeltaOutput(double[] Y) {
        for(int i=0;i<neuronNumber;++i)
            delta[i]=z_prime[i]*(Y[i]-a[i]);
        return delta;
    }

    public double[] evaluateDelta(double[] delta_i, double[][] W) {
        for(int i=0;i<neuronNumber;++i) {
            double tmp=0;
            for(int j=0;j<W.length;++j)
                tmp+=W[j][i]*delta_i[j];
            delta[i]=tmp*z_prime[i];
        }
        return delta;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void updateWeights(double alpha) {
        for(int i=0;i<weights.length;++i)
            for(int j=0;j<weights[0].length;++j)
                weights[i][j]+=alpha*a[i]*delta[i];
    }
}


