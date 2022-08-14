package layer;

import activationFunction.ActivationFunction;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Layer {
    private double[][] weights;
    private double[] biases;
    private ActivationFunction activationFunction;
    private int neuronNumber,inputDim;

    public Layer(int neuronNumber,int inputDim,ActivationFunction activationFunction) {
        this.neuronNumber=neuronNumber;
        this.inputDim=inputDim;
        this.activationFunction=activationFunction;

        weights=new double[neuronNumber][inputDim];

        for(int i=0;i<neuronNumber;++i)
            weights[i]= ThreadLocalRandom.current().doubles(inputDim, 0, 1).toArray();
        biases = ThreadLocalRandom.current().doubles(neuronNumber, 0, 1).toArray();
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
            sum+=biases[i];
            ret[i]=activationFunction.evaluate(sum);
        }

        return ret;

    }

    private void printAll() {
        System.out.println(Arrays.toString(weights[0]));
        System.out.println(Arrays.toString(biases));
    }

    public String getWeights() {
        StringBuilder str = new StringBuilder(100);
        for(int i=0;i<weights.length;++i)
            str.append(Arrays.toString(weights[i])+" | ");
        str.append(" ||||| ");

        return str.toString();
    }
    public String getBiases() {
        return Arrays.toString(biases);
    }


}


