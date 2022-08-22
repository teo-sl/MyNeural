package v2.layer;

import v2.activationFunction.ActivationFunction;
import v2.util.Matrixes;
import v2.util.Vectors;

import java.io.Serializable;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Layer implements Serializable {
    private static double lr=0.1;
    private double[][] weights,delta_w;
    private double[] biases,delta_b;
    private ActivationFunction activationFun;
    private int neuronNumber,inputDim,outputDim;

    private double[] a,z,sigma_prime,delta;

    public Layer(int neuronNumber,int inputDim,ActivationFunction activationFun) {
        this.neuronNumber=neuronNumber;
        this.inputDim=inputDim;
        this.outputDim=neuronNumber;

        this.activationFun=activationFun;

        this.biases=ThreadLocalRandom.current().doubles(neuronNumber, -1, 1).toArray();
        this.delta_b=new double[neuronNumber];

        this.weights=new double[neuronNumber][inputDim];
        this.delta_w=new double[neuronNumber][inputDim];
        for(int i=0;i<neuronNumber;++i)
            this.weights[i]= ThreadLocalRandom.current().doubles(inputDim, -1, 1).toArray();

        this.delta=new double[outputDim];
        this.z=new double[neuronNumber];
        this.a=new double[neuronNumber];
        this.sigma_prime=new double[neuronNumber];

        Arrays.fill(delta,0);


    }

    public double[] feedLayer(double[] input) {
        for(int i=0;i<neuronNumber;++i) {
            z[i]= Vectors.scalarProduct(input,weights[i])+biases[i];
            a[i]=activationFun.evaluate(z[i]);
            sigma_prime[i]=activationFun.derive(z[i]);
        }
        return a;
    }

    public double[] computeDeltaOutput(double[] Y) {
        if(Y.length!=outputDim) throw new IllegalArgumentException("Y size is different from network output size");

        for(int j=0;j<outputDim;++j)
            delta[j]=2*(a[j]-Y[j]);
        return delta;
    }
    public double[] computeDelta(double[] delta_l_suc, double[][] w_l_suc,double[] sigma_prime_l_suc) {
        int n_l_suc=w_l_suc.length;
        Arrays.fill(delta,0);
        for(int k=0;k<outputDim;++k)
            for(int j=0;j<n_l_suc;++j)
                delta[k]+=w_l_suc[j][k]*sigma_prime_l_suc[j]*delta_l_suc[j];
        return delta;
    }
    public void resetAll() {
        Arrays.fill(delta_b,0);
        Matrixes.reset(delta_w);
    }
    public void update_delta_Weights(double[] a_l_prec) {
        if(a_l_prec.length != weights[0].length) throw new IllegalArgumentException("Size of a_l_prec is different from the layer input's size");
        int n_l_prec=a_l_prec.length;
        for(int j=0;j<outputDim;++j)
            for(int k=0;k<n_l_prec;++k)
                delta_w[j][k]-=lr*(a_l_prec[k]*sigma_prime[j]*delta[j]);

    }


    public void update_delta_Biases() {
        for(int j=0;j<outputDim;++j)
            delta_b[j]-=lr*(1*sigma_prime[j]*delta[j]);
    }

    public void updateAll(int n) {
        for(int i=0;i<neuronNumber;++i)
            biases[i]+=delta_b[i]/n;
        for(int i=0;i<neuronNumber;++i)
            for(int j=0;j<inputDim;++j)
                weights[i][j]+=delta_w[i][j]/n;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public ActivationFunction getActivationFun() {
        return activationFun;
    }

    public int getNeuronNumber() {
        return neuronNumber;
    }

    public int getInputDim() {
        return inputDim;
    }

    public int getOutputDim() {
        return outputDim;
    }

    public double[] getA() {
        return a;
    }

    public double[] getZ() {
        return z;
    }

    public double[] getSigma_prime() {
        return sigma_prime;
    }

    public double[] getDelta() {
        return delta;
    }
}
