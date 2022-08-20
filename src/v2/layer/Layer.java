package v2.layer;

import v2.activationFunction.ActivationFunction;
import v2.util.Vectors;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class Layer {
    private double[][] weights;
    private double[] biases;
    private ActivationFunction activationFun;
    private int neuronNumber,inputDim,outputDim;

    private double[] a,z,sigma_prime,delta;

    public Layer(int neuronNumber,int inputDim,ActivationFunction activationFun) {
        this.neuronNumber=neuronNumber;
        this.inputDim=inputDim;
        this.outputDim=neuronNumber;

        this.activationFun=activationFun;

        this.biases=ThreadLocalRandom.current().doubles(neuronNumber, -1, 1).toArray();

        this.weights=new double[neuronNumber][inputDim];
        for(int i=0;i<neuronNumber;++i)
            this.weights[i]= ThreadLocalRandom.current().doubles(inputDim, -1, 1).toArray();
    }

    public double[] feedLayer(double[] input) {
        this.z=new double[neuronNumber];
        this.a=new double[neuronNumber];
        this.sigma_prime=new double[neuronNumber];

        for(int i=0;i<neuronNumber;++i) {
            z[i]= Vectors.scalarProduct(input,weights[i])+biases[i];
            a[i]=activationFun.evaluate(z[i]);
            sigma_prime[i]=activationFun.derive(z[i]);
        }
        return a;
    }

    public double[] computeDeltaOutput(double[] Y) {
        if(Y.length!=outputDim) throw new IllegalArgumentException("Dimensione y non in accordo alla dimensione di output del layer");
        delta=new double[outputDim];
        for(int j=0;j<outputDim;++j)
            delta[j]=2*(a[j]-Y[j]);
        return delta;
    }
    public double[] computeDelta(double[] delta_l_suc, double[][] w_l_suc,double[] sigma_prime_l_suc) {
        int n_l_suc=w_l_suc.length;
        delta=new double[neuronNumber];
        Arrays.fill(delta,0);
        for(int k=0;k<outputDim;++k)
            for(int j=0;j<n_l_suc;++j)
                delta[k]+=w_l_suc[j][k]*sigma_prime_l_suc[j]*delta_l_suc[j];
        return delta;
    }
    public void updateWeights(double[] a_l_prec, double[] X_i,boolean first) {
        int n_l_prec=-1;
        if(!first) {
            if(a_l_prec.length != weights[0].length) throw new IllegalArgumentException("Dimensione input diversa da dimensione di input del layer");
            n_l_prec=a_l_prec.length;
        }
        else {
            if(X_i.length!=weights[0].length) throw new IllegalArgumentException("Dimensione input diversa da dimensione di input del layer (input layer)");
            n_l_prec=X_i.length;
        }
        for(int j=0;j<outputDim;++j)
            for(int k=0;k<n_l_prec;++k)
                if(!first)
                    weights[j][k]-=0.1*(a_l_prec[k]*sigma_prime[j]*delta[j]);
                else
                    weights[j][k]-=0.1*(X_i[k]*sigma_prime[j]*delta[j]);

    }
    public void updateBiases() {
        for(int j=0;j<outputDim;++j)
            biases[j]-=0.1*(1*sigma_prime[j]*delta[j]);
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
