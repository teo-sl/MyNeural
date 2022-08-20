package v1.training;

import v1.network.ANN;

import java.util.*;

public class Trainer {
    private ANN ann;
    private double[][] X;
    private double[][] Y;

    private int batchSize;
    private int nIter;
    private double alpha;

    private Random r = new Random();

    public Trainer(ANN ann, double[][] X, double[][] Y, int batchSize, int nIter,double alpha) {
        if(X.length != Y.length) throw new IllegalArgumentException("Le dimensioni di input e output non coincidono");
        if(X[0].length!=ann.getInputDim()) throw  new IllegalArgumentException("Dimensione di input non in accordo all'input della rete");
        if(Y[0].length != ann.getOutputDim()) throw new IllegalArgumentException("Dimensione dell'output non in accordo a quella della rete");
        this.ann=ann;
        this.X = X;
        this.Y = Y;
        this.batchSize=batchSize;
        this.nIter=nIter;
        this.alpha=alpha;
    }


    public void train() {
        for(int i=0;i<X.length;++i) {
            double[] X_i=X[i];
            double[] Y_i=Y[i];
            ann.evaluate(X_i);

            double[] delta_i=ann.getOutputLayer().evaluateDeltaOutput(Y_i);

            int L=ann.getnLayers();

            double[][] w_i=ann.getOutputLayer().getWeights();
            for(int j=L-2;j>=0;j--) {
                double[] delta_i_new=ann.getNthLayer(j).evaluateDelta(delta_i,w_i);
                delta_i=delta_i_new;
                w_i=ann.getNthLayer(j).getWeights();
            }
            for(int j=0;j<ann.getnLayers();++j)
                ann.getNthLayer(j).updateWeights(alpha);
        }
    }




}
