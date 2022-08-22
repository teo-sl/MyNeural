package v2.ann;

import v2.activationFunction.ActivationFunction;
import v2.layer.Layer;
import v2.util.Matrixes;
import v2.util.Vectors;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class ANN {
    private Layer[] layers;
    private int inputDim,outputDim,numberOfLayer;

    private List<Double> errors = new LinkedList<>();

    public ANN(int inputDim, int[] neuronPerLayer, ActivationFunction[] activationFunctions) {
        if(neuronPerLayer.length==0) throw new IllegalArgumentException("The layer number must be at least one");

        this.inputDim=inputDim;
        this.numberOfLayer=neuronPerLayer.length;
        this.outputDim=neuronPerLayer[numberOfLayer-1];

        this.layers=new Layer[numberOfLayer];

        int tmpInput=inputDim;

        for(int i=0;i<numberOfLayer;++i) {
            layers[i]=new Layer(neuronPerLayer[i],tmpInput,activationFunctions[i]);
            tmpInput=layers[i].getOutputDim();
        }
    }

    public List<Double> getErrors() {
        return errors;
    }

    public double[] evaluate(double[] X) {
        double[] input=X;
        double[] tmp;
        for(int i=0;i<numberOfLayer;++i) {
            tmp=layers[i].feedLayer(input);
            input=tmp;
        }
        return input;
    }

    public void train(double[][] X, double[][] Y) {
        int n_sample_input=X.length, inputDim=X[0].length;
        int n_sample_output=Y.length, outputDim=Y[0].length;

        if(n_sample_input!=n_sample_output) throw new IllegalArgumentException("Size of X is different from the size of Y");
        if(inputDim!=this.inputDim || outputDim!=this.outputDim) throw new IllegalArgumentException("Size of X or Y is different form the the network dimensions");

        Layer l_succ,l_prec,l_curr;
        double[] delta_curr,tmp_delta;

        double[] Y_pred;
        double[] Y_real;

        for(int i=0;i<n_sample_input;++i) {

            Y_pred=this.evaluate(X[i]);
            Y_real=Y[i];
            errors.add(Vectors.squaredError(Y_pred,Y_real));

            delta_curr=layers[numberOfLayer-1].computeDeltaOutput(Y[i]);

            for(int j=numberOfLayer-2;j>=0;--j) {
                l_succ=layers[j+1];
                tmp_delta=layers[j].computeDelta(delta_curr,l_succ.getWeights(),l_succ.getSigma_prime());
                delta_curr=tmp_delta;
            }

            for(int j=numberOfLayer-1;j>=0;--j) {
                l_curr=layers[j];

                if(j>0) {
                    l_prec=layers[j-1];
                    l_curr.updateWeights(l_prec.getA());
                }
                else
                    l_curr.updateWeights(X[i]);
                l_curr.updateBiases();
            }
        }
    }
    public String toString() {
        StringBuilder builder=new StringBuilder(100);
        builder.append("input dim: "+inputDim+"\n");
        builder.append("output dim: "+outputDim+"\n");
        builder.append("number of layers: "+numberOfLayer+"\n");
        int i=0;
        for(Layer l : layers) {
            builder.append("Layer "+i+"\n");
            builder.append("Weights: \n");
            builder.append(Matrixes.printMatrix(l.getWeights()));

            builder.append("Biases: \n");
            builder.append(Arrays.toString(l.getBiases())+"\n");

            i++;
        }

        return builder.toString();
    }
}
