package v1.network;

import v1.activationFunction.ActivationFunction;
import v1.layer.Layer;

public class ANN {
    private int nLayers;
    private int inputDim;
    private Layer[] layers;

    public ANN(int inputDim, int nLayers, int[] neuronsPerLevel, ActivationFunction[] activationFunctionPerLayer) {

        if(nLayers!=activationFunctionPerLayer.length || nLayers!=neuronsPerLevel.length)
            throw new IllegalArgumentException("Errore nella dimensione degli array, devono essere in accordo al numero di v1.layer della rete");

        this.nLayers=nLayers;
        this.inputDim=inputDim;

        layers=new Layer[nLayers];
        for(int i=0;i<nLayers;++i)
            if(i==0)
                layers[i]=new Layer(neuronsPerLevel[i],inputDim+1,activationFunctionPerLayer[i]);
            else
                layers[i]=new Layer(neuronsPerLevel[i],layers[i-1].getOutputDim(),activationFunctionPerLayer[i]);
    }

    public double[] evaluate(double[] input) {
        if(input.length!=inputDim)
            throw new IllegalArgumentException("Dimensione dell'input della rete diversa da quella passata al metodo");
        double[] tmpInput=input;
        double[] tmp;
        for(int i=0;i<nLayers;++i) {
            tmp=layers[i].getOutput(tmpInput);
            tmpInput=tmp;
        }
        return tmpInput;

    }

    public String showANN() {
        StringBuilder str = new StringBuilder(500);
        for(int i=0;i<nLayers;++i) {
            str.append("Layer "+i+"\n");
            str.append("Weights "+layers[i].getWeights()+"\n");
            str.append("\n\n ------------------");

        }

        return str.toString();

    }

    public int getInputDim() {
        return layers[0].getInputDim();
    }
    public int getOutputDim() {
        return layers[layers.length-1].getOutputDim();
    }

    public Layer getOutputLayer() {
        return layers[layers.length-1];
    }
    public Layer getInputLayer() {
        return layers[0];
    }

    public Layer getNthLayer(int i) {
        if(i<0 || i>=nLayers) throw new IllegalArgumentException("Layer selezionato fuori bound");
        return layers[i];
    }

    public int getnLayers() {
        return nLayers;
    }
    /*
    public static void main(String[] args) {
        int inputDim=1;
        int nLayers=3;
        int[] neuronPerLayer={1,2,1};
        ActivationFunction[] activationFunctions=new ActivationFunction[3];
        activationFunctions[0]=new Relu();
        activationFunctions[1]=new Relu();
        activationFunctions[2]=new Relu();

        ANN ann = new ANN(inputDim,nLayers,neuronPerLayer,activationFunctions);
        System.out.println(ann.showANN());





        double[] testInput=new double[1];
        testInput[0]=1;
        System.out.println(Arrays.toString(ann.evaluate(testInput)));
    }

    */
}