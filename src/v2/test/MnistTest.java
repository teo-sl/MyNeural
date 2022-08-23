package v2.test;

import mnistReader.MnistDataReader;
import mnistReader.MnistMatrix;
import v2.activationFunction.ActivationFunction;
import v2.activationFunction.Sigmoid;
import v2.ann.ANN;
import v2.util.Vectors;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MnistTest {
    public static void main(String[] args) throws IOException {
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/t10k-images.idx3-ubyte", "/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/t10k-labels.idx1-ubyte");
        double[][] X=new double[mnistMatrix.length][mnistMatrix[0].getNumberOfRows()*mnistMatrix[0].getNumberOfColumns()];
        double[][] Y=new double[mnistMatrix.length][10];
        for(int i=0;i<mnistMatrix.length;++i) {
            X[i]=Arrays.stream(mnistMatrix[i].getData()).asDoubleStream().toArray();
            Y[i]= Vectors.convertToPositional(mnistMatrix[i].getLabel());
        }

        int inputDim=28*28;
        int nLayers=4;
        int[] neuronPerLayer={32,16,16,10};
        ActivationFunction[] activationFunctions=new ActivationFunction[nLayers];
        activationFunctions[0]=new Sigmoid();
        activationFunctions[1]=new Sigmoid();
        activationFunctions[2]=new Sigmoid();
        activationFunctions[3]=new Sigmoid();

        ANN ann = new ANN(inputDim,neuronPerLayer,activationFunctions);




        int batchsize=512;
        int epochs=1000;

        ann.train(X,Y,batchsize,epochs);
        Random r = new Random();
        int i_test=4321;
        for(int i=0;i<10;++i) {
            i_test=r.nextInt(10000);
            System.out.println("Y real : "+mnistMatrix[i_test].getLabel()+" ||| Y pred : "+ Vectors.getMaxIndex(ann.evaluate(X[i_test]))+" ||||| "+ i_test);
        }
        try {
            PrintWriter writer = new PrintWriter("error.csv", "UTF-8");
            List<Double> tmp = ann.getErrors();
            for(double x : tmp)
                writer.print(x+"\n");
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }

        String fileName= "object1.txt";
        FileOutputStream fos = new FileOutputStream(fileName);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(ann);
        oos.close();
    }


}
