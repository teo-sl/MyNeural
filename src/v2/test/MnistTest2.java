package v2.test;

import mnistReader.MnistDataReader;
import mnistReader.MnistMatrix;
import v2.ann.ANN;
import v2.util.Vectors;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.Random;

public class MnistTest2 {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        //MnistMatrix[] mnistMatrix = new MnistDataReader().readData("/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/t10k-images.idx3-ubyte", "/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/t10k-labels.idx1-ubyte");
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/train-images.idx3-ubyte", "/Users/teodorosullazzo/Documents/git_repos/my_neural_1/src/data/train-labels.idx1-ubyte");
        double[][] X=new double[mnistMatrix.length][mnistMatrix[0].getNumberOfRows()*mnistMatrix[0].getNumberOfColumns()];
        double[][] Y=new double[mnistMatrix.length][10];
        for(int i=0;i<mnistMatrix.length;++i) {
            X[i]= Arrays.stream(mnistMatrix[i].getData()).asDoubleStream().toArray();
            Y[i]= Vectors.convertToPositional(mnistMatrix[i].getLabel());
        }

        String fileName= "object.txt";
        FileInputStream fin = new FileInputStream(fileName);
        ObjectInputStream ois = new ObjectInputStream(fin);
        ANN ann= (ANN) ois.readObject();
        ois.close();

        Random r = new Random();
        int i_test;
        int range=100,k=0;
        for(int i=0;i<range;++i) {
            i_test=r.nextInt(10000);
            System.out.println("Y real : "+mnistMatrix[i_test].getLabel()+" ||| Y pred : "+ Vectors.getMaxIndex(ann.evaluate(X[i_test]))+" ||||| "+ i_test);
            if(mnistMatrix[i_test].getLabel()==Vectors.getMaxIndex(ann.evaluate(X[i_test])))
                k++;
        }
        double tmp=((double) k)/((double)range);
        System.out.println("Correct rate "+k);

    }
}
