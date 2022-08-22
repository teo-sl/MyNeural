package v2.test;



import v2.activationFunction.ActivationFunction;
import v2.activationFunction.Relu;
import v2.activationFunction.Sigmoid;
import v2.ann.ANN;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;
import java.util.Random;

public class Tester {

    public static void main(String[] args) {
        int inputDim=2;
        int nLayers=3;
        int[] neuronPerLayer={2,2,1};
        ActivationFunction[] activationFunctions=new ActivationFunction[3];
        activationFunctions[0]=new Relu();
        activationFunctions[1]=new Relu();
        activationFunctions[2]=new Sigmoid();

        ANN ann = new ANN(inputDim,neuronPerLayer,activationFunctions);
        //System.out.println(ann);

        int p = 100000;
        Random r = new Random();
        double[][] X = new double[p][2];
        double[][] Y = new double[p][1];
        for(int i=0;i<p;i++) {
            int a=r.nextInt(2);
            int b= r.nextInt(2);

            X[i][0]=a;
            X[i][1]=b;
            Y[i][0]=xor(a,b);
        }

        int batchsize=1;
        int epochs=2;

        ann.train(X,Y,batchsize,epochs);
        double[] test_x=new double[2];
        for(int i=0;i<p;++i) {
            int a=r.nextInt(2);
            int b=r.nextInt(2);
            test_x[0]=a;
            test_x[1]=b;
            System.out.println("Y real : "+xor(a,b)+" ||| Y pred : "+ ann.evaluate(test_x)[0]);
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




    }
    public static int xor(int a, int b) {
        if(a==1 && b==1 || a==0 && b==0) return 0;

        return 1;
    }

}
