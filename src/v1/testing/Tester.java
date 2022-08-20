package v1.testing;

import v1.activationFunction.ActivationFunction;
import v1.activationFunction.Relu;
import v1.network.ANN;
import v1.training.Trainer;

import java.util.Random;

public class Tester {

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

        int p = 10000;
        Random r = new Random();
        double[][] X = new double[p][1];
        double[][] Y = new double[p][1];
        for(int i=0;i<p;i++) {
            double tmp=r.nextDouble()*100;
            X[i][0]=tmp;
            Y[i][0]=Math.exp(tmp);
        }

        Trainer trainer = new Trainer(ann,X,Y,-1,10,0.001);
        trainer.train();
        double[] test_x=new double[1];
        for(int i=0;i<50;++i) {
            double tmp=r.nextDouble()*100;
            test_x[0]=tmp;
            System.out.println("Y real : "+Math.exp(tmp)+" ||| Y pred : "+ann.evaluate(test_x)[0]);
        }
    }

}
