package activationFunction;

public class Relu implements ActivationFunction {

    @Override
    public double evaluate(double value) {
        if(value>=0) return value;
        else return 0;
    }
}
