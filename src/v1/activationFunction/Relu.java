package v1.activationFunction;

public class Relu implements ActivationFunction {

    @Override
    public double evaluate(double value) {
        if(value>=0) return value;
        else return 0;
    }

    @Override
    public double derive(double value) {
        return (value>0) ? value : 0;
    }
}
