package v2.activationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double evaluate(double value) {
        return 1/(1+Math.exp(-value));
    }

    @Override
    public double derive(double value) {
        double sigma_x=evaluate(value);
        return sigma_x*(1-sigma_x);
    }

}


